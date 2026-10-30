use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, FLUX_DIFFUSERS_STATE_PLAN, FLUX_LAYOUT_SIGNATURES,
    FLUX_STATE_PLAN_CASES, FluxChromaFinalHead, FluxChromaLayout, FluxChromaVariant,
    ModelFamilyError, ModelProbe, ModelStateLayout, ModelStateTransaction, ModelStore,
    ParserLimits, flux_chroma_configuration_for_probe,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, StreamId, Tensor, TensorBackend,
    TensorDescriptor,
};
use sim_types::CancellationToken;
use std::{collections::BTreeMap, fs, io::Write, path::Path};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn val_model_family_row_001_flux_chroma_adapter_preserves_variant_precedence_and_profiles() {
    let flux = flux_chroma_configuration_for_probe(
        &flux_probe("native", false),
        FluxChromaVariant::Flux,
        "Flux",
    )
    .expect("Flux configuration");
    assert_eq!(flux.layout, FluxChromaLayout::Native);
    assert_eq!((flux.in_channels, flux.out_channels), (16, 16));
    assert_eq!((flux.patch_size, flux.hidden_size), (2, 3_072));
    assert_eq!(flux.vector_input_dimension, Some(768));
    assert!(flux.guidance_embedding);
    assert_eq!(flux.double_block_count, 2);
    assert_eq!(flux.single_block_count, 2);

    let mut scaled_norm = diffusers_probe(4_096, true, true);
    scaled_norm
        .tensor_shapes
        .remove("transformer_blocks.0.attn.norm_k.weight");
    scaled_norm.tensor_shapes.insert(
        "transformer_blocks.0.attn.norm_k.scale".to_owned(),
        vec![128],
    );
    assert_eq!(
        flux_chroma_configuration_for_probe(&scaled_norm, FluxChromaVariant::Flux, "Flux")
            .expect("Diffusers scale key-norm configuration")
            .layout,
        FluxChromaLayout::Diffusers
    );

    let flux2 = flux_chroma_configuration_for_probe(
        &flux_probe("unprefixed", true),
        FluxChromaVariant::Flux2,
        "Flux2",
    )
    .expect("Flux2 configuration");
    assert_eq!(flux2.layout, FluxChromaLayout::Unprefixed);
    assert_eq!((flux2.patch_size, flux2.out_channels), (1, 128));
    assert_eq!(flux2.text_id_dimensions, [3]);

    let chroma =
        flux_chroma_configuration_for_probe(&chroma_probe(), FluxChromaVariant::Chroma, "Chroma")
            .expect("Chroma configuration");
    assert_eq!((chroma.in_channels, chroma.out_channels), (64, 64));
    assert_eq!(chroma.hidden_size, 5_120);

    let radiance = flux_chroma_configuration_for_probe(
        &radiance_probe(),
        FluxChromaVariant::ChromaRadiance,
        "ChromaRadiance",
    )
    .expect("ChromaRadiance configuration");
    assert_eq!(radiance.patch_size, 4);
    assert_eq!(radiance.context_input_dimension, 4_096);
    assert_eq!(radiance.final_head, FluxChromaFinalHead::Linear);
    assert!(radiance.use_x0_prediction);
    assert!(radiance.use_sequential_text_ids);
}

#[test]
fn val_model_detection_001_flux_diffusers_and_longcat_precedence_are_key_derived()
-> Result<(), Box<dyn std::error::Error>> {
    let mut flux_probe = diffusers_probe(4_096, true, true);
    flux_probe
        .metadata
        .insert("model_layout".to_owned(), "prefixed-native".to_owned());
    flux_probe
        .metadata
        .insert("image_model".to_owned(), "chroma".to_owned());
    assert_eq!(
        flux_probe.select_layout(FLUX_LAYOUT_SIGNATURES)?,
        ModelStateLayout::Diffusers
    );
    let flux = flux_chroma_configuration_for_probe(&flux_probe, FluxChromaVariant::Flux, "Flux")?;
    assert_eq!(flux.layout, FluxChromaLayout::Diffusers);
    assert_eq!(flux.context_input_dimension, 4_096);
    assert_eq!(flux.vector_input_dimension, Some(768));
    assert!(flux.guidance_embedding);
    assert_eq!(flux.double_block_count, 2);
    assert_eq!(flux.single_block_count, 2);

    let longcat_probe = diffusers_probe(3_584, false, false);
    let longcat = flux_chroma_configuration_for_probe(
        &longcat_probe,
        FluxChromaVariant::LongCatImage,
        "LongCatImage",
    )?;
    assert_eq!(longcat.layout, FluxChromaLayout::Diffusers);
    assert_eq!(longcat.context_input_dimension, 3_584);
    assert_eq!(longcat.vector_input_dimension, None);
    assert!(!longcat.guidance_embedding);
    assert_eq!(longcat.text_id_dimensions, [1, 2]);
    assert!(matches!(
        flux_chroma_configuration_for_probe(
            &longcat_probe,
            FluxChromaVariant::Flux,
            "FluxSchnell",
        ),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("selected LongCatImage, expected Flux")
    ));

    let schnell = flux_chroma_configuration_for_probe(
        &diffusers_probe(4_096, false, false),
        FluxChromaVariant::Flux,
        "FluxSchnell",
    )?;
    assert!(schnell.text_id_dimensions.is_empty());
    assert!(!schnell.guidance_embedding);
    Ok(())
}

#[test]
fn val_model_format_001_flux_diffusers_layout_comes_from_real_parsed_keys()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("flux-diffusers.safetensors");
    write_diffusers_safetensors(&path)?;
    let mut index = ArtifactIndex::default();
    index.add_root(ArtifactRoot::canonical(
        "flux-adapter",
        "checkpoints",
        directory.path(),
        ["safetensors"],
    )?)?;
    let cancellation = CancellationToken::default();
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("flux-adapter", "flux-diffusers.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    let probe = store.family_probe(&loaded, &cancellation)?;
    assert_eq!(probe.format_identities(), ["safetensors"]);
    assert_eq!(
        probe.select_layout(FLUX_LAYOUT_SIGNATURES)?,
        ModelStateLayout::Diffusers
    );
    let configuration =
        flux_chroma_configuration_for_probe(&probe, FluxChromaVariant::Flux, "Flux")?;
    assert_eq!(configuration.layout, FluxChromaLayout::Diffusers);
    assert_eq!(configuration.hidden_size, 128);
    assert_eq!(configuration.context_input_dimension, 4_096);
    Ok(())
}

#[test]
fn val_model_family_foundation_001_flux_diffusers_plan_converts_transactionally()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(FLUX_STATE_PLAN_CASES.len(), 3);
    assert_eq!(FLUX_STATE_PLAN_CASES[2].layout, ModelStateLayout::Diffusers);
    let plan = FLUX_DIFFUSERS_STATE_PLAN.compile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let source = diffusers_source(&backend, &context)?;
    let mapped = ModelStateTransaction::new(&backend, &context).execute(&plan, DIGEST, &source)?;
    let model = mapped.component("model").ok_or("missing model component")?;
    assert_eq!(
        model["native.double_blocks.0.img_attn.qkv.weight"]
            .descriptor()
            .shape(),
        &[6, 2]
    );
    assert_eq!(
        model["native.single_blocks.0.linear1.weight"]
            .descriptor()
            .shape(),
        &[10, 2]
    );
    for required in [
        "native.img_in.weight",
        "native.txt_in.weight",
        "native.double_blocks.0.img_attn.proj.weight",
        "native.single_blocks.0.linear2.weight",
        "native.final_layer.linear.weight",
    ] {
        assert!(
            model.contains_key(required),
            "missing converted key {required}"
        );
    }
    assert!(model.contains_key("native.diffusers.transformer_blocks.0.attn.to_q.weight"));
    Ok(())
}

#[test]
fn val_cancel_001_flux_diffusers_transaction_cancellation_commits_nothing()
-> Result<(), Box<dyn std::error::Error>> {
    let plan = FLUX_DIFFUSERS_STATE_PLAN.compile()?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let source = diffusers_source(&backend, &context)?;
    let baseline = backend.memory_snapshot().current_bytes;
    cancellation.cancel();
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(&plan, DIGEST, &source),
        Err(ModelFamilyError::Cancelled(_))
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn val_model_family_row_001_flux_chroma_adapter_rejects_malformed_and_cross_family_probes() {
    assert!(matches!(
        flux_chroma_configuration_for_probe(
            &chroma_probe(),
            FluxChromaVariant::Flux,
            "Flux",
        ),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("selected Chroma, expected Flux")
    ));

    let mut duplicate_key_norm = chroma_probe();
    duplicate_key_norm.tensor_shapes.insert(
        "double_blocks.0.img_attn.norm.key_norm.scale".to_string(),
        vec![128],
    );
    assert!(matches!(
        flux_chroma_configuration_for_probe(
            &duplicate_key_norm,
            FluxChromaVariant::Chroma,
            "Chroma",
        ),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("exactly one weight/scale")
    ));

    let mut malformed_patch = radiance_probe();
    malformed_patch
        .tensor_shapes
        .insert("img_in_patch.weight".to_string(), vec![64, 3, 4, 3]);
    assert!(matches!(
        flux_chroma_configuration_for_probe(
            &malformed_patch,
            FluxChromaVariant::ChromaRadiance,
            "ChromaRadiance",
        ),
        Err(ModelFamilyError::InvalidSelectorOutput(message))
            if message.contains("img_in_patch.weight shape")
    ));

    let mut ambiguous = flux_probe("native", false);
    ambiguous
        .tensor_shapes
        .extend(diffusers_probe(4_096, true, true).tensor_shapes);
    assert!(matches!(
        flux_chroma_configuration_for_probe(&ambiguous, FluxChromaVariant::Flux, "Flux"),
        Err(ModelFamilyError::ModelLayoutSelection(message))
            if message.contains("ambiguously match")
    ));
}

#[test]
fn val_model_family_row_001_flux_chroma_adapter_is_the_single_shared_owner() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let owner = fs::read_to_string(crate_root.join("src/flux_chroma_family.rs"))
        .expect("canonical Flux/Chroma adapter source");
    assert_eq!(owner.matches("pub enum FluxChromaVariant").count(), 1);
    assert_eq!(owner.matches("pub enum FluxChromaLayout").count(), 1);
    assert_eq!(
        owner.matches("pub struct FluxChromaConfiguration").count(),
        1
    );

    for row in [
        "flux_sim_model_0077.rs",
        "flux2_sim_model_0078.rs",
        "fluxinpaint_sim_model_0079.rs",
        "fluxschnell_sim_model_0080.rs",
        "chroma_sim_model_0066.rs",
        "chromaradiance_sim_model_0067.rs",
    ] {
        let source = fs::read_to_string(crate_root.join("src/families").join(row))
            .expect("Flux/Chroma family row source");
        assert!(!source.contains("fn require_vector_dimension("));
        assert!(!source.contains("fn consecutive_blocks("));
        assert!(source.contains("flux_chroma_configuration_for_probe("));
    }

    for row in [
        "flux_sim_model_0077.rs",
        "flux2_sim_model_0078.rs",
        "fluxinpaint_sim_model_0079.rs",
        "fluxschnell_sim_model_0080.rs",
    ] {
        let source = fs::read_to_string(crate_root.join("src/families").join(row))
            .expect("Flux family row source");
        for shared_owner in [
            "FLUX_COMPONENTS",
            "FLUX_WEIGHT_RULES",
            "FLUX_MODEL_REQUIRED_KEYS",
            "FLUX_MODEL_OPTIONAL_KEYS",
            "FLUX_FORWARD_PROGRAM",
            "FLUX_STATE_PLAN_CASES",
            "FLUX_COMPONENT_STATE_SCHEMAS",
        ] {
            assert!(
                source.contains(shared_owner),
                "{row} must use canonical {shared_owner}"
            );
            assert!(
                !source.contains(&format!("const {shared_owner}:")),
                "{row} must not redefine canonical {shared_owner}"
            );
        }
        assert!(!source.contains("ModelStateTransformPlanDefinition"));
    }

    for row in ["flux_sim_model_0077.rs", "fluxinpaint_sim_model_0079.rs"] {
        let source = fs::read_to_string(crate_root.join("src/families").join(row))
            .expect("Flux architecture-sharing row source");
        assert!(source.contains("FLUX_ARCHITECTURE_VERSION"));
    }
    for row in [
        "flux_sim_model_0077.rs",
        "fluxinpaint_sim_model_0079.rs",
        "fluxschnell_sim_model_0080.rs",
    ] {
        let source = fs::read_to_string(crate_root.join("src/families").join(row))
            .expect("Flux latent and memory-sharing row source");
        for shared_owner in [
            "FLUX_LATENT_FEATURE_ID",
            "FLUX_LATENT_IDENTIFIER",
            "FLUX_MEMORY_ESTIMATOR",
            "FLUX_MEMORY_USAGE_FACTOR",
        ] {
            assert!(
                source.contains(shared_owner),
                "{row} must use canonical {shared_owner}"
            );
        }
    }

    assert!(
        owner.contains("double_stream_modulation_img.\""),
        "canonical unprefixed plan must own the Flux2 image-modulation mapping"
    );
    assert!(
        owner.contains("double_stream_modulation_txt.\""),
        "canonical unprefixed plan must own the Flux2 text-modulation mapping"
    );
    assert_eq!(owner.matches("pub const FLUX_LAYOUT_SIGNATURES").count(), 1);
    assert_eq!(owner.matches("pub const FLUX_STATE_PLAN_CASES").count(), 1);
    assert!(owner.contains("FLUX_DIFFUSERS_STATE_PLAN"));
    assert!(owner.contains("LongCatImage"));
}

fn flux_probe(layout: &str, flux2: bool) -> ModelProbe {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    let mut tensor_shapes = BTreeMap::from([
        (
            format!("{prefix}double_blocks.0.img_attn.norm.key_norm.weight"),
            vec![128],
        ),
        (format!("{prefix}single_blocks.0.linear.weight"), vec![2, 2]),
        (format!("{prefix}img_in.weight"), vec![3_072, 64]),
        (format!("{prefix}txt_in.weight"), vec![3_072, 4_096]),
        (
            format!("{prefix}vector_in.in_layer.weight"),
            vec![3_072, 768],
        ),
        (
            format!("{prefix}guidance_in.in_layer.weight"),
            vec![3_072, 256],
        ),
    ]);
    if flux2 {
        tensor_shapes.insert(
            format!("{prefix}double_stream_modulation_img.lin.weight"),
            vec![2, 2],
        );
    }
    tensor_shapes.insert(
        format!("{prefix}double_blocks.1.img_attn.norm.key_norm.weight"),
        vec![128],
    );
    tensor_shapes.insert(format!("{prefix}single_blocks.1.linear.weight"), vec![2, 2]);
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

fn diffusers_probe(
    context_dimension: u64,
    vector_embedding: bool,
    guidance_embedding: bool,
) -> ModelProbe {
    let mut tensor_shapes = BTreeMap::from([
        ("x_embedder.weight".to_owned(), vec![3_072, 64]),
        ("x_embedder.bias".to_owned(), vec![3_072]),
        (
            "context_embedder.weight".to_owned(),
            vec![3_072, context_dimension],
        ),
        (
            "transformer_blocks.0.attn.norm_k.weight".to_owned(),
            vec![128],
        ),
        (
            "transformer_blocks.0.attn.to_q.weight".to_owned(),
            vec![3_072, 3_072],
        ),
        (
            "transformer_blocks.0.attn.to_k.weight".to_owned(),
            vec![3_072, 3_072],
        ),
        (
            "transformer_blocks.0.attn.to_v.weight".to_owned(),
            vec![3_072, 3_072],
        ),
        (
            "transformer_blocks.0.attn.to_out.0.weight".to_owned(),
            vec![3_072, 3_072],
        ),
        (
            "transformer_blocks.1.attn.norm_k.weight".to_owned(),
            vec![128],
        ),
        (
            "single_transformer_blocks.0.attn.norm_k.weight".to_owned(),
            vec![128],
        ),
        (
            "single_transformer_blocks.0.attn.to_q.weight".to_owned(),
            vec![3_072, 3_072],
        ),
        (
            "single_transformer_blocks.0.attn.to_k.weight".to_owned(),
            vec![3_072, 3_072],
        ),
        (
            "single_transformer_blocks.0.attn.to_v.weight".to_owned(),
            vec![3_072, 3_072],
        ),
        (
            "single_transformer_blocks.0.proj_mlp.weight".to_owned(),
            vec![12_288, 3_072],
        ),
        (
            "single_transformer_blocks.0.proj_out.weight".to_owned(),
            vec![3_072, 3_072],
        ),
        (
            "single_transformer_blocks.1.attn.norm_k.weight".to_owned(),
            vec![128],
        ),
        ("proj_out.weight".to_owned(), vec![64, 3_072]),
    ]);
    if vector_embedding {
        tensor_shapes.insert(
            "time_text_embed.text_embedder.linear_1.weight".to_owned(),
            vec![3_072, 768],
        );
    }
    if guidance_embedding {
        tensor_shapes.insert(
            "time_text_embed.guidance_embedder.linear_1.weight".to_owned(),
            vec![3_072, 256],
        );
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

fn diffusers_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let entries: &[(&str, &[u64])] = &[
        ("x_embedder.weight", &[2, 2]),
        ("x_embedder.bias", &[2]),
        ("context_embedder.weight", &[2, 2]),
        ("proj_out.weight", &[2, 2]),
        ("transformer_blocks.0.attn.norm_k.weight", &[2]),
        ("transformer_blocks.0.attn.to_q.weight", &[2, 2]),
        ("transformer_blocks.0.attn.to_k.weight", &[2, 2]),
        ("transformer_blocks.0.attn.to_v.weight", &[2, 2]),
        ("transformer_blocks.0.attn.to_out.0.weight", &[2, 2]),
        ("single_transformer_blocks.0.attn.norm_k.weight", &[2]),
        ("single_transformer_blocks.0.attn.to_q.weight", &[2, 2]),
        ("single_transformer_blocks.0.attn.to_k.weight", &[2, 2]),
        ("single_transformer_blocks.0.attn.to_v.weight", &[2, 2]),
        ("single_transformer_blocks.0.proj_mlp.weight", &[4, 2]),
        ("single_transformer_blocks.0.proj_out.weight", &[2, 2]),
    ];
    entries
        .iter()
        .map(|(key, shape)| Ok(((*key).to_owned(), tensor(backend, context, shape)?)))
        .collect()
}

fn tensor(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    shape: &[u64],
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, backend.device(), context.stream)?;
    let elements = usize::try_from(shape.iter().product::<u64>())?;
    Ok(backend
        .upload_f32(descriptor, &vec![1.0; elements], context)?
        .0)
}

fn write_diffusers_safetensors(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let shapes: &[(&str, &[u64])] = &[
        ("x_embedder.weight", &[128, 64]),
        ("x_embedder.bias", &[128]),
        ("context_embedder.weight", &[128, 4_096]),
        ("transformer_blocks.0.attn.norm_k.weight", &[128]),
        ("transformer_blocks.0.attn.to_q.weight", &[128, 128]),
        ("transformer_blocks.0.attn.to_k.weight", &[128, 128]),
        ("transformer_blocks.0.attn.to_v.weight", &[128, 128]),
        ("transformer_blocks.0.attn.to_out.0.weight", &[128, 128]),
        ("single_transformer_blocks.0.attn.norm_k.weight", &[128]),
        ("single_transformer_blocks.0.attn.to_q.weight", &[128, 128]),
        ("single_transformer_blocks.0.attn.to_k.weight", &[128, 128]),
        ("single_transformer_blocks.0.attn.to_v.weight", &[128, 128]),
        ("single_transformer_blocks.0.proj_mlp.weight", &[512, 128]),
        ("single_transformer_blocks.0.proj_out.weight", &[128, 128]),
        ("time_text_embed.text_embedder.linear_1.weight", &[128, 768]),
        (
            "time_text_embed.guidance_embedder.linear_1.weight",
            &[128, 256],
        ),
        ("proj_out.weight", &[64, 128]),
    ];
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".to_owned(),
        serde_json::json!({"model_layout": "native", "image_model": "chroma"}),
    );
    let mut data = Vec::new();
    for (key, shape) in shapes {
        let start = data.len();
        let elements = shape.iter().try_fold(1_u64, |total, dimension| {
            total
                .checked_mul(*dimension)
                .ok_or("fixture shape overflow")
        })?;
        data.resize(
            data.len()
                .checked_add(usize::try_from(
                    elements.checked_mul(4).ok_or("fixture overflow")?,
                )?)
                .ok_or("fixture overflow")?,
            0,
        );
        header.insert(
            (*key).to_owned(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let header = serde_json::to_vec(&header)?;
    let mut file = fs::File::create(path)?;
    file.write_all(&u64::try_from(header.len())?.to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(&data)?;
    Ok(())
}

fn chroma_probe() -> ModelProbe {
    ModelProbe {
        tensor_shapes: BTreeMap::from([
            (
                "double_blocks.0.img_attn.norm.key_norm.weight".to_string(),
                vec![128],
            ),
            ("single_blocks.0.linear.weight".to_string(), vec![2, 2]),
            (
                "distilled_guidance_layer.norms.0.scale".to_string(),
                vec![5_120],
            ),
        ]),
        metadata: BTreeMap::new(),
    }
}

fn radiance_probe() -> ModelProbe {
    let mut tensor_shapes = BTreeMap::from([
        (
            "double_blocks.0.img_attn.norm.key_norm.scale".to_string(),
            vec![128],
        ),
        ("single_blocks.0.linear.weight".to_string(), vec![2, 2]),
        (
            "distilled_guidance_layer.norms.0.weight".to_string(),
            vec![5_120],
        ),
        ("img_in_patch.weight".to_string(), vec![5_120, 3, 4, 4]),
        ("txt_in.weight".to_string(), vec![5_120, 4_096]),
        ("nerf_final_layer.norm.scale".to_string(), vec![64]),
        ("nerf_final_layer.linear.weight".to_string(), vec![3, 64]),
        ("__x0__".to_string(), vec![1]),
        ("__sequential__".to_string(), vec![1]),
    ]);
    for index in 0..4 {
        tensor_shapes.insert(format!("nerf_blocks.{index}.norm.weight"), vec![64]);
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}
