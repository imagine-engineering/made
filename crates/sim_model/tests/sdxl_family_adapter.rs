use sim_model::{
    ModelFamilyError, ModelProbe, ModelStateTransaction, SDXL_CLIP_TARGET, SDXL_COMMON_MAPPING,
    SDXL_DIFFUSERS_STATE_PLAN, SDXL_FORWARD_PROGRAM, SDXL_KOALA_1B_TRANSFORMER_DEPTH,
    SDXL_KOALA_700M_TRANSFORMER_DEPTH, SDXL_LATENT_FORMAT, SDXL_MEMORY_USAGE_FACTOR,
    SDXL_PREFIXED_STATE_PLAN, SDXL_REFINER_CLIP_TARGET, SDXL_REFINER_MEMORY_USAGE_FACTOR,
    SDXL_REFINER_TRANSFORMER_DEPTH, SDXL_SEGMIND_TRANSFORMER_DEPTH, SDXL_SSD1B_TRANSFORMER_DEPTH,
    SDXL_STANDALONE_STATE_PLAN, SDXL_TRANSFORMER_DEPTH, SdxlLayout, SdxlVariant,
    sdxl_common_mapping, sdxl_configuration_for_probe, sdxl_state_plan_for_layout,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, StreamId, Tensor, TensorBackend,
    TensorDescriptor,
};
use sim_types::CancellationToken;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn val_model_detection_001_sdxl_variants_layouts_profiles_and_precedence_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let cases = [
        (
            SdxlVariant::InstructPix2Pix,
            SdxlLayout::PrefixedNative,
            8,
            320,
            2_048,
            2_816,
            10,
            SDXL_TRANSFORMER_DEPTH,
            0.8,
        ),
        (
            SdxlVariant::Refiner,
            SdxlLayout::StandaloneNative,
            4,
            384,
            1_280,
            2_560,
            4,
            SDXL_REFINER_TRANSFORMER_DEPTH,
            1.0,
        ),
        (
            SdxlVariant::Base,
            SdxlLayout::Diffusers,
            4,
            320,
            2_048,
            2_816,
            10,
            SDXL_TRANSFORMER_DEPTH,
            0.8,
        ),
        (
            SdxlVariant::Ssd1B,
            SdxlLayout::PrefixedNative,
            4,
            320,
            2_048,
            2_816,
            4,
            SDXL_SSD1B_TRANSFORMER_DEPTH,
            0.8,
        ),
        (
            SdxlVariant::Koala700M,
            SdxlLayout::StandaloneNative,
            4,
            320,
            2_048,
            2_816,
            5,
            SDXL_KOALA_700M_TRANSFORMER_DEPTH,
            0.8,
        ),
        (
            SdxlVariant::Koala1B,
            SdxlLayout::Diffusers,
            4,
            320,
            2_048,
            2_816,
            6,
            SDXL_KOALA_1B_TRANSFORMER_DEPTH,
            0.8,
        ),
        (
            SdxlVariant::SegmindVega,
            SdxlLayout::PrefixedNative,
            4,
            320,
            2_048,
            2_816,
            2,
            SDXL_SEGMIND_TRANSFORMER_DEPTH,
            0.8,
        ),
    ];
    for (variant, layout, input, model, context, adm, depth, expected_depths, memory) in cases {
        let configuration = sdxl_configuration_for_probe(&probe(layout, variant))?;
        assert_eq!(configuration.variant, variant);
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.in_channels, input);
        assert_eq!(configuration.model_channels, model);
        assert_eq!(configuration.context_dimension, context);
        assert_eq!(configuration.adm_in_channels, adm);
        assert_eq!(configuration.transformer_depth, expected_depths);
        assert_eq!(
            configuration.transformer_depth.iter().copied().max(),
            Some(depth)
        );
        assert_eq!(configuration.memory_usage_factor, memory);
        assert_eq!(configuration.latent_format.feature_id, "COMFY-MODEL-0047");
        assert!(configuration.uses_linear_transformer_projection);
        assert!(!configuration.uses_temporal_attention);
        if variant == SdxlVariant::Refiner {
            assert!(std::ptr::eq(
                configuration.clip_target,
                &SDXL_REFINER_CLIP_TARGET
            ));
        } else {
            assert!(std::ptr::eq(configuration.clip_target, &SDXL_CLIP_TARGET));
        }
    }
    assert_eq!(SDXL_MEMORY_USAGE_FACTOR, 0.8);
    assert_eq!(SDXL_REFINER_MEMORY_USAGE_FACTOR, 1.0);
    Ok(())
}

#[test]
fn val_model_detection_001_sdxl_rejects_partial_mixed_cross_family_gaps_and_bad_shapes() {
    let mut partial = probe(SdxlLayout::StandaloneNative, SdxlVariant::Base);
    partial.tensor_shapes.remove("label_emb.0.0.weight");
    assert!(matches!(
        sdxl_configuration_for_probe(&partial),
        Err(ModelFamilyError::ModelLayoutSelection(_))
    ));

    let mut bad_kernel = probe(SdxlLayout::PrefixedNative, SdxlVariant::Base);
    bad_kernel.tensor_shapes.insert(
        "model.diffusion_model.input_blocks.0.0.weight".to_owned(),
        vec![320, 4, 1, 1],
    );
    assert_invalid(bad_kernel, "input_blocks.0.0.weight shape");

    let mut gap = probe(SdxlLayout::StandaloneNative, SdxlVariant::Base);
    gap.tensor_shapes
        .remove("input_blocks.7.1.transformer_blocks.1.attn2.to_k.weight");
    assert_invalid(gap, "not bounded and consecutive");

    let mut unsupported = probe(SdxlLayout::Diffusers, SdxlVariant::Base);
    unsupported
        .tensor_shapes
        .insert("conv_in.weight".to_owned(), vec![320, 9, 3, 3]);
    assert_invalid(unsupported, "unsupported standard input/depth pair");

    let mut wrong_context = probe(SdxlLayout::PrefixedNative, SdxlVariant::Refiner);
    wrong_context.tensor_shapes.insert(
        "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_k.weight".to_owned(),
        vec![384, 2_048],
    );
    assert_invalid(wrong_context, "unsupported channels/context/ADM profile");

    let mut mixed = probe(SdxlLayout::PrefixedNative, SdxlVariant::Base);
    mixed
        .tensor_shapes
        .extend(probe(SdxlLayout::StandaloneNative, SdxlVariant::Base).tensor_shapes);
    assert!(matches!(
        sdxl_configuration_for_probe(&mixed),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("ambiguously")
    ));

    let sd15 = ModelProbe {
        tensor_shapes: BTreeMap::from([
            ("input_blocks.0.0.weight".to_owned(), vec![320, 4, 3, 3]),
            ("time_embed.0.weight".to_owned(), vec![1_280, 320]),
            ("out.2.weight".to_owned(), vec![4, 320, 3, 3]),
        ]),
        metadata: BTreeMap::new(),
    };
    assert!(matches!(
        sdxl_configuration_for_probe(&sd15),
        Err(ModelFamilyError::ModelLayoutSelection(_))
    ));
}

#[test]
fn val_model_family_row_001_sdxl_native_and_diffusers_plans_are_transactional()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    for layout in [
        SdxlLayout::PrefixedNative,
        SdxlLayout::StandaloneNative,
        SdxlLayout::Diffusers,
    ] {
        let mapped = ModelStateTransaction::new(&backend, &context).execute(
            &sdxl_state_plan_for_layout(layout).compile()?,
            DIGEST,
            &mapping_source(&backend, &context, layout)?,
        )?;
        let denoiser = mapped.component("denoiser").ok_or("missing denoiser")?;
        for required in [
            "native.input_blocks.0.0.weight",
            "native.time_embed.0.weight",
            "native.label_emb.0.0.weight",
            "native.out.2.weight",
        ] {
            assert!(denoiser.contains_key(required), "{layout:?}: {required}");
        }
        assert_eq!(mapped.component("text_encoder").map(BTreeMap::len), Some(2));
        assert_eq!(mapped.component("vae").map(BTreeMap::len), Some(1));
    }
    assert_eq!(
        sdxl_state_plan_for_layout(SdxlLayout::PrefixedNative).encoded_plan,
        SDXL_PREFIXED_STATE_PLAN.encoded_plan
    );
    assert_eq!(
        sdxl_state_plan_for_layout(SdxlLayout::StandaloneNative).encoded_plan,
        SDXL_STANDALONE_STATE_PLAN.encoded_plan
    );
    assert_eq!(
        sdxl_state_plan_for_layout(SdxlLayout::Diffusers).encoded_plan,
        SDXL_DIFFUSERS_STATE_PLAN.encoded_plan
    );
    Ok(())
}

#[test]
fn val_cancel_001_sdxl_mapping_observes_cancellation_without_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(512 * 1024)?,
        &cancellation,
    );
    let source = mapping_source(&backend, &context, SdxlLayout::PrefixedNative)?;
    cancellation.cancel();
    assert!(
        ModelStateTransaction::new(&backend, &context)
            .execute(&SDXL_PREFIXED_STATE_PLAN.compile()?, DIGEST, &source)
            .is_err()
    );
    Ok(())
}

#[test]
fn val_ownership_001_sdxl_has_one_adapter_latent_and_foundational_owners()
-> Result<(), Box<dyn std::error::Error>> {
    assert!(std::ptr::eq(sdxl_common_mapping(), &SDXL_COMMON_MAPPING));
    assert_eq!(
        sdxl_common_mapping().latent_format.feature_id,
        SDXL_LATENT_FORMAT.feature_id
    );
    assert_eq!(sdxl_common_mapping().components.len(), 3);
    assert_eq!(sdxl_common_mapping().component_state_schemas.len(), 3);
    assert_eq!(sdxl_common_mapping().forward_program, SDXL_FORWARD_PROGRAM);

    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repository_root = crate_root
        .parent()
        .and_then(Path::parent)
        .ok_or("sim_model crate is not inside the repository")?;
    let test_path = crate_root.join("tests/sdxl_family_adapter.rs");
    let adapter_path = crate_root.join("src/sdxl_family.rs");
    let latent_path = crate_root.join("src/latent_formats/sdxl_sim_model_0047.rs");
    let foundation_path = crate_root.join("src/model_family.rs");
    let files = rust_files(repository_root)?;
    assert_eq!(
        files_containing(&files, "pub struct SdxlConfiguration", &[&test_path])?,
        vec![adapter_path.clone()]
    );
    let latent_owners = files_containing(
        &files,
        "pub const LATENT_FORMAT: LatentFormatDefinition",
        &[&test_path],
    )?
    .into_iter()
    .filter(|path| {
        fs::read_to_string(path)
            .is_ok_and(|source| source.contains("feature_id: \"COMFY-MODEL-0047\""))
    })
    .collect::<Vec<_>>();
    assert_eq!(latent_owners, vec![latent_path]);
    assert_eq!(
        files_containing(&files, "pub struct ModelStateTransaction", &[&test_path])?,
        vec![foundation_path]
    );
    let adapter = fs::read_to_string(adapter_path)?;
    for forbidden in [
        "struct ModelStateTransaction",
        "struct PatchGraph",
        "fn estimate_model_memory",
        "struct CancellationToken",
    ] {
        assert!(!adapter.contains(forbidden));
    }
    Ok(())
}

fn probe(layout: SdxlLayout, variant: SdxlVariant) -> ModelProbe {
    let (model_channels, in_channels, context, adm, depth, standard) = match variant {
        SdxlVariant::InstructPix2Pix => (320, 8, 2_048, 2_816, 10, true),
        SdxlVariant::Refiner => (384, 4, 1_280, 2_560, 4, true),
        SdxlVariant::Base => (320, 4, 2_048, 2_816, 10, true),
        SdxlVariant::Ssd1B => (320, 4, 2_048, 2_816, 4, true),
        SdxlVariant::Koala700M => (320, 4, 2_048, 2_816, 5, false),
        SdxlVariant::Koala1B => (320, 4, 2_048, 2_816, 6, false),
        SdxlVariant::SegmindVega => (320, 4, 2_048, 2_816, 2, true),
    };
    let mut tensor_shapes = BTreeMap::new();
    match layout {
        SdxlLayout::PrefixedNative | SdxlLayout::StandaloneNative => {
            let prefix = if layout == SdxlLayout::PrefixedNative {
                "model.diffusion_model."
            } else {
                ""
            };
            tensor_shapes.insert(
                format!("{prefix}input_blocks.0.0.weight"),
                vec![model_channels, in_channels, 3, 3],
            );
            tensor_shapes.insert(
                format!("{prefix}time_embed.0.weight"),
                vec![model_channels * 4, model_channels],
            );
            tensor_shapes.insert(
                format!("{prefix}label_emb.0.0.weight"),
                vec![model_channels * 4, adm],
            );
            tensor_shapes.insert(
                format!("{prefix}out.2.weight"),
                vec![4, model_channels, 3, 3],
            );
            if standard {
                tensor_shapes.insert(
                    format!("{prefix}input_blocks.2.0.in_layers.0.weight"),
                    vec![model_channels],
                );
                add_depth(
                    &mut tensor_shapes,
                    &format!("{prefix}input_blocks.7.1.transformer_blocks."),
                    depth,
                    model_channels,
                    context,
                );
            } else {
                add_depth(
                    &mut tensor_shapes,
                    &format!("{prefix}input_blocks.3.1.transformer_blocks."),
                    2,
                    model_channels,
                    context,
                );
                add_depth(
                    &mut tensor_shapes,
                    &format!("{prefix}input_blocks.5.1.transformer_blocks."),
                    depth,
                    model_channels,
                    context,
                );
            }
        }
        SdxlLayout::Diffusers => {
            tensor_shapes.insert(
                "conv_in.weight".to_owned(),
                vec![model_channels, in_channels, 3, 3],
            );
            tensor_shapes.insert(
                "time_embedding.linear_1.weight".to_owned(),
                vec![model_channels * 4, model_channels],
            );
            tensor_shapes.insert(
                "add_embedding.linear_1.weight".to_owned(),
                vec![model_channels * 4, adm],
            );
            tensor_shapes.insert("conv_out.weight".to_owned(), vec![4, model_channels, 3, 3]);
            if standard {
                tensor_shapes.insert(
                    "down_blocks.0.resnets.1.conv1.weight".to_owned(),
                    vec![model_channels],
                );
            } else {
                add_depth(
                    &mut tensor_shapes,
                    "down_blocks.1.attentions.0.transformer_blocks.",
                    2,
                    model_channels,
                    context,
                );
            }
            add_depth(
                &mut tensor_shapes,
                "down_blocks.2.attentions.0.transformer_blocks.",
                depth,
                model_channels,
                context,
            );
        }
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

fn add_depth(
    tensors: &mut BTreeMap<String, Vec<u64>>,
    prefix: &str,
    depth: usize,
    hidden: u64,
    context: u64,
) {
    for index in 0..depth {
        tensors.insert(
            format!("{prefix}{index}.attn2.to_k.weight"),
            vec![hidden, context],
        );
    }
}

fn assert_invalid(probe: ModelProbe, expected: &str) {
    assert!(matches!(
        sdxl_configuration_for_probe(&probe),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains(expected)
    ));
}

fn mapping_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: SdxlLayout,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let keys: &[&str] = match layout {
        SdxlLayout::PrefixedNative => &[
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.time_embed.0.weight",
            "model.diffusion_model.label_emb.0.0.weight",
            "model.diffusion_model.out.2.weight",
            "conditioner.embedders.0.transformer.text_model.weight",
            "conditioner.embedders.1.model.weight",
            "first_stage_model.decoder.weight",
        ],
        SdxlLayout::StandaloneNative => &[
            "input_blocks.0.0.weight",
            "time_embed.0.weight",
            "label_emb.0.0.weight",
            "out.2.weight",
            "text_encoders.clip_l.weight",
            "text_encoders.clip_g.weight",
            "vae.decoder.weight",
        ],
        SdxlLayout::Diffusers => &[
            "conv_in.weight",
            "time_embedding.linear_1.weight",
            "add_embedding.linear_1.weight",
            "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight",
            "conv_out.weight",
            "text_encoder.weight",
            "text_encoder_2.weight",
            "vae.decoder.weight",
        ],
    };
    keys.iter()
        .enumerate()
        .map(|(index, key)| {
            Ok((
                (*key).to_owned(),
                tensor(backend, context, index as f32 + 1.0)?,
            ))
        })
        .collect()
}

fn tensor(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    value: f32,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(vec![1], DType::F32, backend.device(), context.stream)?;
    Ok(backend.upload_f32(descriptor, &[value], context)?.0)
}

fn rust_files(root: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("._"))
            {
                continue;
            }
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name == "target") {
                    continue;
                }
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

fn files_containing(
    files: &[PathBuf],
    needle: &str,
    excluded: &[&PathBuf],
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut matching = Vec::new();
    for path in files {
        if excluded.contains(&path) {
            continue;
        }
        if fs::read_to_string(path)?.contains(needle) {
            matching.push(path.clone());
        }
    }
    Ok(matching)
}
