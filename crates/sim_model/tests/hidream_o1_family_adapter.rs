use sim_model::{
    HIDREAM_O1_ASSISTANT_TOKEN_ID, HIDREAM_O1_BOI_TOKEN_ID, HIDREAM_O1_BOR_TOKEN_ID,
    HIDREAM_O1_BOT_TOKEN_ID, HIDREAM_O1_CLIP_TARGET, HIDREAM_O1_COMPONENT_STATE_SCHEMAS,
    HIDREAM_O1_DEEPSTACK_KEY_FRAGMENT, HIDREAM_O1_EOR_TOKEN_ID, HIDREAM_O1_IM_END_TOKEN_ID,
    HIDREAM_O1_IM_START_TOKEN_ID, HIDREAM_O1_IMAGE_TOKEN_ID, HIDREAM_O1_LATENT_FORMAT,
    HIDREAM_O1_NATIVE_STATE_PLAN, HIDREAM_O1_NEWLINE_TOKEN_ID, HIDREAM_O1_PAD_TOKEN_ID,
    HIDREAM_O1_PATCH_SIZE, HIDREAM_O1_PIXEL_VAE_SENTINEL, HIDREAM_O1_STATE_PLAN_CASES,
    HIDREAM_O1_TEXT_ENCODER_SENTINEL, HIDREAM_O1_TMS_TOKEN_ID, HIDREAM_O1_UNPREFIXED_STATE_PLAN,
    HIDREAM_O1_USER_TOKEN_ID, HIDREAM_O1_VIDEO_TOKEN_ID, HIDREAM_O1_VISION_END_TOKEN_ID,
    HIDREAM_O1_VISION_IMAGE_MEAN, HIDREAM_O1_VISION_IMAGE_STD, HIDREAM_O1_VISION_MERGE_SIZE,
    HIDREAM_O1_VISION_PATCH_SIZE, HIDREAM_O1_VISION_START_TOKEN_ID, HiDreamO1Layout,
    ModelClipModelInvocationDefinition, ModelFamilyError, ModelProbe, ModelStateTransaction,
    ModelStateTransformOperation, hidream_o1_configuration_for_probe,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, StreamId, Tensor, TensorBackend,
    TensorDescriptor,
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::CancellationToken;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn val_model_detection_001_hidream_o1_normalizes_native_and_unprefixed_probes()
-> Result<(), Box<dyn std::error::Error>> {
    let native = hidream_o1_configuration_for_probe(&probe("native"))?;
    let unprefixed = hidream_o1_configuration_for_probe(&probe("unprefixed"))?;

    assert_eq!(native.layout, HiDreamO1Layout::Native);
    assert_eq!(unprefixed.layout, HiDreamO1Layout::Unprefixed);
    assert_eq!(native.patch_size, unprefixed.patch_size);
    assert_eq!(native.input_channels, unprefixed.input_channels);
    assert_eq!(native.patch_dimension, unprefixed.patch_dimension);
    assert_eq!(native.bottleneck_dimension, unprefixed.bottleneck_dimension);
    assert_eq!(native.hidden_size, unprefixed.hidden_size);
    assert_eq!(native.intermediate_size, unprefixed.intermediate_size);
    assert_eq!(native.hidden_layer_count, unprefixed.hidden_layer_count);
    assert_eq!(native.attention_head_count, unprefixed.attention_head_count);
    assert_eq!(native.key_value_head_count, unprefixed.key_value_head_count);
    assert_eq!(
        native.attention_head_dimension,
        unprefixed.attention_head_dimension
    );
    assert_eq!(native.patch_dimension, 3_072);
    Ok(())
}

#[test]
fn val_model_detection_001_hidream_o1_rejects_wrong_partial_cross_family_and_malformed_probes() {
    let mut wrong_layout = probe("native");
    wrong_layout
        .metadata
        .insert("model_layout".to_owned(), "legacy".to_owned());
    assert!(matches!(
        hidream_o1_configuration_for_probe(&wrong_layout),
        Ok(configuration) if configuration.layout == HiDreamO1Layout::Native
    ));

    let mut partial = probe("native");
    partial
        .tensor_shapes
        .remove("model.diffusion_model.x_embedder.proj1.weight");
    assert_invalid(partial, "missing required detector marker");

    let cross_family = ModelProbe {
        tensor_shapes: BTreeMap::from([
            (
                "model.diffusion_model.img_in.weight".to_owned(),
                vec![3_072, 64],
            ),
            (
                "model.diffusion_model.double_blocks.0.img_attn.proj.weight".to_owned(),
                vec![3_072, 3_072],
            ),
        ]),
        metadata: BTreeMap::new(),
    };
    assert!(matches!(
        hidream_o1_configuration_for_probe(&cross_family),
        Err(ModelFamilyError::ModelLayoutSelection(message))
            if message.contains("no supported layout")
    ));

    let mut malformed = probe("unprefixed");
    malformed
        .tensor_shapes
        .insert("x_embedder.proj1.weight".to_owned(), vec![1_024, 4_096]);
    assert_invalid(malformed, "expected [1024, 3072]");
}

#[test]
fn val_model_family_foundation_001_hidream_o1_plans_normalize_and_commit_exact_sentinels()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let native_source = source(&backend, &context, "model.diffusion_model.")?;
    let unprefixed_source = source(&backend, &context, "")?;
    let native = ModelStateTransaction::new(&backend, &context).execute(
        &HIDREAM_O1_NATIVE_STATE_PLAN.compile()?,
        DIGEST,
        &native_source,
    )?;
    let unprefixed = ModelStateTransaction::new(&backend, &context).execute(
        &HIDREAM_O1_UNPREFIXED_STATE_PLAN.compile()?,
        DIGEST,
        &unprefixed_source,
    )?;

    let native_model = native.component("model").ok_or("missing native model")?;
    let unprefixed_model = unprefixed
        .component("model")
        .ok_or("missing unprefixed model")?;
    assert_eq!(
        native_model.keys().collect::<Vec<_>>(),
        unprefixed_model.keys().collect::<Vec<_>>()
    );
    assert!(
        native_model
            .keys()
            .all(|key| !key.contains(HIDREAM_O1_DEEPSTACK_KEY_FRAGMENT))
    );
    assert!(
        unprefixed_model
            .keys()
            .all(|key| !key.contains(HIDREAM_O1_DEEPSTACK_KEY_FRAGMENT))
    );
    assert_eq!(native_model.len(), 5);

    assert_sentinel(
        &backend,
        &context,
        &native,
        "vae",
        HIDREAM_O1_PIXEL_VAE_SENTINEL,
        1.0,
    )?;
    assert_sentinel(
        &backend,
        &context,
        &native,
        "text_encoder",
        HIDREAM_O1_TEXT_ENCODER_SENTINEL,
        0.0,
    )?;
    assert_sentinel(
        &backend,
        &context,
        &unprefixed,
        "vae",
        HIDREAM_O1_PIXEL_VAE_SENTINEL,
        1.0,
    )?;
    assert_sentinel(
        &backend,
        &context,
        &unprefixed,
        "text_encoder",
        HIDREAM_O1_TEXT_ENCODER_SENTINEL,
        0.0,
    )?;

    let native_plan = HIDREAM_O1_NATIVE_STATE_PLAN.compile()?;
    let unprefixed_plan = HIDREAM_O1_UNPREFIXED_STATE_PLAN.compile()?;
    assert_eq!(generate_count(native_plan.operations()), 2);
    assert_eq!(generate_count(unprefixed_plan.operations()), 2);
    assert_eq!(HIDREAM_O1_STATE_PLAN_CASES.len(), 2);
    assert_eq!(HIDREAM_O1_COMPONENT_STATE_SCHEMAS.len(), 3);
    Ok(())
}

#[test]
fn val_cancel_001_hidream_o1_transaction_cancellation_commits_nothing()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    let source = source(&backend, &context, "model.diffusion_model.")?;
    let baseline = backend.memory_snapshot().current_bytes;
    cancellation.cancel();
    let result = ModelStateTransaction::new(&backend, &context).execute(
        &HIDREAM_O1_NATIVE_STATE_PLAN.compile()?,
        DIGEST,
        &source,
    );
    assert!(matches!(result, Err(ModelFamilyError::Cancelled(_))));
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn val_latent_001_hidream_o1_pinned_patch_token_conditioning_and_latent_facts() {
    assert_eq!(HIDREAM_O1_PATCH_SIZE, 32);
    assert_eq!(HIDREAM_O1_VISION_PATCH_SIZE, 16);
    assert_eq!(HIDREAM_O1_VISION_MERGE_SIZE, 2);
    assert_eq!(HIDREAM_O1_VISION_IMAGE_MEAN, [0.5; 3]);
    assert_eq!(HIDREAM_O1_VISION_IMAGE_STD, [0.5; 3]);
    assert_eq!(HIDREAM_O1_PAD_TOKEN_ID, 151_643);
    assert_eq!(HIDREAM_O1_IM_START_TOKEN_ID, 151_644);
    assert_eq!(HIDREAM_O1_IM_END_TOKEN_ID, 151_645);
    assert_eq!(HIDREAM_O1_ASSISTANT_TOKEN_ID, 77_091);
    assert_eq!(HIDREAM_O1_USER_TOKEN_ID, 872);
    assert_eq!(HIDREAM_O1_NEWLINE_TOKEN_ID, 198);
    assert_eq!(HIDREAM_O1_VISION_START_TOKEN_ID, 151_652);
    assert_eq!(HIDREAM_O1_VISION_END_TOKEN_ID, 151_653);
    assert_eq!(HIDREAM_O1_IMAGE_TOKEN_ID, 151_655);
    assert_eq!(HIDREAM_O1_VIDEO_TOKEN_ID, 151_656);
    assert_eq!(HIDREAM_O1_BOI_TOKEN_ID, 151_669);
    assert_eq!(HIDREAM_O1_BOR_TOKEN_ID, 151_670);
    assert_eq!(HIDREAM_O1_EOR_TOKEN_ID, 151_671);
    assert_eq!(HIDREAM_O1_BOT_TOKEN_ID, 151_672);
    assert_eq!(HIDREAM_O1_TMS_TOKEN_ID, 151_673);

    assert_eq!(HIDREAM_O1_LATENT_FORMAT.feature_id, "COMFY-MODEL-0031");
    assert_eq!(HIDREAM_O1_LATENT_FORMAT.identifier, "HiDreamO1Pixel");
    assert_eq!(HIDREAM_O1_LATENT_FORMAT.channels, 3);
    assert_eq!(HIDREAM_O1_LATENT_FORMAT.spatial_downscale_ratio, 1);
    assert_eq!(HIDREAM_O1_LATENT_FORMAT.decoder_name, None);

    assert_eq!(HIDREAM_O1_CLIP_TARGET.candidates.len(), 1);
    let candidate = &HIDREAM_O1_CLIP_TARGET.candidates[0];
    assert_eq!(
        candidate.tokenizer,
        "sim.text_encoders.hidream_o1.HiDreamO1Tokenizer"
    );
    assert_eq!(
        candidate.clip_model,
        "sim.text_encoders.hidream_o1.HiDreamO1TE"
    );
    assert!(matches!(
        candidate.invocation,
        ModelClipModelInvocationDefinition::Reference
    ));
}

#[test]
fn val_ownership_001_hidream_o1_has_one_adapter_and_one_latent_owner()
-> Result<(), Box<dyn std::error::Error>> {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repository_root = crate_root
        .parent()
        .and_then(Path::parent)
        .ok_or("sim_model crate is not inside the repository")?;
    let adapter_path = crate_root.join("src/hidream_o1_family.rs");
    let latent_path = crate_root.join("src/latent_formats/hidreamo1pixel_sim_model_0031.rs");
    let test_path = crate_root.join("tests/hidream_o1_family_adapter.rs");
    let adapter = fs::read_to_string(&adapter_path)?;

    assert_eq!(
        adapter.matches("pub struct HiDreamO1Configuration").count(),
        1
    );
    assert_eq!(adapter.matches("pub enum HiDreamO1Layout").count(), 1);
    assert!(adapter.contains("crate::generated_hidreamo1pixel_comfy_model_0031::LATENT_FORMAT"));
    assert!(!adapter.contains("struct ModelStateTransaction"));
    assert!(!adapter.contains("pub const LATENT_FORMAT: LatentFormatDefinition"));

    let rust_files = rust_files(repository_root)?;
    let adapter_definitions = count_in_files(
        &rust_files,
        "pub struct HiDreamO1Configuration",
        &[&test_path],
    )?;
    assert_eq!(adapter_definitions, 1);
    let patch_owners = count_in_files(
        &rust_files,
        "pub const HIDREAM_O1_PATCH_SIZE",
        &[&test_path],
    )?;
    assert_eq!(patch_owners, 1);
    let deepstack_owners = files_containing(
        &rust_files,
        HIDREAM_O1_DEEPSTACK_KEY_FRAGMENT,
        &[&test_path],
    )?;
    assert_eq!(deepstack_owners, vec![adapter_path]);
    let latent_owners = files_containing(
        &rust_files,
        "pub const LATENT_FORMAT_IDENTIFIER: &str = \"HiDreamO1Pixel\"",
        &[&test_path],
    )?;
    assert_eq!(latent_owners, vec![latent_path]);
    Ok(())
}

fn probe(layout: &str) -> ModelProbe {
    let prefix = if layout == "native" {
        "model.diffusion_model."
    } else {
        ""
    };
    ModelProbe {
        tensor_shapes: BTreeMap::from([
            (
                format!("{prefix}t_embedder1.mlp.0.weight"),
                vec![4_096, 256],
            ),
            (
                format!("{prefix}x_embedder.proj1.weight"),
                vec![1_024, 3_072],
            ),
        ]),
        metadata: BTreeMap::new(),
    }
}

fn assert_invalid(probe: ModelProbe, expected: &str) {
    assert!(matches!(
        hidream_o1_configuration_for_probe(&probe),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains(expected)
    ));
}

fn source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    prefix: &str,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let entries = [
        ("t_embedder1.mlp.0.weight", 1.0),
        ("x_embedder.proj1.weight", 2.0),
        ("visual.patch_embed.proj.weight", 3.0),
        ("visual.deepstack_merger_list.0.weight", 4.0),
        ("language_model.layers.0.self_attn.q_proj.weight", 5.0),
        ("final_layer2.linear.weight", 6.0),
    ];
    let mut source = BTreeMap::new();
    for (key, value) in entries {
        source.insert(format!("{prefix}{key}"), tensor(backend, context, value)?);
    }
    source.insert(
        "vae.decoder.weight".to_owned(),
        tensor(backend, context, 7.0)?,
    );
    source.insert(
        "text_encoders.embedding.weight".to_owned(),
        tensor(backend, context, 8.0)?,
    );
    Ok(source)
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

fn assert_sentinel(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    mapped: &sim_model::MappedModelComponents,
    component: &str,
    key: &str,
    expected: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let component = mapped.component(component).ok_or("missing component")?;
    assert_eq!(component.len(), 1);
    let tensor = component.get(key).ok_or("missing sentinel")?;
    assert_eq!(tensor.descriptor().shape(), &[1]);
    assert_eq!(tensor.descriptor().dtype(), DType::F32);
    assert_eq!(
        &*tensor_to_f32_with_context_exact_native(backend, tensor, context)?,
        &[expected]
    );
    Ok(())
}

fn generate_count(operations: &[ModelStateTransformOperation]) -> usize {
    operations
        .iter()
        .filter(|operation| matches!(operation, ModelStateTransformOperation::Generate { .. }))
        .count()
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

#[test]
fn ownership_scan_ignores_apple_double_metadata() -> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    fs::write(directory.path().join("family.rs"), "fn family() {}\n")?;
    fs::write(directory.path().join("._family.rs"), [0xff])?;
    assert_eq!(
        rust_files(directory.path())?,
        [directory.path().join("family.rs")]
    );
    Ok(())
}

fn count_in_files(
    files: &[PathBuf],
    needle: &str,
    excluded: &[&PathBuf],
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut count = 0;
    for path in files {
        if excluded.contains(&path) {
            continue;
        }
        count += fs::read_to_string(path)?.matches(needle).count();
    }
    Ok(count)
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
