use sim_model::{
    ModelFamilyError, ModelFamilyRegistry, ModelProbe, ModelStateLayout, ModelStateTransaction,
    PatchApplication, PatchKind, PatchOperation, PatchTarget, QWEN_IMAGE_BASE_CONDITIONING_KEYS,
    QWEN_IMAGE_BLOCK_PREFIXES, QWEN_IMAGE_LAYERED_CONDITIONING_KEYS, QwenImageReferenceMethod,
    generated_qwenimage_sim_model_0113 as qwen, qwen_image_checked_patch_graph,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, StreamId, Tensor, TensorBackend,
    generated_sim_operator_indirection_01::{
        tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
};
use sim_types::{CancellationToken, DeviceKind};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};

use super::generated_lumina2_sim_model_0107::support;

const DIGEST: &str = "0113011301130113011301130113011301130113011301130113011301130113";

#[test]
fn source_projection_descriptor_fixture_and_fail_closed_runtime()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(qwen::MODEL_FAMILY_IDENTIFIER, "QwenImage");
    assert_eq!(qwen::MODEL_FAMILY_FEATURE_ID, "COMFY-MODEL-0113");
    assert_eq!(qwen::MODEL_FAMILY_SOURCE_ORDINAL, 77);
    assert_eq!(
        qwen::MODEL_FAMILY_REGISTRATION.source_architecture,
        "model_base.QwenImage"
    );
    assert_eq!(qwen::MODEL_FAMILY_SAMPLING_MULTIPLIER, 1.0);
    assert_eq!(qwen::MODEL_FAMILY_SAMPLING_SHIFT, 1.15);
    assert_eq!(qwen::MODEL_FAMILY_MEMORY_USAGE_FACTOR, 1.8);

    let descriptor = sim_model::describe_model_family(&qwen::MODEL_FAMILY)?;
    assert_eq!(descriptor.architecture_version, "qwen-image-mmdit-v1");
    assert_eq!(descriptor.latent_format, "Wan21");
    assert_eq!(descriptor.supported_dtypes, ["bfloat16", "float32"]);
    assert_eq!(descriptor.supported_devices, [DeviceKind::Cpu]);
    assert_eq!(descriptor.component_graph.len(), 3);

    let aggregate = ModelFamilyRegistry::checked(sim_model::GENERATED_MODEL_FAMILIES)?
        .detect(&probe(ModelStateLayout::PrefixedNative))?;
    assert_eq!(
        aggregate.identity.feature_id(),
        qwen::MODEL_FAMILY_FEATURE_ID,
        "strict QwenImage probe was claimed by {aggregate:?}"
    );

    support::validate_provenance(
        qwen::MODEL_FAMILY_FIXTURE,
        qwen::MODEL_FAMILY_FEATURE_ID,
        qwen::MODEL_FAMILY_IDENTIFIER,
        qwen::MODEL_FAMILY_SOURCE_ORDINAL,
        qwen::MODEL_FAMILY_PROJECTION_SHA256,
    )?;
    verify_catalog_projection()?;
    support::exercise_fixture(
        &qwen::MODEL_FAMILY,
        qwen::MODEL_FAMILY_FIXTURE,
        "qwenimage_comfy_model_0113",
        qwen::MODEL_FAMILY_SOURCE_ORDINAL,
    )?;
    support::assert_leaf_owner("qwenimage_comfy_model_0113", "qwen_image_family")?;
    Ok(())
}

#[test]
fn native_layouts_probe_derived_conditioning_transaction_and_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelFamilyRegistry::checked_registrations(&[
        qwen::MODEL_FAMILY_REGISTRATION,
    ])?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * 1024 * 1024)?,
        &cancellation,
    );

    let mut plan_identities = Vec::new();
    for layout in [
        ModelStateLayout::PrefixedNative,
        ModelStateLayout::StandaloneNative,
    ] {
        let probe = probe(layout);
        let configuration = qwen::configuration_for_probe(&probe)?;
        assert_eq!(configuration.layout, layout);
        assert_eq!(configuration.input_channels, 64);
        assert_eq!(configuration.output_channels, 16);
        assert_eq!(configuration.number_of_layers, 1);
        assert_eq!(configuration.reference_method, QwenImageReferenceMethod::Index);
        assert_eq!(configuration.conditioning_keys, QWEN_IMAGE_BASE_CONDITIONING_KEYS);
        assert_eq!(configuration.latent_format.feature_id, "COMFY-MODEL-0053");
        assert_eq!(configuration.latent_format.identifier, "Wan21");
        assert_eq!(configuration.sampling_shift, qwen::MODEL_FAMILY_SAMPLING_SHIFT);

        let resolved = registry.resolve(&probe)?;
        assert_eq!(resolved.detection().identity.feature_id(), qwen::MODEL_FAMILY_FEATURE_ID);
        assert_eq!(resolved.source_ordinal(), qwen::MODEL_FAMILY_SOURCE_ORDINAL);
        let candidate = resolved
            .clip_target()
            .candidates()
            .first()
            .ok_or("QwenImage CLIP target is missing")?;
        assert_eq!(
            candidate.tokenizer().identifier(),
            "sim.text_encoders.qwen_image.QwenImageTokenizer"
        );
        assert_eq!(
            candidate.clip_model().target().as_str(),
            "sim.text_encoders.qwen_image.te"
        );

        let plan = resolved
            .state_plan()
            .ok_or("QwenImage probe-derived state plan is missing")?;
        plan_identities.push(plan.identity().to_owned());
        let source = mapping_source(&backend, &context, layout)?;
        let mapped = ModelStateTransaction::new(&backend, &context).execute(plan, DIGEST, &source)?;
        let model = mapped.component("model").ok_or("mapped model component")?;
        for key in qwen::MODEL_FAMILY.required_keys {
            assert!(model.contains_key(*key), "missing mapped QwenImage key {key}");
        }
        assert_eq!(mapped.component("vae").map(BTreeMap::len), Some(1));
        assert_eq!(mapped.component("text_encoder").map(BTreeMap::len), Some(1));
        assert_fact(&backend, &context, model, "native.__sampling_shift__", 1.15)?;
        assert_fact(&backend, &context, model, "native.__reference_method__", 0.0)?;
        assert_fact(
            &backend,
            &context,
            model,
            "native.__additional_timestep_condition__",
            0.0,
        )?;
    }
    assert_ne!(plan_identities[0], plan_identities[1]);

    let mut timestep_zero = probe(ModelStateLayout::StandaloneNative);
    timestep_zero
        .tensor_shapes
        .insert("__index_timestep_zero__".to_owned(), vec![]);
    let timestep_zero_configuration = qwen::configuration_for_probe(&timestep_zero)?;
    assert_eq!(
        timestep_zero_configuration.reference_method,
        QwenImageReferenceMethod::IndexTimestepZero
    );
    let timestep_zero_plan = registry.resolve(&timestep_zero)?;

    let mut layered = timestep_zero.clone();
    layered.tensor_shapes.insert(
        "time_text_embed.addition_t_embedding.weight".to_owned(),
        vec![2, 3_072],
    );
    let layered_configuration = qwen::configuration_for_probe(&layered)?;
    assert_eq!(
        layered_configuration.reference_method,
        QwenImageReferenceMethod::NegativeIndex
    );
    assert_eq!(
        layered_configuration.conditioning_keys,
        QWEN_IMAGE_LAYERED_CONDITIONING_KEYS
    );
    assert!(layered_configuration.use_additional_timestep_condition);
    let layered_plan = registry.resolve(&layered)?;
    assert_ne!(
        timestep_zero_plan.state_plan().map(|plan| plan.identity()),
        layered_plan.state_plan().map(|plan| plan.identity())
    );

    assert_eq!(QWEN_IMAGE_BLOCK_PREFIXES.len(), 3);
    let first_patch = patch("first", 0.5);
    let second_patch = patch("second", 0.25);
    let forward = qwen_image_checked_patch_graph(
        DIGEST,
        vec![first_patch.clone(), second_patch.clone()],
    )?;
    let reverse = qwen_image_checked_patch_graph(DIGEST, vec![second_patch, first_patch])?;
    assert_ne!(forward.identity().ordered_digest, reverse.identity().ordered_digest);
    let invalid_patch = PatchOperation {
        identifier: "invalid-output-patch".to_owned(),
        kind: PatchKind::Lora,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.proj_out.weight".to_owned(),
            expected_shape: vec![2, 2],
            values: vec![0.0; 4],
            application: PatchApplication::Add,
        }],
    };
    assert!(qwen_image_checked_patch_graph(DIGEST, vec![invalid_patch]).is_err());

    let mut partial = probe(ModelStateLayout::StandaloneNative);
    partial.tensor_shapes.remove("img_in.weight");
    assert!(matches!(
        registry.resolve(&partial),
        Err(ModelFamilyError::ModelLayoutSelection(_))
    ));
    let mut ambiguous = probe(ModelStateLayout::PrefixedNative);
    ambiguous
        .tensor_shapes
        .extend(probe(ModelStateLayout::StandaloneNative).tensor_shapes);
    assert!(matches!(
        qwen::configuration_for_probe(&ambiguous),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("ambiguously")
    ));
    let diffusers = probe(ModelStateLayout::Diffusers);
    assert!(matches!(
        registry.resolve(&diffusers),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains("Diffusers")
    ));
    let mut misleading = probe(ModelStateLayout::StandaloneNative);
    misleading
        .metadata
        .insert("image_model".to_owned(), "lens".to_owned());
    assert_eq!(
        registry.resolve(&misleading)?.detection().identity.feature_id(),
        qwen::MODEL_FAMILY_FEATURE_ID
    );

    let resolved = registry.resolve(&probe(ModelStateLayout::StandaloneNative))?;
    let plan = resolved.state_plan().ok_or("standalone state plan")?;
    let mut unexpected = mapping_source(
        &backend,
        &context,
        ModelStateLayout::StandaloneNative,
    )?;
    unexpected.insert(
        "diffusers.unexpected.weight".to_owned(),
        tensor(&backend, &context, &[1], &[1.0])?,
    );
    let baseline = backend.memory_snapshot().current_bytes;
    assert!(matches!(
        ModelStateTransaction::new(&backend, &context).execute(plan, DIGEST, &unexpected),
        Err(ModelFamilyError::UnexpectedKeys(keys))
            if keys == ["diffusers.unexpected.weight"]
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::new(113),
        authority.authorize_workspace(4 * 1024 * 1024)?,
        &cancelled,
    );
    let source = mapping_source(
        &backend,
        &context,
        ModelStateLayout::StandaloneNative,
    )?;
    assert!(matches!(
        ModelStateTransaction::new(&backend, &cancelled_context).execute(plan, DIGEST, &source),
        Err(ModelFamilyError::Cancelled(_))
    ));
    Ok(())
}

fn probe(layout: ModelStateLayout) -> ModelProbe {
    let prefix = match layout {
        ModelStateLayout::PrefixedNative => "model.diffusion_model.",
        ModelStateLayout::StandaloneNative => "",
        ModelStateLayout::Diffusers => "transformer.",
    };
    ModelProbe {
        tensor_shapes: BTreeMap::from([
            (format!("{prefix}txt_norm.weight"), vec![3_584]),
            (format!("{prefix}img_in.weight"), vec![3_072, 64]),
            (format!("{prefix}txt_in.weight"), vec![3_072, 3_584]),
            (format!("{prefix}proj_out.weight"), vec![64, 3_072]),
            (
                format!("{prefix}transformer_blocks.0.img_mod.1.weight"),
                vec![18_432, 3_072],
            ),
            (
                format!("{prefix}transformer_blocks.0.txt_mod.1.weight"),
                vec![18_432, 3_072],
            ),
            (
                format!("{prefix}transformer_blocks.0.attn.to_q.weight"),
                vec![3_072, 3_072],
            ),
            (
                format!("{prefix}transformer_blocks.0.attn.norm_q.weight"),
                vec![128],
            ),
        ]),
        metadata: BTreeMap::new(),
    }
}

fn mapping_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: ModelStateLayout,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = match layout {
        ModelStateLayout::PrefixedNative => "model.diffusion_model.",
        ModelStateLayout::StandaloneNative => "",
        ModelStateLayout::Diffusers => return Err("Diffusers mapping is unsupported".into()),
    };
    let mut source = BTreeMap::new();
    for (key, shape) in [
        ("txt_norm.weight", vec![2]),
        ("img_in.weight", vec![2, 2]),
        ("txt_in.weight", vec![2, 2]),
        ("transformer_blocks.0.img_mod.1.weight", vec![2, 2]),
        ("transformer_blocks.0.txt_mod.1.weight", vec![2, 2]),
        ("transformer_blocks.0.attn.to_q.weight", vec![2, 2]),
        ("proj_out.weight", vec![2, 2]),
    ] {
        let elements = usize::try_from(shape.iter().product::<u64>())?;
        source.insert(
            format!("{prefix}{key}"),
            tensor(backend, context, &shape, &vec![1.0; elements])?,
        );
    }
    source.insert(
        "vae.decoder.weight".to_owned(),
        tensor(backend, context, &[1], &[1.0])?,
    );
    source.insert(
        "text_encoders.qwen25_7b.transformer.weight".to_owned(),
        tensor(backend, context, &[1], &[1.0])?,
    );
    Ok(source)
}

fn patch(identifier: &str, value: f32) -> PatchOperation {
    PatchOperation {
        identifier: identifier.to_owned(),
        kind: PatchKind::Lora,
        scale: 1.0,
        targets: vec![PatchTarget {
            key: "native.transformer_blocks.0.img_mod.1.weight".to_owned(),
            expected_shape: vec![2, 2],
            values: vec![value, 0.0, 0.0, value],
            application: PatchApplication::Add,
        }],
    }
}

fn tensor(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    shape: &[u64],
    values: &[f32],
) -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(tensor_from_f32_with_context_exact_native(
        backend,
        shape,
        values,
        DType::F32,
        backend.device(),
        context,
    )?)
}

fn assert_fact(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    model: &BTreeMap<String, Tensor>,
    key: &str,
    expected: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let tensor = model.get(key).ok_or("generated QwenImage conditioning fact")?;
    let values = tensor_to_f32_with_context_exact_native(backend, tensor, context)?;
    assert_eq!(values, [expected]);
    Ok(())
}

fn verify_catalog_projection() -> Result<(), Box<dyn std::error::Error>> {
    let catalog: serde_json::Value = serde_json::from_slice(&std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("catalog/model-families-v1.json"),
    )?)?;
    let row = catalog["models"]
        .as_array()
        .ok_or("model catalog rows")?
        .iter()
        .find(|row| row["feature_id"] == qwen::MODEL_FAMILY_FEATURE_ID)
        .ok_or("QwenImage catalog row")?;
    assert_eq!(row["source_ordinal"], qwen::MODEL_FAMILY_SOURCE_ORDINAL);
    assert_eq!(row["static"]["unet_config"]["value"]["image_model"], "qwen_image");
    assert_eq!(
        sha256(&serde_json::to_vec(row)?),
        qwen::MODEL_FAMILY_CATALOG_ROW_SHA256
    );
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
