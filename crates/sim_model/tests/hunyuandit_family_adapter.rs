use sim_model::{
    HUNYUANDIT_CLIP_TARGET, HUNYUANDIT_COMMON_MAPPING, HUNYUANDIT_DIT1_EXTRA_INPUT,
    HUNYUANDIT_G_DEPTH, HUNYUANDIT_G_HIDDEN_SIZE, HUNYUANDIT_G_MLP_RATIO, HUNYUANDIT_LATENT_FORMAT,
    HUNYUANDIT_LINEAR_END, HUNYUANDIT_PREFIXED_STATE_PLAN, HUNYUANDIT_SAVED_MODEL_STATE_PLAN,
    HUNYUANDIT_STANDALONE_STATE_PLAN, HUNYUANDIT1_LINEAR_END, HunyuanDiTAttentionPrecision,
    HunyuanDiTLayout, HunyuanDiTVariant, ModelDetectionRule, ModelFamilyError, ModelFamilyIdentity,
    ModelParsedFacts, ModelParsedTensorFact, ModelProbe, ModelStateTransaction,
    ModelTensorFactPredicate, ModelTensorFactRelation, ModelTensorFactSubject,
    detect_model_family_rules, hunyuandit_common_mapping, hunyuandit_configuration_for_probe,
    hunyuandit_state_plan_for_layout,
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
const DIT1_FACTS: &[ModelTensorFactPredicate] = &[
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Rank,
        relation: ModelTensorFactRelation::GreaterThanOrEqual,
        value: 2,
    },
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Rank,
        relation: ModelTensorFactRelation::LessThanOrEqual,
        value: 2,
    },
    ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Dimension(1),
        relation: ModelTensorFactRelation::Equal,
        value: HUNYUANDIT_DIT1_EXTRA_INPUT,
    },
];
const DIT1_RULES: &[ModelDetectionRule] = &[ModelDetectionRule::AnyTensorFact {
    keys: &[
        "model.diffusion_model.extra_embedder.0.weight",
        "model.extra_embedder.0.weight",
        "extra_embedder.0.weight",
    ],
    predicates: DIT1_FACTS,
    score: 300,
}];

#[test]
fn val_model_detection_001_hunyuandit_tensor_facts_select_variants_layouts_and_profiles()
-> Result<(), Box<dyn std::error::Error>> {
    let base = hunyuandit_configuration_for_probe(&probe(
        HunyuanDiTLayout::StandaloneNative,
        HunyuanDiTVariant::DiT,
        1_152,
        28,
    ))?;
    assert_eq!(base.variant, HunyuanDiTVariant::DiT);
    assert_eq!(base.layout, HunyuanDiTLayout::StandaloneNative);
    assert_eq!(base.hidden_size, 1_152);
    assert_eq!(base.depth, 28);
    assert_eq!(base.mlp_ratio, 4.0);
    assert!(!base.size_conditioning);
    assert!(!base.style_conditioning);
    assert_eq!(
        base.attention_precision,
        HunyuanDiTAttentionPrecision::Float32
    );
    assert_eq!(base.sampling_linear_end, HUNYUANDIT_LINEAR_END);
    assert_eq!(
        base.latent_format.feature_id,
        HUNYUANDIT_LATENT_FORMAT.feature_id
    );

    let dit1 = hunyuandit_configuration_for_probe(&probe(
        HunyuanDiTLayout::PrefixedNative,
        HunyuanDiTVariant::DiT1,
        HUNYUANDIT_G_HIDDEN_SIZE,
        HUNYUANDIT_G_DEPTH,
    ))?;
    assert_eq!(dit1.variant, HunyuanDiTVariant::DiT1);
    assert_eq!(dit1.layout, HunyuanDiTLayout::PrefixedNative);
    assert_eq!(dit1.extra_input_dimension, HUNYUANDIT_DIT1_EXTRA_INPUT);
    assert_eq!(dit1.mlp_ratio, HUNYUANDIT_G_MLP_RATIO);
    assert!(dit1.size_conditioning);
    assert!(dit1.style_conditioning);
    assert_eq!(
        dit1.attention_precision,
        HunyuanDiTAttentionPrecision::Inherited
    );
    assert_eq!(dit1.sampling_linear_end, HUNYUANDIT1_LINEAR_END);

    let saved_g = hunyuandit_configuration_for_probe(&probe(
        HunyuanDiTLayout::SavedModel,
        HunyuanDiTVariant::DiT,
        HUNYUANDIT_G_HIDDEN_SIZE,
        HUNYUANDIT_G_DEPTH,
    ))?;
    assert_eq!(saved_g.layout, HunyuanDiTLayout::SavedModel);
    assert_eq!(saved_g.variant, HunyuanDiTVariant::DiT);
    assert_eq!(saved_g.mlp_ratio, HUNYUANDIT_G_MLP_RATIO);

    let parsed = ModelProbe::from_parsed_facts(ModelParsedFacts {
        tensors: BTreeMap::from([(
            "extra_embedder.0.weight".to_owned(),
            ModelParsedTensorFact {
                shape: vec![5_632, HUNYUANDIT_DIT1_EXTRA_INPUT],
                storage_dtype: DType::F32.catalog_name().to_owned(),
            },
        )]),
        formats: Vec::new(),
    })?;
    let detection = detect_model_family_rules(
        ModelFamilyIdentity::new("COMFY-MODEL-9088", "HunyuanDiT1Predicate", "v1")?,
        DIT1_RULES,
        &parsed,
    )?;
    assert_eq!(detection.score, 300);
    assert!(parsed.metadata.keys().all(|key| key.starts_with("__sim.")));
    Ok(())
}

#[test]
fn val_model_family_foundation_001_tensor_fact_comparisons_are_bounded_and_typed()
-> Result<(), Box<dyn std::error::Error>> {
    let identity = ModelFamilyIdentity::new("COMFY-MODEL-9087", "TensorFactBounds", "v1")?;
    let mut wrong_width = ModelProbe::default();
    wrong_width
        .tensor_shapes
        .insert("extra_embedder.0.weight".to_owned(), vec![4_608, 1_024]);
    assert!(matches!(
        detect_model_family_rules(identity.clone(), DIT1_RULES, &wrong_width),
        Err(ModelFamilyError::NoDetectionMatch)
    ));

    const BAD_RANK: &[ModelTensorFactPredicate] = &[ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Rank,
        relation: ModelTensorFactRelation::Equal,
        value: 33,
    }];
    const BAD_DIMENSION: &[ModelTensorFactPredicate] = &[ModelTensorFactPredicate {
        subject: ModelTensorFactSubject::Dimension(32),
        relation: ModelTensorFactRelation::NotEqual,
        value: 0,
    }];
    const DUPLICATE: &[ModelTensorFactPredicate] = &[
        ModelTensorFactPredicate {
            subject: ModelTensorFactSubject::Dimension(1),
            relation: ModelTensorFactRelation::GreaterThan,
            value: 1,
        },
        ModelTensorFactPredicate {
            subject: ModelTensorFactSubject::Dimension(1),
            relation: ModelTensorFactRelation::GreaterThan,
            value: 1,
        },
    ];
    for predicates in [BAD_RANK, BAD_DIMENSION, DUPLICATE, &[]] {
        let rules = [ModelDetectionRule::AnyTensorFact {
            keys: &["extra_embedder.0.weight"],
            predicates,
            score: 1,
        }];
        assert!(matches!(
            detect_model_family_rules(identity.clone(), &rules, &wrong_width),
            Err(ModelFamilyError::InvalidDefinition(_))
                | Err(ModelFamilyError::DuplicateDefinitionValue(_))
        ));
    }
    Ok(())
}

#[test]
fn val_model_detection_001_hunyuandit_rejects_partial_mixed_cross_family_and_malformed_probes() {
    let mut partial = probe(
        HunyuanDiTLayout::StandaloneNative,
        HunyuanDiTVariant::DiT,
        1_152,
        2,
    );
    partial.tensor_shapes.remove("extra_embedder.0.weight");
    assert_invalid(partial, "partial marker set");

    let mut malformed = probe(
        HunyuanDiTLayout::StandaloneNative,
        HunyuanDiTVariant::DiT,
        1_152,
        2,
    );
    malformed
        .tensor_shapes
        .insert("extra_embedder.0.weight".to_owned(), vec![4_608, 2_000]);
    assert_invalid(malformed, "neither base DiT");

    let mut bad_rank = probe(
        HunyuanDiTLayout::StandaloneNative,
        HunyuanDiTVariant::DiT,
        1_152,
        2,
    );
    bad_rank
        .tensor_shapes
        .insert("x_embedder.proj.weight".to_owned(), vec![1_152, 4, 2]);
    assert_invalid(bad_rank, "x_embedder.proj.weight shape");

    let mut gap = probe(
        HunyuanDiTLayout::StandaloneNative,
        HunyuanDiTVariant::DiT,
        1_152,
        2,
    );
    gap.tensor_shapes.remove("blocks.1.attn1.qkv.weight");
    gap.tensor_shapes
        .insert("blocks.2.attn1.qkv.weight".to_owned(), vec![3_456, 1_152]);
    assert_invalid(gap, "not consecutive");

    let mut collision = probe(
        HunyuanDiTLayout::StandaloneNative,
        HunyuanDiTVariant::DiT,
        1_152,
        2,
    );
    collision
        .tensor_shapes
        .insert("y_embedder.y_embedding".to_owned(), vec![1, 1_152]);
    assert_invalid(collision, "PixArt cross-family");

    let mut mixed = probe(
        HunyuanDiTLayout::PrefixedNative,
        HunyuanDiTVariant::DiT,
        1_152,
        2,
    );
    let standalone = probe(
        HunyuanDiTLayout::StandaloneNative,
        HunyuanDiTVariant::DiT,
        1_152,
        2,
    );
    mixed.tensor_shapes.extend(standalone.tensor_shapes);
    assert!(matches!(
        hunyuandit_configuration_for_probe(&mixed),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("ambiguously")
    ));

    let pixart_only = ModelProbe {
        tensor_shapes: BTreeMap::from([("y_embedder.y_embedding".to_owned(), vec![1, 1_152])]),
        metadata: BTreeMap::new(),
    };
    assert!(matches!(
        hunyuandit_configuration_for_probe(&pixart_only),
        Err(ModelFamilyError::ModelLayoutSelection(message)) if message.contains("no HunyuanDiT")
    ));
}

#[test]
fn val_model_family_row_001_hunyuandit_state_layouts_map_transactionally()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        &cancellation,
    );
    for layout in [
        HunyuanDiTLayout::PrefixedNative,
        HunyuanDiTLayout::SavedModel,
        HunyuanDiTLayout::StandaloneNative,
    ] {
        let mapped = ModelStateTransaction::new(&backend, &context).execute(
            &hunyuandit_state_plan_for_layout(layout).compile()?,
            DIGEST,
            &mapping_source(&backend, &context, layout)?,
        )?;
        let model = mapped.component("model").ok_or("missing model")?;
        assert!(!model.is_empty());
        assert!(model.keys().all(|key| key.starts_with("native.")));
        assert_eq!(mapped.component("vae").map(BTreeMap::len), Some(1));
        assert_eq!(mapped.component("text_encoder").map(BTreeMap::len), Some(1));
    }
    assert_eq!(
        hunyuandit_state_plan_for_layout(HunyuanDiTLayout::PrefixedNative).encoded_plan,
        HUNYUANDIT_PREFIXED_STATE_PLAN.encoded_plan
    );
    assert_eq!(
        hunyuandit_state_plan_for_layout(HunyuanDiTLayout::SavedModel).encoded_plan,
        HUNYUANDIT_SAVED_MODEL_STATE_PLAN.encoded_plan
    );
    assert_eq!(
        hunyuandit_state_plan_for_layout(HunyuanDiTLayout::StandaloneNative).encoded_plan,
        HUNYUANDIT_STANDALONE_STATE_PLAN.encoded_plan
    );
    Ok(())
}

#[test]
fn val_ownership_001_hunyuandit_has_one_shared_adapter_and_canonical_owners()
-> Result<(), Box<dyn std::error::Error>> {
    assert!(std::ptr::eq(
        hunyuandit_common_mapping(),
        &HUNYUANDIT_COMMON_MAPPING
    ));
    assert!(std::ptr::eq(
        hunyuandit_common_mapping().clip_target,
        &HUNYUANDIT_CLIP_TARGET
    ));
    assert_eq!(
        hunyuandit_common_mapping().latent_format.feature_id,
        HUNYUANDIT_LATENT_FORMAT.feature_id
    );
    assert_eq!(hunyuandit_common_mapping().components.len(), 3);
    assert_eq!(hunyuandit_common_mapping().forward_program.len(), 5);

    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repository_root = crate_root
        .parent()
        .and_then(Path::parent)
        .ok_or("sim_model crate is not inside the repository")?;
    let test_path = crate_root.join("tests/hunyuandit_family_adapter.rs");
    let adapter_path = crate_root.join("src/hunyuandit_family.rs");
    let foundation_path = crate_root.join("src/model_family.rs");
    let files = rust_files(repository_root)?;
    assert_eq!(
        files_containing(&files, "pub struct HunyuanDiTConfiguration", &[&test_path])?,
        vec![adapter_path]
    );
    assert_eq!(
        files_containing(&files, "fn validate_tensor_fact_predicates(", &[&test_path])?,
        vec![foundation_path]
    );
    for forbidden in [
        "struct ModelStateTransaction",
        "struct PatchGraph",
        "fn estimate_model_memory",
        "struct CancellationToken",
    ] {
        assert!(
            !fs::read_to_string(crate_root.join("src/hunyuandit_family.rs"))?.contains(forbidden)
        );
    }
    Ok(())
}

fn probe(
    layout: HunyuanDiTLayout,
    variant: HunyuanDiTVariant,
    hidden_size: u64,
    depth: usize,
) -> ModelProbe {
    let prefix = layout_prefix(layout);
    let extra_input = match variant {
        HunyuanDiTVariant::DiT => 1_024,
        HunyuanDiTVariant::DiT1 => HUNYUANDIT_DIT1_EXTRA_INPUT,
    };
    let mut tensor_shapes = BTreeMap::from([
        (format!("{prefix}mlp_t5.0.weight"), vec![8_192, 2_048]),
        (format!("{prefix}mlp_t5.2.weight"), vec![1_024, 8_192]),
        (
            format!("{prefix}x_embedder.proj.weight"),
            vec![hidden_size, 4, 2, 2],
        ),
        (
            format!("{prefix}extra_embedder.0.weight"),
            vec![hidden_size * 4, extra_input],
        ),
    ]);
    for ordinal in 0..depth {
        tensor_shapes.insert(
            format!("{prefix}blocks.{ordinal}.attn1.qkv.weight"),
            vec![hidden_size * 3, hidden_size],
        );
    }
    ModelProbe {
        tensor_shapes,
        metadata: BTreeMap::new(),
    }
}

fn layout_prefix(layout: HunyuanDiTLayout) -> &'static str {
    match layout {
        HunyuanDiTLayout::PrefixedNative => "model.diffusion_model.",
        HunyuanDiTLayout::SavedModel => "model.",
        HunyuanDiTLayout::StandaloneNative => "",
    }
}

fn assert_invalid(probe: ModelProbe, expected: &str) {
    assert!(matches!(
        hunyuandit_configuration_for_probe(&probe),
        Err(ModelFamilyError::InvalidSelectorOutput(message)) if message.contains(expected)
    ));
}

fn mapping_source(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    layout: HunyuanDiTLayout,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    let prefix = layout_prefix(layout);
    let keys: &[&str] = if layout == HunyuanDiTLayout::StandaloneNative {
        &[
            "mlp_t5.0.weight",
            "text_embedding_padding",
            "pooler.q_proj.weight",
            "x_embedder.proj.weight",
            "t_embedder.mlp.0.weight",
            "extra_embedder.0.weight",
            "blocks.0.attn1.qkv.weight",
            "final_layer.linear.weight",
        ]
    } else {
        &[
            "mlp_t5.0.weight",
            "x_embedder.proj.weight",
            "extra_embedder.0.weight",
            "blocks.0.attn1.qkv.weight",
            "final_layer.linear.weight",
        ]
    };
    let mut source = BTreeMap::new();
    for (index, key) in keys.iter().enumerate() {
        source.insert(
            format!("{prefix}{key}"),
            tensor(backend, context, index as f32 + 1.0)?,
        );
    }
    source.insert(
        "vae.decoder.weight".to_owned(),
        tensor(backend, context, 20.0)?,
    );
    source.insert(
        "text_encoders.hydit.weight".to_owned(),
        tensor(backend, context, 21.0)?,
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
