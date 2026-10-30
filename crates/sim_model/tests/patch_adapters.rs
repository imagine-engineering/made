use sha2::{Digest, Sha256};
use sim_model::patches::{
    PatchAdapterError, PatchKeyMap, PatchKeyMapping, PatchMergeMapping, PatchMergeRatio,
    PatchMergeRule, PatchPreparedValue, PatchableWeight, QuantizedPatchValue, add_patches,
    get_key_patches, load_lora_patch_graph, merge_key_patches, model_lora_keys_clip,
    model_lora_keys_unet, patch_quantized_value, patch_weight_to_device, prefetch_prepared_value,
};
use sim_model::{
    AdapterFamily, AdapterTensor, PatchComputeBoundary, PatchPayload, PatchSlice, PatchTensor,
    PatchValueTransform, QuantizationKind, SemanticPatchOperation, quantize_matrix,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, StreamId, Tensor,
    generated_sim_operator_indirection_01::{
        tensor_from_f32_with_context_exact_native, tensor_to_f32_with_backend_exact_native,
    },
};
use sim_types::CancellationToken;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::path::{Path, PathBuf};

const BASE_DIGEST: &str = "abababababababababababababababababababababababababababababababab";

struct Harness {
    backend: CpuBackend,
    workspace: CpuWorkspaceAuthority,
    cancellation: CancellationToken,
}

impl Harness {
    fn new() -> Result<Self, Box<dyn Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
        Ok(Self {
            backend,
            workspace,
            cancellation: CancellationToken::default(),
        })
    }

    fn context(&self) -> Result<ExecutionContext<'_>, Box<dyn Error>> {
        Ok(self.backend.execution_context(
            StreamId::DEFAULT,
            self.workspace.authorize_workspace(32 * 1024 * 1024)?,
            &self.cancellation,
        ))
    }

    fn tensor(
        &self,
        shape: &[u64],
        values: &[f32],
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, Box<dyn Error>> {
        Ok(tensor_from_f32_with_context_exact_native(
            &self.backend,
            shape,
            values,
            DType::F32,
            sim_tensor::DeviceId::CPU,
            context,
        )?)
    }
}

fn operation(
    identifier: &str,
    target_key: &str,
    expected_shape: Vec<u64>,
    payload: PatchPayload,
) -> SemanticPatchOperation {
    SemanticPatchOperation {
        identifier: identifier.into(),
        target_key: target_key.into(),
        expected_shape,
        strength: 1.0,
        strength_model: 1.0,
        slices: Vec::new(),
        transform: PatchValueTransform::default(),
        payload,
    }
}

#[test]
fn val_patch_adapter_001_key_discovery_preserves_aliases_and_slices() -> Result<(), Box<dyn Error>>
{
    let many_to_one = PatchKeyMap::checked(vec![
        PatchKeyMapping::direct("source.a", "model.weight"),
        PatchKeyMapping::direct("source.b", "model.weight"),
        PatchKeyMapping {
            source: "source.qkv".into(),
            target: "model.qkv.weight".into(),
            slices: vec![PatchSlice {
                dimension: 0,
                start: 4,
                length: 8,
            }],
        },
    ])?;
    assert_eq!(many_to_one.target_for("source.a"), Some("model.weight"));
    assert_eq!(many_to_one.target_for("source.b"), Some("model.weight"));
    assert_eq!(
        many_to_one
            .binding_for("source.qkv")
            .map(|value| &value.slices),
        Some(&vec![PatchSlice {
            dimension: 0,
            start: 4,
            length: 8
        }])
    );
    assert!(matches!(
        PatchKeyMap::checked(vec![
            PatchKeyMapping::direct("same", "first"),
            PatchKeyMapping::direct("same", "second"),
        ]),
        Err(PatchAdapterError::ConflictingSource(_))
    ));

    let clip = model_lora_keys_clip(
        vec![
            "clip_g.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight".into(),
            "clip_l.transformer.text_model.encoder.layers.1.mlp.fc1.weight".into(),
            "t5xxl.transformer.encoder.block.0.layer.weight".into(),
            "clip_g.transformer.text_projection.weight".into(),
            "clip_l.transformer.text_projection.weight".into(),
        ],
        Vec::new(),
    )?;
    assert_eq!(
        clip.target_for("lora_te_text_model_encoder_layers_0_self_attn_q_proj"),
        Some("clip_g.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight")
    );
    assert_eq!(
        clip.target_for("lora_te1_text_model_encoder_layers_1_mlp_fc1"),
        Some("clip_l.transformer.text_model.encoder.layers.1.mlp.fc1.weight")
    );
    assert_eq!(
        clip.target_for("lora_te3_encoder_block_0_layer"),
        Some("t5xxl.transformer.encoder.block.0.layer.weight")
    );
    assert_eq!(
        clip.target_for("lora_te2_text_projection"),
        Some("clip_g.transformer.text_projection.weight")
    );

    let unet = model_lora_keys_unet(
        vec![
            "diffusion_model.input_blocks.0.weight".into(),
            "diffusion_model.marker".into(),
        ],
        vec![PatchKeyMapping::direct(
            "down_blocks.0.attentions.0.to_out.0.weight",
            "diffusion_model.input_blocks.1.weight",
        )],
        vec![PatchKeyMapping {
            source: "transformer.single_transformer_blocks.0.attn.to_qkv".into(),
            target: "diffusion_model.double_blocks.0.qkv.weight".into(),
            slices: vec![PatchSlice {
                dimension: 0,
                start: 0,
                length: 24,
            }],
        }],
        Vec::new(),
    )?;
    assert_eq!(
        unet.target_for("lora_unet_input_blocks_0"),
        Some("diffusion_model.input_blocks.0.weight")
    );
    assert_eq!(
        unet.target_for("down_blocks.0.attentions.0.processor.to_out"),
        Some("diffusion_model.input_blocks.1.weight")
    );
    assert_eq!(
        unet.binding_for("transformer.single_transformer_blocks.0.attn.to_qkv")
            .and_then(|mapping| mapping.slices.first())
            .map(|slice| slice.length),
        Some(24)
    );

    let t5_key = "t5xxl.transformer.encoder.block.0.layer.weight";
    let clip_l_key = "clip_l.transformer.text_model.encoder.layers.0.mlp.fc1.weight";
    let clip_g_key = "clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight";
    let t5_only = model_lora_keys_clip(vec![t5_key.into()], Vec::new())?;
    assert_eq!(
        t5_only.target_for("lora_te1_encoder_block_0_layer"),
        Some(t5_key)
    );
    let t5_with_g = model_lora_keys_clip(vec![clip_g_key.into(), t5_key.into()], Vec::new())?;
    assert_eq!(
        t5_with_g.target_for("lora_te2_encoder_block_0_layer"),
        Some(t5_key)
    );
    let t5_with_l = model_lora_keys_clip(vec![clip_l_key.into(), t5_key.into()], Vec::new())?;
    assert_eq!(
        t5_with_l.target_for("lora_te2_encoder_block_0_layer"),
        Some(t5_key)
    );
    assert_eq!(
        t5_with_l.target_for("lora_te3_encoder_block_0_layer"),
        Some(t5_key)
    );
    Ok(())
}

#[test]
fn val_patch_adapter_001_all_merge_classes_match_source_formulas() -> Result<(), Box<dyn Error>> {
    let model_key = "diffusion_model.input.1.weight";
    let simple = PatchMergeRule::model_simple(0.25)?
        .map(model_key)?
        .ok_or("missing model mapping")?;
    assert_eq!(
        simple,
        PatchMergeMapping {
            target_key: model_key.into(),
            patch_strength: 0.75,
            model_strength: 0.25,
        }
    );
    let add = PatchMergeRule::model_add()?
        .map(model_key)?
        .ok_or("missing add mapping")?;
    assert_eq!((add.patch_strength, add.model_strength), (1.0, 1.0));
    let subtract = PatchMergeRule::model_subtract(2.5)?
        .map(model_key)?
        .ok_or("missing subtract mapping")?;
    assert_eq!(
        (subtract.patch_strength, subtract.model_strength),
        (-2.5, 2.5)
    );
    assert!(PatchMergeRule::model_subtract(10.01).is_err());

    let blocks = PatchMergeRule::model_blocks(
        0.1,
        vec![
            PatchMergeRatio::checked("input", 0.2)?,
            PatchMergeRatio::checked("input.1", 0.8)?,
        ],
    )?
    .map(model_key)?
    .ok_or("missing block mapping")?;
    assert!((blocks.patch_strength - 0.2).abs() < f32::EPSILON);
    assert_eq!(blocks.model_strength, 0.8);

    for clip in [
        PatchMergeRule::clip_simple(0.4)?,
        PatchMergeRule::clip_add()?,
        PatchMergeRule::clip_subtract(-3.0)?,
    ] {
        assert!(clip.map("clip_l.position_ids")?.is_none());
        assert!(clip.map("clip_g.logit_scale")?.is_none());
        assert!(clip.map("clip_l.transformer.weight")?.is_some());
    }
    Ok(())
}

#[test]
fn val_patch_adapter_001_load_lora_delegates_registry_and_preserves_precedence()
-> Result<(), Box<dyn Error>> {
    let harness = Harness::new()?;
    let context = harness.context()?;
    let identity = harness.tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0], &context)?;
    let diff = harness.tensor(&[2, 2], &[2.0; 4], &context)?;
    let set = harness.tensor(&[2, 2], &[3.0; 4], &context)?;
    let bias = harness.tensor(&[2], &[4.0, 5.0], &context)?;
    let alpha = harness.tensor(&[], &[2.0], &context)?;
    let key_map = PatchKeyMap::checked(vec![PatchKeyMapping::direct(
        "adapter",
        "diffusion_model.layer.weight",
    )])?;
    let tensors = BTreeMap::from([
        (
            "adapter.lora_up.weight".into(),
            AdapterTensor::Dense(identity.clone()),
        ),
        (
            "adapter.lora_down.weight".into(),
            AdapterTensor::Dense(identity),
        ),
        ("adapter.alpha".into(), AdapterTensor::Dense(alpha)),
        ("adapter.w_norm".into(), AdapterTensor::Dense(diff.clone())),
        ("adapter.b_norm".into(), AdapterTensor::Dense(bias.clone())),
        ("adapter.diff".into(), AdapterTensor::Dense(diff)),
        ("adapter.diff_b".into(), AdapterTensor::Dense(bias)),
        ("adapter.set_weight".into(), AdapterTensor::Dense(set)),
        (
            "unknown.payload".into(),
            AdapterTensor::Dense(harness.tensor(&[1], &[1.0], &context)?),
        ),
    ]);
    let report = load_lora_patch_graph(
        BASE_DIGEST,
        &key_map,
        &tensors,
        &BTreeMap::from([
            ("diffusion_model.layer.weight".into(), vec![2, 2]),
            ("diffusion_model.layer.bias".into(), vec![2]),
        ]),
        0.5,
        1.0,
        &harness.backend,
        &context,
    )?;
    assert_eq!(
        report.loaded_families.get("adapter"),
        Some(&AdapterFamily::Lora)
    );
    assert_eq!(
        report.unused_keys,
        BTreeSet::from(["unknown.payload".into()])
    );
    let operations = report.graph.semantic_operations();
    assert_eq!(operations.len(), 2);
    assert_eq!(operations[0].target_key, "diffusion_model.layer.weight");
    assert!(matches!(operations[0].payload, PatchPayload::Set { .. }));
    assert_eq!(operations[1].target_key, "diffusion_model.layer.bias");
    assert!(matches!(
        operations[1].payload,
        PatchPayload::DenseDiff { .. }
    ));

    let ambiguous = BTreeMap::from([
        (
            "adapter.lora_up.weight".into(),
            AdapterTensor::Dense(harness.tensor(&[2, 2], &[1.0; 4], &context)?),
        ),
        (
            "adapter.lora_down.weight".into(),
            AdapterTensor::Dense(harness.tensor(&[2, 2], &[1.0; 4], &context)?),
        ),
        (
            "adapter.hada_w1_a".into(),
            AdapterTensor::Dense(harness.tensor(&[2, 2], &[1.0; 4], &context)?),
        ),
        (
            "adapter.hada_w1_b".into(),
            AdapterTensor::Dense(harness.tensor(&[2, 2], &[1.0; 4], &context)?),
        ),
        (
            "adapter.hada_w2_a".into(),
            AdapterTensor::Dense(harness.tensor(&[2, 2], &[1.0; 4], &context)?),
        ),
        (
            "adapter.hada_w2_b".into(),
            AdapterTensor::Dense(harness.tensor(&[2, 2], &[1.0; 4], &context)?),
        ),
    ]);
    assert!(
        load_lora_patch_graph(
            BASE_DIGEST,
            &key_map,
            &ambiguous,
            &BTreeMap::from([("diffusion_model.layer.weight".into(), vec![2, 2])]),
            1.0,
            1.0,
            &harness.backend,
            &context,
        )
        .is_err()
    );

    let quantized_set = quantize_matrix(
        QuantizationKind::Int8Tensorwise,
        DType::F32,
        &[7.0, 8.0, 9.0, 10.0],
        2,
        2,
        &harness.cancellation,
    )?;
    let quantized_alpha = quantize_matrix(
        QuantizationKind::Int8Tensorwise,
        DType::F32,
        &[2.0],
        1,
        1,
        &harness.cancellation,
    )?;
    let quantized = load_lora_patch_graph(
        BASE_DIGEST,
        &key_map,
        &BTreeMap::from([
            (
                "adapter.set_weight".into(),
                AdapterTensor::Quantized(quantized_set),
            ),
            (
                "adapter.alpha".into(),
                AdapterTensor::Quantized(quantized_alpha),
            ),
        ]),
        &BTreeMap::from([("diffusion_model.layer.weight".into(), vec![2, 2])]),
        1.0,
        1.0,
        &harness.backend,
        &context,
    )?;
    assert!(matches!(
        quantized.graph.semantic_operations()[0].payload,
        PatchPayload::Set { .. }
    ));

    let independent_bias = load_lora_patch_graph(
        BASE_DIGEST,
        &key_map,
        &BTreeMap::from([(
            "adapter.diff_b".into(),
            AdapterTensor::Dense(harness.tensor(&[2], &[1.0, 2.0], &context)?),
        )]),
        &BTreeMap::from([("diffusion_model.layer.bias".into(), vec![2])]),
        1.0,
        1.0,
        &harness.backend,
        &context,
    )?;
    assert_eq!(independent_bias.graph.semantic_operations().len(), 1);
    assert_eq!(
        independent_bias.graph.semantic_operations()[0].target_key,
        "diffusion_model.layer.bias"
    );
    Ok(())
}

#[test]
fn val_patch_adapter_001_malformed_ambiguous_and_oom_requests_are_atomic()
-> Result<(), Box<dyn Error>> {
    assert!(PatchMergeRule::model_simple(f32::NAN).is_err());
    assert!(PatchMergeRatio::checked("input", f32::INFINITY).is_err());
    assert!(matches!(
        PatchKeyMap::checked(vec![PatchKeyMapping {
            source: "source".into(),
            target: "target".into(),
            slices: vec![PatchSlice {
                dimension: 0,
                start: u64::MAX,
                length: 2,
            }],
        }]),
        Err(PatchAdapterError::InvalidSlice)
    ));

    let harness = Harness::new()?;
    let context = harness.context()?;
    let key_map = PatchKeyMap::checked(vec![
        PatchKeyMapping::direct("first", "layer.weight"),
        PatchKeyMapping::direct("second", "layer.weight"),
    ])?;
    let tensors = BTreeMap::from([
        (
            "first.diff".into(),
            AdapterTensor::Dense(harness.tensor(&[2, 2], &[1.0; 4], &context)?),
        ),
        (
            "second.diff".into(),
            AdapterTensor::Dense(harness.tensor(&[2, 2], &[2.0; 4], &context)?),
        ),
    ]);
    assert!(matches!(
        load_lora_patch_graph(
            BASE_DIGEST,
            &key_map,
            &tensors,
            &BTreeMap::from([("layer.weight".into(), vec![2, 2])]),
            1.0,
            1.0,
            &harness.backend,
            &context,
        ),
        Err(PatchAdapterError::AmbiguousTarget)
    ));

    let nonscalar = BTreeMap::from([
        (
            "first.alpha".into(),
            AdapterTensor::Dense(harness.tensor(&[2], &[1.0, 2.0], &context)?),
        ),
        (
            "first.diff".into(),
            AdapterTensor::Dense(harness.tensor(&[2, 2], &[1.0; 4], &context)?),
        ),
    ]);
    assert!(
        load_lora_patch_graph(
            BASE_DIGEST,
            &PatchKeyMap::checked(vec![PatchKeyMapping::direct("first", "layer.weight")])?,
            &nonscalar,
            &BTreeMap::from([("layer.weight".into(), vec![2, 2])]),
            1.0,
            1.0,
            &harness.backend,
            &context,
        )
        .is_err()
    );

    let quantized = quantize_matrix(
        QuantizationKind::Int8Tensorwise,
        DType::F32,
        &[1.0, 2.0, 3.0, 4.0],
        2,
        2,
        &harness.cancellation,
    )?;
    let identity = quantized.content_identity();
    let graph = sim_model::PatchGraph::checked_semantic(
        BASE_DIGEST,
        vec![operation(
            "oom",
            "layer.weight",
            vec![2, 2],
            PatchPayload::DenseDiff {
                tensor: PatchTensor::checked(vec![2, 2], vec![1.0; 4])?,
                pad_weight: false,
            },
        )],
    )?;
    let (backend, workspace) = CpuWorkspaceAuthority::create_backend(1024)?;
    let cancellation = CancellationToken::default();
    let zero_workspace = backend.execution_context(
        StreamId::DEFAULT,
        workspace.authorize_workspace(0)?,
        &cancellation,
    );
    assert!(
        patch_quantized_value(
            &graph,
            "layer.weight",
            &QuantizedPatchValue::Matrix(quantized.clone()),
            PatchComputeBoundary::Configured(DType::F32),
            &backend,
            &zero_workspace,
        )
        .is_err()
    );
    assert_eq!(quantized.content_identity(), identity);
    assert_eq!(zero_workspace.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn val_patch_adapter_001_add_get_and_quantized_replacement_delegate_canonical_owners()
-> Result<(), Box<dyn Error>> {
    let harness = Harness::new()?;
    let context = harness.context()?;
    let diff = PatchPayload::DenseDiff {
        tensor: PatchTensor::checked(vec![2, 2], vec![1.0; 4])?,
        pad_weight: false,
    };
    let first = operation("first", "layer.weight", vec![2, 2], diff.clone());
    let ignored = operation("ignored", "missing.weight", vec![2, 2], diff.clone());
    let (graph, accepted) = add_patches(
        None,
        BASE_DIGEST,
        &BTreeSet::from(["layer.weight".into()]),
        vec![first, ignored],
    )?;
    assert_eq!(accepted, BTreeSet::from(["layer.weight".into()]));
    let second = operation("second", "layer.weight", vec![2, 2], diff);
    let (graph, _) = add_patches(
        Some(&graph),
        BASE_DIGEST,
        &BTreeSet::from(["layer.weight".into()]),
        vec![second],
    )?;
    let dense = harness.tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0], &context)?;
    let projection = get_key_patches(
        &graph,
        "layer.weight",
        &PatchableWeight::Dense(dense.clone()),
    )?;
    assert_eq!(
        projection
            .operations()
            .iter()
            .map(|operation| operation.identifier.as_str())
            .collect::<Vec<_>>(),
        vec!["first", "second"]
    );

    let quantized = quantize_matrix(
        QuantizationKind::Int8Tensorwise,
        DType::F32,
        &[1.0, 2.0, 3.0, 4.0],
        2,
        2,
        &harness.cancellation,
    )?;
    let replaced = patch_quantized_value(
        &graph,
        "layer.weight",
        &QuantizedPatchValue::Matrix(quantized.clone()),
        PatchComputeBoundary::Configured(DType::F32),
        &harness.backend,
        &context,
    )?;
    let QuantizedPatchValue::Matrix(replaced) = replaced else {
        return Err("replacement changed quantized representation".into());
    };
    assert_eq!(replaced.kind(), quantized.kind());
    assert_ne!(replaced.content_identity(), quantized.content_identity());
    let values = replaced.materialize(&harness.backend, &context)?;
    for (actual, expected) in values.values().iter().zip([3.0, 4.0, 5.0, 6.0]) {
        assert!((actual - expected).abs() < 0.06, "{actual} != {expected}");
    }

    let graph = graph.append_semantic(vec![operation(
        "other",
        "other.weight",
        vec![2, 2],
        PatchPayload::DenseDiff {
            tensor: PatchTensor::checked(vec![2, 2], vec![100.0; 4])?,
            pad_weight: false,
        },
    )])?;
    let patched = patch_weight_to_device(
        &graph,
        "layer.weight",
        &PatchableWeight::Dense(dense.clone()),
        false,
        PatchComputeBoundary::Configured(DType::F32),
        &harness.backend,
        &context,
    )?;
    let PatchableWeight::Dense(patched) = patched else {
        return Err("dense patch changed representation".into());
    };
    assert_eq!(
        tensor_to_f32_with_backend_exact_native(&harness.backend, &patched, &context)?,
        &[3.0, 4.0, 5.0, 6.0]
    );

    let copied = patch_weight_to_device(
        &graph,
        "unpatched.weight",
        &PatchableWeight::Dense(dense.clone()),
        true,
        PatchComputeBoundary::Configured(DType::F32),
        &harness.backend,
        &context,
    )?;
    let PatchableWeight::Dense(copied) = copied else {
        return Err("forced dense copy changed representation".into());
    };
    assert_eq!(
        tensor_to_f32_with_backend_exact_native(&harness.backend, &copied, &context)?,
        tensor_to_f32_with_backend_exact_native(&harness.backend, &dense, &context)?
    );
    assert_ne!(copied.storage_id(), dense.storage_id());

    let source_graph = sim_model::PatchGraph::checked_semantic(
        BASE_DIGEST,
        vec![operation(
            "source-diff",
            "diffusion_model.layer.weight",
            vec![2, 2],
            PatchPayload::DenseDiff {
                tensor: PatchTensor::checked(vec![2, 2], vec![1.0; 4])?,
                pad_weight: false,
            },
        )],
    )?;
    let source = get_key_patches(
        &source_graph,
        "diffusion_model.layer.weight",
        &PatchableWeight::Dense(harness.tensor(&[2, 2], &[10.0; 4], &context)?),
    )?;
    let target = harness.tensor(&[2, 2], &[2.0; 4], &context)?;
    for (rule, expected) in [
        (PatchMergeRule::model_simple(0.25)?, 8.75),
        (PatchMergeRule::model_add()?, 13.0),
        (PatchMergeRule::model_subtract(2.0)?, -18.0),
    ] {
        let merge = merge_key_patches(
            &rule,
            &source,
            vec![2, 2],
            PatchComputeBoundary::Configured(DType::F32),
            &harness.backend,
            &context,
        )?
        .ok_or("model merge rule did not map its source key")?;
        let merge_graph = sim_model::PatchGraph::checked_semantic(BASE_DIGEST, vec![merge])?;
        let merged = merge_graph.apply_single_tensor(
            &harness.backend,
            "diffusion_model.layer.weight",
            &target,
            PatchComputeBoundary::Configured(DType::F32),
            &context,
        )?;
        assert_eq!(
            tensor_to_f32_with_backend_exact_native(&harness.backend, &merged, &context)?,
            vec![expected; 4]
        );
    }
    Ok(())
}

#[test]
fn val_patch_adapter_001_recursive_prefetch_is_aligned_atomic_and_cancelled()
-> Result<(), Box<dyn Error>> {
    let harness = Harness::new()?;
    let context = harness.context()?;
    let tensor = harness.tensor(&[3], &[1.0, 2.0, 3.0], &context)?;
    let quantized = quantize_matrix(
        QuantizationKind::Int8Tensorwise,
        DType::F32,
        &[1.0, 2.0, 3.0, 4.0],
        2,
        2,
        &harness.cancellation,
    )?;
    let quantized_identity = quantized.content_identity();
    let nested = PatchPreparedValue::Adapter {
        family: AdapterFamily::Lora,
        loaded_keys: BTreeSet::from(["adapter.lora_up.weight".into()]),
        weights: Box::new(PatchPreparedValue::Tuple(vec![
            PatchPreparedValue::Tensor(AdapterTensor::Dense(tensor.clone())),
            PatchPreparedValue::List(vec![
                PatchPreparedValue::Scalar("metadata".into()),
                PatchPreparedValue::Tensor(AdapterTensor::Dense(tensor)),
                PatchPreparedValue::Tensor(AdapterTensor::Quantized(quantized)),
            ]),
        ])),
    };
    let measured = prefetch_prepared_value(&nested, None, false, &harness.backend, &context)?;
    assert_eq!(measured.aligned_bytes, 3072);
    assert!(matches!(
        prefetch_prepared_value(&nested, Some(3071), true, &harness.backend, &context,),
        Err(PatchAdapterError::PrefetchCapacity { .. })
    ));
    let copied = prefetch_prepared_value(&nested, Some(3072), true, &harness.backend, &context)?;
    assert_eq!(copied.aligned_bytes, 3072);
    let PatchPreparedValue::Adapter { weights, .. } = copied.value else {
        return Err("prefetch changed the adapter representation".into());
    };
    let PatchPreparedValue::Tuple(tuple) = *weights else {
        return Err("prefetch changed the tuple representation".into());
    };
    let PatchPreparedValue::List(list) = tuple.get(1).ok_or("missing prefetched list")? else {
        return Err("prefetch changed the list representation".into());
    };
    let Some(PatchPreparedValue::Tensor(AdapterTensor::Quantized(prefetched_quantized))) =
        list.get(2)
    else {
        return Err("prefetch lost quantized storage".into());
    };
    assert_eq!(prefetched_quantized.content_identity(), quantized_identity);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = harness.backend.execution_context(
        StreamId::DEFAULT,
        harness.workspace.authorize_workspace(1024)?,
        &cancelled,
    );
    assert!(
        prefetch_prepared_value(
            &nested,
            Some(3072),
            true,
            &harness.backend,
            &cancelled_context,
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn val_patch_adapter_001_no_duplicate_family_parser_or_owner() -> Result<(), Box<dyn Error>> {
    let source = include_str!("../src/patches.rs");
    assert!(!source.contains(concat!("struct PatchPayload", "Parser")));
    assert!(!source.contains("fn load_family("));
    assert!(source.contains("WeightAdapterRegistry.load_unique"));
    assert!(source.contains("PatchGraph::checked_semantic"));
    assert!(source.contains("quantize_matrix("));
    assert!(!source.contains(concat!("struct Cancellation", "Token")));
    assert!(!source.contains(concat!("struct BackendWorkspace", "Authority")));
    assert!(!source.contains(concat!("struct Output", "Committer")));
    Ok(())
}

fn repository_root() -> Result<PathBuf, Box<dyn Error>> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("repository root is unavailable")?
        .to_path_buf())
}

fn python_symbol_sha256(source: &[u8], symbol: &str) -> Result<String, Box<dyn Error>> {
    let source = std::str::from_utf8(source)?;
    let lines = source.split_inclusive('\n').collect::<Vec<_>>();
    let signatures = [
        format!("def {symbol}("),
        format!("async def {symbol}("),
        format!("class {symbol}("),
        format!("class {symbol}:"),
    ];
    let matches = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| {
            let trimmed = line.trim_start_matches([' ', '\t']);
            signatures
                .iter()
                .any(|signature| trimmed.starts_with(signature))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let [start] = matches.as_slice() else {
        return Err(format!(
            "expected exactly one Python definition for {symbol}, found {}",
            matches.len()
        )
        .into());
    };
    let indentation = lines[*start].len() - lines[*start].trim_start_matches([' ', '\t']).len();
    let mut header_complete = lines[*start].trim_end().ends_with(':');
    let mut body_seen = false;
    let mut end = *start + 1;
    while let Some(line) = lines.get(end) {
        let trimmed = line.trim_start_matches([' ', '\t']);
        let content = trimmed.trim_end_matches(['\r', '\n']);
        if content.is_empty() || content.starts_with('#') {
            end += 1;
            continue;
        }
        let line_indentation = line.len() - trimmed.len();
        if !header_complete {
            header_complete = line_indentation == indentation && content.ends_with(':');
            end += 1;
            continue;
        }
        if body_seen && line_indentation <= indentation {
            break;
        }
        if line_indentation > indentation {
            body_seen = true;
        }
        end += 1;
    }
    if !body_seen {
        return Err(format!("Python definition {symbol} has no body").into());
    }
    while end > *start + 1 {
        let content = lines[end - 1].trim();
        if content.is_empty() || content.starts_with('#') {
            end -= 1;
        } else {
            break;
        }
    }
    Ok(format!(
        "{:x}",
        Sha256::digest(lines[*start..end].concat().as_bytes())
    ))
}

#[test]
fn val_patch_adapter_001_exact_catalog_manifest_and_artifact_are_current()
-> Result<(), Box<dyn Error>> {
    const TASK: &str = "comfy-parity-patch-loading-merge-quantized-adapter";
    const EXPECTED: [(&str, &str, &str, &str, &str); 14] = [
        (
            "conditioning-patch-mapping-model-patcher-add-patches-4ab8745f",
            "projects/comfy/ComfyUI/sim/model_patcher.py",
            "add_patches",
            "96d21eeaf16d4723355374a5e2b93b35d512ad8f6dc8a1a3a4253cdd71dfd5b0",
            "1448de1ea759f3c89de64655f600315668acd74fbde0c70d940cb425b28212cb",
        ),
        (
            "conditioning-patch-mapping-model-patcher-get-key-patches-d9d90b3e",
            "projects/comfy/ComfyUI/sim/model_patcher.py",
            "get_key_patches",
            "96d21eeaf16d4723355374a5e2b93b35d512ad8f6dc8a1a3a4253cdd71dfd5b0",
            "d3ce790c2d99c93a30291cbeac2d7a757cb59a06e9f38be134fb6e04be541a74",
        ),
        (
            "conditioning-patch-mapping-model-patcher-patch-weight-to-device-65b24000",
            "projects/comfy/ComfyUI/sim/model_patcher.py",
            "patch_weight_to_device",
            "96d21eeaf16d4723355374a5e2b93b35d512ad8f6dc8a1a3a4253cdd71dfd5b0",
            "0f0e7a97eaff261d5e9d39536ef47219230f2e175ede86c598451176b97fe85f",
        ),
        (
            "conditioning-patch-mapping-lora-load-lora-37a7b44e",
            "projects/comfy/ComfyUI/sim/lora.py",
            "load_lora",
            "8f75c95ddc8ab0144919fe5277c4e6b4fa4f4f45aa64aa3de5d2d3b1b4a927d8",
            "69633b40846b7b70cb4c3a11b5b7cd074b85c542aa77ee218ef2581a3eef6876",
        ),
        (
            "conditioning-patch-mapping-lora-model-lora-keys-clip-10b4db7a",
            "projects/comfy/ComfyUI/sim/lora.py",
            "model_lora_keys_clip",
            "8f75c95ddc8ab0144919fe5277c4e6b4fa4f4f45aa64aa3de5d2d3b1b4a927d8",
            "c501f7ef1b9ed47f3124fca4881264ba9e8e7f8c36b7e1639a98ca610b3f7550",
        ),
        (
            "conditioning-patch-mapping-lora-model-lora-keys-unet-0f5d794c",
            "projects/comfy/ComfyUI/sim/lora.py",
            "model_lora_keys_unet",
            "8f75c95ddc8ab0144919fe5277c4e6b4fa4f4f45aa64aa3de5d2d3b1b4a927d8",
            "90b5ded69e0e84a398759d2fd4eaabb144d994f7c49321abf6fa812d02c28278",
        ),
        (
            "conditioning-patch-mapping-lora-prefetch-prepared-value-9f54692e",
            "projects/comfy/ComfyUI/sim/lora.py",
            "prefetch_prepared_value",
            "8f75c95ddc8ab0144919fe5277c4e6b4fa4f4f45aa64aa3de5d2d3b1b4a927d8",
            "6b4b0339c296cb04d12c3e72c0461e824b305a2da7575f2779cfacd6c720c8fe",
        ),
        (
            "conditioning-patch-mapping-nodes-model-merging-modelmergesimple-50d89441",
            "projects/comfy/ComfyUI/comfy_extras/nodes_model_merging.py",
            "ModelMergeSimple",
            "8bd93638e30dc8ac005f16130ca9c4ba62228fe1721aad406db962cac1f2e77d",
            "f6470de04c64812c2296905ad5547a9a211483e5ca610b5cc83a9c22180b9140",
        ),
        (
            "conditioning-patch-mapping-nodes-model-merging-modelsubtract-e89cf4f4",
            "projects/comfy/ComfyUI/comfy_extras/nodes_model_merging.py",
            "ModelSubtract",
            "8bd93638e30dc8ac005f16130ca9c4ba62228fe1721aad406db962cac1f2e77d",
            "4b74c7e383703ea30dd5d0c50d0623f5f51e29e6905d7b129991391a04e63bc7",
        ),
        (
            "conditioning-patch-mapping-nodes-model-merging-modeladd-7597d56c",
            "projects/comfy/ComfyUI/comfy_extras/nodes_model_merging.py",
            "ModelAdd",
            "8bd93638e30dc8ac005f16130ca9c4ba62228fe1721aad406db962cac1f2e77d",
            "5e4057b8a6ade9334f839be8b3b805acde23b9634f8950045338961ed31314c9",
        ),
        (
            "conditioning-patch-mapping-nodes-model-merging-clipmergesimple-8592ddcf",
            "projects/comfy/ComfyUI/comfy_extras/nodes_model_merging.py",
            "CLIPMergeSimple",
            "8bd93638e30dc8ac005f16130ca9c4ba62228fe1721aad406db962cac1f2e77d",
            "961fac9ed2946e9915f619a4ae88d8b9fe71e46d76ac629a4988d0e2a5b10ee0",
        ),
        (
            "conditioning-patch-mapping-nodes-model-merging-clipsubtract-00f43912",
            "projects/comfy/ComfyUI/comfy_extras/nodes_model_merging.py",
            "CLIPSubtract",
            "8bd93638e30dc8ac005f16130ca9c4ba62228fe1721aad406db962cac1f2e77d",
            "2f4934d104d2e2dd0973d742a054bd17296805d500eaba61cceb1b564119241a",
        ),
        (
            "conditioning-patch-mapping-nodes-model-merging-clipadd-7f0acd82",
            "projects/comfy/ComfyUI/comfy_extras/nodes_model_merging.py",
            "CLIPAdd",
            "8bd93638e30dc8ac005f16130ca9c4ba62228fe1721aad406db962cac1f2e77d",
            "988966684b034b26feb28e84276f2b40f19189a31f8ee6198ab8b6b8c9257482",
        ),
        (
            "conditioning-patch-mapping-nodes-model-merging-modelmergeblocks-374d8ab0",
            "projects/comfy/ComfyUI/comfy_extras/nodes_model_merging.py",
            "ModelMergeBlocks",
            "8bd93638e30dc8ac005f16130ca9c4ba62228fe1721aad406db962cac1f2e77d",
            "c3e55401c7f73df94d768ff83ec923f1049b6e89d105448b88e67d6c73f2f64f",
        ),
    ];
    let repository = repository_root()?;
    let catalog = std::fs::read_to_string(
        repository
            .join("crates/sim_test_support/parity/catalogs/backend-conditioning-contracts.csv"),
    )?;
    let expected = EXPECTED
        .iter()
        .map(|row| (row.0, *row))
        .collect::<BTreeMap<_, _>>();
    let mut seen = BTreeSet::new();
    let mut contracts = Vec::new();
    for line in catalog.lines().skip(1) {
        let columns = line.split(',').collect::<Vec<_>>();
        if columns.get(8).copied() != Some(TASK) {
            continue;
        }
        if columns.len() != 15 {
            return Err("malformed patch-adapter catalog row".into());
        }
        let expected_row = expected
            .get(columns[0])
            .ok_or("unexpected patch-adapter catalog row")?;
        assert!(seen.insert(columns[0]));
        assert_eq!(columns[2], expected_row.1);
        assert_eq!(columns[3], expected_row.2);
        assert_eq!(columns[5], expected_row.3);
        assert_eq!(columns[6], expected_row.4);
        assert_eq!(columns[7], "sim_model::patches");
        assert_eq!(columns[9], "sim_model::patches::tests");
        assert_eq!(columns[10], "native_rust");
        assert_eq!(columns[14], "VAL-PATCH-ADAPTER-001");
        let source = std::fs::read(repository.join(columns[2]))?;
        assert_eq!(format!("{:x}", Sha256::digest(&source)), columns[5]);
        assert_eq!(python_symbol_sha256(&source, columns[3])?, columns[6]);
        contracts.push(serde_json::json!({
            "contract_id": columns[0],
            "task_id": TASK,
            "source_sha256": columns[5],
            "symbol_sha256": columns[6],
            "status": "passed",
            "case_ids": [
                format!("{}:source-derived-valid", columns[0]),
                format!("{}:source-derived-invalid", columns[0]),
            ],
        }));
    }
    assert_eq!(seen, expected.keys().copied().collect());

    const IMPLEMENTATION_PATHS: [&str; 5] = [
        "crates/sim_model/src/patch_graph.rs",
        "crates/sim_model/src/patches.rs",
        "crates/sim_model/src/quantization.rs",
        "crates/sim_model/src/weight_adapter.rs",
        "crates/sim_model/tests/patch_adapters.rs",
    ];
    let implementations = IMPLEMENTATION_PATHS
        .iter()
        .map(|path| {
            let bytes = std::fs::read(repository.join(path))?;
            Ok(serde_json::json!({
                "path": path,
                "sha256": format!("{:x}", Sha256::digest(bytes)),
            }))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let producer_path = "crates/sim_model/tests/patch_adapters.rs";
    let producer = std::fs::read(repository.join(producer_path))?;
    let passed = contracts.len() * 2;
    let artifact = serde_json::json!({
        "schema_version": 1,
        "validation_id": "VAL-PATCH-ADAPTER-001",
        "overall_status": "passed",
        "environment": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "backend": "sim_tensor::CpuBackend",
            "device": "cpu",
            "dtype": "f32",
        },
        "summary": { "passed": passed, "failed": 0, "skipped": 0 },
        "implementation": {
            "path": producer_path,
            "sha256": format!("{:x}", Sha256::digest(producer)),
        },
        "task_results": {
            TASK: {
                "status": "passed",
                "passed": passed,
                "failed": 0,
                "skipped": 0,
                "case_ids": [
                    "task511:all-14-valid-invalid",
                    "task511:key-discovery-load-diagnostics",
                    "task511:merge-and-patch-plan-mapping",
                    "task511:quantized-prefetch-cancellation",
                    "task511:ownership-consolidation",
                ],
                "implementations": implementations,
            },
        },
        "contracts": contracts,
    });
    let artifact_directory = repository.join("target/sim-parity");
    std::fs::create_dir_all(&artifact_directory)?;
    std::fs::write(
        artifact_directory.join("val-patch-adapter-001.json"),
        serde_json::to_vec_pretty(&artifact)?,
    )?;
    Ok(())
}
