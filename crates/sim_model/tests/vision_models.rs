use sha2::{Digest, Sha256};
use sim_model::{
    ArtifactIndex, ArtifactKey, ArtifactRoot, ModelStore, NativeEfficientNetBlockKind,
    NativeModule, NativeVisionModelError, NativeVisionStateKind, NativeVisionStateSpec,
    ParserLimits, efficientnet_v2_s_exact_native,
    efficientnet_v2_s_features_from_module_with_context,
    load_stage_c_efficientnet_feature_module_from_model_store_with_context,
    load_vision_state_from_model_store_with_context,
    load_vision_state_with_sibling_namespaces_from_model_store_with_context,
    raft_large_exact_native, vision_models::NativeEfficientNetV2SFeatureSource,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    Layout, StreamId, Tensor, TensorDescriptor,
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use std::{collections::BTreeMap, fs};

struct TestBackend {
    backend: CpuBackend,
    workspace_authority: CpuWorkspaceAuthority,
}

impl std::ops::Deref for TestBackend {
    type Target = CpuBackend;

    fn deref(&self) -> &Self::Target {
        &self.backend
    }
}

fn test_backend(memory_limit_bytes: u64) -> Result<TestBackend, Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(memory_limit_bytes)?;
    Ok(TestBackend {
        backend,
        workspace_authority,
    })
}

fn context<'a>(
    backend: &TestBackend,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, Box<dyn std::error::Error>> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        backend
            .workspace_authority
            .authorize_workspace(backend.memory_snapshot().limit_bytes)?,
        cancellation,
    ))
}

#[test]
fn efficientnet_v2_s_has_exact_torchvision_feature_schema() -> Result<(), Box<dyn std::error::Error>>
{
    let cancellation = CancellationToken::default();
    let mut model = efficientnet_v2_s_exact_native(&cancellation)?;
    assert_eq!(model.block_count(), 40);
    assert_eq!(model.parameter_count()?, 21_458_488);
    assert!(model.is_training());
    model.eval();
    assert!(!model.is_training());
    model.train();
    assert!(model.is_training());
    assert_eq!(
        model.stages()[0].block,
        NativeEfficientNetBlockKind::FusedMbConv
    );
    assert_eq!(model.stages()[3].block, NativeEfficientNetBlockKind::MbConv);
    assert_eq!(
        model.stages().map(|stage| stage.layers),
        [2, 4, 4, 6, 9, 15]
    );
    assert_schema_entry(&model, "features.0.0.weight", &[24, 3, 3, 3]);
    assert_schema_entry(&model, "features.1.0.block.0.0.weight", &[24, 24, 3, 3]);
    assert_schema_entry(&model, "features.4.0.block.2.fc1.weight", &[16, 256, 1, 1]);
    assert_schema_entry(&model, "features.6.14.block.3.0.weight", &[256, 1536, 1, 1]);
    assert_schema_entry(&model, "features.7.0.weight", &[1280, 256, 1, 1]);
    assert_schema_entry(&model, "classifier.1.weight", &[1000, 1280]);
    assert!(matches!(
        model.load_state_dict(BTreeMap::new(), &cancellation),
        Err(NativeVisionModelError::MissingState(name)) if name == "features.0.0.weight"
    ));
    assert!(!model.parameters_loaded());
    Ok(())
}

#[test]
fn raft_large_has_exact_native_architecture_and_twelve_update_default()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let mut model = raft_large_exact_native(false, false, &cancellation)?;
    assert_eq!(model.parameter_count()?, 5_257_536);
    assert_eq!(model.default_flow_updates(), 12);
    assert!(model.is_training());
    model.eval();
    assert!(!model.is_training());
    assert_schema_entry(
        &model,
        "feature_encoder.convnormrelu.0.weight",
        &[64, 3, 7, 7],
    );
    assert!(
        model
            .state_schema()
            .iter()
            .all(|spec| !spec.name.starts_with("feature_encoder.convnormrelu.1"))
    );
    assert_schema_entry(&model, "context_encoder.convnormrelu.1.running_var", &[64]);
    assert_schema_entry(
        &model,
        "update_block.motion_encoder.convcorr1.0.weight",
        &[256, 324, 1, 1],
    );
    assert_schema_entry(
        &model,
        "update_block.recurrent_block.convgru1.convz.weight",
        &[128, 384, 1, 5],
    );
    assert_schema_entry(
        &model,
        "update_block.recurrent_block.convgru2.convq.weight",
        &[128, 384, 5, 1],
    );
    assert_schema_entry(&model, "mask_predictor.conv.weight", &[576, 256, 1, 1]);
    Ok(())
}

#[test]
fn constructors_reject_downloads_and_cancel_without_publication() {
    let cancellation = CancellationToken::default();
    assert!(matches!(
        raft_large_exact_native(true, true, &cancellation),
        Err(NativeVisionModelError::Invalid(message)) if message.contains("never downloads")
    ));
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    assert!(matches!(
        efficientnet_v2_s_exact_native(&cancellation),
        Err(NativeVisionModelError::Cancelled)
    ));
    assert!(matches!(
        raft_large_exact_native(false, false, &cancellation),
        Err(NativeVisionModelError::Cancelled)
    ));
}

#[test]
fn efficientnet_feature_only_state_is_strict_atomic_and_semantically_equivalent()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let backend = test_backend(256 * 1024 * 1024)?;
    let mut feature_model = efficientnet_v2_s_exact_native(&cancellation)?;
    let feature_schema = feature_model.feature_state_schema()?.to_vec();
    assert_eq!(feature_model.feature_parameter_count()?, 20_177_488);
    assert!(
        feature_schema
            .iter()
            .all(|spec| spec.name.starts_with("features."))
    );
    assert!(
        feature_schema
            .iter()
            .all(|spec| !spec.name.starts_with("classifier."))
    );

    let full_state = zero_state(&backend, feature_model.state_schema(), &cancellation)?;
    let feature_state = feature_schema
        .iter()
        .map(|spec| {
            full_state
                .get(&spec.name)
                .cloned()
                .map(|tensor| (spec.name.clone(), tensor))
                .ok_or_else(|| format!("missing test feature state {}", spec.name))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    feature_model.load_feature_state_dict(feature_state.clone(), &cancellation)?;
    assert!(feature_model.feature_parameters_loaded());
    assert!(!feature_model.parameters_loaded());
    feature_model.eval();

    let image_values = (0..3 * 32 * 32)
        .map(|index| (index % 101) as f32 / 100.0)
        .collect::<Vec<_>>();
    let image = f32_tensor(&backend, &[1, 3, 32, 32], &image_values, &cancellation)?;
    let feature_output = feature_model.forward_features_with_context(
        &backend,
        &image,
        &context(&backend, &cancellation)?,
    )?;
    assert_eq!(feature_output.descriptor().shape(), &[1, 1280, 1, 1]);
    assert!(matches!(
        feature_model.forward_with_context(&backend, &image, &context(&backend, &cancellation)?),
        Err(NativeVisionModelError::ParametersNotLoaded)
    ));

    let mut full_model = efficientnet_v2_s_exact_native(&cancellation)?;
    full_model.load_state_dict(full_state, &cancellation)?;
    full_model.eval();
    let full_feature_output = full_model.forward_features_with_context(
        &backend,
        &image,
        &context(&backend, &cancellation)?,
    )?;
    assert_eq!(
        hash_tensors(&[&feature_output])?,
        hash_tensors(&[&full_feature_output])?
    );

    let feature_module = feature_model.feature_execution_module(&cancellation)?;
    let wrapped_feature_module =
        NativeModule::module_dict("stage-c.encoder", vec![feature_module.clone()])?;
    let bridged = efficientnet_v2_s_features_from_module_with_context(
        &wrapped_feature_module,
        &backend.backend,
        &image,
        &context(&backend, &cancellation)?,
    )?;
    assert_eq!(
        hash_tensors(&[&bridged])?,
        hash_tensors(&[&feature_output])?
    );

    let duplicate_left =
        NativeModule::module_dict("stage-c.duplicate.left", vec![feature_module.clone()])?;
    let duplicate_right =
        NativeModule::module_dict("stage-c.duplicate.right", vec![feature_module])?;
    let duplicate =
        NativeModule::module_dict("stage-c.duplicate", vec![duplicate_left, duplicate_right])?;
    assert!(matches!(
        efficientnet_v2_s_features_from_module_with_context(
            &duplicate,
            &backend.backend,
            &image,
            &context(&backend, &cancellation)?,
        ),
        Err(NativeVisionModelError::UnexpectedState(name))
            if name == "efficientnet_v2_s.features"
    ));
    let missing = NativeModule::module_dict("stage-c.missing", Vec::new())?;
    assert!(matches!(
        efficientnet_v2_s_features_from_module_with_context(
            &missing,
            &backend.backend,
            &image,
            &context(&backend, &cancellation)?,
        ),
        Err(NativeVisionModelError::MissingState(name))
            if name == "efficientnet_v2_s.features"
    ));

    let preserved_hash = hash_tensors(&[&feature_output])?;
    let mut missing = feature_state.clone();
    missing.remove("features.0.0.weight");
    assert!(matches!(
        feature_model.load_feature_state_dict(missing, &cancellation),
        Err(NativeVisionModelError::MissingState(name)) if name == "features.0.0.weight"
    ));
    let mut unexpected = feature_state.clone();
    unexpected.insert(
        "classifier.1.bias".into(),
        zero_tensor(&backend, &[1000], DType::F32, &cancellation)?,
    );
    assert!(matches!(
        feature_model.load_feature_state_dict(unexpected, &cancellation),
        Err(NativeVisionModelError::UnexpectedState(name)) if name == "classifier.1.bias"
    ));
    let mut wrong_shape = feature_state.clone();
    wrong_shape.insert(
        "features.0.0.weight".into(),
        zero_tensor(&backend, &[1], DType::F32, &cancellation)?,
    );
    assert!(matches!(
        feature_model.load_feature_state_dict(wrong_shape, &cancellation),
        Err(NativeVisionModelError::StateShape { name, .. }) if name == "features.0.0.weight"
    ));
    let mut wrong_dtype = feature_state.clone();
    wrong_dtype.insert(
        "features.0.0.weight".into(),
        zero_tensor(&backend, &[24, 3, 3, 3], DType::I64, &cancellation)?,
    );
    assert!(matches!(
        feature_model.load_feature_state_dict(wrong_dtype, &cancellation),
        Err(NativeVisionModelError::StateDType { name, .. }) if name == "features.0.0.weight"
    ));
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        feature_model.load_feature_state_dict(feature_state, &cancelled),
        Err(NativeVisionModelError::Cancelled)
    ));
    let preserved = feature_model.forward_features_with_context(
        &backend,
        &image,
        &context(&backend, &cancellation)?,
    )?;
    assert_eq!(hash_tensors(&[&preserved])?, preserved_hash);
    assert!(matches!(
        feature_model.forward_features_with_context(
            &backend,
            &image,
            &context(&backend, &cancelled)?
        ),
        Err(NativeVisionModelError::Cancelled)
    ));
    Ok(())
}

#[test]
fn stage_c_model_store_projection_enforces_exact_source_names()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let backend = test_backend(4 * 1024 * 1024)?;
    let directory = tempfile::tempdir()?;
    let cases = [
        (
            "encoder-extra.safetensors",
            "backbone.unexpected",
            NativeEfficientNetV2SFeatureSource::StableCascadeEncoder,
        ),
        (
            "combined-extra.safetensors",
            "encoder.backbone.unexpected",
            NativeEfficientNetV2SFeatureSource::StableCascadeCombined,
        ),
    ];
    for (file_name, tensor_name, _) in cases {
        write_single_f32_safetensors(&directory.path().join(file_name), tensor_name, &[1], &[0.0])?;
    }
    write_single_f32_safetensors(
        &directory.path().join("encoder-wrong-shape.safetensors"),
        "backbone.0.0.weight",
        &[1],
        &[0.0],
    )?;
    write_single_f32_safetensors(
        &directory.path().join("combined-sibling.safetensors"),
        "previewer.blocks.0.weight",
        &[1],
        &[0.0],
    )?;
    write_f32_safetensors(
        &directory.path().join("bounded-sibling.safetensors"),
        &[
            ("previewer.weight", &[1], &[1.5]),
            ("encoder.backbone.delegate", &[1], &[2.5]),
        ],
    )?;
    write_f32_safetensors(
        &directory.path().join("rogue-sibling.safetensors"),
        &[
            ("previewer.weight", &[1], &[1.5]),
            ("encoder.backbone.delegate", &[1], &[2.5]),
            ("rogue.weight", &[1], &[3.5]),
        ],
    )?;
    let root = ArtifactRoot::canonical("vision", "checkpoints", directory.path(), ["safetensors"])?;
    let mut index = ArtifactIndex::default();
    index.add_root(root)?;
    index.refresh(&cancellation)?;
    let mut store = ModelStore::new(ParserLimits::default())?;

    for (file_name, tensor_name, source) in cases {
        let key = ArtifactKey::new("vision", file_name)?;
        let loaded = store.load(&index, &key, &cancellation)?;
        let mut model = efficientnet_v2_s_exact_native(&cancellation)?;
        assert!(matches!(
            model.load_stage_c_features_from_model_store_with_context(
                &backend,
                &store,
                &index,
                &loaded,
                source,
                &context(&backend, &cancellation)?,
            ),
            Err(NativeVisionModelError::UnexpectedState(name)) if name == tensor_name
        ));
        assert!(!model.feature_parameters_loaded());
    }

    let key = ArtifactKey::new("vision", "encoder-wrong-shape.safetensors")?;
    let loaded = store.load(&index, &key, &cancellation)?;
    let mut model = efficientnet_v2_s_exact_native(&cancellation)?;
    assert!(matches!(
        load_stage_c_efficientnet_feature_module_from_model_store_with_context(
            &backend,
            &store,
            &index,
            &loaded,
            NativeEfficientNetV2SFeatureSource::StableCascadeEncoder,
            &context(&backend, &cancellation)?,
        ),
        Err(NativeVisionModelError::StateShape { name, .. }) if name == "backbone.0.0.weight"
    ));
    assert!(!model.feature_parameters_loaded());

    let key = ArtifactKey::new("vision", "combined-sibling.safetensors")?;
    let loaded_sibling = store.load(&index, &key, &cancellation)?;
    assert!(matches!(
        model.load_stage_c_features_from_model_store_with_context(
            &backend,
            &store,
            &index,
            &loaded_sibling,
            NativeEfficientNetV2SFeatureSource::StableCascadeCombined,
            &context(&backend, &cancellation)?,
        ),
        Err(NativeVisionModelError::MissingState(name)) if name == "encoder.backbone.0.0.weight"
    ));
    assert!(!model.feature_parameters_loaded());

    let sibling_schema = [NativeVisionStateSpec {
        name: "previewer.weight".into(),
        shape: vec![1],
        dtype: DType::F32,
        kind: NativeVisionStateKind::Parameter,
    }];
    let key = ArtifactKey::new("vision", "bounded-sibling.safetensors")?;
    let bounded_sibling = store.load(&index, &key, &cancellation)?;
    assert!(matches!(
        load_vision_state_from_model_store_with_context(
            &backend,
            &store,
            &index,
            &bounded_sibling,
            &sibling_schema,
            &context(&backend, &cancellation)?,
        ),
        Err(NativeVisionModelError::UnexpectedState(name))
            if name == "encoder.backbone.delegate"
    ));
    let bounded_state = load_vision_state_with_sibling_namespaces_from_model_store_with_context(
        &backend,
        &store,
        &index,
        &bounded_sibling,
        &sibling_schema,
        &["encoder.backbone."],
        &context(&backend, &cancellation)?,
    )?;
    assert_eq!(bounded_state.len(), 1);
    assert!(bounded_state.contains_key("previewer.weight"));
    assert!(matches!(
        load_vision_state_with_sibling_namespaces_from_model_store_with_context(
            &backend,
            &store,
            &index,
            &bounded_sibling,
            &sibling_schema,
            &["encoder.backbone"],
            &context(&backend, &cancellation)?,
        ),
        Err(NativeVisionModelError::Invalid(message))
            if message.contains("dotted prefixes")
    ));
    let key = ArtifactKey::new("vision", "rogue-sibling.safetensors")?;
    let rogue_sibling = store.load(&index, &key, &cancellation)?;
    assert!(matches!(
        load_vision_state_with_sibling_namespaces_from_model_store_with_context(
            &backend,
            &store,
            &index,
            &rogue_sibling,
            &sibling_schema,
            &["encoder.backbone."],
            &context(&backend, &cancellation)?,
        ),
        Err(NativeVisionModelError::UnexpectedState(name)) if name == "rogue.weight"
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        model.load_stage_c_features_from_model_store_with_context(
            &backend,
            &store,
            &index,
            &loaded,
            NativeEfficientNetV2SFeatureSource::StableCascadeEncoder,
            &context(&backend, &cancelled)?,
        ),
        Err(NativeVisionModelError::Cancelled)
    ));
    assert!(!model.feature_parameters_loaded());
    Ok(())
}

#[test]
fn sparse_state_dictionaries_load_atomically_and_execute_exact_native_forwards()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let backend = test_backend(256 * 1024 * 1024)?;
    let mut efficientnet = efficientnet_v2_s_exact_native(&cancellation)?;
    let mut state = zero_state(&backend, efficientnet.state_schema(), &cancellation)?;
    state.insert(
        "features.7.1.weight".into(),
        f32_tensor(&backend, &[1280], &vec![1.0; 1280], &cancellation)?,
    );
    state.insert(
        "features.7.1.running_var".into(),
        f32_tensor(&backend, &[1280], &vec![1.0; 1280], &cancellation)?,
    );
    state.insert(
        "features.7.1.bias".into(),
        f32_tensor(&backend, &[1280], &vec![0.0; 1280], &cancellation)?,
    );
    state.insert(
        "classifier.1.bias".into(),
        f32_tensor(&backend, &[1000], &vec![0.5; 1000], &cancellation)?,
    );
    let mut classifier_weight = vec![0.0; 1000 * 1280];
    classifier_weight[0] = 64.0;
    state.insert(
        "classifier.1.weight".into(),
        f32_tensor(&backend, &[1000, 1280], &classifier_weight, &cancellation)?,
    );
    let efficientnet_path = [
        "features.0.0.weight",
        "features.2.0.block.0.0.weight",
        "features.2.0.block.1.0.weight",
        "features.3.0.block.0.0.weight",
        "features.3.0.block.1.0.weight",
        "features.4.0.block.0.0.weight",
        "features.4.0.block.1.0.weight",
        "features.4.0.block.3.0.weight",
        "features.5.0.block.0.0.weight",
        "features.5.0.block.1.0.weight",
        "features.5.0.block.3.0.weight",
        "features.6.0.block.0.0.weight",
        "features.6.0.block.1.0.weight",
        "features.6.0.block.3.0.weight",
        "features.7.0.weight",
    ];
    for name in efficientnet_path {
        set_first_state_value(
            &backend,
            &mut state,
            efficientnet.state_schema(),
            name,
            0.25,
            &cancellation,
        )?;
    }
    for prefix in [
        "features.0.1",
        "features.2.0.block.0.1",
        "features.2.0.block.1.1",
        "features.3.0.block.0.1",
        "features.3.0.block.1.1",
        "features.4.0.block.0.1",
        "features.4.0.block.1.1",
        "features.4.0.block.3.1",
        "features.5.0.block.0.1",
        "features.5.0.block.1.1",
        "features.5.0.block.3.1",
        "features.6.0.block.0.1",
        "features.6.0.block.1.1",
        "features.6.0.block.3.1",
        "features.7.1",
    ] {
        set_batch_norm_identity(
            &backend,
            &mut state,
            efficientnet.state_schema(),
            prefix,
            &cancellation,
        )?;
    }
    let valid_efficient_state = state.clone();
    efficientnet.load_state_dict(state, &cancellation)?;
    assert!(matches!(
        efficientnet.forward_with_context(
            &backend,
            &zero_tensor(&backend, &[1, 3, 32, 32], DType::F32, &cancellation)?,
            &context(&backend, &cancellation)?
        ),
        Err(NativeVisionModelError::EvaluationRequired(
            "EfficientNet-V2-S"
        ))
    ));
    efficientnet.eval();
    let image_values = (0..3 * 32 * 32)
        .map(|index| (index % 97) as f32 / 96.0)
        .collect::<Vec<_>>();
    let image = f32_tensor(&backend, &[1, 3, 32, 32], &image_values, &cancellation)?;
    let features = efficientnet.forward_features_with_context(
        &backend,
        &image,
        &context(&backend, &cancellation)?,
    )?;
    assert_eq!(features.descriptor().shape(), &[1, 1280, 1, 1]);
    let feature_values = tensor_to_f32_with_context_exact_native(
        &backend,
        &features,
        &context(&backend, &cancellation)?,
    )?;
    assert!(feature_values.iter().any(|value| *value != 0.0));
    let classes =
        efficientnet.forward_with_context(&backend, &image, &context(&backend, &cancellation)?)?;
    assert_eq!(classes.descriptor().shape(), &[1, 1000]);
    let class_values = tensor_to_f32_with_context_exact_native(
        &backend,
        &classes,
        &context(&backend, &cancellation)?,
    )?;
    assert!(class_values[0] > 0.5);
    assert!(class_values[1..].iter().all(|value| *value == 0.5));
    let efficientnet_hash = hash_tensors(&[&features, &classes])?;
    assert_eq!(
        efficientnet_hash,
        "e320d073312d2d52729cfdf3d09ea964cbade51a78459d71317d150f809ea194"
    );
    let mut perturbed_state = valid_efficient_state.clone();
    set_first_state_value(
        &backend,
        &mut perturbed_state,
        efficientnet.state_schema(),
        "features.0.0.weight",
        0.5,
        &cancellation,
    )?;
    efficientnet.load_state_dict(perturbed_state, &cancellation)?;
    efficientnet.eval();
    let perturbed =
        efficientnet.forward_with_context(&backend, &image, &context(&backend, &cancellation)?)?;
    assert_ne!(hash_tensors(&[&perturbed])?, hash_tensors(&[&classes])?);
    efficientnet.load_state_dict(valid_efficient_state.clone(), &cancellation)?;
    efficientnet.eval();
    assert_eq!(
        hash_tensors(&[&efficientnet.forward_with_context(
            &backend,
            &image,
            &context(&backend, &cancellation)?
        )?])?,
        hash_tensors(&[&classes])?
    );
    let expected_classes_hash = hash_tensors(&[&classes])?;
    let mut unexpected = valid_efficient_state.clone();
    unexpected.insert(
        "unexpected.weight".into(),
        f32_tensor(&backend, &[1], &[1.0], &cancellation)?,
    );
    assert!(matches!(
        efficientnet.load_state_dict(unexpected, &cancellation),
        Err(NativeVisionModelError::UnexpectedState(name)) if name == "unexpected.weight"
    ));
    let mut wrong_shape = valid_efficient_state.clone();
    wrong_shape.insert(
        "classifier.1.bias".into(),
        zero_tensor(&backend, &[999], DType::F32, &cancellation)?,
    );
    assert!(matches!(
        efficientnet.load_state_dict(wrong_shape, &cancellation),
        Err(NativeVisionModelError::StateShape { name, .. }) if name == "classifier.1.bias"
    ));
    let mut wrong_dtype = valid_efficient_state.clone();
    wrong_dtype.insert(
        "classifier.1.bias".into(),
        zero_tensor(&backend, &[1000], DType::I64, &cancellation)?,
    );
    assert!(matches!(
        efficientnet.load_state_dict(wrong_dtype, &cancellation),
        Err(NativeVisionModelError::StateDType { name, .. }) if name == "classifier.1.bias"
    ));
    let mut noncontiguous = valid_efficient_state;
    noncontiguous.insert(
        "classifier.1.bias".into(),
        noncontiguous_f32_tensor(&backend, 1000, &cancellation)?,
    );
    assert!(matches!(
        efficientnet.load_state_dict(noncontiguous, &cancellation),
        Err(NativeVisionModelError::Invalid(message)) if message.contains("contiguous")
    ));
    let preserved =
        efficientnet.forward_with_context(&backend, &image, &context(&backend, &cancellation)?)?;
    assert_eq!(hash_tensors(&[&preserved])?, expected_classes_hash);
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        efficientnet.forward_with_context(&backend, &image, &context(&backend, &cancelled)?),
        Err(NativeVisionModelError::Cancelled)
    ));

    drop(efficientnet);
    let mut raft = raft_large_exact_native(false, false, &cancellation)?;
    let mut state = zero_state(&backend, raft.state_schema(), &cancellation)?;
    let mut feature_stem_weight = vec![0.0; 64 * 3 * 7 * 7];
    feature_stem_weight[24] = 0.25;
    state.insert(
        "feature_encoder.convnormrelu.0.weight".into(),
        f32_tensor(
            &backend,
            &[64, 3, 7, 7],
            &feature_stem_weight,
            &cancellation,
        )?,
    );
    state.insert(
        "feature_encoder.conv.bias".into(),
        f32_tensor(&backend, &[256], &vec![0.0; 256], &cancellation)?,
    );
    for name in [
        "feature_encoder.layer2.0.downsample.0.weight",
        "feature_encoder.layer3.0.downsample.0.weight",
        "feature_encoder.conv.weight",
    ] {
        set_first_state_value(
            &backend,
            &mut state,
            raft.state_schema(),
            name,
            0.25,
            &cancellation,
        )?;
    }
    state.insert(
        "context_encoder.conv.bias".into(),
        f32_tensor(&backend, &[256], &vec![0.0625; 256], &cancellation)?,
    );
    state.insert(
        "update_block.recurrent_block.convgru1.convq.bias".into(),
        f32_tensor(&backend, &[128], &vec![0.03125; 128], &cancellation)?,
    );
    state.insert(
        "mask_predictor.conv.bias".into(),
        f32_tensor(&backend, &[576], &vec![0.015625; 576], &cancellation)?,
    );
    state.insert(
        "update_block.flow_head.conv2.bias".into(),
        f32_tensor(&backend, &[2], &[0.25, -0.5], &cancellation)?,
    );
    raft.load_state_dict(state, &cancellation)?;
    raft.eval();
    let image_values = (0..3 * 128 * 128)
        .map(|index| ((index % 251) as f32 / 250.0 + 1.0) * 0.5)
        .collect::<Vec<_>>();
    let image = f32_tensor(&backend, &[1, 3, 128, 128], &image_values, &cancellation)?;
    let predictions = raft.forward_with_context(
        &backend,
        &image,
        &image,
        1,
        &context(&backend, &cancellation)?,
    )?;
    assert_eq!(predictions.len(), 1);
    assert!(
        predictions
            .iter()
            .all(|prediction| prediction.descriptor().shape() == [1, 2, 128, 128])
    );
    let first = tensor_to_f32_with_context_exact_native(
        &backend,
        &predictions[0],
        &context(&backend, &cancellation)?,
    )?;
    assert!(first.iter().any(|value| *value != 0.0));
    assert_eq!(
        hash_tensors(&predictions.iter().collect::<Vec<_>>())?,
        "d536d191f0d367fdd36962fc6063cad03a65b0100ba12dde12ae5787733ea237"
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        raft.forward_with_context(&backend, &image, &image, 1, &context(&backend, &cancelled)?),
        Err(NativeVisionModelError::Cancelled)
    ));
    drop(raft);

    let mut twelve_step = raft_large_exact_native(false, false, &cancellation)?;
    let mut twelve_step_state = zero_state(&backend, twelve_step.state_schema(), &cancellation)?;
    twelve_step_state.insert(
        "update_block.flow_head.conv2.bias".into(),
        f32_tensor(&backend, &[2], &[0.125, -0.25], &cancellation)?,
    );
    twelve_step.load_state_dict(twelve_step_state, &cancellation)?;
    twelve_step.eval();
    let zero_image = zero_tensor(&backend, &[1, 3, 128, 128], DType::F32, &cancellation)?;
    let first_run = twelve_step.forward_with_context(
        &backend,
        &zero_image,
        &zero_image,
        12,
        &context(&backend, &cancellation)?,
    )?;
    assert_eq!(first_run.len(), 12);
    assert!(
        first_run
            .iter()
            .all(|prediction| prediction.descriptor().shape() == [1, 2, 128, 128])
    );
    let first_values = tensor_to_f32_with_context_exact_native(
        &backend,
        &first_run[0],
        &context(&backend, &cancellation)?,
    )?;
    let last_values = tensor_to_f32_with_context_exact_native(
        &backend,
        &first_run[11],
        &context(&backend, &cancellation)?,
    )?;
    assert!(last_values[0].abs() > first_values[0].abs());
    assert_eq!(
        hash_tensors(&first_run.iter().collect::<Vec<_>>())?,
        "aa1132f4ea023267736c41e075e1404329640b21940781495b216148575e7928"
    );
    Ok(())
}

#[test]
fn model_store_is_the_verified_state_loading_boundary() -> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let backend = test_backend(1024 * 1024)?;
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("vision.safetensors");
    write_single_f32_safetensors(&path, "weight", &[2], &[1.5, -2.25])?;
    write_single_safetensor_bytes(
        &directory.path().join("half.safetensors"),
        "weight",
        "F16",
        &[2],
        &[0x00, 0x3c, 0x00, 0xc0],
    )?;
    write_single_safetensor_bytes(
        &directory.path().join("bfloat16.safetensors"),
        "weight",
        "BF16",
        &[2],
        &[0x80, 0x3f, 0x00, 0xc0],
    )?;
    let i64_payload = [1_i64, -2_i64]
        .into_iter()
        .flat_map(i64::to_le_bytes)
        .collect::<Vec<_>>();
    write_single_safetensor_bytes(
        &directory.path().join("int64.safetensors"),
        "weight",
        "I64",
        &[2],
        &i64_payload,
    )?;
    let root = ArtifactRoot::canonical("vision", "checkpoints", directory.path(), ["safetensors"])?;
    let mut index = ArtifactIndex::default();
    index.add_root(root)?;
    index.refresh(&cancellation)?;
    let key = ArtifactKey::new("vision", "vision.safetensors")?;
    let mut store = ModelStore::new(ParserLimits::default())?;
    let loaded = store.load(&index, &key, &cancellation)?;
    let schema = [NativeVisionStateSpec {
        name: "weight".into(),
        shape: vec![2],
        dtype: DType::F32,
        kind: NativeVisionStateKind::Parameter,
    }];
    let state = load_vision_state_from_model_store_with_context(
        &backend,
        &store,
        &index,
        &loaded,
        &schema,
        &context(&backend, &cancellation)?,
    )?;
    assert_eq!(
        tensor_to_f32_with_context_exact_native(
            &backend,
            state.get("weight").ok_or("missing loaded state")?,
            &context(&backend, &cancellation)?,
        )?,
        [1.5, -2.25]
    );
    for (file_name, dtype, expected_bits) in [
        ("half.safetensors", DType::F16, [0x3c00_u16, 0xc000]),
        ("bfloat16.safetensors", DType::Bf16, [0x3f80_u16, 0xc000]),
    ] {
        let key = ArtifactKey::new("vision", file_name)?;
        let loaded = store.load(&index, &key, &cancellation)?;
        let schema = [NativeVisionStateSpec {
            name: "weight".into(),
            shape: vec![2],
            dtype,
            kind: NativeVisionStateKind::Parameter,
        }];
        let state = load_vision_state_from_model_store_with_context(
            &backend,
            &store,
            &index,
            &loaded,
            &schema,
            &context(&backend, &cancellation)?,
        )?;
        let tensor = state.get("weight").ok_or("missing reduced state")?;
        assert_eq!(tensor.descriptor().dtype(), dtype);
        let expected = expected_bits
            .into_iter()
            .flat_map(u16::to_ne_bytes)
            .collect::<Vec<_>>();
        assert_eq!(tensor.contiguous_bytes()?, expected);
    }
    {
        let key = ArtifactKey::new("vision", "int64.safetensors")?;
        let loaded = store.load(&index, &key, &cancellation)?;
        let schema = [NativeVisionStateSpec {
            name: "weight".into(),
            shape: vec![2],
            dtype: DType::I64,
            kind: NativeVisionStateKind::Parameter,
        }];
        let state = load_vision_state_from_model_store_with_context(
            &backend,
            &store,
            &index,
            &loaded,
            &schema,
            &context(&backend, &cancellation)?,
        )?;
        let tensor = state.get("weight").ok_or("missing integer state")?;
        assert_eq!(tensor.descriptor().dtype(), DType::I64);
        let expected = [1_i64, -2_i64]
            .into_iter()
            .flat_map(i64::to_ne_bytes)
            .collect::<Vec<_>>();
        assert_eq!(tensor.contiguous_bytes()?, expected);
    }

    let exact = backend.workspace_authority.authorize_workspace(8)?;
    let exact_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: exact.clone(),
        rng_phase: None,
        cancellation: &cancellation,
    };
    let state = sim_model::vision_models::load_vision_state_from_model_store_with_context(
        &backend,
        &store,
        &index,
        &loaded,
        &schema,
        &exact_context,
    )?;
    assert_eq!(exact.peak_bytes(), 8);
    assert_eq!(exact.in_use_bytes(), 0);
    assert_eq!(
        tensor_to_f32_with_context_exact_native(
            &backend,
            state.get("weight").ok_or("missing context-loaded state")?,
            &exact_context,
        )?,
        [1.5, -2.25]
    );

    let insufficient = backend.workspace_authority.authorize_workspace(7)?;
    let insufficient_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: insufficient.clone(),
        rng_phase: None,
        cancellation: &cancellation,
    };
    assert!(matches!(
        sim_model::vision_models::load_vision_state_from_model_store_with_context(
            &backend,
            &store,
            &index,
            &loaded,
            &schema,
            &insufficient_context,
        ),
        Err(NativeVisionModelError::TensorStorage(
            sim_tensor::TensorError::WorkspaceAuthorizationExceeded { .. }
        ))
    ));
    assert_eq!(insufficient.in_use_bytes(), 0);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_scratch = backend.workspace_authority.authorize_workspace(8)?;
    let cancelled_context = ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: cancelled_scratch.clone(),
        rng_phase: None,
        cancellation: &cancelled,
    };
    assert!(
        sim_model::vision_models::load_vision_state_from_model_store_with_context(
            &backend,
            &store,
            &index,
            &loaded,
            &schema,
            &cancelled_context,
        )
        .is_err()
    );
    assert_eq!(cancelled_scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn vision_workspace_ownership_inventory_is_bounded() {
    let source = include_str!("../src/vision_models.rs");
    let production = source
        .split("#[cfg(test)]")
        .next()
        .expect("vision source has a production section");
    assert!(production.contains("CpuWorkspaceVec"));
    assert!(production.contains("forward_features_with_context"));
    assert!(production.contains("forward_with_context"));
    assert!(production.contains("load_from_model_store_with_context"));
    assert!(production.contains("load_vision_state_from_model_store_with_context"));
    assert!(!production.contains("authorize_workspace("));
    assert_eq!(production.matches("ScratchReservation::none()").count(), 0);
    assert_eq!(production.matches("tensor_to_f32_exact_native").count(), 0);
    assert_eq!(
        production.matches("tensor_from_f32_exact_native").count(),
        0
    );
}

#[test]
fn invalid_forward_shapes_fail_before_model_execution() -> Result<(), Box<dyn std::error::Error>> {
    let backend = test_backend(4 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let mut efficientnet = efficientnet_v2_s_exact_native(&cancellation)?;
    let image = zero_tensor(&backend, &[1, 3, 32, 32], DType::F32, &cancellation)?;
    assert!(matches!(
        efficientnet.forward_with_context(&backend, &image, &context(&backend, &cancellation)?),
        Err(NativeVisionModelError::ParametersNotLoaded)
    ));
    let mut raft = raft_large_exact_native(false, false, &cancellation)?;
    let invalid = zero_tensor(&backend, &[1, 3, 120, 128], DType::F32, &cancellation)?;
    assert!(matches!(
        raft.forward_with_context(&backend, &invalid, &invalid, 12, &context(&backend, &cancellation)?),
        Err(NativeVisionModelError::Invalid(message)) if message.contains("divisible by eight") || message.contains("at least 128")
    ));
    Ok(())
}

trait HasSchema {
    fn schema(&self) -> &[sim_model::NativeVisionStateSpec];
}

impl HasSchema for sim_model::NativeEfficientNetV2S {
    fn schema(&self) -> &[sim_model::NativeVisionStateSpec] {
        self.state_schema()
    }
}

impl HasSchema for sim_model::NativeRaftLarge {
    fn schema(&self) -> &[sim_model::NativeVisionStateSpec] {
        self.state_schema()
    }
}

fn assert_schema_entry(model: &impl HasSchema, name: &str, shape: &[u64]) {
    let entry = model
        .schema()
        .iter()
        .find(|entry| entry.name == name)
        .unwrap_or_else(|| panic!("missing schema entry {name}"));
    assert_eq!(entry.shape, shape);
}

fn zero_state(
    backend: &TestBackend,
    schema: &[NativeVisionStateSpec],
    cancellation: &CancellationToken,
) -> Result<BTreeMap<String, Tensor>, Box<dyn std::error::Error>> {
    schema
        .iter()
        .map(|spec| {
            Ok((
                spec.name.clone(),
                zero_tensor(backend, &spec.shape, spec.dtype, cancellation)?,
            ))
        })
        .collect()
}

fn zero_tensor(
    backend: &TestBackend,
    shape: &[u64],
    dtype: DType,
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let element_count = shape.iter().try_fold(1_usize, |count, dimension| {
        let dimension =
            usize::try_from(*dimension).map_err(|_| "test tensor dimension overflow")?;
        count
            .checked_mul(dimension)
            .ok_or("test tensor shape overflow")
    })?;
    let byte_count = element_count
        .checked_mul(usize::try_from(dtype.byte_width())?)
        .ok_or("test tensor byte count overflow")?;
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, StreamId::DEFAULT)?;
    let context = context(backend, cancellation)?;
    Ok(backend
        .upload_bytes(descriptor, &vec![0; byte_count], &context)?
        .0)
}

fn f32_tensor(
    backend: &TestBackend,
    shape: &[u64],
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    let context = context(backend, cancellation)?;
    Ok(backend.upload_f32(descriptor, values, &context)?.0)
}

fn set_first_state_value(
    backend: &TestBackend,
    state: &mut BTreeMap<String, Tensor>,
    schema: &[NativeVisionStateSpec],
    name: &str,
    value: f32,
    cancellation: &CancellationToken,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = schema
        .iter()
        .find(|spec| spec.name == name)
        .ok_or_else(|| format!("missing state schema entry {name}"))?;
    if spec.dtype != DType::F32 {
        return Err(format!("state schema entry {name} is not F32").into());
    }
    let mut count = 1_usize;
    for dimension in &spec.shape {
        count = count
            .checked_mul(usize::try_from(*dimension)?)
            .ok_or("state shape overflow")?;
    }
    let mut values = vec![0.0; count];
    let selected_count = if spec.shape.len() == 4 {
        let kernel_height = usize::try_from(spec.shape[2])?;
        let kernel_width = usize::try_from(spec.shape[3])?;
        kernel_height
            .checked_mul(kernel_width)
            .ok_or("state kernel shape overflow")?
    } else {
        1
    };
    let selected = values
        .get_mut(..selected_count)
        .ok_or_else(|| format!("state schema entry {name} is empty"))?;
    selected.fill(value);
    state.insert(
        name.into(),
        f32_tensor(backend, &spec.shape, &values, cancellation)?,
    );
    Ok(())
}

fn set_batch_norm_identity(
    backend: &TestBackend,
    state: &mut BTreeMap<String, Tensor>,
    schema: &[NativeVisionStateSpec],
    prefix: &str,
    cancellation: &CancellationToken,
) -> Result<(), Box<dyn std::error::Error>> {
    for suffix in ["weight", "running_var"] {
        let name = format!("{prefix}.{suffix}");
        let shape = schema
            .iter()
            .find(|spec| spec.name == name)
            .ok_or_else(|| format!("missing batch-normalization schema entry {name}"))?
            .shape
            .clone();
        let count = usize::try_from(
            shape
                .iter()
                .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
                .ok_or("batch-normalization state shape overflow")?,
        )?;
        state.insert(
            name,
            f32_tensor(backend, &shape, &vec![1.0; count], cancellation)?,
        );
    }
    Ok(())
}

fn noncontiguous_f32_tensor(
    backend: &TestBackend,
    length: u64,
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor = TensorDescriptor::new_strided(
        vec![length],
        vec![2],
        0,
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    let storage = descriptor
        .storage_span_bytes()?
        .ok_or("noncontiguous test tensor has no storage span")?;
    let byte_count = usize::try_from(storage.end - storage.start)?;
    let context = context(backend, cancellation)?;
    Ok(backend
        .upload_bytes(descriptor, &vec![0; byte_count], &context)?
        .0)
}

fn write_single_f32_safetensors(
    path: &std::path::Path,
    name: &str,
    shape: &[u64],
    values: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let data = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let mut header = serde_json::to_vec(&serde_json::json!({
        (name): {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [0, data.len()],
        }
    }))?;
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut encoded = Vec::with_capacity(8 + header.len() + data.len());
    encoded.extend_from_slice(&u64::try_from(header.len())?.to_le_bytes());
    encoded.extend_from_slice(&header);
    encoded.extend_from_slice(&data);
    fs::write(path, encoded)?;
    Ok(())
}

fn write_single_safetensor_bytes(
    path: &std::path::Path,
    name: &str,
    dtype: &str,
    shape: &[u64],
    data: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = serde_json::to_vec(&serde_json::json!({
        (name): {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [0, data.len()],
        }
    }))?;
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut encoded = Vec::with_capacity(8 + header.len() + data.len());
    encoded.extend_from_slice(&u64::try_from(header.len())?.to_le_bytes());
    encoded.extend_from_slice(&header);
    encoded.extend_from_slice(data);
    fs::write(path, encoded)?;
    Ok(())
}

fn write_f32_safetensors(
    path: &std::path::Path,
    entries: &[(&str, &[u64], &[f32])],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut data = Vec::new();
    let mut header = serde_json::Map::new();
    for (name, shape, values) in entries {
        let start = data.len();
        data.extend(values.iter().flat_map(|value| value.to_le_bytes()));
        let end = data.len();
        header.insert(
            (*name).to_owned(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
    }
    let mut header = serde_json::to_vec(&header)?;
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut encoded = Vec::with_capacity(8 + header.len() + data.len());
    encoded.extend_from_slice(&u64::try_from(header.len())?.to_le_bytes());
    encoded.extend_from_slice(&header);
    encoded.extend_from_slice(&data);
    fs::write(path, encoded)?;
    Ok(())
}

fn hash_tensors(tensors: &[&Tensor]) -> Result<String, Box<dyn std::error::Error>> {
    let mut digest = Sha256::new();
    for tensor in tensors {
        digest.update(tensor.contiguous_bytes()?);
    }
    Ok(format!("{:x}", digest.finalize()))
}
