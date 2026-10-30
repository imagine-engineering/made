use sim_model::{
    GENERATED_MODEL_FAMILY_REGISTRATIONS, ModelFamilyError, ModelFamilyRegistry, ModelParsedFacts,
    ModelParsedTensorFact, ModelProbe, ModelStateTransaction, NativeFamilyBuildOptions,
    NativeFamilyModel, generated_auraflow_sim_model_0064 as aura,
    generated_qwenimage_sim_model_0113 as qwen,
};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext, StreamId,
    Tensor, generated_native_diffusion::tensor_from_f32,
};
use sim_types::DeviceKind;
use std::{collections::BTreeMap, error::Error};

const MEMORY_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
const ARTIFACT_DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn parsed_probe(tensors: BTreeMap<String, Vec<u64>>) -> Result<ModelProbe, ModelFamilyError> {
    ModelProbe::from_parsed_facts(ModelParsedFacts {
        tensors: tensors
            .into_iter()
            .map(|(key, shape)| {
                (
                    key,
                    ModelParsedTensorFact {
                        shape,
                        storage_dtype: DType::F32.catalog_name().to_owned(),
                    },
                )
            })
            .collect(),
        formats: Vec::new(),
    })
}

fn aura_probe() -> Result<ModelProbe, ModelFamilyError> {
    let mut tensors = BTreeMap::from([
        (
            "model.diffusion_model.cond_seq_linear.weight".to_owned(),
            vec![3_072, 2_048],
        ),
        (
            "model.diffusion_model.positional_encoding".to_owned(),
            vec![1, 1_024, 3_072],
        ),
        (
            "model.diffusion_model.final_linear.weight".to_owned(),
            vec![16, 3_072],
        ),
    ]);
    for ordinal in 0..4 {
        tensors.insert(
            format!("model.diffusion_model.double_layers.{ordinal}.attn.w1q.weight"),
            vec![3_072, 3_072],
        );
    }
    for ordinal in 0..32 {
        tensors.insert(
            format!("model.diffusion_model.single_layers.{ordinal}.attn.w1q.weight"),
            vec![3_072, 3_072],
        );
    }
    parsed_probe(tensors)
}

fn qwen_probe() -> Result<ModelProbe, ModelFamilyError> {
    let mut tensors = BTreeMap::from([
        (
            "model.diffusion_model.txt_norm.weight".to_owned(),
            vec![3_584],
        ),
        (
            "model.diffusion_model.img_in.weight".to_owned(),
            vec![3_072, 64],
        ),
        (
            "model.diffusion_model.txt_in.weight".to_owned(),
            vec![3_072, 3_584],
        ),
        (
            "model.diffusion_model.proj_out.weight".to_owned(),
            vec![64, 3_072],
        ),
    ]);
    for ordinal in 0..60 {
        let prefix = format!("model.diffusion_model.transformer_blocks.{ordinal}");
        tensors.insert(format!("{prefix}.img_mod.1.weight"), vec![18_432, 3_072]);
        tensors.insert(format!("{prefix}.txt_mod.1.weight"), vec![18_432, 3_072]);
        tensors.insert(format!("{prefix}.attn.to_q.weight"), vec![3_072, 3_072]);
        tensors.insert(format!("{prefix}.attn.norm_q.weight"), vec![128]);
    }
    parsed_probe(tensors)
}

fn patterned_values(key: &str, elements: usize) -> Vec<f32> {
    let normalization = key.contains("norm_") || key.ends_with("txt_norm.weight");
    let bias = key.ends_with(".bias");
    let aura_mlp =
        key.contains(".mlpC.") || key.contains(".mlpX.") || key.contains("single_layers.0.mlp.");
    let aura_attention = key.contains("native.double_layers.0.attn.");
    (0..elements)
        .map(|index| {
            if normalization {
                0.95 + (index % 7) as f32 * 0.01
            } else if bias {
                ((index % 11) as f32 - 5.0) * 0.002
            } else if aura_mlp {
                ((index % 17) as f32 - 8.0) * 0.5
            } else if aura_attention {
                ((index % 17) as f32 - 8.0) * 0.1
            } else {
                ((index % 17) as f32 - 8.0) * 0.000_75
            }
        })
        .collect()
}

fn aura_shape(key: &str) -> Result<&'static [u64], Box<dyn Error>> {
    Ok(match key {
        "native.init_x_linear.weight" => &[2, 16],
        "native.init_x_linear.bias" => &[2],
        "native.positional_encoding" => &[1, 16, 2],
        "native.register_tokens" => &[1, 8, 2],
        "native.cond_seq_linear.weight" => &[2, 2_048],
        "native.t_embedder.mlp.0.weight" => &[2, 256],
        "native.t_embedder.mlp.0.bias" | "native.t_embedder.mlp.2.bias" => &[2],
        "native.t_embedder.mlp.2.weight" => &[2, 2],
        "native.double_layers.0.modC.1.weight"
        | "native.double_layers.0.modX.1.weight"
        | "native.single_layers.0.modCX.1.weight" => &[12, 2],
        key if key.contains(".attn.") => &[2, 2],
        key if key.ends_with("c_fc1.weight") || key.ends_with("c_fc2.weight") => &[256, 2],
        key if key.ends_with("c_proj.weight") => &[2, 256],
        "native.modF.1.weight" => &[4, 2],
        "native.final_linear.weight" => &[16, 2],
        _ => return Err(format!("missing Aura projection shape for {key}").into()),
    })
}

fn qwen_shape(key: &str) -> Result<&'static [u64], Box<dyn Error>> {
    Ok(match key {
        "native.img_in.weight" => &[128, 64],
        "native.img_in.bias"
        | "native.txt_in.bias"
        | "native.time_text_embed.timestep_embedder.linear_1.bias"
        | "native.time_text_embed.timestep_embedder.linear_2.bias" => &[128],
        "native.txt_norm.weight" => &[3_584],
        "native.txt_in.weight" => &[128, 3_584],
        "native.time_text_embed.timestep_embedder.linear_1.weight" => &[128, 256],
        "native.time_text_embed.timestep_embedder.linear_2.weight" => &[128, 128],
        key if key.ends_with("img_mod.1.weight") || key.ends_with("txt_mod.1.weight") => {
            &[768, 128]
        }
        key if key.ends_with("img_mod.1.bias") || key.ends_with("txt_mod.1.bias") => &[768],
        key if key.contains("attn.norm_") => &[128],
        key if key.contains("attn.") && key.ends_with(".weight") => &[128, 128],
        key if key.contains("attn.") && key.ends_with(".bias") => &[128],
        key if key.ends_with("mlp.net.0.proj.weight") => &[512, 128],
        key if key.ends_with("mlp.net.0.proj.bias") => &[512],
        key if key.ends_with("mlp.net.2.weight") => &[128, 512],
        key if key.ends_with("mlp.net.2.bias") => &[128],
        "native.norm_out.linear.weight" => &[256, 128],
        "native.norm_out.linear.bias" => &[256],
        "native.proj_out.weight" => &[64, 128],
        "native.proj_out.bias" => &[64],
        "native.__sampling_shift__"
        | "native.__reference_method__"
        | "native.__additional_timestep_condition__" => &[1],
        _ => return Err(format!("missing Qwen projection shape for {key}").into()),
    })
}

fn projection_state(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    qwen_model: bool,
) -> Result<BTreeMap<String, Tensor>, Box<dyn Error>> {
    let keys: Vec<&str> = if qwen_model {
        qwen::DENOISER_INVOCATION_REQUIRED_KEYS
            .iter()
            .copied()
            .chain([
                "native.__sampling_shift__",
                "native.__reference_method__",
                "native.__additional_timestep_condition__",
            ])
            .collect()
    } else {
        aura::DENOISER_INVOCATION_REQUIRED_KEYS.to_vec()
    };
    keys.into_iter()
        .map(|key| {
            let shape = if qwen_model {
                qwen_shape(key)?
            } else {
                aura_shape(key)?
            };
            let elements = shape.iter().try_fold(1_usize, |total, dimension| {
                total.checked_mul(usize::try_from(*dimension).ok()?)
            });
            let elements = elements.ok_or("projection tensor shape overflow")?;
            let values = match key {
                "native.__sampling_shift__" => vec![1.15],
                "native.__reference_method__" | "native.__additional_timestep_condition__" => {
                    vec![0.0]
                }
                _ => patterned_values(key, elements),
            };
            let tensor = tensor_from_f32(backend, shape, &values, context)?;
            Ok((key.to_owned(), tensor))
        })
        .collect()
}

fn options(budget: u64) -> NativeFamilyBuildOptions {
    NativeFamilyBuildOptions {
        dtype: DType::F32,
        device: DeviceKind::Cpu,
        activation_elements: 1,
        memory_budget_bytes: budget,
        allow_unexpected_weights: false,
    }
}

fn bind(
    registry: &ModelFamilyRegistry,
    probe: &ModelProbe,
    state: BTreeMap<String, Tensor>,
    context: &ExecutionContext<'_>,
    budget: u64,
) -> Result<NativeFamilyModel, ModelFamilyError> {
    registry.resolve(probe)?.bind_reduced_execution_projection(
        ARTIFACT_DIGEST,
        state,
        options(budget),
        context,
    )
}

#[test]
fn production_profiles_resolve_before_reduced_execution_binding() -> Result<(), Box<dyn Error>> {
    let registry =
        ModelFamilyRegistry::checked_registrations(GENERATED_MODEL_FAMILY_REGISTRATIONS)?;
    let aura = registry.resolve(&aura_probe()?)?;
    assert_eq!(aura.definition().feature_id, "COMFY-MODEL-0064");
    assert_eq!(aura.source_ordinal(), 22);
    assert_eq!(aura.source_architecture(), "model_base.AuraFlow");
    assert_eq!(aura.profile().latent_identifier, "SDXL");
    assert!(aura.state_plan().is_some());

    let qwen = registry.resolve(&qwen_probe()?)?;
    assert_eq!(qwen.definition().feature_id, "COMFY-MODEL-0113");
    assert_eq!(qwen.source_ordinal(), 77);
    assert_eq!(qwen.source_architecture(), "model_base.QwenImage");
    assert_eq!(qwen.profile().latent_identifier, "Wan21");
    assert_eq!(
        qwen::configuration_for_probe(&qwen_probe()?)?.inner_dimension,
        3_072
    );
    Ok(())
}

#[test]
fn resolved_profiles_bind_deterministic_projection_identity_and_residency()
-> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let workspace = authority.authorize_workspace(MEMORY_LIMIT)?;
    let context = backend.execution_context(StreamId::DEFAULT, workspace, &cancellation);
    let registry =
        ModelFamilyRegistry::checked_registrations(GENERATED_MODEL_FAMILY_REGISTRATIONS)?;
    for (probe, qwen_model) in [(aura_probe()?, false), (qwen_probe()?, true)] {
        let first = bind(
            &registry,
            &probe,
            projection_state(&backend, &context, qwen_model)?,
            &context,
            MEMORY_LIMIT,
        )?;
        let second = bind(
            &registry,
            &probe,
            projection_state(&backend, &context, qwen_model)?,
            &context,
            MEMORY_LIMIT,
        )?;
        assert_eq!(first.source_ordinal(), second.source_ordinal());
        assert_eq!(
            first.weights().cache_identity(),
            second.weights().cache_identity()
        );
        assert_eq!(
            first.execution_projection_identity(),
            second.execution_projection_identity()
        );
        assert_eq!(
            first.execution_projection_state_digest(),
            second.execution_projection_state_digest()
        );
        let binding = first
            .weights()
            .binding()
            .ok_or("missing projection binding")?;
        assert_eq!(binding.source_ordinal(), first.source_ordinal());
        assert_eq!(binding.source_architecture(), first.source_architecture());
        assert_eq!(
            binding.probe_identity(),
            Some(registry.resolve(&probe)?.probe_identity())
        );
        assert!(binding.execution_projection_identity().is_some());
        assert!(binding.execution_projection_state_digest().is_some());
        assert!(first.weights().resident_owned_bytes()? > 0);
        let storage_ids = first
            .weights()
            .tensors()
            .values()
            .map(|tensor| tensor.storage_id().get())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(storage_ids.len(), first.weights().tensors().len());
        let reconstructed = first.with_weights(second.weights().clone())?;
        task376::assert_bound_model_oracle(&first, &backend, &context, qwen_model)?;
        task376::assert_bound_model_oracle(&reconstructed, &backend, &context, qwen_model)?;

        let mut changed_state = projection_state(&backend, &context, qwen_model)?;
        let changed_key = if qwen_model {
            "native.img_in.bias"
        } else {
            "native.init_x_linear.bias"
        };
        let changed_shape = if qwen_model { &[128][..] } else { &[2][..] };
        let changed_values = vec![0.25; changed_shape[0] as usize];
        changed_state.insert(
            changed_key.to_owned(),
            tensor_from_f32(&backend, changed_shape, &changed_values, &context)?,
        );
        let changed = bind(&registry, &probe, changed_state, &context, MEMORY_LIMIT)?;
        assert_ne!(
            first.execution_projection_state_digest(),
            changed.execution_projection_state_digest()
        );
        assert_ne!(
            first.weights().cache_identity(),
            changed.weights().cache_identity()
        );
        assert!(first.with_weights(changed.weights().clone()).is_err());

        let different_artifact = registry
            .resolve(&probe)?
            .bind_reduced_execution_projection(
                "f".repeat(64),
                projection_state(&backend, &context, qwen_model)?,
                options(MEMORY_LIMIT),
                &context,
            )?;
        assert!(
            first
                .with_weights(different_artifact.weights().clone())
                .is_err()
        );

        let mut probe_drift = probe.clone();
        probe_drift
            .metadata
            .insert("fixture.probe-drift".to_owned(), "present".to_owned());
        let different_probe = bind(
            &registry,
            &probe_drift,
            projection_state(&backend, &context, qwen_model)?,
            &context,
            MEMORY_LIMIT,
        )?;
        assert!(
            first
                .with_weights(different_probe.weights().clone())
                .is_err()
        );
    }
    Ok(())
}

#[test]
fn projection_state_confusion_and_marker_drift_fail_closed() -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let workspace = authority.authorize_workspace(MEMORY_LIMIT)?;
    let context = backend.execution_context(StreamId::DEFAULT, workspace, &cancellation);
    let registry =
        ModelFamilyRegistry::checked_registrations(GENERATED_MODEL_FAMILY_REGISTRATIONS)?;
    let qwen_probe = qwen_probe()?;
    let resolved = registry.resolve(&qwen_probe)?;

    let mut timestep_zero_probe = qwen_probe.clone();
    timestep_zero_probe.tensor_shapes.insert(
        "model.diffusion_model.__index_timestep_zero__".to_owned(),
        Vec::new(),
    );
    let resolved_timestep_zero = registry.resolve(&timestep_zero_probe)?;
    assert!(matches!(
        resolved_timestep_zero.bind_reduced_execution_projection(
            ARTIFACT_DIGEST,
            projection_state(&backend, &context, true)?,
            options(MEMORY_LIMIT),
            &context,
        ),
        Err(ModelFamilyError::ExecutionProjectionBinding(_))
    ));

    let mut additional_timestep_probe = qwen_probe.clone();
    additional_timestep_probe.tensor_shapes.insert(
        "model.diffusion_model.time_text_embed.addition_t_embedding.weight".to_owned(),
        vec![2, 3_072],
    );
    let resolved_additional = registry.resolve(&additional_timestep_probe)?;
    assert!(matches!(
        resolved_additional.bind_reduced_execution_projection(
            ARTIFACT_DIGEST,
            projection_state(&backend, &context, true)?,
            options(MEMORY_LIMIT),
            &context,
        ),
        Err(ModelFamilyError::ExecutionProjectionBinding(_))
    ));

    let mut channel_drift_probe = qwen_probe;
    channel_drift_probe.tensor_shapes.insert(
        "model.diffusion_model.img_in.weight".to_owned(),
        vec![3_072, 128],
    );
    let resolved_channel_drift = registry.resolve(&channel_drift_probe)?;
    assert!(matches!(
        resolved_channel_drift.bind_reduced_execution_projection(
            ARTIFACT_DIGEST,
            projection_state(&backend, &context, true)?,
            options(MEMORY_LIMIT),
            &context,
        ),
        Err(ModelFamilyError::ExecutionProjectionBinding(_))
    ));

    let reduced = projection_state(&backend, &context, true)?;
    assert!(matches!(
        resolved.map_primary_weights(
            &ModelStateTransaction::new(&backend, &context),
            ARTIFACT_DIGEST,
            &reduced,
        ),
        Err(ModelFamilyError::ResolvedProbeDrift(_))
    ));

    let mut missing = reduced.clone();
    missing.remove("native.img_in.weight");
    assert!(matches!(
        resolved.bind_reduced_execution_projection(
            ARTIFACT_DIGEST,
            missing,
            options(MEMORY_LIMIT),
            &context,
        ),
        Err(ModelFamilyError::ExecutionProjectionState(_))
    ));

    let mut production_width = reduced.clone();
    production_width.insert(
        "native.img_in.weight".to_owned(),
        tensor_from_f32(&backend, &[3_072, 64], &vec![0.0; 3_072 * 64], &context)?,
    );
    assert!(matches!(
        resolved.bind_reduced_execution_projection(
            ARTIFACT_DIGEST,
            production_width,
            options(MEMORY_LIMIT),
            &context,
        ),
        Err(ModelFamilyError::ExecutionProjectionState(_))
    ));

    let mut marker_drift = reduced;
    marker_drift.insert(
        "native.__reference_method__".to_owned(),
        tensor_from_f32(&backend, &[1], &[1.0], &context)?,
    );
    assert!(matches!(
        resolved.bind_reduced_execution_projection(
            ARTIFACT_DIGEST,
            marker_drift,
            options(MEMORY_LIMIT),
            &context,
        ),
        Err(ModelFamilyError::ExecutionProjectionState(_))
    ));

    let mut nonfinite = projection_state(&backend, &context, true)?;
    nonfinite.insert(
        "native.img_in.bias".to_owned(),
        tensor_from_f32(&backend, &[128], &vec![f32::NAN; 128], &context)?,
    );
    assert!(
        resolved
            .bind_reduced_execution_projection(
                ARTIFACT_DIGEST,
                nonfinite,
                options(MEMORY_LIMIT),
                &context,
            )
            .is_err()
    );

    let mut wrong_dtype_options = options(MEMORY_LIMIT);
    wrong_dtype_options.dtype = DType::Bf16;
    assert!(matches!(
        resolved.bind_reduced_execution_projection(
            ARTIFACT_DIGEST,
            projection_state(&backend, &context, true)?,
            wrong_dtype_options,
            &context,
        ),
        Err(ModelFamilyError::ExecutionProjectionUnavailable(_))
    ));

    let mut wrong_device_options = options(MEMORY_LIMIT);
    wrong_device_options.device = DeviceKind::Metal;
    assert!(matches!(
        resolved.bind_reduced_execution_projection(
            ARTIFACT_DIGEST,
            projection_state(&backend, &context, true)?,
            wrong_device_options,
            &context,
        ),
        Err(ModelFamilyError::ExecutionProjectionUnavailable(_))
    ));

    let other_workspace = authority.authorize_workspace(MEMORY_LIMIT)?;
    let other_context = backend.execution_context(StreamId::new(1), other_workspace, &cancellation);
    assert!(matches!(
        resolved.bind_reduced_execution_projection(
            ARTIFACT_DIGEST,
            projection_state(&backend, &context, true)?,
            options(MEMORY_LIMIT),
            &other_context,
        ),
        Err(ModelFamilyError::ExecutionProjectionState(_))
    ));
    Ok(())
}

#[test]
fn projection_binding_honors_cancellation_and_memory_limits() -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(MEMORY_LIMIT)?;
    let workspace = authority.authorize_workspace(MEMORY_LIMIT)?;
    let context = backend.execution_context(StreamId::DEFAULT, workspace, &cancellation);
    let registry =
        ModelFamilyRegistry::checked_registrations(GENERATED_MODEL_FAMILY_REGISTRATIONS)?;
    let probe = aura_probe()?;
    let state = projection_state(&backend, &context, false)?;
    assert!(matches!(
        bind(&registry, &probe, state, &context, 0),
        Err(ModelFamilyError::OutOfMemory { .. })
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let workspace = authority.authorize_workspace(MEMORY_LIMIT)?;
    let cancelled_context = backend.execution_context(StreamId::DEFAULT, workspace, &cancelled);
    let state = projection_state(&backend, &context, false)?;
    assert!(matches!(
        bind(&registry, &probe, state, &cancelled_context, MEMORY_LIMIT,),
        Err(ModelFamilyError::Cancelled(_))
    ));
    Ok(())
}

mod task376 {
    include!("native_family_model_invocation.rs");

    pub(super) fn assert_bound_model_oracle(
        model: &NativeFamilyModel,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        qwen_model: bool,
    ) -> Result<(), Box<dyn Error>> {
        if qwen_model {
            let latent_values = (0..144)
                .map(|index| (index as f32 - 72.0) * 0.005)
                .collect::<Vec<_>>();
            let latent = tensor_from_f32(backend, &[1, 16, 1, 3, 3], &latent_values, context)?;
            let conditioning = tensor_from_f32(
                backend,
                &[1, 2, 3_584],
                &patterned_values("qwen.conditioning", 2 * 3_584),
                context,
            )?;
            let (identity, entries) = resolved_conditioning(
                model,
                &latent,
                conditioning,
                "qwen.projection.oracle",
                backend,
                context,
            )?;
            let time = tensor_from_f32(backend, &[1], &[0.271_828_18], context)?;
            let mask = tensor_from_f32(backend, &[1, 2], &[0.0, -0.75], context)?;
            let values = invoke_values(
                model,
                &latent,
                &time,
                &identity,
                &entries[0],
                Some(&mask),
                backend,
                context,
            )?;
            assert_independent_oracle(&values, &QWEN_ORACLE_BITS);
        } else {
            let latent_values = (0..36)
                .map(|index| (index as f32 - 18.0) * 0.025)
                .collect::<Vec<_>>();
            let latent = tensor_from_f32(backend, &[1, 4, 3, 3], &latent_values, context)?;
            let conditioning = tensor_from_f32(
                backend,
                &[1, 2, 2_048],
                &patterned_values("aura.conditioning", 2 * 2_048),
                context,
            )?;
            let (identity, entries) = resolved_conditioning(
                model,
                &latent,
                conditioning,
                "aura.projection.oracle",
                backend,
                context,
            )?;
            let time = tensor_from_f32(backend, &[1], &[0.314_159_27], context)?;
            let values = invoke_values(
                model,
                &latent,
                &time,
                &identity,
                &entries[0],
                None,
                backend,
                context,
            )?;
            assert_pinned_auraflow_legacy_oracle(&values)?;
        }
        Ok(())
    }
}
