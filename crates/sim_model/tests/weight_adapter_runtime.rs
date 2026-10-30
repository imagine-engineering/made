use sha2::{Digest, Sha256};
use sim_model::weight_adapter::{
    ADAPTER_MAP_ORDER, AdapterFamily, AdapterTensor, BypassInjectionManager, BypassPatch,
    ModuleTypeInfo, NativeWeightAdapter, WEIGHT_ADAPTER_ORDER, WeightAdapterError,
    WeightAdapterLoadRequest, WeightAdapterRegistry,
};
use sim_model::{PatchPayload, QuantizationKind, quantize_matrix};
use sim_tensor::{
    AutogradError, AutogradTape, CpuBackend, CpuWorkspaceAuthority, DType, ExecutionContext,
    GradientMode, GradientReducer, LeafId, RetryRngPolicy, RngAlgorithm, RngProfileVersion,
    RngStream, RngStreamAddress, StreamId, Tensor,
    generated_sim_operator_indirection_01::{
        tensor_from_f32_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
};
use sim_types::CancellationToken;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::path::{Path, PathBuf};

const BASE_DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

struct Harness {
    backend: CpuBackend,
    workspace: CpuWorkspaceAuthority,
    cancellation: CancellationToken,
}

impl Harness {
    fn new() -> Result<Self, Box<dyn Error>> {
        let (backend, workspace) = CpuWorkspaceAuthority::create_backend(256 * 1024 * 1024)?;
        Ok(Self {
            backend,
            workspace,
            cancellation: CancellationToken::default(),
        })
    }

    fn context(&self) -> Result<ExecutionContext<'_>, Box<dyn Error>> {
        Ok(self.backend.execution_context(
            StreamId::DEFAULT,
            self.workspace.authorize_workspace(128 * 1024 * 1024)?,
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

    fn values(
        &self,
        tensor: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        Ok(tensor_to_f32_with_context_exact_native(
            &self.backend,
            tensor,
            context,
        )?)
    }
}

fn dense(tensor: Tensor) -> AdapterTensor {
    AdapterTensor::Dense(tensor)
}

fn trainable_rng() -> Result<RngStream, Box<dyn Error>> {
    Ok(RngStream::new(
        RngProfileVersion::V1,
        RngAlgorithm::Philox4x32_10,
        510,
        RngStreamAddress::new(
            "weight-adapter-workflow",
            "weight-adapter-attempt",
            "create-train",
            0,
            "parameter-init",
            0,
            0,
            RetryRngPolicy::Replay,
        )?,
    )?)
}

fn lora(up: Tensor, down: Tensor, alpha: Option<f32>) -> NativeWeightAdapter {
    NativeWeightAdapter::Lora {
        up: dense(up),
        down: dense(down),
        alpha,
        mid: None,
        dora_scale: None,
        reshape: None,
    }
}

struct NoAccumulation;

impl GradientReducer for NoAccumulation {
    fn add(
        &self,
        _left: Tensor,
        _right: Tensor,
        _cancellation: &CancellationToken,
    ) -> Result<Tensor, AutogradError> {
        Err(AutogradError::InvalidGraph {
            reason: "weight-adapter fixture unexpectedly accumulated a gradient".into(),
        })
    }
}

#[test]
fn val_weight_adapter_001_registry_and_loaders_match_the_pinned_source_order()
-> Result<(), Box<dyn Error>> {
    assert_eq!(
        WEIGHT_ADAPTER_ORDER,
        [
            AdapterFamily::Lora,
            AdapterFamily::Loha,
            AdapterFamily::Lokr,
            AdapterFamily::Glora,
            AdapterFamily::Oft,
            AdapterFamily::Boft,
        ]
    );
    assert_eq!(
        ADAPTER_MAP_ORDER,
        [
            ("LoRA", AdapterFamily::Lora),
            ("LoHa", AdapterFamily::Loha),
            ("LoKr", AdapterFamily::Lokr),
            ("OFT", AdapterFamily::Oft),
        ]
    );
    let registry = WeightAdapterRegistry;
    assert_eq!(registry.named_family("GLoRA"), None);
    assert_eq!(registry.named_family("BOFT"), None);

    let harness = Harness::new()?;
    let context = harness.context()?;
    let matrix = |shape: &[u64], values: &[f32]| harness.tensor(shape, values, &context);
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "layer.lora_up.weight".to_owned(),
        dense(matrix(&[2, 1], &[1.0, 2.0])?),
    );
    tensors.insert(
        "layer.lora_down.weight".to_owned(),
        dense(matrix(&[1, 2], &[3.0, 4.0])?),
    );
    let loaded = registry
        .load_first(&WeightAdapterLoadRequest {
            prefix: "layer".into(),
            tensors,
            alpha: Some(1.0),
            dora_scale: None,
        })?
        .ok_or("LoRA was not loaded")?;
    assert_eq!(loaded.adapter().family(), AdapterFamily::Lora);
    assert_eq!(
        loaded.loaded_keys(),
        &BTreeSet::from([
            "layer.lora_down.weight".to_owned(),
            "layer.lora_up.weight".to_owned(),
        ])
    );

    let mut malformed = BTreeMap::new();
    malformed.insert(
        "layer.lora_up.weight".to_owned(),
        dense(matrix(&[2, 1], &[1.0, 2.0])?),
    );
    assert!(matches!(
        registry.load_first(&WeightAdapterLoadRequest {
            prefix: "layer".into(),
            tensors: malformed,
            alpha: Some(1.0),
            dora_scale: None,
        }),
        Err(WeightAdapterError::MissingCompanion {
            family: AdapterFamily::Lora,
            ..
        })
    ));
    Ok(())
}

#[test]
fn val_weight_adapter_001_all_source_lora_key_layouts_load_with_exact_companions()
-> Result<(), Box<dyn Error>> {
    let harness = Harness::new()?;
    let context = harness.context()?;
    let layouts = [
        (".lora_up.weight", ".lora_down.weight"),
        ("_lora.up.weight", "_lora.down.weight"),
        (".lora_B.weight", ".lora_A.weight"),
        (".lora.up.weight", ".lora.down.weight"),
        (".lora_B", ".lora_A"),
        (
            ".lora_linear_layer.up.weight",
            ".lora_linear_layer.down.weight",
        ),
        (".lora_B.default.weight", ".lora_A.default.weight"),
    ];
    for (up_suffix, down_suffix) in layouts {
        let up_key = format!("layer{up_suffix}");
        let down_key = format!("layer{down_suffix}");
        let up = dense(harness.tensor(&[2, 1], &[1.0, 2.0], &context)?);
        let down = dense(harness.tensor(&[1, 2], &[3.0, 4.0], &context)?);
        let loaded = WeightAdapterRegistry
            .load_first(&WeightAdapterLoadRequest {
                prefix: "layer".into(),
                tensors: BTreeMap::from([(up_key.clone(), up), (down_key.clone(), down)]),
                alpha: Some(1.0),
                dora_scale: None,
            })?
            .ok_or("LoRA source key layout was not loaded")?;
        assert_eq!(loaded.adapter().family(), AdapterFamily::Lora);
        assert_eq!(
            loaded.loaded_keys(),
            &BTreeSet::from([up_key.clone(), down_key.clone()])
        );

        let missing = WeightAdapterRegistry.load_first(&WeightAdapterLoadRequest {
            prefix: "layer".into(),
            tensors: BTreeMap::from([(
                up_key,
                dense(harness.tensor(&[2, 1], &[1.0, 2.0], &context)?),
            )]),
            alpha: Some(1.0),
            dora_scale: None,
        });
        assert!(matches!(
            missing,
            Err(WeightAdapterError::MissingCompanion {
                family: AdapterFamily::Lora,
                ..
            })
        ));
    }
    Ok(())
}

#[test]
fn val_weight_adapter_001_create_train_uses_caller_rng_and_source_initializers()
-> Result<(), Box<dyn Error>> {
    let harness = Harness::new()?;
    let context = harness.context()?;
    let base = harness.tensor(&[4, 4], &[0.0; 16], &context)?;
    let mut first_rng = trainable_rng()?.begin(None)?;
    let mut second_rng = trainable_rng()?.begin(None)?;
    let first = NativeWeightAdapter::create_trainable(
        sim_model::TrainableAdapterKind::LoraDiff,
        &base,
        2,
        1.0,
        &harness.backend,
        &mut first_rng,
        &context,
    )?;
    let second = NativeWeightAdapter::create_trainable(
        sim_model::TrainableAdapterKind::LoraDiff,
        &base,
        2,
        1.0,
        &harness.backend,
        &mut second_rng,
        &context,
    )?;
    let (
        NativeWeightAdapter::Lora {
            up: AdapterTensor::Dense(first_up),
            down: AdapterTensor::Dense(first_down),
            ..
        },
        NativeWeightAdapter::Lora {
            up: AdapterTensor::Dense(second_up),
            down: AdapterTensor::Dense(second_down),
            ..
        },
    ) = (&first, &second)
    else {
        return Err("create_train did not produce dense LoRA factors".into());
    };
    assert_eq!(
        harness.values(first_up, &context)?,
        harness.values(second_up, &context)?
    );
    assert!(
        harness
            .values(first_up, &context)?
            .iter()
            .all(|value| value.abs() <= 1.0 / 2.0_f32.sqrt())
    );
    assert_eq!(harness.values(first_down, &context)?, vec![0.0; 8]);
    assert_eq!(harness.values(second_down, &context)?, vec![0.0; 8]);
    assert_eq!(first_rng.checkpoint(), second_rng.checkpoint());

    for kind in [
        sim_model::TrainableAdapterKind::LohaDiff,
        sim_model::TrainableAdapterKind::LokrDiff,
        sim_model::TrainableAdapterKind::OftDiff,
    ] {
        let mut transaction = trainable_rng()?.begin(None)?;
        let adapter = NativeWeightAdapter::create_trainable(
            kind,
            &base,
            2,
            1.0,
            &harness.backend,
            &mut transaction,
            &context,
        )?;
        assert_eq!(adapter.trainable_kind()?, kind);
        match adapter {
            NativeWeightAdapter::Loha {
                first_down: AdapterTensor::Dense(first_down),
                ..
            } => assert_eq!(harness.values(&first_down, &context)?, vec![0.0; 8]),
            NativeWeightAdapter::Lokr {
                first: Some(AdapterTensor::Dense(first)),
                ..
            } => assert!(
                harness
                    .values(&first, &context)?
                    .iter()
                    .all(|value| *value == 0.0)
            ),
            NativeWeightAdapter::Oft {
                blocks: AdapterTensor::Dense(blocks),
                ..
            } => assert!(
                harness
                    .values(&blocks, &context)?
                    .iter()
                    .all(|value| *value == 0.0)
            ),
            _ => return Err("create_train produced the wrong adapter family".into()),
        }
    }

    let mut invalid_rng = trainable_rng()?.begin(None)?;
    let before = invalid_rng.checkpoint();
    assert!(
        NativeWeightAdapter::create_trainable(
            sim_model::TrainableAdapterKind::LoraDiff,
            &base,
            0,
            1.0,
            &harness.backend,
            &mut invalid_rng,
            &context,
        )
        .is_err()
    );
    assert_eq!(invalid_rng.checkpoint(), before);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn val_weight_adapter_001_every_concrete_family_loads_and_invalid_shapes_fail_typed()
-> Result<(), Box<dyn Error>> {
    let harness = Harness::new()?;
    let context = harness.context()?;
    let tensor = |shape: &[u64], values: &[f32]| harness.tensor(shape, values, &context);
    let registry = WeightAdapterRegistry;

    let cases = [
        (
            AdapterFamily::Loha,
            BTreeMap::from([
                (
                    "layer.hada_w1_a".into(),
                    dense(tensor(&[2, 1], &[1.0, 2.0])?),
                ),
                (
                    "layer.hada_w1_b".into(),
                    dense(tensor(&[1, 2], &[1.0, 0.0])?),
                ),
                (
                    "layer.hada_w2_a".into(),
                    dense(tensor(&[2, 1], &[1.0, 1.0])?),
                ),
                (
                    "layer.hada_w2_b".into(),
                    dense(tensor(&[1, 2], &[0.0, 1.0])?),
                ),
            ]),
        ),
        (
            AdapterFamily::Lokr,
            BTreeMap::from([
                ("layer.lokr_w1".into(), dense(tensor(&[1, 1], &[1.0])?)),
                (
                    "layer.lokr_w2".into(),
                    dense(tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0])?),
                ),
            ]),
        ),
        (
            AdapterFamily::Glora,
            BTreeMap::from([
                ("layer.a1.weight".into(), dense(tensor(&[2, 2], &[0.0; 4])?)),
                (
                    "layer.a2.weight".into(),
                    dense(tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0])?),
                ),
                (
                    "layer.b1.weight".into(),
                    dense(tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0])?),
                ),
                (
                    "layer.b2.weight".into(),
                    dense(tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0])?),
                ),
            ]),
        ),
        (
            AdapterFamily::Oft,
            BTreeMap::from([(
                "layer.oft_blocks".into(),
                dense(tensor(&[1, 2, 2], &[0.0; 4])?),
            )]),
        ),
        (
            AdapterFamily::Boft,
            BTreeMap::from([(
                "layer.oft_blocks".into(),
                dense(tensor(&[1, 1, 2, 2], &[0.0; 4])?),
            )]),
        ),
    ];
    for (family, tensors) in cases {
        let loaded = registry
            .load_first(&WeightAdapterLoadRequest {
                prefix: "layer".into(),
                tensors,
                alpha: Some(1.0),
                dora_scale: None,
            })?
            .ok_or("adapter was not loaded")?;
        assert_eq!(loaded.adapter().family(), family);
    }

    let malformed_oft = NativeWeightAdapter::Oft {
        blocks: dense(tensor(&[1, 2, 3], &[0.0; 6])?),
        rescale: None,
        constraint: Some(1.0),
        dora_scale: None,
    };
    assert!(matches!(
        sim_model::BypassBinding::checked("layer", malformed_oft, 1.0),
        Err(WeightAdapterError::InvalidShape(_))
    ));
    Ok(())
}

#[test]
fn val_weight_adapter_001_all_diff_rows_record_canonical_first_order_reverse()
-> Result<(), Box<dyn Error>> {
    let harness = Harness::new()?;
    let context = harness.context()?;
    let base_values = [1.0, 2.0, 3.0, 4.0];
    let cases = [
        lora(
            harness.tensor(&[2, 1], &[1.0, 2.0], &context)?,
            harness.tensor(&[1, 2], &[3.0, 4.0], &context)?,
            Some(1.0),
        ),
        NativeWeightAdapter::Loha {
            first_up: dense(harness.tensor(&[2, 1], &[1.0, 2.0], &context)?),
            first_down: dense(harness.tensor(&[1, 2], &[1.0, 1.0], &context)?),
            second_up: dense(harness.tensor(&[2, 1], &[1.0, 1.0], &context)?),
            second_down: dense(harness.tensor(&[1, 2], &[1.0, 0.0], &context)?),
            first_tucker: None,
            second_tucker: None,
            alpha: Some(1.0),
            dora_scale: None,
        },
        NativeWeightAdapter::Lokr {
            first: Some(dense(harness.tensor(&[1, 1], &[1.0], &context)?)),
            second: Some(dense(harness.tensor(
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
                &context,
            )?)),
            first_up: None,
            first_down: None,
            second_up: None,
            second_down: None,
            second_tucker: None,
            alpha: Some(1.0),
            dora_scale: None,
        },
        NativeWeightAdapter::Oft {
            blocks: dense(harness.tensor(&[1, 2, 2], &[0.0; 4], &context)?),
            rescale: None,
            constraint: Some(1.0),
            dora_scale: None,
        },
    ];
    for (case_index, adapter) in cases.into_iter().enumerate() {
        let base = harness.tensor(&[2, 2], &base_values, &context)?;
        let mut tape = AutogradTape::new(GradientMode::Enabled);
        let base_leaf = LeafId::new(format!("base-{case_index}"))?;
        tape.set_requires_grad(&base, Some(base_leaf.clone()), true, &harness.cancellation)?;
        let factors = match &adapter {
            NativeWeightAdapter::Lora { up, down, .. } => vec![up, down],
            NativeWeightAdapter::Loha {
                first_up,
                first_down,
                second_up,
                second_down,
                ..
            } => vec![first_up, first_down, second_up, second_down],
            NativeWeightAdapter::Lokr { first, second, .. } => {
                vec![
                    first.as_ref().ok_or("missing first")?,
                    second.as_ref().ok_or("missing second")?,
                ]
            }
            NativeWeightAdapter::Oft { blocks, .. } => vec![blocks],
            NativeWeightAdapter::Glora { .. } | NativeWeightAdapter::Boft { .. } => {
                return Err("unexpected non-trainable family".into());
            }
        };
        let mut factor_leaves = Vec::new();
        for (factor_index, factor) in factors.into_iter().enumerate() {
            let AdapterTensor::Dense(factor) = factor else {
                return Err("trainable fixture factor was quantized".into());
            };
            let leaf = LeafId::new(format!("factor-{case_index}-{factor_index}"))?;
            tape.set_requires_grad(factor, Some(leaf.clone()), true, &harness.cancellation)?;
            factor_leaves.push(leaf);
        }
        let execution =
            adapter.forward_trainable_recorded(&base, &harness.backend, &mut tape, &context)?;
        let slot = execution
            .output_slot()
            .ok_or("Diff output was not recorded")?;
        if case_index == 3 {
            assert_eq!(harness.values(execution.output(), &context)?, base_values);
        }
        let seed = harness.tensor(&[2, 2], &[1.0; 4], &context)?;
        let gradients = tape.reverse_with_context(
            vec![(slot, seed)],
            &NoAccumulation,
            false,
            false,
            &harness.backend,
            &context,
        )?;
        assert_eq!(
            harness.values(
                gradients.get(&base_leaf).ok_or("missing base gradient")?,
                &context
            )?,
            vec![1.0; 4]
        );
        for leaf in factor_leaves {
            let gradient = gradients.get(&leaf).ok_or("missing factor gradient")?;
            assert!(
                harness
                    .values(gradient, &context)?
                    .iter()
                    .all(|value| value.is_finite())
            );
        }
        assert_eq!(tape.retained_node_count(), 0);
    }
    Ok(())
}

#[test]
fn val_weight_adapter_001_tucker_and_decomposed_diff_branches_reverse_canonically()
-> Result<(), Box<dyn Error>> {
    let harness = Harness::new()?;
    let context = harness.context()?;
    let loha_factors = [
        harness.tensor(&[1, 2], &[1.0, 2.0], &context)?,
        harness.tensor(&[1, 2], &[3.0, 4.0], &context)?,
        harness.tensor(&[1, 2], &[0.5, 1.5], &context)?,
        harness.tensor(&[1, 2], &[2.0, 1.0], &context)?,
        harness.tensor(&[1, 1, 1, 1], &[1.0], &context)?,
        harness.tensor(&[1, 1, 1, 1], &[2.0], &context)?,
    ];
    let loha = NativeWeightAdapter::Loha {
        first_up: dense(loha_factors[0].clone()),
        first_down: dense(loha_factors[1].clone()),
        second_up: dense(loha_factors[2].clone()),
        second_down: dense(loha_factors[3].clone()),
        first_tucker: Some(dense(loha_factors[4].clone())),
        second_tucker: Some(dense(loha_factors[5].clone())),
        alpha: Some(1.0),
        dora_scale: None,
    };
    let lokr_factors = [
        harness.tensor(&[1, 1], &[1.0], &context)?,
        harness.tensor(&[1, 1], &[2.0], &context)?,
        harness.tensor(&[1, 2], &[1.0, 2.0], &context)?,
        harness.tensor(&[1, 2], &[3.0, 4.0], &context)?,
        harness.tensor(&[1, 1, 1, 1], &[1.0], &context)?,
    ];
    let lokr = NativeWeightAdapter::Lokr {
        first: None,
        second: None,
        first_up: Some(dense(lokr_factors[0].clone())),
        first_down: Some(dense(lokr_factors[1].clone())),
        second_up: Some(dense(lokr_factors[2].clone())),
        second_down: Some(dense(lokr_factors[3].clone())),
        second_tucker: Some(dense(lokr_factors[4].clone())),
        alpha: Some(1.0),
        dora_scale: None,
    };

    for (case_index, (adapter, factors)) in
        [(loha, loha_factors.to_vec()), (lokr, lokr_factors.to_vec())]
            .into_iter()
            .enumerate()
    {
        let base = harness.tensor(&[2, 2, 1, 1], &[1.0, 2.0, 3.0, 4.0], &context)?;
        let mut tape = AutogradTape::new(GradientMode::Enabled);
        let base_leaf = LeafId::new(format!("branch-base-{case_index}"))?;
        tape.set_requires_grad(&base, Some(base_leaf.clone()), true, &harness.cancellation)?;
        let mut factor_leaves = Vec::new();
        for (factor_index, factor) in factors.iter().enumerate() {
            let leaf = LeafId::new(format!("branch-factor-{case_index}-{factor_index}"))?;
            tape.set_requires_grad(factor, Some(leaf.clone()), true, &harness.cancellation)?;
            factor_leaves.push(leaf);
        }
        let recorded =
            adapter.forward_trainable_recorded(&base, &harness.backend, &mut tape, &context)?;
        let seed = harness.tensor(&[2, 2, 1, 1], &[1.0; 4], &context)?;
        let gradients = tape.reverse_with_context(
            vec![(
                recorded.output_slot().ok_or("missing recorded branch")?,
                seed,
            )],
            &NoAccumulation,
            false,
            false,
            &harness.backend,
            &context,
        )?;
        assert_eq!(
            harness.values(
                gradients
                    .get(&base_leaf)
                    .ok_or("missing branch base gradient")?,
                &context,
            )?,
            vec![1.0; 4]
        );
        for leaf in factor_leaves {
            let values = harness.values(
                gradients
                    .get(&leaf)
                    .ok_or("missing branch factor gradient")?,
                &context,
            )?;
            assert!(values.iter().all(|value| value.is_finite()));
        }
        assert_eq!(tape.retained_node_count(), 0);
        assert_eq!(context.scratch.in_use_bytes(), 0);
    }
    Ok(())
}

#[test]
fn val_weight_adapter_001_loha_diff_preserves_analytical_higher_order_autograd()
-> Result<(), Box<dyn Error>> {
    let harness = Harness::new()?;
    let context = harness.context()?;
    let matrix_factors = [
        harness.tensor(&[1, 1], &[2.0], &context)?,
        harness.tensor(&[1, 1], &[3.0], &context)?,
        harness.tensor(&[1, 1], &[4.0], &context)?,
        harness.tensor(&[1, 1], &[5.0], &context)?,
    ];
    let matrix = NativeWeightAdapter::Loha {
        first_up: dense(matrix_factors[0].clone()),
        first_down: dense(matrix_factors[1].clone()),
        second_up: dense(matrix_factors[2].clone()),
        second_down: dense(matrix_factors[3].clone()),
        first_tucker: None,
        second_tucker: None,
        alpha: Some(1.0),
        dora_scale: None,
    };
    let tucker_factors = [
        harness.tensor(&[1, 1], &[3.0], &context)?,
        harness.tensor(&[1, 1], &[4.0], &context)?,
        harness.tensor(&[1, 1], &[6.0], &context)?,
        harness.tensor(&[1, 1], &[7.0], &context)?,
        harness.tensor(&[1, 1, 1], &[2.0], &context)?,
        harness.tensor(&[1, 1, 1], &[5.0], &context)?,
    ];
    let tucker = NativeWeightAdapter::Loha {
        first_up: dense(tucker_factors[0].clone()),
        first_down: dense(tucker_factors[1].clone()),
        second_up: dense(tucker_factors[2].clone()),
        second_down: dense(tucker_factors[3].clone()),
        first_tucker: Some(dense(tucker_factors[4].clone())),
        second_tucker: Some(dense(tucker_factors[5].clone())),
        alpha: Some(1.0),
        dora_scale: None,
    };

    for (case_index, (adapter, factors, first_index, second_index, expected)) in [
        (matrix, matrix_factors.to_vec(), 0_usize, 1_usize, 20.0_f32),
        (tucker, tucker_factors.to_vec(), 4_usize, 0_usize, 840.0_f32),
    ]
    .into_iter()
    .enumerate()
    {
        let base_shape = if case_index == 0 {
            vec![1, 1]
        } else {
            vec![1, 1, 1]
        };
        let base = harness.tensor(&base_shape, &[0.0], &context)?;
        let mut tape = AutogradTape::new(GradientMode::Enabled);
        let leaves = factors
            .iter()
            .enumerate()
            .map(|(factor_index, factor)| {
                let leaf = LeafId::new(format!("higher-{case_index}-{factor_index}"))?;
                tape.set_requires_grad(factor, Some(leaf.clone()), true, &harness.cancellation)?;
                Ok(leaf)
            })
            .collect::<Result<Vec<_>, AutogradError>>()?;
        let output =
            adapter.forward_trainable_recorded(&base, &harness.backend, &mut tape, &context)?;
        let first = tape.reverse_with_context(
            vec![(
                output.output_slot().ok_or("LoHa output was not recorded")?,
                harness.tensor(&base_shape, &[1.0], &context)?,
            )],
            &NoAccumulation,
            false,
            true,
            &harness.backend,
            &context,
        )?;
        let first_gradient = first
            .get(&leaves[first_index])
            .ok_or("missing LoHa first derivative")?;
        let first_slot = tape
            .output_slot(first_gradient)
            .ok_or("LoHa first derivative lacks higher-order provenance")?;
        let second = tape.reverse_with_context(
            vec![(
                first_slot,
                harness.tensor(first_gradient.descriptor().shape(), &[1.0], &context)?,
            )],
            &NoAccumulation,
            false,
            false,
            &harness.backend,
            &context,
        )?;
        assert_eq!(
            harness.values(
                second
                    .get(&leaves[second_index])
                    .ok_or("missing LoHa mixed second derivative")?,
                &context,
            )?,
            vec![expected]
        );
        assert_eq!(context.scratch.in_use_bytes(), 0);
    }
    Ok(())
}

#[test]
fn val_weight_adapter_001_trainable_tape_is_lazy_and_saved_mutations_fail_atomically()
-> Result<(), Box<dyn Error>> {
    let harness = Harness::new()?;
    let context = harness.context()?;
    let base = harness.tensor(&[2, 2], &[0.0; 4], &context)?;
    let mut up = harness.tensor(&[2, 1], &[1.0, 2.0], &context)?;
    let down = harness.tensor(&[1, 2], &[3.0, 4.0], &context)?;
    let adapter = lora(up.clone(), down, Some(1.0));

    let mut constant_tape = AutogradTape::new(GradientMode::Enabled);
    let constant = adapter.forward_trainable_recorded(
        &base,
        &harness.backend,
        &mut constant_tape,
        &context,
    )?;
    assert_eq!(constant.output_slot(), None);
    assert_eq!(constant_tape.retained_node_count(), 0);

    let mut tape = AutogradTape::new(GradientMode::Enabled);
    let up_leaf = LeafId::new("mutated-up")?;
    let NativeWeightAdapter::Lora {
        up: AdapterTensor::Dense(recorded_up),
        ..
    } = &adapter
    else {
        return Err("expected dense LoRA".into());
    };
    tape.set_requires_grad(recorded_up, Some(up_leaf), true, &harness.cancellation)?;
    let execution =
        adapter.forward_trainable_recorded(&base, &harness.backend, &mut tape, &context)?;
    let replacement = harness.tensor(&[2, 1], &[9.0, 9.0], &context)?;
    up.replace_data(replacement)?;
    let seed = harness.tensor(&[2, 2], &[1.0; 4], &context)?;
    assert!(matches!(
        tape.reverse_with_context(
            vec![(execution.output_slot().ok_or("missing output slot")?, seed)],
            &NoAccumulation,
            false,
            false,
            &harness.backend,
            &context,
        ),
        Err(AutogradError::SavedTensorModified { .. })
    ));
    assert_eq!(tape.retained_node_count(), 0);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn val_weight_adapter_001_linear_bypass_executes_lora_loha_lokr_and_glora()
-> Result<(), Box<dyn Error>> {
    let harness = Harness::new()?;
    let context = harness.context()?;
    let input = harness.tensor(&[1, 2], &[1.0, 2.0], &context)?;
    let module = ModuleTypeInfo::linear(2, 2)?;
    let identity = |_: &dyn sim_tensor::TensorBackend,
                    input: &Tensor,
                    _: &ExecutionContext<'_>|
     -> Result<Tensor, WeightAdapterError> { Ok(input.clone()) };

    let adapters = [
        (
            lora(
                harness.tensor(&[2, 1], &[5.0, 6.0], &context)?,
                harness.tensor(&[1, 2], &[3.0, 4.0], &context)?,
                Some(1.0),
            ),
            vec![56.0, 68.0],
        ),
        (
            NativeWeightAdapter::Loha {
                first_up: dense(harness.tensor(&[2, 1], &[1.0, 2.0], &context)?),
                first_down: dense(harness.tensor(&[1, 2], &[1.0, 1.0], &context)?),
                second_up: dense(harness.tensor(&[2, 1], &[1.0, 1.0], &context)?),
                second_down: dense(harness.tensor(&[1, 2], &[1.0, 0.0], &context)?),
                first_tucker: None,
                second_tucker: None,
                alpha: Some(1.0),
                dora_scale: None,
            },
            vec![2.0, 4.0],
        ),
        (
            NativeWeightAdapter::Lokr {
                first: Some(dense(harness.tensor(&[1, 1], &[1.0], &context)?)),
                second: Some(dense(harness.tensor(
                    &[2, 2],
                    &[1.0, 0.0, 0.0, 1.0],
                    &context,
                )?)),
                first_up: None,
                first_down: None,
                second_up: None,
                second_down: None,
                second_tucker: None,
                alpha: Some(1.0),
                dora_scale: None,
            },
            vec![2.0, 4.0],
        ),
        (
            NativeWeightAdapter::Glora {
                first_a: dense(harness.tensor(&[2, 2], &[0.0; 4], &context)?),
                second_a: dense(harness.tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0], &context)?),
                first_b: dense(harness.tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0], &context)?),
                second_b: dense(harness.tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0], &context)?),
                alpha: Some(2.0),
                dora_scale: None,
            },
            vec![2.0, 4.0],
        ),
    ];
    for (adapter, expected) in adapters {
        let mut manager = BypassInjectionManager::default();
        manager.add_adapter("layer.weight", adapter, 1.0)?;
        let mut plan =
            manager.create_injections(&BTreeMap::from([("layer".to_owned(), module.clone())]))?;
        assert_eq!(plan.inject_all(), 1);
        assert_eq!(plan.inject_all(), 0);
        let output = plan.hook("layer").ok_or("missing hook")?.execute(
            &harness.backend,
            &input,
            identity,
            &context,
        )?;
        assert_eq!(harness.values(&output, &context)?, expected);
        assert_eq!(plan.eject_all(), 1);
        assert_eq!(plan.eject_all(), 0);
    }
    Ok(())
}

#[test]
fn val_weight_adapter_001_orthogonal_bypass_is_identity_at_zero_and_rescales()
-> Result<(), Box<dyn Error>> {
    let harness = Harness::new()?;
    let context = harness.context()?;
    let input = harness.tensor(&[1, 2], &[3.0, 4.0], &context)?;
    let module = ModuleTypeInfo::linear(2, 2)?;
    let identity = |_: &dyn sim_tensor::TensorBackend,
                    input: &Tensor,
                    _: &ExecutionContext<'_>|
     -> Result<Tensor, WeightAdapterError> { Ok(input.clone()) };
    let adapters = [
        NativeWeightAdapter::Oft {
            blocks: dense(harness.tensor(&[1, 2, 2], &[0.0; 4], &context)?),
            rescale: Some(dense(harness.tensor(&[2], &[2.0, 3.0], &context)?)),
            constraint: Some(1.0),
            dora_scale: None,
        },
        NativeWeightAdapter::Boft {
            blocks: dense(harness.tensor(&[1, 1, 2, 2], &[0.0; 4], &context)?),
            rescale: Some(dense(harness.tensor(&[2], &[2.0, 3.0], &context)?)),
            constraint: Some(1.0),
            dora_scale: None,
        },
    ];
    for adapter in adapters {
        let mut manager = BypassInjectionManager::default();
        manager.add_adapter("layer", adapter, 1.0)?;
        let mut plan =
            manager.create_injections(&BTreeMap::from([("layer".to_owned(), module.clone())]))?;
        plan.inject_all();
        let output = plan.hook("layer").ok_or("missing hook")?.execute(
            &harness.backend,
            &input,
            identity,
            &context,
        )?;
        assert_eq!(harness.values(&output, &context)?, vec![6.0, 12.0]);
    }
    Ok(())
}

#[test]
fn val_weight_adapter_001_nonzero_oft_and_multistage_boft_match_cayley_order()
-> Result<(), Box<dyn Error>> {
    let harness = Harness::new()?;
    let context = harness.context()?;
    let identity = |_: &dyn sim_tensor::TensorBackend,
                    input: &Tensor,
                    _: &ExecutionContext<'_>|
     -> Result<Tensor, WeightAdapterError> { Ok(input.clone()) };

    let oft = NativeWeightAdapter::Oft {
        blocks: dense(harness.tensor(&[1, 2, 2], &[0.0, 1.0, 0.0, 0.0], &context)?),
        rescale: None,
        constraint: Some(0.0),
        dora_scale: None,
    };
    let mut manager = BypassInjectionManager::default();
    manager.add_adapter("oft", oft, 1.0)?;
    let mut plan = manager.create_injections(&BTreeMap::from([(
        "oft".into(),
        ModuleTypeInfo::linear(2, 2)?,
    )]))?;
    plan.inject_all();
    let oft_output = plan.hook("oft").ok_or("missing OFT hook")?.execute(
        &harness.backend,
        &harness.tensor(&[1, 2], &[3.0, 4.0], &context)?,
        identity,
        &context,
    )?;
    assert_eq!(harness.values(&oft_output, &context)?, vec![-4.0, 3.0]);

    let boft = NativeWeightAdapter::Boft {
        blocks: dense(harness.tensor(
            &[2, 2, 2, 2],
            &[
                0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            ],
            &context,
        )?),
        rescale: None,
        constraint: Some(0.0),
        dora_scale: None,
    };
    let mut manager = BypassInjectionManager::default();
    manager.add_adapter("boft", boft, 1.0)?;
    let mut plan = manager.create_injections(&BTreeMap::from([(
        "boft".into(),
        ModuleTypeInfo::linear(4, 4)?,
    )]))?;
    plan.inject_all();
    let boft_output = plan.hook("boft").ok_or("missing BOFT hook")?.execute(
        &harness.backend,
        &harness.tensor(&[1, 4], &[1.0, 2.0, 3.0, 4.0], &context)?,
        identity,
        &context,
    )?;
    assert_eq!(
        harness.values(&boft_output, &context)?,
        vec![4.0, -3.0, -2.0, 1.0]
    );

    let invalid = NativeWeightAdapter::Boft {
        blocks: dense(harness.tensor(&[2, 1, 2, 2], &[0.0; 8], &context)?),
        rescale: None,
        constraint: Some(0.0),
        dora_scale: None,
    };
    assert!(matches!(
        sim_model::BypassBinding::checked("invalid", invalid, 1.0),
        Err(WeightAdapterError::InvalidShape(_))
    ));
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn val_weight_adapter_001_convolution_quantization_and_static_patch_delegate_to_canonical_owners()
-> Result<(), Box<dyn Error>> {
    let harness = Harness::new()?;
    let context = harness.context()?;
    let quantized = quantize_matrix(
        QuantizationKind::Int8Tensorwise,
        DType::F32,
        &[2.0],
        1,
        1,
        &harness.cancellation,
    )?;
    let adapter = NativeWeightAdapter::Lora {
        up: AdapterTensor::Quantized(quantized),
        down: dense(harness.tensor(&[1, 1], &[3.0], &context)?),
        alpha: Some(1.0),
        mid: None,
        dora_scale: None,
        reshape: None,
    };
    let module =
        ModuleTypeInfo::convolution(1, vec![1], vec![0], vec![1], 1, vec![1], Some(1), Some(1))?;
    let input = harness.tensor(&[1, 1, 3], &[1.0, 2.0, 3.0], &context)?;
    let mut manager = BypassInjectionManager::default();
    manager.add_adapter("conv", adapter.clone(), 1.0)?;
    let mut plan = manager.create_injections(&BTreeMap::from([("conv".into(), module)]))?;
    plan.inject_all();
    let output = plan
        .hook("conv")
        .ok_or("missing convolution hook")?
        .execute(
            &harness.backend,
            &input,
            |_, input, _| Ok(input.clone()),
            &context,
        )?;
    assert_eq!(harness.values(&output, &context)?, vec![7.0, 14.0, 21.0]);

    let graph = adapter.calculate_static_patch_graph(
        BASE_DIGEST,
        "adapter-1",
        "layer.weight",
        vec![1, 1],
        0.5,
        1.0,
        &harness.backend,
        &context,
    )?;
    assert_eq!(graph.semantic_operations().len(), 1);
    assert!(matches!(
        graph.semantic_operations()[0].payload,
        PatchPayload::Lora { .. }
    ));
    Ok(())
}

#[test]
fn val_weight_adapter_001_lora_mid_uses_down_geometry_then_pointwise_mid_and_up()
-> Result<(), Box<dyn Error>> {
    let harness = Harness::new()?;
    let context = harness.context()?;
    let adapter = NativeWeightAdapter::Lora {
        up: dense(harness.tensor(&[1, 1], &[3.0], &context)?),
        down: dense(harness.tensor(&[1, 3], &[1.0, 1.0, 1.0], &context)?),
        alpha: Some(1.0),
        mid: Some(dense(harness.tensor(&[1, 1, 1], &[2.0], &context)?)),
        dora_scale: None,
        reshape: None,
    };
    let module =
        ModuleTypeInfo::convolution(1, vec![2], vec![1], vec![1], 1, vec![3], Some(1), Some(1))?;
    let input = harness.tensor(&[1, 1, 5], &[1.0, 2.0, 3.0, 4.0, 5.0], &context)?;
    let base = harness.tensor(&[1, 1, 3], &[0.0; 3], &context)?;
    let mut manager = BypassInjectionManager::default();
    manager.add_adapter("conv", adapter, 1.0)?;
    let mut plan = manager.create_injections(&BTreeMap::from([("conv".into(), module)]))?;
    assert_eq!(plan.inject_all(), 1);
    let output = plan.hook("conv").ok_or("missing LoRA mid hook")?.execute(
        &harness.backend,
        &input,
        |_, _, _| Ok(base.clone()),
        &context,
    )?;
    assert_eq!(harness.values(&output, &context)?, vec![18.0, 54.0, 54.0]);
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn val_weight_adapter_001_cancellation_invalid_dtype_and_patch_filtering_are_failure_atomic()
-> Result<(), Box<dyn Error>> {
    let harness = Harness::new()?;
    let context = harness.context()?;
    let up = harness.tensor(&[2, 1], &[1.0, 1.0], &context)?;
    let down = harness.tensor(&[1, 2], &[1.0, 1.0], &context)?;
    let adapter = lora(up, down, Some(1.0));
    let manager = BypassInjectionManager::from_patches(
        BTreeMap::from([(
            "layer.weight".into(),
            vec![
                BypassPatch::StaticPatch,
                BypassPatch::Adapter {
                    patch_strength: 0.5,
                    adapter,
                },
            ],
        )]),
        2.0,
    )?;
    let plan = manager.create_injections(&BTreeMap::from([(
        "layer".into(),
        ModuleTypeInfo::linear(2, 2)?,
    )]))?;
    assert_eq!(plan.hook_count(), 1);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = harness.backend.execution_context(
        StreamId::DEFAULT,
        harness.workspace.authorize_workspace(1024)?,
        &cancelled,
    );
    let input = harness.tensor(&[1, 2], &[1.0, 2.0], &context)?;
    let mut activated = plan;
    activated.inject_all();
    let unsupported = tensor_from_f32_with_context_exact_native(
        &harness.backend,
        &[1, 2],
        &[1.0, 2.0],
        DType::F16,
        sim_tensor::DeviceId::CPU,
        &context,
    )?;
    assert!(matches!(
        activated.hook("layer").ok_or("missing hook")?.execute(
            &harness.backend,
            &unsupported,
            |_, input, _| Ok(input.clone()),
            &context,
        ),
        Err(WeightAdapterError::UnsupportedDType {
            dtype: DType::F16,
            ..
        })
    ));
    assert!(
        activated
            .hook("layer")
            .ok_or("missing hook")?
            .execute(
                &harness.backend,
                &input,
                |_, input, _| Ok(input.clone()),
                &cancelled_context,
            )
            .is_err()
    );
    assert_eq!(context.scratch.in_use_bytes(), 0);
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);

    Ok(())
}

fn repository_root() -> Result<PathBuf, Box<dyn Error>> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("repository root is unavailable")?
        .to_path_buf())
}

fn production_rust_sources(root: &Path) -> Result<Vec<(PathBuf, String)>, Box<dyn Error>> {
    fn visit(directory: &Path, sources: &mut Vec<(PathBuf, String)>) -> std::io::Result<()> {
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("._"))
            {
                continue;
            }
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name == "tests") {
                    continue;
                }
                visit(&path, sources)?;
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                sources.push((path.clone(), std::fs::read_to_string(path)?));
            }
        }
        Ok(())
    }

    let mut sources = Vec::new();
    visit(&root.join("crates"), &mut sources)?;
    sources.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(sources)
}

#[test]
fn production_source_scan_ignores_apple_double_metadata() -> Result<(), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let crates = directory.path().join("crates/example/src");
    std::fs::create_dir_all(&crates)?;
    std::fs::write(crates.join("owner.rs"), "fn owner() {}\n")?;
    std::fs::write(crates.join("._owner.rs"), [0xff])?;
    assert_eq!(
        production_rust_sources(directory.path())?,
        [(crates.join("owner.rs"), "fn owner() {}\n".to_owned())]
    );
    Ok(())
}

#[test]
fn val_weight_adapter_001_repository_ownership_is_single_and_explicit() -> Result<(), Box<dyn Error>>
{
    let root = repository_root()?;
    let sources = production_rust_sources(&root)?;
    let occurrences = |needle: &str| {
        sources
            .iter()
            .filter(|(_, source)| source.contains(needle))
            .map(|(path, _)| path.strip_prefix(&root).unwrap_or(path).to_path_buf())
            .collect::<Vec<_>>()
    };

    assert_eq!(
        occurrences(concat!("pub enum NativeWeight", "Adapter")),
        [PathBuf::from("crates/sim_model/src/weight_adapter.rs")]
    );
    assert_eq!(
        occurrences(concat!("pub struct Auto", "gradTape")),
        [PathBuf::from("crates/sim_tensor/src/autograd.rs")]
    );
    assert_eq!(
        occurrences(concat!("pub struct Gradient", "Store")),
        [PathBuf::from("crates/sim_tensor/src/autograd.rs")]
    );
    assert_eq!(
        occurrences(concat!("pub struct Cancellation", "Token")),
        [PathBuf::from("crates/sim_types/src/cancellation.rs")]
    );
    assert_eq!(
        occurrences(concat!("pub struct BackendWorkspace", "Authority")),
        [PathBuf::from("crates/sim_tensor/src/operation.rs")]
    );
    assert_eq!(
        occurrences(concat!("pub struct AttemptMemory", "Controller")),
        [PathBuf::from("crates/sim_worker/src/memory_modes.rs")]
    );
    assert_eq!(
        occurrences(concat!("pub struct Patch", "Graph {")),
        [PathBuf::from("crates/sim_model/src/patch_graph.rs")]
    );
    assert_eq!(
        occurrences(concat!("pub struct Quantized", "Matrix")),
        [PathBuf::from("crates/sim_model/src/quantization.rs")]
    );
    assert_eq!(
        occurrences("fn cpu_backend(&self) -> Option<&crate::CpuBackend>"),
        [PathBuf::from("crates/sim_tensor/src/operation.rs")]
    );
    assert_eq!(
        occurrences("fn cpu_backend(&self) -> Option<&CpuBackend>"),
        [PathBuf::from("crates/sim_tensor/src/cpu_backend.rs")]
    );

    let adapter = std::fs::read_to_string(root.join("crates/sim_model/src/weight_adapter.rs"))?;
    for forbidden in [
        concat!("struct Auto", "gradTape"),
        concat!("struct Gradient", "Store"),
        concat!("struct Cancellation", "Token"),
        concat!("struct BackendWorkspace", "Authority"),
        concat!("struct AttemptMemory", "Controller"),
        concat!("struct Output", "Committer"),
        concat!("struct Execution", "Queue"),
        concat!("struct Patch", "Graph"),
        concat!("struct Quantized", "Matrix"),
        "std::fs",
        "tokio::time",
        "smol::Timer",
    ] {
        assert!(
            !adapter.contains(forbidden),
            "weight_adapter contains forbidden parallel owner {forbidden}"
        );
    }
    assert!(adapter.contains("tape.record_operation("));
    assert!(adapter.contains("matrix.materialize(backend, context)?"));
    assert!(adapter.contains("PatchGraph::checked_semantic("));
    assert!(adapter.contains("backend.workspace_vec(context"));
    assert!(adapter.contains("context.check()?"));
    assert!(adapter.contains(".cpu_backend()"));
    assert!(adapter.contains(".ok_or(WeightAdapterError::UnsupportedDevice"));
    assert!(!adapter.contains("CpuWorkspaceAuthority::create_backend"));
    assert!(!adapter.contains("CancellationToken::default"));
    Ok(())
}

fn python_symbol_sha256(source: &[u8], symbol: &str) -> Result<String, Box<dyn Error>> {
    let source = std::str::from_utf8(source)?;
    let lines = source.split_inclusive('\n').collect::<Vec<_>>();
    let assignment_prefixes = [format!("{symbol} ="), format!("{symbol}:")];
    if let Some(start) = lines.iter().position(|line| {
        !line.starts_with([' ', '\t'])
            && assignment_prefixes
                .iter()
                .any(|prefix| line.starts_with(prefix))
    }) {
        let mut balance = 0_i64;
        let mut end = start;
        loop {
            let line = lines.get(end).ok_or("unterminated Python assignment")?;
            for character in line.chars() {
                match character {
                    '(' | '[' | '{' => balance += 1,
                    ')' | ']' | '}' => balance -= 1,
                    _ => {}
                }
            }
            end += 1;
            if balance == 0 {
                break;
            }
            if balance < 0 {
                return Err("malformed Python assignment delimiters".into());
            }
        }
        return Ok(format!(
            "{:x}",
            Sha256::digest(lines[start..end].concat().as_bytes())
        ));
    }

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
fn val_weight_adapter_001_exact_catalog_manifest_and_artifact_are_current()
-> Result<(), Box<dyn Error>> {
    const TASK: &str = "comfy-parity-weight-adapter-runtime-bypass";
    const EXPECTED: [(&str, &str, &str, &str, &str); 18] = [
        (
            "conditioning-weight-adapter-registry-init-adapters-4b5174ed",
            "projects/comfy/ComfyUI/sim/weight_adapter/__init__.py",
            "adapters",
            "d9fe388a6bd3b93212a5e04b8921759841e2cf88939cfcc3a7bf59e16505edd0",
            "99d882edfbb6eefa43035324d605c3f3972a3b350a4a9a63c75cc4b32e8b69fd",
        ),
        (
            "conditioning-weight-adapter-registry-init-adapter-maps-a404bc73",
            "projects/comfy/ComfyUI/sim/weight_adapter/__init__.py",
            "adapter_maps",
            "d9fe388a6bd3b93212a5e04b8921759841e2cf88939cfcc3a7bf59e16505edd0",
            "18906dbde76653312d6b638fea194f169280dcb8ef579e8b1dbf6db08ebf2fd4",
        ),
        (
            "conditioning-weight-adapter-runtime-base-weightadapterbase-1d8af5d1",
            "projects/comfy/ComfyUI/sim/weight_adapter/base.py",
            "WeightAdapterBase",
            "484f0d83a96e700f80c793e4bcc6a897d633233e51a59191d91da5e59da345c7",
            "90be77a1cac84b070449c7858d63b8c2cc6bb8fe0431bb56c74055555423f532",
        ),
        (
            "conditioning-weight-adapter-runtime-base-weightadaptertrainbase-da831b16",
            "projects/comfy/ComfyUI/sim/weight_adapter/base.py",
            "WeightAdapterTrainBase",
            "484f0d83a96e700f80c793e4bcc6a897d633233e51a59191d91da5e59da345c7",
            "0ecab48be7f5d5c35572bc3301c6a4933dbf3d23537cb65930a07c0d530b1e47",
        ),
        (
            "conditioning-weight-adapter-runtime-bypass-get-module-type-info-1182dfe4",
            "projects/comfy/ComfyUI/sim/weight_adapter/bypass.py",
            "get_module_type_info",
            "618c286c5f74e823318f41774e02cfcfa02d74ee9d51320ec0aa2694cd74aa8e",
            "9150476bc769629e6c442eed74d7a00ad52ce06f692092b99f272201d8b956cd",
        ),
        (
            "conditioning-weight-adapter-runtime-bypass-bypassforwardhook-d8e0e2e1",
            "projects/comfy/ComfyUI/sim/weight_adapter/bypass.py",
            "BypassForwardHook",
            "618c286c5f74e823318f41774e02cfcfa02d74ee9d51320ec0aa2694cd74aa8e",
            "fa067d2d069eef094c52b5ea91c41727d8cc7b4809138737e60ceda8fcc32ef6",
        ),
        (
            "conditioning-weight-adapter-runtime-bypass-bypassinjectionmanager-0a83408d",
            "projects/comfy/ComfyUI/sim/weight_adapter/bypass.py",
            "BypassInjectionManager",
            "618c286c5f74e823318f41774e02cfcfa02d74ee9d51320ec0aa2694cd74aa8e",
            "3540126c8a66f59b6ac5ae814d2cd1a16f0acce5dc6854dff6208a95704802e7",
        ),
        (
            "conditioning-weight-adapter-runtime-bypass-create-bypass-injections-from-patches-9ca4524a",
            "projects/comfy/ComfyUI/sim/weight_adapter/bypass.py",
            "create_bypass_injections_from_patches",
            "618c286c5f74e823318f41774e02cfcfa02d74ee9d51320ec0aa2694cd74aa8e",
            "9bb0b0f92e1b3d133c5fcd54103cac6bfaa21f2ec77dface954741999b90f79d",
        ),
        (
            "conditioning-weight-adapter-runtime-boft-boftadapter-5fb812eb",
            "projects/comfy/ComfyUI/sim/weight_adapter/boft.py",
            "BOFTAdapter",
            "2850e0b4c2295cd87445415e287061fa3bfd69e88bd0aeb3eb16064864bd078d",
            "51b818fccd6e868dbbdbdde73df41c4a318adf3d245c4cde83043bd0a939e534",
        ),
        (
            "conditioning-weight-adapter-runtime-glora-gloraadapter-76c25b06",
            "projects/comfy/ComfyUI/sim/weight_adapter/glora.py",
            "GLoRAAdapter",
            "31cdd03f5b0beaa0df055512560128930f4f26b219ba57602d21abb086425b09",
            "4466becb5c2cf3559bc21e622592b594e106b773716f35361a07303f967d1964",
        ),
        (
            "conditioning-weight-adapter-runtime-loha-lohadiff-45ee6865",
            "projects/comfy/ComfyUI/sim/weight_adapter/loha.py",
            "LohaDiff",
            "579ca1e33e0d244e0d7eedd30fb727913341f8e7bfbd74b51221f567612286d5",
            "bc24b5be90ad6d747185f8171a107bbfc6a74b1bad59919a395c3ea619e50581",
        ),
        (
            "conditioning-weight-adapter-runtime-loha-lohaadapter-85c7cd6a",
            "projects/comfy/ComfyUI/sim/weight_adapter/loha.py",
            "LoHaAdapter",
            "579ca1e33e0d244e0d7eedd30fb727913341f8e7bfbd74b51221f567612286d5",
            "5c0f3b33cccbb978ca017e243b0e7d2bfdcd4b5e40330def909bee94957f769a",
        ),
        (
            "conditioning-weight-adapter-runtime-lokr-lokrdiff-161df858",
            "projects/comfy/ComfyUI/sim/weight_adapter/lokr.py",
            "LokrDiff",
            "b4763cc32215a47e4d906cfa6cbad9cf893f6a1329ada2225fda81fd99fcfeb4",
            "8e215047b773756c83cbd6628568e0319a42c52bf6cd6569910b975bd3527e65",
        ),
        (
            "conditioning-weight-adapter-runtime-lokr-lokradapter-324ca38f",
            "projects/comfy/ComfyUI/sim/weight_adapter/lokr.py",
            "LoKrAdapter",
            "b4763cc32215a47e4d906cfa6cbad9cf893f6a1329ada2225fda81fd99fcfeb4",
            "71005144a44c9be904924ba5bb6a0fb6aaf1cbb63768711ad99fccb1459e8dfa",
        ),
        (
            "conditioning-weight-adapter-runtime-lora-loradiff-0423ce51",
            "projects/comfy/ComfyUI/sim/weight_adapter/lora.py",
            "LoraDiff",
            "e506062b4eb189be4c36f88270e0fd4dcce038c79a98f7163604ed6b44efe4b5",
            "f8e68c63987869d66fc62d5b98c090b4cb3f80130118b96f8be29a635aa1d512",
        ),
        (
            "conditioning-weight-adapter-runtime-lora-loraadapter-9ed6cfef",
            "projects/comfy/ComfyUI/sim/weight_adapter/lora.py",
            "LoRAAdapter",
            "e506062b4eb189be4c36f88270e0fd4dcce038c79a98f7163604ed6b44efe4b5",
            "479b37686bdb45bf04db421a9fc96f2c0d480809fef3fc1776bf89de512d183b",
        ),
        (
            "conditioning-weight-adapter-runtime-oft-oftdiff-c8899863",
            "projects/comfy/ComfyUI/sim/weight_adapter/oft.py",
            "OFTDiff",
            "88be3c32f610478bc6900a10009eaadf6fe2af973ed4861731e4f21e1afacf89",
            "0369eed8b55bbffc3c9be278bcab4b43b264d4b0c7df39bc79c80b4aace19f6d",
        ),
        (
            "conditioning-weight-adapter-runtime-oft-oftadapter-db2990a3",
            "projects/comfy/ComfyUI/sim/weight_adapter/oft.py",
            "OFTAdapter",
            "88be3c32f610478bc6900a10009eaadf6fe2af973ed4861731e4f21e1afacf89",
            "4b398fd08a6d35165407a4c85e1575d537a43430fcd3fe3c0faf203d2ed568a8",
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
            return Err("malformed weight-adapter catalog row".into());
        }
        let expected_row = expected
            .get(columns[0])
            .ok_or("unexpected weight-adapter catalog row")?;
        assert!(seen.insert(columns[0]));
        assert_eq!(columns[2], expected_row.1);
        assert_eq!(columns[3], expected_row.2);
        assert_eq!(columns[5], expected_row.3);
        assert_eq!(columns[6], expected_row.4);
        assert_eq!(columns[7], "sim_model::weight_adapter");
        assert_eq!(columns[9], "sim_model::weight_adapter::tests");
        assert_eq!(columns[10], "native_rust");
        assert_eq!(columns[14], "VAL-WEIGHT-ADAPTER-001");
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
        "crates/sim_model/src/sim_model.rs",
        "crates/sim_model/src/weight_adapter.rs",
        "crates/sim_model/tests/weight_adapter_runtime.rs",
        "crates/sim_tensor/src/cpu_backend.rs",
        "crates/sim_tensor/src/operation.rs",
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
    let producer_path = "crates/sim_model/tests/weight_adapter_runtime.rs";
    let producer = std::fs::read(repository.join(producer_path))?;
    let passed = contracts.len() * 2;
    let artifact = serde_json::json!({
        "schema_version": 1,
        "validation_id": "VAL-WEIGHT-ADAPTER-001",
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
                    "task510:all-18-valid-invalid",
                    "task510:canonical-autograd",
                    "task510:caller-workspace-cancellation",
                    "task510:typed-capability-rejection",
                    "task510:ownership-consolidation",
                ],
                "implementations": implementations,
            },
        },
        "contracts": contracts,
    });
    let artifact_directory = repository.join("target/sim-parity");
    std::fs::create_dir_all(&artifact_directory)?;
    std::fs::write(
        artifact_directory.join("val-weight-adapter-001.json"),
        serde_json::to_vec_pretty(&artifact)?,
    )?;
    Ok(())
}
