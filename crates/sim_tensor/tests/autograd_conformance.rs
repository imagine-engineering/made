use serde::Deserialize;
use sim_tensor::{
    AutogradError, AutogradInput, AutogradTape, BackwardRule, CancellationToken, CpuBackend,
    CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext, GradientMode, GradientReducer,
    HigherOrderContext, LeafId, SavedTensor, StorageId, StreamId, Tensor, TensorDescriptor,
    TensorError, TensorId,
    autograd::breadth::{
        AUTOGRAD_CONSTRUCTS, AddAuxLossFunction, AutogradBreadthError, CUSTOM_FUNCTIONS,
        CheckpointCallable, CheckpointFunction, FunctionContext, GradScalerConfig,
        GradScalerOptimizerDecision, GradientStore, HadaWeightFunction, HadaWeightTuckerFunction,
        HigherOrderPolicy, NativeAdam, NativeAdamW, NativeGradScaler, NativeRmsprop, NativeSgd,
        OffloadCheckpointFunction, VectorQuantizeFunction,
    },
};
use std::{
    collections::{BTreeSet, HashMap},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Deserialize)]
struct BreadthFixture {
    schema_version: u32,
    owner_task_id: String,
    catalog_cases: Vec<AutogradFixtureCase>,
    custom_functions: serde_json::Value,
}

#[derive(Deserialize)]
struct AutogradFixtureCase {
    id: String,
    symbol: String,
    execution_case: String,
    source_observations: Vec<serde_json::Value>,
}

fn fixture() -> Result<(CpuBackend, CpuWorkspaceAuthority), TensorError> {
    CpuWorkspaceAuthority::create_backend(32 * 1024 * 1024)
}

fn context<'a>(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, TensorError> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(8 * 1024 * 1024)?,
        cancellation,
    ))
}

fn tensor(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    Ok(backend
        .upload_f32(
            descriptor,
            values,
            &context(backend, authority, cancellation)?,
        )?
        .0)
}

fn values(tensor: &Tensor) -> Result<Vec<f32>, TensorError> {
    tensor
        .contiguous_bytes()?
        .chunks_exact(4)
        .map(|bytes| {
            let array = <[u8; 4]>::try_from(bytes).map_err(|_| TensorError::StorageLength {
                expected: 4,
                actual: u64::try_from(bytes.len()).unwrap_or(0),
            })?;
            Ok(f32::from_ne_bytes(array))
        })
        .collect()
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-4,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn strict_fixture_registry_has_exact_unique_coverage_and_canonical_owners() {
    let catalog = include_str!("../../../.agents/specs/comfy-parity/catalogs/backend-autograd.csv");
    let fixture: BreadthFixture = serde_json::from_str(include_str!(
        "../../sim_test_support/fixtures/autograd/breadth-v1.json"
    ))
    .expect("autograd breadth fixture must parse");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(
        fixture.owner_task_id,
        "comfy-parity-native-autograd-breadth"
    );
    assert_eq!(catalog.lines().skip(1).count(), 36);
    assert_eq!(AUTOGRAD_CONSTRUCTS.len(), 36);
    assert_eq!(fixture.catalog_cases.len(), 36);
    let ids = AUTOGRAD_CONSTRUCTS
        .iter()
        .map(|contract| contract.id)
        .collect::<BTreeSet<_>>();
    let symbols = AUTOGRAD_CONSTRUCTS
        .iter()
        .map(|contract| contract.symbol)
        .collect::<BTreeSet<_>>();
    assert_eq!(ids.len(), 36);
    assert_eq!(symbols.len(), 36);
    let fixture_ids = fixture
        .catalog_cases
        .iter()
        .map(|case| case.id.as_str())
        .collect::<BTreeSet<_>>();
    let fixture_symbols = fixture
        .catalog_cases
        .iter()
        .map(|case| case.symbol.as_str())
        .collect::<BTreeSet<_>>();
    let execution_cases = fixture
        .catalog_cases
        .iter()
        .map(|case| case.execution_case.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(fixture_ids.len(), 36);
    assert_eq!(fixture_symbols.len(), 36);
    assert_eq!(execution_cases.len(), 36);
    assert!(
        fixture
            .catalog_cases
            .iter()
            .all(|case| !case.source_observations.is_empty())
    );
    assert_eq!(ids, fixture_ids.iter().copied().collect::<BTreeSet<_>>());
    assert_eq!(
        symbols,
        fixture_symbols.iter().copied().collect::<BTreeSet<_>>()
    );
    assert!(AUTOGRAD_CONSTRUCTS.iter().all(
        |contract| contract.id.starts_with("COMFY-AUTOGRAD-") && !contract.construct.is_empty()
    ));
    for contract in AUTOGRAD_CONSTRUCTS {
        let prefix = format!("{},{}", contract.id, contract.construct);
        assert!(catalog.lines().skip(1).any(|row| row.starts_with(&prefix)));
        let symbol = format!(",{},", contract.symbol);
        assert!(
            catalog
                .lines()
                .skip(1)
                .any(|row| row.starts_with(contract.id) && row.contains(&symbol))
        );
    }
    let custom_fixture = fixture
        .custom_functions
        .as_object()
        .expect("custom function fixture must be an object");
    assert_eq!(CUSTOM_FUNCTIONS.len(), 7);
    for contract in CUSTOM_FUNCTIONS {
        assert!(fixture_ids.contains(contract.id));
        assert!(fixture_symbols.contains(contract.symbol));
        let catalog_case = fixture
            .catalog_cases
            .iter()
            .find(|case| case.id == contract.id);
        assert!(
            catalog_case.is_some(),
            "custom function catalog case must exist"
        );
        let Some(catalog_case) = catalog_case else {
            continue;
        };
        if contract.symbol == "QuantLinearFunc" {
            assert_eq!(
                contract.fixture,
                ".agents/specs/comfy-parity/fixtures/quant-linear-source-oracle.json#callable"
            );
            assert_eq!(catalog_case.execution_case, "quant_linear_model_adapter");
            assert_eq!(
                catalog_case.source_observations,
                vec![serde_json::json!({
                    "case": "delegated_fixture",
                    "expected": ".agents/specs/comfy-parity/fixtures/quant-linear-source-oracle.json",
                    "sha256": "74acf934871befe3a87a91de6aea430a7ea9a16a821441bd716768dfb1919d0c",
                })]
            );
        } else {
            let key = match contract.symbol {
                "CheckpointFunction" => "checkpoint_function",
                "HadaWeightTucker" => "hada_weight_tucker",
                "AddAuxLoss" => "add_aux_loss",
                "OffloadCheckpointFunction" => "offload_checkpoint",
                "HadaWeight" => "hada_weight",
                symbol => symbol,
            };
            assert!(
                custom_fixture.contains_key(key),
                "missing fixture for {key}"
            );
            let expected_policy = match contract.higher_order {
                HigherOrderPolicy::Analytical => "analytical",
                HigherOrderPolicy::FirstOrderOnly => "first_order_only",
                HigherOrderPolicy::OnceDifferentiable => "once_differentiable",
            };
            assert_eq!(
                custom_fixture[key]["higher_order"].as_str(),
                Some(expected_policy)
            );
        }
        assert!(!contract.fixture.is_empty());
        assert!(contract.forward_arity > 0 && contract.backward_outputs > 0);
    }
    let exact_contracts = CUSTOM_FUNCTIONS
        .iter()
        .map(|contract| {
            (
                contract.symbol,
                contract.forward_arity,
                contract.variadic_inputs,
                contract.forward_outputs,
                contract.backward_inputs,
                contract.backward_outputs,
                contract.higher_order,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        exact_contracts,
        vec![
            (
                "vector_quantize",
                2,
                false,
                2,
                2,
                2,
                HigherOrderPolicy::Analytical
            ),
            (
                "CheckpointFunction",
                3,
                true,
                1,
                1,
                3,
                HigherOrderPolicy::FirstOrderOnly
            ),
            (
                "QuantLinearFunc",
                6,
                false,
                1,
                1,
                6,
                HigherOrderPolicy::OnceDifferentiable
            ),
            (
                "HadaWeightTucker",
                7,
                false,
                1,
                1,
                7,
                HigherOrderPolicy::Analytical
            ),
            (
                "AddAuxLoss",
                2,
                false,
                1,
                1,
                2,
                HigherOrderPolicy::Analytical
            ),
            (
                "OffloadCheckpointFunction",
                2,
                false,
                1,
                1,
                2,
                HigherOrderPolicy::FirstOrderOnly
            ),
            (
                "HadaWeight",
                5,
                false,
                1,
                1,
                5,
                HigherOrderPolicy::Analytical
            ),
        ]
    );
    assert_eq!(
        custom_fixture["checkpoint_function"]["forward_fixture_arity"],
        3
    );
    assert_eq!(
        custom_fixture["checkpoint_function"]["backward_fixture_outputs"],
        3
    );
    assert_eq!(
        custom_fixture["checkpoint_function"]["variadic_tensor_inputs"],
        true
    );
    for contract in CUSTOM_FUNCTIONS {
        assert!(contract.validate_higher_order_request(false).is_ok());
        match contract.higher_order {
            HigherOrderPolicy::Analytical => {
                assert!(contract.validate_higher_order_request(true).is_ok());
            }
            HigherOrderPolicy::FirstOrderOnly | HigherOrderPolicy::OnceDifferentiable => {
                assert!(matches!(
                    contract.validate_higher_order_request(true),
                    Err(AutogradBreadthError::HigherOrderUnavailable { symbol, policy })
                        if symbol == contract.symbol && policy == contract.higher_order
                ));
            }
        }
    }
}

#[test]
fn function_context_checks_versions_non_differentiability_and_release()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let mut source = tensor(&backend, &authority, &[1], &[2.0], &cancellation)?;
    let mut function = FunctionContext::new(vec![true, false]);
    function.save_for_backward(&[&source])?;
    function.mark_non_differentiable(1)?;
    assert!(function.needs_input_grad(0));
    assert!(!function.needs_input_grad(1));
    assert!(function.is_non_differentiable(1));
    source
        .write()?
        .bytes_mut()?
        .copy_from_slice(&3.0_f32.to_ne_bytes());
    assert!(matches!(
        function.saved_tensors(),
        Err(AutogradBreadthError::Autograd(
            AutogradError::SavedTensorModified { .. }
        ))
    ));
    close(&values(&source)?, &[3.0]);
    function.release();
    assert!(matches!(
        function.saved_tensors(),
        Err(AutogradBreadthError::ReleasedContext)
    ));
    Ok(())
}

#[test]
fn mode_stack_is_nested_and_restored_on_error() {
    let cancellation = CancellationToken::default();
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    let result: Result<(), AutogradError> =
        tape.with_mode(GradientMode::NoGrad, &cancellation, |tape| {
            assert_eq!(tape.mode(), GradientMode::NoGrad);
            Err(AutogradError::InvalidGraph {
                reason: "fixture".to_owned(),
            })
        });
    assert!(result.is_err());
    assert_eq!(tape.mode(), GradientMode::Enabled);
}

#[test]
fn gradient_store_publishes_and_zeroes_transactionally() -> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let leaf = LeafId::new("weight")?;
    let gradient = tensor(&backend, &authority, &[2], &[3.0, -2.0], &cancellation)?;
    let mut store = GradientStore::default();
    store.publish(HashMap::from([(leaf.clone(), gradient)]), &cancellation)?;
    store.zero_grad(&backend, &execution, false)?;
    close(
        &values(store.gradient(&leaf).ok_or("missing gradient")?)?,
        &[0.0, 0.0],
    );

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        store.publish(HashMap::new(), &cancelled),
        Err(AutogradError::Cancelled)
    ));
    assert_eq!(store.len(), 1);
    store.zero_grad(&backend, &execution, true)?;
    assert!(store.is_empty());
    Ok(())
}

#[test]
fn canonical_scaler_and_optimizer_owners_preserve_state() -> Result<(), Box<dyn std::error::Error>>
{
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let gradient = tensor(&backend, &authority, &[1], &[0.5], &cancellation)?;

    let mut scaler = NativeGradScaler::new(GradScalerConfig {
        initial_scale: 8.0,
        growth_factor: 2.0,
        backoff_factor: 0.5,
        growth_interval: 1,
        enabled: true,
    })?;
    let loss = tensor(&backend, &authority, &[], &[2.0], &cancellation)?;
    close(
        &values(&scaler.scale_loss_exact_native(&loss, &cancellation)?)?,
        &[16.0],
    );
    let mut scaled_gradient = vec![tensor(&backend, &authority, &[1], &[4.0], &cancellation)?];
    assert!(!scaler.unscale_gradients_exact_native(&mut scaled_gradient, &cancellation)?);
    close(&values(&scaled_gradient[0])?, &[0.5]);
    assert_eq!(
        scaler.optimizer_step_decision_exact_native(&cancellation)?,
        GradScalerOptimizerDecision::Run
    );
    scaler.update_exact_native(&cancellation)?;
    assert_eq!(scaler.scale(), 16.0);

    let parameter = tensor(&backend, &authority, &[1], &[1.0], &cancellation)?;
    let mut sgd = NativeSgd::new_exact_native(1, 0.1, 0.0, 0.0, 0.0, false, false, &cancellation)?;
    let mut sgd_parameters = vec![parameter.clone()];
    sgd.step_with_context_exact_native(
        &backend,
        &mut sgd_parameters,
        std::slice::from_ref(&gradient),
        &execution,
    )?;
    close(&values(&sgd_parameters[0])?, &[0.95]);

    let mut adam = NativeAdam::new_with_context_exact_native(
        &backend,
        std::slice::from_ref(&parameter),
        0.1,
        &execution,
    )?;
    let mut adam_parameters = vec![parameter.clone()];
    adam.step_with_context_exact_native(
        &backend,
        &mut adam_parameters,
        std::slice::from_ref(&gradient),
        &execution,
    )?;
    assert_eq!(adam.steps(), [1]);

    let mut adamw = NativeAdamW::new_with_context_exact_native(
        &backend,
        std::slice::from_ref(&parameter),
        0.1,
        0.9,
        0.999,
        1.0e-8,
        0.01,
        false,
        false,
        &execution,
    )?;
    let mut adamw_parameters = vec![parameter.clone()];
    adamw.step_with_context_exact_native(
        &backend,
        &mut adamw_parameters,
        std::slice::from_ref(&gradient),
        &execution,
    )?;
    assert_eq!(adamw.steps(), [1]);

    let mut rmsprop = NativeRmsprop::new_with_context_exact_native(
        &backend,
        std::slice::from_ref(&parameter),
        0.01,
        0.99,
        1.0e-8,
        0.0,
        0.0,
        false,
        false,
        &execution,
    )?;
    let mut rmsprop_parameters = vec![parameter];
    rmsprop.step_with_context_exact_native(
        &backend,
        &mut rmsprop_parameters,
        std::slice::from_ref(&gradient),
        &execution,
    )?;
    assert_eq!(rmsprop.steps(), [1]);
    Ok(())
}

#[test]
fn vector_quantize_forward_and_vjp_match_source() -> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let input = tensor(
        &backend,
        &authority,
        &[3, 2],
        &[0.1, 0.2, 2.8, 3.2, 0.8, 1.1],
        &cancellation,
    )?;
    let codebook = tensor(
        &backend,
        &authority,
        &[3, 2],
        &[0.0, 0.0, 1.0, 1.0, 3.0, 3.0],
        &cancellation,
    )?;
    let (function, output, indices) =
        VectorQuantizeFunction::forward(&backend, &input, &codebook, [true, true], &execution)?;
    close(&values(&output)?, &[0.0, 0.0, 3.0, 3.0, 1.0, 1.0]);
    assert_eq!(indices.descriptor().dtype(), DType::I64);
    let grad = tensor(
        &backend,
        &authority,
        &[3, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    let [grad_input, grad_codebook] =
        function.backward(&backend, [Some(&grad), None], &execution)?;
    close(
        &values(&grad_input.ok_or("missing input gradient")?)?,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    close(
        &values(&grad_codebook.ok_or("missing codebook gradient")?)?,
        &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0],
    );
    Ok(())
}

#[test]
fn hada_weight_forward_and_vjp_are_analytical() -> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let w1u = tensor(&backend, &authority, &[2, 1], &[1.0, 2.0], &cancellation)?;
    let w1d = tensor(&backend, &authority, &[1, 2], &[3.0, 4.0], &cancellation)?;
    let w2u = tensor(&backend, &authority, &[2, 1], &[2.0, 1.0], &cancellation)?;
    let w2d = tensor(&backend, &authority, &[1, 2], &[1.0, 2.0], &cancellation)?;
    let scale = tensor(&backend, &authority, &[], &[0.5], &cancellation)?;
    let (function, output) = HadaWeightFunction::forward(
        &backend,
        [&w1u, &w1d, &w2u, &w2d],
        &scale,
        [true, true, true, true, false],
        &execution,
    )?;
    close(&values(&output)?, &[3.0, 8.0, 3.0, 8.0]);
    let grad = tensor(&backend, &authority, &[2, 2], &[1.0; 4], &cancellation)?;
    let gradients = function.backward(&backend, Some(&grad), &execution)?;
    assert!(gradients[4].is_none());
    close(
        &values(gradients[0].as_ref().ok_or("missing gradient")?)?,
        &[11.0, 5.5],
    );
    close(
        &values(gradients[1].as_ref().ok_or("missing gradient")?)?,
        &[2.0, 4.0],
    );
    close(
        &values(gradients[2].as_ref().ok_or("missing gradient")?)?,
        &[5.5, 11.0],
    );
    close(
        &values(gradients[3].as_ref().ok_or("missing gradient")?)?,
        &[6.0, 8.0],
    );

    let epsilon = 1.0e-3;
    let plus = hada_objective(&backend, &authority, [1.0 + epsilon, 2.0], &cancellation)?;
    let minus = hada_objective(&backend, &authority, [1.0 - epsilon, 2.0], &cancellation)?;
    assert!((((plus - minus) / (2.0 * epsilon)) - 11.0).abs() < 1.0e-2);
    Ok(())
}

fn hada_objective(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    first_up: [f32; 2],
    cancellation: &CancellationToken,
) -> Result<f32, Box<dyn std::error::Error>> {
    let execution = context(backend, authority, cancellation)?;
    let w1u = tensor(backend, authority, &[2, 1], &first_up, cancellation)?;
    let w1d = tensor(backend, authority, &[1, 2], &[3.0, 4.0], cancellation)?;
    let w2u = tensor(backend, authority, &[2, 1], &[2.0, 1.0], cancellation)?;
    let w2d = tensor(backend, authority, &[1, 2], &[1.0, 2.0], cancellation)?;
    let scale = tensor(backend, authority, &[], &[0.5], cancellation)?;
    let (_, output) = HadaWeightFunction::forward(
        backend,
        [&w1u, &w1d, &w2u, &w2d],
        &scale,
        [false; 5],
        &execution,
    )?;
    Ok(values(&output)?.into_iter().sum())
}

#[test]
fn tucker_forward_and_vjp_match_rank_two_degenerate_case() -> Result<(), Box<dyn std::error::Error>>
{
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let t1 = tensor(&backend, &authority, &[1, 1], &[2.0], &cancellation)?;
    let d1 = tensor(&backend, &authority, &[1, 1], &[3.0], &cancellation)?;
    let u1 = tensor(&backend, &authority, &[1, 1], &[4.0], &cancellation)?;
    let t2 = tensor(&backend, &authority, &[1, 1], &[5.0], &cancellation)?;
    let d2 = tensor(&backend, &authority, &[1, 1], &[6.0], &cancellation)?;
    let u2 = tensor(&backend, &authority, &[1, 1], &[7.0], &cancellation)?;
    let scale = tensor(&backend, &authority, &[], &[0.5], &cancellation)?;
    let (function, output) = HadaWeightTuckerFunction::forward(
        &backend,
        [&t1, &d1, &u1, &t2, &d2, &u2],
        &scale,
        [true, true, true, true, true, true, false],
        &execution,
    )?;
    close(&values(&output)?, &[2520.0]);
    let grad = tensor(&backend, &authority, &[1, 1], &[1.0], &cancellation)?;
    let gradients = function.backward(&backend, Some(&grad), &execution)?;
    assert!(gradients[6].is_none());
    close(
        &values(gradients[0].as_ref().ok_or("missing gradient")?)?,
        &[1260.0],
    );
    close(
        &values(gradients[1].as_ref().ok_or("missing gradient")?)?,
        &[840.0],
    );
    close(
        &values(gradients[2].as_ref().ok_or("missing gradient")?)?,
        &[630.0],
    );
    close(
        &values(gradients[3].as_ref().ok_or("missing gradient")?)?,
        &[504.0],
    );
    close(
        &values(gradients[4].as_ref().ok_or("missing gradient")?)?,
        &[420.0],
    );
    close(
        &values(gradients[5].as_ref().ok_or("missing gradient")?)?,
        &[360.0],
    );
    let expected = [1260.0, 840.0, 630.0, 504.0, 420.0, 360.0];
    let base = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let epsilon = 1.0e-2;
    for index in 0..base.len() {
        let mut plus = base;
        let mut minus = base;
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let numerical = (tucker_objective(&backend, &authority, plus, &cancellation)?
            - tucker_objective(&backend, &authority, minus, &cancellation)?)
            / (2.0 * epsilon);
        assert!(
            (numerical - expected[index]).abs() < 1.0,
            "tucker gradient {index}: {numerical} != {}",
            expected[index]
        );
    }
    Ok(())
}

fn tucker_objective(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    operands: [f32; 6],
    cancellation: &CancellationToken,
) -> Result<f32, Box<dyn std::error::Error>> {
    let execution = context(backend, authority, cancellation)?;
    let t1 = tensor(backend, authority, &[1, 1], &[operands[0]], cancellation)?;
    let d1 = tensor(backend, authority, &[1, 1], &[operands[1]], cancellation)?;
    let u1 = tensor(backend, authority, &[1, 1], &[operands[2]], cancellation)?;
    let t2 = tensor(backend, authority, &[1, 1], &[operands[3]], cancellation)?;
    let d2 = tensor(backend, authority, &[1, 1], &[operands[4]], cancellation)?;
    let u2 = tensor(backend, authority, &[1, 1], &[operands[5]], cancellation)?;
    let scale = tensor(backend, authority, &[], &[0.5], cancellation)?;
    let (_, output) = HadaWeightTuckerFunction::forward(
        backend,
        [&t1, &d1, &u1, &t2, &d2, &u2],
        &scale,
        [false; 7],
        &execution,
    )?;
    values(&output)?
        .first()
        .copied()
        .ok_or_else(|| "missing Tucker output".into())
}

#[test]
fn add_aux_loss_preserves_alias_and_adds_unit_gradient() -> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let input = tensor(&backend, &authority, &[2], &[2.0, 3.0], &cancellation)?;
    let loss = tensor(&backend, &authority, &[], &[7.0], &cancellation)?;
    let (function, output) = AddAuxLossFunction::forward(&input, &loss, true, &execution)?;
    assert_eq!(input.storage_id(), output.storage_id());
    let grad = tensor(&backend, &authority, &[2], &[4.0, 5.0], &cancellation)?;
    let gradients = function.backward(&backend, Some(&grad), &execution)?;
    close(
        &values(gradients[0].as_ref().ok_or("missing gradient")?)?,
        &[4.0, 5.0],
    );
    close(
        &values(gradients[1].as_ref().ok_or("missing gradient")?)?,
        &[1.0],
    );
    Ok(())
}

struct SquareCallable {
    input_tensor_id: TensorId,
    input_storage_id: StorageId,
}
impl CheckpointCallable for SquareCallable {
    fn forward(
        &self,
        backend: &CpuBackend,
        inputs: &[Tensor],
        mode: GradientMode,
        autocast: &sim_tensor::AutocastPolicy,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, AutogradBreadthError> {
        assert_eq!(mode, GradientMode::NoGrad);
        assert!(autocast.enabled());
        assert_eq!(autocast.dtype(), DType::F16);
        assert!(!autocast.cache_enabled());
        let input = inputs
            .first()
            .ok_or_else(|| AutogradBreadthError::InvalidInput("missing input".to_owned()))?;
        assert_eq!(input.tensor_id(), self.input_tensor_id);
        assert_eq!(input.storage_id(), self.input_storage_id);
        let source = values(input)?;
        let output = source.iter().map(|value| value * value).collect::<Vec<_>>();
        let descriptor = TensorDescriptor::contiguous(
            input.descriptor().shape().to_vec(),
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        Ok(vec![backend.upload_f32(descriptor, &output, execution)?.0])
    }
    fn recompute_vjp(
        &self,
        backend: &CpuBackend,
        inputs: &[Tensor],
        parameters: &[Tensor],
        output_gradients: &[Option<Tensor>],
        mode: GradientMode,
        autocast: &sim_tensor::AutocastPolicy,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradBreadthError> {
        assert_eq!(mode, GradientMode::Enabled);
        assert!(autocast.enabled());
        assert_eq!(autocast.dtype(), DType::F16);
        assert!(!autocast.cache_enabled());
        assert!(parameters.is_empty());
        let input = inputs
            .first()
            .ok_or_else(|| AutogradBreadthError::InvalidInput("missing input".to_owned()))?;
        assert_ne!(input.tensor_id(), self.input_tensor_id);
        assert_eq!(input.storage_id(), self.input_storage_id);
        let gradient = output_gradients
            .first()
            .and_then(Option::as_ref)
            .ok_or_else(|| AutogradBreadthError::InvalidInput("missing gradient".to_owned()))?;
        let result = values(input)?
            .iter()
            .zip(values(gradient)?)
            .map(|(value, gradient)| 2.0 * value * gradient)
            .collect::<Vec<_>>();
        let descriptor = TensorDescriptor::contiguous(
            input.descriptor().shape().to_vec(),
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        Ok(vec![Some(
            backend.upload_f32(descriptor, &result, execution)?.0,
        )])
    }
}

struct FailingRecomputeCallable;

impl CheckpointCallable for FailingRecomputeCallable {
    fn forward(
        &self,
        _backend: &CpuBackend,
        inputs: &[Tensor],
        mode: GradientMode,
        _autocast: &sim_tensor::AutocastPolicy,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, AutogradBreadthError> {
        assert_eq!(mode, GradientMode::NoGrad);
        execution.check()?;
        Ok(inputs.to_vec())
    }

    fn recompute_vjp(
        &self,
        _backend: &CpuBackend,
        _inputs: &[Tensor],
        _parameters: &[Tensor],
        _output_gradients: &[Option<Tensor>],
        mode: GradientMode,
        _autocast: &sim_tensor::AutocastPolicy,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradBreadthError> {
        assert_eq!(mode, GradientMode::Enabled);
        execution.check()?;
        Err(AutogradBreadthError::InvalidInput(
            "injected checkpoint recompute failure".to_owned(),
        ))
    }
}

struct CountingRecomputeCallable {
    calls: Arc<AtomicUsize>,
}

impl CheckpointCallable for CountingRecomputeCallable {
    fn forward(
        &self,
        _backend: &CpuBackend,
        inputs: &[Tensor],
        _mode: GradientMode,
        _autocast: &sim_tensor::AutocastPolicy,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, AutogradBreadthError> {
        execution.check()?;
        Ok(inputs.to_vec())
    }

    fn recompute_vjp(
        &self,
        _backend: &CpuBackend,
        inputs: &[Tensor],
        _parameters: &[Tensor],
        _output_gradients: &[Option<Tensor>],
        _mode: GradientMode,
        _autocast: &sim_tensor::AutocastPolicy,
        execution: &ExecutionContext<'_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradBreadthError> {
        execution.check()?;
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(inputs.iter().cloned().map(Some).collect())
    }
}

#[test]
fn checkpoint_policies_reject_create_graph_before_recompute()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let input = tensor(&backend, &authority, &[1], &[2.0], &cancellation)?;
    let gradient = tensor(&backend, &authority, &[1], &[1.0], &cancellation)?;
    let calls = Arc::new(AtomicUsize::new(0));
    let (checkpoint, _) = CheckpointFunction::forward(
        &backend,
        Arc::new(CountingRecomputeCallable {
            calls: calls.clone(),
        }),
        std::slice::from_ref(&input),
        &[],
        vec![true],
        sim_tensor::AutocastPolicy::new(false, DType::F32, false)?,
        &execution,
    )?;
    assert!(matches!(
        checkpoint.backward_with_options(&backend, &[Some(gradient.clone())], true, &execution),
        Err(AutogradBreadthError::HigherOrderUnavailable {
            symbol: "CheckpointFunction",
            policy: HigherOrderPolicy::FirstOrderOnly,
        })
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let (offload, _) = OffloadCheckpointFunction::forward(
        &backend,
        Arc::new(CountingRecomputeCallable {
            calls: calls.clone(),
        }),
        &input,
        true,
        sim_tensor::AutocastPolicy::new(false, DType::F32, false)?,
        &execution,
    )?;
    assert!(matches!(
        offload.backward_with_options(&backend, Some(gradient), true, &execution),
        Err(AutogradBreadthError::HigherOrderUnavailable {
            symbol: "OffloadCheckpointFunction",
            policy: HigherOrderPolicy::FirstOrderOnly,
        })
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[test]
fn checkpoint_and_offload_recompute_vjp_and_release_callable()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let input = tensor(&backend, &authority, &[2], &[2.0, 3.0], &cancellation)?;
    let gradient = tensor(&backend, &authority, &[2], &[1.0, 2.0], &cancellation)?;
    let callable = Arc::new(SquareCallable {
        input_tensor_id: input.tensor_id(),
        input_storage_id: input.storage_id(),
    });
    let (checkpoint, outputs) = CheckpointFunction::forward(
        &backend,
        callable.clone(),
        std::slice::from_ref(&input),
        &[],
        vec![true],
        sim_tensor::AutocastPolicy::new(true, DType::F16, false)?,
        &execution,
    )?;
    assert_eq!(Arc::strong_count(&callable), 2);
    close(&values(&outputs[0])?, &[4.0, 9.0]);
    let gradients =
        checkpoint.backward_source_arity(&backend, &[Some(gradient.clone())], &execution)?;
    assert_eq!(Arc::strong_count(&callable), 1);
    assert_eq!(gradients.len(), 3);
    assert!(gradients[0].is_none() && gradients[1].is_none());
    close(
        &values(gradients[2].as_ref().ok_or("missing gradient")?)?,
        &[4.0, 12.0],
    );
    let (offload, output) = OffloadCheckpointFunction::forward(
        &backend,
        callable.clone(),
        &input,
        true,
        sim_tensor::AutocastPolicy::new(true, DType::F16, false)?,
        &execution,
    )?;
    assert_eq!(Arc::strong_count(&callable), 2);
    close(&values(&output)?, &[4.0, 9.0]);
    let offload_gradients = offload.backward(&backend, Some(gradient.clone()), &execution)?;
    close(
        &values(offload_gradients[0].as_ref().ok_or("missing gradient")?)?,
        &[4.0, 12.0],
    );
    assert!(offload_gradients[1].is_none());
    assert_eq!(Arc::strong_count(&callable), 1);
    let (offload_without_gradient, _) = OffloadCheckpointFunction::forward(
        &backend,
        callable.clone(),
        &input,
        false,
        sim_tensor::AutocastPolicy::new(true, DType::F16, false)?,
        &execution,
    )?;
    assert!(offload_without_gradient.backward(&backend, Some(gradient), &execution)?[0].is_none());
    assert_eq!(Arc::strong_count(&callable), 1);

    let failing_callable = Arc::new(FailingRecomputeCallable);
    let (failing_checkpoint, failing_outputs) = CheckpointFunction::forward(
        &backend,
        failing_callable.clone(),
        std::slice::from_ref(&input),
        &[],
        vec![true],
        sim_tensor::AutocastPolicy::new(false, DType::F32, true)?,
        &execution,
    )?;
    assert_eq!(Arc::strong_count(&failing_callable), 2);
    let failing_output = failing_outputs
        .first()
        .ok_or("missing failing checkpoint output")?
        .clone();
    assert!(matches!(
        failing_checkpoint.backward(
            &backend,
            &[Some(failing_output)],
            &execution,
        ),
        Err(AutogradBreadthError::InvalidInput(reason))
            if reason == "injected checkpoint recompute failure"
    ));
    assert_eq!(Arc::strong_count(&failing_callable), 1);
    Ok(())
}

#[test]
fn custom_functions_reject_wrong_stream_and_gradient_shape_before_compute()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let default_execution = context(&backend, &authority, &cancellation)?;
    let other_execution = backend.execution_context(
        StreamId::new(1),
        authority.authorize_workspace(8 * 1024 * 1024)?,
        &cancellation,
    );
    let input = tensor(
        &backend,
        &authority,
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let codebook = tensor(
        &backend,
        &authority,
        &[2, 2],
        &[0.0, 0.0, 1.0, 1.0],
        &cancellation,
    )?;
    assert!(matches!(
        VectorQuantizeFunction::forward(
            &backend,
            &input,
            &codebook,
            [true, true],
            &other_execution,
        ),
        Err(AutogradBreadthError::Tensor(
            TensorError::StreamMismatch { .. }
        ))
    ));

    let half_input = sim_tensor::generated_sim_operator_indirection_01::tensor_from_f32_with_context_exact_native(
        &backend,
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        DType::F16,
        DeviceId::CPU,
        &default_execution,
    )?;
    assert!(matches!(
        VectorQuantizeFunction::forward(
            &backend,
            &half_input,
            &codebook,
            [true, true],
            &default_execution,
        ),
        Err(AutogradBreadthError::UnsupportedDType {
            dtype: DType::F16,
            ..
        })
    ));

    let (function, _, _) = VectorQuantizeFunction::forward(
        &backend,
        &input,
        &codebook,
        [true, true],
        &default_execution,
    )?;
    let wrong_shape = tensor(&backend, &authority, &[1, 4], &[1.0; 4], &cancellation)?;
    assert!(matches!(
        function.backward(&backend, [Some(&wrong_shape), None], &default_execution),
        Err(AutogradBreadthError::InvalidInput(_))
    ));
    Ok(())
}

struct IdentityRule;
impl BackwardRule for IdentityRule {
    fn vjp(
        &self,
        output_gradients: &[Option<Tensor>],
        _saved: &[SavedTensor],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, sim_tensor::AutogradError> {
        Ok(vec![output_gradients.first().cloned().flatten()])
    }
}
struct AddReducer;
impl GradientReducer for AddReducer {
    fn add(
        &self,
        left: Tensor,
        right: Tensor,
        _cancellation: &CancellationToken,
    ) -> Result<Tensor, sim_tensor::AutogradError> {
        let descriptor = left.descriptor().clone();
        let sum = values(&left)?
            .into_iter()
            .zip(values(&right)?)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        let (backend, authority) = fixture()?;
        let cancellation = CancellationToken::default();
        Ok(backend
            .upload_f32(
                descriptor,
                &sum,
                &context(&backend, &authority, &cancellation)?,
            )?
            .0)
    }

    fn add_higher_order(
        &self,
        left: Tensor,
        right: Tensor,
        context: &mut HigherOrderContext<'_, '_>,
    ) -> Result<Tensor, AutogradError> {
        let output = self.add(
            left.clone(),
            right.clone(),
            context.execution().cancellation,
        )?;
        context.record_operation(
            &[&left, &right],
            &[&output],
            &[true],
            Vec::new(),
            Arc::new(AnalyticalAddRule),
        )?;
        Ok(output)
    }
}

struct AnalyticalAddRule;

impl BackwardRule for AnalyticalAddRule {
    fn vjp(
        &self,
        output_gradients: &[Option<Tensor>],
        _saved: &[SavedTensor],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let gradient = output_gradients.first().cloned().flatten();
        Ok(vec![gradient.clone(), gradient])
    }
}

struct AnalyticalSquareRule;

impl BackwardRule for AnalyticalSquareRule {
    fn vjp(
        &self,
        output_gradients: &[Option<Tensor>],
        saved: &[SavedTensor],
        cancellation: &CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        cancellation.check().map_err(|_| AutogradError::Cancelled)?;
        let Some(output_gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None, None]);
        };
        let input = saved.first().ok_or_else(|| AutogradError::InvalidGraph {
            reason: "square rule is missing its input".to_owned(),
        })?;
        let gradient = scalar_product(&[input.tensor(), &output_gradient], 2.0)?;
        Ok(vec![Some(gradient), None])
    }

    fn higher_order_policy(&self) -> HigherOrderPolicy {
        HigherOrderPolicy::Analytical
    }

    fn symbol(&self) -> &'static str {
        "analytical_square"
    }

    fn vjp_higher_order(
        &self,
        output_gradients: &[Option<Tensor>],
        saved: &[SavedTensor],
        context: &mut HigherOrderContext<'_, '_>,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        let Some(output_gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None, None]);
        };
        let input = saved.first().ok_or_else(|| AutogradError::InvalidGraph {
            reason: "square rule is missing its input".to_owned(),
        })?;
        let gradient = scalar_product(&[input.tensor(), &output_gradient], 2.0)?;
        context.record_operation(
            &[input.tensor(), &output_gradient],
            &[&gradient],
            &[true],
            vec![
                SavedTensor::capture(input.tensor()),
                SavedTensor::capture(&output_gradient),
            ],
            Arc::new(AnalyticalSquareGradientRule),
        )?;
        Ok(vec![Some(gradient), None])
    }
}

struct AnalyticalSquareGradientRule;

impl BackwardRule for AnalyticalSquareGradientRule {
    fn vjp(
        &self,
        output_gradients: &[Option<Tensor>],
        saved: &[SavedTensor],
        cancellation: &CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        cancellation.check().map_err(|_| AutogradError::Cancelled)?;
        let Some(output_gradient) = output_gradients.first().cloned().flatten() else {
            return Ok(vec![None, None]);
        };
        let input = saved.first().ok_or_else(|| AutogradError::InvalidGraph {
            reason: "square-gradient rule is missing its input".to_owned(),
        })?;
        let first_gradient = saved.get(1).ok_or_else(|| AutogradError::InvalidGraph {
            reason: "square-gradient rule is missing its incoming gradient".to_owned(),
        })?;
        Ok(vec![
            Some(scalar_product(
                &[first_gradient.tensor(), &output_gradient],
                2.0,
            )?),
            Some(scalar_product(&[input.tensor(), &output_gradient], 2.0)?),
        ])
    }
}

struct RejectedHigherOrderRule {
    policy: HigherOrderPolicy,
    vjp_calls: Arc<AtomicUsize>,
}

impl BackwardRule for RejectedHigherOrderRule {
    fn vjp(
        &self,
        output_gradients: &[Option<Tensor>],
        _saved: &[SavedTensor],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        self.vjp_calls.fetch_add(1, Ordering::SeqCst);
        Ok(vec![output_gradients.first().cloned().flatten()])
    }

    fn higher_order_policy(&self) -> HigherOrderPolicy {
        self.policy
    }

    fn symbol(&self) -> &'static str {
        "rejected_higher_order_rule"
    }
}

fn scalar_product(inputs: &[&Tensor], factor: f32) -> Result<Tensor, AutogradError> {
    let first = inputs.first().ok_or_else(|| AutogradError::InvalidGraph {
        reason: "scalar product requires an input".to_owned(),
    })?;
    let mut value = factor;
    for input in inputs {
        let values = values(input)?;
        let scalar = values
            .first()
            .copied()
            .ok_or_else(|| AutogradError::InvalidGraph {
                reason: "scalar product received an empty tensor".to_owned(),
            })?;
        value *= scalar;
    }
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    Ok(backend
        .upload_f32(
            first.descriptor().clone(),
            &[value],
            &context(&backend, &authority, &cancellation)?,
        )?
        .0)
}

#[test]
fn retained_backward_repeats_then_terminal_backward_releases_tape()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let leaf = LeafId::new("x")?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    let outputs = tape
        .record(
            vec![AutogradInput::Leaf(leaf.clone())],
            1,
            vec![],
            Arc::new(IdentityRule),
        )?
        .ok_or("node not recorded")?;
    let seed = tensor(&backend, &authority, &[], &[2.0], &cancellation)?;
    let first = tape.backward_retain_graph(
        vec![(outputs[0], seed.clone())],
        &AddReducer,
        &cancellation,
        true,
    )?;
    let second =
        tape.backward_retain_graph(vec![(outputs[0], seed)], &AddReducer, &cancellation, false)?;
    close(
        &values(first.get(&leaf).ok_or("missing gradient")?)?,
        &[2.0],
    );
    close(
        &values(second.get(&leaf).ok_or("missing gradient")?)?,
        &[2.0],
    );
    assert_eq!(tape.retained_node_count(), 0);
    assert_eq!(tape.state(), &sim_tensor::TapeState::Completed);
    Ok(())
}

#[test]
fn val_autograd_001_analytical_create_graph_records_square_and_branch_second_derivatives()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let leaf = LeafId::new("x")?;
    let input = tensor(&backend, &authority, &[], &[3.0], &cancellation)?;
    let output = tensor(&backend, &authority, &[], &[9.0], &cancellation)?;
    let seed = tensor(&backend, &authority, &[], &[1.0], &cancellation)?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    tape.set_requires_grad(&input, Some(leaf.clone()), true, &cancellation)?;

    let first_slot = tape
        .record_operation(
            &[&input, &input],
            &[&output],
            &[true],
            vec![SavedTensor::capture(&input)],
            Arc::new(AnalyticalSquareRule),
        )?
        .and_then(|slots| slots.first().copied())
        .ok_or("square was not recorded")?;
    let second_output = tensor(&backend, &authority, &[], &[9.0], &cancellation)?;
    let second_slot = tape
        .record_operation(
            &[&input, &input],
            &[&second_output],
            &[true],
            vec![SavedTensor::capture(&input)],
            Arc::new(AnalyticalSquareRule),
        )?
        .and_then(|slots| slots.first().copied())
        .ok_or("second square was not recorded")?;

    let first_gradients = tape.reverse_with_context(
        vec![(first_slot, seed.clone()), (second_slot, seed.clone())],
        &AddReducer,
        false,
        true,
        &backend,
        &execution,
    )?;
    let first_gradient = first_gradients.get(&leaf).ok_or("missing first gradient")?;
    close(&values(first_gradient)?, &[12.0]);
    let gradient_slot = tape
        .output_slot(first_gradient)
        .ok_or("first gradient has no recorded output slot")?;
    let second_gradients = tape.reverse_with_context(
        vec![(gradient_slot, seed)],
        &AddReducer,
        false,
        false,
        &backend,
        &execution,
    )?;
    close(
        &values(
            second_gradients
                .get(&leaf)
                .ok_or("missing second gradient")?,
        )?,
        &[4.0],
    );
    assert_eq!(tape.state(), &sim_tensor::TapeState::Completed);
    Ok(())
}

#[test]
fn add_aux_loss_records_an_analytical_create_graph_without_detaching()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let input_leaf = LeafId::new("input")?;
    let incoming_leaf = LeafId::new("add-aux-incoming-gradient")?;
    let input = tensor(&backend, &authority, &[1], &[2.0], &cancellation)?;
    let loss = tensor(&backend, &authority, &[1], &[7.0], &cancellation)?;
    let seed = tensor(&backend, &authority, &[1], &[1.0], &cancellation)?;
    let second_seed = tensor(&backend, &authority, &[1], &[1.0], &cancellation)?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    tape.set_requires_grad(&input, Some(input_leaf.clone()), true, &cancellation)?;
    tape.set_requires_grad(&seed, Some(incoming_leaf.clone()), true, &cancellation)?;
    let (_function, output, slot) =
        AddAuxLossFunction::forward_recorded(&backend, &mut tape, &input, &loss, true, &execution)?;
    let gradients = tape.reverse_with_context(
        vec![(slot.ok_or("AddAuxLoss was not recorded")?, seed)],
        &AddReducer,
        false,
        true,
        &backend,
        &execution,
    )?;
    let input_gradient = gradients
        .get(&input_leaf)
        .ok_or("missing AddAuxLoss input gradient")?;
    close(&values(input_gradient)?, &[1.0]);
    let gradient_slot = tape
        .output_slot(input_gradient)
        .ok_or("AddAuxLoss gradient was detached")?;
    let second = tape.reverse_with_context(
        vec![(gradient_slot, second_seed)],
        &AddReducer,
        false,
        false,
        &backend,
        &execution,
    )?;
    close(
        &values(
            second
                .get(&incoming_leaf)
                .ok_or("missing AddAuxLoss second backward")?,
        )?,
        &[1.0],
    );
    assert_eq!(values(&output)?, vec![2.0]);
    Ok(())
}

#[test]
fn reached_non_analytical_policies_reject_before_vjp_or_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    for policy in [
        HigherOrderPolicy::FirstOrderOnly,
        HigherOrderPolicy::OnceDifferentiable,
    ] {
        let leaf = LeafId::new(format!("x-{policy:?}"))?;
        let input = tensor(&backend, &authority, &[], &[3.0], &cancellation)?;
        let output = tensor(&backend, &authority, &[], &[3.0], &cancellation)?;
        let seed = tensor(&backend, &authority, &[], &[1.0], &cancellation)?;
        let vjp_calls = Arc::new(AtomicUsize::new(0));
        let mut tape = AutogradTape::new(GradientMode::Enabled);
        tape.set_requires_grad(&input, Some(leaf), true, &cancellation)?;
        let slot = tape
            .record_operation(
                &[&input],
                &[&output],
                &[true],
                Vec::new(),
                Arc::new(RejectedHigherOrderRule {
                    policy,
                    vjp_calls: vjp_calls.clone(),
                }),
            )?
            .and_then(|slots| slots.first().copied())
            .ok_or("policy rule was not recorded")?;
        let mut store = GradientStore::default();
        assert!(matches!(
            tape.reverse_and_publish_with_context(
                vec![(slot, seed)],
                &AddReducer,
                false,
                true,
                None,
                &mut store,
                &backend,
                &execution,
            ),
            Err(AutogradError::HigherOrderUnavailable {
                symbol: "rejected_higher_order_rule",
                policy: actual,
            }) if actual == policy
        ));
        assert_eq!(vjp_calls.load(Ordering::SeqCst), 0);
        assert!(store.is_empty());
        assert!(matches!(tape.state(), sim_tensor::TapeState::Faulted(_)));
    }
    Ok(())
}

#[test]
fn public_backward_create_graph_publishes_a_recorded_gradient()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let leaf = LeafId::new("public-backward-x")?;
    let input = tensor(&backend, &authority, &[], &[3.0], &cancellation)?;
    let output = tensor(&backend, &authority, &[], &[9.0], &cancellation)?;
    let seed = tensor(&backend, &authority, &[], &[1.0], &cancellation)?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    tape.set_requires_grad(&input, Some(leaf.clone()), true, &cancellation)?;
    let slot = tape
        .record_operation(
            &[&input, &input],
            &[&output],
            &[true],
            vec![SavedTensor::capture(&input)],
            Arc::new(AnalyticalSquareRule),
        )?
        .and_then(|slots| slots.first().copied())
        .ok_or("square was not recorded")?;
    let mut store = GradientStore::default();
    sim_tensor::generated_elementwise_or_runtime_operation_21::backward_method_with_context_exact_native(
        &backend,
        &mut tape,
        slot,
        &output,
        Some(seed.clone()),
        Some(std::slice::from_ref(&leaf)),
        &AddReducer,
        &mut store,
        false,
        true,
        &execution,
    )?;
    let first_gradient = store.gradient(&leaf).ok_or("missing published gradient")?;
    close(&values(first_gradient)?, &[6.0]);
    let first_gradient_slot = tape
        .output_slot(first_gradient)
        .ok_or("published gradient has no output slot")?;
    let second = tape.reverse_with_context(
        vec![(first_gradient_slot, seed)],
        &AddReducer,
        false,
        false,
        &backend,
        &execution,
    )?;
    close(
        &values(second.get(&leaf).ok_or("missing second derivative")?)?,
        &[2.0],
    );
    Ok(())
}

#[test]
fn vector_quantize_create_graph_executes_scatter_second_backward()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let input = tensor(
        &backend,
        &authority,
        &[3, 2],
        &[0.1, 0.2, 2.8, 3.2, 0.8, 1.1],
        &cancellation,
    )?;
    let codebook = tensor(
        &backend,
        &authority,
        &[3, 2],
        &[0.0, 0.0, 1.0, 1.0, 3.0, 3.0],
        &cancellation,
    )?;
    let incoming = tensor(
        &backend,
        &authority,
        &[3, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &cancellation,
    )?;
    let second_seed = tensor(&backend, &authority, &[3, 2], &[1.0; 6], &cancellation)?;
    let input_leaf = LeafId::new("vq-input")?;
    let codebook_leaf = LeafId::new("vq-codebook")?;
    let incoming_leaf = LeafId::new("vq-incoming-gradient")?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    tape.set_requires_grad(&input, Some(input_leaf), true, &cancellation)?;
    tape.set_requires_grad(&codebook, Some(codebook_leaf.clone()), true, &cancellation)?;
    tape.set_requires_grad(&incoming, Some(incoming_leaf.clone()), true, &cancellation)?;
    let (_function, _output, _indices, slot) = VectorQuantizeFunction::forward_recorded(
        &backend,
        &mut tape,
        &input,
        &codebook,
        [true, true],
        &execution,
    )?;
    let first = tape.reverse_with_context(
        vec![(slot.ok_or("vector_quantize was not recorded")?, incoming)],
        &AddReducer,
        false,
        true,
        &backend,
        &execution,
    )?;
    let codebook_gradient = first
        .get(&codebook_leaf)
        .ok_or("missing vector_quantize codebook gradient")?;
    let gradient_slot = tape
        .output_slot(codebook_gradient)
        .ok_or("vector_quantize gradient was detached")?;
    let second = tape.reverse_with_context(
        vec![(gradient_slot, second_seed)],
        &AddReducer,
        false,
        false,
        &backend,
        &execution,
    )?;
    close(
        &values(
            second
                .get(&incoming_leaf)
                .ok_or("missing vector_quantize second backward")?,
        )?,
        &[1.0; 6],
    );
    Ok(())
}

#[test]
fn hada_weight_create_graph_executes_mixed_second_derivative()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let tensors = [
        tensor(&backend, &authority, &[1, 1], &[2.0], &cancellation)?,
        tensor(&backend, &authority, &[1, 1], &[3.0], &cancellation)?,
        tensor(&backend, &authority, &[1, 1], &[5.0], &cancellation)?,
        tensor(&backend, &authority, &[1, 1], &[7.0], &cancellation)?,
    ];
    let scale = tensor(&backend, &authority, &[1, 1], &[11.0], &cancellation)?;
    let seed = tensor(&backend, &authority, &[1, 1], &[1.0], &cancellation)?;
    let leaves = (0..5)
        .map(|index| LeafId::new(format!("hada-{index}")))
        .collect::<Result<Vec<_>, _>>()?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    for (tensor, leaf) in tensors.iter().chain(std::iter::once(&scale)).zip(&leaves) {
        tape.set_requires_grad(tensor, Some(leaf.clone()), true, &cancellation)?;
    }
    let (_function, _output, slot) = HadaWeightFunction::forward_recorded(
        &backend,
        &mut tape,
        [&tensors[0], &tensors[1], &tensors[2], &tensors[3]],
        &scale,
        [true; 5],
        &execution,
    )?;
    let first = tape.reverse_with_context(
        vec![(slot.ok_or("HadaWeight was not recorded")?, seed.clone())],
        &AddReducer,
        false,
        true,
        &backend,
        &execution,
    )?;
    close(
        &values(first.get(&leaves[0]).ok_or("missing Hada gradient")?)?,
        &[1155.0],
    );
    let gradient_slot = tape
        .output_slot(first.get(&leaves[0]).ok_or("missing Hada gradient")?)
        .ok_or("Hada gradient was detached")?;
    let second = tape.reverse_with_context(
        vec![(gradient_slot, seed)],
        &AddReducer,
        false,
        false,
        &backend,
        &execution,
    )?;
    close(
        &values(
            second
                .get(&leaves[1])
                .ok_or("missing Hada mixed derivative")?,
        )?,
        &[385.0],
    );
    Ok(())
}

#[test]
fn hada_tucker_create_graph_executes_mixed_second_derivative()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let operand_values = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let tensors = operand_values
        .into_iter()
        .map(|value| tensor(&backend, &authority, &[1, 1], &[value], &cancellation))
        .collect::<Result<Vec<_>, _>>()?;
    let scale = tensor(&backend, &authority, &[1, 1], &[11.0], &cancellation)?;
    let seed = tensor(&backend, &authority, &[1, 1], &[1.0], &cancellation)?;
    let leaves = (0..7)
        .map(|index| LeafId::new(format!("tucker-{index}")))
        .collect::<Result<Vec<_>, _>>()?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    for (tensor, leaf) in tensors.iter().chain(std::iter::once(&scale)).zip(&leaves) {
        tape.set_requires_grad(tensor, Some(leaf.clone()), true, &cancellation)?;
    }
    let (_function, _output, slot) = HadaWeightTuckerFunction::forward_recorded(
        &backend,
        &mut tape,
        [
            &tensors[0],
            &tensors[1],
            &tensors[2],
            &tensors[3],
            &tensors[4],
            &tensors[5],
        ],
        &scale,
        [true; 7],
        &execution,
    )?;
    let first = tape.reverse_with_context(
        vec![(
            slot.ok_or("HadaWeightTucker was not recorded")?,
            seed.clone(),
        )],
        &AddReducer,
        false,
        true,
        &backend,
        &execution,
    )?;
    close(
        &values(first.get(&leaves[0]).ok_or("missing Tucker gradient")?)?,
        &[27720.0],
    );
    let gradient_slot = tape
        .output_slot(first.get(&leaves[0]).ok_or("missing Tucker gradient")?)
        .ok_or("Tucker gradient was detached")?;
    let second = tape.reverse_with_context(
        vec![(gradient_slot, seed)],
        &AddReducer,
        false,
        false,
        &backend,
        &execution,
    )?;
    close(
        &values(
            second
                .get(&leaves[1])
                .ok_or("missing Tucker mixed derivative")?,
        )?,
        &[9240.0],
    );
    Ok(())
}

#[test]
fn cancellation_is_precedence_safe_and_does_not_publish_partial_results()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = fixture()?;
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let execution = context(&backend, &authority, &cancellation)?;
    let input = tensor(
        &backend,
        &authority,
        &[1, 2],
        &[0.0, 0.0],
        &CancellationToken::default(),
    )?;
    let codebook = tensor(
        &backend,
        &authority,
        &[1, 2],
        &[0.0, 0.0],
        &CancellationToken::default(),
    )?;
    assert!(matches!(
        VectorQuantizeFunction::forward(&backend, &input, &codebook, [true, true], &execution),
        Err(AutogradBreadthError::Cancelled)
    ));
    Ok(())
}
