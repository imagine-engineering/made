use sha2::{Digest, Sha256};
use sim_tensor::CpuWorkspaceAuthority;
use sim_tensor::{
    CancellationToken, CpuBackend, DType, DecodedScalar, DeviceId,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, GradScalerConfig, GradScalerOptimizerDecision,
    RetryRngPolicy, RngAlgorithm, RngError, RngProfileVersion, RngStream, RngStreamAddress, Scalar,
    StreamId, Tensor, TensorDescriptor,
    generated_elementwise_or_runtime_operation_04::{
        ElementwiseRuntimePartFourError, KaimingMode, KaimingNonlinearity,
        deg2rad_jvp_with_context_exact_native, deg2rad_vjp_with_context_exact_native,
        deg2rad_with_context_exact_native, grad_scaler_exact_native,
        isfinite_with_context_exact_native, kaiming_uniform_in_place_exact_native,
        nan_to_num_jvp_with_context_exact_native, nan_to_num_vjp_with_context_exact_native,
        nan_to_num_with_context_exact_native, relu_jvp_with_context_exact_native,
        relu_vjp_with_context_exact_native, relu_with_context_exact_native,
        tensor_eq_scalar_with_context_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::DeviceKind;
use std::{collections::BTreeSet, fs, path::Path};

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-30F43E74E34C",
    "COMFY-TENSOR-OP-2C340B4A7331",
    "COMFY-TENSOR-OP-2881ABE3D797",
    "COMFY-TENSOR-OP-28BA5C917CFB",
    "COMFY-TENSOR-OP-310F887878BC",
    "COMFY-TENSOR-OP-2ED7F479B4BD",
    "COMFY-TENSOR-OP-2C5A78E85B7F",
    "COMFY-TENSOR-OP-2E9C9B320055",
    "COMFY-TENSOR-OP-290B200830F8",
    "COMFY-TENSOR-OP-3118FDEB2829",
    "COMFY-TENSOR-OP-2D7E75B69A4E",
    "COMFY-TENSOR-OP-289B94EF73DE",
];

fn backend() -> Result<(CpuBackend, CpuWorkspaceAuthority), Box<dyn std::error::Error>> {
    Ok(CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?)
}

#[test]
fn workspace_authorization_is_exact_bounded_and_convergent_for_part_four()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &authority,
        &[4],
        &[1.0, f32::INFINITY, f32::NAN, -2.0],
        &cancellation,
    )?;

    let authorization = authority.authorize_workspace(4)?;
    let context =
        backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancellation);
    isfinite_with_context_exact_native(&backend, &input, &context)?;
    assert_eq!(authorization.peak_bytes(), 4);
    assert_eq!(authorization.in_use_bytes(), 0);

    let insufficient = authority.authorize_workspace(3)?;
    let context = backend.execution_context(StreamId::DEFAULT, insufficient.clone(), &cancellation);
    assert!(isfinite_with_context_exact_native(&backend, &input, &context).is_err());
    assert_eq!(insufficient.in_use_bytes(), 0);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let authorization = authority.authorize_workspace(4)?;
    let context = backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancelled);
    assert!(isfinite_with_context_exact_native(&backend, &input, &context).is_err());
    assert_eq!(authorization.in_use_bytes(), 0);
    Ok(())
}

fn upload_f32(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        cancellation,
    );
    Ok(backend.upload_f32(descriptor, values, &context)?.0)
}

fn values(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(tensor_to_f32_with_context_exact_native(
        backend,
        tensor,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            cancellation,
        ),
    )?)
}

fn upload_complex64(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    values: &[(f32, f32)],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor = TensorDescriptor::contiguous(
        vec![u64::try_from(values.len())?],
        DType::Complex64,
        DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for (real, imaginary) in values {
        bytes.extend_from_slice(&real.to_ne_bytes());
        bytes.extend_from_slice(&imaginary.to_ne_bytes());
    }
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        cancellation,
    );
    Ok(backend.upload_bytes(descriptor, &bytes, &context)?.0)
}

fn upload_i32(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    values: &[i32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor = TensorDescriptor::contiguous(
        vec![u64::try_from(values.len())?],
        DType::I32,
        DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        cancellation,
    );
    Ok(backend.upload_bytes(descriptor, &bytes, &context)?.0)
}

fn complex_values(tensor: &Tensor) -> Result<Vec<(f64, f64)>, Box<dyn std::error::Error>> {
    let mut values = Vec::new();
    for index in 0..tensor.descriptor().shape()[0] {
        match tensor
            .descriptor()
            .dtype()
            .decode_scalar(tensor.element_bytes(&[index])?)?
        {
            DecodedScalar::Complex { real, imaginary } => values.push((real, imaginary)),
            _ => return Err("expected complex tensor value".into()),
        }
    }
    Ok(values)
}

fn rng_stream() -> Result<RngStream, Box<dyn std::error::Error>> {
    Ok(RngStream::new(
        RngProfileVersion::V1,
        RngAlgorithm::Mt19937,
        17,
        RngStreamAddress::new(
            "workflow",
            "attempt",
            "initializer",
            0,
            "parameter-init",
            0,
            0,
            RetryRngPolicy::Replay,
        )?,
    )?)
}

#[test]
fn resolution_slice_seals_exactly_the_assigned_contracts_and_fixtures()
-> Result<(), Box<dyn std::error::Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_04")
        .ok_or("elementwise/runtime part-four resolution slice is missing")?;
    assert_eq!(slice.len(), IDS.len());
    assert_eq!(
        slice
            .iter()
            .map(|contract| contract.operation_id)
            .collect::<BTreeSet<_>>(),
        IDS.into_iter().collect()
    );
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is missing")?;
    for contract in slice.iter() {
        assert_eq!(
            contract.owner_task_id,
            "comfy-parity-tensor-ops-elementwise-or-runtime-operation-comfy-tensor-op-2881abe3d797"
        );
        let bytes = fs::read(workspace.join(contract.evidence_fixture))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            contract.evidence_fixture_sha256
        );
        let callable = match contract.operation_id {
            "COMFY-TENSOR-OP-30F43E74E34C" => "tensor_eq_scalar_with_context_exact_native",
            "COMFY-TENSOR-OP-2C340B4A7331" => "grad_scaler_exact_native",
            "COMFY-TENSOR-OP-2881ABE3D797" => "compile_exact_native",
            "COMFY-TENSOR-OP-28BA5C917CFB" => "cuda_device_count_exact_native",
            "COMFY-TENSOR-OP-310F887878BC" => "cuda_is_available_exact_native",
            "COMFY-TENSOR-OP-2ED7F479B4BD" => "deg2rad_with_context_exact_native",
            "COMFY-TENSOR-OP-2C5A78E85B7F" => "isfinite_with_context_exact_native",
            "COMFY-TENSOR-OP-2E9C9B320055" => "mps_empty_cache_exact_native",
            "COMFY-TENSOR-OP-290B200830F8" => "nan_to_num_with_context_exact_native",
            "COMFY-TENSOR-OP-3118FDEB2829" => "kaiming_uniform_in_place_exact_native",
            "COMFY-TENSOR-OP-2D7E75B69A4E" => "remove_parametrizations_with_context_exact_native",
            "COMFY-TENSOR-OP-289B94EF73DE" => "relu_with_context_exact_native",
            _ => return Err("unexpected Task 47 operation identifier".into()),
        };
        assert!(contract.rust_signature.contains(callable));
        if contract.rust_signature.contains("ExecutionContext") {
            assert!(contract.rust_signature.contains("ExecutionContext<'_>"));
        }
    }
    Ok(())
}

#[test]
fn equality_delegates_and_numeric_predicates_preserve_exact_shapes()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &authority,
        &[5],
        &[0.0, 180.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
        &cancellation,
    )?;
    let equality = tensor_eq_scalar_with_context_exact_native(
        &backend,
        &input,
        Scalar::Float(0.0),
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(equality.contiguous_bytes()?, [1, 0, 0, 0, 0]);
    let finite = isfinite_with_context_exact_native(
        &backend,
        &input,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(finite.contiguous_bytes()?, [1, 1, 0, 0, 0]);
    let radians = deg2rad_with_context_exact_native(
        &backend,
        &upload_f32(&backend, &authority, &[2], &[0.0, 180.0], &cancellation)?,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &radians, &cancellation)?,
        [0.0, std::f32::consts::PI]
    );
    let tangent = deg2rad_jvp_with_context_exact_native(
        &backend,
        &upload_f32(&backend, &authority, &[1], &[180.0], &cancellation)?,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &tangent, &cancellation)?,
        [std::f32::consts::PI]
    );
    Ok(())
}

#[test]
fn nan_to_num_and_relu_forward_gradients_match_the_canonical_profiles()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &authority,
        &[5],
        &[f32::NAN, f32::NEG_INFINITY, -2.0, 3.0, f32::INFINITY],
        &cancellation,
    )?;
    let replaced = nan_to_num_with_context_exact_native(
        &backend,
        &input,
        Some(7.0),
        Some(9.0),
        Some(-9.0),
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &replaced, &cancellation)?,
        [7.0, -9.0, -2.0, 3.0, 9.0]
    );
    let tangent = nan_to_num_jvp_with_context_exact_native(
        &backend,
        &input,
        &upload_f32(&backend, &authority, &[5], &[1.0; 5], &cancellation)?,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &tangent, &cancellation)?,
        [0.0, 0.0, 1.0, 1.0, 0.0]
    );
    let complex_input = upload_complex64(
        &backend,
        &authority,
        &[
            (f32::NAN, 2.0),
            (3.0, f32::INFINITY),
            (f32::NEG_INFINITY, 4.0),
        ],
        &cancellation,
    )?;
    let complex_replaced = nan_to_num_with_context_exact_native(
        &backend,
        &complex_input,
        None,
        None,
        None,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        complex_values(&complex_replaced)?,
        [(0.0, 2.0), (3.0, f32::MAX as f64), (f32::MIN as f64, 4.0)]
    );
    let complex_gradient = upload_complex64(
        &backend,
        &authority,
        &[(11.0, 12.0), (13.0, 14.0), (15.0, 16.0)],
        &cancellation,
    )?;
    let complex_vjp = nan_to_num_vjp_with_context_exact_native(
        &backend,
        &complex_input,
        &complex_gradient,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        complex_values(&complex_vjp)?,
        [(0.0, 12.0), (13.0, 0.0), (0.0, 16.0)]
    );

    let relu_input = upload_f32(
        &backend,
        &authority,
        &[4],
        &[-2.0, -0.0, 3.0, f32::NAN],
        &cancellation,
    )?;
    let relu = relu_with_context_exact_native(
        &backend,
        &relu_input,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    let relu_values = values(&backend, &authority, &relu, &cancellation)?;
    assert_eq!(&relu_values[..3], &[0.0, 0.0, 3.0]);
    assert!(relu_values[3].is_nan());
    let gradient = relu_vjp_with_context_exact_native(
        &backend,
        &relu_input,
        &upload_f32(&backend, &authority, &[4], &[2.0; 4], &cancellation)?,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &gradient, &cancellation)?,
        [0.0, 0.0, 2.0, 0.0]
    );
    assert!(
        relu_with_context_exact_native(
            &backend,
            &upload_i32(&backend, &authority, &[-1, 2], &cancellation)?,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancellation,
            )
        )
        .is_err()
    );
    assert!(
        relu_vjp_with_context_exact_native(
            &backend,
            &relu_input,
            &upload_complex64(&backend, &authority, &[(1.0, 0.0); 4], &cancellation)?,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancellation,
            )
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn grad_scaler_owns_transactional_optimizer_stage_and_nonfinite_backoff()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let mut scaler = grad_scaler_exact_native(
        GradScalerConfig {
            initial_scale: 8.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2,
            enabled: true,
        },
        &cancellation,
    )?;
    let loss = upload_f32(&backend, &authority, &[], &[2.0], &cancellation)?;
    let scaled = scaler.scale_loss_exact_native(&loss, &cancellation)?;
    assert_eq!(
        values(&backend, &authority, &scaled, &cancellation)?,
        [16.0]
    );
    let mut gradients = vec![upload_f32(
        &backend,
        &authority,
        &[2],
        &[8.0, f32::INFINITY],
        &cancellation,
    )?];
    assert!(scaler.unscale_gradients_exact_native(&mut gradients, &cancellation)?);
    let gradient_values = values(&backend, &authority, &gradients[0], &cancellation)?;
    assert_eq!(gradient_values[0], 1.0);
    assert!(gradient_values[1].is_infinite());
    assert_eq!(
        scaler.optimizer_step_decision_exact_native(&cancellation)?,
        GradScalerOptimizerDecision::SkipNonFinite
    );
    scaler.update_exact_native(&cancellation)?;
    assert_eq!(scaler.scale(), 4.0);
    assert_eq!(scaler.growth_tracker(), 0);
    assert!(scaler.update_exact_native(&cancellation).is_err());

    let mut atomic_scaler = grad_scaler_exact_native(
        GradScalerConfig {
            initial_scale: 4.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2,
            enabled: true,
        },
        &cancellation,
    )?;
    let mut staged_gradients = vec![
        upload_f32(&backend, &authority, &[1], &[8.0], &cancellation)?,
        upload_complex64(&backend, &authority, &[(1.0, 0.0)], &cancellation)?,
    ];
    let original = values(&backend, &authority, &staged_gradients[0], &cancellation)?;
    assert!(
        atomic_scaler
            .unscale_gradients_exact_native(&mut staged_gradients, &cancellation)
            .is_err()
    );
    assert_eq!(
        values(&backend, &authority, &staged_gradients[0], &cancellation)?,
        original
    );
    let mut finite = vec![upload_f32(
        &backend,
        &authority,
        &[1],
        &[8.0],
        &cancellation,
    )?];
    assert!(!atomic_scaler.unscale_gradients_exact_native(&mut finite, &cancellation)?);
    assert_eq!(
        atomic_scaler.optimizer_step_decision_exact_native(&cancellation)?,
        GradScalerOptimizerDecision::Run
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(atomic_scaler.update_exact_native(&cancelled).is_err());
    assert_eq!(atomic_scaler.scale(), 4.0);
    assert_eq!(atomic_scaler.growth_tracker(), 0);
    atomic_scaler.update_exact_native(&cancellation)?;
    assert_eq!(atomic_scaler.scale(), 4.0);
    assert_eq!(atomic_scaler.growth_tracker(), 1);

    let mut second_finite = vec![upload_f32(
        &backend,
        &authority,
        &[1],
        &[4.0],
        &cancellation,
    )?];
    assert!(!atomic_scaler.unscale_gradients_exact_native(&mut second_finite, &cancellation)?);
    assert_eq!(
        atomic_scaler.optimizer_step_decision_exact_native(&cancellation)?,
        GradScalerOptimizerDecision::Run
    );
    atomic_scaler.update_exact_native(&cancellation)?;
    assert_eq!(atomic_scaler.scale(), 8.0);
    assert_eq!(atomic_scaler.growth_tracker(), 0);

    let mut disabled = grad_scaler_exact_native(
        GradScalerConfig {
            enabled: false,
            ..GradScalerConfig::default()
        },
        &cancellation,
    )?;
    let disabled_scaled = disabled.scale_loss_exact_native(&loss, &cancellation)?;
    assert_eq!(disabled_scaled.storage_id(), loss.storage_id());
    assert!(!disabled.unscale_gradients_exact_native(&mut [], &cancellation)?);
    assert_eq!(
        disabled.optimizer_step_decision_exact_native(&cancellation)?,
        GradScalerOptimizerDecision::Run
    );
    disabled.update_exact_native(&cancellation)?;
    Ok(())
}

#[test]
fn kaiming_uniform_is_replay_deterministic_bounded_and_cancellation_atomic()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let mut first = upload_f32(&backend, &authority, &[3, 4], &[0.0; 12], &cancellation)?;
    let mut second = upload_f32(&backend, &authority, &[3, 4], &[0.0; 12], &cancellation)?;
    let first_checkpoint = kaiming_uniform_in_place_exact_native(
        &mut first,
        rng_stream()?.begin(None)?,
        5.0_f64.sqrt(),
        KaimingMode::FanIn,
        KaimingNonlinearity::LeakyRelu,
        &cancellation,
    )?;
    let second_checkpoint = kaiming_uniform_in_place_exact_native(
        &mut second,
        rng_stream()?.begin(None)?,
        5.0_f64.sqrt(),
        KaimingMode::FanIn,
        KaimingNonlinearity::LeakyRelu,
        &cancellation,
    )?;
    assert_eq!(first_checkpoint, second_checkpoint);
    let first_values = values(&backend, &authority, &first, &cancellation)?;
    assert_eq!(
        first_values,
        values(&backend, &authority, &second, &cancellation)?
    );
    assert!(first_values.iter().all(|value| value.abs() <= 0.5));

    let cuda = DeviceId::new(DeviceKind::Cuda, 0);
    let mismatched_stream = RngStream::new(
        RngProfileVersion::V2,
        RngAlgorithm::Philox4x32_10,
        17,
        RngStreamAddress::for_device(
            "workflow",
            "attempt",
            "initializer",
            0,
            "parameter-init",
            0,
            0,
            RetryRngPolicy::Replay,
            cuda,
        )?,
    )?;
    let mut mismatched = upload_f32(&backend, &authority, &[2, 2], &[7.0; 4], &cancellation)?;
    assert!(matches!(
        kaiming_uniform_in_place_exact_native(
            &mut mismatched,
            mismatched_stream.begin(None)?,
            0.0,
            KaimingMode::FanIn,
            KaimingNonlinearity::Relu,
            &cancellation,
        ),
        Err(ElementwiseRuntimePartFourError::Rng(
            RngError::DeviceMismatch {
                expected: DeviceId::CPU,
                actual,
            }
        )) if actual == cuda
    ));
    assert_eq!(
        values(&backend, &authority, &mismatched, &cancellation)?,
        [7.0; 4]
    );

    let mut cancelled_tensor = upload_f32(&backend, &authority, &[2, 2], &[3.0; 4], &cancellation)?;
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(
        kaiming_uniform_in_place_exact_native(
            &mut cancelled_tensor,
            rng_stream()?.begin(None)?,
            0.0,
            KaimingMode::FanOut,
            KaimingNonlinearity::Relu,
            &cancelled,
        )
        .is_err()
    );
    assert_eq!(
        values(&backend, &authority, &cancelled_tensor, &cancellation)?,
        [3.0; 4]
    );

    let mut empty = upload_f32(&backend, &authority, &[0, 4], &[], &cancellation)?;
    let empty_storage = empty.storage_id();
    let expected_empty_checkpoint = rng_stream()?.begin(None)?.commit();
    let empty_checkpoint = kaiming_uniform_in_place_exact_native(
        &mut empty,
        rng_stream()?.begin(None)?,
        f64::NAN,
        KaimingMode::FanOut,
        KaimingNonlinearity::LeakyRelu,
        &cancellation,
    )?;
    assert_eq!(empty.storage_id(), empty_storage);
    assert_eq!(empty_checkpoint, expected_empty_checkpoint);
    Ok(())
}

#[test]
fn every_local_task47_adapter_honors_pre_cancellation_before_validation_or_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let live = CancellationToken::default();
    let input = upload_f32(&backend, &authority, &[2], &[1.0, 2.0], &live)?;
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        &cancelled,
    );
    assert!(matches!(
        tensor_eq_scalar_with_context_exact_native(
            &backend,
            &input,
            Scalar::Float(1.0),
            &cancelled_context,
        ),
        Err(ElementwiseRuntimePartFourError::Cancelled)
    ));
    assert!(matches!(
        deg2rad_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartFourError::Cancelled)
    ));
    assert!(matches!(
        deg2rad_vjp_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartFourError::Cancelled)
    ));
    assert!(matches!(
        deg2rad_jvp_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartFourError::Cancelled)
    ));
    assert!(matches!(
        isfinite_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartFourError::Cancelled)
    ));
    assert!(matches!(
        nan_to_num_with_context_exact_native(
            &backend,
            &input,
            None,
            None,
            None,
            &cancelled_context,
        ),
        Err(ElementwiseRuntimePartFourError::Cancelled)
    ));
    assert!(matches!(
        nan_to_num_vjp_with_context_exact_native(&backend, &input, &input, &cancelled_context,),
        Err(ElementwiseRuntimePartFourError::Cancelled)
    ));
    assert!(matches!(
        nan_to_num_jvp_with_context_exact_native(&backend, &input, &input, &cancelled_context,),
        Err(ElementwiseRuntimePartFourError::Cancelled)
    ));
    assert!(matches!(
        relu_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartFourError::Cancelled)
    ));
    assert!(matches!(
        relu_vjp_with_context_exact_native(&backend, &input, &input, &cancelled_context),
        Err(ElementwiseRuntimePartFourError::Cancelled)
    ));
    assert!(matches!(
        relu_jvp_with_context_exact_native(&backend, &input, &input, &cancelled_context),
        Err(ElementwiseRuntimePartFourError::Cancelled)
    ));
    assert!(matches!(
        grad_scaler_exact_native(
            GradScalerConfig {
                initial_scale: f64::NAN,
                ..GradScalerConfig::default()
            },
            &cancelled,
        ),
        Err(ElementwiseRuntimePartFourError::Cancelled)
    ));
    let mut mutable = input;
    let original_storage = mutable.storage_id();
    let original_bytes = mutable.contiguous_bytes()?.to_vec();
    assert!(matches!(
        kaiming_uniform_in_place_exact_native(
            &mut mutable,
            rng_stream()?.begin(None)?,
            f64::NAN,
            KaimingMode::FanIn,
            KaimingNonlinearity::Relu,
            &cancelled,
        ),
        Err(ElementwiseRuntimePartFourError::Cancelled)
    ));
    assert_eq!(mutable.storage_id(), original_storage);
    assert_eq!(mutable.contiguous_bytes()?, original_bytes);
    assert_eq!(cancelled_context.scratch.peak_bytes(), 0);
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn deg2rad_f32_rounding_and_derivatives_match_pinned_source_bits()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let input_bits = [
        0x3d4ccccd_u32,
        0x3eb33333_u32,
        0x3f000000_u32,
        0x410a6666_u32,
        0x41700000_u32,
        0x41b40000_u32,
        0x41f00000_u32,
        0x4233cccd_u32,
        0x42340000_u32,
        0x42700000_u32,
        0x42a9e666_u32,
        0x42aa0000_u32,
    ];
    let expected_bits = [
        0x3a64c388_u32,
        0x3bc82b17_u32,
        0x3c0efa35_u32,
        0x3e1a9822_u32,
        0x3e860a92_u32,
        0x3ec90fdb_u32,
        0x3f060a92_u32,
        0x3f48d6aa_u32,
        0x3f490fdb_u32,
        0x3f860a92_u32,
        0x3fbdc7b6_u32,
        0x3fbde44e_u32,
    ];
    let input_values = input_bits.map(f32::from_bits);
    let input = upload_f32(&backend, &authority, &[12], &input_values, &cancellation)?;
    let authorization = authority.authorize_workspace(1024 * 1024)?;
    let context =
        backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancellation);
    let outputs = [
        deg2rad_with_context_exact_native(&backend, &input, &context)?,
        deg2rad_vjp_with_context_exact_native(&backend, &input, &context)?,
        deg2rad_jvp_with_context_exact_native(&backend, &input, &context)?,
    ];
    for output in outputs {
        let actual = values(&backend, &authority, &output, &cancellation)?;
        assert_eq!(
            actual.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            expected_bits
        );
    }
    assert_eq!(authorization.in_use_bytes(), 0);
    Ok(())
}
