use sha2::{Digest, Sha256};
use sim_tensor::CpuWorkspaceAuthority;
use sim_tensor::{
    BackendCapabilityMatrix, CancellationToken, CpuBackend, DType, DecodedScalar, DeviceId,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, GradientMode, Scalar, StreamId, Tensor,
    TensorDescriptor,
    generated_elementwise_or_runtime_operation_02::{
        ElementwiseRuntimePartTwoError, acos_jvp_with_context_exact_native,
        acos_vjp_with_context_exact_native, acos_with_context_exact_native,
        adamw_with_context_exact_native, ceil_jvp_with_context_exact_native,
        ceil_vjp_with_context_exact_native, ceil_with_context_exact_native,
        cudnn_is_available_exact_native, equal_scalar_with_context_exact_native,
        is_inference_mode_enabled_exact_native, jit_is_scripting_exact_native,
        log_jvp_with_context_exact_native, log_vjp_with_context_exact_native,
        log_with_context_exact_native, polar_with_context_exact_native,
        tanh_jvp_with_context_exact_native, tanh_vjp_with_context_exact_native,
        tanh_with_context_exact_native, view_as_complex_exact_native,
        view_as_complex_jvp_exact_native, view_as_complex_vjp_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use std::{collections::BTreeSet, fs, path::Path};

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-0FB8594194A8",
    "COMFY-TENSOR-OP-10A0FC173128",
    "COMFY-TENSOR-OP-10FC4A6ED9AA",
    "COMFY-TENSOR-OP-11C887BB4214",
    "COMFY-TENSOR-OP-14218933001D",
    "COMFY-TENSOR-OP-147180FA6AF4",
    "COMFY-TENSOR-OP-14815FE141B4",
    "COMFY-TENSOR-OP-1599F5E140D0",
    "COMFY-TENSOR-OP-1602683BB161",
    "COMFY-TENSOR-OP-160E75523010",
    "COMFY-TENSOR-OP-190DE0F94657",
    "COMFY-TENSOR-OP-1912E4160DE1",
];

fn backend() -> Result<(CpuBackend, CpuWorkspaceAuthority), Box<dyn std::error::Error>> {
    Ok(CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?)
}

#[test]
fn workspace_authorization_is_exact_bounded_and_convergent_for_part_two()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &authority,
        &[4],
        &[1.0, 2.0, 1.0, 3.0],
        &cancellation,
    )?;

    let authorization = authority.authorize_workspace(4)?;
    let context =
        backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancellation);
    equal_scalar_with_context_exact_native(&backend, &input, Scalar::Float(1.0), &context)?;
    assert_eq!(authorization.peak_bytes(), 4);
    assert_eq!(authorization.in_use_bytes(), 0);

    let insufficient = authority.authorize_workspace(3)?;
    let context = backend.execution_context(StreamId::DEFAULT, insufficient.clone(), &cancellation);
    assert!(
        equal_scalar_with_context_exact_native(&backend, &input, Scalar::Float(1.0), &context)
            .is_err()
    );
    assert_eq!(insufficient.in_use_bytes(), 0);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let authorization = authority.authorize_workspace(4)?;
    let context = backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancelled);
    assert!(
        equal_scalar_with_context_exact_native(&backend, &input, Scalar::Float(1.0), &context)
            .is_err()
    );
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

fn upload_i64(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    values: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor = TensorDescriptor::contiguous(
        vec![u64::try_from(values.len())?],
        DType::I64,
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

#[test]
fn resolution_slice_seals_exactly_the_assigned_contracts_and_fixtures()
-> Result<(), Box<dyn std::error::Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_02")
        .ok_or("elementwise/runtime part-two resolution slice is missing")?;
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
            "comfy-parity-tensor-ops-elementwise-or-runtime-operation-comfy-tensor-op-0fb8594194a8"
        );
        let bytes = fs::read(workspace.join(contract.evidence_fixture))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            contract.evidence_fixture_sha256
        );
        let callable = match contract.operation_id {
            "COMFY-TENSOR-OP-0FB8594194A8" => "acos_with_context_exact_native",
            "COMFY-TENSOR-OP-10A0FC173128" => "polar_with_context_exact_native",
            "COMFY-TENSOR-OP-10FC4A6ED9AA" => "tanh_with_context_exact_native",
            "COMFY-TENSOR-OP-11C887BB4214" => "view_as_complex_exact_native",
            "COMFY-TENSOR-OP-14218933001D" => "equal_scalar_with_context_exact_native",
            "COMFY-TENSOR-OP-147180FA6AF4" => "npu_memory_stats_exact_native",
            "COMFY-TENSOR-OP-14815FE141B4" => "cudnn_is_available_exact_native",
            "COMFY-TENSOR-OP-1599F5E140D0" => "ceil_with_context_exact_native",
            "COMFY-TENSOR-OP-1602683BB161" => "adamw_with_context_exact_native",
            "COMFY-TENSOR-OP-160E75523010" => "is_inference_mode_enabled_exact_native",
            "COMFY-TENSOR-OP-190DE0F94657" => "jit_is_scripting_exact_native",
            "COMFY-TENSOR-OP-1912E4160DE1" => "log_with_context_exact_native",
            _ => return Err("unexpected Task 45 operation identifier".into()),
        };
        assert!(contract.rust_signature.contains(callable));
        if contract.rust_signature.contains("ExecutionContext") {
            assert!(contract.rust_signature.contains("ExecutionContext<'_>"));
        }
    }
    Ok(())
}

#[test]
fn real_unary_adapters_preserve_domains_and_analytical_gradients()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &authority,
        &[4],
        &[-1.25, -0.0, 0.5, 1.25],
        &cancellation,
    )?;
    assert_eq!(
        values(
            &backend,
            &authority,
            &ceil_with_context_exact_native(
                &backend,
                &input,
                &backend.execution_context(
                    StreamId::DEFAULT,
                    authority.authorize_workspace(1024 * 1024)?,
                    &cancellation,
                )
            )?,
            &cancellation
        )?,
        [-1.0, -0.0, 1.0, 2.0]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &ceil_vjp_with_context_exact_native(
                &backend,
                &input,
                &backend.execution_context(
                    StreamId::DEFAULT,
                    authority.authorize_workspace(1024 * 1024)?,
                    &cancellation,
                )
            )?,
            &cancellation
        )?,
        [0.0; 4]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &ceil_jvp_with_context_exact_native(
                &backend,
                &input,
                &backend.execution_context(
                    StreamId::DEFAULT,
                    authority.authorize_workspace(1024 * 1024)?,
                    &cancellation,
                )
            )?,
            &cancellation
        )?,
        [0.0; 4]
    );

    let input = upload_f32(&backend, &authority, &[3], &[-0.5, 0.0, 0.5], &cancellation)?;
    let gradient = upload_f32(&backend, &authority, &[3], &[1.0, 2.0, 3.0], &cancellation)?;
    let tanh = tanh_with_context_exact_native(
        &backend,
        &input,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    let tanh_values = values(&backend, &authority, &tanh, &cancellation)?;
    assert!((tanh_values[0] - (-0.5_f32).tanh()).abs() < 0.000_001);
    assert_eq!(tanh_values[1], 0.0);
    let tanh_vjp = tanh_vjp_with_context_exact_native(
        &backend,
        &input,
        &gradient,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    let tanh_jvp = tanh_jvp_with_context_exact_native(
        &backend,
        &input,
        &gradient,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &tanh_vjp, &cancellation)?,
        values(&backend, &authority, &tanh_jvp, &cancellation)?
    );

    let acos_input = upload_f32(&backend, &authority, &[3], &[-1.0, 0.0, 1.0], &cancellation)?;
    let acos = values(
        &backend,
        &authority,
        &acos_with_context_exact_native(
            &backend,
            &acos_input,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancellation,
            ),
        )?,
        &cancellation,
    )?;
    assert!((acos[0] - std::f32::consts::PI).abs() < 0.000_001);
    assert!((acos[1] - std::f32::consts::FRAC_PI_2).abs() < 0.000_001);
    assert_eq!(acos[2], 0.0);
    let center = upload_f32(&backend, &authority, &[1], &[0.0], &cancellation)?;
    let unit = upload_f32(&backend, &authority, &[1], &[1.0], &cancellation)?;
    assert_eq!(
        values(
            &backend,
            &authority,
            &acos_vjp_with_context_exact_native(
                &backend,
                &center,
                &unit,
                &backend.execution_context(
                    StreamId::DEFAULT,
                    authority.authorize_workspace(1024 * 1024)?,
                    &cancellation,
                )
            )?,
            &cancellation
        )?,
        [-1.0]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &acos_jvp_with_context_exact_native(
                &backend,
                &center,
                &unit,
                &backend.execution_context(
                    StreamId::DEFAULT,
                    authority.authorize_workspace(1024 * 1024)?,
                    &cancellation,
                )
            )?,
            &cancellation
        )?,
        [-1.0]
    );

    let positive = upload_f32(
        &backend,
        &authority,
        &[2],
        &[1.0, std::f32::consts::E],
        &cancellation,
    )?;
    let logarithms = values(
        &backend,
        &authority,
        &log_with_context_exact_native(
            &backend,
            &positive,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancellation,
            ),
        )?,
        &cancellation,
    )?;
    assert_eq!(logarithms[0], 0.0);
    assert!((logarithms[1] - 1.0).abs() < 0.000_001);
    let log_gradient = upload_f32(&backend, &authority, &[2], &[2.0, 4.0], &cancellation)?;
    let vjp = log_vjp_with_context_exact_native(
        &backend,
        &positive,
        &log_gradient,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    let jvp = log_jvp_with_context_exact_native(
        &backend,
        &positive,
        &log_gradient,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &vjp, &cancellation)?,
        values(&backend, &authority, &jvp, &cancellation)?
    );
    Ok(())
}

#[test]
fn metadata_equality_and_unavailable_device_queries_use_canonical_owners()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let indices = upload_i64(&backend, &authority, &[0, 1, 0, 3], &cancellation)?;
    let equal = equal_scalar_with_context_exact_native(
        &backend,
        &indices,
        Scalar::Signed(0),
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(equal.contiguous_bytes()?, [1, 0, 1, 0]);
    let floating = upload_f32(
        &backend,
        &authority,
        &[3],
        &[-0.0, 0.0, f32::NAN],
        &cancellation,
    )?;
    let equal = equal_scalar_with_context_exact_native(
        &backend,
        &floating,
        Scalar::Float(0.0),
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(equal.contiguous_bytes()?, [1, 1, 0]);

    assert!(is_inference_mode_enabled_exact_native(
        GradientMode::Inference,
        &cancellation
    )?);
    assert!(!is_inference_mode_enabled_exact_native(
        GradientMode::Enabled,
        &cancellation
    )?);
    assert!(!jit_is_scripting_exact_native(&cancellation)?);
    let cpu = BackendCapabilityMatrix::for_native_device(DeviceId::CPU)?;
    assert!(!cudnn_is_available_exact_native(&cpu, &cancellation)?);
    Ok(())
}

#[test]
fn polar_and_view_as_complex_preserve_complex_values_and_aliasing()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let magnitude = upload_f32(&backend, &authority, &[2], &[1.0, 2.0], &cancellation)?;
    let angle = upload_f32(
        &backend,
        &authority,
        &[2],
        &[0.0, std::f32::consts::FRAC_PI_2],
        &cancellation,
    )?;
    let polar = polar_with_context_exact_native(
        &backend,
        &magnitude,
        &angle,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(polar.descriptor().dtype(), DType::Complex64);
    assert_eq!(
        polar
            .descriptor()
            .dtype()
            .decode_scalar(polar.element_bytes(&[0])?)?,
        DecodedScalar::Complex {
            real: 1.0,
            imaginary: 0.0
        }
    );
    let second = polar
        .descriptor()
        .dtype()
        .decode_scalar(polar.element_bytes(&[1])?)?;
    assert!(
        matches!(second, DecodedScalar::Complex { real, imaginary } if real.abs() < 0.000_001 && (imaginary - 2.0).abs() < 0.000_001)
    );

    let pairs = upload_f32(
        &backend,
        &authority,
        &[2, 2],
        &[1.0, -2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let storage = pairs.storage_id();
    let complex = view_as_complex_exact_native(&pairs, &cancellation)?;
    assert_eq!(complex.storage_id(), storage);
    assert_eq!(complex.descriptor().shape(), [2]);
    assert_eq!(complex.descriptor().dtype(), DType::Complex64);
    assert_eq!(
        complex
            .descriptor()
            .dtype()
            .decode_scalar(complex.element_bytes(&[0])?)?,
        DecodedScalar::Complex {
            real: 1.0,
            imaginary: -2.0
        }
    );
    let complex_tangent = view_as_complex_jvp_exact_native(&pairs, &cancellation)?;
    assert_eq!(complex_tangent.storage_id(), pairs.storage_id());
    let input_gradient = view_as_complex_vjp_exact_native(&complex, &cancellation)?;
    assert_eq!(input_gradient.storage_id(), pairs.storage_id());
    assert_eq!(input_gradient.descriptor().shape(), [2, 2]);
    assert_eq!(
        values(&backend, &authority, &input_gradient, &cancellation)?,
        [1.0, -2.0, 3.0, 4.0]
    );
    Ok(())
}

#[test]
fn adamw_stages_parameter_and_moment_updates_before_commit()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let mut parameters = vec![upload_f32(
        &backend,
        &authority,
        &[2],
        &[1.0, 2.0],
        &cancellation,
    )?];
    let mut gradients = vec![upload_f32(
        &backend,
        &authority,
        &[2],
        &[0.1, -0.2],
        &cancellation,
    )?];
    let mut averages = vec![upload_f32(
        &backend,
        &authority,
        &[2],
        &[0.0, 0.0],
        &cancellation,
    )?];
    let mut average_squares = vec![upload_f32(
        &backend,
        &authority,
        &[2],
        &[0.0, 0.0],
        &cancellation,
    )?];
    let mut maxima = vec![upload_f32(
        &backend,
        &authority,
        &[2],
        &[0.0, 0.0],
        &cancellation,
    )?];
    adamw_with_context_exact_native(
        &backend,
        &mut parameters,
        &gradients,
        &mut averages,
        &mut average_squares,
        &mut maxima,
        &[1],
        true,
        0.9,
        0.999,
        0.01,
        0.01,
        1.0e-8,
        false,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    let parameter_values = values(&backend, &authority, &parameters[0], &cancellation)?;
    assert!((parameter_values[0] - 0.9899).abs() < 0.000_01);
    assert!((parameter_values[1] - 2.0098).abs() < 0.000_01);
    let average_values = values(&backend, &authority, &averages[0], &cancellation)?;
    assert!((average_values[0] - 0.01).abs() < 0.000_001);
    assert!((average_values[1] - (-0.02)).abs() < 0.000_001);
    let squares = values(&backend, &authority, &average_squares[0], &cancellation)?;
    let maximum_values = values(&backend, &authority, &maxima[0], &cancellation)?;
    assert_eq!(squares, maximum_values);

    let before_parameter = values(&backend, &authority, &parameters[0], &cancellation)?;
    let before_average = values(&backend, &authority, &averages[0], &cancellation)?;
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(
        adamw_with_context_exact_native(
            &backend,
            &mut parameters,
            &gradients,
            &mut averages,
            &mut average_squares,
            &mut maxima,
            &[2],
            true,
            0.9,
            0.999,
            0.01,
            0.01,
            1.0e-8,
            false,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancelled,
            ),
        )
        .is_err()
    );
    assert_eq!(
        values(&backend, &authority, &parameters[0], &cancellation)?,
        before_parameter
    );
    assert_eq!(
        values(&backend, &authority, &averages[0], &cancellation)?,
        before_average
    );

    parameters.push(upload_f32(
        &backend,
        &authority,
        &[1],
        &[3.0],
        &cancellation,
    )?);
    gradients.push(upload_f32(
        &backend,
        &authority,
        &[2],
        &[0.3, 0.4],
        &cancellation,
    )?);
    averages.push(upload_f32(
        &backend,
        &authority,
        &[1],
        &[0.0],
        &cancellation,
    )?);
    average_squares.push(upload_f32(
        &backend,
        &authority,
        &[1],
        &[0.0],
        &cancellation,
    )?);
    maxima.push(upload_f32(
        &backend,
        &authority,
        &[1],
        &[0.0],
        &cancellation,
    )?);
    let before_first_parameter = values(&backend, &authority, &parameters[0], &cancellation)?;
    let before_first_average = values(&backend, &authority, &averages[0], &cancellation)?;
    assert!(
        adamw_with_context_exact_native(
            &backend,
            &mut parameters,
            &gradients,
            &mut averages,
            &mut average_squares,
            &mut maxima,
            &[2, 1],
            true,
            0.9,
            0.999,
            0.01,
            0.01,
            1.0e-8,
            false,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancellation,
            ),
        )
        .is_err()
    );
    assert_eq!(
        values(&backend, &authority, &parameters[0], &cancellation)?,
        before_first_parameter
    );
    assert_eq!(
        values(&backend, &authority, &averages[0], &cancellation)?,
        before_first_average
    );
    Ok(())
}

#[test]
fn every_local_task45_adapter_honors_pre_cancellation_before_validation_or_publication()
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
        ceil_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartTwoError::Cancelled)
    ));
    assert!(matches!(
        tanh_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartTwoError::Cancelled)
    ));
    assert!(matches!(
        acos_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartTwoError::Cancelled)
    ));
    assert!(matches!(
        equal_scalar_with_context_exact_native(
            &backend,
            &input,
            Scalar::Float(1.0),
            &cancelled_context,
        ),
        Err(ElementwiseRuntimePartTwoError::Cancelled)
    ));
    assert!(matches!(
        log_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartTwoError::Cancelled)
    ));
    assert!(matches!(
        polar_with_context_exact_native(&backend, &input, &input, &cancelled_context),
        Err(ElementwiseRuntimePartTwoError::Cancelled)
    ));
    assert!(matches!(
        view_as_complex_exact_native(&input, &cancelled),
        Err(ElementwiseRuntimePartTwoError::Cancelled)
    ));
    let capabilities = BackendCapabilityMatrix::for_native_device(DeviceId::CPU)?;
    assert!(matches!(
        cudnn_is_available_exact_native(&capabilities, &cancelled),
        Err(ElementwiseRuntimePartTwoError::Cancelled)
    ));
    assert!(matches!(
        is_inference_mode_enabled_exact_native(GradientMode::Inference, &cancelled),
        Err(ElementwiseRuntimePartTwoError::Cancelled)
    ));
    assert!(matches!(
        jit_is_scripting_exact_native(&cancelled),
        Err(ElementwiseRuntimePartTwoError::Cancelled)
    ));
    let mut parameters = Vec::new();
    let gradients = Vec::new();
    let mut averages = Vec::new();
    let mut average_squares = Vec::new();
    let mut maxima = Vec::new();
    assert!(matches!(
        adamw_with_context_exact_native(
            &backend,
            &mut parameters,
            &gradients,
            &mut averages,
            &mut average_squares,
            &mut maxima,
            &[],
            false,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            false,
            &cancelled_context,
        ),
        Err(ElementwiseRuntimePartTwoError::Cancelled)
    ));
    assert_eq!(cancelled_context.scratch.peak_bytes(), 0);
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}
