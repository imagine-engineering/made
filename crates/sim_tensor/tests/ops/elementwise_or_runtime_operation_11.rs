use std::{collections::BTreeSet, fs, path::Path};

use sha2::{Digest, Sha256};
use sim_tensor::CpuWorkspaceAuthority;
use sim_tensor::{
    BackendCapabilityMatrix, CancellationToken, CpuBackend, DType, DeviceId, ExecutionContext,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, GradientMode, NativeDeviceProperties, Scalar,
    StreamId, Tensor, TensorDescriptor, TensorError,
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_11::{
        ElementwiseRuntimePartElevenError, NativeAdam, TensorListValue,
        clamp_method_jvp_with_context_exact_native, clamp_method_vjp_with_context_exact_native,
        clamp_method_with_context_exact_native, enable_grad_exact_native,
        isclose_with_context_exact_native, maximum_jvp_with_context_exact_native,
        maximum_vjp_with_context_exact_native, maximum_with_context_exact_native,
        npu_get_device_name_exact_native, round_function_jvp_with_context_exact_native,
        round_function_vjp_with_context_exact_native, round_function_with_context_exact_native,
        sign_jvp_with_context_exact_native, sign_vjp_with_context_exact_native,
        sign_with_context_exact_native, tolist_exact_native,
        zeros_in_place_with_context_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::DeviceKind;

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-7E962991728B",
    "COMFY-TENSOR-OP-7E09C5749B60",
    "COMFY-TENSOR-OP-7F86521B5D09",
    "COMFY-TENSOR-OP-7A0F5559B701",
    "COMFY-TENSOR-OP-7DB0B0EC6483",
    "COMFY-TENSOR-OP-82BC07D67AFD",
    "COMFY-TENSOR-OP-791664AD5273",
    "COMFY-TENSOR-OP-7EA8F732F7A7",
    "COMFY-TENSOR-OP-790CDE1EBF17",
    "COMFY-TENSOR-OP-8162B4C00596",
    "COMFY-TENSOR-OP-82310D1230AF",
    "COMFY-TENSOR-OP-80B937845579",
];

fn context<'a>(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, TensorError> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        cancellation,
    ))
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
    Ok(backend
        .upload_f32(
            descriptor,
            values,
            &context(backend, authority, cancellation)?,
        )?
        .0)
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

fn require_cancelled<T>(
    result: Result<T, ElementwiseRuntimePartElevenError>,
) -> Result<(), Box<dyn std::error::Error>> {
    match result {
        Err(ElementwiseRuntimePartElevenError::Cancelled) => Ok(()),
        Err(error) => Err(format!("expected cancellation, received {error}").into()),
        Ok(_) => Err("cancelled operation unexpectedly succeeded".into()),
    }
}

#[test]
fn canonical_adapters_preserve_value_mutation_and_gradient_semantics()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4096)?,
        &cancellation,
    );
    let input = upload_f32(
        &backend,
        &authority,
        &[2, 3],
        &[-2.0, -0.0, 0.5, 1.5, 2.5, f32::NAN],
        &cancellation,
    )?;
    let clamped = clamp_method_with_context_exact_native(
        &backend,
        &input,
        Some(Scalar::Float(-1.0)),
        Some(Scalar::Float(2.0)),
        &execution,
    )?;
    let clamped_values = values(&backend, &authority, &clamped, &cancellation)?;
    assert_eq!(&clamped_values[..5], &[-1.0, -0.0, 0.5, 1.5, 2.0]);
    assert!(clamped_values[5].is_nan());
    let TensorListValue::List(rows) = tolist_exact_native(&clamped, &cancellation)? else {
        return Err("tolist did not preserve the outer dimension".into());
    };
    assert_eq!(
        rows.first(),
        Some(&TensorListValue::List(vec![
            TensorListValue::Real(-1.0),
            TensorListValue::Real(-0.0),
            TensorListValue::Real(0.5),
        ]))
    );
    let Some(TensorListValue::List(last_row)) = rows.get(1) else {
        return Err("tolist did not preserve the final row".into());
    };
    assert_eq!(
        last_row.get(..2),
        Some([TensorListValue::Real(1.5), TensorListValue::Real(2.0)].as_slice())
    );
    assert!(matches!(last_row.get(2), Some(TensorListValue::Real(value)) if value.is_nan()));
    assert_eq!(
        enable_grad_exact_native(&cancellation)?,
        GradientMode::Enabled
    );

    let rounded = round_function_with_context_exact_native(&backend, &input, &execution)?;
    let rounded_values = values(&backend, &authority, &rounded, &cancellation)?;
    assert_eq!(&rounded_values[..5], &[-2.0, -0.0, 0.0, 2.0, 2.0]);
    assert!(rounded_values[5].is_nan());
    let signed = sign_with_context_exact_native(&backend, &input, &execution)?;
    let signed_values = values(&backend, &authority, &signed, &cancellation)?;
    assert_eq!(&signed_values[..5], &[-1.0, -0.0, 1.0, 1.0, 1.0]);
    assert!(signed_values[1].is_sign_negative());
    assert!(signed_values[5].is_nan());

    let tangent = upload_f32(&backend, &authority, &[2, 3], &[1.0; 6], &cancellation)?;
    for derivative in [
        clamp_method_vjp_with_context_exact_native(
            &backend,
            &input,
            Some(-1.0),
            Some(2.0),
            &tangent,
            &execution,
        )?,
        clamp_method_jvp_with_context_exact_native(
            &backend,
            &input,
            Some(-1.0),
            Some(2.0),
            &tangent,
            &execution,
        )?,
    ] {
        assert_eq!(
            values(&backend, &authority, &derivative, &cancellation)?,
            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        );
    }
    for derivative in [
        round_function_vjp_with_context_exact_native(&backend, &input, &tangent, &execution)?,
        round_function_jvp_with_context_exact_native(&backend, &input, &tangent, &execution)?,
        sign_vjp_with_context_exact_native(&backend, &input, &tangent, &execution)?,
        sign_jvp_with_context_exact_native(&backend, &input, &tangent, &execution)?,
    ] {
        assert_eq!(
            values(&backend, &authority, &derivative, &cancellation)?,
            [0.0; 6]
        );
    }
    let mut initialized = input.clone();
    zeros_in_place_with_context_exact_native(&backend, &mut initialized, &execution)?;
    assert_eq!(
        values(&backend, &authority, &initialized, &cancellation)?,
        [0.0; 6]
    );
    assert!(values(&backend, &authority, &input, &cancellation)?[0] < 0.0);
    Ok(())
}

#[test]
fn isclose_and_maximum_are_broadcast_exact_with_analytical_derivatives()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &authority,
        &[2, 2],
        &[1.0, 3.0, f32::NAN, 4.0],
        &cancellation,
    )?;
    let other = upload_f32(&backend, &authority, &[2], &[1.000_001, 2.0], &cancellation)?;
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024)?,
        &cancellation,
    );
    let close = isclose_with_context_exact_native(
        &backend, &input, &other, 1.0e-5, 1.0e-8, false, &execution,
    )?;
    assert_eq!(close.host_storage_bytes()?, &[1, 0, 0, 0]);

    let left = upload_f32(&backend, &authority, &[2, 1], &[1.0, 3.0], &cancellation)?;
    let right = upload_f32(&backend, &authority, &[1, 2], &[2.0, 3.0], &cancellation)?;
    let maximum = maximum_with_context_exact_native(
        &backend,
        &left,
        ElementwiseOperand::Tensor(&right),
        &execution,
    )?;
    assert_eq!(
        values(&backend, &authority, &maximum, &cancellation)?,
        [2.0, 3.0, 3.0, 3.0]
    );
    let output_gradient = upload_f32(&backend, &authority, &[2, 2], &[1.0; 4], &cancellation)?;
    let vjp = maximum_vjp_with_context_exact_native(
        &backend,
        &left,
        ElementwiseOperand::Tensor(&right),
        &output_gradient,
        &execution,
    )?;
    assert_eq!(
        values(&backend, &authority, &vjp.input, &cancellation)?,
        [0.0, 1.5]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            vjp.other.as_ref().ok_or("other VJP missing")?,
            &cancellation
        )?,
        [1.0, 1.5]
    );
    let input_tangent = upload_f32(&backend, &authority, &[2, 1], &[10.0, 20.0], &cancellation)?;
    let other_tangent = upload_f32(&backend, &authority, &[1, 2], &[30.0, 40.0], &cancellation)?;
    let jvp = maximum_jvp_with_context_exact_native(
        &backend,
        &left,
        ElementwiseOperand::Tensor(&right),
        &input_tangent,
        Some(&other_tangent),
        &execution,
    )?;
    assert_eq!(
        values(&backend, &authority, &jvp, &cancellation)?,
        [30.0, 40.0, 20.0, 30.0]
    );
    Ok(())
}

#[test]
fn workspace_authority_isclose_rejects_underauthorization_and_converges()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &authority,
        &[4],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let other = upload_f32(
        &backend,
        &authority,
        &[4],
        &[1.0, 2.0, 0.0, 4.0],
        &cancellation,
    )?;
    let baseline = backend.memory_snapshot().current_bytes;

    let underauthorized = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(3)?,
        &cancellation,
    );
    assert!(
        isclose_with_context_exact_native(
            &backend,
            &input,
            &other,
            0.0,
            0.0,
            false,
            &underauthorized,
        )
        .is_err()
    );
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4)?,
        &cancellation,
    );
    let result =
        isclose_with_context_exact_native(&backend, &input, &other, 0.0, 0.0, false, &execution)?;
    assert_eq!(result.host_storage_bytes()?, &[1, 1, 0, 1]);
    drop(result);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4)?,
        &cancelled,
    );
    assert!(
        isclose_with_context_exact_native(
            &backend,
            &input,
            &other,
            0.0,
            0.0,
            false,
            &cancelled_context,
        )
        .is_err()
    );
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn facade_derivatives_use_exact_caller_workspace_and_converge()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &authority,
        &[4],
        &[-2.0, -0.0, 1.0, 3.0],
        &cancellation,
    )?;
    let tangent = upload_f32(&backend, &authority, &[4], &[1.0; 4], &cancellation)?;

    let insufficient = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(15)?,
        &cancellation,
    );
    for result in [
        clamp_method_vjp_with_context_exact_native(
            &backend,
            &input,
            Some(-1.0),
            Some(2.0),
            &tangent,
            &insufficient,
        ),
        round_function_vjp_with_context_exact_native(&backend, &input, &tangent, &insufficient),
        sign_vjp_with_context_exact_native(&backend, &input, &tangent, &insufficient),
    ] {
        assert!(result.is_err());
        assert_eq!(insufficient.scratch.in_use_bytes(), 0);
    }

    let exact = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(16)?,
        &cancellation,
    );
    let output = sign_jvp_with_context_exact_native(&backend, &input, &tangent, &exact)?;
    assert_eq!(
        values(&backend, &authority, &output, &cancellation)?,
        [0.0; 4]
    );
    assert_eq!(exact.scratch.peak_bytes(), 16);
    assert_eq!(exact.scratch.in_use_bytes(), 0);
    drop(output);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(16)?,
        &cancelled,
    );
    assert!(
        round_function_jvp_with_context_exact_native(
            &backend,
            &input,
            &tangent,
            &cancelled_context,
        )
        .is_err()
    );
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn adam_and_npu_name_are_focused_adapters_with_checked_failure_paths()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let mut parameters = vec![upload_f32(
        &backend,
        &authority,
        &[1],
        &[1.0],
        &cancellation,
    )?];
    let gradients = vec![upload_f32(
        &backend,
        &authority,
        &[1],
        &[0.5],
        &cancellation,
    )?];
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4096)?,
        &cancellation,
    );
    let mut optimizer =
        NativeAdam::new_with_context_exact_native(&backend, &parameters, 0.1, &execution)?;
    optimizer.step_with_context_exact_native(&backend, &mut parameters, &gradients, &execution)?;
    assert_eq!(optimizer.steps(), [1]);
    assert!((values(&backend, &authority, &parameters[0], &cancellation)?[0] - 0.9).abs() < 1.0e-5);

    let npu = DeviceId::new(DeviceKind::Npu, 3);
    let properties = NativeDeviceProperties::new(
        npu,
        "Zed Native NPU Fixture",
        16 * 1024 * 1024,
        1,
        0,
        Some("zed-npu-v1".to_owned()),
        true,
    )?;
    let capabilities = BackendCapabilityMatrix::new_with_properties(
        npu,
        Vec::new(),
        Vec::new(),
        Some(properties),
    )?;
    assert_eq!(
        npu_get_device_name_exact_native(&capabilities, npu, &cancellation)?,
        "Zed Native NPU Fixture"
    );
    assert!(npu_get_device_name_exact_native(&capabilities, DeviceId::CPU, &cancellation).is_err());
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(enable_grad_exact_native(&cancelled).is_err());
    let cancelled_execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4096)?,
        &cancelled,
    );
    assert!(
        NativeAdam::new_with_context_exact_native(
            &backend,
            &parameters,
            0.1,
            &cancelled_execution,
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn every_local_task54_adapter_honors_pre_cancellation_before_validation_or_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let live = CancellationToken::default();
    let input = upload_f32(&backend, &authority, &[2], &[1.0, -2.0], &live)?;
    let tangent = upload_f32(&backend, &authority, &[2], &[3.0, 4.0], &live)?;
    let mut parameters = vec![input.clone()];
    let gradients = vec![tangent.clone()];
    let live_context = context(&backend, &authority, &live)?;
    let mut optimizer =
        NativeAdam::new_with_context_exact_native(&backend, &parameters, 0.001, &live_context)?;
    let parameter_before = values(&backend, &authority, &parameters[0], &live)?;

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(&backend, &authority, &cancelled)?;

    require_cancelled(clamp_method_with_context_exact_native(
        &backend,
        &input,
        Some(Scalar::Float(2.0)),
        Some(Scalar::Float(-2.0)),
        &cancelled_context,
    ))?;
    require_cancelled(clamp_method_vjp_with_context_exact_native(
        &backend,
        &input,
        Some(2.0),
        Some(-2.0),
        &tangent,
        &cancelled_context,
    ))?;
    require_cancelled(clamp_method_jvp_with_context_exact_native(
        &backend,
        &input,
        Some(2.0),
        Some(-2.0),
        &tangent,
        &cancelled_context,
    ))?;
    require_cancelled(tolist_exact_native(&input, &cancelled))?;
    require_cancelled(enable_grad_exact_native(&cancelled))?;
    require_cancelled(isclose_with_context_exact_native(
        &backend,
        &input,
        &input,
        -1.0,
        -1.0,
        false,
        &cancelled_context,
    ))?;
    require_cancelled(maximum_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Scalar(Scalar::Float(f64::INFINITY)),
        &cancelled_context,
    ))?;
    require_cancelled(maximum_vjp_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Scalar(Scalar::Float(0.0)),
        &tangent,
        &cancelled_context,
    ))?;
    require_cancelled(maximum_jvp_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Scalar(Scalar::Float(0.0)),
        &tangent,
        Some(&tangent),
        &cancelled_context,
    ))?;
    let mut zero_target = input.clone();
    require_cancelled(zeros_in_place_with_context_exact_native(
        &backend,
        &mut zero_target,
        &cancelled_context,
    ))?;
    require_cancelled(npu_get_device_name_exact_native(
        &BackendCapabilityMatrix::new(DeviceId::CPU, Vec::new(), Vec::new())?,
        DeviceId::CPU,
        &cancelled,
    ))?;
    require_cancelled(NativeAdam::new_with_context_exact_native(
        &backend,
        &[],
        f32::NAN,
        &cancelled_context,
    ))?;
    require_cancelled(optimizer.step_with_context_exact_native(
        &backend,
        &mut parameters,
        &gradients,
        &cancelled_context,
    ))?;
    require_cancelled(round_function_with_context_exact_native(
        &backend,
        &input,
        &cancelled_context,
    ))?;
    require_cancelled(round_function_vjp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        &cancelled_context,
    ))?;
    require_cancelled(round_function_jvp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        &cancelled_context,
    ))?;
    require_cancelled(sign_with_context_exact_native(
        &backend,
        &input,
        &cancelled_context,
    ))?;
    require_cancelled(sign_vjp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        &cancelled_context,
    ))?;
    require_cancelled(sign_jvp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        &cancelled_context,
    ))?;

    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    assert_eq!(optimizer.steps(), [0]);
    assert_eq!(
        values(&backend, &authority, &parameters[0], &live)?,
        parameter_before
    );
    assert_eq!(
        values(&backend, &authority, &zero_target, &live)?,
        [1.0, -2.0]
    );
    Ok(())
}

#[test]
fn resolution_contracts_are_unique_and_sealed_by_their_fixtures()
-> Result<(), Box<dyn std::error::Error>> {
    let owner =
        "comfy-parity-tensor-ops-elementwise-or-runtime-operation-comfy-tensor-op-790cde1ebf17";
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_11")
        .ok_or("part-eleven resolution slice is missing")?;
    assert!(slice.contracts.iter().all(|contract| {
        !contract.rust_signature.contains("&ExecutionContext)")
            && (!contract.rust_signature.contains("ExecutionContext")
                || contract.rust_signature.contains("ExecutionContext<'_>"))
    }));
    assert!(slice.contracts.iter().any(|contract| {
        contract.operation_id == "COMFY-TENSOR-OP-790CDE1EBF17"
            && contract
                .rust_signature
                .contains("NativeAdam::new_with_context_exact_native")
    }));
    assert_eq!(slice.len(), IDS.len());
    let ids = IDS.into_iter().collect::<BTreeSet<_>>();
    assert_eq!(ids.len(), IDS.len());
    let mut overloads = BTreeSet::new();
    let mut digests = BTreeSet::new();
    for contract in slice.contracts {
        assert!(ids.contains(contract.operation_id));
        assert_eq!(contract.owner_task_id, owner);
        assert!(overloads.insert(contract.overload_id));
        assert!(digests.insert(contract.evidence_fixture_sha256));
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .ok_or("workspace root missing")?
            .join(contract.evidence_fixture);
        let bytes = fs::read(path)?;
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            contract.evidence_fixture_sha256
        );
    }
    Ok(())
}
