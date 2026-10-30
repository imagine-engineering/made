use sha2::{Digest, Sha256};
use sim_tensor::CpuWorkspaceAuthority;
use sim_tensor::{
    AutogradTape, BackendCapabilityMatrix, CancellationToken, CpuBackend, DType, DecodedScalar,
    DeviceId, ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, GradientMode,
    NativeDeviceProperties, StreamId, Tensor, TensorDescriptor, TensorError,
    generated_elementwise_or_runtime_operation_02::acos_with_context_exact_native,
    generated_elementwise_or_runtime_operation_03::{
        ElementwiseOperand, sigmoid_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_05::{
        div_with_context_exact_native, sqrt_with_context_exact_native,
    },
    generated_elementwise_or_runtime_operation_06::{
        DivisionRoundingMode, ElementwiseRuntimePartSixError,
        acos_method_jvp_with_context_exact_native, acos_method_vjp_with_context_exact_native,
        acos_method_with_context_exact_native, bool_method_with_context_exact_native,
        checkpoint_exact_native, div_function_jvp_with_context_exact_native,
        div_function_vjp_with_context_exact_native, div_function_with_context_exact_native,
        inference_mode_exact_native, jit_is_tracing_exact_native,
        round_method_jvp_with_context_exact_native, round_method_vjp_with_context_exact_native,
        round_method_with_context_exact_native, sigmoid_method_jvp_with_context_exact_native,
        sigmoid_method_vjp_with_context_exact_native, sigmoid_method_with_context_exact_native,
        sinc_function_jvp_with_context_exact_native, sinc_function_vjp_with_context_exact_native,
        sinc_function_with_context_exact_native, sqrt_function_jvp_with_context_exact_native,
        sqrt_function_vjp_with_context_exact_native, sqrt_function_with_context_exact_native,
        unique_method_with_context_exact_native, xpu_get_device_properties_exact_native,
    },
    generated_elementwise_or_runtime_operation_10::unique_with_context_exact_native as generic_unique_with_context_exact_native,
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::DeviceKind;
use std::{cell::RefCell, collections::BTreeSet, fs, path::Path};

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-4B4746D5885A",
    "COMFY-TENSOR-OP-472F133627A1",
    "COMFY-TENSOR-OP-4B6925D60ACD",
    "COMFY-TENSOR-OP-51695C0FE8D8",
    "COMFY-TENSOR-OP-4685F95970C6",
    "COMFY-TENSOR-OP-4D087B722410",
    "COMFY-TENSOR-OP-42AC47EA61EE",
    "COMFY-TENSOR-OP-49949B24BFD5",
    "COMFY-TENSOR-OP-4BE7FEEFD9EF",
    "COMFY-TENSOR-OP-4130D690D4B2",
    "COMFY-TENSOR-OP-5278A14360E3",
    "COMFY-TENSOR-OP-48C9CD534224",
];

fn backend() -> Result<(CpuBackend, CpuWorkspaceAuthority), Box<dyn std::error::Error>> {
    Ok(CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?)
}

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

fn upload_i32(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    values: &[i32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::I32, DeviceId::CPU, StreamId::DEFAULT)?;
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    Ok(backend
        .upload_bytes(
            descriptor,
            &bytes,
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

fn decoded_flat(tensor: &Tensor) -> Result<Vec<DecodedScalar>, Box<dyn std::error::Error>> {
    let count = usize::try_from(tensor.descriptor().element_count()?)?;
    let mut values = Vec::with_capacity(count);
    for linear_index in 0..count {
        let mut linear = linear_index;
        let mut indices = vec![0; tensor.descriptor().rank()];
        for (slot, dimension) in indices.iter_mut().zip(tensor.descriptor().shape()).rev() {
            let dimension = usize::try_from(*dimension)?;
            *slot = u64::try_from(linear % dimension)?;
            linear /= dimension;
        }
        values.push(
            tensor
                .descriptor()
                .dtype()
                .decode_scalar(tensor.element_bytes(&indices)?)?,
        );
    }
    Ok(values)
}

#[test]
fn resolution_slice_seals_exactly_the_assigned_contracts_and_fixtures()
-> Result<(), Box<dyn std::error::Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_06")
        .ok_or("elementwise/runtime part-six resolution slice is missing")?;
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
            "comfy-parity-tensor-ops-elementwise-or-runtime-operation-comfy-tensor-op-4130d690d4b2"
        );
        let bytes = fs::read(workspace.join(contract.evidence_fixture))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            contract.evidence_fixture_sha256
        );
        let callable = match contract.operation_id {
            "COMFY-TENSOR-OP-4B4746D5885A" => "acos_method_with_context_exact_native",
            "COMFY-TENSOR-OP-472F133627A1" => "bool_method_with_context_exact_native",
            "COMFY-TENSOR-OP-4B6925D60ACD" => "round_method_with_context_exact_native",
            "COMFY-TENSOR-OP-51695C0FE8D8" => "sigmoid_method_with_context_exact_native",
            "COMFY-TENSOR-OP-4685F95970C6" => "unique_method_with_context_exact_native",
            "COMFY-TENSOR-OP-4D087B722410" => "div_function_with_context_exact_native",
            "COMFY-TENSOR-OP-42AC47EA61EE" => "inference_mode_exact_native",
            "COMFY-TENSOR-OP-49949B24BFD5" => "jit_is_tracing_exact_native",
            "COMFY-TENSOR-OP-4BE7FEEFD9EF" => "sinc_function_with_context_exact_native",
            "COMFY-TENSOR-OP-4130D690D4B2" => "sqrt_function_with_context_exact_native",
            "COMFY-TENSOR-OP-5278A14360E3" => "checkpoint_exact_native",
            "COMFY-TENSOR-OP-48C9CD534224" => "xpu_get_device_properties_exact_native",
            _ => return Err("unexpected Task 49 operation identifier".into()),
        };
        assert!(contract.rust_signature.contains(callable));
        if contract.rust_signature.contains("ExecutionContext") {
            assert!(contract.rust_signature.contains("ExecutionContext<'_>"));
        }
        if contract.operation_id == "COMFY-TENSOR-OP-4D087B722410" {
            assert!(contract.rust_signature.contains("ElementwiseOperand<'_>"));
        }
    }
    Ok(())
}

#[test]
fn method_and_function_facades_delegate_to_the_canonical_math_owners()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let unit = upload_f32(&backend, &authority, &[3], &[-0.5, 0.0, 0.5], &cancellation)?;
    assert_eq!(
        acos_method_with_context_exact_native(&backend, &unit, &execution)?.contiguous_bytes()?,
        acos_with_context_exact_native(
            &backend,
            &unit,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancellation,
            )
        )?
        .contiguous_bytes()?
    );
    assert_eq!(
        sigmoid_method_with_context_exact_native(&backend, &unit, &execution)?
            .contiguous_bytes()?,
        sigmoid_with_context_exact_native(
            &backend,
            &unit,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancellation,
            )
        )?
        .contiguous_bytes()?
    );

    let numerator = upload_f32(&backend, &authority, &[2], &[8.0, 9.0], &cancellation)?;
    let facade_division = div_function_with_context_exact_native(
        &backend,
        &numerator,
        ElementwiseOperand::Scalar(sim_tensor::Scalar::Float(2.0)),
        None,
        &execution,
    )?;
    let canonical_division = div_with_context_exact_native(
        &backend,
        &numerator,
        ElementwiseOperand::Scalar(sim_tensor::Scalar::Float(2.0)),
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        facade_division.contiguous_bytes()?,
        canonical_division.contiguous_bytes()?
    );
    assert_eq!(
        sqrt_function_with_context_exact_native(&backend, &numerator, &execution)?
            .contiguous_bytes()?,
        sqrt_with_context_exact_native(
            &backend,
            &numerator,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancellation,
            )
        )?
        .contiguous_bytes()?
    );
    Ok(())
}

#[test]
fn workspace_authority_is_exact_bounded_and_converges_for_round()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &authority,
        &[4],
        &[1.25, 1.35, -1.25, -1.35],
        &cancellation,
    )?;
    let exact = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(16)?,
        &cancellation,
    );
    round_method_with_context_exact_native(&backend, &input, 1, &exact)?;
    assert_eq!(exact.scratch.peak_bytes(), 16);
    assert_eq!(exact.scratch.in_use_bytes(), 0);

    let insufficient = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(15)?,
        &cancellation,
    );
    assert!(round_method_with_context_exact_native(&backend, &input, 1, &insufficient).is_err());
    assert_eq!(insufficient.scratch.in_use_bytes(), 0);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(16)?,
        &cancelled,
    );
    assert!(
        unique_method_with_context_exact_native(
            &backend,
            &input,
            true,
            true,
            true,
            &cancelled_context,
        )
        .is_err()
    );
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn round_and_sinc_use_canonical_cpu_primitives_with_exact_gradients()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let input = upload_f32(
        &backend,
        &authority,
        &[5],
        &[0.5, 1.5, 2.5, -0.5, -1.5],
        &cancellation,
    )?;
    let rounded = round_method_with_context_exact_native(&backend, &input, 0, &execution)?;
    assert_eq!(
        values(&backend, &authority, &rounded, &cancellation)?,
        [0.0, 2.0, 2.0, -0.0, -2.0]
    );
    let gradient = upload_f32(&backend, &authority, &[5], &[1.0; 5], &cancellation)?;
    assert_eq!(
        values(
            &backend,
            &authority,
            &round_method_vjp_with_context_exact_native(&backend, &input, &gradient, &execution)?,
            &cancellation,
        )?,
        [0.0; 5]
    );
    let decimal = upload_f32(&backend, &authority, &[2], &[1.25, 1.35], &cancellation)?;
    assert_eq!(
        values(
            &backend,
            &authority,
            &round_method_with_context_exact_native(&backend, &decimal, 1, &execution)?,
            &cancellation,
        )?,
        [1.2, 1.4]
    );

    let sinc_input = upload_f32(&backend, &authority, &[3], &[0.0, 0.5, 1.0], &cancellation)?;
    let sinc = values(
        &backend,
        &authority,
        &sinc_function_with_context_exact_native(&backend, &sinc_input, &execution)?,
        &cancellation,
    )?;
    assert_eq!(sinc[0], 1.0);
    assert!((sinc[1] - 2.0 / std::f32::consts::PI).abs() < 1e-6);
    assert!(sinc[2].abs() < 1e-6);
    let sinc_gradient = values(
        &backend,
        &authority,
        &sinc_function_vjp_with_context_exact_native(
            &backend,
            &sinc_input,
            &upload_f32(&backend, &authority, &[3], &[1.0; 3], &cancellation)?,
            &execution,
        )?,
        &cancellation,
    )?;
    assert_eq!(sinc_gradient[0], 0.0);
    assert!((sinc_gradient[1] + 4.0 / std::f32::consts::PI).abs() < 1e-6);
    Ok(())
}

#[test]
fn bool_and_unique_preserve_dtype_order_inverse_counts_and_cancellation()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let booleans = bool_method_with_context_exact_native(
        &backend,
        &upload_f32(
            &backend,
            &authority,
            &[3],
            &[0.0, -1.0, f32::NAN],
            &cancellation,
        )?,
        &execution,
    )?;
    assert_eq!(
        decoded_flat(&booleans)?,
        [
            DecodedScalar::Boolean(false),
            DecodedScalar::Boolean(true),
            DecodedScalar::Boolean(true),
        ]
    );

    let input = upload_i32(&backend, &authority, &[5], &[3, 1, 3, -2, 1], &cancellation)?;
    let unique =
        unique_method_with_context_exact_native(&backend, &input, true, true, true, &execution)?;
    assert_eq!(
        decoded_flat(&unique.values)?,
        [
            DecodedScalar::Signed(-2),
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(3),
        ]
    );
    assert_eq!(
        decoded_flat(
            unique
                .inverse_indices
                .as_ref()
                .ok_or("unique inverse indices are missing")?
        )?,
        [
            DecodedScalar::Signed(2),
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(2),
            DecodedScalar::Signed(0),
            DecodedScalar::Signed(1),
        ]
    );
    assert_eq!(
        decoded_flat(unique.counts.as_ref().ok_or("unique counts are missing")?)?,
        [
            DecodedScalar::Signed(1),
            DecodedScalar::Signed(2),
            DecodedScalar::Signed(2),
        ]
    );
    let generic = generic_unique_with_context_exact_native(
        &backend, &input, false, true, true, None, &execution,
    )?;
    assert_eq!(
        generic.values.contiguous_bytes()?,
        unique.values.contiguous_bytes()?
    );
    assert_eq!(
        generic
            .inverse_indices
            .as_ref()
            .ok_or("generic unique inverse indices are missing")?
            .contiguous_bytes()?,
        unique
            .inverse_indices
            .as_ref()
            .ok_or("method unique inverse indices are missing")?
            .contiguous_bytes()?
    );
    assert_eq!(
        generic
            .counts
            .as_ref()
            .ok_or("generic unique counts are missing")?
            .contiguous_bytes()?,
        unique
            .counts
            .as_ref()
            .ok_or("method unique counts are missing")?
            .contiguous_bytes()?
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_execution = context(&backend, &authority, &cancelled)?;
    assert!(
        unique_method_with_context_exact_native(
            &backend,
            &input,
            true,
            true,
            true,
            &cancelled_execution,
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn inference_tracing_checkpoint_and_xpu_properties_use_explicit_canonical_state()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let mode = inference_mode_exact_native(true, GradientMode::Enabled, &cancellation)?;
    assert_eq!(mode, GradientMode::Inference);
    assert_eq!(AutogradTape::new(mode).mode(), GradientMode::Inference);
    assert_eq!(
        inference_mode_exact_native(false, GradientMode::NoGrad, &cancellation)?,
        GradientMode::NoGrad
    );
    assert!(!jit_is_tracing_exact_native(&cancellation)?);

    let input = upload_f32(&backend, &authority, &[2], &[2.0, 3.0], &cancellation)?;
    let modes = RefCell::new(Vec::new());
    let execution = checkpoint_exact_native(
        std::slice::from_ref(&input),
        true,
        &cancellation,
        |inputs, mode, token| {
            token.check()?;
            modes.borrow_mut().push(mode);
            Ok(inputs.to_vec())
        },
    )?;
    assert!(execution.use_reentrant());
    assert_eq!(execution.outputs().len(), 1);
    let replayed = execution.recompute_exact_native(&cancellation, |inputs, mode, token| {
        token.check()?;
        modes.borrow_mut().push(mode);
        Ok(inputs.to_vec())
    })?;
    assert_eq!(replayed[0].contiguous_bytes()?, input.contiguous_bytes()?);
    assert_eq!(
        modes.into_inner(),
        [GradientMode::NoGrad, GradientMode::Enabled]
    );

    let xpu = DeviceId::new(DeviceKind::Xpu, 3);
    let properties = NativeDeviceProperties::new(
        xpu,
        "native-xpu-fixture",
        16 * 1024 * 1024,
        2,
        1,
        Some("xe2".to_owned()),
        true,
    )?;
    let matrix = BackendCapabilityMatrix::new_with_properties(
        xpu,
        Vec::new(),
        Vec::new(),
        Some(properties.clone()),
    )?;
    assert_eq!(
        xpu_get_device_properties_exact_native(&matrix, xpu, &cancellation)?,
        properties
    );
    assert!(xpu_get_device_properties_exact_native(&matrix, DeviceId::CPU, &cancellation).is_err());
    Ok(())
}

#[test]
fn every_task49_adapter_honors_pre_cancellation_before_validation_or_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let live = CancellationToken::default();
    let input = upload_f32(&backend, &authority, &[1], &[0.5], &live)?;
    let input_bytes = input.contiguous_bytes()?.to_vec();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = context(&backend, &authority, &cancelled)?;
    let scalar = ElementwiseOperand::Scalar(sim_tensor::Scalar::Float(2.0));

    assert!(matches!(
        acos_method_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        acos_method_vjp_with_context_exact_native(&backend, &input, &input, &cancelled_context),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        acos_method_jvp_with_context_exact_native(&backend, &input, &input, &cancelled_context),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        bool_method_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        sigmoid_method_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        sigmoid_method_vjp_with_context_exact_native(&backend, &input, &input, &cancelled_context),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        sigmoid_method_jvp_with_context_exact_native(&backend, &input, &input, &cancelled_context),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        round_method_with_context_exact_native(&backend, &input, i32::MAX, &cancelled_context),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        round_method_vjp_with_context_exact_native(&backend, &input, &input, &cancelled_context),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        round_method_jvp_with_context_exact_native(&backend, &input, &input, &cancelled_context),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        unique_method_with_context_exact_native(
            &backend,
            &input,
            true,
            true,
            true,
            &cancelled_context
        ),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        div_function_with_context_exact_native(
            &backend,
            &input,
            scalar,
            Some(DivisionRoundingMode::Floor),
            &cancelled_context
        ),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        div_function_vjp_with_context_exact_native(
            &backend,
            &input,
            scalar,
            Some(DivisionRoundingMode::Trunc),
            &input,
            &cancelled_context
        ),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        div_function_jvp_with_context_exact_native(
            &backend,
            &input,
            scalar,
            None,
            &input,
            Some(&input),
            &cancelled_context
        ),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        inference_mode_exact_native(true, GradientMode::Enabled, &cancelled),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        jit_is_tracing_exact_native(&cancelled),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        sinc_function_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        sinc_function_vjp_with_context_exact_native(&backend, &input, &input, &cancelled_context),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        sinc_function_jvp_with_context_exact_native(&backend, &input, &input, &cancelled_context),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        sqrt_function_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        sqrt_function_vjp_with_context_exact_native(&backend, &input, &input, &cancelled_context),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(matches!(
        sqrt_function_jvp_with_context_exact_native(&backend, &input, &input, &cancelled_context),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));

    let forward_called = RefCell::new(false);
    assert!(matches!(
        checkpoint_exact_native(std::slice::from_ref(&input), true, &cancelled, |_, _, _| {
            *forward_called.borrow_mut() = true;
            Ok(Vec::new())
        }),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(!*forward_called.borrow());
    let execution =
        checkpoint_exact_native(std::slice::from_ref(&input), true, &live, |inputs, _, _| {
            Ok(inputs.to_vec())
        })?;
    let replay_called = RefCell::new(false);
    assert!(matches!(
        execution.recompute_exact_native(&cancelled, |_, _, _| {
            *replay_called.borrow_mut() = true;
            Ok(Vec::new())
        }),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert!(!*replay_called.borrow());

    let cpu_matrix = BackendCapabilityMatrix::for_native_device(DeviceId::CPU)?;
    assert!(matches!(
        xpu_get_device_properties_exact_native(&cpu_matrix, DeviceId::CPU, &cancelled),
        Err(ElementwiseRuntimePartSixError::Cancelled)
    ));
    assert_eq!(cancelled_context.scratch.peak_bytes(), 0);
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    assert_eq!(input.contiguous_bytes()?, input_bytes);
    Ok(())
}
