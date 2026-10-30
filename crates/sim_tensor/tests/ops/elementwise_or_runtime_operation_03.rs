use sha2::{Digest, Sha256};
use sim_tensor::CpuWorkspaceAuthority;
use sim_tensor::{
    CancellationToken, CpuBackend, DType, DeviceId, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES,
    GradientMode, Layout, Scalar, StreamId, Tensor, TensorDescriptor, ViewAccess,
    generated_elementwise_or_runtime_operation_03::{
        ElementwiseOperand, ElementwiseRuntimePartThreeError, TorchArchiveValue,
        add_in_place_with_context_exact_native, clamp_in_place_with_context_exact_native,
        cudnn_convolution_jvp_with_context_exact_native,
        cudnn_convolution_vjp_with_context_exact_native,
        cudnn_convolution_with_context_exact_native, data_ptr_exact_native,
        expm1_jvp_with_context_exact_native, expm1_vjp_with_context_exact_native,
        expm1_with_context_exact_native, floor_jvp_with_context_exact_native,
        floor_vjp_with_context_exact_native, floor_with_context_exact_native,
        greater_with_context_exact_native, no_grad_exact_native,
        real_add_with_context_exact_native, real_lerp_tensor_weight_with_context_exact_native,
        real_multiply_with_context_exact_native, sigmoid_jvp_with_context_exact_native,
        sigmoid_vjp_with_context_exact_native, sigmoid_with_context_exact_native,
        torch_save_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use std::{collections::BTreeSet, fs, path::Path};

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-1ED3CF790B68",
    "COMFY-TENSOR-OP-26EF6B18C684",
    "COMFY-TENSOR-OP-1F39246F0FAD",
    "COMFY-TENSOR-OP-1B60D420F7C7",
    "COMFY-TENSOR-OP-231238FDA88D",
    "COMFY-TENSOR-OP-1C55B11AD08B",
    "COMFY-TENSOR-OP-22DF6A4C26CC",
    "COMFY-TENSOR-OP-2673CE820FAC",
    "COMFY-TENSOR-OP-2464198E16CB",
    "COMFY-TENSOR-OP-1917B7227A5C",
    "COMFY-TENSOR-OP-263D166C9D1F",
    "COMFY-TENSOR-OP-2255F11A43BA",
];

fn backend() -> Result<(CpuBackend, CpuWorkspaceAuthority), Box<dyn std::error::Error>> {
    Ok(CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?)
}

#[test]
fn workspace_authorization_is_exact_bounded_and_failure_atomic_for_part_three()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &authority,
        &[4],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;

    let authorization = authority.authorize_workspace(4)?;
    let context =
        backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancellation);
    greater_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Scalar(Scalar::Float(2.0)),
        &context,
    )?;
    assert_eq!(authorization.peak_bytes(), 4);
    assert_eq!(authorization.in_use_bytes(), 0);

    let mut candidate = input.clone();
    let before = candidate.contiguous_bytes()?.to_vec();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let authorization = authority.authorize_workspace(16)?;
    let context = backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancelled);
    assert!(
        add_in_place_with_context_exact_native(
            &backend,
            &mut candidate,
            ElementwiseOperand::Scalar(Scalar::Float(1.0)),
            Scalar::Float(1.0),
            &context,
        )
        .is_err()
    );
    assert_eq!(candidate.contiguous_bytes()?, before.as_slice());
    assert_eq!(authorization.in_use_bytes(), 0);

    let insufficient = authority.authorize_workspace(3)?;
    let context = backend.execution_context(StreamId::DEFAULT, insufficient.clone(), &cancellation);
    assert!(
        greater_with_context_exact_native(
            &backend,
            &input,
            ElementwiseOperand::Scalar(Scalar::Float(2.0)),
            &context
        )
        .is_err()
    );
    assert_eq!(insufficient.in_use_bytes(), 0);
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

fn upload_real(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    values: &[f32],
    dtype: DType,
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, StreamId::DEFAULT)?;
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend(dtype.encode_scalar(
            Scalar::Float(f64::from(*value)),
            "real-arithmetic-test",
            DeviceId::CPU,
        )?);
    }
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        cancellation,
    );
    Ok(backend.upload_bytes(descriptor, &bytes, &context)?.0)
}

fn real_values(input: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let dtype = input.descriptor().dtype();
    let width = usize::try_from(dtype.byte_width())?;
    input
        .contiguous_bytes()?
        .chunks_exact(width)
        .map(|bytes| match dtype.decode_scalar(bytes)? {
            sim_tensor::DecodedScalar::Real(value) => Ok(value as f32),
            _ => Err("expected a real tensor".into()),
        })
        .collect()
}

#[test]
fn bounded_real_arithmetic_preserves_dtype_broadcasting_and_failure_atomicity()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        let memory_before = backend.memory_snapshot().current_bytes;
        let input = upload_real(
            &backend,
            &authority,
            &[1, 2, 1, 2],
            &[-2.0, 4.0, 3.0, -5.0],
            dtype,
            &cancellation,
        )?;
        let input_bytes = input.contiguous_bytes()?.to_vec();
        let channel_scale = upload_real(
            &backend,
            &authority,
            &[1, 2, 1, 1],
            &[0.5, -2.0],
            dtype,
            &cancellation,
        )?;
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(64)?,
            &cancellation,
        );
        let multiplied = real_multiply_with_context_exact_native(
            &backend,
            &input,
            ElementwiseOperand::Tensor(&channel_scale),
            &context,
        )?;
        assert_eq!(real_values(&multiplied)?, [-1.0, 2.0, -6.0, 10.0]);
        let added = real_add_with_context_exact_native(&backend, &input, &multiplied, &context)?;
        assert_eq!(real_values(&added)?, [-3.0, 6.0, -3.0, 5.0]);
        let scaled = real_multiply_with_context_exact_native(
            &backend,
            &input,
            ElementwiseOperand::Scalar(Scalar::Float(0.25)),
            &context,
        )?;
        assert_eq!(real_values(&scaled)?, [-0.5, 1.0, 0.75, -1.25]);

        let start = upload_real(
            &backend,
            &authority,
            &[1, 3, 1, 2],
            &[0.0, 10.0, 10.0, 20.0, 20.0, 30.0],
            dtype,
            &cancellation,
        )?;
        let end = upload_real(
            &backend,
            &authority,
            &[1, 3, 1, 2],
            &[10.0, 20.0, 20.0, 30.0, 30.0, 40.0],
            dtype,
            &cancellation,
        )?;
        let weight = upload_real(
            &backend,
            &authority,
            &[1, 1, 1, 2],
            &[0.25, 0.75],
            dtype,
            &cancellation,
        )?;
        let lerped = real_lerp_tensor_weight_with_context_exact_native(
            &backend, &start, &end, &weight, &context,
        )?;
        assert_eq!(real_values(&lerped)?, [2.5, 17.5, 12.5, 27.5, 22.5, 37.5]);
        assert_eq!(lerped.descriptor().dtype(), dtype);
        assert_eq!(lerped.descriptor().stream(), StreamId::DEFAULT);
        assert_ne!(lerped.storage_id(), start.storage_id());
        assert_eq!(input.contiguous_bytes()?, input_bytes);
        assert_eq!(context.scratch.in_use_bytes(), 0);

        let required = dtype
            .byte_width()
            .checked_mul(4)
            .ok_or("workspace size overflow")?;
        let insufficient = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(required - 1)?,
            &cancellation,
        );
        assert!(
            real_add_with_context_exact_native(&backend, &input, &multiplied, &insufficient,)
                .is_err()
        );
        assert_eq!(insufficient.scratch.in_use_bytes(), 0);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(0)?,
            &cancelled,
        );
        assert!(matches!(
            real_multiply_with_context_exact_native(
                &backend,
                &input,
                ElementwiseOperand::Tensor(&channel_scale),
                &cancelled_context,
            ),
            Err(ElementwiseRuntimePartThreeError::Cancelled)
        ));
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
        drop(lerped);
        drop(weight);
        drop(end);
        drop(start);
        drop(scaled);
        drop(added);
        drop(multiplied);
        drop(channel_scale);
        drop(input);
        assert_eq!(backend.memory_snapshot().current_bytes, memory_before);
    }
    Ok(())
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

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|candidate| candidate == needle)
}

#[test]
fn resolution_slice_seals_exactly_the_assigned_contracts_and_fixtures()
-> Result<(), Box<dyn std::error::Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_03")
        .ok_or("elementwise/runtime part-three resolution slice is missing")?;
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
            "comfy-parity-tensor-ops-elementwise-or-runtime-operation-comfy-tensor-op-1917b7227a5c"
        );
        let bytes = fs::read(workspace.join(contract.evidence_fixture))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            contract.evidence_fixture_sha256
        );
        let callable = match contract.operation_id {
            "COMFY-TENSOR-OP-1917B7227A5C" => "sigmoid_with_context_exact_native",
            "COMFY-TENSOR-OP-1B60D420F7C7" => "cuda_memory_summary_exact_native",
            "COMFY-TENSOR-OP-1C55B11AD08B" => "floor_with_context_exact_native",
            "COMFY-TENSOR-OP-1ED3CF790B68" => "add_in_place_with_context_exact_native",
            "COMFY-TENSOR-OP-1F39246F0FAD" => "data_ptr_exact_native",
            "COMFY-TENSOR-OP-2255F11A43BA" => "xpu_device_count_exact_native",
            "COMFY-TENSOR-OP-22DF6A4C26CC" => "greater_with_context_exact_native",
            "COMFY-TENSOR-OP-231238FDA88D" => "cudnn_convolution_with_context_exact_native",
            "COMFY-TENSOR-OP-2464198E16CB" => "torch_save_exact_native",
            "COMFY-TENSOR-OP-263D166C9D1F" => "expm1_with_context_exact_native",
            "COMFY-TENSOR-OP-2673CE820FAC" => "no_grad_exact_native",
            "COMFY-TENSOR-OP-26EF6B18C684" => "clamp_in_place_with_context_exact_native",
            _ => return Err("unexpected Task 46 operation identifier".into()),
        };
        assert!(contract.rust_signature.contains(callable));
        if contract.rust_signature.contains("ExecutionContext") {
            assert!(contract.rust_signature.contains("ExecutionContext<'_>"));
        }
    }
    Ok(())
}

#[test]
fn in_place_operations_stage_copy_on_write_and_publish_nothing_when_cancelled()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let mut input = upload_f32(
        &backend,
        &authority,
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let original = input.clone();
    let operand = upload_f32(&backend, &authority, &[2], &[10.0, 20.0], &cancellation)?;
    let original_storage = input.storage_id();
    add_in_place_with_context_exact_native(
        &backend,
        &mut input,
        ElementwiseOperand::Tensor(&operand),
        Scalar::Float(0.5),
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &input, &cancellation)?,
        [6.0, 12.0, 8.0, 14.0]
    );
    assert_eq!(
        values(&backend, &authority, &original, &cancellation)?,
        [1.0, 2.0, 3.0, 4.0]
    );
    assert_ne!(input.storage_id(), original_storage);

    clamp_in_place_with_context_exact_native(
        &backend,
        &mut input,
        Some(Scalar::Float(7.0)),
        Some(Scalar::Float(12.0)),
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &input, &cancellation)?,
        [7.0, 12.0, 8.0, 12.0]
    );
    clamp_in_place_with_context_exact_native(
        &backend,
        &mut input,
        Some(Scalar::Float(5.0)),
        Some(Scalar::Float(2.0)),
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(
        values(&backend, &authority, &input, &cancellation)?,
        [2.0; 4]
    );

    let before = values(&backend, &authority, &input, &cancellation)?;
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(
        add_in_place_with_context_exact_native(
            &backend,
            &mut input,
            ElementwiseOperand::Scalar(Scalar::Float(9.0)),
            Scalar::Float(1.0),
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancelled,
            ),
        )
        .is_err()
    );
    assert_eq!(values(&backend, &authority, &input, &cancellation)?, before);

    let expanding = upload_f32(&backend, &authority, &[3, 2, 2], &[1.0; 12], &cancellation)?;
    assert!(
        add_in_place_with_context_exact_native(
            &backend,
            &mut input,
            ElementwiseOperand::Tensor(&expanding),
            Scalar::Float(1.0),
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancellation,
            ),
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn data_ptr_uses_the_canonical_storage_and_checked_view_offset()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(&backend, &authority, &[3], &[1.0, 2.0, 3.0], &cancellation)?;
    let base = data_ptr_exact_native(&input, &cancellation)?;
    let descriptor = TensorDescriptor::new_strided(
        vec![2],
        vec![1],
        1,
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    let view = input.view(descriptor, ViewAccess::ReadOnly)?;
    assert_eq!(data_ptr_exact_native(&view, &cancellation)?, base + 4);
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(data_ptr_exact_native(&view, &cancelled).is_err());
    Ok(())
}

#[test]
fn greater_uses_canonical_right_aligned_broadcasting() -> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let left = upload_f32(&backend, &authority, &[2, 1], &[1.0, 4.0], &cancellation)?;
    let right = upload_f32(
        &backend,
        &authority,
        &[1, 3],
        &[0.0, 2.0, 5.0],
        &cancellation,
    )?;
    let output = greater_with_context_exact_native(
        &backend,
        &left,
        ElementwiseOperand::Tensor(&right),
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        ),
    )?;
    assert_eq!(output.descriptor().shape(), [2, 3]);
    assert_eq!(output.contiguous_bytes()?, [1, 0, 0, 1, 1, 0]);
    Ok(())
}

#[test]
fn cudnn_source_facade_delegates_forward_vjp_and_jvp_to_native_convolution()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let (backend, authority) = backend()?;
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        &cancellation,
    );
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = cudnn_convolution_with_context_exact_native(
        &input,
        &[1, 1, 2, 2, 2],
        &[2.0],
        &[1, 1, 1, 1, 1],
        vec![0, 0, 0],
        vec![1, 1, 1],
        vec![1, 1, 1],
        1,
        false,
        false,
        true,
        DeviceId::CPU,
        &execution,
    )?;
    assert_eq!(result.shape, [1, 1, 2, 2, 2]);
    assert_eq!(result.values, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    assert!(
        cudnn_convolution_with_context_exact_native(
            &input,
            &[1, 1, 2, 2, 2],
            &[2.0],
            &[1, 1, 1, 1, 1],
            vec![0; 3],
            vec![1; 3],
            vec![1; 3],
            1,
            true,
            false,
            true,
            DeviceId::CPU,
            &execution,
        )
        .is_err()
    );

    let gradient = [1.0; 8];
    let vjp = cudnn_convolution_vjp_with_context_exact_native(
        &input,
        &[1, 1, 2, 2, 2],
        &[2.0],
        &[1, 1, 1, 1, 1],
        &gradient,
        vec![0; 3],
        vec![1; 3],
        vec![1; 3],
        1,
        DeviceId::CPU,
        &execution,
    )?;
    assert_eq!(vjp.input, [2.0; 8]);
    assert_eq!(vjp.weight, [36.0]);
    let jvp = cudnn_convolution_jvp_with_context_exact_native(
        &input,
        &[1.0; 8],
        &[1, 1, 2, 2, 2],
        &[2.0],
        &[0.0],
        &[1, 1, 1, 1, 1],
        vec![0; 3],
        vec![1; 3],
        vec![1; 3],
        1,
        DeviceId::CPU,
        &execution,
    )?;
    assert_eq!(jvp.values, [2.0; 8]);
    Ok(())
}

#[test]
fn unary_operations_preserve_edge_semantics_and_analytical_gradients()
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
    let floor = values(
        &backend,
        &authority,
        &floor_with_context_exact_native(
            &backend,
            &input,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancellation,
            ),
        )?,
        &cancellation,
    )?;
    assert_eq!(floor, [-2.0, -0.0, 0.0, 1.0]);
    assert!(floor[1].is_sign_negative());
    assert_eq!(
        values(
            &backend,
            &authority,
            &floor_vjp_with_context_exact_native(
                &backend,
                &input,
                &backend.execution_context(
                    StreamId::DEFAULT,
                    authority.authorize_workspace(1024 * 1024)?,
                    &cancellation,
                )
            )?,
            &cancellation,
        )?,
        [0.0; 4]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &floor_jvp_with_context_exact_native(
                &backend,
                &input,
                &backend.execution_context(
                    StreamId::DEFAULT,
                    authority.authorize_workspace(1024 * 1024)?,
                    &cancellation,
                )
            )?,
            &cancellation,
        )?,
        [0.0; 4]
    );

    let input = upload_f32(&backend, &authority, &[3], &[-1.0, 0.0, 1.0], &cancellation)?;
    let gradient = upload_f32(&backend, &authority, &[3], &[1.0, 2.0, 3.0], &cancellation)?;
    let sigmoid = values(
        &backend,
        &authority,
        &sigmoid_with_context_exact_native(
            &backend,
            &input,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancellation,
            ),
        )?,
        &cancellation,
    )?;
    assert_eq!(sigmoid[1], 0.5);
    let sigmoid_vjp = values(
        &backend,
        &authority,
        &sigmoid_vjp_with_context_exact_native(
            &backend,
            &input,
            &gradient,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancellation,
            ),
        )?,
        &cancellation,
    )?;
    let sigmoid_jvp = values(
        &backend,
        &authority,
        &sigmoid_jvp_with_context_exact_native(
            &backend,
            &input,
            &gradient,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancellation,
            ),
        )?,
        &cancellation,
    )?;
    assert_eq!(sigmoid_vjp, sigmoid_jvp);
    assert!((sigmoid_vjp[1] - 0.5).abs() < 0.000_001);

    let expm1 = values(
        &backend,
        &authority,
        &expm1_with_context_exact_native(
            &backend,
            &input,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancellation,
            ),
        )?,
        &cancellation,
    )?;
    assert_eq!(expm1[1], 0.0);
    let expm1_vjp = values(
        &backend,
        &authority,
        &expm1_vjp_with_context_exact_native(
            &backend,
            &input,
            &gradient,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancellation,
            ),
        )?,
        &cancellation,
    )?;
    let expm1_jvp = values(
        &backend,
        &authority,
        &expm1_jvp_with_context_exact_native(
            &backend,
            &input,
            &gradient,
            &backend.execution_context(
                StreamId::DEFAULT,
                authority.authorize_workspace(1024 * 1024)?,
                &cancellation,
            ),
        )?,
        &cancellation,
    )?;
    assert_eq!(expm1_vjp, expm1_jvp);
    assert_eq!(expm1_vjp[1], 2.0);
    Ok(())
}

#[test]
fn no_grad_uses_the_canonical_autograd_mode() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(
        no_grad_exact_native(&CancellationToken::default())?,
        GradientMode::NoGrad
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(no_grad_exact_native(&cancelled).is_err());
    Ok(())
}

#[test]
fn torch_save_builds_deterministic_archive_content_without_owning_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let tensor = upload_f32(&backend, &authority, &[2], &[1.0, 2.0], &cancellation)?;
    let value = TorchArchiveValue::Tuple(vec![
        TorchArchiveValue::String("fixture".to_owned()),
        TorchArchiveValue::Tensor(tensor.clone()),
        TorchArchiveValue::Tensor(tensor),
    ]);
    let first = torch_save_exact_native(&value, &cancellation)?;
    let second = torch_save_exact_native(&value, &cancellation)?;
    assert_eq!(first, second);
    assert!(first.starts_with(b"PK\x03\x04"));
    assert!(contains_bytes(&first, b"archive/data.pkl"));
    assert!(contains_bytes(&first, b"archive/data/0"));
    assert!(!contains_bytes(&first, b"archive/data/1"));
    assert!(contains_bytes(&first, b"zed-native-sim-tensor-v1"));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(torch_save_exact_native(&value, &cancelled).is_err());
    Ok(())
}

#[test]
fn every_local_task46_adapter_honors_pre_cancellation_before_validation_or_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let live = CancellationToken::default();
    let input = upload_f32(&backend, &authority, &[2], &[1.0, 2.0], &live)?;
    let mut mutable = input.clone();
    let original_storage = mutable.storage_id();
    let original_bytes = mutable.contiguous_bytes()?.to_vec();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        &cancelled,
    );
    assert!(matches!(
        add_in_place_with_context_exact_native(
            &backend,
            &mut mutable,
            ElementwiseOperand::Scalar(Scalar::Float(1.0)),
            Scalar::Float(1.0),
            &cancelled_context,
        ),
        Err(ElementwiseRuntimePartThreeError::Cancelled)
    ));
    assert!(matches!(
        clamp_in_place_with_context_exact_native(
            &backend,
            &mut mutable,
            None,
            None,
            &cancelled_context,
        ),
        Err(ElementwiseRuntimePartThreeError::Cancelled)
    ));
    assert_eq!(mutable.storage_id(), original_storage);
    assert_eq!(mutable.contiguous_bytes()?, original_bytes);
    assert!(matches!(
        cudnn_convolution_with_context_exact_native(
            &[],
            &[],
            &[],
            &[],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            0,
            true,
            true,
            false,
            DeviceId::CPU,
            &cancelled_context,
        ),
        Err(ElementwiseRuntimePartThreeError::Cancelled)
    ));
    assert!(matches!(
        data_ptr_exact_native(&input, &cancelled),
        Err(ElementwiseRuntimePartThreeError::Cancelled)
    ));
    assert!(matches!(
        floor_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartThreeError::Cancelled)
    ));
    assert!(matches!(
        greater_with_context_exact_native(
            &backend,
            &input,
            ElementwiseOperand::Scalar(Scalar::Float(1.0)),
            &cancelled_context,
        ),
        Err(ElementwiseRuntimePartThreeError::Cancelled)
    ));
    assert!(matches!(
        no_grad_exact_native(&cancelled),
        Err(ElementwiseRuntimePartThreeError::Cancelled)
    ));
    assert!(matches!(
        sigmoid_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartThreeError::Cancelled)
    ));
    assert!(matches!(
        expm1_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartThreeError::Cancelled)
    ));
    assert!(matches!(
        torch_save_exact_native(
            &TorchArchiveValue::String("cancelled".to_owned()),
            &cancelled,
        ),
        Err(ElementwiseRuntimePartThreeError::Cancelled)
    ));
    assert_eq!(cancelled_context.scratch.peak_bytes(), 0);
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}
