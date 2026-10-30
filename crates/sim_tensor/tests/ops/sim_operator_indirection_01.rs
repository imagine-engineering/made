use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, Layout, StreamId, TensorDescriptor, ViewAccess,
    generated_accelerated_attention_kernel_01::{
        AttentionKernelKind, AttentionKernelRequest, AttentionLayout, AttentionShape,
        CheckedAttentionInvocation,
    },
    generated_sim_operator_indirection_01::{
        ConvolutionGeometry, OperatorIndirectionError, cast_to_with_context_exact_native,
        convolution_jvp_with_context_exact_native, convolution_vjp_with_context_exact_native,
        convolution_with_context_exact_native, linear_jvp_with_context_exact_native,
        linear_vjp_with_context_exact_native, linear_with_context_exact_native,
        scaled_dot_product_attention_jvp_with_context_exact_native,
        scaled_dot_product_attention_vjp_with_context_exact_native,
        scaled_dot_product_attention_with_context_exact_native,
        tensor_to_f32_with_context_exact_native,
    },
};
use sim_types::DeviceKind;
use std::{collections::BTreeSet, fs, path::Path};

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-97205767AA40",
    "COMFY-TENSOR-OP-23DA7F686728",
    "COMFY-TENSOR-OP-56B106D5BEE7",
    "COMFY-TENSOR-OP-4B62764DCD01",
    "COMFY-TENSOR-OP-6CF91D19480B",
    "COMFY-TENSOR-OP-227F5D04687A",
    "COMFY-TENSOR-OP-6F126397E86F",
    "COMFY-TENSOR-OP-5D8E418C8374",
    "COMFY-TENSOR-OP-7ADDDB2261D6",
    "COMFY-TENSOR-OP-4C30712EC2F7",
    "COMFY-TENSOR-OP-5EAFEF13DE9D",
    "COMFY-TENSOR-OP-86BEA4A2DC25",
];

fn backend() -> Result<(CpuBackend, CpuWorkspaceAuthority), Box<dyn std::error::Error>> {
    Ok(CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?)
}

fn upload_f32(
    backend: &CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<sim_tensor::Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

#[test]
fn resolution_slice_seals_exactly_the_assigned_contracts_and_fixtures()
-> Result<(), Box<dyn std::error::Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "sim_operator_indirection_01")
        .ok_or("operator-indirection resolution slice is missing")?;
    assert_eq!(slice.len(), IDS.len());
    let actual = slice
        .iter()
        .map(|contract| contract.operation_id)
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, IDS.into_iter().collect());
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is missing")?;
    for contract in slice.iter() {
        assert_eq!(
            contract.owner_task_id,
            "comfy-parity-tensor-ops-comfy-operator-indirection-comfy-tensor-op-227f5d04687a"
        );
        let bytes = fs::read(workspace.join(contract.evidence_fixture))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            contract.evidence_fixture_sha256
        );
        match contract.operation_id {
            "COMFY-TENSOR-OP-97205767AA40" => assert!(
                contract
                    .rust_signature
                    .contains("NativeModule::cast_bias_weight_with_context_exact_native")
            ),
            "COMFY-TENSOR-OP-23DA7F686728" => assert!(
                contract
                    .rust_signature
                    .contains("cast_modules_with_vbar_with_context_exact_native")
            ),
            _ => {}
        }
        assert!(
            !contract
                .rust_signature
                .contains("cast_bias_weight_exact_native")
        );
        assert!(
            !contract
                .rust_signature
                .contains("cast_modules_with_vbar_exact_native")
        );
    }
    Ok(())
}

#[test]
fn cast_uses_canonical_codecs_alias_rules_and_typed_device_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = backend()?;
    let cancellation = CancellationToken::default();
    let upload_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(0)?,
        &cancellation,
    );
    let input = upload_f32(&backend, &[4], &[-1.25, 0.0, 1.5, 31.75], &upload_context)?;
    let alias_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(0)?,
        &cancellation,
    );
    let alias = cast_to_with_context_exact_native(
        &backend,
        &input,
        DType::F32,
        DeviceId::CPU,
        false,
        false,
        &alias_context,
    )?;
    assert_eq!(alias.storage_id(), input.storage_id());
    let copied = cast_to_with_context_exact_native(
        &backend,
        &input,
        DType::F32,
        DeviceId::CPU,
        false,
        true,
        &alias_context,
    )?;
    assert_ne!(copied.storage_id(), input.storage_id());
    let half_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(4 * DType::F16.byte_width())?,
        &cancellation,
    );
    let half = cast_to_with_context_exact_native(
        &backend,
        &input,
        DType::F16,
        DeviceId::CPU,
        true,
        true,
        &half_context,
    )?;
    let round_trip_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(4 * DType::F32.byte_width())?,
        &cancellation,
    );
    let round_trip = tensor_to_f32_with_context_exact_native(&backend, &half, &round_trip_context)?;
    assert_eq!(round_trip, vec![-1.25, 0.0, 1.5, 31.75]);

    let matrix = upload_f32(
        &backend,
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &upload_context,
    )?;
    let transposed = matrix.view(
        TensorDescriptor::new_strided(
            vec![3, 2],
            vec![1, 3],
            0,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?,
        ViewAccess::ReadOnly,
    )?;
    let transposed_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(6 * DType::F32.byte_width())?,
        &cancellation,
    );
    assert_eq!(
        tensor_to_f32_with_context_exact_native(&backend, &transposed, &transposed_context)?,
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );

    let complex_descriptor =
        TensorDescriptor::contiguous(vec![1], DType::Complex64, DeviceId::CPU, StreamId::DEFAULT)?;
    let mut complex_bytes = 3.5_f32.to_ne_bytes().to_vec();
    complex_bytes.extend_from_slice(&(-2.0_f32).to_ne_bytes());
    let complex = backend
        .upload_bytes(complex_descriptor, &complex_bytes, &upload_context)?
        .0;
    let real_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(DType::F32.byte_width())?,
        &cancellation,
    );
    let real = cast_to_with_context_exact_native(
        &backend,
        &complex,
        DType::F32,
        DeviceId::CPU,
        false,
        true,
        &real_context,
    )?;
    assert_eq!(
        tensor_to_f32_with_context_exact_native(&backend, &real, &alias_context)?,
        vec![3.5]
    );

    assert!(matches!(
        cast_to_with_context_exact_native(
            &backend,
            &input,
            DType::F32,
            DeviceId::new(DeviceKind::Cuda, 0),
            false,
            false,
            &alias_context,
        ),
        Err(OperatorIndirectionError::UnsupportedDevice { .. })
    ));
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(4 * DType::F16.byte_width())?,
        &cancelled,
    );
    assert!(matches!(
        cast_to_with_context_exact_native(
            &backend,
            &input,
            DType::F16,
            DeviceId::CPU,
            false,
            true,
            &cancelled_context,
        ),
        Err(OperatorIndirectionError::Cancelled)
    ));
    Ok(())
}

#[test]
fn cast_staging_uses_exact_caller_workspace_and_converges() -> Result<(), Box<dyn std::error::Error>>
{
    let (backend, workspace_authority) = backend()?;
    let cancellation = CancellationToken::default();
    let upload_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(0)?,
        &cancellation,
    );
    let input = upload_f32(&backend, &[4], &[-1.25, 0.0, 1.5, 31.75], &upload_context)?;
    let original = input.contiguous_bytes()?.to_vec();
    let required = 4 * DType::F16.byte_width();

    let exact = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(required)?,
        &cancellation,
    );
    let output = cast_to_with_context_exact_native(
        &backend,
        &input,
        DType::F16,
        DeviceId::CPU,
        false,
        true,
        &exact,
    )?;
    assert_eq!(output.descriptor().dtype(), DType::F16);
    assert_eq!(exact.scratch.peak_bytes(), required);
    assert_eq!(exact.scratch.in_use_bytes(), 0);

    let insufficient = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(required - 1)?,
        &cancellation,
    );
    assert!(
        cast_to_with_context_exact_native(
            &backend,
            &input,
            DType::F16,
            DeviceId::CPU,
            false,
            true,
            &insufficient,
        )
        .is_err()
    );
    assert_eq!(insufficient.scratch.in_use_bytes(), 0);
    assert_eq!(input.contiguous_bytes()?, original);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(required)?,
        &cancelled,
    );
    assert!(
        cast_to_with_context_exact_native(
            &backend,
            &input,
            DType::F16,
            DeviceId::CPU,
            false,
            true,
            &cancelled_context,
        )
        .is_err()
    );
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    assert_eq!(input.contiguous_bytes()?, original);
    Ok(())
}

#[test]
fn cast_staging_obeys_backend_capacity_without_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) =
        CpuWorkspaceAuthority::create_backend(4 * DType::F32.byte_width())?;
    let cancellation = CancellationToken::default();
    let upload_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(0)?,
        &cancellation,
    );
    let input = upload_f32(&backend, &[4], &[-1.25, 0.0, 1.5, 31.75], &upload_context)?;
    let original = input.contiguous_bytes()?.to_vec();
    let required = 4 * DType::F16.byte_width();
    let context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(required)?,
        &cancellation,
    );
    assert!(matches!(
        cast_to_with_context_exact_native(
            &backend,
            &input,
            DType::F16,
            DeviceId::CPU,
            false,
            true,
            &context,
        ),
        Err(OperatorIndirectionError::Tensor(
            sim_tensor::TensorError::AllocationFailed { .. }
        ))
    ));
    assert_eq!(context.scratch.in_use_bytes(), 0);
    assert_eq!(input.contiguous_bytes()?, original);
    assert_eq!(backend.memory_snapshot().current_bytes, 16);
    Ok(())
}

#[test]
fn linear_forward_vjp_and_jvp_obey_the_adjoint_identity() -> Result<(), Box<dyn std::error::Error>>
{
    let cancellation = CancellationToken::default();
    let (backend, workspace_authority) = backend()?;
    let caller_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(0)?,
        &cancellation,
    );
    let input = [1.0, -2.0, 0.5, 3.0];
    let weight = [0.5, 1.0, -1.0, 2.0, 0.25, -0.5];
    let bias = [0.25, -0.75, 1.0];
    let forward = linear_with_context_exact_native(
        &input,
        &[2, 2],
        &weight,
        &[3, 2],
        Some(&bias),
        DeviceId::CPU,
        &caller_context,
    )?;
    assert_eq!(forward.shape, vec![2, 3]);
    assert_eq!(forward.values, vec![-1.25, -5.75, 2.25, 3.5, 4.75, -0.375]);
    let input_tangent = [0.1, -0.2, 0.3, -0.4];
    let weight_tangent = [0.2, 0.1, -0.3, 0.4, 0.5, -0.2];
    let bias_tangent = [0.05, -0.1, 0.2];
    let upstream = [0.5, -1.0, 0.25, 0.75, 0.4, -0.3];
    let jvp = linear_jvp_with_context_exact_native(
        &input,
        &input_tangent,
        &[2, 2],
        &weight,
        &weight_tangent,
        &[3, 2],
        Some(&bias),
        Some(&bias_tangent),
        DeviceId::CPU,
        &caller_context,
    )?;
    let vjp = linear_vjp_with_context_exact_native(
        &input,
        &[2, 2],
        &weight,
        &[3, 2],
        Some(&bias),
        &upstream,
        DeviceId::CPU,
        &caller_context,
    )?;
    let lhs = dot(&jvp.values, &upstream);
    let rhs = dot(&input_tangent, &vjp.input)
        + dot(&weight_tangent, &vjp.weight)
        + dot(
            &bias_tangent,
            vjp.bias.as_deref().ok_or("bias VJP is missing")?,
        );
    assert_close(lhs, rhs, 0.00001);

    let empty = linear_with_context_exact_native(
        &[],
        &[0, 2],
        &weight,
        &[3, 2],
        Some(&bias),
        DeviceId::CPU,
        &caller_context,
    )?;
    assert_eq!(empty.shape, vec![0, 3]);
    assert!(empty.values.is_empty());
    assert!(matches!(
        linear_jvp_with_context_exact_native(
            &input,
            &input_tangent,
            &[2, 2],
            &weight,
            &weight_tangent,
            &[3, 2],
            None,
            Some(&bias_tangent),
            DeviceId::CPU,
            &caller_context,
        ),
        Err(OperatorIndirectionError::Invalid(_))
    ));
    assert!(matches!(
        linear_with_context_exact_native(
            &input,
            &[2, 2],
            &weight,
            &[3, 2],
            Some(&bias),
            DeviceId::new(DeviceKind::Cuda, 0),
            &caller_context,
        ),
        Err(OperatorIndirectionError::UnsupportedDevice { .. })
    ));
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(0)?,
        &cancelled,
    );
    assert!(matches!(
        linear_with_context_exact_native(
            &input,
            &[2, 2],
            &weight,
            &[3, 2],
            Some(&bias),
            DeviceId::CPU,
            &cancelled_context,
        ),
        Err(OperatorIndirectionError::Cancelled)
    ));
    Ok(())
}

#[test]
fn convolution_supports_grouped_three_dimensional_and_transposed_geometry()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let (backend, workspace_authority) = backend()?;
    let caller_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(0)?,
        &cancellation,
    );
    let conv2d =
        ConvolutionGeometry::new(2, vec![1, 1], vec![0, 0], vec![1, 1], 1, false, vec![0, 0])?;
    assert_eq!(
        conv2d.checked_output_shape(&[1, 1, 3, 3], &[1, 1, 2, 2], None)?,
        [1, 1, 2, 2]
    );
    assert!(matches!(
        conv2d.checked_output_shape(&[1, 1, 3, 3], &[1, 1, 2, 2], Some(&[2])),
        Err(OperatorIndirectionError::Invalid(
            "convolution bias must match output channels"
        ))
    ));
    let output = convolution_with_context_exact_native(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 1, 3, 3],
        &[1.0; 4],
        &[1, 1, 2, 2],
        None,
        &conv2d,
        DeviceId::CPU,
        &caller_context,
    )?;
    assert_eq!(output.shape, vec![1, 1, 2, 2]);
    assert_eq!(output.values, vec![12.0, 16.0, 24.0, 28.0]);
    let empty = convolution_with_context_exact_native(
        &[],
        &[0, 1, 3, 3],
        &[1.0; 4],
        &[1, 1, 2, 2],
        None,
        &conv2d,
        DeviceId::CPU,
        &caller_context,
    )?;
    assert_eq!(empty.shape, vec![0, 1, 2, 2]);
    assert!(empty.values.is_empty());

    let conv3d =
        ConvolutionGeometry::new(3, vec![1; 3], vec![0; 3], vec![1; 3], 1, false, vec![0; 3])?;
    let output3d = convolution_with_context_exact_native(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 1, 2, 2, 2],
        &[1.0; 8],
        &[1, 1, 2, 2, 2],
        Some(&[0.5]),
        &conv3d,
        DeviceId::CPU,
        &caller_context,
    )?;
    assert_eq!(output3d.values, vec![36.5]);

    let transpose = ConvolutionGeometry::new(1, vec![2], vec![0], vec![1], 1, true, vec![0])?;
    let transposed = convolution_with_context_exact_native(
        &[1.0, 2.0],
        &[1, 1, 2],
        &[1.0, 2.0],
        &[1, 1, 2],
        None,
        &transpose,
        DeviceId::CPU,
        &caller_context,
    )?;
    assert_eq!(transposed.shape, vec![1, 1, 4]);
    assert_eq!(transposed.values, vec![1.0, 2.0, 2.0, 4.0]);

    let transpose2d =
        ConvolutionGeometry::new(2, vec![1, 1], vec![0, 0], vec![1, 1], 1, true, vec![0, 0])?;
    let transposed2d = convolution_with_context_exact_native(
        &[2.0],
        &[1, 1, 1, 1],
        &[1.0, 2.0, 3.0, 4.0],
        &[1, 1, 2, 2],
        None,
        &transpose2d,
        DeviceId::CPU,
        &caller_context,
    )?;
    assert_eq!(transposed2d.shape, vec![1, 1, 2, 2]);
    assert_eq!(transposed2d.values, vec![2.0, 4.0, 6.0, 8.0]);

    let grouped = ConvolutionGeometry::new(1, vec![1], vec![0], vec![1], 2, false, vec![0])?;
    let grouped_output = convolution_with_context_exact_native(
        &[1.0, 2.0, 3.0, 4.0],
        &[1, 2, 2],
        &[2.0, 3.0],
        &[2, 1, 1],
        None,
        &grouped,
        DeviceId::CPU,
        &caller_context,
    )?;
    assert_eq!(grouped_output.values, vec![2.0, 4.0, 9.0, 12.0]);
    Ok(())
}

#[test]
fn every_assigned_convolution_geometry_preserves_cancellation_and_device_boundaries()
-> Result<(), Box<dyn std::error::Error>> {
    let active = CancellationToken::default();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let (backend, workspace_authority) = backend()?;
    let active_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(0)?,
        &active,
    );
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(0)?,
        &cancelled,
    );
    let cases = [
        (
            "conv2d",
            vec![1.0; 9],
            vec![1, 1, 3, 3],
            vec![1.0; 4],
            vec![1, 1, 2, 2],
            ConvolutionGeometry::new(2, vec![1, 1], vec![0, 0], vec![1, 1], 1, false, vec![0, 0])?,
        ),
        (
            "conv3d",
            vec![1.0; 8],
            vec![1, 1, 2, 2, 2],
            vec![1.0; 8],
            vec![1, 1, 2, 2, 2],
            ConvolutionGeometry::new(3, vec![1; 3], vec![0; 3], vec![1; 3], 1, false, vec![0; 3])?,
        ),
        (
            "conv_transpose1d",
            vec![1.0, 2.0],
            vec![1, 1, 2],
            vec![1.0, 2.0],
            vec![1, 1, 2],
            ConvolutionGeometry::new(1, vec![2], vec![0], vec![1], 1, true, vec![0])?,
        ),
        (
            "conv_transpose2d",
            vec![2.0],
            vec![1, 1, 1, 1],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 1, 2, 2],
            ConvolutionGeometry::new(2, vec![1, 1], vec![0, 0], vec![1, 1], 1, true, vec![0, 0])?,
        ),
    ];
    for (name, input, input_shape, weight, weight_shape, geometry) in cases {
        assert!(
            matches!(
                convolution_with_context_exact_native(
                    &input,
                    &input_shape,
                    &weight,
                    &weight_shape,
                    None,
                    &geometry,
                    DeviceId::new(DeviceKind::Cuda, 0),
                    &active_context,
                ),
                Err(OperatorIndirectionError::UnsupportedDevice { .. })
            ),
            "{name} accepted an uncertified device"
        );
        assert!(
            matches!(
                convolution_with_context_exact_native(
                    &input,
                    &input_shape,
                    &weight,
                    &weight_shape,
                    None,
                    &geometry,
                    DeviceId::CPU,
                    &cancelled_context,
                ),
                Err(OperatorIndirectionError::Cancelled)
            ),
            "{name} did not preserve cancellation"
        );
    }
    Ok(())
}

#[test]
fn convolution_vjp_and_jvp_share_one_checked_connection_map()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let (backend, workspace_authority) = backend()?;
    let caller_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(0)?,
        &cancellation,
    );
    let geometry =
        ConvolutionGeometry::new(2, vec![1, 1], vec![1, 1], vec![1, 1], 1, false, vec![0, 0])?;
    let input = [1.0, -2.0, 0.5, 3.0];
    let input_tangent = [0.1, 0.2, -0.3, 0.4];
    let weight = [0.5, -1.0, 2.0, 0.25];
    let weight_tangent = [-0.2, 0.3, 0.1, -0.4];
    let bias = [0.25];
    let bias_tangent = [-0.15];
    let forward = convolution_with_context_exact_native(
        &input,
        &[1, 1, 2, 2],
        &weight,
        &[1, 1, 2, 2],
        Some(&bias),
        &geometry,
        DeviceId::CPU,
        &caller_context,
    )?;
    let upstream = vec![0.2, -0.1, 0.3, 0.5, -0.4, 0.7, 0.1, -0.2, 0.6];
    assert_eq!(forward.values.len(), upstream.len());
    let jvp = convolution_jvp_with_context_exact_native(
        &input,
        &input_tangent,
        &[1, 1, 2, 2],
        &weight,
        &weight_tangent,
        &[1, 1, 2, 2],
        Some(&bias),
        Some(&bias_tangent),
        &geometry,
        DeviceId::CPU,
        &caller_context,
    )?;
    let vjp = convolution_vjp_with_context_exact_native(
        &input,
        &[1, 1, 2, 2],
        &weight,
        &[1, 1, 2, 2],
        Some(&bias),
        &upstream,
        &geometry,
        DeviceId::CPU,
        &caller_context,
    )?;
    let lhs = dot(&jvp.values, &upstream);
    let rhs = dot(&input_tangent, &vjp.input)
        + dot(&weight_tangent, &vjp.weight)
        + dot(
            &bias_tangent,
            vjp.bias.as_deref().ok_or("bias VJP is missing")?,
        );
    assert_close(lhs, rhs, 0.00001);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(0)?,
        &cancelled,
    );
    assert!(matches!(
        convolution_with_context_exact_native(
            &input,
            &[1, 1, 2, 2],
            &weight,
            &[1, 1, 2, 2],
            Some(&bias),
            &geometry,
            DeviceId::CPU,
            &cancelled_context
        ),
        Err(OperatorIndirectionError::Cancelled)
    ));
    Ok(())
}

#[test]
fn attention_facade_delegates_forward_and_gradients_to_checked_invocation()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let request = AttentionKernelRequest {
        kind: AttentionKernelKind::ReferenceSdp,
        device: DeviceId::CPU,
        layout: AttentionLayout::Nhd,
        shape: AttentionShape {
            batch: 1,
            query_tokens: 2,
            key_tokens: 2,
            heads: 1,
            head_dimension: 2,
            value_dimension: 2,
        },
        scale: Some(1.0),
        causal: false,
        dropout_probability: 0.0,
    };
    let query = [1.0, 0.0, 0.0, 1.0];
    let key = [1.0, 0.0, 0.0, 1.0];
    let value = [2.0, 1.0, -1.0, 3.0];
    let (backend, workspace_authority) = backend()?;
    let attention_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(2 * DType::F32.byte_width())?,
        &cancellation,
    );
    let actual = scaled_dot_product_attention_with_context_exact_native(
        &backend,
        request,
        &query,
        &key,
        &value,
        None,
        &attention_context,
    )?;
    let expected = CheckedAttentionInvocation::new(request, &query, &key, &value, None)?
        .execute_with_context(&backend, 1, &attention_context)?;
    assert_eq!(actual, expected);
    assert_eq!(attention_context.scratch.peak_bytes(), 8);
    assert_eq!(attention_context.scratch.in_use_bytes(), 0);
    let query_tangent = [0.1, -0.2, 0.3, 0.2];
    let key_tangent = [-0.1, 0.4, 0.2, -0.3];
    let value_tangent = [0.3, 0.1, -0.2, 0.5];
    let upstream = [0.5, -0.3, 0.2, 0.7];
    let gradient_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(4 * DType::F32.byte_width())?,
        &cancellation,
    );
    let jvp = scaled_dot_product_attention_jvp_with_context_exact_native(
        &backend,
        request,
        &query,
        &key,
        &value,
        None,
        &query_tangent,
        &key_tangent,
        &value_tangent,
        &gradient_context,
    )?;
    let vjp = scaled_dot_product_attention_vjp_with_context_exact_native(
        &backend,
        request,
        &query,
        &key,
        &value,
        None,
        &upstream,
        &gradient_context,
    )?;
    let lhs = dot(&jvp, &upstream);
    let rhs = dot(&query_tangent, &vjp.query)
        + dot(&key_tangent, &vjp.key)
        + dot(&value_tangent, &vjp.value);
    assert_close(lhs, rhs, 0.00001);
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(8)?,
        &cancelled,
    );
    assert!(matches!(
        scaled_dot_product_attention_with_context_exact_native(
            &backend,
            request,
            &query,
            &key,
            &value,
            None,
            &cancelled_context,
        ),
        Err(OperatorIndirectionError::Cancelled)
    ));
    let unsupported_request = AttentionKernelRequest {
        device: DeviceId::new(DeviceKind::Cuda, 0),
        ..request
    };
    assert!(matches!(
        scaled_dot_product_attention_with_context_exact_native(
            &backend,
            unsupported_request,
            &query,
            &key,
            &value,
            None,
            &attention_context,
        ),
        Err(OperatorIndirectionError::UnsupportedDevice { .. })
    ));
    Ok(())
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual} with tolerance {tolerance}"
    );
}
