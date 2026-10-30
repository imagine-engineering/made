use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, StreamId, TensorDescriptor,
    generated_sim_operator_indirection_01::{
        ConvolutionGeometry, ConvolutionPaddingMode, OperatorIndirectionError,
        convolution_jvp_with_context_exact_native, convolution_vjp_with_context_exact_native,
        convolution_with_context_exact_native, tensor_to_f32_with_context_exact_native,
    },
    generated_sim_operator_indirection_02::cast_to_input_with_context_exact_native,
};
use std::{collections::BTreeSet, fs, path::Path};

const IDS: [&str; 7] = [
    "COMFY-TENSOR-OP-A0BD98DDA517",
    "COMFY-TENSOR-OP-A553C4928CA6",
    "COMFY-TENSOR-OP-A88C934F4A40",
    "COMFY-TENSOR-OP-C9049FCF1A75",
    "COMFY-TENSOR-OP-D63C669FCD27",
    "COMFY-TENSOR-OP-DAC4074BC3B2",
    "COMFY-TENSOR-OP-FDDDAF202C6D",
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
        .find(|slice| slice.module_name == "sim_operator_indirection_02")
        .ok_or("operator-indirection part-two resolution slice is missing")?;
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
            "comfy-parity-tensor-ops-comfy-operator-indirection-comfy-tensor-op-a0bd98dda517"
        );
        let bytes = fs::read(workspace.join(contract.evidence_fixture))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            contract.evidence_fixture_sha256
        );
        let callable = match contract.operation_id {
            "COMFY-TENSOR-OP-A0BD98DDA517" => {
                "NativeModule::zero_init_parameter_with_context_exact_native"
            }
            "COMFY-TENSOR-OP-A553C4928CA6" => "disable_weight_init_conv1d_exact_native",
            "COMFY-TENSOR-OP-A88C934F4A40" => "disable_weight_init_layer_norm_exact_native",
            "COMFY-TENSOR-OP-C9049FCF1A75" => "disable_weight_init_group_norm_exact_native",
            "COMFY-TENSOR-OP-D63C669FCD27" => "pick_operations_exact_native",
            "COMFY-TENSOR-OP-DAC4074BC3B2" => "manual_cast_linear_exact_native",
            "COMFY-TENSOR-OP-FDDDAF202C6D" => "cast_to_input_with_context_exact_native",
            _ => return Err("unexpected Task 43 operation identifier".into()),
        };
        assert!(contract.rust_signature.contains(callable));
        assert!(
            !contract
                .rust_signature
                .contains("zero_init_parameter_exact_native")
        );
    }
    Ok(())
}

#[test]
fn cast_to_input_delegates_dtype_device_alias_and_cancellation_semantics()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, workspace_authority) = backend()?;
    let cancellation = CancellationToken::default();
    let upload_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(0)?,
        &cancellation,
    );
    let weight = upload_f32(&backend, &[3], &[1.25, -2.5, 3.75], &upload_context)?;
    let input_f32 = upload_f32(&backend, &[1], &[0.0], &upload_context)?;
    let alias_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(0)?,
        &cancellation,
    );
    let alias = cast_to_input_with_context_exact_native(
        &backend,
        &weight,
        &input_f32,
        false,
        false,
        &alias_context,
    )?;
    assert_eq!(alias.storage_id(), weight.storage_id());

    let input_cast_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(DType::F16.byte_width())?,
        &cancellation,
    );
    let input_f16 =
        sim_tensor::generated_sim_operator_indirection_01::cast_to_with_context_exact_native(
            &backend,
            &input_f32,
            DType::F16,
            DeviceId::CPU,
            false,
            true,
            &input_cast_context,
        )?;
    let required = 3 * DType::F16.byte_width();
    let cast_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(required)?,
        &cancellation,
    );
    let cast = cast_to_input_with_context_exact_native(
        &backend,
        &weight,
        &input_f16,
        true,
        true,
        &cast_context,
    )?;
    assert_eq!(cast.descriptor().dtype(), DType::F16);
    assert_eq!(cast_context.scratch.peak_bytes(), required);
    assert_eq!(cast_context.scratch.in_use_bytes(), 0);
    let decode_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(3 * DType::F32.byte_width())?,
        &cancellation,
    );
    assert_eq!(
        tensor_to_f32_with_context_exact_native(&backend, &cast, &decode_context)?,
        vec![1.25, -2.5, 3.75]
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(required)?,
        &cancelled,
    );
    assert!(matches!(
        cast_to_input_with_context_exact_native(
            &backend,
            &weight,
            &input_f16,
            false,
            true,
            &cancelled_context,
        ),
        Err(OperatorIndirectionError::Cancelled)
    ));
    Ok(())
}

#[test]
fn convolution_padding_modes_share_forward_and_gradient_coordinate_mapping()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let (backend, workspace_authority) = backend()?;
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(0)?,
        &cancellation,
    );
    let run = |mode| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let geometry = ConvolutionGeometry::new_with_padding_mode(
            1,
            vec![1],
            vec![1],
            vec![1],
            1,
            false,
            vec![0],
            mode,
        )?;
        Ok(convolution_with_context_exact_native(
            &[1.0, 2.0, 3.0],
            &[1, 1, 3],
            &[1.0, 1.0, 1.0],
            &[1, 1, 3],
            None,
            &geometry,
            DeviceId::CPU,
            &execution,
        )?
        .values)
    };
    assert_eq!(run(ConvolutionPaddingMode::Zeros)?, vec![3.0, 6.0, 5.0]);
    assert_eq!(run(ConvolutionPaddingMode::Replicate)?, vec![4.0, 6.0, 8.0]);
    assert_eq!(run(ConvolutionPaddingMode::Reflect)?, vec![5.0, 6.0, 7.0]);
    assert_eq!(run(ConvolutionPaddingMode::Circular)?, vec![6.0, 6.0, 6.0]);

    let geometry = ConvolutionGeometry::new_with_padding_mode(
        1,
        vec![1],
        vec![1],
        vec![1],
        1,
        false,
        vec![0],
        ConvolutionPaddingMode::Replicate,
    )?;
    let vjp = convolution_vjp_with_context_exact_native(
        &[1.0, 2.0, 3.0],
        &[1, 1, 3],
        &[1.0, 1.0, 1.0],
        &[1, 1, 3],
        None,
        &[1.0, 1.0, 1.0],
        &geometry,
        DeviceId::CPU,
        &execution,
    )?;
    assert_eq!(vjp.input, vec![3.0, 3.0, 3.0]);
    assert_eq!(vjp.weight, vec![4.0, 6.0, 8.0]);
    let input_tangent = [1.0, 0.0, -1.0];
    let weight_tangent = [0.5, 0.0, -0.5];
    let jvp = convolution_jvp_with_context_exact_native(
        &[1.0, 2.0, 3.0],
        &input_tangent,
        &[1, 1, 3],
        &[1.0, 1.0, 1.0],
        &weight_tangent,
        &[1, 1, 3],
        None,
        None,
        &geometry,
        DeviceId::CPU,
        &execution,
    )?;
    assert_eq!(jvp.values, vec![1.5, -1.0, -2.5]);
    let jvp_projection = jvp.values.iter().sum::<f32>();
    let vjp_projection = input_tangent
        .iter()
        .zip(&vjp.input)
        .map(|(tangent, gradient)| tangent * gradient)
        .sum::<f32>()
        + weight_tangent
            .iter()
            .zip(&vjp.weight)
            .map(|(tangent, gradient)| tangent * gradient)
            .sum::<f32>();
    assert_eq!(jvp_projection, vjp_projection);
    Ok(())
}
