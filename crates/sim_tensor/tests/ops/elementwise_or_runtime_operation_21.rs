use std::{collections::BTreeSet, error::Error, fs, path::Path, sync::Arc};

use sha2::{Digest, Sha256};
use sim_tensor::{
    AutogradError, AutogradInput, AutogradTape, BackwardRule, CancellationToken, CpuBackend,
    CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, GradientMode, GradientReducer, GradientStore,
    LeafId, SavedTensor, StreamId, TapeState, Tensor, TensorDescriptor, TensorError,
    generated_elementwise_or_runtime_operation_21::{
        ElementwiseRuntimePartTwentyOneError,
        addcmul_jvp_with_context_exact_native as addcmul_jvp_exact_native,
        addcmul_vjp_with_context_exact_native as addcmul_vjp_exact_native,
        addcmul_with_context_exact_native as addcmul_exact_native,
        backward_method_with_context_exact_native,
        bitwise_or_with_context_exact_native as bitwise_or_exact_native,
        cumprod_jvp_with_context_exact_native as cumprod_jvp_exact_native,
        cumprod_vjp_with_context_exact_native as cumprod_vjp_exact_native,
        cumprod_with_context_exact_native as cumprod_exact_native,
        exp_jvp_with_context_exact_native as exp_jvp_exact_native,
        exp_vjp_with_context_exact_native as exp_vjp_exact_native,
        exp_with_context_exact_native as exp_exact_native, get_default_dtype_exact_native,
        is_grad_enabled_exact_native, kron_jvp_with_context_exact_native as kron_jvp_exact_native,
        kron_vjp_with_context_exact_native as kron_vjp_exact_native,
        kron_vjp_with_context_exact_native, kron_with_context_exact_native as kron_exact_native,
        neg_jvp_with_context_exact_native as neg_jvp_exact_native,
        neg_vjp_with_context_exact_native as neg_vjp_exact_native,
        neg_with_context_exact_native as neg_exact_native,
        sin_jvp_with_context_exact_native as sin_jvp_exact_native,
        sin_vjp_with_context_exact_native as sin_vjp_exact_native,
        sin_with_context_exact_native as sin_exact_native, unfold_exact_native,
        unfold_jvp_exact_native, unfold_vjp_with_context_exact_native as unfold_vjp_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-ED2FCEFE4ECE",
    "COMFY-TENSOR-OP-E9EE26A3960C",
    "COMFY-TENSOR-OP-ECF812A5CF81",
    "COMFY-TENSOR-OP-E78AD841C264",
    "COMFY-TENSOR-OP-EBFD0D7FDA6D",
    "COMFY-TENSOR-OP-E8EA8CB65E2C",
    "COMFY-TENSOR-OP-ECAAC1BA206A",
    "COMFY-TENSOR-OP-EC849F37A5FD",
    "COMFY-TENSOR-OP-E8537C6996DA",
    "COMFY-TENSOR-OP-F122D7D4E807",
    "COMFY-TENSOR-OP-EA16F5C2EAC6",
    "COMFY-TENSOR-OP-E851105B589B",
];

fn context<'a>(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, sim_tensor::TensorError> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(0)?,
        cancellation,
    ))
}

fn authorized_context<'a>(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, sim_tensor::TensorError> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(1024 * 1024)?,
        cancellation,
    ))
}

fn upload_f32(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    Ok(backend
        .upload_f32(
            descriptor,
            values,
            &context(backend, workspace_authority, cancellation)?,
        )?
        .0)
}

fn upload_integer(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    dtype: DType,
    values: &[u64],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), dtype, DeviceId::CPU, StreamId::DEFAULT)?;
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend(dtype.encode_scalar(
            sim_tensor::Scalar::Unsigned(*value),
            "task-63-test",
            DeviceId::CPU,
        )?);
    }
    Ok(backend
        .upload_bytes(
            descriptor,
            &bytes,
            &context(backend, workspace_authority, cancellation)?,
        )?
        .0)
}

fn values(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let byte_count = tensor
        .descriptor()
        .element_count()?
        .checked_mul(4)
        .ok_or("tensor-to-f32 workspace overflow")?;
    let execution = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority.authorize_workspace(byte_count)?,
        cancellation,
    );
    Ok(tensor_to_f32_with_context_exact_native(
        backend, tensor, &execution,
    )?)
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "expected {expected}, got {actual}"
        );
    }
}

#[track_caller]
fn assert_cancelled<T>(result: Result<T, ElementwiseRuntimePartTwentyOneError>) {
    assert!(matches!(
        result,
        Err(ElementwiseRuntimePartTwentyOneError::Cancelled)
    ));
}

#[test]
fn workspace_kron_vjp_accounts_simultaneous_gradients_exactly() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2],
        &[1.0, 2.0],
        &cancellation,
    )?;
    let other = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[3.0, 4.0, 5.0],
        &cancellation,
    )?;
    let gradient = upload_f32(
        &backend,
        &workspace_authority,
        &[6],
        &[1.0; 6],
        &cancellation,
    )?;
    let scratch = workspace_authority.authorize_workspace(20)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch.clone(), &cancellation);
    let gradients =
        kron_vjp_with_context_exact_native(&backend, &input, &other, &gradient, &execution)?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &gradients.input,
            &cancellation,
        )?,
        &[12.0, 12.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &gradients.other,
            &cancellation,
        )?,
        &[3.0, 3.0, 3.0],
    );
    assert_eq!(scratch.peak_bytes(), 20);
    assert_eq!(scratch.in_use_bytes(), 0);

    let too_small = workspace_authority.authorize_workspace(19)?;
    let execution = backend.execution_context(StreamId::DEFAULT, too_small.clone(), &cancellation);
    assert!(matches!(
        kron_vjp_with_context_exact_native(&backend, &input, &other, &gradient, &execution),
        Err(sim_tensor::generated_elementwise_or_runtime_operation_21::ElementwiseRuntimePartTwentyOneError::Tensor(
            TensorError::WorkspaceAuthorizationExceeded { .. }
        ))
    ));
    assert_eq!(too_small.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn task_64_resolution_slice_seals_all_twelve_unique_contracts() -> Result<(), Box<dyn Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_21")
        .ok_or("Task 64 resolution slice is missing")?;
    assert_eq!(slice.contracts.len(), IDS.len());
    assert_eq!(
        slice
            .contracts
            .iter()
            .map(|contract| contract.operation_id)
            .collect::<BTreeSet<_>>(),
        IDS.into_iter().collect()
    );
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root")?;
    let mut overloads = BTreeSet::new();
    let mut digests = BTreeSet::new();
    for contract in slice.contracts {
        assert!(overloads.insert(contract.overload_id));
        assert!(digests.insert(contract.evidence_fixture_sha256));
        let bytes = fs::read(workspace.join(contract.evidence_fixture))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            contract.evidence_fixture_sha256
        );
    }
    Ok(())
}

#[test]
fn unary_adapters_and_request_scoped_queries_preserve_canonical_semantics()
-> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[-1.0, 0.0, 1.0],
        &cancellation,
    )?;
    let tangent = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[1.0, 2.0, 3.0],
        &cancellation,
    )?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &exp_exact_native(
                &backend,
                &input,
                &authorized_context(&backend, &workspace_authority, &cancellation)?,
            )?,
            &cancellation,
        )?,
        &[(-1.0_f32).exp(), 1.0, 1.0_f32.exp()],
    );
    let exp_vjp = values(
        &backend,
        &workspace_authority,
        &exp_vjp_exact_native(
            &backend,
            &input,
            &tangent,
            &authorized_context(&backend, &workspace_authority, &cancellation)?,
        )?,
        &cancellation,
    )?;
    assert_close(&exp_vjp, &[(-1.0_f32).exp(), 2.0, 3.0 * 1.0_f32.exp()]);
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &exp_jvp_exact_native(
                &backend,
                &input,
                &tangent,
                &authorized_context(&backend, &workspace_authority, &cancellation)?,
            )?,
            &cancellation,
        )?,
        &exp_vjp,
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &neg_exact_native(
                &backend,
                &input,
                &authorized_context(&backend, &workspace_authority, &cancellation)?,
            )?,
            &cancellation,
        )?,
        &[1.0, -0.0, -1.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &neg_vjp_exact_native(
                &backend,
                &tangent,
                &authorized_context(&backend, &workspace_authority, &cancellation)?,
            )?,
            &cancellation,
        )?,
        &[-1.0, -2.0, -3.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &neg_jvp_exact_native(
                &backend,
                &tangent,
                &authorized_context(&backend, &workspace_authority, &cancellation)?,
            )?,
            &cancellation,
        )?,
        &[-1.0, -2.0, -3.0],
    );
    let sine = sin_exact_native(
        &backend,
        &input,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_close(
        &values(&backend, &workspace_authority, &sine, &cancellation)?,
        &[(-1.0_f32).sin(), 0.0, 1.0_f32.sin()],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &sin_vjp_exact_native(
                &backend,
                &input,
                &tangent,
                &authorized_context(&backend, &workspace_authority, &cancellation)?,
            )?,
            &cancellation,
        )?,
        &values(
            &backend,
            &workspace_authority,
            &sin_jvp_exact_native(
                &backend,
                &input,
                &tangent,
                &authorized_context(&backend, &workspace_authority, &cancellation)?,
            )?,
            &cancellation,
        )?,
    );
    assert_eq!(get_default_dtype_exact_native(&cancellation)?, DType::F32);
    assert!(is_grad_enabled_exact_native(
        GradientMode::Enabled,
        &cancellation
    )?);
    assert!(!is_grad_enabled_exact_native(
        GradientMode::NoGrad,
        &cancellation
    )?);
    assert!(!is_grad_enabled_exact_native(
        GradientMode::Inference,
        &cancellation
    )?);
    Ok(())
}

#[test]
fn unfold_is_a_read_only_view_with_zero_aware_derivatives() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 4],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &cancellation,
    )?;
    let mut unfolded = unfold_exact_native(&input, 1, 3, 1, &cancellation)?;
    assert_eq!(unfolded.descriptor().shape(), &[2, 2, 3]);
    assert_eq!(unfolded.descriptor().strides(), &[4, 1, 1]);
    assert_eq!(unfolded.storage_id(), input.storage_id());
    assert_close(
        &values(&backend, &workspace_authority, &unfolded, &cancellation)?,
        &[1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 6.0, 7.0, 8.0],
    );
    assert!(unfolded.write().is_err());
    let output_gradient = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2, 3],
        &[1.0; 12],
        &cancellation,
    )?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &unfold_vjp_exact_native(
                &backend,
                &input,
                &output_gradient,
                1,
                3,
                1,
                &authorized_context(&backend, &workspace_authority, &cancellation)?,
            )?,
            &cancellation,
        )?,
        &[1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0],
    );
    assert_eq!(
        unfold_jvp_exact_native(&input, 1, 3, 1, &cancellation)?
            .descriptor()
            .shape(),
        &[2, 2, 3]
    );
    Ok(())
}

#[test]
fn composed_arithmetic_bitwise_and_cumulative_adapters_preserve_semantics()
-> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2],
        &[1.0, 2.0],
        &cancellation,
    )?;
    let one = upload_f32(
        &backend,
        &workspace_authority,
        &[2],
        &[2.0, 3.0],
        &cancellation,
    )?;
    let two = upload_f32(
        &backend,
        &workspace_authority,
        &[2],
        &[4.0, 5.0],
        &cancellation,
    )?;
    let tangent = upload_f32(
        &backend,
        &workspace_authority,
        &[2],
        &[1.0, 1.0],
        &cancellation,
    )?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &addcmul_exact_native(
                &backend,
                &input,
                &one,
                &two,
                0.5,
                &authorized_context(&backend, &workspace_authority, &cancellation)?,
            )?,
            &cancellation,
        )?,
        &[5.0, 9.5],
    );
    let gradients = addcmul_vjp_exact_native(
        &backend,
        &input,
        &one,
        &two,
        0.5,
        &tangent,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &gradients.input,
            &cancellation,
        )?,
        &[1.0, 1.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &gradients.tensor_one,
            &cancellation,
        )?,
        &[2.0, 2.5],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &gradients.tensor_two,
            &cancellation,
        )?,
        &[1.0, 1.5],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &addcmul_jvp_exact_native(
                &backend,
                &input,
                &one,
                &two,
                &tangent,
                &tangent,
                &tangent,
                0.5,
                &authorized_context(&backend, &workspace_authority, &cancellation)?,
            )?,
            &cancellation,
        )?,
        &[4.0, 5.0],
    );

    let left = upload_integer(
        &backend,
        &workspace_authority,
        &[3],
        DType::U8,
        &[1, 2, 4],
        &cancellation,
    )?;
    let right = upload_integer(
        &backend,
        &workspace_authority,
        &[3],
        DType::U16,
        &[8, 4, 1],
        &cancellation,
    )?;
    let bitwise = bitwise_or_exact_native(
        &backend,
        &left,
        &right,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(bitwise.descriptor().dtype(), DType::U16);
    assert_close(
        &values(&backend, &workspace_authority, &bitwise, &cancellation)?,
        &[9.0, 6.0, 5.0],
    );

    let cumulative_input = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[1.0, 2.0, 3.0],
        &cancellation,
    )?;
    let cumulative_tangent = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[1.0, 1.0, 1.0],
        &cancellation,
    )?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &cumprod_exact_native(
                &backend,
                &cumulative_input,
                0,
                None,
                &authorized_context(&backend, &workspace_authority, &cancellation)?,
            )?,
            &cancellation,
        )?,
        &[1.0, 2.0, 6.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &cumprod_vjp_exact_native(
                &backend,
                &cumulative_input,
                &cumulative_tangent,
                0,
                &authorized_context(&backend, &workspace_authority, &cancellation)?,
            )?,
            &cancellation,
        )?,
        &[9.0, 4.0, 2.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &cumprod_jvp_exact_native(
                &backend,
                &cumulative_input,
                &cumulative_tangent,
                0,
                &authorized_context(&backend, &workspace_authority, &cancellation)?,
            )?,
            &cancellation,
        )?,
        &[1.0, 3.0, 11.0],
    );
    let zero_input = upload_f32(
        &backend,
        &workspace_authority,
        &[3],
        &[4.0, 0.0, 6.0],
        &cancellation,
    )?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &cumprod_vjp_exact_native(
                &backend,
                &zero_input,
                &cumulative_tangent,
                0,
                &authorized_context(&backend, &workspace_authority, &cancellation)?,
            )?,
            &cancellation,
        )?,
        &[1.0, 28.0, 0.0],
    );
    Ok(())
}

#[test]
fn kron_and_backward_execute_native_derivative_paths() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &workspace_authority,
        &[2],
        &[1.0, 2.0],
        &cancellation,
    )?;
    let other = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
        &cancellation,
    )?;
    let output = kron_exact_native(
        &backend,
        &input,
        &other,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(output.descriptor().shape(), &[2, 4]);
    assert_close(
        &values(&backend, &workspace_authority, &output, &cancellation)?,
        &[1.0, 2.0, 2.0, 4.0, 3.0, 4.0, 6.0, 8.0],
    );
    let output_gradient = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 4],
        &[1.0; 8],
        &cancellation,
    )?;
    let gradients = kron_vjp_exact_native(
        &backend,
        &input,
        &other,
        &output_gradient,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &gradients.input,
            &cancellation,
        )?,
        &[10.0, 10.0],
    );
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &gradients.other,
            &cancellation,
        )?,
        &[3.0, 3.0, 3.0, 3.0],
    );
    let input_tangent = upload_f32(
        &backend,
        &workspace_authority,
        &[2],
        &[1.0, 1.0],
        &cancellation,
    )?;
    let other_tangent = upload_f32(
        &backend,
        &workspace_authority,
        &[2, 2],
        &[0.0; 4],
        &cancellation,
    )?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            &kron_jvp_exact_native(
                &backend,
                &input,
                &other,
                &input_tangent,
                &other_tangent,
                &authorized_context(&backend, &workspace_authority, &cancellation)?,
            )?,
            &cancellation,
        )?,
        &[1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0],
    );

    let leaf = LeafId::new("task_64_leaf")?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    let output_slot = tape
        .record(
            vec![AutogradInput::Leaf(leaf.clone())],
            1,
            Vec::new(),
            Arc::new(PassThroughRule),
        )?
        .ok_or("enabled tape did not record")?
        .remove(0);
    let scalar = upload_f32(&backend, &workspace_authority, &[1], &[7.0], &cancellation)?;
    let scratch = workspace_authority.authorize_workspace(4)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
    let mut gradient_store = GradientStore::default();
    backward_method_with_context_exact_native(
        &backend,
        &mut tape,
        output_slot,
        &scalar,
        None,
        Some(std::slice::from_ref(&leaf)),
        &NoAccumulation,
        &mut gradient_store,
        false,
        false,
        &execution,
    )?;
    assert_close(
        &values(
            &backend,
            &workspace_authority,
            gradient_store
                .gradient(&leaf)
                .ok_or("leaf gradient missing")?,
            &cancellation,
        )?,
        &[1.0],
    );
    Ok(())
}

#[test]
fn task_64_every_public_tensor_adapter_observes_cancellation_before_validation()
-> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let live = CancellationToken::default();
    let input = upload_f32(&backend, &workspace_authority, &[2], &[1.0, 2.0], &live)?;
    let mismatched = upload_f32(&backend, &workspace_authority, &[1], &[1.0], &live)?;
    let integer = upload_integer(
        &backend,
        &workspace_authority,
        &[2],
        DType::U8,
        &[1, 2],
        &live,
    )?;
    let leaf = LeafId::new("task_64_cancelled_leaf")?;
    let mut tape = AutogradTape::new(GradientMode::Enabled);
    let output_slot = tape
        .record(
            vec![AutogradInput::Leaf(leaf)],
            1,
            Vec::new(),
            Arc::new(PassThroughRule),
        )?
        .ok_or("enabled tape did not record")?
        .remove(0);

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let scratch = workspace_authority.authorize_workspace(1024 * 1024)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch.clone(), &cancelled);
    let mut gradient_store = GradientStore::default();

    assert_cancelled(backward_method_with_context_exact_native(
        &backend,
        &mut tape,
        output_slot,
        &mismatched,
        None,
        None,
        &NoAccumulation,
        &mut gradient_store,
        true,
        true,
        &execution,
    ));
    assert_eq!(tape.state(), &TapeState::Active);
    assert_eq!(tape.retained_node_count(), 1);

    assert_cancelled(exp_exact_native(&backend, &integer, &execution));
    assert_cancelled(exp_vjp_exact_native(
        &backend,
        &integer,
        &mismatched,
        &execution,
    ));
    assert_cancelled(exp_jvp_exact_native(
        &backend,
        &integer,
        &mismatched,
        &execution,
    ));
    assert_cancelled(neg_exact_native(&backend, &integer, &execution));
    assert_cancelled(neg_vjp_exact_native(&backend, &integer, &execution));
    assert_cancelled(neg_jvp_exact_native(&backend, &integer, &execution));

    assert_cancelled(unfold_exact_native(&input, 99, 0, 0, &cancelled));
    assert_cancelled(unfold_vjp_exact_native(
        &backend,
        &integer,
        &mismatched,
        99,
        0,
        0,
        &execution,
    ));
    assert_cancelled(unfold_jvp_exact_native(&input, 99, 0, 0, &cancelled));

    assert_cancelled(addcmul_exact_native(
        &backend,
        &integer,
        &mismatched,
        &input,
        1.0,
        &execution,
    ));
    assert_cancelled(addcmul_vjp_exact_native(
        &backend,
        &integer,
        &mismatched,
        &input,
        1.0,
        &mismatched,
        &execution,
    ));
    assert_cancelled(addcmul_jvp_exact_native(
        &backend,
        &integer,
        &mismatched,
        &input,
        &mismatched,
        &input,
        &integer,
        1.0,
        &execution,
    ));
    assert_cancelled(bitwise_or_exact_native(
        &backend,
        &input,
        &mismatched,
        &execution,
    ));
    assert_cancelled(cumprod_exact_native(
        &backend,
        &input,
        99,
        Some(DType::Bool),
        &execution,
    ));
    assert_cancelled(cumprod_vjp_exact_native(
        &backend,
        &input,
        &mismatched,
        99,
        &execution,
    ));
    assert_cancelled(cumprod_jvp_exact_native(
        &backend,
        &input,
        &mismatched,
        99,
        &execution,
    ));
    assert_cancelled(get_default_dtype_exact_native(&cancelled));
    assert_cancelled(is_grad_enabled_exact_native(
        GradientMode::Enabled,
        &cancelled,
    ));

    assert_cancelled(kron_exact_native(
        &backend,
        &integer,
        &mismatched,
        &execution,
    ));
    assert_cancelled(kron_vjp_exact_native(
        &backend,
        &integer,
        &mismatched,
        &input,
        &execution,
    ));
    assert_cancelled(kron_jvp_exact_native(
        &backend,
        &integer,
        &mismatched,
        &input,
        &integer,
        &execution,
    ));
    assert_cancelled(sin_exact_native(&backend, &integer, &execution));
    assert_cancelled(sin_vjp_exact_native(
        &backend,
        &integer,
        &mismatched,
        &execution,
    ));
    assert_cancelled(sin_jvp_exact_native(
        &backend,
        &integer,
        &mismatched,
        &execution,
    ));

    assert_eq!(scratch.peak_bytes(), 0);
    assert_eq!(scratch.in_use_bytes(), 0);
    Ok(())
}

struct PassThroughRule;

impl BackwardRule for PassThroughRule {
    fn vjp(
        &self,
        output_gradients: &[Option<Tensor>],
        _saved_tensors: &[SavedTensor],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<Option<Tensor>>, AutogradError> {
        Ok(vec![output_gradients.first().cloned().flatten()])
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
            reason: "Task 64 reducer should not accumulate".to_owned(),
        })
    }
}
