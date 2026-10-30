use std::{collections::BTreeSet, error::Error, fs, path::Path};

use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, StreamId, Tensor, TensorDescriptor, TensorError,
    generated_elementwise_or_runtime_operation_23::{
        ElementwiseRuntimePartTwentyThreeError, NativeRmsprop,
        cartesian_prod_with_context_exact_native as cartesian_prod_exact_native,
        cartesian_prod_with_context_exact_native,
    },
};

const IDS: [&str; 2] = [
    "COMFY-TENSOR-OP-FEFD7C671451",
    "COMFY-TENSOR-OP-FCDA841034ED",
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

fn upload_i64(
    backend: &CpuBackend,
    workspace_authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    values: &[i64],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::I64, DeviceId::CPU, StreamId::DEFAULT)?;
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    Ok(backend
        .upload_bytes(
            descriptor,
            &bytes,
            &context(backend, workspace_authority, cancellation)?,
        )?
        .0)
}

fn f32_values(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut values = Vec::new();
    for bytes in tensor.contiguous_bytes()?.chunks_exact(4) {
        let encoded: [u8; 4] = bytes.try_into()?;
        values.push(f32::from_ne_bytes(encoded));
    }
    Ok(values)
}

fn i64_values(tensor: &Tensor) -> Result<Vec<i64>, Box<dyn Error>> {
    let mut values = Vec::new();
    for bytes in tensor.contiguous_bytes()?.chunks_exact(8) {
        let encoded: [u8; 8] = bytes.try_into()?;
        values.push(i64::from_ne_bytes(encoded));
    }
    Ok(values)
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

#[test]
fn workspace_cartesian_product_has_exact_peak_and_failure_convergence() -> Result<(), Box<dyn Error>>
{
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let left = upload_i64(&backend, &workspace_authority, &[2], &[1, 2], &cancellation)?;
    let right = upload_i64(
        &backend,
        &workspace_authority,
        &[2],
        &[10, 20],
        &cancellation,
    )?;
    let scratch = workspace_authority.authorize_workspace(64)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch.clone(), &cancellation);
    let output = cartesian_prod_with_context_exact_native(
        &backend,
        &[left.clone(), right.clone()],
        &execution,
    )?;
    assert_eq!(i64_values(&output)?, [1, 10, 1, 20, 2, 10, 2, 20]);
    assert_eq!(scratch.peak_bytes(), 64);
    assert_eq!(scratch.in_use_bytes(), 0);

    let too_small = workspace_authority.authorize_workspace(63)?;
    let execution = backend.execution_context(StreamId::DEFAULT, too_small.clone(), &cancellation);
    assert!(matches!(
        cartesian_prod_with_context_exact_native(
            &backend,
            &[left.clone(), right.clone()],
            &execution,
        ),
        Err(sim_tensor::generated_elementwise_or_runtime_operation_23::ElementwiseRuntimePartTwentyThreeError::Tensor(
            TensorError::WorkspaceAuthorizationExceeded { .. }
        ))
    ));
    assert_eq!(too_small.in_use_bytes(), 0);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_scratch = workspace_authority.authorize_workspace(64)?;
    let execution =
        backend.execution_context(StreamId::DEFAULT, cancelled_scratch.clone(), &cancelled);
    assert!(
        cartesian_prod_with_context_exact_native(&backend, &[left, right], &execution).is_err()
    );
    assert_eq!(cancelled_scratch.peak_bytes(), 0);
    assert_eq!(cancelled_scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn task_66_resolution_slice_seals_both_unique_contracts() -> Result<(), Box<dyn Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_23")
        .ok_or("Task 66 resolution slice is missing")?;
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
fn cartesian_prod_preserves_dtype_order_empty_and_single_input_behavior()
-> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let vertical = upload_i64(&backend, &workspace_authority, &[2], &[1, 2], &cancellation)?;
    let horizontal = upload_i64(
        &backend,
        &workspace_authority,
        &[3],
        &[10, 20, 30],
        &cancellation,
    )?;
    let output = cartesian_prod_exact_native(
        &backend,
        &[vertical.clone(), horizontal],
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(output.descriptor().shape(), &[6, 2]);
    assert_eq!(output.descriptor().dtype(), DType::I64);
    assert_ne!(output.storage_id(), vertical.storage_id());
    assert_eq!(
        i64_values(&output)?,
        [1, 10, 1, 20, 1, 30, 2, 10, 2, 20, 2, 30]
    );

    let single = cartesian_prod_exact_native(
        &backend,
        std::slice::from_ref(&vertical),
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(single.descriptor().shape(), &[2]);
    assert_eq!(i64_values(&single)?, [1, 2]);
    assert_ne!(single.storage_id(), vertical.storage_id());

    let empty = upload_i64(&backend, &workspace_authority, &[0], &[], &cancellation)?;
    let empty_product = cartesian_prod_exact_native(
        &backend,
        &[vertical.clone(), empty],
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert_eq!(empty_product.descriptor().shape(), &[0, 2]);
    assert!(empty_product.contiguous_bytes()?.is_empty());

    let matrix = upload_i64(
        &backend,
        &workspace_authority,
        &[1, 2],
        &[1, 2],
        &cancellation,
    )?;
    assert!(
        cartesian_prod_exact_native(
            &backend,
            &[matrix, vertical],
            &authorized_context(&backend, &workspace_authority, &cancellation)?
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn rmsprop_default_equations_and_state_are_exact() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let mut parameters = vec![upload_f32(
        &backend,
        &workspace_authority,
        &[2],
        &[1.0, -2.0],
        &cancellation,
    )?];
    let gradients = vec![upload_f32(
        &backend,
        &workspace_authority,
        &[2],
        &[0.5, -0.25],
        &cancellation,
    )?];
    let mut optimizer = NativeRmsprop::new_with_context_exact_native(
        &backend,
        &parameters,
        0.1,
        0.99,
        1.0e-8,
        0.0,
        0.0,
        false,
        false,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    optimizer.step_with_context_exact_native(
        &backend,
        &mut parameters,
        &gradients,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    let expected = [
        1.0 - 0.1 * 0.5 / (0.0025_f32.sqrt() + 1.0e-8),
        -2.0 - 0.1 * -0.25 / (0.000625_f32.sqrt() + 1.0e-8),
    ];
    assert_close(&f32_values(&parameters[0])?, &expected);
    assert_close(
        &f32_values(&optimizer.square_averages()[0])?,
        &[0.0025, 0.000625],
    );
    assert_eq!(optimizer.steps(), [1]);
    assert!(optimizer.momentum_buffers().is_empty());
    assert!(optimizer.gradient_averages().is_empty());
    Ok(())
}

#[test]
fn centered_momentum_rmsprop_stages_all_state_and_rolls_back_on_cancel()
-> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let mut parameters = vec![
        upload_f32(&backend, &workspace_authority, &[1], &[2.0], &cancellation)?,
        upload_f32(&backend, &workspace_authority, &[1], &[-1.0], &cancellation)?,
    ];
    let gradients = vec![
        upload_f32(&backend, &workspace_authority, &[1], &[0.25], &cancellation)?,
        upload_f32(&backend, &workspace_authority, &[1], &[-0.5], &cancellation)?,
    ];
    let mut optimizer = NativeRmsprop::new_with_context_exact_native(
        &backend,
        &parameters,
        0.05,
        0.9,
        1.0e-6,
        0.1,
        0.5,
        true,
        false,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    optimizer.step_with_context_exact_native(
        &backend,
        &mut parameters,
        &gradients,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;

    let directions: [f32; 2] = [0.25 + 0.1 * 2.0, -0.5 + 0.1 * -1.0];
    let square_averages = directions.map(|gradient| 0.1 * gradient * gradient);
    let gradient_averages = directions.map(|gradient| 0.1 * gradient);
    let momentum = [0, 1].map(|index| {
        directions[index]
            / ((square_averages[index] - gradient_averages[index] * gradient_averages[index])
                .sqrt()
                + 1.0e-6)
    });
    assert_close(&f32_values(&parameters[0])?, &[2.0 - 0.05 * momentum[0]]);
    assert_close(&f32_values(&parameters[1])?, &[-1.0 - 0.05 * momentum[1]]);
    assert_close(
        &f32_values(&optimizer.square_averages()[0])?,
        &[square_averages[0]],
    );
    assert_close(
        &f32_values(&optimizer.gradient_averages()[1])?,
        &[gradient_averages[1]],
    );
    assert_close(
        &f32_values(&optimizer.momentum_buffers()[0])?,
        &[momentum[0]],
    );

    let parameter_ids = parameters
        .iter()
        .map(Tensor::storage_id)
        .collect::<Vec<_>>();
    let square_ids = optimizer
        .square_averages()
        .iter()
        .map(Tensor::storage_id)
        .collect::<Vec<_>>();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(
        optimizer
            .step_with_context_exact_native(
                &backend,
                &mut parameters,
                &gradients,
                &authorized_context(&backend, &workspace_authority, &cancelled)?
            )
            .is_err()
    );
    assert_eq!(optimizer.steps(), [1, 1]);
    assert_eq!(
        parameters
            .iter()
            .map(Tensor::storage_id)
            .collect::<Vec<_>>(),
        parameter_ids
    );
    assert_eq!(
        optimizer
            .square_averages()
            .iter()
            .map(Tensor::storage_id)
            .collect::<Vec<_>>(),
        square_ids
    );
    Ok(())
}

#[test]
fn task_66_invalid_inputs_and_cancellation_fail_closed() -> Result<(), Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let parameter = upload_f32(&backend, &workspace_authority, &[1], &[1.0], &cancellation)?;
    assert!(
        NativeRmsprop::new_with_context_exact_native(
            &backend,
            std::slice::from_ref(&parameter),
            -0.1,
            0.99,
            1.0e-8,
            0.0,
            0.0,
            false,
            false,
            &authorized_context(&backend, &workspace_authority, &cancellation)?,
        )
        .is_err()
    );
    assert!(
        cartesian_prod_exact_native(
            &backend,
            &[],
            &authorized_context(&backend, &workspace_authority, &cancellation)?
        )
        .is_err()
    );
    let integral = upload_i64(&backend, &workspace_authority, &[1], &[1], &cancellation)?;
    assert!(
        cartesian_prod_exact_native(
            &backend,
            &[parameter.clone(), integral.clone()],
            &authorized_context(&backend, &workspace_authority, &cancellation)?,
        )
        .is_err()
    );
    assert!(
        NativeRmsprop::new_with_context_exact_native(
            &backend,
            &[integral],
            0.01,
            0.99,
            1.0e-8,
            0.0,
            0.0,
            false,
            false,
            &authorized_context(&backend, &workspace_authority, &cancellation)?,
        )
        .is_err()
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_scratch = workspace_authority.authorize_workspace(1024 * 1024)?;
    let cancelled_context =
        backend.execution_context(StreamId::DEFAULT, cancelled_scratch.clone(), &cancelled);
    assert!(matches!(
        cartesian_prod_exact_native(&backend, &[], &cancelled_context),
        Err(ElementwiseRuntimePartTwentyThreeError::Cancelled)
    ));
    assert!(matches!(
        NativeRmsprop::new_with_context_exact_native(
            &backend,
            &[],
            -0.01,
            0.99,
            1.0e-8,
            0.0,
            0.0,
            false,
            false,
            &cancelled_context,
        ),
        Err(ElementwiseRuntimePartTwentyThreeError::Cancelled)
    ));

    let mut parameters = vec![parameter];
    let parameter_storage = parameters[0].storage_id();
    let mut optimizer = NativeRmsprop::new_with_context_exact_native(
        &backend,
        &parameters,
        0.01,
        0.99,
        1.0e-8,
        0.0,
        0.0,
        false,
        false,
        &authorized_context(&backend, &workspace_authority, &cancellation)?,
    )?;
    assert!(matches!(
        optimizer.step_with_context_exact_native(
            &backend,
            &mut parameters,
            &[],
            &cancelled_context,
        ),
        Err(ElementwiseRuntimePartTwentyThreeError::Cancelled)
    ));
    assert_eq!(parameters[0].storage_id(), parameter_storage);
    assert_eq!(optimizer.steps(), [0]);
    assert_eq!(cancelled_scratch.peak_bytes(), 0);
    assert_eq!(cancelled_scratch.in_use_bytes(), 0);
    Ok(())
}
