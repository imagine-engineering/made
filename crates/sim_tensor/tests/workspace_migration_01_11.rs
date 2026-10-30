use sim_tensor::CpuWorkspaceAuthority;
use sim_tensor::{
    CancellationToken, CpuBackend, DType, DeviceId, StreamId, TensorDescriptor,
    generated_elementwise_or_runtime_operation_01::abs_vjp_with_context_exact_native,
};
use std::{fs, path::Path};

fn upload_f32(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<sim_tensor::Tensor, Box<dyn std::error::Error>> {
    let descriptor = TensorDescriptor::contiguous(
        vec![u64::try_from(values.len())?],
        DType::F32,
        DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        cancellation,
    );
    Ok(backend.upload_f32(descriptor, values, &context)?.0)
}

#[test]
fn canonical_workspace_path_is_exact_bounded_and_converges()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(&backend, &authority, &[-2.0, 0.0, 3.0], &cancellation)?;
    let output_gradient = upload_f32(&backend, &authority, &[1.0, 2.0, 4.0], &cancellation)?;

    let buffer_bytes = 3 * u64::try_from(std::mem::size_of::<f32>())?;
    let simultaneous_bytes = buffer_bytes.checked_mul(3).ok_or("workspace overflow")?;
    let exact = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(simultaneous_bytes)?,
        &cancellation,
    );
    let output = abs_vjp_with_context_exact_native(&backend, &input, &output_gradient, &exact)?;
    assert_eq!(output.descriptor().shape(), &[3]);
    assert_eq!(exact.scratch.peak_bytes(), simultaneous_bytes);
    assert_eq!(exact.scratch.in_use_bytes(), 0);

    let insufficient = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(simultaneous_bytes - 1)?,
        &cancellation,
    );
    assert!(
        abs_vjp_with_context_exact_native(&backend, &input, &output_gradient, &insufficient,)
            .is_err()
    );
    assert_eq!(insufficient.scratch.in_use_bytes(), 0);

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(simultaneous_bytes)?,
        &cancelled,
    );
    assert!(
        abs_vjp_with_context_exact_native(&backend, &input, &output_gradient, &cancelled_context,)
            .is_err()
    );
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn migration_inventory_is_bounded_and_cannot_mint_authority()
-> Result<(), Box<dyn std::error::Error>> {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for part in 1..=11 {
        let source = fs::read_to_string(crate_root.join(format!(
            "src/ops/elementwise_or_runtime_operation_{part:02}.rs"
        )))?;
        assert!(
            source.contains("with_context_exact_native"),
            "part {part:02} has no canonical ExecutionContext entry point"
        );
        assert_eq!(
            source.matches("ScratchReservation::none()").count(),
            0,
            "part {part:02} must not retain compatibility scratch constructors"
        );
        assert!(
            !source.contains("Option<&ExecutionContext")
                && !source.contains("legacy_context")
                && !source.contains("transitional_context")
                && !source.contains("allow_legacy"),
            "part {part:02} must not retain alternate untracked workspace paths"
        );
        assert!(
            !source.contains("authorize_workspace("),
            "part {part:02} must not mint workspace authority"
        );
    }
    Ok(())
}
