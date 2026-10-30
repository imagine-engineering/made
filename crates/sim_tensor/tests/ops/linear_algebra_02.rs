use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DecodedScalar, DeviceId,
    ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, StreamId, Tensor,
    TensorDescriptor,
    generated_linear_algebra_01::transpose_last_two_with_context_exact_native,
    generated_linear_algebra_02::{
        LinearAlgebraPartTwoError, bmm_jvp_with_context_exact_native,
        bmm_vjp_with_context_exact_native,
        bmm_with_context_exact_native, norm_jvp_with_context_exact_native,
        norm_vjp_with_context_exact_native, norm_with_context_exact_native,
        svd_jvp_with_context_exact_native, svd_vjp_with_context_exact_native,
        svd_with_context_exact_native,
    },
};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, ops::Deref, path::Path};

struct TestBackend {
    backend: CpuBackend,
    workspace_authority: CpuWorkspaceAuthority,
    workspace_limit: u64,
}

impl TestBackend {
    fn new(workspace_limit: u64) -> Result<Self, Box<dyn std::error::Error>> {
        let (backend, workspace_authority) =
            CpuWorkspaceAuthority::create_backend(workspace_limit)?;
        Ok(Self {
            backend,
            workspace_authority,
            workspace_limit,
        })
    }

    fn execution<'a>(
        &self,
        cancellation: &'a CancellationToken,
    ) -> Result<ExecutionContext<'a>, Box<dyn std::error::Error>> {
        self.execution_with_workspace(self.workspace_limit, cancellation)
    }

    fn execution_with_workspace<'a>(
        &self,
        bytes: u64,
        cancellation: &'a CancellationToken,
    ) -> Result<ExecutionContext<'a>, Box<dyn std::error::Error>> {
        Ok(self.backend.execution_context(
            StreamId::DEFAULT,
            self.workspace_authority.authorize_workspace(bytes)?,
            cancellation,
        ))
    }
}

impl Deref for TestBackend {
    type Target = CpuBackend;

    fn deref(&self) -> &Self::Target {
        &self.backend
    }
}

fn upload_f32(
    backend: &TestBackend,
    shape: &[u64],
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    Ok(backend
        .upload_f32(descriptor, values, &backend.execution(cancellation)?)?
        .0)
}

fn upload_bool(
    backend: &TestBackend,
    shape: &[u64],
    values: &[bool],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor = TensorDescriptor::contiguous(
        shape.to_vec(),
        DType::Bool,
        DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    let bytes = values.iter().copied().map(u8::from).collect::<Vec<_>>();
    Ok(backend
        .upload_bytes(descriptor, &bytes, &backend.execution(cancellation)?)?
        .0)
}

fn values(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let count = usize::try_from(tensor.descriptor().element_count()?)?;
    let mut result = Vec::with_capacity(count);
    for linear in 0..count {
        let mut remainder = linear;
        let mut indices = vec![0_u64; tensor.descriptor().rank()];
        for (index, dimension) in indices.iter_mut().zip(tensor.descriptor().shape()).rev() {
            let dimension = usize::try_from(*dimension)?;
            *index = u64::try_from(remainder % dimension)?;
            remainder /= dimension;
        }
        match tensor
            .descriptor()
            .dtype()
            .decode_scalar(tensor.element_bytes(&indices)?)?
        {
            DecodedScalar::Real(value) => result.push(value as f32),
            _ => return Err("expected a real tensor".into()),
        }
    }
    Ok(result)
}

fn assert_cancelled<T>(
    result: Result<T, LinearAlgebraPartTwoError>,
    context: &ExecutionContext<'_>,
) {
    assert!(matches!(result, Err(LinearAlgebraPartTwoError::Cancelled)));
    assert_eq!(context.scratch.peak_bytes(), 0);
    assert_eq!(context.scratch.in_use_bytes(), 0);
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index}: expected {expected}, got {actual}"
        );
    }
}

fn dot(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| f64::from(*left) * f64::from(*right))
        .sum()
}

#[test]
fn bmm_delegates_forward_vjp_and_jvp_to_canonical_backend() -> Result<(), Box<dyn std::error::Error>>
{
    let backend = TestBackend::new(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let input = upload_f32(&backend, &[1, 2, 2], &[1.0, 2.0, 3.0, 4.0], &cancellation)?;
    let mat2 = upload_f32(&backend, &[1, 2, 2], &[5.0, 6.0, 7.0, 8.0], &cancellation)?;
    let output = bmm_with_context_exact_native(&backend, &input, &mat2, &execution)?;
    assert_eq!(output.descriptor().shape(), &[1, 2, 2]);
    assert_ne!(output.storage_id(), input.storage_id());
    assert_close(&values(&output)?, &[19.0, 22.0, 43.0, 50.0], 1e-6);

    let output_gradient = upload_f32(&backend, &[1, 2, 2], &[1.0; 4], &cancellation)?;
    let gradients =
        bmm_vjp_with_context_exact_native(&backend, &input, &mat2, &output_gradient, &execution)?;
    assert_close(&values(&gradients.input)?, &[11.0, 15.0, 11.0, 15.0], 1e-6);
    assert_close(&values(&gradients.mat2)?, &[4.0, 4.0, 6.0, 6.0], 1e-6);

    let input_tangent = upload_f32(&backend, &[1, 2, 2], &[1.0; 4], &cancellation)?;
    let mat2_tangent = upload_f32(&backend, &[1, 2, 2], &[0.0; 4], &cancellation)?;
    let tangent = bmm_jvp_with_context_exact_native(
        &backend,
        &input,
        &mat2,
        &input_tangent,
        &mat2_tangent,
        &execution,
    )?;
    assert_close(&values(&tangent)?, &[12.0, 14.0, 12.0, 14.0], 1e-6);
    Ok(())
}

#[test]
fn norm_delegates_euclidean_reduction_and_derivatives() -> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let input = upload_f32(&backend, &[2, 2], &[3.0, 4.0, 0.0, 0.0], &cancellation)?;
    let norm =
        norm_with_context_exact_native(&backend, &input, 2.0, Some(&[-1]), true, None, &execution)?;
    assert_eq!(norm.descriptor().shape(), &[2, 1]);
    assert_close(&values(&norm)?, &[5.0, 0.0], 1e-6);
    let global =
        norm_with_context_exact_native(&backend, &input, 2.0, None, false, None, &execution)?;
    assert!(global.descriptor().shape().is_empty());
    assert_close(&values(&global)?, &[5.0], 1e-6);

    let upstream = upload_f32(&backend, &[2, 1], &[2.0, 7.0], &cancellation)?;
    let gradient = norm_vjp_with_context_exact_native(
        &backend,
        &input,
        &upstream,
        2.0,
        Some(&[-1]),
        true,
        None,
        &execution,
    )?;
    assert_close(&values(&gradient)?, &[1.2, 1.6, 0.0, 0.0], 1e-6);
    let input_tangent = upload_f32(&backend, &[2, 2], &[1.0, 2.0, 3.0, 4.0], &cancellation)?;
    let tangent = norm_jvp_with_context_exact_native(
        &backend,
        &input,
        &input_tangent,
        2.0,
        Some(&[-1]),
        true,
        None,
        &execution,
    )?;
    assert_close(&values(&tangent)?, &[2.2, 0.0], 1e-6);
    Ok(())
}

#[test]
fn svd_reconstructs_and_has_analytic_distinct_spectrum_derivatives()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let input = upload_f32(&backend, &[2, 2], &[3.0, 0.0, 0.0, 2.0], &cancellation)?;
    let decomposition = svd_with_context_exact_native(&backend, &input, false, &execution)?;
    assert_eq!(decomposition.u.descriptor().shape(), &[2, 2]);
    assert_eq!(decomposition.s.descriptor().shape(), &[2]);
    assert_eq!(decomposition.vh.descriptor().shape(), &[2, 2]);
    assert_close(&values(&decomposition.s)?, &[3.0, 2.0], 1e-5);
    let u = values(&decomposition.u)?;
    let singular = values(&decomposition.s)?;
    let vh = values(&decomposition.vh)?;
    let reconstructed = [
        u[0] * singular[0] * vh[0] + u[1] * singular[1] * vh[2],
        u[0] * singular[0] * vh[1] + u[1] * singular[1] * vh[3],
        u[2] * singular[0] * vh[0] + u[3] * singular[1] * vh[2],
        u[2] * singular[0] * vh[1] + u[3] * singular[1] * vh[3],
    ];
    assert_close(&reconstructed, &[3.0, 0.0, 0.0, 2.0], 2e-5);

    let input_tangent = upload_f32(&backend, &[2, 2], &[0.5, 0.0, 0.0, -0.25], &cancellation)?;
    let tangent =
        svd_jvp_with_context_exact_native(&backend, &input, false, &input_tangent, &execution)?;
    assert_close(&values(&tangent.s)?, &[0.5, -0.25], 2e-5);
    assert_close(&values(&tangent.u)?, &[0.0; 4], 2e-5);
    assert_close(&values(&tangent.vh)?, &[0.0; 4], 2e-5);

    let u_gradient = upload_f32(&backend, &[2, 2], &[0.0; 4], &cancellation)?;
    let s_gradient = upload_f32(&backend, &[2], &[2.0, 4.0], &cancellation)?;
    let vh_gradient = upload_f32(&backend, &[2, 2], &[0.0; 4], &cancellation)?;
    let gradient = svd_vjp_with_context_exact_native(
        &backend,
        &input,
        false,
        &u_gradient,
        &s_gradient,
        &vh_gradient,
        &execution,
    )?;
    assert_close(&values(&gradient)?, &[2.0, 0.0, 0.0, 4.0], 2e-5);
    Ok(())
}

#[test]
fn canonical_svd_leases_exact_workspace_for_strided_forward_and_derivatives()
-> Result<(), Box<dyn std::error::Error>> {
    const GENEROUS: u64 = 1024 * 1024;

    let backend = TestBackend::new(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let contiguous = upload_f32(&backend, &[2, 2], &[3.0, 0.0, 0.0, 2.0], &cancellation)?;
    let input = transpose_last_two_with_context_exact_native(
        &contiguous,
        &backend.execution(&cancellation)?,
    )?;
    let tangent = upload_f32(&backend, &[2, 2], &[0.5, 0.0, 0.0, -0.25], &cancellation)?;

    let forward_probe = backend.execution_with_workspace(GENEROUS, &cancellation)?;
    let decomposition = svd_with_context_exact_native(&backend, &input, false, &forward_probe)?;
    assert_close(&values(&decomposition.s)?, &[3.0, 2.0], 1.0e-5);
    assert_ne!(decomposition.u.storage_id(), input.storage_id());
    let forward_peak = forward_probe.scratch.peak_bytes();
    assert!(forward_peak > 0);
    assert_eq!(forward_probe.scratch.in_use_bytes(), 0);

    let forward_exact = backend.execution_with_workspace(forward_peak, &cancellation)?;
    svd_with_context_exact_native(&backend, &input, false, &forward_exact)?;
    assert_eq!(forward_exact.scratch.peak_bytes(), forward_peak);
    let forward_insufficient = backend.execution_with_workspace(forward_peak - 1, &cancellation)?;
    assert!(svd_with_context_exact_native(&backend, &input, false, &forward_insufficient).is_err());
    assert_eq!(forward_insufficient.scratch.in_use_bytes(), 0);

    let jvp_probe = backend.execution_with_workspace(GENEROUS, &cancellation)?;
    let directional =
        svd_jvp_with_context_exact_native(&backend, &input, false, &tangent, &jvp_probe)?;
    assert_close(&values(&directional.s)?, &[0.5, -0.25], 2.0e-5);
    let jvp_peak = jvp_probe.scratch.peak_bytes();
    assert!(jvp_peak >= forward_peak);
    assert_eq!(jvp_probe.scratch.in_use_bytes(), 0);
    let jvp_exact = backend.execution_with_workspace(jvp_peak, &cancellation)?;
    svd_jvp_with_context_exact_native(&backend, &input, false, &tangent, &jvp_exact)?;
    assert_eq!(jvp_exact.scratch.peak_bytes(), jvp_peak);

    let u_gradient = upload_f32(&backend, &[2, 2], &[0.0; 4], &cancellation)?;
    let s_gradient = upload_f32(&backend, &[2], &[2.0, 4.0], &cancellation)?;
    let vh_gradient = upload_f32(&backend, &[2, 2], &[0.0; 4], &cancellation)?;
    let vjp_probe = backend.execution_with_workspace(GENEROUS, &cancellation)?;
    let gradient = svd_vjp_with_context_exact_native(
        &backend,
        &input,
        false,
        &u_gradient,
        &s_gradient,
        &vh_gradient,
        &vjp_probe,
    )?;
    assert_close(&values(&gradient)?, &[2.0, 0.0, 0.0, 4.0], 2.0e-5);
    let vjp_peak = vjp_probe.scratch.peak_bytes();
    assert!(vjp_peak >= forward_peak);
    assert_eq!(vjp_probe.scratch.in_use_bytes(), 0);
    let vjp_exact = backend.execution_with_workspace(vjp_peak, &cancellation)?;
    svd_vjp_with_context_exact_native(
        &backend,
        &input,
        false,
        &u_gradient,
        &s_gradient,
        &vh_gradient,
        &vjp_exact,
    )?;
    assert_eq!(vjp_exact.scratch.peak_bytes(), vjp_peak);

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = backend.execution_with_workspace(vjp_peak, &cancelled)?;
    assert!(
        svd_vjp_with_context_exact_native(
            &backend,
            &input,
            false,
            &u_gradient,
            &s_gradient,
            &vh_gradient,
            &cancelled_context,
        )
        .is_err()
    );
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn canonical_bmm_uses_zero_scratch_and_norm_delegates_to_leased_owner()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(&backend, &[1, 2, 2], &[1.0, 2.0, 3.0, 4.0], &cancellation)?;
    let mat2 = upload_f32(&backend, &[1, 2, 2], &[5.0, 6.0, 7.0, 8.0], &cancellation)?;
    let zero = backend.execution_with_workspace(0, &cancellation)?;
    let output = bmm_with_context_exact_native(&backend, &input, &mat2, &zero)?;
    assert_close(&values(&output)?, &[19.0, 22.0, 43.0, 50.0], 1.0e-6);
    assert_eq!(zero.scratch.peak_bytes(), 0);

    let matrix = upload_f32(&backend, &[2, 2], &[3.0, 4.0, 5.0, 12.0], &cancellation)?;
    let norm_context = backend.execution_with_workspace(48, &cancellation)?;
    let norms = norm_with_context_exact_native(
        &backend,
        &matrix,
        2.0,
        Some(&[1]),
        false,
        None,
        &norm_context,
    )?;
    assert_close(&values(&norms)?, &[5.0, 13.0], 1.0e-6);
    assert_eq!(norm_context.scratch.peak_bytes(), 48);
    assert_eq!(norm_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn svd_vjp_is_the_adjoint_of_its_jvp() -> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let input = upload_f32(&backend, &[2, 2], &[3.0, 1.0, 0.0, 2.0], &cancellation)?;
    let input_tangent = upload_f32(&backend, &[2, 2], &[0.2, -0.1, 0.3, 0.4], &cancellation)?;
    let u_gradient = upload_f32(&backend, &[2, 2], &[0.1, -0.2, 0.3, 0.4], &cancellation)?;
    let s_gradient = upload_f32(&backend, &[2], &[0.5, -0.7], &cancellation)?;
    let vh_gradient = upload_f32(&backend, &[2, 2], &[-0.1, 0.2, 0.6, -0.3], &cancellation)?;
    let tangent =
        svd_jvp_with_context_exact_native(&backend, &input, false, &input_tangent, &execution)?;
    let gradient = svd_vjp_with_context_exact_native(
        &backend,
        &input,
        false,
        &u_gradient,
        &s_gradient,
        &vh_gradient,
        &execution,
    )?;
    let left = dot(&values(&tangent.u)?, &values(&u_gradient)?)
        + dot(&values(&tangent.s)?, &values(&s_gradient)?)
        + dot(&values(&tangent.vh)?, &values(&vh_gradient)?);
    let right = dot(&values(&input_tangent)?, &values(&gradient)?);
    assert!(
        (left - right).abs() <= 2e-4,
        "adjoint mismatch: {left} != {right}"
    );
    Ok(())
}

#[test]
fn svd_supports_batches_and_full_rectangular_bases() -> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let batched = upload_f32(
        &backend,
        &[2, 2, 2],
        &[3.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 1.0],
        &cancellation,
    )?;
    let batched = svd_with_context_exact_native(&backend, &batched, false, &execution)?;
    assert_eq!(batched.u.descriptor().shape(), &[2, 2, 2]);
    assert_eq!(batched.s.descriptor().shape(), &[2, 2]);
    assert_eq!(batched.vh.descriptor().shape(), &[2, 2, 2]);
    assert_close(&values(&batched.s)?, &[3.0, 2.0, 4.0, 1.0], 1e-5);

    let tall = upload_f32(
        &backend,
        &[3, 2],
        &[3.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        &cancellation,
    )?;
    let full = svd_with_context_exact_native(&backend, &tall, true, &execution)?;
    assert_eq!(full.u.descriptor().shape(), &[3, 3]);
    assert_eq!(full.s.descriptor().shape(), &[2]);
    assert_eq!(full.vh.descriptor().shape(), &[2, 2]);
    let reduced = svd_with_context_exact_native(&backend, &tall, false, &execution)?;
    assert_eq!(reduced.u.descriptor().shape(), &[3, 2]);
    assert_eq!(reduced.s.descriptor().shape(), &[2]);
    assert_eq!(reduced.vh.descriptor().shape(), &[2, 2]);

    let wide = upload_f32(
        &backend,
        &[2, 3],
        &[3.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        &cancellation,
    )?;
    let full = svd_with_context_exact_native(&backend, &wide, true, &execution)?;
    assert_eq!(full.u.descriptor().shape(), &[2, 2]);
    assert_eq!(full.s.descriptor().shape(), &[2]);
    assert_eq!(full.vh.descriptor().shape(), &[3, 3]);
    let reduced = svd_with_context_exact_native(&backend, &wide, false, &execution)?;
    assert_eq!(reduced.u.descriptor().shape(), &[2, 2]);
    assert_eq!(reduced.s.descriptor().shape(), &[2]);
    assert_eq!(reduced.vh.descriptor().shape(), &[2, 3]);
    Ok(())
}

#[test]
fn linear_algebra_boundaries_are_typed_and_checked() -> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(16 * 1024 * 1024)?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let rank_one = upload_f32(&backend, &[2], &[1.0, 2.0], &cancellation)?;
    let rank_two = upload_f32(&backend, &[1, 2], &[1.0, 2.0], &cancellation)?;
    assert!(bmm_with_context_exact_native(&backend, &rank_one, &rank_one, &execution).is_err());
    assert!(svd_with_context_exact_native(&backend, &rank_one, false, &execution).is_err());
    assert!(
        norm_with_context_exact_native(
            &backend,
            &rank_two,
            2.0,
            Some(&[0, 0]),
            false,
            None,
            &execution,
        )
        .is_err()
    );
    let boolean = upload_bool(&backend, &[1, 1], &[true], &cancellation)?;
    assert!(
        norm_with_context_exact_native(&backend, &boolean, 2.0, None, false, None, &execution,)
            .is_err()
    );
    assert!(svd_with_context_exact_native(&backend, &boolean, false, &execution).is_err());
    Ok(())
}

#[test]
fn cancellation_precedes_invalid_linear_algebra_inputs() -> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new(16 * 1024 * 1024)?;
    let setup = CancellationToken::default();
    let invalid = upload_f32(&backend, &[1], &[1.0], &setup)?;
    let invalid_before = invalid.contiguous_bytes()?.to_vec();
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let execution = backend.execution(&cancellation)?;

    assert_cancelled(
        bmm_with_context_exact_native(&backend, &invalid, &invalid, &execution),
        &execution,
    );
    assert_cancelled(
        bmm_vjp_with_context_exact_native(
            &backend,
            &invalid,
            &invalid,
            &invalid,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        bmm_jvp_with_context_exact_native(
            &backend,
            &invalid,
            &invalid,
            &invalid,
            &invalid,
            &execution,
        ),
        &execution,
    );

    assert_cancelled(
        norm_with_context_exact_native(
            &backend,
            &invalid,
            2.0,
            Some(&[0, 0]),
            false,
            None,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        norm_vjp_with_context_exact_native(
            &backend,
            &invalid,
            &invalid,
            2.0,
            Some(&[0, 0]),
            false,
            None,
            &execution,
        ),
        &execution,
    );
    assert_cancelled(
        norm_jvp_with_context_exact_native(
            &backend,
            &invalid,
            &invalid,
            2.0,
            Some(&[0, 0]),
            false,
            None,
            &execution,
        ),
        &execution,
    );

    assert_cancelled(
        svd_with_context_exact_native(&backend, &invalid, false, &execution),
        &execution,
    );
    assert_cancelled(
        svd_jvp_with_context_exact_native(&backend, &invalid, false, &invalid, &execution),
        &execution,
    );
    assert_cancelled(
        svd_vjp_with_context_exact_native(
            &backend,
            &invalid,
            false,
            &invalid,
            &invalid,
            &invalid,
            &execution,
        ),
        &execution,
    );

    assert_eq!(invalid.contiguous_bytes()?, invalid_before);
    Ok(())
}

#[test]
fn resolution_slice_is_unique_and_fixtures_are_runtime_sealed()
-> Result<(), Box<dyn std::error::Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "linear_algebra_02")
        .ok_or("linear-algebra part-two resolution slice is missing")?;
    assert_eq!(slice.len(), 3);
    assert_eq!(
        slice
            .iter()
            .map(|contract| contract.operation_id)
            .collect::<BTreeSet<_>>(),
        [
            "COMFY-TENSOR-OP-A5D623C79A18",
            "COMFY-TENSOR-OP-B42F17255D7D",
            "COMFY-TENSOR-OP-C31767F422EE",
        ]
        .into_iter()
        .collect()
    );
    let owner = "comfy-parity-tensor-ops-linear-algebra-comfy-tensor-op-a5d623c79a18";
    for contract in slice.iter() {
        assert_eq!(contract.owner_task_id, owner);
        let fixture = fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(contract.evidence_fixture),
        )?;
        assert_eq!(
            format!("{:x}", Sha256::digest(fixture)),
            contract.evidence_fixture_sha256
        );
    }
    Ok(())
}
