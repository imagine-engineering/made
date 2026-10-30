use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DecodedScalar, DeviceId,
    ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, StreamId, Tensor,
    TensorDescriptor, TensorError, ViewAccess,
    generated_spectral_transform_01::{
        FFTN_OPERATION_ID, FFTSHIFT_OPERATION_ID, IFFTN_OPERATION_ID, IFFTSHIFT_OPERATION_ID,
        SpectralTransformError, fftn_jvp_with_context_exact_native,
        fftn_vjp_with_context_exact_native, fftn_with_context_exact_native,
        fftshift_jvp_with_context_exact_native, fftshift_vjp_with_context_exact_native,
        fftshift_with_context_exact_native, ifftn_jvp_with_context_exact_native,
        ifftn_vjp_with_context_exact_native, ifftn_with_context_exact_native,
        ifftshift_jvp_with_context_exact_native, ifftshift_vjp_with_context_exact_native,
        ifftshift_with_context_exact_native,
    },
};
use std::{collections::BTreeSet, fs, path::Path};

const IDS: [&str; 4] = [
    FFTN_OPERATION_ID,
    FFTSHIFT_OPERATION_ID,
    IFFTN_OPERATION_ID,
    IFFTSHIFT_OPERATION_ID,
];

struct TestBackend {
    backend: CpuBackend,
    authority: CpuWorkspaceAuthority,
    limit: u64,
}

impl TestBackend {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let limit = 64 * 1024 * 1024;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(limit)?;
        Ok(Self {
            backend,
            authority,
            limit,
        })
    }

    fn execution<'a>(
        &self,
        cancellation: &'a CancellationToken,
    ) -> Result<ExecutionContext<'a>, Box<dyn std::error::Error>> {
        self.execution_with_limit(cancellation, self.limit)
    }

    fn execution_with_limit<'a>(
        &self,
        cancellation: &'a CancellationToken,
        limit: u64,
    ) -> Result<ExecutionContext<'a>, Box<dyn std::error::Error>> {
        Ok(self.backend.execution_context(
            StreamId::DEFAULT,
            self.authority.authorize_workspace(limit)?,
            cancellation,
        ))
    }

    fn upload_f32(
        &self,
        shape: &[u64],
        values: &[f32],
        cancellation: &CancellationToken,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let descriptor = TensorDescriptor::contiguous(
            shape.to_vec(),
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        Ok(self
            .backend
            .upload_f32(descriptor, values, &self.execution(cancellation)?)?
            .0)
    }

    fn upload_complex64(
        &self,
        shape: &[u64],
        values: &[(f32, f32)],
        cancellation: &CancellationToken,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for &(real, imaginary) in values {
            bytes.extend(DType::Complex64.encode_decoded_scalar(
                DecodedScalar::Complex {
                    real: f64::from(real),
                    imaginary: f64::from(imaginary),
                },
                "task-90-test-upload",
                DeviceId::CPU,
            )?);
        }
        let descriptor = TensorDescriptor::contiguous(
            shape.to_vec(),
            DType::Complex64,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        Ok(self
            .backend
            .upload_bytes(descriptor, &bytes, &self.execution(cancellation)?)?
            .0)
    }
}

fn complex_values(tensor: &Tensor) -> Result<Vec<(f32, f32)>, Box<dyn std::error::Error>> {
    let count = tensor.descriptor().element_count()?;
    let mut values = Vec::with_capacity(usize::try_from(count)?);
    for linear_index in 0..count {
        let DecodedScalar::Complex { real, imaginary } = tensor
            .descriptor()
            .dtype()
            .decode_scalar(tensor.linear_element_bytes(linear_index)?)?
        else {
            return Err("expected complex tensor".into());
        };
        values.push((real as f32, imaginary as f32));
    }
    Ok(values)
}

fn real_values(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let count = tensor.descriptor().element_count()?;
    let mut values = Vec::with_capacity(usize::try_from(count)?);
    for linear_index in 0..count {
        let DecodedScalar::Real(value) = tensor
            .descriptor()
            .dtype()
            .decode_scalar(tensor.linear_element_bytes(linear_index)?)?
        else {
            return Err("expected real tensor".into());
        };
        values.push(value as f32);
    }
    Ok(values)
}

#[test]
fn double_precision_fft_retains_sub_f32_values_and_its_adjoint()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let input_values = [1.0_f64 + 1e-10, 2.0, -3.0, 4.0];
    let descriptor =
        TensorDescriptor::contiguous(vec![4], DType::F64, DeviceId::CPU, StreamId::DEFAULT)?;
    let bytes = input_values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let input = backend
        .backend
        .upload_bytes(descriptor, &bytes, &execution)?
        .0;
    let spectrum = fftn_with_context_exact_native(&backend.backend, &input, &[0], &execution)?;
    assert_eq!(spectrum.descriptor().dtype(), DType::Complex128);
    let expected = [
        (4.0 + 1e-10, 0.0),
        (4.0 + 1e-10, 2.0),
        (-8.0 + 1e-10, 0.0),
        (4.0 + 1e-10, -2.0),
    ];
    for (index, (expected_real, expected_imaginary)) in expected.into_iter().enumerate() {
        let DecodedScalar::Complex { real, imaginary } = DType::Complex128
            .decode_scalar(spectrum.linear_element_bytes(u64::try_from(index)?)?)?
        else {
            return Err("expected double precision spectrum".into());
        };
        assert!((real - expected_real).abs() <= 1e-12);
        assert!((imaginary - expected_imaginary).abs() <= 1e-12);
    }
    let restored = ifftn_with_context_exact_native(&backend.backend, &spectrum, &[0], &execution)?;
    let adjoint =
        fftn_vjp_with_context_exact_native(&backend.backend, &input, &spectrum, &[0], &execution)?;
    assert_eq!(adjoint.descriptor().dtype(), DType::F64);
    for (index, expected) in input_values.into_iter().enumerate() {
        let index = u64::try_from(index)?;
        let DecodedScalar::Complex { real, imaginary } =
            DType::Complex128.decode_scalar(restored.linear_element_bytes(index)?)?
        else {
            return Err("expected double precision inverse".into());
        };
        let DecodedScalar::Real(gradient) =
            DType::F64.decode_scalar(adjoint.linear_element_bytes(index)?)?
        else {
            return Err("expected double precision gradient".into());
        };
        assert!((real - expected).abs() <= 1e-12);
        assert!(imaginary.abs() <= 1e-12);
        assert!((gradient - 4.0 * expected).abs() <= 1e-12);
    }
    let wrong_gradient = backend.upload_complex64(&[4], &[(1.0, 0.0); 4], &cancellation)?;
    assert!(matches!(
        fftn_vjp_with_context_exact_native(
            &backend.backend,
            &input,
            &wrong_gradient,
            &[0],
            &execution
        ),
        Err(SpectralTransformError::UnsupportedDType { .. })
    ));
    let constrained = backend.execution_with_limit(&cancellation, 127)?;
    assert!(fftn_with_context_exact_native(&backend.backend, &input, &[0], &constrained).is_err());
    assert_eq!(constrained.scratch.in_use_bytes(), 0);
    Ok(())
}

fn close_complex(actual: &[(f32, f32)], expected: &[(f32, f32)], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual.0 - expected.0).abs() <= tolerance);
        assert!((actual.1 - expected.1).abs() <= tolerance);
    }
}

fn real_inner(left: &[(f32, f32)], right: &[(f32, f32)]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.0 * right.0 + left.1 * right.1)
        .sum()
}

#[test]
fn two_dimensional_freelunch_sequence_round_trips_exactly() -> Result<(), Box<dyn std::error::Error>>
{
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let input = backend.upload_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0], &cancellation)?;
    let spectrum = fftn_with_context_exact_native(&backend.backend, &input, &[-2, -1], &execution)?;
    close_complex(
        &complex_values(&spectrum)?,
        &[(10.0, 0.0), (-2.0, 0.0), (-4.0, 0.0), (0.0, 0.0)],
        1e-5,
    );
    assert_ne!(input.storage_id(), spectrum.storage_id());
    assert!(spectrum.descriptor().is_contiguous()?);

    let shifted =
        fftshift_with_context_exact_native(&backend.backend, &spectrum, &[-2, -1], &execution)?;
    close_complex(
        &complex_values(&shifted)?,
        &[(0.0, 0.0), (-4.0, 0.0), (-2.0, 0.0), (10.0, 0.0)],
        1e-5,
    );
    let unshifted =
        ifftshift_with_context_exact_native(&backend.backend, &shifted, &[-2, -1], &execution)?;
    let restored =
        ifftn_with_context_exact_native(&backend.backend, &unshifted, &[-2, -1], &execution)?;
    close_complex(
        &complex_values(&restored)?,
        &[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
        1e-5,
    );
    Ok(())
}

#[test]
fn non_power_of_two_fft_and_odd_shifts_match_reference_values()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let input = backend.upload_f32(&[3], &[1.0, 2.0, 3.0], &cancellation)?;
    let spectrum = fftn_with_context_exact_native(&backend.backend, &input, &[0], &execution)?;
    close_complex(
        &complex_values(&spectrum)?,
        &[(6.0, 0.0), (-1.5, 0.866_025_4), (-1.5, -0.866_025_4)],
        2e-5,
    );

    let odd = backend.upload_f32(&[5], &[0.0, 1.0, 2.0, 3.0, 4.0], &cancellation)?;
    let shifted = fftshift_with_context_exact_native(&backend.backend, &odd, &[0], &execution)?;
    assert_eq!(real_values(&shifted)?, [3.0, 4.0, 0.0, 1.0, 2.0]);
    let restored =
        ifftshift_with_context_exact_native(&backend.backend, &shifted, &[0], &execution)?;
    assert_eq!(real_values(&restored)?, [0.0, 1.0, 2.0, 3.0, 4.0]);

    let promoted = fftn_with_context_exact_native(&backend.backend, &odd, &[], &execution)?;
    close_complex(
        &complex_values(&promoted)?,
        &[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
        0.0,
    );
    Ok(())
}

#[test]
fn logical_strided_input_is_transformed_without_copying_descriptor_rules()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let contiguous = backend.upload_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &cancellation)?;
    let input = contiguous.view(
        contiguous.descriptor().permuted_view(&[1, 0])?,
        ViewAccess::ReadOnly,
    )?;
    assert!(!input.descriptor().is_contiguous()?);
    let spectrum = fftn_with_context_exact_native(&backend.backend, &input, &[0, 1], &execution)?;
    let restored =
        ifftn_with_context_exact_native(&backend.backend, &spectrum, &[0, 1], &execution)?;
    close_complex(
        &complex_values(&restored)?,
        &[
            (1.0, 0.0),
            (4.0, 0.0),
            (2.0, 0.0),
            (5.0, 0.0),
            (3.0, 0.0),
            (6.0, 0.0),
        ],
        2e-5,
    );
    assert!(spectrum.descriptor().is_contiguous()?);
    Ok(())
}

#[test]
fn non_power_of_two_workspace_is_exact_failure_atomic_and_convergent()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let input = backend.upload_f32(&[3], &[1.0, 2.0, 3.0], &cancellation)?;
    let baseline = backend.backend.memory_snapshot().current_bytes;
    let underauthorized = backend.execution_with_limit(&cancellation, 71)?;
    assert!(matches!(
        fftn_with_context_exact_native(&backend.backend, &input, &[0], &underauthorized),
        Err(SpectralTransformError::Tensor(
            TensorError::WorkspaceAuthorizationExceeded {
                requested: 24,
                authorized: 71,
                in_use: 48,
            }
        ))
    ));
    assert_eq!(backend.backend.memory_snapshot().current_bytes, baseline);

    let exact = backend.execution_with_limit(&cancellation, 72)?;
    let output = fftn_with_context_exact_native(&backend.backend, &input, &[0], &exact)?;
    close_complex(
        &complex_values(&output)?,
        &[(6.0, 0.0), (-1.5, 0.866_025_4), (-1.5, -0.866_025_4)],
        2e-5,
    );
    drop(output);
    assert_eq!(backend.backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn fft_and_ifft_analytic_jvp_vjp_are_adjoint() -> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let input = backend.upload_f32(&[2, 2], &[0.5, -1.0, 2.0, 0.25], &cancellation)?;
    let tangent = backend.upload_f32(&[2, 2], &[0.2, -0.1, 0.4, 0.3], &cancellation)?;
    let cotangent = backend.upload_complex64(
        &[2, 2],
        &[(0.5, -0.2), (-0.4, 0.3), (0.1, 0.7), (0.8, -0.6)],
        &cancellation,
    )?;
    let jvp = fftn_jvp_with_context_exact_native(
        &backend.backend,
        &input,
        &tangent,
        &[0, 1],
        &execution,
    )?;
    let vjp = fftn_vjp_with_context_exact_native(
        &backend.backend,
        &input,
        &cotangent,
        &[0, 1],
        &execution,
    )?;
    let lhs = real_inner(&complex_values(&jvp)?, &complex_values(&cotangent)?);
    let rhs: f32 = real_values(&tangent)?
        .iter()
        .zip(real_values(&vjp)?)
        .map(|(left, right)| left * right)
        .sum();
    assert!((lhs - rhs).abs() <= 2e-5, "{lhs} != {rhs}");

    let complex_input = backend.upload_complex64(
        &[3],
        &[(1.0, 0.5), (-0.25, 0.75), (2.0, -1.0)],
        &cancellation,
    )?;
    let complex_tangent =
        backend.upload_complex64(&[3], &[(0.2, 0.1), (0.4, -0.3), (-0.5, 0.7)], &cancellation)?;
    let complex_cotangent =
        backend.upload_complex64(&[3], &[(0.8, -0.2), (0.3, 0.9), (-0.6, 0.1)], &cancellation)?;
    let inverse_jvp = ifftn_jvp_with_context_exact_native(
        &backend.backend,
        &complex_input,
        &complex_tangent,
        &[0],
        &execution,
    )?;
    let inverse_vjp = ifftn_vjp_with_context_exact_native(
        &backend.backend,
        &complex_input,
        &complex_cotangent,
        &[0],
        &execution,
    )?;
    let lhs = real_inner(
        &complex_values(&inverse_jvp)?,
        &complex_values(&complex_cotangent)?,
    );
    let rhs = real_inner(
        &complex_values(&complex_tangent)?,
        &complex_values(&inverse_vjp)?,
    );
    assert!((lhs - rhs).abs() <= 2e-5, "{lhs} != {rhs}");
    Ok(())
}

#[test]
fn shift_jvp_and_vjp_use_the_forward_and_inverse_permutations()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let tangent = backend.upload_f32(&[5], &[1.0, 2.0, 3.0, 4.0, 5.0], &cancellation)?;
    let forward =
        fftshift_jvp_with_context_exact_native(&backend.backend, &tangent, &[0], &execution)?;
    assert_eq!(real_values(&forward)?, [4.0, 5.0, 1.0, 2.0, 3.0]);
    let backward =
        fftshift_vjp_with_context_exact_native(&backend.backend, &forward, &[0], &execution)?;
    assert_eq!(real_values(&backward)?, [1.0, 2.0, 3.0, 4.0, 5.0]);
    let inverse_forward =
        ifftshift_jvp_with_context_exact_native(&backend.backend, &tangent, &[0], &execution)?;
    let inverse_backward = ifftshift_vjp_with_context_exact_native(
        &backend.backend,
        &inverse_forward,
        &[0],
        &execution,
    )?;
    assert_eq!(real_values(&inverse_backward)?, [1.0, 2.0, 3.0, 4.0, 5.0]);
    Ok(())
}

#[test]
fn dimensions_dtypes_empty_axes_and_cancellation_are_checked()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let execution = backend.execution(&cancellation)?;
    let input = backend.upload_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0], &cancellation)?;
    assert!(matches!(
        fftn_with_context_exact_native(&backend.backend, &input, &[0, -2], &execution),
        Err(SpectralTransformError::Invalid { .. })
    ));
    assert!(matches!(
        fftn_with_context_exact_native(&backend.backend, &input, &[2], &execution),
        Err(SpectralTransformError::Invalid { .. })
    ));
    assert!(matches!(
        ifftn_with_context_exact_native(&backend.backend, &input, &[0], &execution),
        Err(SpectralTransformError::UnsupportedDType { .. })
    ));
    let empty = backend.upload_f32(&[0, 2], &[], &cancellation)?;
    assert!(matches!(
        fftn_with_context_exact_native(&backend.backend, &empty, &[0], &execution),
        Err(SpectralTransformError::Invalid { .. })
    ));

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_execution = backend.execution(&cancelled)?;
    assert!(matches!(
        ifftn_with_context_exact_native(&backend.backend, &input, &[99], &cancelled_execution),
        Err(SpectralTransformError::Cancelled)
    ));
    Ok(())
}

#[test]
fn task_55_is_the_only_fft_kernel_owner() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root unavailable")?;
    let task_55 = fs::read_to_string(
        root.join("crates/sim_tensor/src/ops/elementwise_or_runtime_operation_12.rs"),
    )?;
    let task_90 =
        fs::read_to_string(root.join("crates/sim_tensor/src/ops/spectral_transform_01.rs"))?;
    assert_eq!(task_55.matches("fn complex_fft_in_place<").count(), 1);
    assert!(task_55.contains("complex_fft_in_place(backend, &mut frame_values, false, context)"));
    assert!(task_90.contains("complex_fft_in_place(backend, &mut line, inverse, context)"));
    assert!(!task_90.contains("consts::TAU"));
    Ok(())
}

#[test]
fn all_four_resolutions_are_unique_and_runtime_hash_sealed()
-> Result<(), Box<dyn std::error::Error>> {
    let contracts = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .flat_map(|slice| slice.iter())
        .filter(|contract| contract.resolution_module == "spectral_transform_01")
        .collect::<Vec<_>>();
    assert_eq!(contracts.len(), IDS.len());
    let operation_ids = contracts
        .iter()
        .map(|contract| contract.operation_id)
        .collect::<BTreeSet<_>>();
    assert_eq!(operation_ids, IDS.into_iter().collect());
    assert_eq!(
        contracts
            .iter()
            .map(|contract| contract.overload_id)
            .collect::<BTreeSet<_>>()
            .len(),
        IDS.len()
    );
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root unavailable")?;
    for contract in contracts {
        assert_eq!(
            contract.owner_task_id,
            "comfy-parity-tensor-ops-spectral-transform-comfy-tensor-op-2c39e32acd3c"
        );
        assert_ne!(
            contract.baseline_fixture_sha256,
            contract.evidence_fixture_sha256
        );
        let bytes = fs::read(root.join(contract.evidence_fixture))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            contract.evidence_fixture_sha256
        );
        let fixture: serde_json::Value = serde_json::from_slice(&bytes)?;
        assert_eq!(
            fixture
                .get("operation_id")
                .and_then(serde_json::Value::as_str),
            Some(contract.operation_id)
        );
    }
    Ok(())
}
