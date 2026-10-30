use std::{collections::BTreeSet, error::Error, fs, ops::Deref, path::Path};

use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, ResizeMode, StreamId, Tensor, TensorDescriptor,
    generated_external_tensor_kernel_01::{
        ExternalTensorKernelPartOneError, NativeBilinearBoundary, NativeBiquadCoefficients,
        NativeLinearBoundary, NativeMelNormalization, NativeMelScale, NativeMelScaleConfiguration,
        NativeMelSpectrogramConfiguration, NativeMorphologyOperation, NativeRearrangePlan,
        NativeResampleConfiguration, RoiAlignConfiguration, biquad_with_context_exact_native,
        checked_bilinear_weights, checked_linear_weights,
        equalizer_biquad_with_context_exact_native,
        mel_scale_project_vjp_with_context_exact_native,
        mel_scale_project_with_context_exact_native, mel_spectrogram_with_context_exact_native,
        native_morphology_with_context_exact, normalize_jvp_with_context_exact_native,
        normalize_vjp_with_context_exact_native, normalize_with_context_exact_native,
        rearrange_jvp_with_context_exact_native,
        rearrange_jvp_with_context_exact_native_for_operation,
        rearrange_tensor_with_context_exact_native_for_operation,
        rearrange_vjp_with_context_exact_native,
        rearrange_vjp_with_context_exact_native_for_operation, rearrange_with_context_exact_native,
        rearrange_with_context_exact_native_for_operation, resample_jvp_with_context_exact_native,
        resample_vjp_with_context_exact_native, resample_with_context_exact_native,
        resize_with_context_exact_native, roi_align_jvp_with_context_exact_native,
        roi_align_vjp_with_context_exact_native, roi_align_with_context_exact_native,
        to_tensor_with_context_exact_native, treble_biquad_with_context_exact_native,
    },
};

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-41E0A53BDA36",
    "COMFY-TENSOR-OP-338E50E9975A",
    "COMFY-TENSOR-OP-165C20AC8DD8",
    "COMFY-TENSOR-OP-363BE404A764",
    "COMFY-TENSOR-OP-0607DAA06439",
    "COMFY-TENSOR-OP-0A14AB1C4005",
    "COMFY-TENSOR-OP-49A168D86220",
    "COMFY-TENSOR-OP-367BF5D133D8",
    "COMFY-TENSOR-OP-0ABA532316FA",
    "COMFY-TENSOR-OP-0AB66ED3B4C2",
    "COMFY-TENSOR-OP-0882F83B464A",
    "COMFY-TENSOR-OP-2799F344E971",
];

struct TestBackend {
    backend: CpuBackend,
    workspace_authority: CpuWorkspaceAuthority,
}

impl Deref for TestBackend {
    type Target = CpuBackend;

    fn deref(&self) -> &Self::Target {
        &self.backend
    }
}

fn context<'a>(
    backend: &TestBackend,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, Box<dyn Error>> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        backend.workspace_authority.authorize_workspace(0)?,
        cancellation,
    ))
}

fn authorized_context<'a>(
    backend: &TestBackend,
    cancellation: &'a CancellationToken,
    bytes: u64,
) -> Result<ExecutionContext<'a>, Box<dyn Error>> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        backend.workspace_authority.authorize_workspace(bytes)?,
        cancellation,
    ))
}

fn backend_with_limit(memory_limit_bytes: u64) -> Result<TestBackend, Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(memory_limit_bytes)?;
    Ok(TestBackend {
        backend,
        workspace_authority,
    })
}

fn backend() -> Result<TestBackend, Box<dyn Error>> {
    backend_with_limit(64 * 1024 * 1024)
}

fn upload_f32(
    backend: &TestBackend,
    shape: &[u64],
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    Ok(backend
        .upload_f32(descriptor, values, &context(backend, cancellation)?)?
        .0)
}

fn upload_u8(
    backend: &TestBackend,
    shape: &[u64],
    values: &[u8],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::U8, DeviceId::CPU, StreamId::DEFAULT)?;
    Ok(backend
        .upload_bytes(descriptor, values, &context(backend, cancellation)?)?
        .0)
}

fn f32_values(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn Error>> {
    tensor
        .contiguous_bytes()?
        .chunks_exact(4)
        .map(|bytes| Ok(f32::from_ne_bytes(bytes.try_into()?)))
        .collect()
}

fn assert_cancelled<T>(result: Result<T, ExternalTensorKernelPartOneError>) {
    match result {
        Ok(_) => panic!("pre-cancelled operation unexpectedly succeeded"),
        Err(error) => assert!(
            matches!(error, ExternalTensorKernelPartOneError::Cancelled),
            "pre-cancelled operation returned the wrong typed error: {error}"
        ),
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }
}

#[test]
fn task_67_resolution_slice_seals_twelve_distinct_contracts() -> Result<(), Box<dyn Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "external_tensor_kernel_01")
        .ok_or("Task 67 resolution slice is missing")?;
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
        let evidence: serde_json::Value =
            serde_json::from_slice(&fs::read(workspace.join(contract.evidence_fixture))?)?;
        assert_eq!(evidence["operation_id"], contract.operation_id);
        assert_eq!(evidence["overload_id"], contract.overload_id);
        if [
            "COMFY-TENSOR-OP-338E50E9975A",
            "COMFY-TENSOR-OP-165C20AC8DD8",
            "COMFY-TENSOR-OP-363BE404A764",
        ]
        .contains(&contract.operation_id)
        {
            assert_eq!(evidence["source_profile"]["dependency"], "kornia");
            assert_eq!(evidence["source_profile"]["version"], "0.8.2");
            let observations = evidence["source_observations"]
                .as_array()
                .ok_or("morphology source observations are missing")?;
            let observation_ids = observations
                .iter()
                .filter_map(|observation| observation["id"].as_str())
                .collect::<BTreeSet<_>>();
            assert!(observation_ids.contains("cancellation_precedes_invalid"));
            assert!(observation_ids.contains("canonical_owner_and_adapters"));
            assert!(
                observation_ids
                    .iter()
                    .any(|observation_id| observation_id.starts_with("asymmetric_sparse_"))
            );
            assert!(
                observation_ids
                    .iter()
                    .any(|observation_id| observation_id.starts_with("all_zero_"))
            );
            let owner = observations
                .iter()
                .find(|observation| {
                    observation["id"].as_str() == Some("canonical_owner_and_adapters")
                })
                .ok_or("canonical morphology owner observation is missing")?;
            assert_eq!(owner["expected"]["owner"], "native_morphology_exact");
            assert_eq!(owner["expected"]["adapter_owns_traversal"], false);
        }
    }
    Ok(())
}

#[test]
fn rearrange_plan_forward_inverse_and_cancellation_are_checked() -> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &[1, 1, 2, 2, 2],
        &[0., 1., 2., 3., 4., 5., 6., 7.],
        &cancellation,
    )?;
    let plan = NativeRearrangePlan::patch_embedding(input.descriptor().shape(), 2, 1, 2)?;
    let execution = authorized_context(&backend, &cancellation, 16 * 1024 * 1024)?;
    let output = rearrange_with_context_exact_native(&backend, &input, &plan, &execution)?;
    assert_eq!(output.descriptor().shape(), &[1, 1, 2, 1, 4]);
    assert_close(
        &f32_values(&output)?,
        &[0., 1., 4., 5., 2., 3., 6., 7.],
        0.0,
    );
    let inverse = rearrange_vjp_with_context_exact_native(&backend, &output, &plan, &execution)?;
    assert_close(
        &f32_values(&inverse)?,
        &[0., 1., 2., 3., 4., 5., 6., 7.],
        0.0,
    );

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = authorized_context(&backend, &cancelled, 0)?;
    assert!(
        rearrange_with_context_exact_native(&backend, &input, &plan, &cancelled_context).is_err()
    );
    Ok(())
}

#[test]
fn symbolic_rearrange_is_axis_bounded_and_operation_aware() -> Result<(), Box<dyn Error>> {
    const EINOPS_OPERATION: &str = "COMFY-TENSOR-OP-A56F89536902";

    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &cancellation, 16 * 1024 * 1024)?;
    let input = upload_f32(
        &backend,
        &[2, 3],
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        &cancellation,
    )?;
    let transpose = NativeRearrangePlan::from_atomic_axes(
        EINOPS_OPERATION,
        vec![2, 3],
        vec![2, 3],
        vec![vec![0], vec![1]],
        vec![vec![1], vec![0]],
    )?;
    assert!(transpose.is_symbolic());
    let transposed = rearrange_with_context_exact_native_for_operation(
        &backend,
        &input,
        &transpose,
        EINOPS_OPERATION,
        &execution,
    )?;
    assert_eq!(transposed.descriptor().shape(), &[3, 2]);
    assert_eq!(transposed.storage_id(), input.storage_id());
    let mut transposed_values = Vec::new();
    for y in 0..3 {
        for x in 0..2 {
            transposed_values.push(f32::from_ne_bytes(
                transposed.element_bytes(&[y, x])?.try_into()?,
            ));
        }
    }
    assert_close(&transposed_values, &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0], 0.0);

    let reshaped_plan = NativeRearrangePlan::from_atomic_axes(
        EINOPS_OPERATION,
        vec![2, 3],
        vec![2, 3],
        vec![vec![0], vec![1]],
        vec![vec![0, 1]],
    )?;
    let reshaped = rearrange_with_context_exact_native_for_operation(
        &backend,
        &input,
        &reshaped_plan,
        EINOPS_OPERATION,
        &execution,
    )?;
    assert_eq!(reshaped.descriptor().shape(), &[6]);
    assert_eq!(reshaped.storage_id(), input.storage_id());

    let task_67_fresh =
        rearrange_with_context_exact_native(&backend, &input, &reshaped_plan, &execution)?;
    assert_ne!(task_67_fresh.storage_id(), input.storage_id());
    assert_close(&f32_values(&task_67_fresh)?, &f32_values(&input)?, 0.0);

    let materialized_transpose = NativeRearrangePlan::checked_for_operation(
        EINOPS_OPERATION,
        vec![2, 3],
        vec![3, 2],
        vec![0, 3, 1, 4, 2, 5],
    )?;
    let generic_output = rearrange_tensor_with_context_exact_native_for_operation(
        &*backend,
        &input,
        &materialized_transpose,
        EINOPS_OPERATION,
        &execution,
    )?;
    assert_eq!(generic_output.descriptor().shape(), &[3, 2]);
    assert_ne!(generic_output.storage_id(), input.storage_id());
    assert_close(
        &f32_values(&generic_output)?,
        &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0],
        0.0,
    );

    let upstream = upload_f32(
        &backend,
        &[3, 2],
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        &cancellation,
    )?;
    let input_gradient = rearrange_vjp_with_context_exact_native_for_operation(
        &backend,
        &upstream,
        &transpose,
        EINOPS_OPERATION,
        &execution,
    )?;
    assert_eq!(input_gradient.storage_id(), upstream.storage_id());
    let mut gradient_values = Vec::new();
    for y in 0..2 {
        for x in 0..3 {
            gradient_values.push(f32::from_ne_bytes(
                input_gradient.element_bytes(&[y, x])?.try_into()?,
            ));
        }
    }
    assert_close(&gradient_values, &[0.0, 2.0, 4.0, 1.0, 3.0, 5.0], 0.0);

    let wrong_shape = upload_f32(&backend, &[6], &[0.0; 6], &cancellation)?;
    let error = rearrange_with_context_exact_native_for_operation(
        &backend,
        &wrong_shape,
        &transpose,
        EINOPS_OPERATION,
        &execution,
    )
    .expect_err("operation-aware validation must reject a shape mismatch");
    assert!(error.to_string().contains(EINOPS_OPERATION));
    let error =
        NativeRearrangePlan::checked_for_operation(EINOPS_OPERATION, vec![2], vec![2], vec![0, 0])
            .expect_err("operation-aware explicit mapping validation must reject duplicates");
    assert!(error.to_string().contains(EINOPS_OPERATION));
    Ok(())
}

#[test]
fn canonical_bilinear_owner_distinguishes_roi_and_zero_padding() -> Result<(), Box<dyn Error>> {
    const DEFORM_CONV_OPERATION: &str = "COMFY-TENSOR-OP-9E730487CA71";

    let roi = checked_bilinear_weights(
        2,
        2,
        -0.5,
        0.5,
        NativeBilinearBoundary::RoiAlign,
        "COMFY-TENSOR-OP-0ABA532316FA",
    )?;
    assert_eq!(roi.len(), 4);
    assert_eq!(roi.iter().map(|sample| sample.weight).sum::<f32>(), 1.0);
    assert!(
        roi.iter()
            .all(|sample| sample.weight == 0.0 || sample.source_y == 0)
    );

    let zero = checked_bilinear_weights(
        2,
        2,
        -0.5,
        0.5,
        NativeBilinearBoundary::ZeroPadding,
        DEFORM_CONV_OPERATION,
    )?;
    assert_eq!(zero.len(), 2);
    assert_close(
        &zero.iter().map(|sample| sample.weight).collect::<Vec<_>>(),
        &[0.25, 0.25],
        0.0,
    );
    assert_close(
        &zero
            .iter()
            .map(|sample| sample.derivative_y)
            .collect::<Vec<_>>(),
        &[0.5, 0.5],
        0.0,
    );
    assert!(
        checked_bilinear_weights(
            2,
            2,
            f32::NAN,
            0.0,
            NativeBilinearBoundary::ZeroPadding,
            DEFORM_CONV_OPERATION,
        )
        .is_err()
    );
    let border = checked_bilinear_weights(
        2,
        2,
        -4.0,
        0.5,
        NativeBilinearBoundary::Border,
        DEFORM_CONV_OPERATION,
    )?;
    assert_close(
        &border
            .iter()
            .map(|sample| sample.weight)
            .collect::<Vec<_>>(),
        &[0.5, 0.5, 0.0, 0.0],
        0.0,
    );
    assert!(border.iter().all(|sample| sample.derivative_y == 0.0));
    let linear =
        checked_linear_weights(3, 1.25, NativeLinearBoundary::Border, DEFORM_CONV_OPERATION)?;
    assert_eq!(linear.len(), 2);
    assert_close(
        &linear
            .iter()
            .map(|sample| sample.weight)
            .collect::<Vec<_>>(),
        &[0.75, 0.25],
        0.0,
    );
    assert!(
        checked_linear_weights(
            3,
            f32::MAX,
            NativeLinearBoundary::ZeroPadding,
            DEFORM_CONV_OPERATION,
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn morphology_compositions_use_one_geodesic_flat_kernel() -> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &cancellation, 16 * 1024 * 1024)?;
    let input = upload_f32(
        &backend,
        &[1, 1, 3, 3],
        &[0., 0., 0., 0., 1., 0., 0., 0., 0.],
        &cancellation,
    )?;
    let kernel = upload_f32(&backend, &[3, 3], &[1.; 9], &cancellation)?;
    let closing = native_morphology_with_context_exact(
        &backend,
        &input,
        &kernel,
        NativeMorphologyOperation::Closing,
        &execution,
    )?;
    let opening = native_morphology_with_context_exact(
        &backend,
        &input,
        &kernel,
        NativeMorphologyOperation::Opening,
        &execution,
    )?;
    let gradient = native_morphology_with_context_exact(
        &backend,
        &input,
        &kernel,
        NativeMorphologyOperation::Gradient,
        &execution,
    )?;
    assert_close(&f32_values(&closing)?, &[1.; 9], 0.0);
    assert_close(&f32_values(&opening)?, &[0.; 9], 0.0);
    assert_close(&f32_values(&gradient)?, &[1.; 9], 0.0);

    let asymmetric_input = upload_f32(
        &backend,
        &[1, 1, 3, 4],
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &cancellation,
    )?;
    let asymmetric_kernel = upload_f32(
        &backend,
        &[2, 3],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &cancellation,
    )?;
    let asymmetric_closing = native_morphology_with_context_exact(
        &backend,
        &asymmetric_input,
        &asymmetric_kernel,
        NativeMorphologyOperation::Closing,
        &execution,
    )?;
    let asymmetric_opening = native_morphology_with_context_exact(
        &backend,
        &asymmetric_input,
        &asymmetric_kernel,
        NativeMorphologyOperation::Opening,
        &execution,
    )?;
    let asymmetric_gradient = native_morphology_with_context_exact(
        &backend,
        &asymmetric_input,
        &asymmetric_kernel,
        NativeMorphologyOperation::Gradient,
        &execution,
    )?;
    assert_close(
        &f32_values(&asymmetric_closing)?,
        &[
            2.0, 3.0, 4.0, -9_996.0, 6.0, 2.0, 3.0, 4.0, 10.0, 6.0, 7.0, 8.0,
        ],
        0.0,
    );
    assert_close(
        &f32_values(&asymmetric_opening)?,
        &[
            2.0, 3.0, 4.0, -9_996.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0,
        ],
        0.0,
    );
    assert_close(
        &f32_values(&asymmetric_gradient)?,
        &[
            1.0, 1.0, 1.0, -10_000.0, 1.0, 6.0, 6.0, 1.0, 1.0, 6.0, 6.0, 1.0,
        ],
        0.0,
    );

    let all_zero_kernel = upload_f32(&backend, &[2, 2], &[0.0; 4], &cancellation)?;
    for (operation, expected) in [
        (
            NativeMorphologyOperation::Closing,
            [1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 3.0, 5.0, 5.0, 6.0, 7.0],
        ),
        (
            NativeMorphologyOperation::Opening,
            [1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 3.0, 5.0, 5.0, 6.0, 7.0],
        ),
        (
            NativeMorphologyOperation::Gradient,
            [
                -20_000.0, -19_999.0, -19_999.0, -19_999.0, -19_996.0, -19_995.0, -19_995.0,
                -19_995.0, -19_996.0, -19_995.0, -19_995.0, -19_995.0,
            ],
        ),
    ] {
        let output = native_morphology_with_context_exact(
            &backend,
            &asymmetric_input,
            &all_zero_kernel,
            operation,
            &execution,
        )?;
        assert_close(&f32_values(&output)?, &expected, 0.0);
    }

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let malformed_kernel = upload_u8(&backend, &[1, 1], &[1], &cancellation)?;
    let cancelled_context = authorized_context(&backend, &cancelled, 0)?;
    let error = native_morphology_with_context_exact(
        &backend,
        &asymmetric_input,
        &malformed_kernel,
        NativeMorphologyOperation::Closing,
        &cancelled_context,
    )
    .expect_err("cancellation must take precedence over malformed morphology input");
    assert_eq!(
        error.to_string(),
        "external tensor kernel execution was cancelled"
    );
    Ok(())
}

#[test]
fn audio_biquads_and_bandlimited_resample_are_deterministic() -> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &cancellation, 16 * 1024 * 1024)?;
    let waveform = upload_f32(&backend, &[1, 4], &[0.25, -0.5, 0.75, -0.25], &cancellation)?;
    let equalized = equalizer_biquad_with_context_exact_native(
        &backend, &waveform, 48_000, 1_000.0, 0.0, 0.707, &execution,
    )?;
    let trebled = treble_biquad_with_context_exact_native(
        &backend, &waveform, 48_000, 0.0, 3_000.0, 0.707, &execution,
    )?;
    assert_close(&f32_values(&equalized)?, &[0.25, -0.5, 0.75, -0.25], 1.0e-6);
    assert_close(&f32_values(&trebled)?, &[0.25, -0.5, 0.75, -0.25], 1.0e-6);
    let boosted_equalizer = equalizer_biquad_with_context_exact_native(
        &backend, &waveform, 48_000, 1_000.0, 6.0, 0.707, &execution,
    )?;
    assert_close(
        &f32_values(&boosted_equalizer)?,
        &[0.26526278, -0.50211763, 0.76319385, -0.20832232],
        1.0e-6,
    );
    let boosted_treble = treble_biquad_with_context_exact_native(
        &backend, &waveform, 48_000, 6.0, 3_000.0, 0.707, &execution,
    )?;
    assert_close(
        &f32_values(&boosted_treble)?,
        &[0.44916528, -0.9817641, 1.0, -0.6230713],
        1.0e-6,
    );

    let constant = upload_f32(&backend, &[1, 4], &[1.; 4], &cancellation)?;
    let configuration = NativeResampleConfiguration::torchaudio_default(4, 8);
    let resampled =
        resample_with_context_exact_native(&backend, &constant, configuration, &execution)?;
    assert_eq!(resampled.descriptor().shape(), &[1, 8]);
    let repeated =
        resample_with_context_exact_native(&backend, &constant, configuration, &execution)?;
    assert_close(&f32_values(&resampled)?, &f32_values(&repeated)?, 0.0);
    assert!(
        f32_values(&resampled)?
            .iter()
            .all(|value| value.is_finite())
    );
    let signal = upload_f32(&backend, &[1, 4], &[0.0, 1.0, 0.0, -1.0], &cancellation)?;
    let signal_output =
        resample_with_context_exact_native(&backend, &signal, configuration, &execution)?;
    assert_close(
        &f32_values(&signal_output)?,
        &[
            0.0042705993,
            0.5452181,
            0.99754024,
            0.8074259,
            0.0,
            -0.8074259,
            -0.99754024,
            -0.5452181,
        ],
        2.0e-6,
    );
    let tangent =
        resample_jvp_with_context_exact_native(&backend, &constant, configuration, &execution)?;
    assert_close(&f32_values(&tangent)?, &f32_values(&resampled)?, 0.0);
    let upstream = upload_f32(&backend, &[1, 8], &[1.; 8], &cancellation)?;
    let input_gradient = resample_vjp_with_context_exact_native(
        &backend,
        &[1, 4],
        &upstream,
        configuration,
        &execution,
    )?;
    assert_eq!(input_gradient.descriptor().shape(), &[1, 4]);
    let lhs: f32 = f32_values(&resampled)?.iter().sum();
    let rhs: f32 = f32_values(&input_gradient)?.iter().sum();
    assert!((lhs - rhs).abs() <= 1.0e-5);
    Ok(())
}

#[test]
fn mel_spectrogram_and_roi_align_execute_native_analytical_paths() -> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let waveform = upload_f32(
        &backend,
        &[8],
        &[0., 1., 0., -1., 0., 1., 0., -1.],
        &cancellation,
    )?;
    let mel_configuration = NativeMelSpectrogramConfiguration {
        sample_rate: 8,
        n_fft: 4,
        win_length: Some(4),
        hop_length: Some(2),
        f_min: 0.0,
        f_max: Some(4.0),
        n_mels: 2,
        power: 2.0,
        center: true,
        normalized: false,
        mel_scale: NativeMelScale::Slaney,
        mel_normalization: NativeMelNormalization::Slaney,
    };
    let mel_context = backend.execution_context(
        StreamId::DEFAULT,
        backend.workspace_authority.authorize_workspace(960)?,
        &cancellation,
    );
    let mel = mel_spectrogram_with_context_exact_native(
        &backend,
        &waveform,
        mel_configuration,
        &mel_context,
    )?;
    assert_eq!(mel.descriptor().shape()[0], 2);
    assert!(
        f32_values(&mel)?
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    );
    let batched_waveform_values = [0., 1., 0., -1., 0., 1., 0., -1.].repeat(4);
    let batched_waveform = upload_f32(
        &backend,
        &[2, 2, 8],
        &batched_waveform_values,
        &cancellation,
    )?;
    let batched_mel = mel_spectrogram_with_context_exact_native(
        &backend,
        &batched_waveform,
        mel_configuration,
        &mel_context,
    )?;
    assert_eq!(batched_mel.descriptor().shape(), &[2, 2, 2, 5]);
    for batch in f32_values(&batched_mel)?.chunks_exact(f32_values(&mel)?.len()) {
        assert_close(batch, &f32_values(&mel)?, 1.0e-6);
    }
    assert_eq!(mel_context.scratch.peak_bytes(), 960);
    assert_eq!(mel_context.scratch.in_use_bytes(), 0);

    let insufficient_mel = authorized_context(&backend, &cancellation, 959)?;
    let original_waveform = batched_waveform.contiguous_bytes()?.to_vec();
    assert!(
        mel_spectrogram_with_context_exact_native(
            &backend,
            &batched_waveform,
            mel_configuration,
            &insufficient_mel,
        )
        .is_err()
    );
    assert_eq!(insufficient_mel.scratch.in_use_bytes(), 0);
    assert_eq!(batched_waveform.contiguous_bytes()?, original_waveform);

    let image = upload_f32(&backend, &[1, 1, 2, 2], &[1., 2., 3., 4.], &cancellation)?;
    let boxes = upload_f32(&backend, &[1, 4], &[0., 0., 1., 1.], &cancellation)?;
    let configuration = RoiAlignConfiguration {
        output_height: 1,
        output_width: 1,
        spatial_scale_numerator: 1,
        spatial_scale_denominator: 1,
        sampling_ratio: 1,
        aligned: false,
    };
    let aligned = roi_align_with_context_exact_native(
        &backend,
        &image,
        std::slice::from_ref(&boxes),
        configuration,
        &mel_context,
    )?;
    assert_close(&f32_values(&aligned)?, &[2.5], 1.0e-6);
    let tangent = roi_align_jvp_with_context_exact_native(
        &backend,
        &image,
        std::slice::from_ref(&boxes),
        configuration,
        &mel_context,
    )?;
    assert_close(&f32_values(&tangent)?, &[2.5], 1.0e-6);
    let upstream = upload_f32(&backend, &[1, 1, 1, 1], &[1.], &cancellation)?;
    let gradient = roi_align_vjp_with_context_exact_native(
        &backend,
        &image,
        std::slice::from_ref(&boxes),
        &upstream,
        configuration,
        &mel_context,
    )?;
    assert_close(&f32_values(&gradient)?, &[0.25; 4], 1.0e-6);
    Ok(())
}

#[test]
fn normalize_resize_and_native_image_conversion_preserve_boundary_semantics()
-> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &cancellation, 16 * 1024 * 1024)?;
    let input = upload_f32(&backend, &[3, 1, 1], &[0.0, 0.5, 1.0], &cancellation)?;
    let normalized =
        normalize_with_context_exact_native(&backend, &input, &[0.5], &[0.5], &execution)?;
    assert_close(&f32_values(&normalized)?, &[-1.0, 0.0, 1.0], 0.0);
    let tangent = normalize_jvp_with_context_exact_native(&backend, &input, &[0.5], &execution)?;
    let gradient = normalize_vjp_with_context_exact_native(&backend, &input, &[0.5], &execution)?;
    assert_close(&f32_values(&tangent)?, &[0.0, 1.0, 2.0], 0.0);
    assert_close(&f32_values(&gradient)?, &[0.0, 1.0, 2.0], 0.0);

    let image = to_tensor_with_context_exact_native(
        &backend,
        &[0, 64, 128, 255, 32, 16],
        1,
        2,
        3,
        StreamId::DEFAULT,
        &execution,
    )?;
    assert_eq!(image.descriptor().shape(), &[3, 1, 2]);
    assert_close(
        &f32_values(&image)?,
        &[
            0.0,
            1.0,
            64.0 / 255.0,
            32.0 / 255.0,
            128.0 / 255.0,
            16.0 / 255.0,
        ],
        1.0e-7,
    );

    let u8_image = upload_u8(&backend, &[1, 2, 2], &[0, 64, 128, 255], &cancellation)?;
    let resized = resize_with_context_exact_native(
        &backend,
        &u8_image,
        1,
        1,
        ResizeMode::Bicubic,
        true,
        &execution,
    )?;
    assert_eq!(resized.descriptor().shape(), &[1, 1, 1]);
    assert_eq!(resized.descriptor().dtype(), DType::U8);
    assert_eq!(resized.contiguous_bytes()?, &[112]);
    Ok(())
}

#[test]
fn canonical_workspace_is_exact_atomic_and_convergent() -> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let image = [0_u8, 64, 128, 255, 32, 16];
    let exact_bytes = u64::try_from(image.len() * std::mem::size_of::<f32>())?;

    let foreign_backend = backend_with_limit(64 * 1024 * 1024)?;
    let foreign_context = context(&foreign_backend, &cancellation)?;
    let foreign_authority_error = to_tensor_with_context_exact_native(
        &backend,
        &image,
        1,
        2,
        3,
        StreamId::DEFAULT,
        &foreign_context,
    )
    .expect_err("canonical staging must reject another backend's authority");
    assert!(
        foreign_authority_error
            .to_string()
            .contains("workspace authorization")
    );

    let exact = authorized_context(&backend, &cancellation, exact_bytes)?;
    let tensor =
        to_tensor_with_context_exact_native(&backend, &image, 1, 2, 3, StreamId::DEFAULT, &exact)?;
    assert_eq!(tensor.descriptor().shape(), &[3, 1, 2]);
    assert_eq!(exact.scratch.peak_bytes(), exact_bytes);
    assert_eq!(exact.scratch.in_use_bytes(), 0);

    let waveform = upload_f32(&backend, &[1, 4], &[0.25, -0.5, 0.75, -0.25], &cancellation)?;
    let biquad_context = authorized_context(&backend, &cancellation, 32)?;
    equalizer_biquad_with_context_exact_native(
        &backend,
        &waveform,
        48_000,
        1_000.0,
        3.0,
        0.707,
        &biquad_context,
    )?;
    assert_eq!(biquad_context.scratch.peak_bytes(), 32);
    assert_eq!(biquad_context.scratch.in_use_bytes(), 0);

    let morphology_input = upload_f32(
        &backend,
        &[1, 1, 3, 3],
        &[0., 0., 0., 0., 1., 0., 0., 0., 0.],
        &cancellation,
    )?;
    let morphology_kernel = upload_f32(&backend, &[3, 3], &[1.; 9], &cancellation)?;
    let morphology_context = authorized_context(&backend, &cancellation, 153)?;
    native_morphology_with_context_exact(
        &backend,
        &morphology_input,
        &morphology_kernel,
        NativeMorphologyOperation::Gradient,
        &morphology_context,
    )?;
    assert_eq!(morphology_context.scratch.peak_bytes(), 153);
    assert_eq!(morphology_context.scratch.in_use_bytes(), 0);

    let roi_input = upload_f32(&backend, &[1, 1, 2, 2], &[1., 2., 3., 4.], &cancellation)?;
    let boxes = upload_f32(&backend, &[1, 4], &[0., 0., 1., 1.], &cancellation)?;
    let roi_context = authorized_context(&backend, &cancellation, 36)?;
    roi_align_with_context_exact_native(
        &backend,
        &roi_input,
        &[boxes],
        RoiAlignConfiguration {
            output_height: 1,
            output_width: 1,
            spatial_scale_numerator: 1,
            spatial_scale_denominator: 1,
            sampling_ratio: 1,
            aligned: false,
        },
        &roi_context,
    )?;
    assert_eq!(roi_context.scratch.peak_bytes(), 36);
    assert_eq!(roi_context.scratch.in_use_bytes(), 0);

    let insufficient = authorized_context(&backend, &cancellation, exact_bytes - 1)?;
    assert!(
        to_tensor_with_context_exact_native(
            &backend,
            &image,
            1,
            2,
            3,
            StreamId::DEFAULT,
            &insufficient,
        )
        .is_err()
    );
    assert_eq!(insufficient.scratch.in_use_bytes(), 0);
    assert_eq!(image, [0, 64, 128, 255, 32, 16]);

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = authorized_context(&backend, &cancelled, exact_bytes)?;
    assert!(
        to_tensor_with_context_exact_native(
            &backend,
            &image,
            1,
            2,
            3,
            StreamId::DEFAULT,
            &cancelled_context,
        )
        .is_err()
    );
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn canonical_context_inventory_has_no_legacy_authority() -> Result<(), Box<dyn Error>> {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/ops/external_tensor_kernel_01.rs"),
    )?;
    assert!(!source.contains("authorize_workspace("));
    assert_eq!(source.matches("ScratchReservation::none()").count(), 0);
    for canonical in [
        "rearrange_with_context_exact_native",
        "native_morphology_with_context_exact",
        "biquad_with_context_exact_native",
        "resample_with_context_exact_native",
        "mel_scale_project_with_context_exact_native",
        "mel_spectrogram_with_context_exact_native",
        "roi_align_with_context_exact_native",
        "normalize_with_context_exact_native",
        "resize_with_context_exact_native",
        "to_tensor_with_context_exact_native",
    ] {
        assert!(
            source.contains(canonical),
            "missing canonical path {canonical}"
        );
    }
    Ok(())
}

#[test]
fn backend_capacity_oom_is_atomic_and_convergent() -> Result<(), Box<dyn Error>> {
    let backend = backend_with_limit(32)?;
    let cancellation = CancellationToken::default();
    let waveform = upload_f32(&backend, &[1, 4], &[0.25, -0.5, 0.75, -0.25], &cancellation)?;
    let original = waveform.contiguous_bytes()?.to_vec();
    let baseline = backend.memory_snapshot().current_bytes;
    assert_eq!(baseline, 16);
    let execution = authorized_context(&backend, &cancellation, 32)?;

    assert!(matches!(
        equalizer_biquad_with_context_exact_native(
            &backend, &waveform, 48_000, 1_000.0, 3.0, 0.707, &execution,
        ),
        Err(ExternalTensorKernelPartOneError::Tensor(
            sim_tensor::TensorError::AllocationFailed { requested: 16, .. }
        ))
    ));
    assert_eq!(waveform.contiguous_bytes()?, original);
    assert_eq!(execution.scratch.peak_bytes(), 32);
    assert_eq!(execution.scratch.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn invalid_external_boundaries_fail_typed_without_partial_publication() -> Result<(), Box<dyn Error>>
{
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = authorized_context(&backend, &cancellation, 16 * 1024 * 1024)?;
    assert!(NativeRearrangePlan::checked(vec![2], vec![2], vec![0, 0]).is_err());

    let image = upload_f32(&backend, &[1, 1, 2, 2], &[1., 2., 3., 4.], &cancellation)?;
    let empty_kernel = upload_f32(&backend, &[0, 0], &[], &cancellation)?;
    assert!(
        native_morphology_with_context_exact(
            &backend,
            &image,
            &empty_kernel,
            NativeMorphologyOperation::Opening,
            &execution,
        )
        .is_err()
    );
    let wrong_dtype_kernel = upload_u8(&backend, &[1, 1], &[1], &cancellation)?;
    for (operation, operation_id) in [
        (
            NativeMorphologyOperation::Dilation,
            "COMFY-TENSOR-OP-AF5C2820E4C3",
        ),
        (
            NativeMorphologyOperation::Erosion,
            "COMFY-TENSOR-OP-9236C1C08976",
        ),
        (
            NativeMorphologyOperation::TopHat,
            "COMFY-TENSOR-OP-AC69F309A190",
        ),
        (
            NativeMorphologyOperation::BottomHat,
            "COMFY-TENSOR-OP-C5A306EB73FD",
        ),
    ] {
        let error = native_morphology_with_context_exact(
            &backend,
            &image,
            &wrong_dtype_kernel,
            operation,
            &execution,
        )
        .expect_err("wrong morphology kernel dtype must fail");
        assert!(error.to_string().contains(operation_id));
    }
    assert!(
        equalizer_biquad_with_context_exact_native(
            &backend, &image, 0, 1_000.0, 3.0, 0.707, &execution,
        )
        .is_err()
    );
    assert!(
        normalize_with_context_exact_native(&backend, &image, &[0.0], &[0.0], &execution).is_err()
    );
    assert!(
        resize_with_context_exact_native(
            &backend,
            &image,
            1,
            1,
            ResizeMode::NearestExact,
            true,
            &execution,
        )
        .is_err()
    );
    assert!(
        to_tensor_with_context_exact_native(
            &backend,
            &[0, 1],
            1,
            1,
            3,
            StreamId::DEFAULT,
            &execution,
        )
        .is_err()
    );
    let boxes = upload_f32(&backend, &[1, 4], &[0., 0., 1., 1.], &cancellation)?;
    assert!(
        roi_align_with_context_exact_native(
            &backend,
            &image,
            &[boxes.clone(), boxes],
            RoiAlignConfiguration {
                output_height: 1,
                output_width: 1,
                spatial_scale_numerator: 1,
                spatial_scale_denominator: 1,
                sampling_ratio: 1,
                aligned: false,
            },
            &execution,
        )
        .is_err()
    );

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_context = authorized_context(&backend, &cancelled, 0)?;
    let memory_before_cancellation = backend.memory_snapshot();
    let plan = NativeRearrangePlan::checked(vec![1], vec![1], vec![0])?;
    for result in [
        rearrange_with_context_exact_native(&backend, &image, &plan, &cancelled_context),
        rearrange_with_context_exact_native_for_operation(
            &backend,
            &image,
            &plan,
            "task-67-cancelled-rearrange",
            &cancelled_context,
        ),
        rearrange_jvp_with_context_exact_native(&backend, &image, &plan, &cancelled_context),
        rearrange_vjp_with_context_exact_native(&backend, &image, &plan, &cancelled_context),
        rearrange_jvp_with_context_exact_native_for_operation(
            &backend,
            &image,
            &plan,
            "task-67-cancelled-rearrange-jvp",
            &cancelled_context,
        ),
        rearrange_vjp_with_context_exact_native_for_operation(
            &backend,
            &image,
            &plan,
            "task-67-cancelled-rearrange-vjp",
            &cancelled_context,
        ),
    ] {
        assert_cancelled(result);
    }
    assert_cancelled(native_morphology_with_context_exact(
        &backend,
        &image,
        &empty_kernel,
        NativeMorphologyOperation::Opening,
        &cancelled_context,
    ));
    assert_cancelled(biquad_with_context_exact_native(
        &backend,
        &image,
        NativeBiquadCoefficients {
            b0: f64::NAN,
            b1: 0.0,
            b2: 0.0,
            a0: 0.0,
            a1: 0.0,
            a2: 0.0,
        },
        false,
        "task-67-cancelled-biquad",
        &cancelled_context,
    ));
    assert_cancelled(equalizer_biquad_with_context_exact_native(
        &backend,
        &image,
        0,
        0.0,
        f64::NAN,
        0.0,
        &cancelled_context,
    ));
    assert_cancelled(treble_biquad_with_context_exact_native(
        &backend,
        &image,
        0,
        f64::NAN,
        0.0,
        0.0,
        &cancelled_context,
    ));
    assert_cancelled(resample_with_context_exact_native(
        &backend,
        &image,
        NativeResampleConfiguration::torchaudio_default(0, 0),
        &cancelled_context,
    ));
    assert_cancelled(resample_jvp_with_context_exact_native(
        &backend,
        &image,
        NativeResampleConfiguration::torchaudio_default(0, 0),
        &cancelled_context,
    ));
    assert_cancelled(resample_vjp_with_context_exact_native(
        &backend,
        &[],
        &image,
        NativeResampleConfiguration::torchaudio_default(0, 0),
        &cancelled_context,
    ));
    let invalid_mel_scale = NativeMelScaleConfiguration {
        n_mels: 0,
        sample_rate: 0,
        f_min: f64::NAN,
        f_max: None,
        n_stft: 0,
        mel_scale: NativeMelScale::Slaney,
        mel_normalization: NativeMelNormalization::Slaney,
    };
    assert_cancelled(mel_scale_project_with_context_exact_native(
        &backend,
        &image,
        invalid_mel_scale,
        "task-67-cancelled-mel-scale",
        &cancelled_context,
    ));
    assert_cancelled(mel_scale_project_vjp_with_context_exact_native(
        &backend,
        &image,
        &image,
        invalid_mel_scale,
        "task-67-cancelled-mel-scale-vjp",
        &cancelled_context,
    ));
    assert_cancelled(mel_spectrogram_with_context_exact_native(
        &backend,
        &image,
        NativeMelSpectrogramConfiguration {
            sample_rate: 0,
            n_fft: 0,
            win_length: None,
            hop_length: None,
            f_min: f64::NAN,
            f_max: None,
            n_mels: 0,
            power: 0.0,
            center: false,
            normalized: false,
            mel_scale: NativeMelScale::Slaney,
            mel_normalization: NativeMelNormalization::Slaney,
        },
        &cancelled_context,
    ));
    assert_cancelled(normalize_with_context_exact_native(
        &backend,
        &image,
        &[],
        &[],
        &cancelled_context,
    ));
    assert_cancelled(resize_with_context_exact_native(
        &backend,
        &image,
        0,
        0,
        ResizeMode::NearestExact,
        true,
        &cancelled_context,
    ));
    assert_cancelled(to_tensor_with_context_exact_native(
        &backend,
        &[],
        0,
        0,
        0,
        StreamId::DEFAULT,
        &cancelled_context,
    ));
    assert_cancelled(roi_align_with_context_exact_native(
        &backend,
        &image,
        &[],
        RoiAlignConfiguration {
            output_height: 0,
            output_width: 0,
            spatial_scale_numerator: 1,
            spatial_scale_denominator: 0,
            sampling_ratio: -2,
            aligned: false,
        },
        &cancelled_context,
    ));
    assert_cancelled(roi_align_jvp_with_context_exact_native(
        &backend,
        &image,
        &[],
        RoiAlignConfiguration {
            output_height: 0,
            output_width: 0,
            spatial_scale_numerator: 1,
            spatial_scale_denominator: 0,
            sampling_ratio: -2,
            aligned: false,
        },
        &cancelled_context,
    ));
    assert_cancelled(roi_align_vjp_with_context_exact_native(
        &backend,
        &image,
        &[],
        &image,
        RoiAlignConfiguration {
            output_height: 0,
            output_width: 0,
            spatial_scale_numerator: 1,
            spatial_scale_denominator: 0,
            sampling_ratio: -2,
            aligned: false,
        },
        &cancelled_context,
    ));
    assert_cancelled(normalize_jvp_with_context_exact_native(
        &backend,
        &image,
        &[],
        &cancelled_context,
    ));
    assert_cancelled(normalize_vjp_with_context_exact_native(
        &backend,
        &image,
        &[],
        &cancelled_context,
    ));
    assert_eq!(cancelled_context.scratch.peak_bytes(), 0);
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot(), memory_before_cancellation);
    Ok(())
}
