use std::{error::Error, ops::Deref};

use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    Rgb8ImageTensor, StreamId, Tensor, TensorDescriptor, TensorError,
    generated_external_tensor_kernel_01::{
        NativeMelNormalization, NativeMelScale, NativeMelScaleConfiguration,
    },
    generated_external_tensor_kernel_03::{
        ExternalTensorKernelPartThreeError, NativeBoxFormat, NativeTensorTransform,
        bass_biquad_with_context_exact_native, bottom_hat_with_context_exact_native,
        box_convert_jvp_with_context_exact_native, box_convert_vjp_with_context_exact_native,
        box_convert_with_context_exact_native, compose_jvp_with_context_exact_native,
        compose_vjp_with_context_exact_native, compose_with_context_exact_native,
        lab_to_rgb_jvp_with_context_exact_native, lab_to_rgb_vjp_with_context_exact_native,
        lab_to_rgb_with_context_exact_native, mel_scale_jvp_with_context_exact_native,
        mel_scale_vjp_with_context_exact_native, mel_scale_with_context_exact_native,
        to_tensor_with_context_exact_native,
    },
};

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
    bytes: u64,
) -> Result<ExecutionContext<'a>, Box<dyn Error>> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        backend.workspace_authority.authorize_workspace(bytes)?,
        cancellation,
    ))
}

fn backend() -> Result<TestBackend, Box<dyn Error>> {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
    Ok(TestBackend {
        backend,
        workspace_authority,
    })
}

fn upload_f32(
    backend: &TestBackend,
    shape: &[u64],
    values: &[f32],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn upload_u8(
    backend: &TestBackend,
    shape: &[u64],
    values: &[u8],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::U8, DeviceId::CPU, StreamId::DEFAULT)?;
    Ok(backend.upload_bytes(descriptor, values, context)?.0)
}

fn f32_values(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn Error>> {
    fn visit(
        tensor: &Tensor,
        shape: &[u64],
        indices: &mut Vec<u64>,
        output: &mut Vec<f32>,
    ) -> Result<(), Box<dyn Error>> {
        if indices.len() == shape.len() {
            output.push(f32::from_ne_bytes(
                tensor.element_bytes(indices)?.try_into()?,
            ));
            return Ok(());
        }
        let axis = indices.len();
        for index in 0..shape[axis] {
            indices.push(index);
            visit(tensor, shape, indices, output)?;
            indices.pop();
        }
        Ok(())
    }

    let mut output = Vec::new();
    visit(
        tensor,
        tensor.descriptor().shape(),
        &mut Vec::new(),
        &mut output,
    )?;
    Ok(output)
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: expected {expected}, got {actual}"
        );
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn assert_cancelled<T>(result: Result<T, ExternalTensorKernelPartThreeError>) {
    match result {
        Ok(_) => panic!("pre-cancelled operation unexpectedly succeeded"),
        Err(error) => assert!(matches!(
            error,
            ExternalTensorKernelPartThreeError::Cancelled
        )),
    }
}

#[test]
fn box_adapter_workspace_is_exact_bounded_cancel_safe_and_convergent() -> Result<(), Box<dyn Error>>
{
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation, 64 * 1024 * 1024)?;
    let input = upload_f32(
        &backend,
        &[2, 4],
        &[1.0, 2.0, 3.0, 4.0, 4.0, 6.0, 2.0, 8.0],
        &execution,
    )?;

    let exact = backend.workspace_authority.authorize_workspace(32)?;
    let execution = backend.execution_context(StreamId::DEFAULT, exact.clone(), &cancellation);
    let output = box_convert_with_context_exact_native(
        &backend,
        &input,
        NativeBoxFormat::CenterXyWidthHeight,
        NativeBoxFormat::Xyxy,
        &execution,
    )?;
    assert_eq!(output.descriptor().shape(), input.descriptor().shape());
    assert_eq!(exact.peak_bytes(), 32);
    assert_eq!(exact.in_use_bytes(), 0);

    let baseline = backend.memory_snapshot().current_bytes;
    let insufficient = backend.workspace_authority.authorize_workspace(31)?;
    let execution =
        backend.execution_context(StreamId::DEFAULT, insufficient.clone(), &cancellation);
    assert!(matches!(
        box_convert_with_context_exact_native(
            &backend,
            &input,
            NativeBoxFormat::CenterXyWidthHeight,
            NativeBoxFormat::Xyxy,
            &execution,
        ),
        Err(sim_tensor::generated_external_tensor_kernel_03::ExternalTensorKernelPartThreeError::Tensor(
            TensorError::WorkspaceAuthorizationExceeded { .. }
        ))
    ));
    assert_eq!(insufficient.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let scratch = backend.workspace_authority.authorize_workspace(32)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch.clone(), &cancelled);
    assert!(
        box_convert_with_context_exact_native(
            &backend,
            &input,
            NativeBoxFormat::CenterXyWidthHeight,
            NativeBoxFormat::Xyxy,
            &execution,
        )
        .is_err()
    );
    assert_eq!(scratch.peak_bytes(), 0);
    assert_eq!(scratch.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn lab_to_rgb_forward_and_analytical_maps_use_shared_color_traversal() -> Result<(), Box<dyn Error>>
{
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation, 64 * 1024 * 1024)?;
    let input = upload_f32(&backend, &[1, 3, 1, 1], &[50.0, 20.0, -30.0], &execution)?;
    let tangent = upload_f32(&backend, &[1, 3, 1, 1], &[0.3, -0.2, 0.1], &execution)?;
    let upstream = upload_f32(&backend, &[1, 3, 1, 1], &[-0.4, 0.7, 0.2], &execution)?;
    let output = lab_to_rgb_with_context_exact_native(&backend, &input, &execution)?;
    let jvp = lab_to_rgb_jvp_with_context_exact_native(&backend, &input, &tangent, &execution)?;
    let vjp = lab_to_rgb_vjp_with_context_exact_native(&backend, &input, &upstream, &execution)?;
    assert_close(
        &f32_values(&output)?,
        &[0.496_361_43, 0.429_257_63, 0.666_838_9],
        3.0e-5,
    );
    assert_close(
        &f32_values(&jvp)?,
        &[0.002_043_65, 0.003_422_78, 0.002_498_04],
        2.0e-6,
    );
    assert_close(
        &f32_values(&vjp)?,
        &[0.004_859_13, -0.004_847_7, -0.003_491_85],
        2.0e-6,
    );
    assert!(
        (dot(&f32_values(&jvp)?, &f32_values(&upstream)?)
            - dot(&f32_values(&tangent)?, &f32_values(&vjp)?))
        .abs()
            <= 2.0e-7
    );
    Ok(())
}

#[test]
fn bottom_hat_delegates_asymmetric_kornia_geodesic_semantics() -> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let scratch = backend
        .workspace_authority
        .authorize_workspace(64 * 1024 * 1024)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
    let input = upload_f32(
        &backend,
        &[1, 1, 3, 4],
        &(1..=12).map(|value| value as f32).collect::<Vec<_>>(),
        &execution,
    )?;
    let kernel = upload_f32(
        &backend,
        &[2, 3],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &execution,
    )?;
    let output = bottom_hat_with_context_exact_native(&backend, &input, &kernel, &execution)?;
    assert_close(
        &f32_values(&output)?,
        &[
            1.0, 1.0, 1.0, -10_000.0, 1.0, -4.0, -4.0, -4.0, 1.0, -4.0, -4.0, -4.0,
        ],
        0.0,
    );
    Ok(())
}

#[test]
fn bass_coefficients_delegate_the_canonical_biquad_recurrence() -> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let scratch = backend
        .workspace_authority
        .authorize_workspace(64 * 1024 * 1024)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
    let waveform = upload_f32(&backend, &[1, 4], &[0.25, -0.5, 0.75, -0.25], &execution)?;
    let output = bass_biquad_with_context_exact_native(
        &backend, &waveform, 48_000, 6.0, 100.0, 0.707, &execution,
    )?;
    assert_close(
        &f32_values(&output)?,
        &[0.250_962_2, -0.500_003, 0.750_959_16, -0.247_119_52],
        1.0e-6,
    );
    Ok(())
}

fn mel_configuration() -> NativeMelScaleConfiguration {
    NativeMelScaleConfiguration {
        n_mels: 2,
        sample_rate: 8,
        f_min: 0.0,
        f_max: Some(4.0),
        n_stft: 3,
        mel_scale: NativeMelScale::Slaney,
        mel_normalization: NativeMelNormalization::Slaney,
    }
}

#[test]
fn mel_scale_forward_jvp_and_transpose_vjp_share_one_filter_bank() -> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let scratch = backend
        .workspace_authority
        .authorize_workspace(64 * 1024 * 1024)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
    let input = upload_f32(
        &backend,
        &[3, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &execution,
    )?;
    let output =
        mel_scale_with_context_exact_native(&backend, &input, mel_configuration(), &execution)?;
    let jvp =
        mel_scale_jvp_with_context_exact_native(&backend, &input, mel_configuration(), &execution)?;
    assert_eq!(output.descriptor().shape(), &[2, 2]);
    assert_close(&f32_values(&output)?, &[1.125, 1.5, 1.125, 1.5], 1.0e-6);
    assert_close(&f32_values(&jvp)?, &f32_values(&output)?, 0.0);
    let upstream = upload_f32(&backend, &[2, 2], &[1.0, 2.0, 3.0, 4.0], &execution)?;
    let vjp = mel_scale_vjp_with_context_exact_native(
        &backend,
        &input,
        &upstream,
        mel_configuration(),
        &execution,
    )?;
    assert_close(&f32_values(&vjp)?, &[0.0, 0.0, 1.5, 2.25, 0.0, 0.0], 1.0e-6);
    assert!(
        (dot(&f32_values(&output)?, &f32_values(&upstream)?)
            - dot(&f32_values(&input)?, &f32_values(&vjp)?))
        .abs()
            <= 1.0e-6
    );
    Ok(())
}

#[test]
fn box_convert_forward_jvp_vjp_and_empty_publication_are_checked() -> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation, 64 * 1024 * 1024)?;
    let input = upload_f32(&backend, &[1, 1, 4], &[0.25, 0.75, 0.5, 0.25], &execution)?;
    let tangent = upload_f32(&backend, &[1, 1, 4], &[0.3, -0.2, 0.1, 0.4], &execution)?;
    let upstream = upload_f32(&backend, &[1, 1, 4], &[-0.4, 0.7, 0.2, -0.1], &execution)?;
    let output = box_convert_with_context_exact_native(
        &backend,
        &input,
        NativeBoxFormat::CenterXyWidthHeight,
        NativeBoxFormat::Xyxy,
        &execution,
    )?;
    let jvp = box_convert_jvp_with_context_exact_native(
        &backend,
        &tangent,
        NativeBoxFormat::CenterXyWidthHeight,
        NativeBoxFormat::Xyxy,
        &execution,
    )?;
    let vjp = box_convert_vjp_with_context_exact_native(
        &backend,
        &upstream,
        NativeBoxFormat::CenterXyWidthHeight,
        NativeBoxFormat::Xyxy,
        &execution,
    )?;
    assert_close(&f32_values(&output)?, &[0.0, 0.625, 0.5, 0.875], 0.0);
    assert_close(&f32_values(&jvp)?, &[0.25, -0.4, 0.35, 0.0], 1.0e-7);
    assert_close(&f32_values(&vjp)?, &[-0.2, 0.6, 0.3, -0.4], 1.0e-7);
    assert!(
        (dot(&f32_values(&jvp)?, &f32_values(&upstream)?)
            - dot(&f32_values(&tangent)?, &f32_values(&vjp)?))
        .abs()
            <= 1.0e-7
    );
    let empty = upload_f32(&backend, &[0, 4], &[], &execution)?;
    let empty_output = box_convert_with_context_exact_native(
        &backend,
        &empty,
        NativeBoxFormat::CenterXyWidthHeight,
        NativeBoxFormat::Xyxy,
        &execution,
    )?;
    assert_eq!(empty_output.descriptor().shape(), &[0, 4]);
    assert_ne!(empty_output.storage_id(), empty.storage_id());
    Ok(())
}

#[test]
fn compose_orders_canonical_normalization_and_preserves_empty_identity_alias()
-> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let scratch = backend
        .workspace_authority
        .authorize_workspace(64 * 1024 * 1024)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
    let input = upload_f32(
        &backend,
        &[1, 1, 1, 5],
        &[0.0, 0.25, 0.5, 0.75, 1.0],
        &execution,
    )?;
    let transforms = [NativeTensorTransform::Normalize {
        mean: vec![0.5],
        standard_deviation: vec![0.5],
    }];
    let output = compose_with_context_exact_native(&backend, &input, &transforms, &execution)?;
    assert_close(&f32_values(&output)?, &[-1.0, -0.5, 0.0, 0.5, 1.0], 0.0);
    let tangent = upload_f32(
        &backend,
        &[1, 1, 1, 5],
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        &execution,
    )?;
    let jvp = compose_jvp_with_context_exact_native(&backend, &tangent, &transforms, &execution)?;
    let vjp = compose_vjp_with_context_exact_native(&backend, &tangent, &transforms, &execution)?;
    assert_close(&f32_values(&jvp)?, &[2.0, 4.0, 6.0, 8.0, 10.0], 0.0);
    assert_close(&f32_values(&vjp)?, &[2.0, 4.0, 6.0, 8.0, 10.0], 0.0);
    let identity = compose_with_context_exact_native(&backend, &input, &[], &execution)?;
    assert_eq!(identity.storage_id(), input.storage_id());
    Ok(())
}

#[test]
fn to_tensor_borrows_canonical_rgb8_storage_and_delegates_hwc_conversion()
-> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let scratch = backend
        .workspace_authority
        .authorize_workspace(64 * 1024 * 1024)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
    let image = Rgb8ImageTensor::from_tensor(upload_u8(
        &backend,
        &[1, 2, 3],
        &[0, 127, 255, 64, 128, 192],
        &execution,
    )?)?;
    let output = to_tensor_with_context_exact_native(&backend, &image, &execution)?;
    assert_eq!(output.descriptor().shape(), &[3, 1, 2]);
    assert_close(
        &f32_values(&output)?,
        &[
            0.0,
            64.0 / 255.0,
            127.0 / 255.0,
            128.0 / 255.0,
            1.0,
            192.0 / 255.0,
        ],
        0.0,
    );
    Ok(())
}

#[test]
fn pre_cancelled_external_kernels_take_precedence_over_malformed_inputs()
-> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let setup = CancellationToken::default();
    let setup_execution = context(&backend, &setup, 64 * 1024 * 1024)?;
    let wrong_dtype = upload_u8(&backend, &[1, 4], &[0, 0, 0, 0], &setup_execution)?;
    assert!(
        box_convert_with_context_exact_native(
            &backend,
            &wrong_dtype,
            NativeBoxFormat::CenterXyWidthHeight,
            NativeBoxFormat::Xyxy,
            &setup_execution,
        )
        .is_err()
    );
    let invalid_lab = upload_f32(&backend, &[1, 2, 1, 1], &[0.0, 1.0], &setup_execution)?;
    let invalid_waveform = upload_f32(&backend, &[], &[1.0], &setup_execution)?;
    let invalid_mel = upload_f32(&backend, &[2, 2], &[1.0; 4], &setup_execution)?;
    let invalid_box = upload_f32(&backend, &[3], &[1.0, 2.0, 3.0], &setup_execution)?;
    let invalid_kernel = upload_u8(&backend, &[1, 1], &[1], &setup_execution)?;
    let image = Rgb8ImageTensor::from_tensor(upload_u8(
        &backend,
        &[1, 1, 3],
        &[0, 127, 255],
        &setup_execution,
    )?)?;
    let baseline = backend.memory_snapshot();
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let scratch = backend.workspace_authority.authorize_workspace(0)?;
    let cancelled_execution =
        backend.execution_context(StreamId::DEFAULT, scratch.clone(), &cancelled);
    assert_cancelled(lab_to_rgb_with_context_exact_native(
        &backend,
        &invalid_lab,
        &cancelled_execution,
    ));
    assert_cancelled(lab_to_rgb_jvp_with_context_exact_native(
        &backend,
        &invalid_lab,
        &invalid_box,
        &cancelled_execution,
    ));
    assert_cancelled(lab_to_rgb_vjp_with_context_exact_native(
        &backend,
        &invalid_lab,
        &invalid_box,
        &cancelled_execution,
    ));
    assert_cancelled(bottom_hat_with_context_exact_native(
        &backend,
        &invalid_lab,
        &invalid_kernel,
        &cancelled_execution,
    ));
    assert_cancelled(bass_biquad_with_context_exact_native(
        &backend,
        &invalid_waveform,
        0,
        f64::NAN,
        0.0,
        0.0,
        &cancelled_execution,
    ));
    assert_cancelled(mel_scale_with_context_exact_native(
        &backend,
        &invalid_mel,
        NativeMelScaleConfiguration {
            n_mels: 0,
            ..mel_configuration()
        },
        &cancelled_execution,
    ));
    assert_cancelled(mel_scale_jvp_with_context_exact_native(
        &backend,
        &invalid_mel,
        NativeMelScaleConfiguration {
            n_mels: 0,
            ..mel_configuration()
        },
        &cancelled_execution,
    ));
    assert_cancelled(mel_scale_vjp_with_context_exact_native(
        &backend,
        &invalid_mel,
        &invalid_box,
        NativeMelScaleConfiguration {
            n_mels: 0,
            ..mel_configuration()
        },
        &cancelled_execution,
    ));
    assert_cancelled(box_convert_with_context_exact_native(
        &backend,
        &invalid_box,
        NativeBoxFormat::CenterXyWidthHeight,
        NativeBoxFormat::Xyxy,
        &cancelled_execution,
    ));
    assert_cancelled(box_convert_jvp_with_context_exact_native(
        &backend,
        &invalid_box,
        NativeBoxFormat::CenterXyWidthHeight,
        NativeBoxFormat::Xyxy,
        &cancelled_execution,
    ));
    assert_cancelled(box_convert_vjp_with_context_exact_native(
        &backend,
        &invalid_box,
        NativeBoxFormat::CenterXyWidthHeight,
        NativeBoxFormat::Xyxy,
        &cancelled_execution,
    ));
    assert_cancelled(compose_with_context_exact_native(
        &backend,
        &invalid_lab,
        &[NativeTensorTransform::Normalize {
            mean: vec![],
            standard_deviation: vec![],
        }],
        &cancelled_execution,
    ));
    assert_cancelled(compose_jvp_with_context_exact_native(
        &backend,
        &invalid_lab,
        &[NativeTensorTransform::Normalize {
            mean: vec![],
            standard_deviation: vec![],
        }],
        &cancelled_execution,
    ));
    assert_cancelled(compose_vjp_with_context_exact_native(
        &backend,
        &invalid_lab,
        &[NativeTensorTransform::Normalize {
            mean: vec![],
            standard_deviation: vec![],
        }],
        &cancelled_execution,
    ));
    assert_cancelled(to_tensor_with_context_exact_native(
        &backend,
        &image,
        &cancelled_execution,
    ));
    assert_eq!(scratch.peak_bytes(), 0);
    assert_eq!(scratch.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot(), baseline);
    Ok(())
}
