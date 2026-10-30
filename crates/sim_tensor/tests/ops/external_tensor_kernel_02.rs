use std::{collections::BTreeMap, error::Error, ops::Deref};

use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    StreamId, Tensor, TensorDescriptor, TensorError,
    generated_external_tensor_kernel_02::{
        CANNY_OPERATION_ID, DEFORM_CONV2D_OPERATION_ID, DILATION_OPERATION_ID,
        EINOPS_REARRANGE_OPERATION_ID, EROSION_OPERATION_ID, ExternalTensorKernelPartTwoError,
        NativeDeformConv2dConfiguration, RGB_TO_LAB_OPERATION_ID, RGB_TO_YCBCR_OPERATION_ID,
        TO_PIL_IMAGE_OPERATION_ID, TOP_HAT_OPERATION_ID, YCBCR_TO_RGB_OPERATION_ID,
        canny_with_context_exact_native, deform_conv2d_jvp_with_context_exact_native,
        deform_conv2d_vjp_with_context_exact_native, deform_conv2d_with_context_exact_native,
        dilation_with_context_exact_native, einops_rearrange_jvp_with_context_exact_native,
        einops_rearrange_vjp_with_context_exact_native, einops_rearrange_with_context_exact_native,
        erosion_with_context_exact_native, rgb_to_lab_jvp_with_context_exact_native,
        rgb_to_lab_vjp_with_context_exact_native, rgb_to_lab_with_context_exact_native,
        rgb_to_ycbcr_jvp_with_context_exact_native, rgb_to_ycbcr_vjp_with_context_exact_native,
        rgb_to_ycbcr_with_context_exact_native, to_pil_image_with_context_exact_native,
        top_hat_with_context_exact_native, ycbcr_to_rgb_jvp_with_context_exact_native,
        ycbcr_to_rgb_vjp_with_context_exact_native, ycbcr_to_rgb_with_context_exact_native,
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

fn logical_u8_values(tensor: &Tensor) -> Result<Vec<u8>, Box<dyn Error>> {
    fn visit(
        tensor: &Tensor,
        shape: &[u64],
        indices: &mut Vec<u64>,
        output: &mut Vec<u8>,
    ) -> Result<(), Box<dyn Error>> {
        if indices.len() == shape.len() {
            output.push(tensor.element_bytes(indices)?[0]);
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

fn assert_cancelled<T>(result: Result<T, ExternalTensorKernelPartTwoError>) {
    match result {
        Ok(_) => panic!("pre-cancelled operation unexpectedly succeeded"),
        Err(error) => assert!(
            matches!(error, ExternalTensorKernelPartTwoError::Cancelled),
            "pre-cancelled operation returned the wrong typed error: {error}"
        ),
    }
}

#[test]
fn color_adapter_workspace_is_exact_bounded_cancel_safe_and_convergent()
-> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation, 64 * 1024 * 1024)?;
    let input = upload_f32(&backend, &[1, 3, 1, 1], &[0.2, 0.4, 0.6], &execution)?;

    let exact = backend.workspace_authority.authorize_workspace(12)?;
    let execution = backend.execution_context(StreamId::DEFAULT, exact.clone(), &cancellation);
    let output = rgb_to_lab_with_context_exact_native(&backend, &input, &execution)?;
    assert_eq!(output.descriptor().shape(), input.descriptor().shape());
    assert_eq!(exact.peak_bytes(), 12);
    assert_eq!(exact.in_use_bytes(), 0);

    let baseline = backend.memory_snapshot().current_bytes;
    let insufficient = backend.workspace_authority.authorize_workspace(11)?;
    let execution =
        backend.execution_context(StreamId::DEFAULT, insufficient.clone(), &cancellation);
    assert!(matches!(
        rgb_to_lab_with_context_exact_native(&backend, &input, &execution),
        Err(sim_tensor::generated_external_tensor_kernel_02::ExternalTensorKernelPartTwoError::Tensor(
            TensorError::WorkspaceAuthorizationExceeded { .. }
        ))
    ));
    assert_eq!(insufficient.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let zero = backend.workspace_authority.authorize_workspace(0)?;
    let execution = backend.execution_context(StreamId::DEFAULT, zero.clone(), &cancellation);
    assert!(matches!(
        rgb_to_lab_with_context_exact_native(&backend, &input, &execution),
        Err(sim_tensor::generated_external_tensor_kernel_02::ExternalTensorKernelPartTwoError::Tensor(
            TensorError::WorkspaceAuthorizationExceeded { .. }
        ))
    ));
    assert_eq!(zero.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let scratch = backend.workspace_authority.authorize_workspace(12)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch.clone(), &cancelled);
    assert!(rgb_to_lab_with_context_exact_native(&backend, &input, &execution).is_err());
    assert_eq!(scratch.peak_bytes(), 0);
    assert_eq!(scratch.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn einops_group_ellipsis_units_views_and_byte_exact_dtype_are_native() -> Result<(), Box<dyn Error>>
{
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation, 64 * 1024 * 1024)?;
    let input = upload_u8(
        &backend,
        &[2, 6],
        &(0_u8..12).collect::<Vec<_>>(),
        &execution,
    )?;
    let axes = BTreeMap::from([("h".to_owned(), 2)]);
    let split = einops_rearrange_with_context_exact_native(
        &backend,
        &input,
        "b (h w) -> b h w",
        &axes,
        &execution,
    )?;
    assert_eq!(split.descriptor().shape(), &[2, 2, 3]);
    assert_eq!(split.descriptor().dtype(), DType::U8);
    assert_eq!(split.storage_id(), input.storage_id());
    assert_eq!(logical_u8_values(&split)?, (0_u8..12).collect::<Vec<_>>());

    let merged = einops_rearrange_with_context_exact_native(
        &backend,
        &split,
        "b h w -> b (h w)",
        &BTreeMap::new(),
        &execution,
    )?;
    assert_eq!(merged.descriptor().shape(), &[2, 6]);
    assert_eq!(merged.storage_id(), input.storage_id());
    assert_eq!(logical_u8_values(&merged)?, (0_u8..12).collect::<Vec<_>>());

    let ellipsis_input = upload_u8(
        &backend,
        &[2, 3, 4],
        &(0_u8..24).collect::<Vec<_>>(),
        &execution,
    )?;
    let ellipsis = einops_rearrange_with_context_exact_native(
        &backend,
        &ellipsis_input,
        "... c -> c ...",
        &BTreeMap::new(),
        &execution,
    )?;
    assert_eq!(ellipsis.descriptor().shape(), &[4, 2, 3]);
    assert_eq!(ellipsis.storage_id(), ellipsis_input.storage_id());
    assert_eq!(
        logical_u8_values(&ellipsis)?,
        vec![
            0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23
        ]
    );

    let inserted = einops_rearrange_with_context_exact_native(
        &backend,
        &input,
        "b c -> b 1 c",
        &BTreeMap::new(),
        &execution,
    )?;
    assert_eq!(inserted.descriptor().shape(), &[2, 1, 6]);
    assert_eq!(inserted.storage_id(), input.storage_id());
    let removed = einops_rearrange_with_context_exact_native(
        &backend,
        &inserted,
        "b 1 c -> b c",
        &BTreeMap::new(),
        &execution,
    )?;
    assert_eq!(removed.descriptor().shape(), &[2, 6]);
    assert_eq!(removed.storage_id(), input.storage_id());
    Ok(())
}

#[test]
fn einops_derivatives_empty_inputs_axis_errors_and_cancellation_are_checked()
-> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation, 64 * 1024 * 1024)?;
    let tangent = upload_f32(
        &backend,
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &execution,
    )?;
    let jvp = einops_rearrange_jvp_with_context_exact_native(
        &backend,
        &tangent,
        "b c -> c b",
        &BTreeMap::new(),
        &execution,
    )?;
    assert_close(&f32_values(&jvp)?, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 0.0);
    let vjp = einops_rearrange_vjp_with_context_exact_native(
        &backend,
        &jvp,
        &[2, 3],
        "b c -> c b",
        &BTreeMap::new(),
        &execution,
    )?;
    assert_close(&f32_values(&vjp)?, &f32_values(&tangent)?, 0.0);

    let empty = upload_u8(&backend, &[0, 3], &[], &execution)?;
    let empty_output = einops_rearrange_with_context_exact_native(
        &backend,
        &empty,
        "b c -> c b",
        &BTreeMap::new(),
        &execution,
    )?;
    assert_eq!(empty_output.descriptor().shape(), &[3, 0]);
    assert_eq!(empty_output.storage_id(), empty.storage_id());
    assert!(logical_u8_values(&empty_output)?.is_empty());
    let empty_inverse = einops_rearrange_vjp_with_context_exact_native(
        &backend,
        &empty_output,
        &[0, 3],
        "b c -> c b",
        &BTreeMap::new(),
        &execution,
    )?;
    assert_eq!(empty_inverse.descriptor().shape(), &[0, 3]);
    assert_eq!(empty_inverse.storage_id(), empty.storage_id());
    assert!(logical_u8_values(&empty_inverse)?.is_empty());

    for (pattern, shape, axes) in [
        ("b b -> b", vec![2, 2], BTreeMap::new()),
        ("b c -> b d", vec![2, 3], BTreeMap::new()),
        ("(h w) -> h w", vec![6], BTreeMap::new()),
        (
            "b c -> c b",
            vec![2, 3],
            BTreeMap::from([("unknown".to_owned(), 2)]),
        ),
    ] {
        let input = upload_u8(
            &backend,
            &shape,
            &vec![0; shape.iter().product::<u64>() as usize],
            &execution,
        )?;
        let error = einops_rearrange_with_context_exact_native(
            &backend, &input, pattern, &axes, &execution,
        )
        .expect_err("invalid einops recipe must fail");
        assert!(error.to_string().contains(EINOPS_REARRANGE_OPERATION_ID));
    }

    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_execution = context(&backend, &cancelled, 0)?;
    assert!(
        einops_rearrange_with_context_exact_native(
            &backend,
            &tangent,
            "b c -> c b",
            &BTreeMap::new(),
            &cancelled_execution,
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn kornia_color_vectors_and_analytical_derivatives_match_contracts() -> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation, 64 * 1024 * 1024)?;
    let rgb = upload_f32(
        &backend,
        &[1, 3, 1, 3],
        &[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        &execution,
    )?;
    let lab = rgb_to_lab_with_context_exact_native(&backend, &rgb, &execution)?;
    assert_close(
        &f32_values(&lab)?,
        &[
            0.0,
            100.0,
            53.240_585,
            0.0,
            -0.002_443_790_4,
            80.092_31,
            0.0,
            0.004_649_162_3,
            67.202_74,
        ],
        2.0e-3,
    );

    let primal = upload_f32(&backend, &[1, 3, 1, 1], &[0.2, 0.4, 0.7], &execution)?;
    let tangent = upload_f32(&backend, &[1, 3, 1, 1], &[0.3, -0.2, 0.1], &execution)?;
    let upstream = upload_f32(&backend, &[1, 3, 1, 1], &[-0.4, 0.7, 0.2], &execution)?;
    let jvp = rgb_to_lab_jvp_with_context_exact_native(&backend, &primal, &tangent, &execution)?;
    let vjp = rgb_to_lab_vjp_with_context_exact_native(&backend, &primal, &upstream, &execution)?;
    assert_close(
        &[dot(&f32_values(&jvp)?, &f32_values(&upstream)?)],
        &[dot(&f32_values(&tangent)?, &f32_values(&vjp)?)],
        2.0e-4,
    );
    let epsilon = 1.0e-3_f32;
    let plus = upload_f32(
        &backend,
        &[1, 3, 1, 1],
        &[
            0.2 + 0.3 * epsilon,
            0.4 - 0.2 * epsilon,
            0.7 + 0.1 * epsilon,
        ],
        &execution,
    )?;
    let minus = upload_f32(
        &backend,
        &[1, 3, 1, 1],
        &[
            0.2 - 0.3 * epsilon,
            0.4 + 0.2 * epsilon,
            0.7 - 0.1 * epsilon,
        ],
        &execution,
    )?;
    let finite_difference = f32_values(&rgb_to_lab_with_context_exact_native(
        &backend, &plus, &execution,
    )?)?
    .into_iter()
    .zip(f32_values(&rgb_to_lab_with_context_exact_native(
        &backend, &minus, &execution,
    )?)?)
    .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
    .collect::<Vec<_>>();
    assert_close(&f32_values(&jvp)?, &finite_difference, 2.0e-2);

    let ycbcr = rgb_to_ycbcr_with_context_exact_native(&backend, &rgb, &execution)?;
    assert_close(
        &f32_values(&ycbcr)?,
        &[0.0, 1.0, 0.299, 0.5, 0.5, 0.331_364, 0.5, 0.5, 0.999_813],
        2.0e-6,
    );
    let roundtrip = ycbcr_to_rgb_with_context_exact_native(&backend, &ycbcr, &execution)?;
    assert_close(&f32_values(&roundtrip)?, &f32_values(&rgb)?, 6.0e-4);

    let linear_jvp = rgb_to_ycbcr_jvp_with_context_exact_native(&backend, &tangent, &execution)?;
    let linear_vjp = rgb_to_ycbcr_vjp_with_context_exact_native(&backend, &upstream, &execution)?;
    assert_close(
        &[dot(&f32_values(&linear_jvp)?, &f32_values(&upstream)?)],
        &[dot(&f32_values(&tangent)?, &f32_values(&linear_vjp)?)],
        2.0e-6,
    );
    Ok(())
}

#[test]
fn ycbcr_inverse_derivatives_apply_output_clamp_masks() -> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation, 64 * 1024 * 1024)?;
    let primal = upload_f32(
        &backend,
        &[1, 3, 1, 2],
        &[0.5, 0.0, 0.5, 0.0, 0.5, 1.0],
        &execution,
    )?;
    let tangent = upload_f32(
        &backend,
        &[1, 3, 1, 2],
        &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        &execution,
    )?;
    let upstream = upload_f32(
        &backend,
        &[1, 3, 1, 2],
        &[0.2, 0.2, -0.4, -0.4, 0.7, 0.7],
        &execution,
    )?;
    let output = ycbcr_to_rgb_with_context_exact_native(&backend, &primal, &execution)?;
    assert_close(
        &f32_values(&output)?,
        &[0.5, 0.7015, 0.5, 0.0, 0.5, 0.0],
        2.0e-6,
    );
    let jvp = ycbcr_to_rgb_jvp_with_context_exact_native(&backend, &primal, &tangent, &execution)?;
    assert_close(
        &f32_values(&jvp)?,
        &[5.209, 5.209, -1.83, 0.0, 4.546, 0.0],
        2.0e-5,
    );
    let vjp = ycbcr_to_rgb_vjp_with_context_exact_native(&backend, &primal, &upstream, &execution)?;
    assert_close(
        &[dot(&f32_values(&jvp)?, &f32_values(&upstream)?)],
        &[dot(&f32_values(&tangent)?, &f32_values(&vjp)?)],
        2.0e-5,
    );
    Ok(())
}

#[test]
fn canny_is_deterministic_for_constant_and_step_edges_and_checks_boundaries()
-> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation, 64 * 1024 * 1024)?;
    let constant = upload_f32(&backend, &[1, 1, 7, 7], &[0.0; 49], &execution)?;
    let constant_output =
        canny_with_context_exact_native(&backend, &constant, 0.1, 0.2, &execution)?;
    assert_close(&f32_values(constant_output.magnitude())?, &[0.0; 49], 0.0);
    assert_close(&f32_values(constant_output.edges())?, &[0.0; 49], 0.0);

    let step_values = (0..7)
        .flat_map(|_| (0..7).map(|x| if x < 3 { 0.0 } else { 1.0 }))
        .collect::<Vec<_>>();
    let step = upload_f32(&backend, &[1, 1, 7, 7], &step_values, &execution)?;
    let first = canny_with_context_exact_native(&backend, &step, 0.1, 0.2, &execution)?;
    let second = canny_with_context_exact_native(&backend, &step, 0.1, 0.2, &execution)?;
    assert_eq!(
        first.magnitude().contiguous_bytes()?,
        second.magnitude().contiguous_bytes()?
    );
    assert_eq!(
        first.edges().contiguous_bytes()?,
        second.edges().contiguous_bytes()?
    );
    let edge_values = f32_values(first.edges())?;
    assert!(edge_values.contains(&1.0));
    assert!(edge_values.iter().all(|value| matches!(*value, 0.0 | 1.0)));

    for (low, high) in [(0.0, 0.2), (0.3, 0.2), (0.1, 1.0), (f32::NAN, 0.2)] {
        let error = canny_with_context_exact_native(&backend, &step, low, high, &execution)
            .expect_err("invalid Canny thresholds must fail");
        assert!(error.to_string().contains(CANNY_OPERATION_ID));
    }
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_execution = context(&backend, &cancelled, 0)?;
    assert!(
        canny_with_context_exact_native(&backend, &step, 0.1, 0.2, &cancelled_execution).is_err()
    );
    Ok(())
}

#[test]
fn canny_adapter_uses_exact_simultaneous_workspace_and_rejects_one_byte_short()
-> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation, 64 * 1024 * 1024)?;
    let input = upload_f32(&backend, &[1, 1, 7, 7], &[0.0; 49], &execution)?;

    let exact = backend.workspace_authority.authorize_workspace(637)?;
    let execution = backend.execution_context(StreamId::DEFAULT, exact.clone(), &cancellation);
    canny_with_context_exact_native(&backend, &input, 0.1, 0.2, &execution)?;
    assert_eq!(exact.peak_bytes(), 637);
    assert_eq!(exact.in_use_bytes(), 0);

    let baseline = backend.memory_snapshot().current_bytes;
    let insufficient = backend.workspace_authority.authorize_workspace(636)?;
    let execution =
        backend.execution_context(StreamId::DEFAULT, insufficient.clone(), &cancellation);
    assert!(canny_with_context_exact_native(&backend, &input, 0.1, 0.2, &execution).is_err());
    assert_eq!(insufficient.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn canny_diagonal_nms_directions_match_kornia_0_8_2_exact_fixtures() -> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation, 64 * 1024 * 1024)?;
    let cases = [
        (
            (0..7)
                .flat_map(|y| (0..7).map(move |x| if x >= y { 1.0 } else { 0.0 }))
                .collect::<Vec<_>>(),
            vec![
                0.751_405_66,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.063_018_8,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.035_280_5,
                2.492_661_2,
                2.597_311_5,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.624_189_9,
                2.628_247_5,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.624_189_9,
                2.597_311_5,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.492_661_2,
                2.063_019,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.035_280_7,
                0.0,
                0.751_405_9,
            ],
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                1.0,
            ],
        ),
        (
            (0..7)
                .flat_map(|y| (0..7).map(move |x| if x + y >= 6 { 1.0 } else { 0.0 }))
                .collect::<Vec<_>>(),
            vec![
                0.0,
                0.0,
                0.0,
                0.0,
                2.035_280_7,
                0.0,
                0.751_405_9,
                0.0,
                0.0,
                0.0,
                0.0,
                2.492_661,
                2.063_019,
                0.0,
                0.0,
                0.0,
                0.0,
                2.624_189_6,
                2.597_311_5,
                0.0,
                0.0,
                0.0,
                0.0,
                2.624_189_6,
                2.628_247_5,
                0.0,
                0.0,
                0.0,
                2.035_280_7,
                2.492_661_2,
                2.597_311_5,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.063_018_8,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.751_405_9,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0,
            ],
        ),
    ];

    for (input_values, expected_magnitude, expected_edges) in cases {
        let input = upload_f32(&backend, &[1, 1, 7, 7], &input_values, &execution)?;
        let output = canny_with_context_exact_native(&backend, &input, 0.1, 0.2, &execution)?;
        assert_close(
            &f32_values(output.magnitude())?,
            &expected_magnitude,
            2.0e-5,
        );
        assert_close(&f32_values(output.edges())?, &expected_edges, 0.0);
    }
    Ok(())
}

#[test]
fn morphology_facades_delegate_exact_semantics_and_report_their_contract_ids()
-> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let scratch = backend
        .workspace_authority
        .authorize_workspace(64 * 1024 * 1024)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
    let input = upload_f32(
        &backend,
        &[1, 1, 3, 3],
        &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        &execution,
    )?;
    let kernel = upload_f32(&backend, &[3, 3], &[1.0; 9], &execution)?;
    assert_close(
        &f32_values(&dilation_with_context_exact_native(
            &backend, &input, &kernel, &execution,
        )?)?,
        &[1.0; 9],
        0.0,
    );
    assert_close(
        &f32_values(&erosion_with_context_exact_native(
            &backend, &input, &kernel, &execution,
        )?)?,
        &[0.0; 9],
        0.0,
    );
    assert_close(
        &f32_values(&top_hat_with_context_exact_native(
            &backend, &input, &kernel, &execution,
        )?)?,
        &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        0.0,
    );

    let wrong_kernel = upload_u8(&backend, &[1, 1], &[1], &execution)?;
    for (result, operation_id) in [
        (
            dilation_with_context_exact_native(&backend, &input, &wrong_kernel, &execution),
            DILATION_OPERATION_ID,
        ),
        (
            erosion_with_context_exact_native(&backend, &input, &wrong_kernel, &execution),
            EROSION_OPERATION_ID,
        ),
        (
            top_hat_with_context_exact_native(&backend, &input, &wrong_kernel, &execution),
            TOP_HAT_OPERATION_ID,
        ),
    ] {
        assert!(
            result
                .expect_err("wrong morphology dtype must fail")
                .to_string()
                .contains(operation_id)
        );
    }
    Ok(())
}

#[test]
fn morphology_sparse_asymmetric_and_geodesic_boundaries_match_kornia_0_8_2()
-> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let scratch = backend
        .workspace_authority
        .authorize_workspace(64 * 1024 * 1024)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
    let input = upload_f32(
        &backend,
        &[1, 1, 3, 4],
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &execution,
    )?;
    let asymmetric = upload_f32(
        &backend,
        &[2, 3],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &execution,
    )?;
    assert_close(
        &f32_values(&dilation_with_context_exact_native(
            &backend,
            &input,
            &asymmetric,
            &execution,
        )?)?,
        &[
            2.0, 3.0, 4.0, -9_996.0, 6.0, 7.0, 8.0, 4.0, 10.0, 11.0, 12.0, 8.0,
        ],
        0.0,
    );
    assert_close(
        &f32_values(&erosion_with_context_exact_native(
            &backend,
            &input,
            &asymmetric,
            &execution,
        )?)?,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 9.0, 5.0, 6.0, 7.0],
        0.0,
    );

    let boundary = upload_f32(
        &backend,
        &[3, 3],
        &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &execution,
    )?;
    assert_close(
        &f32_values(&dilation_with_context_exact_native(
            &backend, &input, &boundary, &execution,
        )?)?,
        &[
            6.0, 7.0, 8.0, -9_992.0, 10.0, 11.0, 12.0, -9_988.0, -9_990.0, -9_989.0, -9_988.0,
            -9_988.0,
        ],
        0.0,
    );
    assert_close(
        &f32_values(&erosion_with_context_exact_native(
            &backend, &input, &boundary, &execution,
        )?)?,
        &[
            10_000.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0, 1.0, 2.0, 3.0, 10_000.0, 5.0, 6.0,
            7.0,
        ],
        0.0,
    );
    Ok(())
}

#[test]
fn morphology_all_zero_flat_kernel_matches_kornia_0_8_2_finite_infinity()
-> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let scratch = backend
        .workspace_authority
        .authorize_workspace(64 * 1024 * 1024)?;
    let execution = backend.execution_context(StreamId::DEFAULT, scratch, &cancellation);
    let input = upload_f32(
        &backend,
        &[1, 1, 3, 4],
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &execution,
    )?;
    let kernel = upload_f32(&backend, &[2, 2], &[0.0; 4], &execution)?;
    assert_close(
        &f32_values(&dilation_with_context_exact_native(
            &backend, &input, &kernel, &execution,
        )?)?,
        &[
            -9_999.0, -9_998.0, -9_997.0, -9_996.0, -9_995.0, -9_994.0, -9_993.0, -9_992.0,
            -9_991.0, -9_990.0, -9_989.0, -9_988.0,
        ],
        0.0,
    );
    assert_close(
        &f32_values(&erosion_with_context_exact_native(
            &backend, &input, &kernel, &execution,
        )?)?,
        &[
            10_001.0, 10_001.0, 10_002.0, 10_003.0, 10_001.0, 10_001.0, 10_002.0, 10_003.0,
            10_005.0, 10_005.0, 10_006.0, 10_007.0,
        ],
        0.0,
    );
    Ok(())
}

#[test]
fn deform_conv_matches_zero_offset_convolution_and_fractional_zero_boundary_sampling()
-> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation, 64 * 1024 * 1024)?;
    let input = upload_f32(
        &backend,
        &[1, 1, 3, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &execution,
    )?;
    let weight = upload_f32(&backend, &[1, 1, 2, 2], &[1.0; 4], &execution)?;
    let offset = upload_f32(&backend, &[1, 8, 2, 2], &[0.0; 32], &execution)?;
    let output = deform_conv2d_with_context_exact_native(
        &backend,
        &input,
        &offset,
        &weight,
        None,
        NativeDeformConv2dConfiguration::default(),
        None,
        &execution,
    )?;
    assert_close(&f32_values(&output)?, &[12.0, 16.0, 24.0, 28.0], 0.0);

    let small = upload_f32(&backend, &[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0], &execution)?;
    let point_weight = upload_f32(&backend, &[1, 1, 1, 1], &[1.0], &execution)?;
    let fractional_offset = upload_f32(
        &backend,
        &[1, 2, 2, 2],
        &[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        &execution,
    )?;
    let fractional = deform_conv2d_with_context_exact_native(
        &backend,
        &small,
        &fractional_offset,
        &point_weight,
        None,
        NativeDeformConv2dConfiguration::default(),
        None,
        &execution,
    )?;
    assert_close(&f32_values(&fractional)?, &[2.5, 1.5, 1.75, 1.0], 1.0e-6);
    Ok(())
}

#[test]
fn deform_conv_adapter_leases_tensor_copies_and_output_staging_exactly()
-> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation, 64 * 1024 * 1024)?;
    let input = upload_f32(&backend, &[1, 1, 3, 3], &[1.0; 9], &execution)?;
    let weight = upload_f32(&backend, &[1, 1, 2, 2], &[1.0; 4], &execution)?;
    let offset = upload_f32(&backend, &[1, 8, 2, 2], &[0.0; 32], &execution)?;

    let exact = backend.workspace_authority.authorize_workspace(196)?;
    let execution = backend.execution_context(StreamId::DEFAULT, exact.clone(), &cancellation);
    deform_conv2d_with_context_exact_native(
        &backend,
        &input,
        &offset,
        &weight,
        None,
        NativeDeformConv2dConfiguration::default(),
        None,
        &execution,
    )?;
    assert_eq!(exact.peak_bytes(), 196);
    assert_eq!(exact.in_use_bytes(), 0);

    let baseline = backend.memory_snapshot().current_bytes;
    let insufficient = backend.workspace_authority.authorize_workspace(195)?;
    let execution =
        backend.execution_context(StreamId::DEFAULT, insufficient.clone(), &cancellation);
    assert!(
        deform_conv2d_with_context_exact_native(
            &backend,
            &input,
            &offset,
            &weight,
            None,
            NativeDeformConv2dConfiguration::default(),
            None,
            &execution,
        )
        .is_err()
    );
    assert_eq!(insufficient.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);
    Ok(())
}

#[test]
fn deform_conv_supports_mask_groups_bias_and_nontrivial_geometry() -> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation, 64 * 1024 * 1024)?;
    let grouped_input = upload_f32(
        &backend,
        &[1, 2, 2, 2],
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        &execution,
    )?;
    let grouped_weight = upload_f32(&backend, &[2, 1, 1, 1], &[2.0, 3.0], &execution)?;
    let grouped_offset = upload_f32(&backend, &[1, 4, 2, 2], &[0.0; 16], &execution)?;
    let grouped_mask = upload_f32(&backend, &[1, 2, 2, 2], &[0.5; 8], &execution)?;
    let bias = upload_f32(&backend, &[2], &[1.0, -1.0], &execution)?;
    let grouped = deform_conv2d_with_context_exact_native(
        &backend,
        &grouped_input,
        &grouped_offset,
        &grouped_weight,
        Some(&bias),
        NativeDeformConv2dConfiguration::default(),
        Some(&grouped_mask),
        &execution,
    )?;
    assert_close(
        &f32_values(&grouped)?,
        &[2.0, 3.0, 4.0, 5.0, 14.0, 29.0, 44.0, 59.0],
        0.0,
    );

    let geometry_input = upload_f32(&backend, &[1, 1, 4, 4], &[1.0; 16], &execution)?;
    let geometry_weight = upload_f32(&backend, &[1, 1, 2, 2], &[1.0; 4], &execution)?;
    let geometry_offset = upload_f32(&backend, &[1, 8, 2, 2], &[0.0; 32], &execution)?;
    let geometry = deform_conv2d_with_context_exact_native(
        &backend,
        &geometry_input,
        &geometry_offset,
        &geometry_weight,
        None,
        NativeDeformConv2dConfiguration {
            stride: [2, 2],
            padding: [1, 1],
            dilation: [2, 2],
        },
        None,
        &execution,
    )?;
    assert_close(&f32_values(&geometry)?, &[1.0, 2.0, 2.0, 4.0], 0.0);
    Ok(())
}

#[test]
fn deform_conv_analytical_jvp_vjp_match_finite_difference_and_adjoint() -> Result<(), Box<dyn Error>>
{
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation, 64 * 1024 * 1024)?;
    let input_values = [1.0, 2.0, 3.0, 4.0];
    let offset_values = [0.2, 0.2, -0.2, -0.2, 0.3, -0.3, 0.3, -0.3];
    let weight_values = [0.7];
    let bias_values = [0.1];
    let mask_values = [0.8, 0.9, 1.1, 1.2];
    let input_tangent_values = [0.1, -0.2, 0.3, -0.4];
    let offset_tangent_values = [0.05, -0.03, 0.02, -0.04, -0.02, 0.04, -0.01, 0.03];
    let weight_tangent_values = [-0.15];
    let bias_tangent_values = [0.25];
    let mask_tangent_values = [0.07, -0.06, 0.05, -0.04];
    let input = upload_f32(&backend, &[1, 1, 2, 2], &input_values, &execution)?;
    let offset = upload_f32(&backend, &[1, 2, 2, 2], &offset_values, &execution)?;
    let weight = upload_f32(&backend, &[1, 1, 1, 1], &weight_values, &execution)?;
    let bias = upload_f32(&backend, &[1], &bias_values, &execution)?;
    let mask = upload_f32(&backend, &[1, 1, 2, 2], &mask_values, &execution)?;
    let input_tangent = upload_f32(&backend, &[1, 1, 2, 2], &input_tangent_values, &execution)?;
    let offset_tangent = upload_f32(&backend, &[1, 2, 2, 2], &offset_tangent_values, &execution)?;
    let weight_tangent = upload_f32(&backend, &[1, 1, 1, 1], &weight_tangent_values, &execution)?;
    let bias_tangent = upload_f32(&backend, &[1], &bias_tangent_values, &execution)?;
    let mask_tangent = upload_f32(&backend, &[1, 1, 2, 2], &mask_tangent_values, &execution)?;
    let configuration = NativeDeformConv2dConfiguration::default();
    let jvp = deform_conv2d_jvp_with_context_exact_native(
        &backend,
        &input,
        Some(&input_tangent),
        &offset,
        Some(&offset_tangent),
        &weight,
        Some(&weight_tangent),
        Some(&bias),
        Some(&bias_tangent),
        configuration,
        Some(&mask),
        Some(&mask_tangent),
        &execution,
    )?;

    let epsilon = 1.0e-3_f32;
    let perturb = |primal: &[f32], tangent: &[f32], direction: f32| {
        primal
            .iter()
            .zip(tangent)
            .map(|(primal, tangent)| primal + direction * epsilon * tangent)
            .collect::<Vec<_>>()
    };
    let evaluate = |direction: f32| -> Result<Vec<f32>, Box<dyn Error>> {
        let input = upload_f32(
            &backend,
            &[1, 1, 2, 2],
            &perturb(&input_values, &input_tangent_values, direction),
            &execution,
        )?;
        let offset = upload_f32(
            &backend,
            &[1, 2, 2, 2],
            &perturb(&offset_values, &offset_tangent_values, direction),
            &execution,
        )?;
        let weight = upload_f32(
            &backend,
            &[1, 1, 1, 1],
            &perturb(&weight_values, &weight_tangent_values, direction),
            &execution,
        )?;
        let bias = upload_f32(
            &backend,
            &[1],
            &perturb(&bias_values, &bias_tangent_values, direction),
            &execution,
        )?;
        let mask = upload_f32(
            &backend,
            &[1, 1, 2, 2],
            &perturb(&mask_values, &mask_tangent_values, direction),
            &execution,
        )?;
        f32_values(&deform_conv2d_with_context_exact_native(
            &backend,
            &input,
            &offset,
            &weight,
            Some(&bias),
            configuration,
            Some(&mask),
            &execution,
        )?)
    };
    let finite_difference = evaluate(1.0)?
        .into_iter()
        .zip(evaluate(-1.0)?)
        .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
        .collect::<Vec<_>>();
    assert_close(&f32_values(&jvp)?, &finite_difference, 5.0e-4);

    let upstream_values = [0.2, -0.4, 0.7, 0.1];
    let upstream = upload_f32(&backend, &[1, 1, 2, 2], &upstream_values, &execution)?;
    let vjp = deform_conv2d_vjp_with_context_exact_native(
        &backend,
        &input,
        &offset,
        &weight,
        Some(&bias),
        configuration,
        Some(&mask),
        &upstream,
        &execution,
    )?;
    let adjoint = dot(&input_tangent_values, &f32_values(&vjp.input)?)
        + dot(&offset_tangent_values, &f32_values(&vjp.offset)?)
        + dot(&weight_tangent_values, &f32_values(&vjp.weight)?)
        + dot(
            &bias_tangent_values,
            &f32_values(vjp.bias.as_ref().ok_or("missing bias gradient")?)?,
        )
        + dot(
            &mask_tangent_values,
            &f32_values(vjp.mask.as_ref().ok_or("missing mask gradient")?)?,
        );
    assert_close(
        &[dot(&f32_values(&jvp)?, &upstream_values)],
        &[adjoint],
        3.0e-5,
    );
    Ok(())
}

#[test]
fn deform_conv_rejects_invalid_geometry_and_honors_pre_cancellation() -> Result<(), Box<dyn Error>>
{
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation, 64 * 1024 * 1024)?;
    let input = upload_f32(&backend, &[1, 1, 2, 2], &[1.0; 4], &execution)?;
    let weight = upload_f32(&backend, &[1, 1, 1, 1], &[1.0], &execution)?;
    let wrong_offset = upload_f32(&backend, &[1, 1, 2, 2], &[0.0; 4], &execution)?;
    let error = deform_conv2d_with_context_exact_native(
        &backend,
        &input,
        &wrong_offset,
        &weight,
        None,
        NativeDeformConv2dConfiguration::default(),
        None,
        &execution,
    )
    .expect_err("invalid offset channels must fail");
    assert!(error.to_string().contains(DEFORM_CONV2D_OPERATION_ID));

    let offset = upload_f32(&backend, &[1, 2, 2, 2], &[0.0; 8], &execution)?;
    assert!(
        deform_conv2d_with_context_exact_native(
            &backend,
            &input,
            &offset,
            &weight,
            None,
            NativeDeformConv2dConfiguration {
                stride: [0, 1],
                ..NativeDeformConv2dConfiguration::default()
            },
            None,
            &execution,
        )
        .is_err()
    );
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_execution = context(&backend, &cancelled, 0)?;
    assert!(
        deform_conv2d_with_context_exact_native(
            &backend,
            &input,
            &offset,
            &weight,
            None,
            NativeDeformConv2dConfiguration::default(),
            None,
            &cancelled_execution,
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn pre_cancelled_external_kernels_take_precedence_over_malformed_inputs()
-> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let setup = CancellationToken::default();
    let setup_execution = context(&backend, &setup, 0)?;
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let cancelled_execution = context(&backend, &cancelled, 0)?;

    let scalar_f32 = upload_f32(&backend, &[1], &[0.0], &setup_execution)?;
    let scalar_u8 = upload_u8(&backend, &[1], &[0], &setup_execution)?;
    let malformed_offset = upload_f32(&backend, &[1], &[0.0], &setup_execution)?;
    let memory_before_cancellation = backend.memory_snapshot();
    assert_cancelled(einops_rearrange_with_context_exact_native(
        &backend,
        &scalar_f32,
        "(",
        &BTreeMap::new(),
        &cancelled_execution,
    ));
    assert_cancelled(einops_rearrange_jvp_with_context_exact_native(
        &backend,
        &scalar_f32,
        "(",
        &BTreeMap::new(),
        &cancelled_execution,
    ));
    assert_cancelled(einops_rearrange_vjp_with_context_exact_native(
        &backend,
        &scalar_f32,
        &[],
        "(",
        &BTreeMap::new(),
        &cancelled_execution,
    ));

    assert_cancelled(rgb_to_lab_with_context_exact_native(
        &backend,
        &scalar_u8,
        &cancelled_execution,
    ));
    assert_cancelled(rgb_to_lab_jvp_with_context_exact_native(
        &backend,
        &scalar_u8,
        &scalar_u8,
        &cancelled_execution,
    ));
    assert_cancelled(rgb_to_lab_vjp_with_context_exact_native(
        &backend,
        &scalar_u8,
        &scalar_u8,
        &cancelled_execution,
    ));
    assert_cancelled(rgb_to_ycbcr_with_context_exact_native(
        &backend,
        &scalar_u8,
        &cancelled_execution,
    ));
    assert_cancelled(rgb_to_ycbcr_jvp_with_context_exact_native(
        &backend,
        &scalar_u8,
        &cancelled_execution,
    ));
    assert_cancelled(rgb_to_ycbcr_vjp_with_context_exact_native(
        &backend,
        &scalar_u8,
        &cancelled_execution,
    ));
    assert_cancelled(ycbcr_to_rgb_with_context_exact_native(
        &backend,
        &scalar_u8,
        &cancelled_execution,
    ));
    assert_cancelled(ycbcr_to_rgb_jvp_with_context_exact_native(
        &backend,
        &scalar_u8,
        &scalar_u8,
        &cancelled_execution,
    ));
    assert_cancelled(ycbcr_to_rgb_vjp_with_context_exact_native(
        &backend,
        &scalar_u8,
        &scalar_u8,
        &cancelled_execution,
    ));
    assert_cancelled(canny_with_context_exact_native(
        &backend,
        &scalar_u8,
        f32::NAN,
        -1.0,
        &cancelled_execution,
    ));

    assert_cancelled(deform_conv2d_with_context_exact_native(
        &backend,
        &scalar_f32,
        &malformed_offset,
        &scalar_f32,
        None,
        NativeDeformConv2dConfiguration {
            stride: [0, 0],
            padding: [0, 0],
            dilation: [0, 0],
        },
        None,
        &cancelled_execution,
    ));
    assert_cancelled(deform_conv2d_vjp_with_context_exact_native(
        &backend,
        &scalar_f32,
        &malformed_offset,
        &scalar_f32,
        None,
        NativeDeformConv2dConfiguration {
            stride: [0, 0],
            padding: [0, 0],
            dilation: [0, 0],
        },
        None,
        &scalar_f32,
        &cancelled_execution,
    ));
    assert_cancelled(deform_conv2d_jvp_with_context_exact_native(
        &backend,
        &scalar_f32,
        None,
        &malformed_offset,
        None,
        &scalar_f32,
        None,
        None,
        None,
        NativeDeformConv2dConfiguration {
            stride: [0, 0],
            padding: [0, 0],
            dilation: [0, 0],
        },
        None,
        None,
        &cancelled_execution,
    ));

    assert_cancelled(dilation_with_context_exact_native(
        &backend,
        &scalar_u8,
        &scalar_u8,
        &cancelled_execution,
    ));
    assert_cancelled(erosion_with_context_exact_native(
        &backend,
        &scalar_u8,
        &scalar_u8,
        &cancelled_execution,
    ));
    assert_cancelled(top_hat_with_context_exact_native(
        &backend,
        &scalar_u8,
        &scalar_u8,
        &cancelled_execution,
    ));
    assert_cancelled(to_pil_image_with_context_exact_native(
        &backend,
        &scalar_u8,
        &cancelled_execution,
    ));
    assert_eq!(cancelled_execution.scratch.peak_bytes(), 0);
    assert_eq!(cancelled_execution.scratch.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot(), memory_before_cancellation);
    Ok(())
}

#[test]
fn to_pil_rgb8_interleaving_roundtrips_through_canonical_tensor_rearrange()
-> Result<(), Box<dyn Error>> {
    let backend = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &cancellation, 64 * 1024 * 1024)?;
    let chw = upload_u8(
        &backend,
        &[3, 2, 2],
        &[1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24],
        &execution,
    )?;
    let exact = backend.workspace_authority.authorize_workspace(12)?;
    let execution = backend.execution_context(StreamId::DEFAULT, exact.clone(), &cancellation);
    let image = to_pil_image_with_context_exact_native(&backend, &chw, &execution)?;
    assert_eq!(exact.peak_bytes(), 12);
    assert_eq!(exact.in_use_bytes(), 0);
    assert_eq!(image.dimensions()?, (2, 2));
    assert_eq!(
        image.as_u8_slice()?,
        &[1, 11, 21, 2, 12, 22, 3, 13, 23, 4, 14, 24]
    );
    let roundtrip = einops_rearrange_with_context_exact_native(
        &backend,
        image.tensor(),
        "h w c -> c h w",
        &BTreeMap::new(),
        &execution,
    )?;
    assert_eq!(roundtrip.descriptor().shape(), &[3, 2, 2]);
    assert_eq!(logical_u8_values(&roundtrip)?, chw.contiguous_bytes()?);

    let baseline = backend.memory_snapshot().current_bytes;
    let insufficient = backend.workspace_authority.authorize_workspace(11)?;
    let execution =
        backend.execution_context(StreamId::DEFAULT, insufficient.clone(), &cancellation);
    assert!(matches!(
        to_pil_image_with_context_exact_native(&backend, &chw, &execution),
        Err(sim_tensor::generated_external_tensor_kernel_02::ExternalTensorKernelPartTwoError::Tensor(
            TensorError::WorkspaceAuthorizationExceeded { .. }
        ))
    ));
    assert_eq!(insufficient.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, baseline);

    let invalid = upload_u8(&backend, &[1, 2, 2], &[0; 4], &execution)?;
    let error = to_pil_image_with_context_exact_native(&backend, &invalid, &execution)
        .expect_err("non-RGB logical image must fail");
    assert!(error.to_string().contains("three channels"));

    assert_eq!(RGB_TO_LAB_OPERATION_ID, "COMFY-TENSOR-OP-4F9C05E204D4");
    assert_eq!(RGB_TO_YCBCR_OPERATION_ID, "COMFY-TENSOR-OP-A555F803F554");
    assert_eq!(YCBCR_TO_RGB_OPERATION_ID, "COMFY-TENSOR-OP-9EF1D9EB674A");
    assert_eq!(TO_PIL_IMAGE_OPERATION_ID, "COMFY-TENSOR-OP-B7926028DA57");
    Ok(())
}
