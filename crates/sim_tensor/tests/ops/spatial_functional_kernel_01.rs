use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
    Scalar, StreamId, TensorDescriptor,
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
    generated_spatial_functional_kernel_01::{
        AVG_POOL_1D_OPERATION_ID, AveragePoolConfiguration, ConvolutionConfiguration,
        GRID_SAMPLE_OPERATION_ID, GridPaddingMode, GridSampleConfiguration, GridSampleMode,
        INTERPOLATE_OPERATION_ID, InterpolateConfiguration, InterpolateMode,
        SpatialFunctionalKernelError, average_pool_1d_with_context_exact_native,
        average_pool_2d_tensor_with_context_exact_native,
        average_pool_2d_with_context_exact_native, average_pool_3d_with_context_exact_native,
        average_pool_jvp_with_context_exact_native, average_pool_vjp_with_context_exact_native,
        bislerp_tensor_with_context_exact_native, conv_1d_with_context_exact_native,
        conv_2d_tensor_with_context_exact_native, conv_2d_with_context_exact_native,
        conv_3d_tensor_with_context_exact_native, conv_3d_with_context_exact_native,
        conv_transpose_1d_with_context_exact_native,
        conv_transpose_2d_tensor_with_context_exact_native,
        conv_transpose_2d_with_context_exact_native, conv_transpose_3d_with_context_exact_native,
        convolution_jvp_with_context_exact_native, convolution_vjp_with_context_exact_native,
        grid_sample_jvp_with_context_exact_native, grid_sample_tensor_with_context_exact_native,
        grid_sample_vjp_with_context_exact_native, grid_sample_with_context_exact_native,
        interpolate_jvp_with_context_exact_native, interpolate_tensor_with_context_exact_native,
        interpolate_vjp_with_context_exact_native, interpolate_with_context_exact_native,
        max_pool_2d_jvp_with_context_exact_native, max_pool_2d_vjp_with_context_exact_native,
        max_pool_2d_with_context_exact_native, pixel_shuffle_nd_tensor_with_context_exact_native,
    },
};
use std::{fs, ops::Deref, path::Path};

struct TestBackend {
    backend: CpuBackend,
    authority: CpuWorkspaceAuthority,
    limit: u64,
}

#[test]
fn latent_upscale_shuffle_and_bislerp_oracles_are_source_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let upload = |shape: Vec<u64>, values: &[f32]| {
        let descriptor =
            TensorDescriptor::contiguous(shape, DType::F32, DeviceId::CPU, context.stream)?;
        Ok::<_, Box<dyn std::error::Error>>(backend.upload_f32(descriptor, values, &context)?.0)
    };

    let temporal = upload(vec![1, 2, 2, 1, 1], &[10.0, 20.0, 11.0, 21.0])?;
    let temporal =
        pixel_shuffle_nd_tensor_with_context_exact_native(&backend, &temporal, 1, 2, &context)?;
    assert_eq!(temporal.descriptor().shape(), &[1, 1, 4, 1, 1]);
    close(
        &tensor_to_f32_with_context_exact_native(&backend, &temporal, &context)?,
        &[10.0, 11.0, 20.0, 21.0],
        0.0,
    );

    let spatiotemporal = upload(
        vec![1, 8, 1, 1, 1],
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )?;
    let spatiotemporal = pixel_shuffle_nd_tensor_with_context_exact_native(
        &backend,
        &spatiotemporal,
        3,
        2,
        &context,
    )?;
    assert_eq!(spatiotemporal.descriptor().shape(), &[1, 1, 2, 2, 2]);
    close(
        &tensor_to_f32_with_context_exact_native(&backend, &spatiotemporal, &context)?,
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        0.0,
    );

    let coincident = upload(vec![1, 2, 1, 2], &[3.0, 3.0, 4.0, 4.0])?;
    let coincident =
        bislerp_tensor_with_context_exact_native(&backend, &coincident, 3, 1, &context)?;
    close(
        &tensor_to_f32_with_context_exact_native(&backend, &coincident, &context)?,
        &[3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
        0.0,
    );

    let antipodal = upload(vec![1, 2, 1, 2], &[1.0, -1.0, 0.0, 0.0])?;
    let antipodal = bislerp_tensor_with_context_exact_native(&backend, &antipodal, 3, 1, &context)?;
    close(
        &tensor_to_f32_with_context_exact_native(&backend, &antipodal, &context)?,
        &[1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        0.0,
    );
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn latent_upscale_shuffle_and_bislerp_preserve_typed_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let descriptor = TensorDescriptor::contiguous(
        vec![1, 8, 1, 1, 1],
        DType::F32,
        DeviceId::CPU,
        context.stream,
    )?;
    let input = backend.upload_f32(descriptor, &[0.0; 8], &context)?.0;
    let image_descriptor =
        TensorDescriptor::contiguous(vec![1, 2, 1, 2], DType::F32, DeviceId::CPU, context.stream)?;
    let image = backend
        .upload_f32(image_descriptor, &[1.0, -1.0, 0.0, 0.0], &context)?
        .0;

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.backend.execution_context(
        StreamId::DEFAULT,
        backend.authority.authorize_workspace(1024)?,
        &cancelled,
    );
    assert!(matches!(
        pixel_shuffle_nd_tensor_with_context_exact_native(
            &backend,
            &input,
            3,
            2,
            &cancelled_context,
        ),
        Err(SpatialFunctionalKernelError::Cancelled)
    ));
    assert!(matches!(
        bislerp_tensor_with_context_exact_native(&backend, &image, 3, 1, &cancelled_context),
        Err(SpatialFunctionalKernelError::Cancelled)
    ));

    let constrained_context = backend.backend.execution_context(
        StreamId::DEFAULT,
        backend.authority.authorize_workspace(8)?,
        &cancellation,
    );
    for error in [
        pixel_shuffle_nd_tensor_with_context_exact_native(
            &backend,
            &input,
            3,
            2,
            &constrained_context,
        )
        .expect_err("pixel shuffle must preserve a typed resource failure"),
        bislerp_tensor_with_context_exact_native(&backend, &image, 3, 1, &constrained_context)
            .expect_err("bislerp must preserve a typed resource failure"),
    ] {
        assert!(
            matches!(error, SpatialFunctionalKernelError::Tensor(_)),
            "typed tensor failure was stringified: {error:?}"
        );
    }
    assert_eq!(constrained_context.scratch.in_use_bytes(), 0);
    Ok(())
}

impl TestBackend {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let limit = 16 * 1024 * 1024;
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
        Ok(self.backend.execution_context(
            StreamId::DEFAULT,
            self.authority.authorize_workspace(self.limit)?,
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

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[test]
fn tensor_convolution_adapters_are_bounded_fresh_and_transpose_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let upload = |shape: Vec<u64>, values: &[f32]| {
        let descriptor =
            TensorDescriptor::contiguous(shape, DType::F32, DeviceId::CPU, context.stream)?;
        Ok::<_, Box<dyn std::error::Error>>(backend.upload_f32(descriptor, values, &context)?.0)
    };
    let input = upload(
        vec![1, 1, 3, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )?;
    let weight = upload(vec![1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0])?;
    let bias = upload(vec![1], &[0.5])?;
    let configuration = ConvolutionConfiguration {
        stride: vec![1, 1],
        padding: vec![0, 0],
        dilation: vec![1, 1],
        groups: 1,
        output_padding: vec![0, 0],
    };
    let output = conv_2d_tensor_with_context_exact_native(
        &*backend,
        &input,
        &weight,
        Some(&bias),
        &configuration,
        &context,
    )?;
    close(
        &tensor_to_f32_with_context_exact_native(&backend, &output, &context)?,
        &[37.5, 47.5, 67.5, 77.5],
        0.0,
    );
    assert_ne!(output.storage_id(), input.storage_id());

    let volume = upload(
        vec![1, 1, 2, 2, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )?;
    let volume_weight = upload(vec![1, 1, 2, 2, 2], &[1.0; 8])?;
    let volume_bias = upload(vec![1], &[0.25])?;
    let volume_configuration = ConvolutionConfiguration {
        stride: vec![1, 1, 1],
        padding: vec![0, 0, 0],
        dilation: vec![1, 1, 1],
        groups: 1,
        output_padding: vec![0, 0, 0],
    };
    let volume_output = conv_3d_tensor_with_context_exact_native(
        &*backend,
        &volume,
        &volume_weight,
        Some(&volume_bias),
        &volume_configuration,
        &context,
    )?;
    assert_eq!(volume_output.descriptor().shape(), &[1, 1, 1, 1, 1]);
    close(
        &tensor_to_f32_with_context_exact_native(&backend, &volume_output, &context)?,
        &[36.25],
        0.0,
    );
    assert_ne!(volume_output.storage_id(), volume.storage_id());

    let transpose_input = upload(vec![1, 1, 1, 1], &[2.0])?;
    let transpose_weight = upload(vec![1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0])?;
    let transpose = conv_transpose_2d_tensor_with_context_exact_native(
        &*backend,
        &transpose_input,
        &transpose_weight,
        None,
        &configuration,
        &context,
    )?;
    close(
        &tensor_to_f32_with_context_exact_native(&backend, &transpose, &context)?,
        &[2.0, 4.0, 6.0, 8.0],
        0.0,
    );
    assert_eq!(context.scratch.in_use_bytes(), 0);
    Ok(())
}

#[test]
fn average_pool_dimensions_divisors_and_derivatives_are_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let one = AveragePoolConfiguration {
        kernel_size: vec![2],
        stride: Some(vec![1]),
        padding: vec![0],
        ceil_mode: false,
        count_include_pad: true,
        divisor_override: None,
    };
    let output = average_pool_1d_with_context_exact_native(
        &[1.0, 3.0, 5.0],
        &[1, 1, 3],
        &one,
        DeviceId::CPU,
        &context,
    )?;
    close(&output.values, &[2.0, 4.0], 0.0);
    let vjp = average_pool_vjp_with_context_exact_native(
        AVG_POOL_1D_OPERATION_ID,
        1,
        &[1.0, 3.0, 5.0],
        &[1, 1, 3],
        &one,
        &[1.0, 1.0],
        DeviceId::CPU,
        &context,
    )?;
    close(&vjp.input, &[0.5, 1.0, 0.5], 0.0);
    let jvp = average_pool_jvp_with_context_exact_native(
        AVG_POOL_1D_OPERATION_ID,
        1,
        &[1.0, 1.0, 1.0],
        &[1, 1, 3],
        &one,
        DeviceId::CPU,
        &context,
    )?;
    close(&jvp.values, &[1.0, 1.0], 0.0);

    let two = AveragePoolConfiguration {
        kernel_size: vec![2, 2],
        stride: Some(vec![2, 2]),
        padding: vec![1, 1],
        ceil_mode: true,
        count_include_pad: false,
        divisor_override: None,
    };
    let output = average_pool_2d_with_context_exact_native(
        &[1.0, 2.0, 3.0, 4.0],
        &[1, 1, 2, 2],
        &two,
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(output.shape, [1, 1, 2, 2]);
    close(&output.values, &[1.0, 2.0, 3.0, 4.0], 0.0);

    let three = AveragePoolConfiguration {
        kernel_size: vec![1, 1, 2],
        stride: None,
        padding: vec![0, 0, 0],
        ceil_mode: false,
        count_include_pad: true,
        divisor_override: Some(4),
    };
    let output = average_pool_3d_with_context_exact_native(
        &[2.0, 6.0],
        &[1, 1, 1, 1, 2],
        &three,
        DeviceId::CPU,
        &context,
    )?;
    close(&output.values, &[2.0], 0.0);
    Ok(())
}

#[test]
fn average_pool_tensor_is_bounded_dtype_preserving_and_failure_atomic()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let configuration = AveragePoolConfiguration {
        kernel_size: vec![2, 2],
        stride: Some(vec![2, 2]),
        padding: vec![0, 0],
        ceil_mode: false,
        count_include_pad: true,
        divisor_override: None,
    };
    let input_values = (1..=16).map(|value| value as f64).collect::<Vec<_>>();
    let expected = [3.5, 5.5, 11.5, 13.5];

    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        let bytes = input_values
            .iter()
            .map(|value| {
                dtype.encode_scalar(
                    Scalar::Float(*value),
                    "average-pool-tensor-test",
                    DeviceId::CPU,
                )
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let descriptor =
            TensorDescriptor::contiguous(vec![1, 1, 4, 4], dtype, DeviceId::CPU, context.stream)?;
        let input = backend.upload_bytes(descriptor, &bytes, &context)?.0;
        let input_bytes = input.contiguous_bytes()?.to_vec();
        let output = average_pool_2d_tensor_with_context_exact_native(
            &backend,
            &input,
            &configuration,
            &context,
        )?;
        assert_eq!(output.descriptor().shape(), [1, 1, 2, 2]);
        assert_eq!(output.descriptor().dtype(), dtype);
        assert_eq!(output.descriptor().device(), DeviceId::CPU);
        assert_eq!(output.descriptor().stream(), StreamId::DEFAULT);
        assert_ne!(output.storage_id(), input.storage_id());
        close(
            &tensor_to_f32_with_context_exact_native(&backend, &output, &context)?,
            &expected,
            0.0,
        );
        assert_eq!(input.contiguous_bytes()?, input_bytes);
        assert_eq!(context.scratch.in_use_bytes(), 0);
    }

    let descriptor =
        TensorDescriptor::contiguous(vec![1, 1, 4, 4], DType::F32, DeviceId::CPU, context.stream)?;
    let input = backend
        .upload_f32(
            descriptor,
            &input_values
                .iter()
                .map(|value| *value as f32)
                .collect::<Vec<_>>(),
            &context,
        )?
        .0;
    let before_failure = backend.memory_snapshot();
    let limited_context = backend.backend.execution_context(
        StreamId::DEFAULT,
        backend.authority.authorize_workspace(79)?,
        &cancellation,
    );
    assert!(matches!(
        average_pool_2d_tensor_with_context_exact_native(
            &backend,
            &input,
            &configuration,
            &limited_context,
        ),
        Err(SpatialFunctionalKernelError::Tensor(
            sim_tensor::TensorError::WorkspaceAuthorizationExceeded { .. }
        ))
    ));
    assert_eq!(limited_context.scratch.in_use_bytes(), 0);
    assert_eq!(
        backend.memory_snapshot().current_bytes,
        before_failure.current_bytes
    );

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.backend.execution_context(
        StreamId::DEFAULT,
        backend.authority.authorize_workspace(backend.limit)?,
        &cancelled,
    );
    assert!(matches!(
        average_pool_2d_tensor_with_context_exact_native(
            &backend,
            &input,
            &configuration,
            &cancelled_context,
        ),
        Err(SpatialFunctionalKernelError::Cancelled)
    ));
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    assert_eq!(
        backend.memory_snapshot().current_bytes,
        before_failure.current_bytes
    );
    Ok(())
}

#[test]
fn all_convolution_dimensions_delegate_one_geometry_owner() -> Result<(), Box<dyn std::error::Error>>
{
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let configuration = |dimensions: usize, transposed: bool| ConvolutionConfiguration {
        stride: vec![1; dimensions],
        padding: vec![0; dimensions],
        dilation: vec![1; dimensions],
        groups: 1,
        output_padding: vec![if transposed { 0 } else { 0 }; dimensions],
    };
    let one = conv_1d_with_context_exact_native(
        &[1.0, 2.0, 3.0],
        &[1, 1, 3],
        &[2.0, 1.0],
        &[1, 1, 2],
        Some(&[0.5]),
        &configuration(1, false),
        DeviceId::CPU,
        &context,
    )?;
    close(&one.values, &[4.5, 7.5], 0.0);
    let two = conv_2d_with_context_exact_native(
        &[1.0, 2.0, 3.0, 4.0],
        &[1, 1, 2, 2],
        &[1.0],
        &[1, 1, 1, 1],
        None,
        &configuration(2, false),
        DeviceId::CPU,
        &context,
    )?;
    close(&two.values, &[1.0, 2.0, 3.0, 4.0], 0.0);
    let three = conv_3d_with_context_exact_native(
        &[1.0, 2.0],
        &[1, 1, 1, 1, 2],
        &[2.0],
        &[1, 1, 1, 1, 1],
        None,
        &configuration(3, false),
        DeviceId::CPU,
        &context,
    )?;
    close(&three.values, &[2.0, 4.0], 0.0);
    let transpose_one = conv_transpose_1d_with_context_exact_native(
        &[1.0, 2.0],
        &[1, 1, 2],
        &[1.0, 2.0],
        &[1, 1, 2],
        None,
        &configuration(1, true),
        DeviceId::CPU,
        &context,
    )?;
    close(&transpose_one.values, &[1.0, 4.0, 4.0], 0.0);
    let transpose_two = conv_transpose_2d_with_context_exact_native(
        &[3.0],
        &[1, 1, 1, 1],
        &[2.0],
        &[1, 1, 1, 1],
        None,
        &configuration(2, true),
        DeviceId::CPU,
        &context,
    )?;
    close(&transpose_two.values, &[6.0], 0.0);
    let transpose_three = conv_transpose_3d_with_context_exact_native(
        &[4.0],
        &[1, 1, 1, 1, 1],
        &[3.0],
        &[1, 1, 1, 1, 1],
        None,
        &configuration(3, true),
        DeviceId::CPU,
        &context,
    )?;
    close(&transpose_three.values, &[12.0], 0.0);

    let vjp = convolution_vjp_with_context_exact_native(
        sim_tensor::generated_spatial_functional_kernel_01::CONV_1D_OPERATION_ID,
        1,
        false,
        &[1.0, 2.0, 3.0],
        &[1, 1, 3],
        &[2.0, 1.0],
        &[1, 1, 2],
        None,
        &[1.0, 1.0],
        &configuration(1, false),
        DeviceId::CPU,
        &context,
    )?;
    close(&vjp.input, &[2.0, 3.0, 1.0], 0.0);
    let jvp = convolution_jvp_with_context_exact_native(
        sim_tensor::generated_spatial_functional_kernel_01::CONV_1D_OPERATION_ID,
        1,
        false,
        &[1.0, 2.0, 3.0],
        &[1.0; 3],
        &[1, 1, 3],
        &[2.0, 1.0],
        &[0.0; 2],
        &[1, 1, 2],
        None,
        None,
        &configuration(1, false),
        DeviceId::CPU,
        &context,
    )?;
    close(&jvp.values, &[3.0, 3.0], 0.0);
    Ok(())
}

#[test]
fn grid_sample_modes_boundaries_and_derivatives_are_native()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = [1.0, 2.0, 3.0, 4.0];
    let center_grid = [0.0, 0.0];
    let bilinear = GridSampleConfiguration {
        mode: GridSampleMode::Bilinear,
        padding_mode: GridPaddingMode::Zeros,
        align_corners: true,
    };
    let output = grid_sample_with_context_exact_native(
        &input,
        &[1, 1, 2, 2],
        &center_grid,
        &[1, 1, 1, 2],
        bilinear,
        DeviceId::CPU,
        &context,
    )?;
    close(&output.values, &[2.5], 0.0);
    let vjp = grid_sample_vjp_with_context_exact_native(
        &input,
        &[1, 1, 2, 2],
        &center_grid,
        &[1, 1, 1, 2],
        bilinear,
        &[1.0],
        DeviceId::CPU,
        &context,
    )?;
    close(&vjp.input, &[0.25; 4], 0.0);
    close(&vjp.grid, &[0.5, 1.0], 1.0e-6);
    let jvp = grid_sample_jvp_with_context_exact_native(
        &input,
        &[1.0; 4],
        &[1, 1, 2, 2],
        &center_grid,
        &[1.0, 0.0],
        &[1, 1, 1, 2],
        bilinear,
        DeviceId::CPU,
        &context,
    )?;
    close(&jvp.values, &[1.5], 1.0e-6);

    for (mode, padding, expected) in [
        (GridSampleMode::Nearest, GridPaddingMode::Border, 2.0),
        (GridSampleMode::Bilinear, GridPaddingMode::Reflection, 3.0),
        (GridSampleMode::Bicubic, GridPaddingMode::Border, 2.0),
    ] {
        let grid = [3.0, -3.0];
        let output = grid_sample_with_context_exact_native(
            &input,
            &[1, 1, 2, 2],
            &grid,
            &[1, 1, 1, 2],
            GridSampleConfiguration {
                mode,
                padding_mode: padding,
                align_corners: true,
            },
            DeviceId::CPU,
            &context,
        )?;
        close(&output.values, &[expected], 1.0e-5);
    }
    Ok(())
}

#[test]
fn grid_sample_tensor_is_bounded_dtype_preserving_and_failure_atomic()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let upload = |shape: Vec<u64>, dtype: DType, values: &[f64]| {
        let bytes = values
            .iter()
            .map(|value| {
                dtype.encode_scalar(
                    Scalar::Float(*value),
                    "grid-sample-tensor-test",
                    DeviceId::CPU,
                )
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let descriptor = TensorDescriptor::contiguous(shape, dtype, DeviceId::CPU, context.stream)?;
        Ok::<_, Box<dyn std::error::Error>>(backend.upload_bytes(descriptor, &bytes, &context)?.0)
    };
    let grid = upload(vec![1, 1, 1, 2], DType::F32, &[0.0, 0.0])?;
    let configuration = GridSampleConfiguration {
        mode: GridSampleMode::Bilinear,
        padding_mode: GridPaddingMode::Border,
        align_corners: true,
    };

    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        let input = upload(vec![1, 1, 2, 2], dtype, &[1.0, 2.0, 3.0, 4.0])?;
        let input_bytes = input.contiguous_bytes()?.to_vec();
        let output = grid_sample_tensor_with_context_exact_native(
            &backend,
            &input,
            &grid,
            configuration,
            &context,
        )?;
        assert_eq!(output.descriptor().shape(), [1, 1, 1, 1]);
        assert_eq!(output.descriptor().dtype(), dtype);
        assert_eq!(output.descriptor().device(), DeviceId::CPU);
        assert_eq!(output.descriptor().stream(), StreamId::DEFAULT);
        assert_ne!(output.storage_id(), input.storage_id());
        let values = tensor_to_f32_with_context_exact_native(&backend, &output, &context)?;
        close(&values, &[2.5], 0.0);
        assert_eq!(input.contiguous_bytes()?, input_bytes);
        assert_eq!(context.scratch.in_use_bytes(), 0);
    }

    let input = upload(vec![1, 1, 2, 2], DType::F32, &[1.0, 2.0, 3.0, 4.0])?;
    let half_grid = upload(vec![1, 1, 1, 2], DType::F16, &[0.0, 0.0])?;
    assert!(matches!(
        grid_sample_tensor_with_context_exact_native(
            &backend,
            &input,
            &half_grid,
            configuration,
            &context,
        ),
        Err(SpatialFunctionalKernelError::Tensor(
            sim_tensor::TensorError::DTypeMismatch {
                expected: DType::F32,
                actual: DType::F16,
            }
        ))
    ));
    let before_failure = backend.memory_snapshot();
    let limited_context = backend.backend.execution_context(
        StreamId::DEFAULT,
        backend.authority.authorize_workspace(23)?,
        &cancellation,
    );
    let workspace_error = grid_sample_tensor_with_context_exact_native(
        &backend,
        &input,
        &grid,
        configuration,
        &limited_context,
    );
    assert!(matches!(
        workspace_error,
        Err(SpatialFunctionalKernelError::Tensor(
            sim_tensor::TensorError::WorkspaceAuthorizationExceeded { .. }
        ))
    ));
    assert_eq!(limited_context.scratch.in_use_bytes(), 0);
    assert_eq!(
        backend.memory_snapshot().current_bytes,
        before_failure.current_bytes
    );

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.backend.execution_context(
        StreamId::DEFAULT,
        backend.authority.authorize_workspace(backend.limit)?,
        &cancelled,
    );
    assert!(matches!(
        grid_sample_tensor_with_context_exact_native(
            &backend,
            &input,
            &grid,
            configuration,
            &cancelled_context,
        ),
        Err(SpatialFunctionalKernelError::Cancelled)
    ));
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    assert_eq!(
        backend.memory_snapshot().current_bytes,
        before_failure.current_bytes
    );
    Ok(())
}

#[test]
fn grid_sample_analytic_maps_match_finite_differences_and_adjoint_identity()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = [1.0, -0.5, 2.0, 4.0, -2.0, 3.0, 0.25, 1.5];
    let input_tangent = [0.2, -0.4, 0.7, 0.1, 0.5, 0.3, -0.2, 0.8];
    let grid = [0.2, -0.3];
    let grid_tangent = [0.35, -0.25];
    let output_cotangent = [0.7, -1.1];
    for mode in [GridSampleMode::Bilinear, GridSampleMode::Bicubic] {
        for padding_mode in [
            GridPaddingMode::Zeros,
            GridPaddingMode::Border,
            GridPaddingMode::Reflection,
        ] {
            let configuration = GridSampleConfiguration {
                mode,
                padding_mode,
                align_corners: false,
            };
            let jvp = grid_sample_jvp_with_context_exact_native(
                &input,
                &input_tangent,
                &[1, 2, 2, 2],
                &grid,
                &grid_tangent,
                &[1, 1, 1, 2],
                configuration,
                DeviceId::CPU,
                &context,
            )?;
            let vjp = grid_sample_vjp_with_context_exact_native(
                &input,
                &[1, 2, 2, 2],
                &grid,
                &[1, 1, 1, 2],
                configuration,
                &output_cotangent,
                DeviceId::CPU,
                &context,
            )?;
            let adjoint_left = dot(&jvp.values, &output_cotangent);
            let adjoint_right = dot(&input_tangent, &vjp.input) + dot(&grid_tangent, &vjp.grid);
            close(&[adjoint_left], &[adjoint_right], 2.0e-5);

            let epsilon = 1.0e-3_f32;
            let perturbed_input = input
                .iter()
                .zip(input_tangent)
                .map(|(value, tangent)| value + epsilon * tangent)
                .collect::<Vec<_>>();
            let perturbed_grid = grid
                .iter()
                .zip(grid_tangent)
                .map(|(value, tangent)| value + epsilon * tangent)
                .collect::<Vec<_>>();
            let base = grid_sample_with_context_exact_native(
                &input,
                &[1, 2, 2, 2],
                &grid,
                &[1, 1, 1, 2],
                configuration,
                DeviceId::CPU,
                &context,
            )?;
            let perturbed = grid_sample_with_context_exact_native(
                &perturbed_input,
                &[1, 2, 2, 2],
                &perturbed_grid,
                &[1, 1, 1, 2],
                configuration,
                DeviceId::CPU,
                &context,
            )?;
            let finite_difference = perturbed
                .values
                .iter()
                .zip(&base.values)
                .map(|(perturbed, base)| (perturbed - base) / epsilon)
                .collect::<Vec<_>>();
            close(&jvp.values, &finite_difference, 3.0e-3);
        }
    }
    Ok(())
}

#[test]
fn interpolate_tensor_is_bounded_dtype_preserving_and_failure_atomic()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let upload = |dtype: DType| {
        let bytes = [1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(|value| {
                dtype.encode_scalar(
                    Scalar::Float(value),
                    "interpolate-tensor-test",
                    DeviceId::CPU,
                )
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let descriptor =
            TensorDescriptor::contiguous(vec![1, 1, 2, 2], dtype, DeviceId::CPU, context.stream)?;
        Ok::<_, Box<dyn std::error::Error>>(backend.upload_bytes(descriptor, &bytes, &context)?.0)
    };
    let configuration = InterpolateConfiguration {
        output_size: Some(vec![3, 3]),
        scale_factor: None,
        mode: InterpolateMode::Bilinear,
        align_corners: Some(false),
        recompute_scale_factor: None,
        antialias: false,
    };
    let expected = [1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0];

    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        let input = upload(dtype)?;
        let input_bytes = input.contiguous_bytes()?.to_vec();
        let output = interpolate_tensor_with_context_exact_native(
            &backend,
            &input,
            &configuration,
            &context,
        )?;
        assert_eq!(output.descriptor().shape(), [1, 1, 3, 3]);
        assert_eq!(output.descriptor().dtype(), dtype);
        assert_eq!(output.descriptor().device(), DeviceId::CPU);
        assert_eq!(output.descriptor().stream(), StreamId::DEFAULT);
        assert_ne!(output.storage_id(), input.storage_id());
        let values = tensor_to_f32_with_context_exact_native(&backend, &output, &context)?;
        close(&values, &expected, 0.0);
        assert_eq!(input.contiguous_bytes()?, input_bytes);
        assert_eq!(context.scratch.in_use_bytes(), 0);
    }

    let input = upload(DType::F32)?;
    let before_failure = backend.memory_snapshot();
    let limited_context = backend.backend.execution_context(
        StreamId::DEFAULT,
        backend.authority.authorize_workspace(51)?,
        &cancellation,
    );
    assert!(matches!(
        interpolate_tensor_with_context_exact_native(
            &backend,
            &input,
            &configuration,
            &limited_context,
        ),
        Err(SpatialFunctionalKernelError::Tensor(
            sim_tensor::TensorError::WorkspaceAuthorizationExceeded { .. }
        ))
    ));
    assert_eq!(limited_context.scratch.in_use_bytes(), 0);
    assert_eq!(
        backend.memory_snapshot().current_bytes,
        before_failure.current_bytes
    );

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.backend.execution_context(
        StreamId::DEFAULT,
        backend.authority.authorize_workspace(backend.limit)?,
        &cancelled,
    );
    assert!(matches!(
        interpolate_tensor_with_context_exact_native(
            &backend,
            &input,
            &configuration,
            &cancelled_context,
        ),
        Err(SpatialFunctionalKernelError::Cancelled)
    ));
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    assert_eq!(
        backend.memory_snapshot().current_bytes,
        before_failure.current_bytes
    );
    Ok(())
}

#[test]
fn interpolate_supports_all_spatial_ranks_modes_and_analytic_maps()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let linear = InterpolateConfiguration {
        output_size: Some(vec![3]),
        scale_factor: None,
        mode: InterpolateMode::Linear,
        align_corners: Some(true),
        recompute_scale_factor: None,
        antialias: false,
    };
    let output = interpolate_with_context_exact_native(
        &[1.0, 3.0],
        &[1, 1, 2],
        &linear,
        DeviceId::CPU,
        &context,
    )?;
    close(&output.values, &[1.0, 2.0, 3.0], 0.0);
    let vjp = interpolate_vjp_with_context_exact_native(
        &[1.0, 3.0],
        &[1, 1, 2],
        &linear,
        &[1.0; 3],
        DeviceId::CPU,
        &context,
    )?;
    close(&vjp, &[1.5, 1.5], 0.0);
    let jvp = interpolate_jvp_with_context_exact_native(
        &[2.0, 4.0],
        &[1, 1, 2],
        &linear,
        DeviceId::CPU,
        &context,
    )?;
    close(&jvp.values, &[2.0, 3.0, 4.0], 0.0);

    for (mode, output_size) in [
        (InterpolateMode::Nearest, vec![3, 3]),
        (InterpolateMode::NearestExact, vec![3, 3]),
        (InterpolateMode::Bilinear, vec![3, 3]),
        (InterpolateMode::Bicubic, vec![3, 3]),
        (InterpolateMode::Area, vec![1, 1]),
    ] {
        let configuration = InterpolateConfiguration {
            output_size: Some(output_size),
            scale_factor: None,
            mode,
            align_corners: if matches!(mode, InterpolateMode::Bilinear | InterpolateMode::Bicubic) {
                Some(false)
            } else {
                None
            },
            recompute_scale_factor: None,
            antialias: matches!(mode, InterpolateMode::Bilinear | InterpolateMode::Bicubic),
        };
        let output = interpolate_with_context_exact_native(
            &[1.0, 2.0, 3.0, 4.0],
            &[1, 1, 2, 2],
            &configuration,
            DeviceId::CPU,
            &context,
        )?;
        assert_eq!(
            output.shape[2..],
            configuration.output_size.as_deref().unwrap_or_default()[..]
        );
        assert!(output.values.iter().all(|value| value.is_finite()));
    }

    let trilinear = InterpolateConfiguration {
        output_size: Some(vec![2, 2, 3]),
        scale_factor: None,
        mode: InterpolateMode::Trilinear,
        align_corners: Some(true),
        recompute_scale_factor: None,
        antialias: false,
    };
    let output = interpolate_with_context_exact_native(
        &[1.0, 3.0],
        &[1, 1, 1, 1, 2],
        &trilinear,
        DeviceId::CPU,
        &context,
    )?;
    close(
        &output.values,
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        0.0,
    );
    Ok(())
}

#[test]
fn interpolate_maps_preserve_constants_and_satisfy_the_adjoint_identity()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let cases = [
        (
            vec![1, 1, 3],
            InterpolateConfiguration {
                output_size: Some(vec![5]),
                scale_factor: None,
                mode: InterpolateMode::Linear,
                align_corners: Some(false),
                recompute_scale_factor: None,
                antialias: false,
            },
        ),
        (
            vec![1, 1, 3, 3],
            InterpolateConfiguration {
                output_size: Some(vec![2, 2]),
                scale_factor: None,
                mode: InterpolateMode::Bicubic,
                align_corners: Some(false),
                recompute_scale_factor: None,
                antialias: true,
            },
        ),
        (
            vec![1, 1, 2, 2, 2],
            InterpolateConfiguration {
                output_size: Some(vec![3, 3, 3]),
                scale_factor: None,
                mode: InterpolateMode::Trilinear,
                align_corners: Some(true),
                recompute_scale_factor: None,
                antialias: false,
            },
        ),
    ];
    for (input_shape, configuration) in cases {
        let input_count = input_shape.iter().product::<usize>();
        let input = (0..input_count)
            .map(|index| index as f32 * 0.25 - 0.5)
            .collect::<Vec<_>>();
        let input_tangent = (0..input_count)
            .map(|index| (index as f32 + 1.0) * -0.1)
            .collect::<Vec<_>>();
        let constant = vec![2.25; input_count];
        let constant_output = interpolate_with_context_exact_native(
            &constant,
            &input_shape,
            &configuration,
            DeviceId::CPU,
            &context,
        )?;
        close(
            &constant_output.values,
            &vec![2.25; constant_output.values.len()],
            2.0e-5,
        );
        let jvp = interpolate_jvp_with_context_exact_native(
            &input_tangent,
            &input_shape,
            &configuration,
            DeviceId::CPU,
            &context,
        )?;
        let output_cotangent = (0..jvp.values.len())
            .map(|index| (index as f32 + 0.5) * 0.03)
            .collect::<Vec<_>>();
        let vjp = interpolate_vjp_with_context_exact_native(
            &input,
            &input_shape,
            &configuration,
            &output_cotangent,
            DeviceId::CPU,
            &context,
        )?;
        close(
            &[dot(&jvp.values, &output_cotangent)],
            &[dot(&input_tangent, &vjp)],
            2.0e-4,
        );
    }
    Ok(())
}

#[test]
fn max_pool_adapter_preserves_selection_and_derivatives() -> Result<(), Box<dyn std::error::Error>>
{
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let input = [1.0, 4.0, 3.0, 2.0];
    let output = max_pool_2d_with_context_exact_native(
        &input,
        &[1, 1, 2, 2],
        [2, 2],
        None,
        [0, 0],
        [1, 1],
        false,
        DeviceId::CPU,
        &context,
    )?;
    close(&output.values, &[4.0], 0.0);
    let vjp = max_pool_2d_vjp_with_context_exact_native(
        &input,
        &[1, 1, 2, 2],
        [2, 2],
        None,
        [0, 0],
        [1, 1],
        false,
        &[2.0],
        DeviceId::CPU,
        &context,
    )?;
    close(&vjp.input, &[0.0, 2.0, 0.0, 0.0], 0.0);
    let jvp = max_pool_2d_jvp_with_context_exact_native(
        &input,
        &[10.0, 20.0, 30.0, 40.0],
        &[1, 1, 2, 2],
        [2, 2],
        None,
        [0, 0],
        [1, 1],
        false,
        DeviceId::CPU,
        &context,
    )?;
    close(&jvp.values, &[20.0], 0.0);
    Ok(())
}

#[test]
fn cancellation_and_invalid_geometries_fail_before_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let context = backend.execution(&cancellation)?;
    let error = grid_sample_with_context_exact_native(
        &[1.0],
        &[1, 1, 1, 1],
        &[0.0, 0.0],
        &[1, 1, 1, 2],
        GridSampleConfiguration {
            mode: GridSampleMode::Bilinear,
            padding_mode: GridPaddingMode::Zeros,
            align_corners: false,
        },
        DeviceId::CPU,
        &context,
    )
    .expect_err("cancelled execution must fail");
    assert!(matches!(error, SpatialFunctionalKernelError::Cancelled));

    let active = CancellationToken::default();
    let context = backend.execution(&active)?;
    let error = interpolate_with_context_exact_native(
        &[1.0],
        &[1, 1, 1],
        &InterpolateConfiguration {
            output_size: Some(vec![2]),
            scale_factor: Some(vec![2.0]),
            mode: InterpolateMode::Nearest,
            align_corners: None,
            recompute_scale_factor: None,
            antialias: false,
        },
        DeviceId::CPU,
        &context,
    )
    .expect_err("ambiguous interpolation geometry must fail");
    assert!(matches!(
        error,
        SpatialFunctionalKernelError::Invalid {
            operation: INTERPOLATE_OPERATION_ID,
            ..
        }
    ));
    let error = interpolate_with_context_exact_native(
        &[1.0],
        &[1, 1, 1],
        &InterpolateConfiguration {
            output_size: Some(vec![2]),
            scale_factor: None,
            mode: InterpolateMode::Nearest,
            align_corners: None,
            recompute_scale_factor: Some(true),
            antialias: false,
        },
        DeviceId::CPU,
        &context,
    )
    .expect_err("recomputation with an explicit output size must fail");
    assert!(matches!(
        error,
        SpatialFunctionalKernelError::Invalid { .. }
    ));
    let error = grid_sample_with_context_exact_native(
        &[1.0, 2.0, 3.0, 4.0],
        &[1, 1, 2, 2],
        &[f32::MAX, f32::MAX],
        &[1, 1, 1, 2],
        GridSampleConfiguration {
            mode: GridSampleMode::Bicubic,
            padding_mode: GridPaddingMode::Border,
            align_corners: true,
        },
        DeviceId::CPU,
        &context,
    )
    .expect_err("unrepresentable bicubic coordinates must fail without panicking");
    assert!(matches!(
        error,
        SpatialFunctionalKernelError::ShapeOverflow {
            operation: GRID_SAMPLE_OPERATION_ID,
            ..
        }
    ));
    assert_eq!(GRID_SAMPLE_OPERATION_ID, "COMFY-TENSOR-OP-A90AB43A3320");
    Ok(())
}

#[test]
fn all_twelve_contracts_are_build_sealed_against_runtime_fixtures()
-> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixtures = root
        .join("crates/sim_test_support/fixtures/tensor_operations/spatial_functional_kernel_01");
    let mut runtime_digests = Vec::new();
    for entry in fs::read_dir(fixtures)? {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("json") {
            runtime_digests.push(format!("{:x}", Sha256::digest(fs::read(path)?)));
        }
    }
    assert_eq!(runtime_digests.len(), 12);
    assert_eq!(
        sim_tensor::GENERATED_BUILD_SEALED_OPERATION_RESOLUTIONS
            .iter()
            .filter(|(_, _, module, digest)| {
                *module == "spatial_functional_kernel_01"
                    && runtime_digests.iter().any(|runtime| runtime == digest)
            })
            .count(),
        12
    );
    let contracts = sim_tensor::GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .filter(|slice| slice.module_name == "spatial_functional_kernel_01")
        .flat_map(|slice| slice.contracts)
        .collect::<Vec<_>>();
    assert_eq!(contracts.len(), 12);
    let operation_ids = contracts
        .iter()
        .map(|contract| contract.operation_id)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(operation_ids.len(), 12);
    Ok(())
}
