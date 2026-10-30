use sim_tensor::{
    CancellationToken, CpuWorkspaceAuthority, DType, DeviceId,
    GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, OperationContractId, Scalar, StreamId, Tensor,
    TensorDescriptor,
    generated_activation_normalization_functional_01::{
        BATCH_NORM_OPERATION_ID, BatchNormTensorDirection, FunctionalError, GELU_OPERATION_ID,
        GROUP_NORM_OPERATION_ID, GeluApproximation, LAYER_NORM_OPERATION_ID,
        LEAKY_RELU_OPERATION_ID, NORMALIZE_OPERATION_ID, RELU_OPERATION_ID, RMS_NORM_OPERATION_ID,
        SILU_OPERATION_ID, SOFTMAX_OPERATION_ID, batch_norm_jvp_with_context_exact_native,
        batch_norm_tensor_with_context_exact_native, batch_norm_vjp_with_context_exact_native,
        batch_norm_with_context_exact_native, channel_layer_norm_tensor_with_context_exact_native,
        channel_standardize_tensor_with_context_exact_native, gelu_jvp_with_context_exact_native,
        gelu_scalar_exact_native, gelu_vjp_with_context_exact_native,
        gelu_with_context_exact_native, group_norm_jvp_with_context_exact_native,
        group_norm_tensor_with_context_exact_native, group_norm_vjp_with_context_exact_native,
        group_norm_with_context_exact_native, layer_norm_jvp_with_context_exact_native,
        layer_norm_vjp_with_context_exact_native, layer_norm_with_context_exact_native,
        leaky_relu_jvp_with_context_exact_native, leaky_relu_tensor_with_context_exact_native,
        leaky_relu_vjp_with_context_exact_native, leaky_relu_with_context_exact_native,
        leaky_relu_with_context_exact_native_in_place, normalize_jvp_with_context_exact_native,
        normalize_vjp_with_context_exact_native, normalize_with_context_exact_native,
        relu_jvp_with_context_exact_native, relu_vjp_with_context_exact_native,
        relu_with_context_exact_native, relu_with_context_exact_native_in_place,
        rms_norm_jvp_with_context_exact_native, rms_norm_vjp_with_context_exact_native,
        rms_norm_with_context_exact_native, silu_jvp_with_context_exact_native,
        silu_tensor_with_context_exact_native, silu_vjp_with_context_exact_native,
        silu_with_context_exact_native, silu_with_context_exact_native_in_place,
        softmax_jvp_with_context_exact_native, softmax_tensor_with_context_exact_native,
        softmax_vjp_with_context_exact_native, softmax_with_context_exact_native,
    },
};
use sim_types::DeviceKind;

const OPERATION_IDS: [&str; 10] = [
    BATCH_NORM_OPERATION_ID,
    GELU_OPERATION_ID,
    GROUP_NORM_OPERATION_ID,
    LAYER_NORM_OPERATION_ID,
    LEAKY_RELU_OPERATION_ID,
    NORMALIZE_OPERATION_ID,
    RELU_OPERATION_ID,
    RMS_NORM_OPERATION_ID,
    SILU_OPERATION_ID,
    SOFTMAX_OPERATION_ID,
];

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index}: expected {expected}, got {actual}"
        );
    }
}

fn upload_tensor(
    backend: &sim_tensor::CpuBackend,
    shape: &[u64],
    values: &[f32],
    context: &sim_tensor::ExecutionContext<'_>,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, context.stream)?;
    Ok(backend.upload_f32(descriptor, values, context)?.0)
}

fn tensor_f32_values(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(tensor
        .contiguous_bytes()?
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

#[test]
fn leaky_relu_tensor_preserves_low_precision_descriptor_and_storage_independence()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        let memory_before = backend.memory_snapshot().current_bytes;
        let descriptor =
            TensorDescriptor::contiguous(vec![5], dtype, DeviceId::CPU, StreamId::DEFAULT)?;
        let mut bytes = Vec::new();
        for value in [-4.0_f32, -1.0, 0.0, 1.0, 4.0] {
            bytes.extend(dtype.encode_scalar(
                Scalar::Float(f64::from(value)),
                "leaky-relu-tensor-test",
                DeviceId::CPU,
            )?);
        }
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(0)?,
            &cancellation,
        );
        let input = backend.upload_bytes(descriptor, &bytes, &context)?.0;
        let output = leaky_relu_tensor_with_context_exact_native(&backend, &input, 0.2, &context)?;
        assert_eq!(output.descriptor().dtype(), dtype);
        assert_eq!(output.descriptor().shape(), [5]);
        assert_eq!(output.descriptor().stream(), StreamId::DEFAULT);
        assert_ne!(output.storage_id(), input.storage_id());
        assert_eq!(input.contiguous_bytes()?, bytes);
        let width = usize::try_from(dtype.byte_width())?;
        let values = output
            .contiguous_bytes()?
            .chunks_exact(width)
            .map(|bytes| match dtype.decode_scalar(bytes)? {
                sim_tensor::DecodedScalar::Real(value) => Ok(value as f32),
                _ => Err::<f32, Box<dyn std::error::Error>>("expected real output".into()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (actual, expected) in values.iter().zip([-0.8, -0.2, 0.0, 1.0, 4.0]) {
            assert!((actual - expected).abs() <= 0.004);
        }
        assert_eq!(context.scratch.in_use_bytes(), 0);
        drop(output);
        drop(input);
        assert_eq!(backend.memory_snapshot().current_bytes, memory_before);
    }
    Ok(())
}

#[test]
fn tensor_adapters_preserve_canonical_normalization_equations_and_context()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        &cancellation,
    );
    let input = upload_tensor(&backend, &[1, 2, 1, 2], &[1.0, 3.0, 5.0, 7.0], &context)?;
    let weight = upload_tensor(&backend, &[2], &[2.0, 0.5], &context)?;
    let bias = upload_tensor(&backend, &[2], &[1.0, -1.0], &context)?;

    let softmax = softmax_tensor_with_context_exact_native(&backend, &input, -1, &context)?;
    let silu = silu_tensor_with_context_exact_native(&backend, &input, &context)?;
    assert_close(
        &tensor_f32_values(&silu)?,
        &[0.731_058_6, 2.857_722_3, 4.966_536, 6.993_622_6],
        1.0e-6,
    );
    assert_ne!(silu.storage_id(), input.storage_id());
    let softmax_values = tensor_f32_values(&softmax)?;
    assert_close(
        &softmax_values,
        &[0.119_202_92, 0.880_797, 0.119_202_92, 0.880_797],
        1.0e-6,
    );

    let group = group_norm_tensor_with_context_exact_native(
        &backend,
        &input,
        2,
        Some(&weight),
        Some(&bias),
        1.0e-6,
        &context,
    )?;
    assert_close(
        &tensor_f32_values(&group)?,
        &[-1.0, 3.0, -1.5, -0.5],
        2.0e-6,
    );

    let channel = channel_layer_norm_tensor_with_context_exact_native(
        &backend, &input, None, None, 1.0e-6, &context,
    )?;
    assert_close(
        &tensor_f32_values(&channel)?,
        &[-1.0, -1.0, 1.0, 1.0],
        2.0e-6,
    );

    let mean = upload_tensor(&backend, &[2], &[1.0, 5.0], &context)?;
    let variance = upload_tensor(&backend, &[2], &[4.0, 4.0], &context)?;
    let standard_deviation = upload_tensor(&backend, &[2], &[2.0, 4.0], &context)?;
    let standardized = channel_standardize_tensor_with_context_exact_native(
        &backend,
        &input,
        &mean,
        &standard_deviation,
        &context,
    )?;
    assert_close(
        &tensor_f32_values(&standardized)?,
        &[0.0, 1.0, 0.0, 0.5],
        1.0e-6,
    );
    let normalized = batch_norm_tensor_with_context_exact_native(
        &backend,
        &input,
        &mean,
        &variance,
        Some(&weight),
        Some(&bias),
        1.0e-6,
        BatchNormTensorDirection::Normalize,
        &context,
    )?;
    let restored = batch_norm_tensor_with_context_exact_native(
        &backend,
        &normalized,
        &mean,
        &variance,
        Some(&weight),
        Some(&bias),
        1.0e-6,
        BatchNormTensorDirection::Denormalize,
        &context,
    )?;
    assert_close(
        &tensor_f32_values(&restored)?,
        &[1.0, 3.0, 5.0, 7.0],
        1.0e-5,
    );
    assert_eq!(softmax.descriptor().dtype(), DType::F32);
    assert_eq!(softmax.descriptor().device(), DeviceId::CPU);
    assert_eq!(softmax.descriptor().stream(), StreamId::DEFAULT);
    let half_bytes = [0.0_f64, 1.0]
        .into_iter()
        .map(|value| DType::F16.encode_scalar(Scalar::Float(value), "softmax-test", DeviceId::CPU))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let half_descriptor =
        TensorDescriptor::contiguous(vec![1, 2], DType::F16, DeviceId::CPU, context.stream)?;
    let half = backend
        .upload_bytes(half_descriptor, &half_bytes, &context)?
        .0;
    let half_softmax = softmax_tensor_with_context_exact_native(&backend, &half, -1, &context)?;
    assert_eq!(half_softmax.descriptor().dtype(), DType::F16);
    let decoded = half_softmax
        .contiguous_bytes()?
        .chunks_exact(2)
        .map(|bytes| DType::F16.decode_scalar(bytes))
        .collect::<Result<Vec<_>, _>>()?;
    assert!(
        matches!(decoded.as_slice(), [sim_tensor::DecodedScalar::Real(left), sim_tensor::DecodedScalar::Real(right)] if (*left - 0.269).abs() < 0.002 && (*right - 0.731).abs() < 0.002)
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        &cancelled,
    );
    assert!(matches!(
        channel_standardize_tensor_with_context_exact_native(
            &backend,
            &input,
            &mean,
            &input,
            &cancelled_context,
        ),
        Err(FunctionalError::Cancelled)
    ));
    assert!(matches!(
        silu_tensor_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(FunctionalError::Cancelled)
    ));
    Ok(())
}

fn add_scaled(values: &[f32], direction: &[f32], scale: f32) -> Vec<f32> {
    values
        .iter()
        .zip(direction)
        .map(|(value, direction)| value + scale * direction)
        .collect()
}

fn finite_difference(
    values: &[f32],
    direction: &[f32],
    function: impl Fn(&[f32]) -> Vec<f32>,
) -> Vec<f32> {
    let epsilon = 0.000_5;
    let plus = function(&add_scaled(values, direction, epsilon));
    let minus = function(&add_scaled(values, direction, -epsilon));
    plus.iter()
        .zip(minus)
        .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
        .collect()
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn assert_cancelled(operation_id: &str, result: Result<Vec<f32>, FunctionalError>) {
    assert!(
        matches!(result, Err(FunctionalError::Cancelled)),
        "{operation_id} did not preserve canonical cancellation"
    );
}

fn assert_unsupported(operation_id: &str, result: Result<Vec<f32>, FunctionalError>) {
    assert!(
        matches!(result, Err(FunctionalError::UnsupportedDevice { .. })),
        "{operation_id} did not reject an uncertified device"
    );
}

#[test]
fn all_ten_functional_contracts_are_build_sealed_once() {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "activation_normalization_functional_01")
        .unwrap_or_else(|| {
            panic!("functional activation/normalization resolution slice is missing")
        });
    assert_eq!(slice.len(), OPERATION_IDS.len());
    for operation_id in OPERATION_IDS {
        assert_eq!(
            slice
                .contracts
                .iter()
                .filter(|contract| contract.operation_id == operation_id)
                .count(),
            1
        );
        assert!(
            OperationContractId::new(operation_id).is_ok(),
            "invalid operation ID {operation_id}"
        );
    }
}

#[test]
fn activations_match_exact_out_of_place_and_transactional_in_place_semantics() {
    let cancellation = CancellationToken::default();
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)
        .unwrap_or_else(|error| panic!("backend construction failed: {error}"));
    let caller_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(16)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancellation,
    );
    let input = [-1.0, 0.0, 1.0, f32::NAN];
    let relu = relu_with_context_exact_native(&backend, &input, DeviceId::CPU, &caller_context)
        .unwrap_or_else(|error| panic!("ReLU failed: {error}"));
    assert_eq!(&relu[..3], &[0.0, 0.0, 1.0]);
    assert!(relu[3].is_nan());
    assert!(input[3].is_nan());

    let leaky = leaky_relu_with_context_exact_native(
        &backend,
        &input[..3],
        0.1,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("leaky ReLU failed: {error}"));
    assert_close(&leaky, &[-0.1, 0.0, 1.0], 0.000_001);
    let silu =
        silu_with_context_exact_native(&backend, &input[..3], DeviceId::CPU, &caller_context)
            .unwrap_or_else(|error| panic!("SiLU failed: {error}"));
    assert_close(&silu, &[-0.268_941_43, 0.0, 0.731_058_6], 0.000_001);
    let gelu = gelu_with_context_exact_native(
        &backend,
        &input[..3],
        GeluApproximation::None,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("GELU failed: {error}"));
    assert_close(&gelu, &[-0.158_655_26, 0.0, 0.841_344_7], 0.000_001);
    assert_close(
        &[gelu_scalar_exact_native(-1.0, GeluApproximation::None)],
        &[gelu[0]],
        0.000_001,
    );

    let mut in_place = [-2.0, 3.0];
    relu_with_context_exact_native_in_place(
        &backend,
        &mut in_place,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("in-place ReLU failed: {error}"));
    assert_eq!(in_place, [0.0, 3.0]);
    leaky_relu_with_context_exact_native_in_place(
        &backend,
        &mut in_place,
        0.2,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("in-place leaky ReLU failed: {error}"));
    silu_with_context_exact_native_in_place(
        &backend,
        &mut in_place,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("in-place SiLU failed: {error}"));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(16)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancelled,
    );
    let before = in_place;
    assert!(matches!(
        relu_with_context_exact_native_in_place(
            &backend,
            &mut in_place,
            DeviceId::CPU,
            &cancelled_context
        ),
        Err(FunctionalError::Cancelled)
    ));
    assert_eq!(in_place, before);
    assert!(matches!(
        relu_with_context_exact_native(
            &backend,
            &[1.0],
            DeviceId::new(DeviceKind::Cuda, 0),
            &caller_context
        ),
        Err(FunctionalError::UnsupportedDevice { .. })
    ));
}

#[test]
fn softmax_and_vector_normalization_cover_axes_zero_rows_and_boundaries() {
    let cancellation = CancellationToken::default();
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)
        .unwrap_or_else(|error| panic!("backend construction failed: {error}"));
    let caller_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(16)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancellation,
    );
    let input = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0];
    let output = softmax_with_context_exact_native(
        &backend,
        &input,
        &[2, 3],
        -1,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("softmax failed: {error}"));
    assert_close(
        &output,
        &[
            0.090_030_57,
            0.244_728_48,
            0.665_240_94,
            0.665_240_94,
            0.244_728_48,
            0.090_030_57,
        ],
        0.000_001,
    );
    let vectors = [3.0, 4.0, 0.0, 0.0];
    let normalized = normalize_with_context_exact_native(
        &backend,
        &vectors,
        &[2, 2],
        2.0,
        &[-1],
        1.0e-12,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("normalize failed: {error}"));
    assert_close(&normalized, &[0.6, 0.8, 0.0, 0.0], 0.000_001);
    let volume = normalize_with_context_exact_native(
        &backend,
        &[1.0; 8],
        &[1, 2, 2, 2],
        2.0,
        &[-1, -2, -3],
        1.0e-12,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("multi-axis normalize failed: {error}"));
    assert_close(&volume, &[1.0 / 8.0_f32.sqrt(); 8], 0.000_001);
    assert!(matches!(
        softmax_with_context_exact_native(
            &backend,
            &input,
            &[2, 3],
            2,
            DeviceId::CPU,
            &caller_context
        ),
        Err(FunctionalError::InvalidDimension { .. })
    ));
    assert!(matches!(
        normalize_with_context_exact_native(
            &backend,
            &vectors,
            &[2, 2],
            2.0,
            &[-1, 1],
            1.0e-12,
            DeviceId::CPU,
            &caller_context
        ),
        Err(FunctionalError::DuplicateDimension { .. })
    ));
}

#[test]
fn layer_rms_group_and_batch_norm_match_affine_and_running_stat_rules() {
    let cancellation = CancellationToken::default();
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)
        .unwrap_or_else(|error| panic!("backend construction failed: {error}"));
    let caller_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(16)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancellation,
    );
    let input = [1.0, 3.0, 2.0, 4.0];
    let layer = layer_norm_with_context_exact_native(
        &backend,
        &input,
        &[2, 2],
        &[2],
        Some(&[2.0, 0.5]),
        Some(&[1.0, -1.0]),
        1.0e-6,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("layer norm failed: {error}"));
    assert_close(
        &layer,
        &[-0.999_999, -0.500_000_24, -0.999_999, -0.500_000_24],
        0.000_002,
    );
    let rms = rms_norm_with_context_exact_native(
        &backend,
        &input,
        &[2, 2],
        &[2],
        Some(&[1.0, 2.0]),
        Some(1.0e-6),
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("RMS norm failed: {error}"));
    assert_close(
        &rms,
        &[
            1.0 / 5.000_001_f32.sqrt(),
            6.0 / 5.000_001_f32.sqrt(),
            2.0 / 10.000_001_f32.sqrt(),
            8.0 / 10.000_001_f32.sqrt(),
        ],
        0.000_001,
    );
    let group = group_norm_with_context_exact_native(
        &backend,
        &input,
        &[1, 2, 1, 2],
        1,
        None,
        None,
        1.0e-5,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("group norm failed: {error}"));
    assert_close(
        &group,
        &[-1.341_635_5, 0.447_211_83, -0.447_211_83, 1.341_635_5],
        0.000_01,
    );

    let batch_input = [1.0, 10.0, 3.0, 14.0];
    let mut running_mean = [0.0, 0.0];
    let mut running_variance = [1.0, 1.0];
    let batch = batch_norm_with_context_exact_native(
        &backend,
        &batch_input,
        &[2, 2, 1],
        Some(&mut running_mean),
        Some(&mut running_variance),
        None,
        None,
        true,
        0.5,
        1.0e-5,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("batch norm failed: {error}"));
    assert_close(
        &batch,
        &[-0.999_995, -0.999_998_75, 0.999_995, 0.999_998_75],
        0.000_01,
    );
    assert_close(&running_mean, &[1.0, 6.0], 0.000_001);
    assert_close(&running_variance, &[1.5, 4.5], 0.000_001);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(16 * 1024 * 1024)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancelled,
    );
    let mean_before = running_mean;
    let variance_before = running_variance;
    assert!(matches!(
        batch_norm_with_context_exact_native(
            &backend,
            &batch_input,
            &[2, 2, 1],
            Some(&mut running_mean),
            Some(&mut running_variance),
            None,
            None,
            true,
            0.5,
            1.0e-5,
            DeviceId::CPU,
            &cancelled_context
        ),
        Err(FunctionalError::Cancelled)
    ));
    assert_eq!(running_mean, mean_before);
    assert_eq!(running_variance, variance_before);
}

#[test]
fn elementwise_softmax_and_normalize_gradients_are_analytical() {
    let cancellation = CancellationToken::default();
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)
        .unwrap_or_else(|error| panic!("backend construction failed: {error}"));
    let caller_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(16 * 1024 * 1024)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancellation,
    );
    let input = [-0.7, 0.2, 1.1, -1.3];
    let tangent = [0.3, -0.5, 0.2, 0.7];
    let upstream = [0.9, -0.4, 0.6, 0.1];

    let relu_jvp = relu_jvp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("ReLU JVP failed: {error}"));
    let relu_finite = finite_difference(&input, &tangent, |values| {
        relu_with_context_exact_native(&backend, values, DeviceId::CPU, &caller_context)
            .unwrap_or_else(|error| panic!("ReLU finite forward failed: {error}"))
    });
    assert_close(&relu_jvp, &relu_finite, 0.000_3);
    let relu_vjp = relu_vjp_with_context_exact_native(
        &backend,
        &input,
        &upstream,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("ReLU VJP failed: {error}"));
    assert!((dot(&relu_jvp, &upstream) - dot(&tangent, &relu_vjp)).abs() <= 0.000_01);

    let leaky_relu_jvp = leaky_relu_jvp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        0.2,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("leaky ReLU JVP failed: {error}"));
    let leaky_relu_finite = finite_difference(&input, &tangent, |values| {
        leaky_relu_with_context_exact_native(&backend, values, 0.2, DeviceId::CPU, &caller_context)
            .unwrap_or_else(|error| panic!("leaky ReLU finite forward failed: {error}"))
    });
    assert_close(&leaky_relu_jvp, &leaky_relu_finite, 0.000_3);
    let leaky_relu_vjp = leaky_relu_vjp_with_context_exact_native(
        &backend,
        &input,
        &upstream,
        0.2,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("leaky ReLU VJP failed: {error}"));
    assert!((dot(&leaky_relu_jvp, &upstream) - dot(&tangent, &leaky_relu_vjp)).abs() <= 0.000_01);

    let silu_jvp = silu_jvp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("SiLU JVP failed: {error}"));
    let silu_finite = finite_difference(&input, &tangent, |values| {
        silu_with_context_exact_native(&backend, values, DeviceId::CPU, &caller_context)
            .unwrap_or_else(|error| panic!("SiLU finite forward failed: {error}"))
    });
    assert_close(&silu_jvp, &silu_finite, 0.000_3);
    let silu_vjp = silu_vjp_with_context_exact_native(
        &backend,
        &input,
        &upstream,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("SiLU VJP failed: {error}"));
    assert!((dot(&silu_jvp, &upstream) - dot(&tangent, &silu_vjp)).abs() <= 0.000_01);

    let gelu_jvp = gelu_jvp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        GeluApproximation::Tanh,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("GELU JVP failed: {error}"));
    let gelu_finite = finite_difference(&input, &tangent, |values| {
        gelu_with_context_exact_native(
            &backend,
            values,
            GeluApproximation::Tanh,
            DeviceId::CPU,
            &caller_context,
        )
        .unwrap_or_else(|error| panic!("GELU finite forward failed: {error}"))
    });
    assert_close(&gelu_jvp, &gelu_finite, 0.000_3);
    let gelu_vjp = gelu_vjp_with_context_exact_native(
        &backend,
        &input,
        &upstream,
        GeluApproximation::Tanh,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("GELU VJP failed: {error}"));
    assert!((dot(&gelu_jvp, &upstream) - dot(&tangent, &gelu_vjp)).abs() <= 0.000_01);

    let softmax_jvp = softmax_jvp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        &[2, 2],
        -1,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("softmax JVP failed: {error}"));
    let softmax_vjp = softmax_vjp_with_context_exact_native(
        &backend,
        &input,
        &upstream,
        &[2, 2],
        -1,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("softmax VJP failed: {error}"));
    assert!((dot(&softmax_jvp, &upstream) - dot(&tangent, &softmax_vjp)).abs() <= 0.000_01);

    let normalize_jvp = normalize_jvp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        &[2, 2],
        2.0,
        &[-1],
        1.0e-6,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("normalize JVP failed: {error}"));
    let normalize_vjp = normalize_vjp_with_context_exact_native(
        &backend,
        &input,
        &upstream,
        &[2, 2],
        2.0,
        &[-1],
        1.0e-6,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("normalize VJP failed: {error}"));
    assert!((dot(&normalize_jvp, &upstream) - dot(&tangent, &normalize_vjp)).abs() <= 0.000_01);
}

#[test]
fn every_functional_contract_preserves_cancellation_and_rejects_uncertified_devices() {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)
        .unwrap_or_else(|error| panic!("backend construction failed: {error}"));
    let active = CancellationToken::default();
    let active_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(16 * 1024 * 1024)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &active,
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(16 * 1024 * 1024)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancelled,
    );
    let input = [0.2, -0.4, 0.7, 1.1];
    let uncertified = DeviceId::new(DeviceKind::Cuda, 0);

    assert_cancelled(
        RELU_OPERATION_ID,
        relu_with_context_exact_native(&backend, &input, DeviceId::CPU, &cancelled_context),
    );
    assert_cancelled(
        LEAKY_RELU_OPERATION_ID,
        leaky_relu_with_context_exact_native(
            &backend,
            &input,
            0.2,
            DeviceId::CPU,
            &cancelled_context,
        ),
    );
    assert_cancelled(
        SILU_OPERATION_ID,
        silu_with_context_exact_native(&backend, &input, DeviceId::CPU, &cancelled_context),
    );
    assert_cancelled(
        GELU_OPERATION_ID,
        gelu_with_context_exact_native(
            &backend,
            &input,
            GeluApproximation::Tanh,
            DeviceId::CPU,
            &cancelled_context,
        ),
    );
    assert_cancelled(
        SOFTMAX_OPERATION_ID,
        softmax_with_context_exact_native(
            &backend,
            &input,
            &[2, 2],
            -1,
            DeviceId::CPU,
            &cancelled_context,
        ),
    );
    assert_cancelled(
        NORMALIZE_OPERATION_ID,
        normalize_with_context_exact_native(
            &backend,
            &input,
            &[2, 2],
            2.0,
            &[-1],
            1.0e-6,
            DeviceId::CPU,
            &cancelled_context,
        ),
    );
    assert_cancelled(
        LAYER_NORM_OPERATION_ID,
        layer_norm_with_context_exact_native(
            &backend,
            &input,
            &[2, 2],
            &[2],
            None,
            None,
            1.0e-5,
            DeviceId::CPU,
            &cancelled_context,
        ),
    );
    assert_cancelled(
        RMS_NORM_OPERATION_ID,
        rms_norm_with_context_exact_native(
            &backend,
            &input,
            &[2, 2],
            &[2],
            None,
            Some(1.0e-5),
            DeviceId::CPU,
            &cancelled_context,
        ),
    );
    assert_cancelled(
        GROUP_NORM_OPERATION_ID,
        group_norm_with_context_exact_native(
            &backend,
            &input,
            &[1, 2, 1, 2],
            1,
            None,
            None,
            1.0e-5,
            DeviceId::CPU,
            &cancelled_context,
        ),
    );
    assert_cancelled(
        BATCH_NORM_OPERATION_ID,
        batch_norm_with_context_exact_native(
            &backend,
            &input,
            &[2, 2, 1],
            None,
            None,
            None,
            None,
            true,
            0.1,
            1.0e-5,
            DeviceId::CPU,
            &cancelled_context,
        ),
    );

    assert_unsupported(
        RELU_OPERATION_ID,
        relu_with_context_exact_native(&backend, &input, uncertified, &active_context),
    );
    assert_unsupported(
        LEAKY_RELU_OPERATION_ID,
        leaky_relu_with_context_exact_native(&backend, &input, 0.2, uncertified, &active_context),
    );
    assert_unsupported(
        SILU_OPERATION_ID,
        silu_with_context_exact_native(&backend, &input, uncertified, &active_context),
    );
    assert_unsupported(
        GELU_OPERATION_ID,
        gelu_with_context_exact_native(
            &backend,
            &input,
            GeluApproximation::Tanh,
            uncertified,
            &active_context,
        ),
    );
    assert_unsupported(
        SOFTMAX_OPERATION_ID,
        softmax_with_context_exact_native(
            &backend,
            &input,
            &[2, 2],
            -1,
            uncertified,
            &active_context,
        ),
    );
    assert_unsupported(
        NORMALIZE_OPERATION_ID,
        normalize_with_context_exact_native(
            &backend,
            &input,
            &[2, 2],
            2.0,
            &[-1],
            1.0e-6,
            uncertified,
            &active_context,
        ),
    );
    assert_unsupported(
        LAYER_NORM_OPERATION_ID,
        layer_norm_with_context_exact_native(
            &backend,
            &input,
            &[2, 2],
            &[2],
            None,
            None,
            1.0e-5,
            uncertified,
            &active_context,
        ),
    );
    assert_unsupported(
        RMS_NORM_OPERATION_ID,
        rms_norm_with_context_exact_native(
            &backend,
            &input,
            &[2, 2],
            &[2],
            None,
            Some(1.0e-5),
            uncertified,
            &active_context,
        ),
    );
    assert_unsupported(
        GROUP_NORM_OPERATION_ID,
        group_norm_with_context_exact_native(
            &backend,
            &input,
            &[1, 2, 1, 2],
            1,
            None,
            None,
            1.0e-5,
            uncertified,
            &active_context,
        ),
    );
    assert_unsupported(
        BATCH_NORM_OPERATION_ID,
        batch_norm_with_context_exact_native(
            &backend,
            &input,
            &[2, 2, 1],
            None,
            None,
            None,
            None,
            true,
            0.1,
            1.0e-5,
            uncertified,
            &active_context,
        ),
    );
}

#[test]
fn affine_normalization_vjps_and_jvps_are_adjoint_and_match_finite_difference() {
    let cancellation = CancellationToken::default();
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)
        .unwrap_or_else(|error| panic!("backend construction failed: {error}"));
    let caller_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(16 * 1024 * 1024)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancellation,
    );
    let input = [0.2, -0.4, 0.7, 1.1];
    let tangent = [0.1, -0.3, 0.5, 0.2];
    let upstream = [0.8, -0.2, 0.4, -0.7];
    let weight = [1.2, 0.7];
    let layer_jvp = layer_norm_jvp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        &[2, 2],
        &[2],
        Some(&weight),
        None,
        None,
        1.0e-5,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("layer norm JVP failed: {error}"));
    let layer_vjp = layer_norm_vjp_with_context_exact_native(
        &backend,
        &input,
        &upstream,
        &[2, 2],
        &[2],
        Some(&weight),
        None,
        1.0e-5,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("layer norm VJP failed: {error}"));
    assert!((dot(&layer_jvp, &upstream) - dot(&tangent, &layer_vjp.input)).abs() <= 0.000_02);
    let layer_finite = finite_difference(&input, &tangent, |values| {
        layer_norm_with_context_exact_native(
            &backend,
            values,
            &[2, 2],
            &[2],
            Some(&weight),
            None,
            1.0e-5,
            DeviceId::CPU,
            &caller_context,
        )
        .unwrap_or_else(|error| panic!("layer norm finite forward failed: {error}"))
    });
    assert_close(&layer_jvp, &layer_finite, 0.003);

    let rms_jvp = rms_norm_jvp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        &[2, 2],
        &[2],
        Some(&weight),
        None,
        Some(1.0e-5),
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("RMS norm JVP failed: {error}"));
    let rms_vjp = rms_norm_vjp_with_context_exact_native(
        &backend,
        &input,
        &upstream,
        &[2, 2],
        &[2],
        Some(&weight),
        Some(1.0e-5),
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("RMS norm VJP failed: {error}"));
    assert!((dot(&rms_jvp, &upstream) - dot(&tangent, &rms_vjp.input)).abs() <= 0.000_02);

    let group_jvp = group_norm_jvp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        &[1, 2, 1, 2],
        1,
        Some(&weight),
        None,
        None,
        1.0e-5,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("group norm JVP failed: {error}"));
    let group_vjp = group_norm_vjp_with_context_exact_native(
        &backend,
        &input,
        &upstream,
        &[1, 2, 1, 2],
        1,
        Some(&weight),
        None,
        1.0e-5,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("group norm VJP failed: {error}"));
    assert!((dot(&group_jvp, &upstream) - dot(&tangent, &group_vjp.input)).abs() <= 0.000_02);

    let batch_jvp = batch_norm_jvp_with_context_exact_native(
        &backend,
        &input,
        &tangent,
        &[2, 2, 1],
        None,
        None,
        Some(&weight),
        None,
        None,
        true,
        1.0e-5,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("batch norm JVP failed: {error}"));
    let batch_vjp = batch_norm_vjp_with_context_exact_native(
        &backend,
        &input,
        &upstream,
        &[2, 2, 1],
        None,
        None,
        Some(&weight),
        None,
        true,
        1.0e-5,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("batch norm VJP failed: {error}"));
    assert!((dot(&batch_jvp, &upstream) - dot(&tangent, &batch_vjp.input)).abs() <= 0.000_02);
}

#[test]
fn canonical_normalization_and_in_place_staging_use_exact_workspace() {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1_024)
        .unwrap_or_else(|error| panic!("CPU backend construction failed: {error}"));
    let cancellation = CancellationToken::default();
    let caller_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(16)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancellation,
    );
    let input = [3.0, 4.0, 5.0, 12.0];
    let gradient = [0.2, -0.1, 0.3, 0.7];
    let exact = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(16)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancellation,
    );
    let canonical = normalize_vjp_with_context_exact_native(
        &backend,
        &input,
        &gradient,
        &[2, 2],
        2.0,
        &[-1],
        1.0e-6,
        DeviceId::CPU,
        &exact,
    )
    .unwrap_or_else(|error| panic!("canonical normalization VJP failed: {error}"));
    let second_lease_result = normalize_vjp_with_context_exact_native(
        &backend,
        &input,
        &gradient,
        &[2, 2],
        2.0,
        &[-1],
        1.0e-6,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("normalization VJP with a second lease failed: {error}"));
    assert_close(&canonical, &second_lease_result, 0.000_001);
    assert_eq!(exact.scratch.in_use_bytes(), 0);
    assert_eq!(exact.scratch.peak_bytes(), 16);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);

    let zero_scratch = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(0)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancellation,
    );
    assert_eq!(
        relu_with_context_exact_native(&backend, &[-2.0, 3.0], DeviceId::CPU, &zero_scratch)
            .unwrap_or_else(|error| panic!("zero-scratch ReLU failed: {error}")),
        [0.0, 3.0]
    );

    let underauthorized = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(15)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancellation,
    );
    assert!(
        normalize_vjp_with_context_exact_native(
            &backend,
            &input,
            &gradient,
            &[2, 2],
            2.0,
            &[-1],
            1.0e-6,
            DeviceId::CPU,
            &underauthorized,
        )
        .is_err()
    );
    assert_eq!(underauthorized.scratch.in_use_bytes(), 0);

    let mut in_place = [-2.0, 3.0, -4.0, 5.0];
    relu_with_context_exact_native_in_place(&backend, &mut in_place, DeviceId::CPU, &exact)
        .unwrap_or_else(|error| panic!("canonical in-place ReLU failed: {error}"));
    assert_eq!(in_place, [0.0, 3.0, 0.0, 5.0]);
    assert_eq!(exact.scratch.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
}

#[test]
fn canonical_softmax_derivatives_lease_the_exact_intermediate() {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(1_024)
        .unwrap_or_else(|error| panic!("CPU backend construction failed: {error}"));
    let cancellation = CancellationToken::default();
    let caller_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(16)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancellation,
    );
    let input = [-0.7, 0.2, 1.1, -1.3];
    let direction = [0.3, -0.5, 0.2, 0.7];
    let required = 16;
    let exact = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(required)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancellation,
    );
    let canonical_vjp = softmax_vjp_with_context_exact_native(
        &backend,
        &input,
        &direction,
        &[2, 2],
        -1,
        DeviceId::CPU,
        &exact,
    )
    .unwrap_or_else(|error| panic!("canonical softmax VJP failed: {error}"));
    let second_lease_vjp = softmax_vjp_with_context_exact_native(
        &backend,
        &input,
        &direction,
        &[2, 2],
        -1,
        DeviceId::CPU,
        &caller_context,
    )
    .unwrap_or_else(|error| panic!("softmax VJP with a second lease failed: {error}"));
    assert_close(&canonical_vjp, &second_lease_vjp, 0.000_001);
    let canonical_jvp = softmax_jvp_with_context_exact_native(
        &backend,
        &input,
        &direction,
        &[2, 2],
        -1,
        DeviceId::CPU,
        &exact,
    )
    .unwrap_or_else(|error| panic!("canonical softmax JVP failed: {error}"));
    assert_close(&canonical_jvp, &second_lease_vjp, 0.000_001);
    assert_eq!(exact.scratch.peak_bytes(), required);
    assert_eq!(exact.scratch.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);

    let insufficient = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(required - 1)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancellation,
    );
    assert!(
        softmax_vjp_with_context_exact_native(
            &backend,
            &input,
            &direction,
            &[2, 2],
            -1,
            DeviceId::CPU,
            &insufficient,
        )
        .is_err()
    );
    assert_eq!(insufficient.scratch.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(required)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancelled,
    );
    assert!(
        softmax_jvp_with_context_exact_native(
            &backend,
            &input,
            &direction,
            &[2, 2],
            -1,
            DeviceId::CPU,
            &cancelled_context,
        )
        .is_err()
    );
    assert_eq!(cancelled_context.scratch.peak_bytes(), 0);
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
}

#[test]
fn batch_stat_staging_is_failure_atomic_under_authorization_cancel_and_oom() {
    let (backend, workspace_authority) = CpuWorkspaceAuthority::create_backend(64)
        .unwrap_or_else(|error| panic!("CPU backend construction failed: {error}"));
    let cancellation = CancellationToken::default();
    let input = [1.0, 10.0, 3.0, 14.0];
    let mut running_mean = [0.0, 0.0];
    let mut running_variance = [1.0, 1.0];
    let success = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(16)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancellation,
    );
    let mut success_mean = running_mean;
    let mut success_variance = running_variance;
    batch_norm_with_context_exact_native(
        &backend,
        &input,
        &[2, 2, 1],
        Some(&mut success_mean),
        Some(&mut success_variance),
        None,
        None,
        true,
        0.5,
        1.0e-5,
        DeviceId::CPU,
        &success,
    )
    .unwrap_or_else(|error| panic!("canonical batch normalization failed: {error}"));
    assert_eq!(success.scratch.peak_bytes(), 16);
    assert_eq!(success.scratch.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    let underauthorized = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(15)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancellation,
    );
    assert!(
        batch_norm_with_context_exact_native(
            &backend,
            &input,
            &[2, 2, 1],
            Some(&mut running_mean),
            Some(&mut running_variance),
            None,
            None,
            true,
            0.5,
            1.0e-5,
            DeviceId::CPU,
            &underauthorized,
        )
        .is_err()
    );
    assert_eq!(running_mean, [0.0, 0.0]);
    assert_eq!(running_variance, [1.0, 1.0]);
    assert_eq!(underauthorized.scratch.in_use_bytes(), 0);

    let occupied_context = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(64)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancellation,
    );
    let occupied = backend
        .reserve_workspace(&occupied_context, 64)
        .unwrap_or_else(|error| panic!("persistent workspace reservation failed: {error}"));
    let exact = backend.execution_context(
        StreamId::DEFAULT,
        workspace_authority
            .authorize_workspace(16)
            .unwrap_or_else(|error| panic!("workspace authorization failed: {error}")),
        &cancellation,
    );
    assert!(
        batch_norm_with_context_exact_native(
            &backend,
            &input,
            &[2, 2, 1],
            Some(&mut running_mean),
            Some(&mut running_variance),
            None,
            None,
            true,
            0.5,
            1.0e-5,
            DeviceId::CPU,
            &exact,
        )
        .is_err()
    );
    assert_eq!(running_mean, [0.0, 0.0]);
    assert_eq!(running_variance, [1.0, 1.0]);
    assert_eq!(exact.scratch.in_use_bytes(), 0);
    drop(occupied);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);

    cancellation.cancel();
    let before_mean = running_mean;
    let before_variance = running_variance;
    assert!(
        batch_norm_with_context_exact_native(
            &backend,
            &input,
            &[2, 2, 1],
            Some(&mut running_mean),
            Some(&mut running_variance),
            None,
            None,
            true,
            0.5,
            1.0e-5,
            DeviceId::CPU,
            &exact,
        )
        .is_err()
    );
    assert_eq!(running_mean, before_mean);
    assert_eq!(running_variance, before_variance);
    assert_eq!(exact.scratch.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
}
