use half::f16;
use serde::Deserialize;
use sim_tensor::{
    BinaryOperation, CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId,
    ExecutionContext, ImageTensor, IndexSpec, Layout, Mt19937, OPERATION_CONTRACTS,
    OperationContractId, Philox4x32, ResizeCrop, ResizeMode, ResizeSpec, Scalar, ScalarSide,
    StreamId, Tensor, TensorBackend, TensorDescriptor, TensorError, UnaryOperation, ViewAccess,
};
use std::{
    collections::BTreeMap,
    collections::HashSet,
    error::Error,
    io,
    sync::{Arc, mpsc},
    time::Duration,
};

#[derive(Deserialize)]
struct ResizeOracle {
    inputs: BTreeMap<String, ResizeOracleInput>,
    cases: Vec<ResizeOracleCase>,
}

#[derive(Deserialize)]
struct ResizeOracleInput {
    shape_bhwc: [u64; 4],
    values_flat_bhwc_f32: Vec<f32>,
}

#[derive(Deserialize)]
struct ResizeOracleCase {
    arguments: ResizeOracleArguments,
    comparison: ResizeOracleComparison,
    id: String,
    input_id: String,
    output: ResizeOracleOutput,
}

#[derive(Deserialize)]
struct ResizeOracleArguments {
    crop: String,
    height: u64,
    upscale_method: String,
    width: u64,
}

#[derive(Deserialize)]
struct ResizeOracleComparison {
    #[serde(default)]
    absolute_tolerance: f32,
    #[serde(default)]
    relative_tolerance: f32,
    #[serde(default)]
    alias_required: bool,
}

#[derive(Deserialize)]
struct ResizeOracleOutput {
    shape_bhwc: [u64; 4],
    values_flat_bhwc_f32: Vec<f32>,
    #[serde(default)]
    values_flat_bhwc_u8: Option<Vec<u8>>,
}

fn context<'a>(
    authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> ExecutionContext<'a> {
    ExecutionContext {
        stream: StreamId::DEFAULT,
        scratch: authority
            .authorize_workspace(0)
            .expect("workspace authorization"),
        rng_phase: None,
        cancellation,
    }
}

fn descriptor(shape: Vec<u64>, dtype: DType) -> TensorDescriptor {
    match TensorDescriptor::contiguous(shape, dtype, DeviceId::CPU, StreamId::DEFAULT) {
        Ok(value) => value,
        Err(error) => panic!("test descriptor failed: {error}"),
    }
}

fn f32_tensor(shape: Vec<u64>, values: &[f32]) -> Tensor {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    tensor_from_bytes(descriptor(shape, DType::F32), bytes)
}

fn tensor_from_bytes(descriptor: TensorDescriptor, bytes: Vec<u8>) -> Tensor {
    let memory_limit = u64::try_from(bytes.len())
        .expect("test tensor length")
        .saturating_add(16);
    let (backend, authority) =
        CpuWorkspaceAuthority::create_backend(memory_limit).expect("test CPU backend");
    let cancellation = CancellationToken::default();
    let upload_context = ExecutionContext {
        stream: descriptor.stream(),
        scratch: authority
            .authorize_workspace(0)
            .expect("workspace authorization"),
        rng_phase: None,
        cancellation: &cancellation,
    };
    match backend.upload_bytes(descriptor, &bytes, &upload_context) {
        Ok((value, _)) => value,
        Err(error) => panic!("test tensor failed: {error}"),
    }
}

fn f32_values(tensor: &Tensor) -> Vec<f32> {
    let bytes = match tensor.contiguous_bytes() {
        Ok(value) => value,
        Err(error) => panic!("contiguous output failed: {error}"),
    };
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = match chunk.try_into() {
                Ok(value) => value,
                Err(error) => panic!("f32 chunk failed: {error}"),
            };
            f32::from_ne_bytes(bytes)
        })
        .collect()
}

#[test]
fn public_reference_projection_is_limited_to_the_generated_closed_catalog() {
    assert!(OperationContractId::new("zed.native-internal.forged").is_err());
    let references = OPERATION_CONTRACTS
        .iter()
        .filter_map(|contract| contract.typed_reference())
        .collect::<Vec<_>>();
    assert_eq!(references.len(), 82);
    assert_eq!(
        references
            .iter()
            .map(|contract| contract.operation_id())
            .collect::<HashSet<_>>()
            .len(),
        82
    );
    assert_eq!(
        references
            .iter()
            .map(|contract| contract.semantic())
            .collect::<HashSet<_>>()
            .len(),
        79
    );
    assert!(
        references
            .iter()
            .all(|contract| !contract.canonical_target().is_empty())
    );
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

fn oracle_resize_mode(value: &str) -> Result<ResizeMode, io::Error> {
    match value {
        "nearest-exact" => Ok(ResizeMode::NearestExact),
        "bilinear" => Ok(ResizeMode::Bilinear),
        "area" => Ok(ResizeMode::Area),
        "bicubic" => Ok(ResizeMode::Bicubic),
        "lanczos" => Ok(ResizeMode::Lanczos),
        value => Err(io::Error::other(format!(
            "unknown resize oracle mode {value}"
        ))),
    }
}

fn oracle_resize_crop(value: &str) -> Result<ResizeCrop, io::Error> {
    match value {
        "disabled" => Ok(ResizeCrop::Disabled),
        "center" => Ok(ResizeCrop::Center),
        value => Err(io::Error::other(format!(
            "unknown resize oracle crop {value}"
        ))),
    }
}

#[test]
fn val_tensor_001_checked_in_sim_image_resize_oracle_matches_native_execution()
-> Result<(), Box<dyn Error>> {
    const FIXTURE: &[u8] = include_bytes!(
        "../../sim_test_support/fixtures/tensor_operations/image_resize_foundation.json"
    );
    let oracle: ResizeOracle = serde_json::from_slice(FIXTURE)?;
    assert_eq!(oracle.cases.len(), 11);

    let cancellation = CancellationToken::default();
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
    let context = context(&authority, &cancellation);
    for case in oracle.cases {
        let input = oracle.inputs.get(&case.input_id).ok_or_else(|| {
            io::Error::other(format!(
                "resize oracle case {} references missing input {}",
                case.id, case.input_id
            ))
        })?;
        let [batch, height, width, channels] = input.shape_bhwc;
        let image = ImageTensor::from_f32(
            &backend,
            &context,
            batch,
            height,
            width,
            channels,
            &input.values_flat_bhwc_f32,
        )?;
        let resized = image.resize(
            case.arguments.width,
            case.arguments.height,
            oracle_resize_mode(&case.arguments.upscale_method)?,
            oracle_resize_crop(&case.arguments.crop)?,
            &backend,
            &context,
        )?;
        assert_eq!(
            resized.dimensions()?,
            tuple_dimensions(case.output.shape_bhwc),
            "resize oracle shape mismatch for {}",
            case.id
        );
        if case.comparison.alias_required {
            assert_eq!(
                resized.tensor().storage_id(),
                image.tensor().storage_id(),
                "resize oracle required input alias for {}",
                case.id
            );
        }
        let actual = resized.as_f32_slice()?;
        assert_eq!(actual.len(), case.output.values_flat_bhwc_f32.len());
        for (index, (actual, expected)) in actual
            .iter()
            .zip(&case.output.values_flat_bhwc_f32)
            .enumerate()
        {
            let tolerance = case
                .comparison
                .absolute_tolerance
                .max(case.comparison.relative_tolerance * actual.abs().max(expected.abs()));
            assert!(
                (actual - expected).abs() <= tolerance,
                "resize oracle {} element {index}: expected {expected}, got {actual}, tolerance {tolerance}",
                case.id
            );
        }
        if let Some(expected_bytes) = case.output.values_flat_bhwc_u8 {
            let actual_bytes = actual
                .iter()
                .map(|value| (value * 255.0).round().clamp(0.0, 255.0) as u8)
                .collect::<Vec<_>>();
            assert_eq!(
                actual_bytes, expected_bytes,
                "resize oracle quantized output mismatch for {}",
                case.id
            );
            assert!(
                actual
                    .iter()
                    .zip(&actual_bytes)
                    .all(|(value, byte)| { *value == f32::from(*byte) / 255.0 }),
                "resize oracle {} must decode exact quantized bytes",
                case.id
            );
        }
    }
    Ok(())
}

fn tuple_dimensions(shape: [u64; 4]) -> (u64, u64, u64, u64) {
    (shape[0], shape[1], shape[2], shape[3])
}

#[test]
fn allocation_is_aligned_bounded_and_released() {
    let (backend, authority) = match CpuWorkspaceAuthority::create_backend(16) {
        Ok(value) => value,
        Err(error) => panic!("CPU backend failed: {error}"),
    };
    let token = CancellationToken::default();
    let context = context(&authority, &token);
    let (tensor, event) = match backend.allocate(descriptor(vec![2], DType::F32), &context) {
        Ok(value) => value,
        Err(error) => panic!("allocation failed: {error}"),
    };
    let bytes = match tensor.host_storage_bytes() {
        Ok(value) => value,
        Err(error) => panic!("host storage failed: {error}"),
    };
    assert_eq!(bytes.as_ptr().align_offset(16), 0);
    assert_eq!(event.sequence(), 1);
    assert_eq!(backend.memory_snapshot().current_bytes, 16);
    assert!(matches!(
        backend.allocate(descriptor(vec![1], DType::F32), &context,),
        Err(TensorError::AllocationFailed { requested: 16, .. })
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, 16);
    drop(tensor);
    assert_eq!(
        backend.memory_snapshot(),
        sim_tensor::CpuMemorySnapshot {
            limit_bytes: 16,
            current_bytes: 0,
            peak_bytes: 16,
        }
    );
}

#[test]
fn cpu_instance_properties_match_the_injected_allocation_ceiling() {
    let (backend, authority) = match CpuWorkspaceAuthority::create_backend(33) {
        Ok(value) => value,
        Err(error) => panic!("CPU backend failed: {error}"),
    };
    let properties = match backend.capabilities().device_properties() {
        Some(value) => value,
        None => panic!("constructed CPU backend has no native properties"),
    };
    assert_eq!(properties.device(), DeviceId::CPU);
    assert_eq!(properties.name(), "Zed native Rust CPU");
    assert_eq!(properties.total_memory_bytes(), 33);
    assert_eq!(
        properties.total_memory_bytes(),
        authority.memory_snapshot().limit_bytes
    );
    assert_eq!(properties.architecture(), Some(std::env::consts::ARCH));
    assert!(!properties.has_fp16());
    assert!(matches!(
        CpuWorkspaceAuthority::create_backend(0),
        Err(TensorError::Faulted { .. })
    ));
}

#[test]
fn copy_on_write_is_charged_until_each_storage_is_released() {
    let (backend, authority) = match CpuWorkspaceAuthority::create_backend(32) {
        Ok(value) => value,
        Err(error) => panic!("CPU backend failed: {error}"),
    };
    let token = CancellationToken::default();
    let context = context(&authority, &token);
    let (mut first, _) = match backend.allocate(descriptor(vec![2], DType::F32), &context) {
        Ok(value) => value,
        Err(error) => panic!("allocation failed: {error}"),
    };
    let second = first.clone();
    {
        let mut write = match first.write() {
            Ok(value) => value,
            Err(error) => panic!("COW lease failed: {error}"),
        };
        let value = match write.element_bytes_mut(&[0]) {
            Ok(value) => value,
            Err(error) => panic!("COW element failed: {error}"),
        };
        value.copy_from_slice(&1_f32.to_ne_bytes());
    }
    assert_eq!(backend.memory_snapshot().current_bytes, 32);
    drop(second);
    assert_eq!(backend.memory_snapshot().current_bytes, 16);
    drop(first);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    assert_eq!(backend.memory_snapshot().peak_bytes, 32);
}

#[test]
fn aligned_capacity_is_the_memory_accounting_unit() {
    let token = CancellationToken::default();
    for (logical_bytes, limit, expected) in [
        (1_u64, 15_u64, None),
        (1, 16, Some(16)),
        (15, 16, Some(16)),
        (16, 16, Some(16)),
        (17, 31, None),
        (17, 32, Some(32)),
    ] {
        let (backend, authority) =
            CpuWorkspaceAuthority::create_backend(limit).expect("CPU backend");
        let result = backend.allocate(
            descriptor(vec![logical_bytes], DType::U8),
            &context(&authority, &token),
        );
        match expected {
            Some(expected) => {
                let (tensor, _) = result.expect("aligned allocation fits");
                assert_eq!(backend.memory_snapshot().current_bytes, expected);
                drop(tensor);
                assert_eq!(backend.memory_snapshot().current_bytes, 0);
            }
            None => assert!(matches!(result, Err(TensorError::AllocationFailed { .. }))),
        }
    }
}

#[test]
fn workspace_authorization_is_bound_to_its_issuing_backend() {
    let (first, first_authority) =
        CpuWorkspaceAuthority::create_backend(64).expect("first CPU backend");
    let (second, _second_authority) =
        CpuWorkspaceAuthority::create_backend(64).expect("second CPU backend");
    let cancellation = CancellationToken::default();

    let authorization = first_authority
        .authorize_workspace(16)
        .expect("workspace authorization");
    let foreign_context =
        second.execution_context(StreamId::DEFAULT, authorization.clone(), &cancellation);
    assert!(matches!(
        second.reserve_workspace(&foreign_context, 1),
        Err(TensorError::WorkspaceAuthorizationMismatch {
            expected_backend,
            actual_backend,
            actual_authority,
            ..
        }) if expected_backend != actual_backend && actual_authority != 0
    ));
    assert_eq!(authorization.in_use_bytes(), 0);
    assert_eq!(first.memory_snapshot().current_bytes, 0);
    assert_eq!(second.memory_snapshot().current_bytes, 0);
}

#[test]
fn overlapping_workspace_leases_obey_the_exact_authorization_and_capacity() {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(48).expect("CPU backend");
    let cancellation = CancellationToken::default();
    let authorization = authority
        .authorize_workspace(32)
        .expect("workspace authorization");
    let context =
        backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancellation);

    let ceiling_minus_one = backend
        .reserve_workspace(&context, 31)
        .expect("ceiling-minus-one lease");
    assert_eq!(ceiling_minus_one.bytes(), 31);
    assert_eq!(authorization.in_use_bytes(), 31);
    assert_eq!(backend.memory_snapshot().current_bytes, 32);

    let final_byte = backend
        .reserve_workspace(&context, 1)
        .expect("remaining authorization byte");
    assert_eq!(authorization.in_use_bytes(), 32);
    assert_eq!(authorization.peak_bytes(), 32);
    assert_eq!(backend.memory_snapshot().current_bytes, 48);
    assert!(matches!(
        backend.reserve_workspace(&context, 1),
        Err(TensorError::WorkspaceAuthorizationExceeded {
            requested: 1,
            authorized: 32,
            in_use: 32,
        })
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, 48);

    drop(final_byte);
    assert_eq!(authorization.in_use_bytes(), 31);
    assert_eq!(backend.memory_snapshot().current_bytes, 32);
    drop(ceiling_minus_one);
    assert_eq!(authorization.in_use_bytes(), 0);
    assert_eq!(
        backend.memory_snapshot(),
        sim_tensor::CpuMemorySnapshot {
            limit_bytes: 48,
            current_bytes: 0,
            peak_bytes: 48,
        }
    );
}

#[test]
fn workspace_authorization_and_capacity_are_atomic_across_threads() {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32).expect("CPU backend");
    let cancellation = CancellationToken::default();
    let authorization = authority
        .authorize_workspace(32)
        .expect("workspace authorization");
    let (acquired_sender, acquired_receiver) = mpsc::channel();
    let (first_release_sender, first_release_receiver) = mpsc::channel();
    let (second_release_sender, second_release_receiver) = mpsc::channel();

    std::thread::scope(|scope| {
        let first_backend = &backend;
        let first_cancellation = &cancellation;
        let first_authorization = authorization.clone();
        let first_acquired_sender = acquired_sender.clone();
        scope.spawn(move || {
            let context = first_backend.execution_context(
                StreamId::DEFAULT,
                first_authorization,
                first_cancellation,
            );
            let lease = first_backend
                .reserve_workspace(&context, 16)
                .expect("first concurrent lease");
            first_acquired_sender
                .send(())
                .expect("report first concurrent lease");
            first_release_receiver
                .recv_timeout(Duration::from_secs(5))
                .expect("release first concurrent lease");
            drop(lease);
        });

        let second_backend = &backend;
        let second_cancellation = &cancellation;
        let second_authorization = authorization.clone();
        scope.spawn(move || {
            let context = second_backend.execution_context(
                StreamId::DEFAULT,
                second_authorization,
                second_cancellation,
            );
            let lease = second_backend
                .reserve_workspace(&context, 16)
                .expect("second concurrent lease");
            acquired_sender
                .send(())
                .expect("report second concurrent lease");
            second_release_receiver
                .recv_timeout(Duration::from_secs(5))
                .expect("release second concurrent lease");
            drop(lease);
        });

        acquired_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("observe first concurrent lease");
        acquired_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("observe second concurrent lease");
        assert_eq!(authorization.in_use_bytes(), 32);
        assert_eq!(backend.memory_snapshot().current_bytes, 32);
        let context =
            backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancellation);
        assert!(matches!(
            backend.reserve_workspace(&context, 1),
            Err(TensorError::WorkspaceAuthorizationExceeded {
                requested: 1,
                authorized: 32,
                in_use: 32,
            })
        ));
        first_release_sender.send(()).expect("release first worker");
        second_release_sender
            .send(())
            .expect("release second worker");
    });

    assert_eq!(authorization.in_use_bytes(), 0);
    assert_eq!(authorization.peak_bytes(), 32);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    assert_eq!(backend.memory_snapshot().peak_bytes, 32);
}

#[test]
fn workspace_and_tensor_storage_share_one_capacity_tracker() {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(32).expect("CPU backend");
    let cancellation = CancellationToken::default();
    let base_context = context(&authority, &cancellation);
    let (tensor, _) = backend
        .allocate(descriptor(vec![1], DType::U8), &base_context)
        .expect("tracked tensor allocation");
    assert_eq!(backend.memory_snapshot().current_bytes, 16);

    let authorization = authority
        .authorize_workspace(16)
        .expect("workspace authorization");
    let workspace_context =
        backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancellation);
    let workspace = backend
        .reserve_workspace(&workspace_context, 16)
        .expect("workspace lease");
    assert_eq!(backend.memory_snapshot().current_bytes, 32);
    let before_failed_output = backend.memory_snapshot();
    assert!(matches!(
        backend.allocate(descriptor(vec![1], DType::U8), &base_context),
        Err(TensorError::AllocationFailed { requested: 16, .. })
    ));
    assert_eq!(
        backend.memory_snapshot().current_bytes,
        before_failed_output.current_bytes
    );
    assert_eq!(authorization.in_use_bytes(), 16);

    drop(workspace);
    assert_eq!(backend.memory_snapshot().current_bytes, 16);
    assert_eq!(authorization.in_use_bytes(), 0);
    drop(tensor);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    assert_eq!(backend.memory_snapshot().peak_bytes, 32);
    let (recovered, event) = backend
        .allocate(descriptor(vec![1], DType::U8), &base_context)
        .expect("allocation after released workspace");
    assert_eq!(event.sequence(), 2, "failed output was not published");
    drop(recovered);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
}

#[test]
fn workspace_vector_is_bounded_and_releases_on_cancellation() {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(16).expect("CPU backend");
    let cancellation = CancellationToken::default();
    let authorization = authority
        .authorize_workspace(16)
        .expect("workspace authorization");
    let context =
        backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancellation);
    let mut workspace = backend
        .workspace_vec::<u32>(&context, 4)
        .expect("workspace vector");
    assert_eq!(workspace.capacity(), 4);
    for value in 0..4 {
        workspace.try_push(value).expect("bounded workspace push");
    }
    assert_eq!(&*workspace, &[0, 1, 2, 3]);
    assert!(matches!(
        workspace.try_push(4),
        Err(TensorError::WorkspaceAuthorizationExceeded {
            requested: 4,
            authorized: 16,
            in_use: 16,
        })
    ));
    assert_eq!(workspace.len(), 4);
    assert_eq!(backend.memory_snapshot().current_bytes, 16);
    drop(workspace);
    assert_eq!(authorization.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);

    cancellation.cancel();
    assert!(matches!(
        backend.workspace_vec::<u8>(&context, 1),
        Err(TensorError::Cancelled)
    ));
    assert_eq!(authorization.in_use_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn fallible_workspace_vector_allocation_rolls_back_every_lease() {
    let (backend, authority) =
        CpuWorkspaceAuthority::create_backend(u64::MAX).expect("CPU backend");
    let cancellation = CancellationToken::default();
    let capacity = (isize::MAX as usize)
        .checked_add(1)
        .expect("capacity above Vec byte limit");
    let requested = u64::try_from(capacity).expect("64-bit capacity");
    let authorization = authority
        .authorize_workspace(requested)
        .expect("workspace authorization");
    let context =
        backend.execution_context(StreamId::DEFAULT, authorization.clone(), &cancellation);

    assert!(matches!(
        backend.workspace_vec::<u8>(&context, capacity),
        Err(TensorError::AllocationFailed {
            requested: actual,
            ..
        }) if actual == requested
    ));
    assert_eq!(authorization.in_use_bytes(), 0);
    assert_eq!(authorization.peak_bytes(), requested);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    assert_eq!(backend.memory_snapshot().peak_bytes, requested);
}

#[test]
fn read_only_views_cannot_mint_writable_access() {
    let tensor = f32_tensor(vec![2], &[1.0, 2.0]);
    let read_only = tensor
        .view(tensor.descriptor().clone(), ViewAccess::ReadOnly)
        .expect("read-only view");
    assert!(matches!(
        read_only.view(read_only.descriptor().clone(), ViewAccess::Writable),
        Err(TensorError::ReadOnlyView)
    ));
}

#[test]
fn copy_preserves_logical_order_for_negative_strides() {
    let (backend, authority) = match CpuWorkspaceAuthority::create_backend(64) {
        Ok(value) => value,
        Err(error) => panic!("CPU backend failed: {error}"),
    };
    let base = f32_tensor(vec![3], &[1.0, 2.0, 3.0]);
    let reversed = match TensorDescriptor::new_strided(
        vec![3],
        vec![-1],
        2,
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        StreamId::DEFAULT,
    ) {
        Ok(value) => value,
        Err(error) => panic!("reversed descriptor failed: {error}"),
    };
    let reversed = match base.view(reversed, ViewAccess::ReadOnly) {
        Ok(value) => value,
        Err(error) => panic!("reversed view failed: {error}"),
    };
    let token = CancellationToken::default();
    let (copied, _) = match backend.copy(
        &reversed,
        descriptor(vec![3], DType::F32),
        &context(&authority, &token),
    ) {
        Ok(value) => value,
        Err(error) => panic!("strided copy failed: {error}"),
    };
    assert_eq!(f32_values(&copied), vec![3.0, 2.0, 1.0]);
}

#[test]
fn copy_preserves_channels_last_logical_order() {
    let (backend, authority) = match CpuWorkspaceAuthority::create_backend(128) {
        Ok(value) => value,
        Err(error) => panic!("CPU backend failed: {error}"),
    };
    let channels_last = match TensorDescriptor::channels_last(
        vec![1, 2, 2, 2],
        DType::F32,
        DeviceId::CPU,
        StreamId::DEFAULT,
    ) {
        Ok(value) => value,
        Err(error) => panic!("channels-last descriptor failed: {error}"),
    };
    let physical_values = [0.0_f32, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0];
    let physical_bytes = physical_values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let input = tensor_from_bytes(channels_last, physical_bytes);
    let token = CancellationToken::default();
    let (copied, _) = match backend.copy(
        &input,
        descriptor(vec![1, 2, 2, 2], DType::F32),
        &context(&authority, &token),
    ) {
        Ok(value) => value,
        Err(error) => panic!("channels-last copy failed: {error}"),
    };
    assert_eq!(
        f32_values(&copied),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    );
}

#[test]
fn scalar_fill_handles_half_boolean_complex_and_range_errors() {
    let (backend, authority) = match CpuWorkspaceAuthority::create_backend(128) {
        Ok(value) => value,
        Err(error) => panic!("CPU backend failed: {error}"),
    };
    let token = CancellationToken::default();
    let context = context(&authority, &token);
    let (half, _) = match backend.fill(
        Scalar::Float(0.5),
        descriptor(vec![2], DType::F16),
        &context,
    ) {
        Ok(value) => value,
        Err(error) => panic!("half fill failed: {error}"),
    };
    let half_bytes = match half.contiguous_bytes() {
        Ok(value) => value,
        Err(error) => panic!("half bytes failed: {error}"),
    };
    for chunk in half_bytes.chunks_exact(2) {
        let bits: [u8; 2] = match chunk.try_into() {
            Ok(value) => value,
            Err(error) => panic!("half chunk failed: {error}"),
        };
        assert_eq!(f16::from_bits(u16::from_ne_bytes(bits)), f16::from_f32(0.5));
    }

    let (boolean, _) = match backend.fill(
        Scalar::Float(f64::NAN),
        descriptor(vec![1], DType::Bool),
        &context,
    ) {
        Ok(value) => value,
        Err(error) => panic!("boolean fill failed: {error}"),
    };
    assert!(matches!(boolean.contiguous_bytes(), Ok([1])));

    let (complex, _) = match backend.fill(
        Scalar::Signed(-2),
        descriptor(vec![], DType::Complex64),
        &context,
    ) {
        Ok(value) => value,
        Err(error) => panic!("complex fill failed: {error}"),
    };
    let mut expected_complex = (-2_f32).to_ne_bytes().to_vec();
    expected_complex.extend_from_slice(&0_f32.to_ne_bytes());
    assert_eq!(complex.contiguous_bytes(), Ok(expected_complex.as_slice()));

    assert!(matches!(
        backend.fill(Scalar::Signed(-1), descriptor(vec![1], DType::U8), &context,),
        Err(TensorError::InvalidNumeric { .. })
    ));
}

#[test]
fn allocation_and_copy_cover_every_declared_dtype() {
    let (backend, authority) = match CpuWorkspaceAuthority::create_backend(1024) {
        Ok(value) => value,
        Err(error) => panic!("CPU backend failed: {error}"),
    };
    let token = CancellationToken::default();
    let context = context(&authority, &token);
    let dtypes = [
        DType::F64,
        DType::F32,
        DType::F16,
        DType::Bf16,
        DType::I64,
        DType::I32,
        DType::I16,
        DType::I8,
        DType::U64,
        DType::U32,
        DType::U16,
        DType::U8,
        DType::Bool,
        DType::Complex64,
        DType::Complex128,
        DType::Float8E4m3Fn,
        DType::Float8E5m2,
        DType::Float8E4m3Fnuz,
        DType::Float8E5m2Fnuz,
        DType::Float8E8m0Fnu,
    ];
    for dtype in dtypes {
        let length = match usize::try_from(dtype.byte_width()) {
            Ok(value) => value,
            Err(error) => panic!("dtype width conversion failed: {error}"),
        };
        let source = tensor_from_bytes(descriptor(vec![1], dtype), vec![0x5a; length]);
        let (copied, _) = match backend.copy(&source, descriptor(vec![1], dtype), &context) {
            Ok(value) => value,
            Err(error) => panic!("{dtype:?} copy failed: {error}"),
        };
        assert_eq!(copied.contiguous_bytes(), Ok(&vec![0x5a; length][..]));
        drop(copied);
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
    }
}

#[test]
fn scalar_fill_covers_every_certified_dtype() {
    let (backend, authority) = match CpuWorkspaceAuthority::create_backend(1024) {
        Ok(value) => value,
        Err(error) => panic!("CPU backend failed: {error}"),
    };
    let token = CancellationToken::default();
    let context = context(&authority, &token);
    let dtypes = [
        DType::F64,
        DType::F32,
        DType::F16,
        DType::Bf16,
        DType::I64,
        DType::I32,
        DType::I16,
        DType::I8,
        DType::U64,
        DType::U32,
        DType::U16,
        DType::U8,
        DType::Bool,
        DType::Complex64,
        DType::Complex128,
        DType::Float8E4m3Fn,
        DType::Float8E5m2,
        DType::Float8E4m3Fnuz,
        DType::Float8E5m2Fnuz,
    ];
    for dtype in dtypes {
        let (filled, _) =
            match backend.fill(Scalar::Unsigned(1), descriptor(vec![1], dtype), &context) {
                Ok(value) => value,
                Err(error) => panic!("{dtype:?} fill failed: {error}"),
            };
        assert_eq!(filled.storage_byte_len(), dtype.byte_width());
        drop(filled);
    }
    assert!(matches!(
        backend.fill(
            Scalar::Unsigned(1),
            descriptor(vec![1], DType::Float8E8m0Fnu),
            &context,
        ),
        Err(TensorError::UnsupportedCapability { .. })
    ));
}

#[test]
fn unary_image_and_numeric_kernels_are_deterministic_and_cancel_before_allocation() {
    let (backend, authority) = match CpuWorkspaceAuthority::create_backend(128) {
        Ok(value) => value,
        Err(error) => panic!("CPU backend failed: {error}"),
    };
    let input = f32_tensor(vec![3], &[0.0, 0.25, 1.0]);
    let token = CancellationToken::default();
    let (inverted, _) = match backend.unary(
        UnaryOperation::InvertUnitInterval,
        &input,
        descriptor(vec![3], DType::F32),
        &context(&authority, &token),
    ) {
        Ok(value) => value,
        Err(error) => panic!("image invert failed: {error}"),
    };
    assert_eq!(f32_values(&inverted), vec![1.0, 0.75, 0.0]);

    let numeric = f32_tensor(vec![3], &[1.0, f32::INFINITY, f32::NAN]);
    let (finite, _) = match backend.unary(
        UnaryOperation::IsFinite,
        &numeric,
        descriptor(vec![3], DType::Bool),
        &context(&authority, &token),
    ) {
        Ok(value) => value,
        Err(error) => panic!("isfinite failed: {error}"),
    };
    assert_eq!(finite.contiguous_bytes(), Ok(&[1, 0, 0][..]));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let before = backend.memory_snapshot().current_bytes;
    assert!(matches!(
        backend.unary(
            UnaryOperation::Negate,
            &input,
            descriptor(vec![3], DType::F32),
            &context(&authority, &cancelled),
        ),
        Err(TensorError::Cancelled)
    ));
    assert_eq!(backend.memory_snapshot().current_bytes, before);
}

#[test]
fn cancellation_during_a_kernel_releases_uncommitted_storage() {
    let element_count = 1_000_000_usize;
    let element_count_u64 = match u64::try_from(element_count) {
        Ok(value) => value,
        Err(error) => panic!("test element count conversion failed: {error}"),
    };
    let memory_limit = match element_count_u64.checked_mul(4) {
        Some(value) => value,
        None => panic!("test memory size overflowed"),
    };
    let (backend, authority) = match CpuWorkspaceAuthority::create_backend(memory_limit) {
        Ok(value) => value,
        Err(error) => panic!("CPU backend failed: {error}"),
    };
    let backend = Arc::new(backend);
    let input = f32_tensor(vec![element_count_u64], &vec![1.0; element_count]);
    let cancellation = CancellationToken::default();
    let thread_backend = backend.clone();
    let thread_cancellation = cancellation.clone();
    let thread_authorization = authority
        .authorize_workspace(0)
        .expect("workspace authorization");
    let worker = std::thread::spawn(move || {
        let thread_context = thread_backend.execution_context(
            StreamId::DEFAULT,
            thread_authorization,
            &thread_cancellation,
        );
        thread_backend.unary(
            UnaryOperation::Negate,
            &input,
            descriptor(vec![element_count_u64], DType::F32),
            &thread_context,
        )
    });
    let mut observed_allocation = false;
    for _ in 0..1_000_000 {
        if backend.memory_snapshot().current_bytes == memory_limit {
            observed_allocation = true;
            cancellation.cancel();
            break;
        }
        std::thread::yield_now();
    }
    assert!(
        observed_allocation,
        "kernel output allocation was not observed"
    );
    let result = match worker.join() {
        Ok(value) => value,
        Err(_) => panic!("kernel worker panicked"),
    };
    assert!(matches!(result, Err(TensorError::Cancelled)));
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
}

#[test]
fn binary_tensor_and_scalar_rules_match_reference_semantics() {
    let (backend, authority) = match CpuWorkspaceAuthority::create_backend(256) {
        Ok(value) => value,
        Err(error) => panic!("CPU backend failed: {error}"),
    };
    let input = f32_tensor(vec![2], &[5.0, -5.0]);
    let token = CancellationToken::default();
    let (remainder, _) = match backend.binary_scalar(
        BinaryOperation::Remainder,
        &input,
        Scalar::Float(3.0),
        ScalarSide::Right,
        descriptor(vec![2], DType::F32),
        &context(&authority, &token),
    ) {
        Ok(value) => value,
        Err(error) => panic!("scalar remainder failed: {error}"),
    };
    assert_eq!(f32_values(&remainder), vec![2.0, 1.0]);

    let (greater, _) = match backend.binary_scalar(
        BinaryOperation::Greater,
        &input,
        Scalar::Float(0.0),
        ScalarSide::Right,
        descriptor(vec![2], DType::Bool),
        &context(&authority, &token),
    ) {
        Ok(value) => value,
        Err(error) => panic!("scalar comparison failed: {error}"),
    };
    assert_eq!(greater.contiguous_bytes(), Ok(&[1, 0][..]));

    let scalar = f32_tensor(vec![], &[2.0]);
    let (sum, _) = match backend.binary(
        BinaryOperation::Add,
        &input,
        &scalar,
        descriptor(vec![2], DType::F32),
        &context(&authority, &token),
    ) {
        Ok(value) => value,
        Err(error) => panic!("rank-zero binary failed: {error}"),
    };
    assert_eq!(f32_values(&sum), vec![7.0, -3.0]);
}

#[test]
fn select_and_narrow_copy_exact_logical_elements_and_reject_unowned_indexing() {
    let (backend, authority) = match CpuWorkspaceAuthority::create_backend(256) {
        Ok(value) => value,
        Err(error) => panic!("CPU backend failed: {error}"),
    };
    let input = f32_tensor(vec![2, 3], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let token = CancellationToken::default();
    let context = context(&authority, &token);
    let (selected, _) = match backend.indexing(
        &IndexSpec::Select {
            dimension: 1,
            index: -1,
        },
        std::slice::from_ref(&input),
        descriptor(vec![2], DType::F32),
        &context,
    ) {
        Ok(value) => value,
        Err(error) => panic!("select failed: {error}"),
    };
    assert_eq!(f32_values(&selected), vec![2.0, 5.0]);

    let (narrowed, _) = match backend.indexing(
        &IndexSpec::Narrow {
            dimension: 1,
            start: -2,
            length: 2,
        },
        std::slice::from_ref(&input),
        descriptor(vec![2, 2], DType::F32),
        &context,
    ) {
        Ok(value) => value,
        Err(error) => panic!("narrow failed: {error}"),
    };
    assert_eq!(f32_values(&narrowed), vec![1.0, 2.0, 4.0, 5.0]);

    assert!(matches!(
        backend.indexing(
            &IndexSpec::Gather { dimension: 0 },
            std::slice::from_ref(&input),
            descriptor(vec![2, 3], DType::F32),
            &context,
        ),
        Err(TensorError::UnsupportedCapability { .. })
    ));
}

fn resize(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    input: &Tensor,
    width: u64,
    height: u64,
    mode: ResizeMode,
    crop: ResizeCrop,
) -> Tensor {
    let token = CancellationToken::default();
    let [batch, channels, _, _] = input.descriptor().shape() else {
        panic!("resize test input must be rank four");
    };
    let output_shape = vec![*batch, *channels, height, width];
    match backend.resize(
        ResizeSpec {
            width,
            height,
            mode,
            crop,
            antialias: false,
            align_corners: false,
        },
        input,
        descriptor(output_shape, DType::F32),
        &context(authority, &token),
    ) {
        Ok((tensor, _)) => tensor,
        Err(error) => panic!("resize failed: {error}"),
    }
}

#[test]
fn resize_primitives_cover_the_native_image_slice() {
    let (backend, authority) = match CpuWorkspaceAuthority::create_backend(4096) {
        Ok(value) => value,
        Err(error) => panic!("CPU backend failed: {error}"),
    };
    let input = f32_tensor(vec![1, 1, 2, 2], &[0.0, 1.0, 2.0, 3.0]);
    let nearest = resize(
        &backend,
        &authority,
        &input,
        4,
        4,
        ResizeMode::NearestExact,
        ResizeCrop::Disabled,
    );
    assert_eq!(
        f32_values(&nearest),
        vec![
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0,
        ]
    );

    let bilinear = resize(
        &backend,
        &authority,
        &input,
        3,
        3,
        ResizeMode::Bilinear,
        ResizeCrop::Disabled,
    );
    assert_close(
        &f32_values(&bilinear),
        &[0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 2.5, 3.0],
        1e-6,
    );

    let area = resize(
        &backend,
        &authority,
        &input,
        1,
        1,
        ResizeMode::Area,
        ResizeCrop::Disabled,
    );
    assert_eq!(f32_values(&area), vec![1.5]);

    let bicubic = resize(
        &backend,
        &authority,
        &input,
        2,
        2,
        ResizeMode::Bicubic,
        ResizeCrop::Disabled,
    );
    assert_close(&f32_values(&bicubic), &[0.0, 1.0, 2.0, 3.0], 1e-6);

    let lanczos_input = f32_tensor(vec![1, 1, 2, 2], &[0.0, 0.25, 0.5, 1.0]);
    let lanczos = resize(
        &backend,
        &authority,
        &lanczos_input,
        2,
        2,
        ResizeMode::Lanczos,
        ResizeCrop::Disabled,
    );
    assert_close(
        &f32_values(&lanczos),
        &[0.0, 63.0 / 255.0, 127.0 / 255.0, 1.0],
        1e-6,
    );
}

#[test]
fn center_crop_matches_source_aspect_rule() {
    let (backend, authority) = match CpuWorkspaceAuthority::create_backend(1024) {
        Ok(value) => value,
        Err(error) => panic!("CPU backend failed: {error}"),
    };
    let input = f32_tensor(vec![1, 1, 2, 4], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let cropped = resize(
        &backend,
        &authority,
        &input,
        2,
        2,
        ResizeMode::NearestExact,
        ResizeCrop::Center,
    );
    assert_eq!(f32_values(&cropped), vec![1.0, 2.0, 5.0, 6.0]);
}

#[test]
fn cpu_events_are_bounded_and_reject_foreign_backends() {
    let (backend, authority) = match CpuWorkspaceAuthority::create_backend(1) {
        Ok(value) => value,
        Err(error) => panic!("CPU backend failed: {error}"),
    };
    let token = CancellationToken::default();
    let default_context = context(&authority, &token);
    let first = match backend.record_event(&default_context) {
        Ok(value) => value,
        Err(error) => panic!("first event failed: {error}"),
    };
    let second = match backend.record_event(&default_context) {
        Ok(value) => value,
        Err(error) => panic!("second event failed: {error}"),
    };
    assert_eq!((first.sequence(), second.sequence()), (1, 2));
    assert!(backend.wait_event(first, &default_context).is_ok());
    let (other_backend, other_authority) =
        CpuWorkspaceAuthority::create_backend(1).expect("second CPU backend");
    let other_context = context(&other_authority, &token);
    let foreign = other_backend
        .record_event(&other_context)
        .expect("foreign backend event");
    assert!(matches!(
        backend.wait_event(foreign, &default_context),
        Err(TensorError::Faulted { .. })
    ));
    let stream_context = ExecutionContext {
        stream: StreamId::new(7),
        scratch: authority
            .authorize_workspace(0)
            .expect("workspace authorization"),
        rng_phase: None,
        cancellation: &token,
    };
    let stream_event = backend
        .record_event(&stream_context)
        .expect("second stream event");
    assert_eq!(stream_event.sequence(), 3);
    assert!(matches!(
        backend.wait_event(stream_event, &default_context),
        Err(TensorError::StreamMismatch { .. })
    ));
    let mut last_sequence = stream_event.sequence();
    for ordinal in 0..10_000 {
        let stream_context = ExecutionContext {
            stream: StreamId::new(100 + ordinal),
            scratch: authority
                .authorize_workspace(0)
                .expect("workspace authorization"),
            rng_phase: None,
            cancellation: &token,
        };
        last_sequence = backend
            .record_event(&stream_context)
            .expect("bounded event sequence")
            .sequence();
    }
    assert_eq!(last_sequence, 10_003);
}

#[test]
fn every_tensor_primitive_rejects_cross_stream_inputs_before_allocation() {
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024).expect("CPU backend");
    let cancellation = CancellationToken::default();
    let default_context = context(&authority, &cancellation);
    let foreign_descriptor = TensorDescriptor::contiguous(
        vec![1, 1, 1, 1],
        DType::F32,
        DeviceId::CPU,
        StreamId::new(7),
    )
    .expect("foreign stream descriptor");
    let foreign = tensor_from_bytes(foreign_descriptor, 1_f32.to_ne_bytes().to_vec());
    let output = descriptor(vec![1, 1, 1, 1], DType::F32);
    let assert_stream_mismatch = |result: Result<(), TensorError>| {
        assert!(matches!(result, Err(TensorError::StreamMismatch { .. })));
        assert_eq!(backend.memory_snapshot().current_bytes, 0);
    };
    assert_stream_mismatch(
        backend
            .copy(&foreign, output.clone(), &default_context)
            .map(|_| ()),
    );
    assert_stream_mismatch(
        backend
            .unary(
                UnaryOperation::Negate,
                &foreign,
                output.clone(),
                &default_context,
            )
            .map(|_| ()),
    );
    assert_stream_mismatch(
        backend
            .binary(
                BinaryOperation::Add,
                &foreign,
                &foreign,
                output.clone(),
                &default_context,
            )
            .map(|_| ()),
    );
    assert_stream_mismatch(
        backend
            .binary_scalar(
                BinaryOperation::Add,
                &foreign,
                Scalar::Float(1.0),
                ScalarSide::Right,
                output.clone(),
                &default_context,
            )
            .map(|_| ()),
    );
    assert_stream_mismatch(
        backend
            .indexing(
                &IndexSpec::Narrow {
                    dimension: 0,
                    start: 0,
                    length: 1,
                },
                std::slice::from_ref(&foreign),
                output.clone(),
                &default_context,
            )
            .map(|_| ()),
    );
    assert_stream_mismatch(
        backend
            .resize(
                ResizeSpec {
                    width: 1,
                    height: 1,
                    mode: ResizeMode::NearestExact,
                    crop: ResizeCrop::Disabled,
                    antialias: false,
                    align_corners: false,
                },
                &foreign,
                output,
                &default_context,
            )
            .map(|_| ()),
    );
}

#[test]
fn cpu_rng_foundations_retain_standard_vectors() {
    let mut mt = Mt19937::from_seed(5489);
    assert_eq!(
        (0..5).map(|_| mt.next_u32()).collect::<Vec<_>>(),
        vec![
            3_499_211_612,
            581_869_302,
            3_890_346_734,
            3_586_334_585,
            545_404_204,
        ]
    );
    assert_eq!(
        Philox4x32::generate([0; 4], [0; 2]),
        [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
    );
}
