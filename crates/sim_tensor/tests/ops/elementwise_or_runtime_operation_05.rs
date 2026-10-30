use sha2::{Digest, Sha256};
use sim_tensor::CpuWorkspaceAuthority;
use sim_tensor::{
    BackendCapabilityMatrix, CachedAllocationOwner, CancellationToken, CpuBackend, DType,
    DecodedScalar, DeviceId, ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES,
    NativeDeviceProperties, Scalar, StreamId, Tensor, TensorDescriptor, TensorError,
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_05::{
        BitwiseShiftOperand, ElementwiseRuntimePartFiveError,
        bitwise_left_shift_with_context_exact_native, constant_in_place_with_context_exact_native,
        count_nonzero_with_context_exact_native, cuda_get_allocator_backend_exact_native,
        cuda_set_device_exact_native, div_jvp_with_context_exact_native,
        div_vjp_with_context_exact_native, div_with_context_exact_native,
        item_with_context_exact_native, minimum_jvp_with_context_exact_native,
        minimum_vjp_with_context_exact_native, minimum_with_context_exact_native,
        sin_jvp_with_context_exact_native, sin_vjp_with_context_exact_native,
        sin_with_context_exact_native, sqrt_jvp_with_context_exact_native,
        sqrt_vjp_with_context_exact_native, sqrt_with_context_exact_native,
        xpu_get_device_name_exact_native, zero_in_place_with_context_exact_native,
    },
    generated_sim_operator_indirection_01::tensor_to_f32_with_context_exact_native,
};
use sim_types::DeviceKind;
use std::{collections::BTreeSet, fs, path::Path};

const IDS: [&str; 12] = [
    "COMFY-TENSOR-OP-365E27719CFD",
    "COMFY-TENSOR-OP-3D0519DB53BD",
    "COMFY-TENSOR-OP-332E7E59DC10",
    "COMFY-TENSOR-OP-3D09997B7D21",
    "COMFY-TENSOR-OP-33CCFBAAA7B3",
    "COMFY-TENSOR-OP-384A8C6954B8",
    "COMFY-TENSOR-OP-3ADC7A3998E4",
    "COMFY-TENSOR-OP-40FEFA2DEAC6",
    "COMFY-TENSOR-OP-3A3C79159CBC",
    "COMFY-TENSOR-OP-388D285AB0F7",
    "COMFY-TENSOR-OP-37266D0A196F",
    "COMFY-TENSOR-OP-3A641CA3FC0F",
];

fn backend() -> Result<(CpuBackend, CpuWorkspaceAuthority), Box<dyn std::error::Error>> {
    Ok(CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?)
}

fn upload_f32(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    Ok(backend
        .upload_f32(
            descriptor,
            values,
            &context(backend, authority, cancellation)?,
        )?
        .0)
}

fn upload_i32(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    shape: &[u64],
    values: &[i32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::I32, DeviceId::CPU, StreamId::DEFAULT)?;
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    Ok(backend
        .upload_bytes(
            descriptor,
            &bytes,
            &context(backend, authority, cancellation)?,
        )?
        .0)
}

fn context<'a>(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    cancellation: &'a CancellationToken,
) -> Result<ExecutionContext<'a>, TensorError> {
    Ok(backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(1024 * 1024)?,
        cancellation,
    ))
}

fn values(
    backend: &CpuBackend,
    authority: &CpuWorkspaceAuthority,
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(tensor_to_f32_with_context_exact_native(
        backend,
        tensor,
        &backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            cancellation,
        ),
    )?)
}

fn i64_scalar(tensor: &Tensor) -> Result<i64, Box<dyn std::error::Error>> {
    match tensor
        .descriptor()
        .dtype()
        .decode_scalar(tensor.element_bytes(&[])?)?
    {
        DecodedScalar::Signed(value) => Ok(value),
        _ => Err("expected signed scalar tensor".into()),
    }
}

#[test]
fn resolution_slice_seals_exactly_the_assigned_contracts_and_fixtures()
-> Result<(), Box<dyn std::error::Error>> {
    let slice = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .find(|slice| slice.module_name == "elementwise_or_runtime_operation_05")
        .ok_or("elementwise/runtime part-five resolution slice is missing")?;
    assert_eq!(slice.len(), IDS.len());
    assert_eq!(
        slice
            .iter()
            .map(|contract| contract.operation_id)
            .collect::<BTreeSet<_>>(),
        IDS.into_iter().collect()
    );
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root is missing")?;
    for contract in slice.iter() {
        assert_eq!(
            contract.owner_task_id,
            "comfy-parity-tensor-ops-elementwise-or-runtime-operation-comfy-tensor-op-332e7e59dc10"
        );
        let bytes = fs::read(workspace.join(contract.evidence_fixture))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            contract.evidence_fixture_sha256
        );
        let callable = match contract.operation_id {
            "COMFY-TENSOR-OP-365E27719CFD" => "div_with_context_exact_native",
            "COMFY-TENSOR-OP-3D0519DB53BD" => "item_with_context_exact_native",
            "COMFY-TENSOR-OP-332E7E59DC10" => "sin_with_context_exact_native",
            "COMFY-TENSOR-OP-3D09997B7D21" => "sqrt_with_context_exact_native",
            "COMFY-TENSOR-OP-33CCFBAAA7B3" => "zero_in_place_with_context_exact_native",
            "COMFY-TENSOR-OP-384A8C6954B8" => "bitwise_left_shift_with_context_exact_native",
            "COMFY-TENSOR-OP-3ADC7A3998E4" => "count_nonzero_with_context_exact_native",
            "COMFY-TENSOR-OP-40FEFA2DEAC6" => "cuda_get_allocator_backend_exact_native",
            "COMFY-TENSOR-OP-3A3C79159CBC" => "cuda_set_device_exact_native",
            "COMFY-TENSOR-OP-388D285AB0F7" => "minimum_with_context_exact_native",
            "COMFY-TENSOR-OP-37266D0A196F" => "constant_in_place_with_context_exact_native",
            "COMFY-TENSOR-OP-3A641CA3FC0F" => "xpu_get_device_name_exact_native",
            _ => return Err("unexpected Task 48 operation identifier".into()),
        };
        assert!(contract.rust_signature.contains(callable));
        if contract.rust_signature.contains("ExecutionContext") {
            assert!(contract.rust_signature.contains("ExecutionContext<'_>"));
        }
        if contract.operation_id == "COMFY-TENSOR-OP-3A3C79159CBC" {
            assert!(
                contract
                    .rust_signature
                    .contains("fn cuda_set_device_exact_native<'a>")
            );
            assert!(
                contract
                    .rust_signature
                    .contains("Result<&'a BackendCapabilityMatrix")
            );
        }
    }
    Ok(())
}

#[test]
fn divide_sine_and_square_root_reuse_canonical_primitives_with_gradients()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let input = upload_f32(&backend, &authority, &[2, 1], &[2.0, 8.0], &cancellation)?;
    let denominator = upload_f32(&backend, &authority, &[1, 2], &[2.0, 4.0], &cancellation)?;
    let divided = div_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Tensor(&denominator),
        &execution,
    )?;
    assert_eq!(divided.descriptor().shape(), [2, 2]);
    assert_eq!(
        values(&backend, &authority, &divided, &cancellation)?,
        [1.0, 0.5, 4.0, 2.0]
    );

    let gradient = upload_f32(&backend, &authority, &[2, 2], &[1.0; 4], &cancellation)?;
    let vjp = div_vjp_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Tensor(&denominator),
        &gradient,
        &execution,
    )?;
    assert_eq!(
        values(&backend, &authority, &vjp.input, &cancellation)?,
        [0.75, 0.75]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &vjp.other.ok_or("missing denominator gradient")?,
            &cancellation
        )?,
        [-2.5, -0.625]
    );
    let jvp = div_jvp_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Tensor(&denominator),
        &upload_f32(&backend, &authority, &[2, 1], &[1.0, 2.0], &cancellation)?,
        Some(&upload_f32(
            &backend,
            &authority,
            &[1, 2],
            &[0.5, 1.0],
            &cancellation,
        )?),
        &execution,
    )?;
    assert_eq!(
        values(&backend, &authority, &jvp, &cancellation)?,
        [0.25, 0.125, 0.0, 0.0]
    );

    let angles = upload_f32(
        &backend,
        &authority,
        &[2],
        &[0.0, std::f32::consts::FRAC_PI_2],
        &cancellation,
    )?;
    let sine = sin_with_context_exact_native(&backend, &angles, &execution)?;
    assert_eq!(
        values(&backend, &authority, &sine, &cancellation)?,
        [0.0, 1.0]
    );
    let tangent = sin_jvp_with_context_exact_native(
        &backend,
        &angles,
        &upload_f32(&backend, &authority, &[2], &[2.0, 2.0], &cancellation)?,
        &execution,
    )?;
    let tangent_values = values(&backend, &authority, &tangent, &cancellation)?;
    assert_eq!(tangent_values[0], 2.0);
    assert!(tangent_values[1].abs() < 1e-6);

    let roots = upload_f32(&backend, &authority, &[2], &[4.0, 9.0], &cancellation)?;
    assert_eq!(
        values(
            &backend,
            &authority,
            &sqrt_with_context_exact_native(&backend, &roots, &execution)?,
            &cancellation
        )?,
        [2.0, 3.0]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &sqrt_vjp_with_context_exact_native(
                &backend,
                &roots,
                &upload_f32(&backend, &authority, &[2], &[1.0, 1.0], &cancellation)?,
                &execution,
            )?,
            &cancellation,
        )?,
        [0.25, 1.0 / 6.0]
    );
    Ok(())
}

#[test]
fn item_count_nonzero_and_left_shift_preserve_scalar_and_integer_semantics()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let scalar = upload_i32(&backend, &authority, &[], &[7], &cancellation)?;
    assert_eq!(
        item_with_context_exact_native(&scalar, &execution)?,
        DecodedScalar::Signed(7)
    );
    let nonscalar = upload_i32(&backend, &authority, &[2], &[7, 8], &cancellation)?;
    assert!(item_with_context_exact_native(&nonscalar, &execution).is_err());

    let input = upload_i32(&backend, &authority, &[4], &[0, -2, 0, 9], &cancellation)?;
    assert_eq!(
        i64_scalar(&count_nonzero_with_context_exact_native(
            &backend, &input, &execution
        )?)?,
        2
    );
    let shifted = bitwise_left_shift_with_context_exact_native(
        &backend,
        &input,
        BitwiseShiftOperand::Scalar(3),
        &execution,
    )?;
    let decoded = (0..4)
        .map(|index| {
            shifted
                .descriptor()
                .dtype()
                .decode_scalar(shifted.element_bytes(&[index])?)
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(
        decoded,
        [
            DecodedScalar::Signed(0),
            DecodedScalar::Signed(-16),
            DecodedScalar::Signed(0),
            DecodedScalar::Signed(72),
        ]
    );
    Ok(())
}

#[test]
fn zero_and_constant_stage_one_atomic_canonical_fill() -> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let mut tensor = upload_f32(&backend, &authority, &[3], &[1.0, 2.0, 3.0], &cancellation)?;
    let original = tensor.clone();
    constant_in_place_with_context_exact_native(
        &backend,
        &mut tensor,
        Scalar::Float(4.5),
        &execution,
    )?;
    assert_eq!(
        values(&backend, &authority, &tensor, &cancellation)?,
        [4.5; 3]
    );
    assert_eq!(
        values(&backend, &authority, &original, &cancellation)?,
        [1.0, 2.0, 3.0]
    );
    zero_in_place_with_context_exact_native(&backend, &mut tensor, &execution)?;
    assert_eq!(
        values(&backend, &authority, &tensor, &cancellation)?,
        [0.0; 3]
    );

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_execution = context(&backend, &authority, &cancelled)?;
    let snapshot = tensor.contiguous_bytes()?.to_vec();
    assert!(
        constant_in_place_with_context_exact_native(
            &backend,
            &mut tensor,
            Scalar::Float(9.0),
            &cancelled_execution,
        )
        .is_err()
    );
    assert_eq!(tensor.contiguous_bytes()?, snapshot);
    Ok(())
}

#[test]
fn minimum_broadcasts_propagates_nan_and_splits_tie_gradients()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let execution = context(&backend, &authority, &cancellation)?;
    let input = upload_f32(
        &backend,
        &authority,
        &[3],
        &[1.0, 2.0, f32::NAN],
        &cancellation,
    )?;
    let other = upload_f32(&backend, &authority, &[3], &[2.0, 2.0, 3.0], &cancellation)?;
    let output = minimum_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Tensor(&other),
        &execution,
    )?;
    let output_values = values(&backend, &authority, &output, &cancellation)?;
    assert_eq!(&output_values[..2], &[1.0, 2.0]);
    assert!(output_values[2].is_nan());
    let gradient = upload_f32(&backend, &authority, &[3], &[1.0; 3], &cancellation)?;
    let vjp = minimum_vjp_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Tensor(&other),
        &gradient,
        &execution,
    )?;
    assert_eq!(
        values(&backend, &authority, &vjp.input, &cancellation)?[..2],
        [1.0, 0.5]
    );
    assert_eq!(
        values(
            &backend,
            &authority,
            &vjp.other.ok_or("missing minimum gradient")?,
            &cancellation
        )?[..2],
        [0.0, 0.5]
    );
    let minimum_tangent = minimum_jvp_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Tensor(&other),
        &upload_f32(&backend, &authority, &[3], &[2.0, 4.0, 8.0], &cancellation)?,
        Some(&upload_f32(
            &backend,
            &authority,
            &[3],
            &[1.0, 2.0, 3.0],
            &cancellation,
        )?),
        &execution,
    )?;
    assert_eq!(
        values(&backend, &authority, &minimum_tangent, &cancellation)?,
        [2.0, 3.0, 0.0]
    );
    Ok(())
}

#[test]
fn workspace_authority_is_exact_bounded_and_converges_after_failures()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let cancellation = CancellationToken::default();
    let input = upload_f32(
        &backend,
        &authority,
        &[4],
        &[2.0, 4.0, 8.0, 16.0],
        &cancellation,
    )?;
    let tangent = upload_f32(&backend, &authority, &[4], &[1.0; 4], &cancellation)?;

    let exact = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(4 * std::mem::size_of::<f32>() as u64)?,
        &cancellation,
    );
    div_jvp_with_context_exact_native(
        &backend,
        &input,
        ElementwiseOperand::Scalar(Scalar::Float(2.0)),
        &tangent,
        None,
        &exact,
    )?;
    assert_eq!(exact.scratch.peak_bytes(), 16);
    assert_eq!(exact.scratch.in_use_bytes(), 0);

    let insufficient = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(15)?,
        &cancellation,
    );
    assert!(matches!(
        div_jvp_with_context_exact_native(
            &backend,
            &input,
            ElementwiseOperand::Scalar(Scalar::Float(2.0)),
            &tangent,
            None,
            &insufficient,
        ),
        Err(sim_tensor::generated_elementwise_or_runtime_operation_05::ElementwiseRuntimePartFiveError::Tensor(
            TensorError::WorkspaceAuthorizationExceeded { .. }
        ))
    ));
    assert_eq!(insufficient.scratch.in_use_bytes(), 0);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(16)?,
        &cancelled,
    );
    assert!(
        div_jvp_with_context_exact_native(
            &backend,
            &input,
            ElementwiseOperand::Scalar(Scalar::Float(2.0)),
            &tangent,
            None,
            &cancelled_context,
        )
        .is_err()
    );
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}

struct CudaAllocatorOwner {
    device: DeviceId,
}

impl CachedAllocationOwner for CudaAllocatorOwner {
    fn cache_device(&self) -> DeviceId {
        self.device
    }

    fn release_cached_allocations(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<u64, TensorError> {
        cancellation.check()?;
        Ok(0)
    }
}

#[test]
fn device_adapters_project_only_canonical_owner_state() -> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let cuda = DeviceId::new(DeviceKind::Cuda, 2);
    let cuda_matrix = BackendCapabilityMatrix::new(cuda, Vec::new(), Vec::new())?;
    let matrices = [cuda_matrix];
    assert_eq!(
        cuda_set_device_exact_native(&matrices, cuda, &cancellation)?.device(),
        cuda
    );
    let allocator = CudaAllocatorOwner { device: cuda };
    assert_eq!(
        cuda_get_allocator_backend_exact_native(&allocator, &cancellation)?,
        "zed-native-cuda-caching-v1"
    );

    let xpu = DeviceId::new(DeviceKind::Xpu, 1);
    let properties = NativeDeviceProperties::new(
        xpu,
        "Intel Arc Native Fixture",
        8 * 1024 * 1024,
        1,
        0,
        Some("xe-hpg".to_owned()),
        true,
    )?;
    let xpu_matrix = BackendCapabilityMatrix::new_with_properties(
        xpu,
        Vec::new(),
        Vec::new(),
        Some(properties),
    )?;
    assert_eq!(
        xpu_get_device_name_exact_native(&xpu_matrix, xpu, &cancellation)?,
        "Intel Arc Native Fixture"
    );
    assert!(cuda_set_device_exact_native(&matrices, DeviceId::CPU, &cancellation).is_err());
    let (backend, _authority) = backend()?;
    assert!(cuda_get_allocator_backend_exact_native(&backend, &cancellation).is_err());
    assert!(xpu_get_device_name_exact_native(&xpu_matrix, cuda, &cancellation).is_err());
    Ok(())
}

#[test]
fn every_task48_adapter_honors_pre_cancellation_before_validation_or_publication()
-> Result<(), Box<dyn std::error::Error>> {
    let (backend, authority) = backend()?;
    let live = CancellationToken::default();
    let input = upload_f32(&backend, &authority, &[2], &[1.0, 2.0], &live)?;
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = backend.execution_context(
        StreamId::DEFAULT,
        authority.authorize_workspace(0)?,
        &cancelled,
    );
    let scalar = ElementwiseOperand::Scalar(Scalar::Float(1.0));
    assert!(matches!(
        div_with_context_exact_native(&backend, &input, scalar, &cancelled_context),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    assert!(matches!(
        div_vjp_with_context_exact_native(&backend, &input, scalar, &input, &cancelled_context,),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    assert!(matches!(
        div_jvp_with_context_exact_native(
            &backend,
            &input,
            scalar,
            &input,
            Some(&input),
            &cancelled_context,
        ),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    assert!(matches!(
        item_with_context_exact_native(&input, &cancelled_context),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    assert!(matches!(
        sin_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    assert!(matches!(
        sin_vjp_with_context_exact_native(&backend, &input, &input, &cancelled_context),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    assert!(matches!(
        sin_jvp_with_context_exact_native(&backend, &input, &input, &cancelled_context),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    assert!(matches!(
        sqrt_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    assert!(matches!(
        sqrt_vjp_with_context_exact_native(&backend, &input, &input, &cancelled_context),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    assert!(matches!(
        sqrt_jvp_with_context_exact_native(&backend, &input, &input, &cancelled_context),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    assert!(matches!(
        bitwise_left_shift_with_context_exact_native(
            &backend,
            &input,
            BitwiseShiftOperand::Scalar(u32::MAX),
            &cancelled_context,
        ),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    assert!(matches!(
        count_nonzero_with_context_exact_native(&backend, &input, &cancelled_context),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    assert!(matches!(
        minimum_with_context_exact_native(&backend, &input, scalar, &cancelled_context),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    assert!(matches!(
        minimum_vjp_with_context_exact_native(&backend, &input, scalar, &input, &cancelled_context,),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    assert!(matches!(
        minimum_jvp_with_context_exact_native(
            &backend,
            &input,
            scalar,
            &input,
            Some(&input),
            &cancelled_context,
        ),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    let mut mutable = input;
    let storage = mutable.storage_id();
    let bytes = mutable.contiguous_bytes()?.to_vec();
    assert!(matches!(
        zero_in_place_with_context_exact_native(&backend, &mut mutable, &cancelled_context),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    assert!(matches!(
        constant_in_place_with_context_exact_native(
            &backend,
            &mut mutable,
            Scalar::Float(f64::NAN),
            &cancelled_context,
        ),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    assert_eq!(mutable.storage_id(), storage);
    assert_eq!(mutable.contiguous_bytes()?, bytes);
    assert!(matches!(
        cuda_get_allocator_backend_exact_native(&backend, &cancelled),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    assert!(matches!(
        cuda_set_device_exact_native(&[], DeviceId::CPU, &cancelled),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    let cpu_matrix = BackendCapabilityMatrix::for_native_device(DeviceId::CPU)?;
    assert!(matches!(
        xpu_get_device_name_exact_native(&cpu_matrix, DeviceId::CPU, &cancelled),
        Err(ElementwiseRuntimePartFiveError::Cancelled)
    ));
    assert_eq!(cancelled_context.scratch.peak_bytes(), 0);
    assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);
    Ok(())
}
