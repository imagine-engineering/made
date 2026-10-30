use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DecodedScalar, DeviceId,
    ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, Layout, MemoryFormatReference,
    StreamId, Tensor, TensorDescriptor, ViewAccess,
    generated_storage_dtype_device_01::{
        CLONE_OPERATION_ID, CONTIGUOUS_OPERATION_ID, COPY_OPERATION_ID, CPU_OPERATION_ID,
        CUDA_OPERATION_ID, FLOAT_OPERATION_ID, HALF_OPERATION_ID, NUMPY_OPERATION_ID,
        StorageDTypeDeviceError, TO_OPERATION_ID, TYPE_AS_OPERATION_ID, TYPE_OPERATION_ID,
        TensorTypeRequest, TensorTypeResult, cast_jvp_with_context_exact_native,
        cast_vjp_with_context_exact_native, clone_with_context_exact_native,
        contiguous_with_context_exact_native, copy_with_context_exact_native,
        cpu_with_context_exact_native, cuda_with_context_exact_native,
        float_with_context_exact_native, half_with_context_exact_native,
        identity_vjp_with_context_exact_native, numpy_exact_native,
        tensor_type_with_context_exact_native, to_with_context_exact_native,
        type_as_with_context_exact_native,
    },
};
use sim_types::DeviceKind;
use std::{collections::BTreeSet, fs, path::Path};

const IDS: [&str; 11] = [
    CLONE_OPERATION_ID,
    CONTIGUOUS_OPERATION_ID,
    COPY_OPERATION_ID,
    CPU_OPERATION_ID,
    CUDA_OPERATION_ID,
    FLOAT_OPERATION_ID,
    HALF_OPERATION_ID,
    NUMPY_OPERATION_ID,
    TO_OPERATION_ID,
    TYPE_OPERATION_ID,
    TYPE_AS_OPERATION_ID,
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

    fn context<'a>(
        &self,
        cancellation: &'a CancellationToken,
    ) -> Result<ExecutionContext<'a>, Box<dyn std::error::Error>> {
        self.context_with_scratch(cancellation, self.limit)
    }

    fn context_with_scratch<'a>(
        &self,
        cancellation: &'a CancellationToken,
        bytes: u64,
    ) -> Result<ExecutionContext<'a>, Box<dyn std::error::Error>> {
        Ok(self.backend.execution_context(
            StreamId::DEFAULT,
            self.authority.authorize_workspace(bytes)?,
            cancellation,
        ))
    }

    fn tensor(
        &self,
        shape: &[u64],
        values: &[f32],
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let descriptor = TensorDescriptor::contiguous(
            shape.to_vec(),
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        Ok(self.backend.upload_f32(descriptor, values, context)?.0)
    }
}

fn real_values(tensor: &Tensor) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let count = tensor.descriptor().element_count()?;
    let mut values = Vec::with_capacity(usize::try_from(count)?);
    for index in 0..count {
        let value = tensor
            .descriptor()
            .dtype()
            .decode_scalar(tensor.linear_element_bytes(index)?)?;
        values.push(match value {
            DecodedScalar::Real(value) => value,
            DecodedScalar::Signed(value) => value as f64,
            DecodedScalar::Unsigned(value) => value as f64,
            DecodedScalar::Boolean(value) => f64::from(value),
            DecodedScalar::Complex { .. } => return Err("expected a real tensor".into()),
        });
    }
    Ok(values)
}

#[test]
fn clone_is_deep_contiguous_is_conditional_and_formats_are_descriptor_owned()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[2, 3], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &context)?;

    let cloned = clone_with_context_exact_native(
        &test.backend,
        &input,
        MemoryFormatReference::PreserveFormat,
        &context,
    )?;
    assert_ne!(cloned.storage_id(), input.storage_id());
    assert_eq!(cloned.descriptor(), input.descriptor());
    assert_eq!(real_values(&cloned)?, real_values(&input)?);

    let already = contiguous_with_context_exact_native(
        &test.backend,
        &input,
        MemoryFormatReference::Layout(Layout::Contiguous),
        &context,
    )?;
    assert_eq!(already.storage_id(), input.storage_id());

    let transposed_descriptor = input.descriptor().permuted_view(&[1, 0])?;
    let transposed = input.view(transposed_descriptor, ViewAccess::ReadOnly)?;
    let contiguous = contiguous_with_context_exact_native(
        &test.backend,
        &transposed,
        MemoryFormatReference::Layout(Layout::Contiguous),
        &context,
    )?;
    assert_ne!(contiguous.storage_id(), input.storage_id());
    assert!(contiguous.descriptor().is_contiguous()?);
    assert_eq!(
        real_values(&contiguous)?,
        vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
    );
    Ok(())
}

#[test]
fn copy_broadcasts_converts_and_commits_only_after_complete_staging()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let source = test.tensor(&[1, 3], &[1.25, -2.5, 3.75], &context)?;
    let mut destination = test.tensor(&[2, 3], &[9.0; 6], &context)?;
    let previous_version = destination.storage_version();
    let returned =
        copy_with_context_exact_native(&test.backend, &mut destination, &source, true, &context)?;
    assert_eq!(
        real_values(&destination)?,
        vec![1.25, -2.5, 3.75, 1.25, -2.5, 3.75]
    );
    assert_eq!(returned.storage_id(), destination.storage_id());
    assert_eq!(destination.storage_version(), previous_version + 1);

    let mut insufficient = test.tensor(&[2, 3], &[7.0; 6], &context)?;
    let insufficient_context = test.context_with_scratch(&cancellation, 23)?;
    assert!(matches!(
        copy_with_context_exact_native(
            &test.backend,
            &mut insufficient,
            &source,
            false,
            &insufficient_context,
        ),
        Err(StorageDTypeDeviceError::Tensor(
            sim_tensor::TensorError::WorkspaceAuthorizationExceeded { .. }
        ))
    ));
    assert_eq!(real_values(&insufficient)?, vec![7.0; 6]);
    assert_eq!(insufficient.storage_version(), 1);

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = test.context(&cancelled)?;
    assert!(matches!(
        copy_with_context_exact_native(
            &test.backend,
            &mut insufficient,
            &source,
            false,
            &cancelled_context,
        ),
        Err(StorageDTypeDeviceError::Cancelled {
            operation: COPY_OPERATION_ID
        })
    ));
    assert_eq!(real_values(&insufficient)?, vec![7.0; 6]);
    Ok(())
}

#[test]
fn to_cpu_float_half_type_and_type_as_share_the_canonical_cast_adapter()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[3], &[1.25, -2.5, 3.75], &context)?;

    let cpu = cpu_with_context_exact_native(
        &test.backend,
        &input,
        MemoryFormatReference::PreserveFormat,
        &context,
    )?;
    assert_eq!(cpu.storage_id(), input.storage_id());
    let half = half_with_context_exact_native(
        &test.backend,
        &input,
        MemoryFormatReference::PreserveFormat,
        &context,
    )?;
    assert_eq!(half.descriptor().dtype(), DType::F16);
    let float = float_with_context_exact_native(
        &test.backend,
        &half,
        MemoryFormatReference::PreserveFormat,
        &context,
    )?;
    assert_eq!(float.descriptor().dtype(), DType::F32);
    assert_eq!(real_values(&float)?, vec![1.25, -2.5, 3.75]);

    let same = to_with_context_exact_native(
        &test.backend,
        &input,
        None,
        None,
        false,
        false,
        None,
        &context,
    )?;
    assert_eq!(same.storage_id(), input.storage_id());
    let copied = to_with_context_exact_native(
        &test.backend,
        &input,
        None,
        None,
        false,
        true,
        None,
        &context,
    )?;
    assert_ne!(copied.storage_id(), input.storage_id());

    assert!(matches!(
        tensor_type_with_context_exact_native(
            &test.backend,
            &input,
            TensorTypeRequest::Query,
            &context,
        )?,
        TensorTypeResult::Name("torch.FloatTensor")
    ));
    let TensorTypeResult::Tensor(typed) = tensor_type_with_context_exact_native(
        &test.backend,
        &input,
        TensorTypeRequest::Convert(DType::F64),
        &context,
    )?
    else {
        return Err("type conversion did not return a tensor".into());
    };
    assert_eq!(typed.descriptor().dtype(), DType::F64);
    let type_as = type_as_with_context_exact_native(&test.backend, &input, &half, &context)?;
    assert_eq!(type_as.descriptor().dtype(), DType::F16);

    assert!(matches!(
        cuda_with_context_exact_native(
            &input,
            Some(2),
            false,
            MemoryFormatReference::PreserveFormat,
            &cancellation,
        ),
        Err(StorageDTypeDeviceError::UnsupportedDevice {
            operation: CUDA_OPERATION_ID,
            device,
        }) if device == DeviceId::new(DeviceKind::Cuda, 2)
    ));
    Ok(())
}

#[test]
fn numpy_is_an_immutable_native_borrow_with_exact_strided_metadata()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[2, 3], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &context)?;
    let transposed = input.view(
        input.descriptor().permuted_view(&[1, 0])?,
        ViewAccess::ReadOnly,
    )?;
    let array = numpy_exact_native(&transposed, &cancellation)?;
    assert_eq!(array.shape(), &[3, 2]);
    assert_eq!(array.dtype(), DType::F32);
    assert_eq!(array.rank(), 2);
    assert_eq!(array.stride_bytes(0)?, 4);
    assert_eq!(array.stride_bytes(1)?, 12);
    assert_eq!(array.offset_bytes()?, 0);
    assert_eq!(array.storage_bytes()?.len(), 24);
    assert_eq!(array.element_bytes(&[2, 1])?, 5.0_f32.to_ne_bytes());
    Ok(())
}

#[test]
fn analytical_identity_and_cast_maps_preserve_shape_layout_and_dtype()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let primal = test.tensor(&[2], &[1.0, 2.0], &context)?;
    let gradient = test.tensor(&[2], &[3.0, 4.0], &context)?;
    let identity =
        identity_vjp_with_context_exact_native(&test.backend, &primal, &gradient, &context)?;
    assert_eq!(identity.descriptor(), primal.descriptor());
    assert_eq!(real_values(&identity)?, vec![3.0, 4.0]);

    let half_tangent = cast_jvp_with_context_exact_native(
        &test.backend,
        &gradient,
        DType::F16,
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(half_tangent.descriptor().dtype(), DType::F16);
    let pulled_back =
        cast_vjp_with_context_exact_native(&test.backend, &primal, &half_tangent, &context)?;
    assert_eq!(pulled_back.descriptor().dtype(), DType::F32);
    assert_eq!(real_values(&pulled_back)?, vec![3.0, 4.0]);

    let integer_descriptor =
        TensorDescriptor::contiguous(vec![2], DType::I32, DeviceId::CPU, StreamId::DEFAULT)?;
    let integer = test
        .backend
        .upload_bytes(
            integer_descriptor,
            &[1_i32.to_ne_bytes(), 2_i32.to_ne_bytes()].concat(),
            &context,
        )?
        .0;
    assert!(matches!(
        cast_jvp_with_context_exact_native(
            &test.backend,
            &integer,
            DType::F32,
            DeviceId::CPU,
            &context,
        ),
        Err(StorageDTypeDeviceError::NonDifferentiable {
            dtype: DType::I32,
            ..
        })
    ));
    Ok(())
}

#[test]
fn invalid_formats_broadcasts_and_cancellation_have_typed_precedence()
-> Result<(), Box<dyn std::error::Error>> {
    let test = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = test.context(&cancellation)?;
    let input = test.tensor(&[2], &[1.0, 2.0], &context)?;
    assert!(matches!(
        contiguous_with_context_exact_native(
            &test.backend,
            &input,
            MemoryFormatReference::Layout(Layout::Strided),
            &context,
        ),
        Err(StorageDTypeDeviceError::Invalid { .. })
    ));
    let mut destination = test.tensor(&[2], &[0.0, 0.0], &context)?;
    let source = test.tensor(&[3], &[1.0, 2.0, 3.0], &context)?;
    assert!(matches!(
        copy_with_context_exact_native(&test.backend, &mut destination, &source, false, &context,),
        Err(StorageDTypeDeviceError::Invalid {
            operation: COPY_OPERATION_ID,
            ..
        })
    ));

    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_context = test.context(&cancelled)?;
    assert!(matches!(
        contiguous_with_context_exact_native(
            &test.backend,
            &input,
            MemoryFormatReference::Layout(Layout::Strided),
            &cancelled_context,
        ),
        Err(StorageDTypeDeviceError::Cancelled {
            operation: CONTIGUOUS_OPERATION_ID
        })
    ));
    assert!(matches!(
        copy_with_context_exact_native(
            &test.backend,
            &mut destination,
            &source,
            true,
            &cancelled_context,
        ),
        Err(StorageDTypeDeviceError::Cancelled {
            operation: COPY_OPERATION_ID
        })
    ));
    assert!(matches!(
        identity_vjp_with_context_exact_native(&test.backend, &input, &source, &cancelled_context,),
        Err(StorageDTypeDeviceError::Cancelled {
            operation: CLONE_OPERATION_ID
        })
    ));
    assert!(matches!(
        cast_vjp_with_context_exact_native(&test.backend, &input, &source, &cancelled_context,),
        Err(StorageDTypeDeviceError::Cancelled {
            operation: TO_OPERATION_ID
        })
    ));

    let integer_descriptor =
        TensorDescriptor::contiguous(vec![2], DType::I32, DeviceId::CPU, StreamId::DEFAULT)?;
    let integer = test
        .backend
        .upload_bytes(
            integer_descriptor,
            &[1_i32.to_ne_bytes(), 2_i32.to_ne_bytes()].concat(),
            &context,
        )?
        .0;
    assert!(matches!(
        cast_jvp_with_context_exact_native(
            &test.backend,
            &integer,
            DType::F32,
            DeviceId::CPU,
            &cancelled_context,
        ),
        Err(StorageDTypeDeviceError::Cancelled {
            operation: TO_OPERATION_ID
        })
    ));
    Ok(())
}

#[test]
fn authoritative_owners_are_reused_without_competing_foundations()
-> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("workspace root unavailable")?;
    let source =
        fs::read_to_string(root.join("crates/sim_tensor/src/ops/storage_dtype_device_01.rs"))?;
    let tensor = fs::read_to_string(root.join("crates/sim_tensor/src/sim_tensor.rs"))?;
    let dtypes = fs::read_to_string(root.join("crates/sim_tensor/src/dtypes.rs"))?;
    let cast =
        fs::read_to_string(root.join("crates/sim_tensor/src/ops/sim_operator_indirection_01.rs"))?;
    assert_eq!(
        tensor.matches(concat!("pub struct ", "Tensor {")).count(),
        1
    );
    assert_eq!(tensor.matches(concat!("struct ", "Storage {")).count(), 1);
    assert_eq!(dtypes.matches(concat!("pub enum ", "DType {")).count(), 1);
    assert_eq!(
        cast.matches(concat!("pub fn ", "cast_to_with_context_exact_native("))
            .count(),
        1
    );
    assert!(source.contains("cast_to_with_context_exact_native("));
    assert!(source.matches(".copy(").count() >= 4);
    assert!(source.contains("descriptor().preserving_format_for("));
    assert!(!source.contains(concat!("pub struct ", "Storage {")));
    assert!(!source.contains(concat!("pub enum ", "DType {")));
    assert!(!source.contains(concat!("pub struct ", "DeviceId {")));
    assert!(!source.contains(concat!("pub struct Cancellation", "Token")));
    Ok(())
}

#[test]
fn all_eleven_resolutions_are_unique_and_runtime_hash_sealed()
-> Result<(), Box<dyn std::error::Error>> {
    let contracts = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .flat_map(|slice| slice.iter())
        .filter(|contract| contract.resolution_module == "storage_dtype_device_01")
        .collect::<Vec<_>>();
    assert_eq!(contracts.len(), IDS.len());
    assert_eq!(
        contracts
            .iter()
            .map(|contract| contract.operation_id)
            .collect::<BTreeSet<_>>(),
        IDS.into_iter().collect()
    );
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
            "comfy-parity-tensor-ops-storage-dtype-device-comfy-tensor-op-00f639d6c8a7"
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
