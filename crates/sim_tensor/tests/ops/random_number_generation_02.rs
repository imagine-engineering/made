use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId,
    ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, Layout, RngAlgorithm,
    RngError, RngProfileVersion, RngStream, RngStreamAddress, RetryRngPolicy, StreamId,
    TensorDescriptor,
    generated_random_number_generation_01::{
        RandomNumberGenerationPartOneError, randn_like_with_context_exact_native,
    },
    generated_random_number_generation_02::{
        RANDN_OPERATION_ID, RandomNumberGenerationPartTwoError,
        randn_with_context_exact_native,
    },
};
use sim_types::DeviceKind;
use sha2::{Digest, Sha256};
use std::{fs, ops::Deref, path::Path};

struct TestBackend {
    backend: CpuBackend,
    authority: CpuWorkspaceAuthority,
}

impl TestBackend {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
        Ok(Self { backend, authority })
    }

    fn execution<'a>(
        &self,
        cancellation: &'a CancellationToken,
    ) -> Result<ExecutionContext<'a>, Box<dyn std::error::Error>> {
        Ok(self.backend.execution_context(
            StreamId::DEFAULT,
            self.authority.authorize_workspace(16 * 1024 * 1024)?,
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

fn stream(
    seed: u64,
    phase: &str,
    device: DeviceId,
) -> Result<RngStream, Box<dyn std::error::Error>> {
    let address = RngStreamAddress::for_device(
        "workflow",
        "attempt",
        "task-82",
        0,
        phase,
        0,
        0,
        RetryRngPolicy::Replay,
        device,
    )?;
    Ok(RngStream::new(
        RngProfileVersion::V2,
        RngAlgorithm::Philox4x32_10,
        seed,
        address,
    )?)
}

fn f32_values(tensor: &sim_tensor::Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    tensor
        .contiguous_bytes()?
        .chunks_exact(4)
        .map(|bytes| {
            let bytes: [u8; 4] = bytes.try_into()?;
            Ok(f32::from_ne_bytes(bytes))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()
}

#[test]
fn randn_replays_advances_and_reuses_the_canonical_normal_transform()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let generator = stream(41, "forward", DeviceId::CPU)?;
    let first = randn_with_context_exact_native(
        &backend,
        &[2, 3],
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        generator.begin(None)?,
        &context,
    )?;
    let replay = randn_with_context_exact_native(
        &backend,
        &[2, 3],
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        generator.begin(None)?,
        &context,
    )?;
    assert_eq!(first.tensor.contiguous_bytes()?, replay.tensor.contiguous_bytes()?);
    assert_eq!(first.tensor.descriptor().shape(), [2, 3]);
    assert_eq!(first.tensor.descriptor().layout(), Layout::Contiguous);
    let values = f32_values(&first.tensor)?;
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().any(|value| *value != 0.0));

    let checkpoint = first.transaction.commit();
    let advanced = randn_with_context_exact_native(
        &backend,
        &[2, 3],
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        generator.begin(Some(checkpoint))?,
        &context,
    )?;
    assert_ne!(advanced.tensor.contiguous_bytes()?, replay.tensor.contiguous_bytes()?);

    let descriptor = TensorDescriptor::contiguous(
        vec![5],
        DType::F32,
        DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    let input = backend
        .upload_f32(descriptor, &[0.0; 5], &context)?
        .0;
    let odd_shape = stream(43, "shared-transform", DeviceId::CPU)?;
    let randn = randn_with_context_exact_native(
        &backend,
        &[5],
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        odd_shape.begin(None)?,
        &context,
    )?;
    let randn_like =
        randn_like_with_context_exact_native(&backend, &input, odd_shape.begin(None)?, &context)?;
    assert_eq!(randn.tensor.contiguous_bytes()?, randn_like.tensor.contiguous_bytes()?);
    Ok(())
}

#[test]
fn randn_supports_source_used_dtypes_and_zero_sized_shapes()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let generator = stream(47, "dtype", DeviceId::CPU)?;
    for dtype in [DType::F64, DType::F32, DType::F16, DType::Bf16] {
        let output = randn_with_context_exact_native(
            &backend,
            &[2, 3],
            dtype,
            Layout::Strided,
            DeviceId::CPU,
            generator.begin(None)?,
            &context,
        )?;
        assert_eq!(output.tensor.descriptor().dtype(), dtype);
        assert_eq!(output.tensor.descriptor().shape(), [2, 3]);
    }

    let empty_stream = stream(53, "empty", DeviceId::CPU)?;
    let initial_checkpoint = empty_stream.begin(None)?.commit();
    let empty = randn_with_context_exact_native(
        &backend,
        &[2, 0, 3],
        DType::F32,
        Layout::Strided,
        DeviceId::CPU,
        empty_stream.begin(None)?,
        &context,
    )?;
    assert_eq!(empty.tensor.descriptor().shape(), [2, 0, 3]);
    assert!(empty.tensor.contiguous_bytes()?.is_empty());
    assert_eq!(empty.transaction.commit(), initial_checkpoint);
    Ok(())
}

#[test]
fn randn_rejects_layout_dtype_and_rng_device_mismatches()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let cpu = stream(59, "invalid", DeviceId::CPU)?;
    assert!(matches!(
        randn_with_context_exact_native(
            &backend,
            &[2],
            DType::F32,
            Layout::Contiguous,
            DeviceId::CPU,
            cpu.begin(None)?,
            &context,
        ),
        Err(RandomNumberGenerationPartTwoError::UnsupportedLayout { .. })
    ));
    assert!(matches!(
        randn_with_context_exact_native(
            &backend,
            &[2],
            DType::I64,
            Layout::Strided,
            DeviceId::CPU,
            cpu.begin(None)?,
            &context,
        ),
        Err(RandomNumberGenerationPartTwoError::Canonical(
            RandomNumberGenerationPartOneError::UnsupportedDType { operation, .. }
        )) if operation == RANDN_OPERATION_ID
    ));

    let cuda = DeviceId::new(DeviceKind::Cuda, 0);
    assert!(matches!(
        randn_with_context_exact_native(
            &backend,
            &[2],
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            stream(59, "mismatch", cuda)?.begin(None)?,
            &context,
        ),
        Err(RandomNumberGenerationPartTwoError::Canonical(
            RandomNumberGenerationPartOneError::Rng(RngError::DeviceMismatch {
                expected: DeviceId::CPU,
                actual,
            })
        )) if actual == cuda
    ));
    assert!(matches!(
        randn_with_context_exact_native(
            &backend,
            &[2],
            DType::F32,
            Layout::Strided,
            cuda,
            cpu.begin(None)?,
            &context,
        ),
        Err(RandomNumberGenerationPartTwoError::UnsupportedDevice { device, .. }) if device == cuda
    ));
    Ok(())
}

#[test]
fn cancellation_precedes_every_invalid_randn_argument()
-> Result<(), Box<dyn std::error::Error>> {
    assert!(matches!(
        RandomNumberGenerationPartTwoError::from(RandomNumberGenerationPartOneError::from(
            RngError::Cancelled,
        )),
        RandomNumberGenerationPartTwoError::Cancelled
    ));
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let context = backend.execution(&cancellation)?;
    let cuda = DeviceId::new(DeviceKind::Cuda, 0);
    assert!(matches!(
        randn_with_context_exact_native(
            &backend,
            &[u64::MAX, 2],
            DType::I64,
            Layout::ChannelsLast,
            cuda,
            stream(61, "cancelled", cuda)?.begin(None)?,
            &context,
        ),
        Err(RandomNumberGenerationPartTwoError::Cancelled)
    ));
    Ok(())
}

#[test]
fn randn_resolution_is_unique_source_profiled_and_hash_sealed()
-> Result<(), Box<dyn std::error::Error>> {
    let contracts = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .flat_map(|contracts| contracts.iter())
        .filter(|contract| contract.operation_id == RANDN_OPERATION_ID)
        .collect::<Vec<_>>();
    assert_eq!(contracts.len(), 1);
    let contract = contracts
        .first()
        .ok_or("randn resolution was not generated")?;
    assert_eq!(contract.resolution_module, "random_number_generation_02");
    assert_eq!(
        contract.owner_task_id,
        "comfy-parity-tensor-ops-random-number-generation-comfy-tensor-op-fd729b8a5363"
    );
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = fs::read(workspace.join(contract.evidence_fixture))?;
    assert_eq!(
        format!("{:x}", Sha256::digest(&fixture)),
        contract.evidence_fixture_sha256
    );
    let evidence: serde_json::Value = serde_json::from_slice(&fixture)?;
    assert_eq!(evidence["source_profile"]["dependency"], "pytorch");
    assert_eq!(
        evidence["source_profile"]["fingerprint_sha256"],
        "48f4835af39b753fb2e637ec17813716024e08952e82e6e4e536a0fcfd944d0e"
    );
    assert_eq!(
        evidence["source_observations"]
            .as_array()
            .map(Vec::len),
        Some(5)
    );
    Ok(())
}
