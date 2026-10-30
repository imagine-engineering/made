use sim_tensor::{
    BrownianTree, CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId,
    ExecutionContext, GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES, RngAlgorithm, RngError,
    RngProfileVersion, RngStream, RngStreamAddress, RetryRngPolicy, StreamId, Tensor,
    TensorDescriptor,
    generated_random_number_generation_01::{
        BROWNIAN_TREE_OPERATION_ID, GENERATOR_MANUAL_SEED_OPERATION_ID, GENERATOR_OPERATION_ID,
        MANUAL_SEED_OPERATION_ID, MULTINOMIAL_OPERATION_ID, NORMAL_INITIALIZER_OPERATION_ID,
        RAND_OPERATION_ID, RANDINT_OPERATION_ID, RANDN_LIKE_OPERATION_ID, RANDPERM_OPERATION_ID,
        RandomNumberGenerationPartOneError, SOBOL_ENGINE_OPERATION_ID,
        UNIFORM_INITIALIZER_OPERATION_ID, brownian_tree_exact_native,
        generator_exact_native, generator_manual_seed_exact_native, manual_seed_exact_native,
        multinomial_with_context_exact_native, normal_in_place_exact_native,
        rand_with_context_exact_native, randint_with_context_exact_native,
        randn_like_with_context_exact_native,
        randperm_with_context_exact_native, sobol_draw_with_context_exact_native,
        sobol_engine_exact_native, uniform_in_place_exact_native,
    },
};
use sim_types::DeviceKind;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, ops::Deref, path::Path};

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

fn address(phase: &str) -> Result<RngStreamAddress, Box<dyn std::error::Error>> {
    Ok(RngStreamAddress::new(
        "workflow",
        "attempt",
        "task-81",
        0,
        phase,
        0,
        0,
        RetryRngPolicy::Replay,
    )?)
}

fn stream(seed: u64, phase: &str) -> Result<RngStream, Box<dyn std::error::Error>> {
    Ok(RngStream::new(
        RngProfileVersion::V1,
        RngAlgorithm::Philox4x32_10,
        seed,
        address(phase)?,
    )?)
}

fn stream_for_device(
    seed: u64,
    phase: &str,
    device: DeviceId,
) -> Result<RngStream, Box<dyn std::error::Error>> {
    Ok(RngStream::new(
        RngProfileVersion::V2,
        RngAlgorithm::Philox4x32_10,
        seed,
        RngStreamAddress::for_device(
            "workflow",
            "attempt",
            "task-81",
            0,
            phase,
            0,
            0,
            RetryRngPolicy::Replay,
            device,
        )?,
    )?)
}

fn upload_f32(
    backend: &TestBackend,
    shape: &[u64],
    values: &[f32],
    cancellation: &CancellationToken,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
    Ok(backend
        .upload_f32(descriptor, values, &backend.execution(cancellation)?)?
        .0)
}

fn f32_values(tensor: &Tensor) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(tensor
        .contiguous_bytes()?
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

fn i64_values(tensor: &Tensor) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    Ok(tensor
        .contiguous_bytes()?
        .chunks_exact(8)
        .map(|bytes| {
            i64::from_ne_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])
        })
        .collect())
}

#[test]
fn generator_facades_reseed_the_single_canonical_stream()
-> Result<(), Box<dyn std::error::Error>> {
    let cancellation = CancellationToken::default();
    let generated = generator_exact_native(
        RngProfileVersion::V1,
        RngAlgorithm::Mt19937,
        7,
        address("generator")?,
        &cancellation,
    )?;
    assert_eq!(generated.seed(), 7);
    let reseeded = generator_manual_seed_exact_native(&generated, -1, &cancellation)?;
    assert_eq!(reseeded.seed(), u64::MAX - 1);
    assert_eq!(generated.seed(), 7);
    let manual = manual_seed_exact_native(
        RngProfileVersion::V1,
        RngAlgorithm::Mt19937,
        42,
        address("manual")?,
        &cancellation,
    )?;
    assert_eq!(manual.seed(), 42);
    let first = manual.begin(None)?.next_u32(&cancellation)?;
    let replay = manual.begin(None)?.next_u32(&cancellation)?;
    assert_eq!(first, replay);
    Ok(())
}

#[test]
fn multinomial_validates_rows_and_replays_transactionally()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let generator = stream(11, "multinomial")?;
    let first = multinomial_with_context_exact_native(
        &[0.0, 1.0, 3.0, 2.0, 0.0, 1.0],
        2,
        3,
        2,
        false,
        generator.begin(None)?,
        DeviceId::CPU,
        &context,
    )?;
    let replay = multinomial_with_context_exact_native(
        &[0.0, 1.0, 3.0, 2.0, 0.0, 1.0],
        2,
        3,
        2,
        false,
        generator.begin(None)?,
        DeviceId::CPU,
        &context,
    )?;
    assert_eq!(first.shape, [2, 2]);
    assert_eq!(first.indices, replay.indices);
    let mut second_row = first.indices[2..].to_vec();
    second_row.sort_unstable();
    assert_eq!(second_row, [0, 2]);
    assert!(matches!(
        multinomial_with_context_exact_native(
            &[1.0, f64::NAN],
            1,
            2,
            1,
            false,
            generator.begin(None)?,
            DeviceId::CPU,
            &context,
        ),
        Err(RandomNumberGenerationPartOneError::Invalid { .. })
    ));
    Ok(())
}

#[test]
fn initializers_are_copy_on_write_and_publish_only_after_success()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let mut tensor = upload_f32(&backend, &[4], &[9.0; 4], &cancellation)?;
    let alias = tensor.clone();
    let transaction = uniform_in_place_exact_native(
        &mut tensor,
        -2.0,
        2.0,
        stream(13, "uniform")?.begin(None)?,
        &cancellation,
    )?;
    assert!(f32_values(&tensor)?.iter().all(|value| (-2.0..2.0).contains(value)));
    assert_eq!(f32_values(&alias)?, [9.0; 4]);
    let checkpoint = transaction.commit();
    normal_in_place_exact_native(
        &mut tensor,
        3.0,
        0.0,
        stream(13, "uniform")?.begin(Some(checkpoint))?,
        &cancellation,
    )?;
    assert_eq!(f32_values(&tensor)?, [3.0; 4]);

    let before_invalid = tensor.contiguous_bytes()?.to_vec();
    assert!(matches!(
        uniform_in_place_exact_native(
            &mut tensor,
            2.0,
            -2.0,
            stream(13, "invalid")?.begin(None)?,
            &cancellation,
        ),
        Err(RandomNumberGenerationPartOneError::Invalid { .. })
    ));
    assert_eq!(tensor.contiguous_bytes()?, before_invalid);
    Ok(())
}

#[test]
fn sobol_and_brownian_state_are_deterministic_and_additive()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;
    let engine = sobol_engine_exact_native(3, false, 123, &cancellation)?;
    let (points, mut engine) = sobol_draw_with_context_exact_native(&backend, engine, 4, &context)?;
    assert_eq!(
        f32_values(&points)?,
        [
            0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.75, 0.25, 0.25, 0.25, 0.75, 0.75,
        ]
    );
    let continued = engine.draw(1, &cancellation)?;
    engine.reset();
    let replay = engine.draw(5, &cancellation)?;
    assert_eq!(continued, replay[12..]);
    let mut scrambled = sobol_engine_exact_native(3, true, 123, &cancellation)?;
    let scrambled_first = scrambled.draw(8, &cancellation)?;
    let mut scrambled_replay = sobol_engine_exact_native(3, true, 123, &cancellation)?;
    assert_eq!(scrambled_first, scrambled_replay.draw(8, &cancellation)?);
    assert_ne!(scrambled_first[..12], replay[..12]);

    let mut brownian = brownian_tree_exact_native(0.0, vec![0.0, 0.0], 1.0, 99, &cancellation)?;
    let left = brownian.increment(0.0, 0.5, &cancellation)?;
    let right = brownian.increment(0.5, 1.0, &cancellation)?;
    let whole = brownian.increment(0.0, 1.0, &cancellation)?;
    for ((left, right), whole) in left.iter().zip(right).zip(whole) {
        assert!((left + right - whole).abs() <= 1.0e-12);
    }
    let replay_tree = BrownianTree::new(0.0, vec![0.0, 0.0], 1.0, 99)?;
    assert_eq!(brownian.dimension(), replay_tree.dimension());
    Ok(())
}

#[test]
fn random_tensor_operations_preserve_shape_dtype_and_replay()
-> Result<(), Box<dyn std::error::Error>> {
    let backend = TestBackend::new()?;
    let cancellation = CancellationToken::default();
    let context = backend.execution(&cancellation)?;

    let generator = stream(17, "rand")?;
    let first = rand_with_context_exact_native(
        &backend,
        &[2, 3],
        DType::F32,
        generator.begin(None)?,
        &context,
    )?;
    let replay = rand_with_context_exact_native(
        &backend,
        &[2, 3],
        DType::F32,
        generator.begin(None)?,
        &context,
    )?;
    assert_eq!(first.tensor.descriptor().shape(), [2, 3]);
    assert_eq!(f32_values(&first.tensor)?, f32_values(&replay.tensor)?);
    assert!(f32_values(&first.tensor)?.iter().all(|value| (0.0..1.0).contains(value)));

    let integers = randint_with_context_exact_native(
        &backend,
        -3,
        5,
        &[64],
        stream(19, "randint")?.begin(None)?,
        &context,
    )?;
    assert!(i64_values(&integers.tensor)?
        .iter()
        .all(|value| (-3..5).contains(value)));

    let input = upload_f32(&backend, &[5], &[0.0; 5], &cancellation)?;
    let normal = randn_like_with_context_exact_native(
        &backend,
        &input,
        stream(23, "randn")?.begin(None)?,
        &context,
    )?;
    assert_eq!(normal.tensor.descriptor(), input.descriptor());
    assert!(f32_values(&normal.tensor)?.iter().all(|value| value.is_finite()));

    let cuda = DeviceId::new(DeviceKind::Cuda, 0);
    assert!(matches!(
        rand_with_context_exact_native(
            &backend,
            &[2],
            DType::F32,
            stream_for_device(23, "device-mismatch", cuda)?.begin(None)?,
            &context,
        ),
        Err(RandomNumberGenerationPartOneError::Rng(
            RngError::DeviceMismatch {
                expected: DeviceId::CPU,
                actual,
            }
        )) if actual == cuda
    ));

    let permutation = randperm_with_context_exact_native(
        &backend,
        128,
        stream(29, "randperm")?.begin(None)?,
        &context,
    )?;
    let mut values = i64_values(&permutation.tensor)?;
    values.sort_unstable();
    assert_eq!(values, (0..128).collect::<Vec<_>>());
    Ok(())
}

#[test]
fn cancellation_precedes_invalid_inputs_and_leaves_mutations_unchanged()
-> Result<(), Box<dyn std::error::Error>> {
    assert!(matches!(
        RandomNumberGenerationPartOneError::from(RngError::Cancelled),
        RandomNumberGenerationPartOneError::Cancelled
    ));
    let backend = TestBackend::new()?;
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let context = backend.execution(&cancelled)?;
    let mut input = upload_f32(&backend, &[2], &[1.0, 2.0], &CancellationToken::default())?;
    let before = input.contiguous_bytes()?.to_vec();
    assert!(matches!(
        normal_in_place_exact_native(
            &mut input,
            f64::NAN,
            -1.0,
            stream(31, "cancel-normal")?.begin(None)?,
            &cancelled,
        ),
        Err(RandomNumberGenerationPartOneError::Cancelled)
    ));
    assert_eq!(input.contiguous_bytes()?, before);
    assert!(matches!(
        randint_with_context_exact_native(
            &backend,
            4,
            4,
            &[u64::MAX],
            stream(31, "cancel-randint")?.begin(None)?,
            &context,
        ),
        Err(RandomNumberGenerationPartOneError::Cancelled)
    ));
    assert!(matches!(
        generator_exact_native(
            RngProfileVersion::V1,
            RngAlgorithm::Philox4x32_10,
            0,
            address("cancel-generator")?,
            &cancelled,
        ),
        Err(RandomNumberGenerationPartOneError::Cancelled)
    ));
    Ok(())
}

#[test]
fn operation_contracts_are_unique_and_evidence_is_exact()
-> Result<(), Box<dyn std::error::Error>> {
    let expected = BTreeSet::from([
        GENERATOR_OPERATION_ID,
        GENERATOR_MANUAL_SEED_OPERATION_ID,
        MANUAL_SEED_OPERATION_ID,
        MULTINOMIAL_OPERATION_ID,
        NORMAL_INITIALIZER_OPERATION_ID,
        UNIFORM_INITIALIZER_OPERATION_ID,
        SOBOL_ENGINE_OPERATION_ID,
        RAND_OPERATION_ID,
        RANDINT_OPERATION_ID,
        RANDN_LIKE_OPERATION_ID,
        RANDPERM_OPERATION_ID,
        BROWNIAN_TREE_OPERATION_ID,
    ]);
    let contracts = GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .flat_map(|contracts| contracts.iter())
        .filter(|contract| expected.contains(contract.operation_id))
        .collect::<Vec<_>>();
    assert_eq!(contracts.len(), expected.len());
    assert_eq!(
        contracts
            .iter()
            .map(|contract| contract.operation_id)
            .collect::<BTreeSet<_>>(),
        expected
    );
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for contract in contracts {
        let fixture = fs::read(workspace.join(contract.evidence_fixture))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(fixture)),
            contract.evidence_fixture_sha256
        );
    }
    Ok(())
}
