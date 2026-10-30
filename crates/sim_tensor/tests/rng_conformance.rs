use sim_tensor::{
    BrownianTree, CancellationToken, CompatibilityRngTransaction, DeviceId, Mt19937,
    NativeRngExecutionProfile, Philox4x32, RNG_COMPATIBILITY_CONTRACTS, RetryRngPolicy,
    RngAlgorithm, RngCheckpoint, RngCompatibilityError, RngCompatibilityOperation,
    RngCompatibilityRequest, RngContractAvailability, RngError, RngExecutionScope,
    RngGenerationPlacement, RngProfileVersion, RngSeedTransform, RngStream, RngStreamAddress,
};
use sim_types::DeviceKind;
use std::{collections::BTreeMap, error::Error, fs, path::Path};

fn request(
    phase_seed: i128,
    batch: u64,
    retry: u32,
    retry_policy: RetryRngPolicy,
    ordinal: u64,
    placement: RngGenerationPlacement,
    scope: RngExecutionScope,
) -> RngCompatibilityRequest {
    request_with_transform(
        phase_seed,
        batch,
        retry,
        retry_policy,
        ordinal,
        RngSeedTransform::TorchSigned64,
        placement,
        scope,
    )
}

#[allow(clippy::too_many_arguments)]
fn request_with_transform(
    phase_seed: i128,
    batch: u64,
    retry: u32,
    retry_policy: RetryRngPolicy,
    ordinal: u64,
    seed_transform: RngSeedTransform,
    placement: RngGenerationPlacement,
    scope: RngExecutionScope,
) -> RngCompatibilityRequest {
    RngCompatibilityRequest::new(
        "workflow",
        "attempt",
        "node",
        0,
        ordinal,
        batch,
        retry,
        retry_policy,
        phase_seed,
        seed_transform,
        placement,
        scope,
    )
}

fn cpu_request(seed: i128) -> RngCompatibilityRequest {
    request(
        seed,
        0,
        0,
        RetryRngPolicy::Replay,
        0,
        RngGenerationPlacement::Native(DeviceId::CPU),
        RngExecutionScope::Production,
    )
}

fn open_cpu(
    rng_id: &str,
    seed: i128,
) -> Result<CompatibilityRngTransaction, RngCompatibilityError> {
    CompatibilityRngTransaction::open(
        rng_id,
        cpu_request(seed),
        None,
        &CancellationToken::default(),
    )
}

fn parse_csv_record(record: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut characters = record.chars().peekable();
    let mut quoted = false;
    while let Some(character) = characters.next() {
        match character {
            '"' if quoted && characters.peek() == Some(&'"') => {
                field.push('"');
                characters.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(std::mem::take(&mut field));
            }
            character => field.push(character),
        }
    }
    if quoted {
        return Err("unterminated CSV field".into());
    }
    fields.push(field);
    Ok(fields)
}

#[test]
fn all_54_catalog_rows_have_exact_typed_contracts() -> Result<(), Box<dyn Error>> {
    let catalog_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/sim_test_support/parity/catalogs/backend-rng.csv");
    let catalog = fs::read_to_string(catalog_path)?;
    let mut lines = catalog.lines();
    let header = lines.next().ok_or("RNG catalog is empty")?;
    let columns = parse_csv_record(header)?;
    let column = |name: &str| {
        columns
            .iter()
            .position(|candidate| candidate == name)
            .ok_or_else(|| format!("RNG catalog column {name} is missing"))
    };
    let rng_id = column("rng_id")?;
    let phase = column("phase")?;
    let symbol = column("symbol")?;
    let seededness = column("seededness")?;
    let seeds = column("seed_expressions")?;
    let generators = column("generator_expressions")?;
    let devices = column("device_expressions")?;
    let availability = column("availability")?;
    let mut catalog_rows = BTreeMap::new();
    for line in lines {
        let fields = parse_csv_record(line)?;
        let id = fields.get(rng_id).ok_or("RNG row has no id")?.clone();
        catalog_rows.insert(id, fields);
    }

    assert_eq!(catalog_rows.len(), 54);
    assert_eq!(RNG_COMPATIBILITY_CONTRACTS.len(), 54);
    let mut compiled = BTreeMap::new();
    for contract in RNG_COMPATIBILITY_CONTRACTS {
        assert!(compiled.insert(contract.rng_id(), *contract).is_none());
        let fields = catalog_rows
            .get(contract.rng_id())
            .ok_or("compiled RNG contract is absent from the catalog")?;
        assert_eq!(
            fields.get(phase).map(String::as_str),
            Some(contract.phase().as_str())
        );
        assert_eq!(
            fields.get(symbol).map(String::as_str),
            Some(contract.symbol())
        );
        assert_eq!(
            fields.get(seeds).map(String::as_str),
            Some(contract.seed_expressions())
        );
        assert_eq!(
            fields.get(generators).map(String::as_str),
            Some(contract.generator_expressions())
        );
        assert_eq!(
            fields.get(devices).map(String::as_str),
            Some(contract.device_expressions())
        );
        let expected_seededness = match contract.seededness() {
            sim_tensor::RngSeededness::StateConstructorOrSnapshot => {
                "state-constructor-or-snapshot"
            }
            sim_tensor::RngSeededness::StateMutator => "state-mutator",
            sim_tensor::RngSeededness::ExplicitSeedOrGenerator => "explicit-seed-or-generator",
            sim_tensor::RngSeededness::ImplicitObjectState => "implicit-global-or-object-state",
            sim_tensor::RngSeededness::ExplicitOrObjectState => {
                "explicit-seed-or-generator | implicit-global-or-object-state"
            }
            sim_tensor::RngSeededness::EntropyDefault => "entropy-default",
        };
        assert_eq!(
            fields.get(seededness).map(String::as_str),
            Some(expected_seededness)
        );
        let expected_availability = match contract.availability() {
            RngContractAvailability::Active => "active",
            RngContractAvailability::Conditional => "conditional",
            RngContractAvailability::DeveloperOnly => "developer-only",
        };
        assert_eq!(
            fields.get(availability).map(String::as_str),
            Some(expected_availability)
        );
    }
    assert_eq!(compiled.len(), catalog_rows.len());
    Ok(())
}

#[test]
fn cpu_and_device_profiles_match_certified_vectors() -> Result<(), Box<dyn Error>> {
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

    let cpu = open_cpu("COMFY-RNG-8A604BD4AC80", 17)?;
    assert_eq!(cpu.profile(), NativeRngExecutionProfile::CpuMt19937V1);
    let cuda = DeviceId::new(DeviceKind::Cuda, 2);
    let native = CompatibilityRngTransaction::open(
        "COMFY-RNG-8D517574907F",
        request(
            17,
            0,
            0,
            RetryRngPolicy::Replay,
            0,
            RngGenerationPlacement::Native(cuda),
            RngExecutionScope::Production,
        ),
        None,
        &CancellationToken::default(),
    )?;
    assert_eq!(
        native.profile(),
        NativeRngExecutionProfile::DevicePhilox4x32_10V1
    );
    assert_eq!(native.generation_device(), cuda);
    assert_eq!(native.output_device(), cuda);

    let transfer = CompatibilityRngTransaction::open(
        "COMFY-RNG-B35F0F617BFA",
        request(
            17,
            0,
            0,
            RetryRngPolicy::Replay,
            0,
            RngGenerationPlacement::CpuSeededTransfer {
                output_device: cuda,
            },
            RngExecutionScope::Production,
        ),
        None,
        &CancellationToken::default(),
    )?;
    assert_eq!(transfer.profile(), NativeRngExecutionProfile::CpuMt19937V1);
    assert_eq!(transfer.generation_device(), DeviceId::CPU);
    assert_eq!(transfer.output_device(), cuda);
    assert_ne!(native.checkpoint(), transfer.checkpoint());
    Ok(())
}

#[test]
fn phase_batch_retry_ordinal_contract_and_device_identity_are_independent()
-> Result<(), Box<dyn Error>> {
    let checkpoint = |rng_id: &str,
                      batch: u64,
                      retry: u32,
                      policy: RetryRngPolicy,
                      ordinal: u64,
                      placement: RngGenerationPlacement|
     -> Result<_, RngCompatibilityError> {
        Ok(CompatibilityRngTransaction::open(
            rng_id,
            request(
                99,
                batch,
                retry,
                policy,
                ordinal,
                placement,
                RngExecutionScope::Production,
            ),
            None,
            &CancellationToken::default(),
        )?
        .checkpoint())
    };
    let base = checkpoint(
        "COMFY-RNG-8A604BD4AC80",
        0,
        0,
        RetryRngPolicy::Replay,
        0,
        RngGenerationPlacement::Native(DeviceId::CPU),
    )?;
    assert_ne!(
        base,
        checkpoint(
            "COMFY-RNG-8A604BD4AC80",
            1,
            0,
            RetryRngPolicy::Replay,
            0,
            RngGenerationPlacement::Native(DeviceId::CPU),
        )?
    );
    assert_eq!(
        base,
        checkpoint(
            "COMFY-RNG-8A604BD4AC80",
            0,
            17,
            RetryRngPolicy::Replay,
            0,
            RngGenerationPlacement::Native(DeviceId::CPU),
        )?
    );
    assert_ne!(
        base,
        checkpoint(
            "COMFY-RNG-8A604BD4AC80",
            0,
            1,
            RetryRngPolicy::Advance,
            0,
            RngGenerationPlacement::Native(DeviceId::CPU),
        )?
    );
    assert_ne!(
        base,
        checkpoint(
            "COMFY-RNG-8A604BD4AC80",
            0,
            0,
            RetryRngPolicy::Replay,
            1,
            RngGenerationPlacement::Native(DeviceId::CPU),
        )?
    );
    assert_ne!(
        base,
        checkpoint(
            "COMFY-RNG-1D16866E414E",
            0,
            0,
            RetryRngPolicy::Replay,
            0,
            RngGenerationPlacement::Native(DeviceId::CPU),
        )?
    );
    assert_ne!(
        base,
        checkpoint(
            "COMFY-RNG-8D517574907F",
            0,
            0,
            RetryRngPolicy::Replay,
            0,
            RngGenerationPlacement::Native(DeviceId::CPU),
        )?
    );

    let cuda_zero = DeviceId::new(DeviceKind::Cuda, 0);
    let cuda_one = DeviceId::new(DeviceKind::Cuda, 1);
    assert_ne!(
        checkpoint(
            "COMFY-RNG-8D517574907F",
            0,
            0,
            RetryRngPolicy::Replay,
            0,
            RngGenerationPlacement::Native(cuda_zero),
        )?,
        checkpoint(
            "COMFY-RNG-8D517574907F",
            0,
            0,
            RetryRngPolicy::Replay,
            0,
            RngGenerationPlacement::Native(cuda_one),
        )?
    );
    Ok(())
}

#[test]
fn canonical_stream_identity_separates_every_declared_component() -> Result<(), Box<dyn Error>> {
    let checkpoint = |workflow: &str,
                      attempt: &str,
                      node: &str,
                      output: u32,
                      phase: &str,
                      batch: u64,
                      retry: u32,
                      retry_policy: RetryRngPolicy,
                      device: DeviceId,
                      seed: u64,
                      algorithm: RngAlgorithm|
     -> Result<RngCheckpoint, Box<dyn Error>> {
        Ok(RngStream::new(
            RngProfileVersion::V2,
            algorithm,
            seed,
            RngStreamAddress::for_device(
                workflow,
                attempt,
                node,
                output,
                phase,
                batch,
                retry,
                retry_policy,
                device,
            )?,
        )?
        .begin(None)?
        .commit())
    };
    let cpu = DeviceId::CPU;
    let base = checkpoint(
        "workflow",
        "attempt",
        "node",
        0,
        "noise",
        0,
        0,
        RetryRngPolicy::Advance,
        cpu,
        7,
        RngAlgorithm::Philox4x32_10,
    )?;
    let variants = [
        checkpoint(
            "workflow-2",
            "attempt",
            "node",
            0,
            "noise",
            0,
            0,
            RetryRngPolicy::Advance,
            cpu,
            7,
            RngAlgorithm::Philox4x32_10,
        )?,
        checkpoint(
            "workflow",
            "attempt-2",
            "node",
            0,
            "noise",
            0,
            0,
            RetryRngPolicy::Advance,
            cpu,
            7,
            RngAlgorithm::Philox4x32_10,
        )?,
        checkpoint(
            "workflow",
            "attempt",
            "node-2",
            0,
            "noise",
            0,
            0,
            RetryRngPolicy::Advance,
            cpu,
            7,
            RngAlgorithm::Philox4x32_10,
        )?,
        checkpoint(
            "workflow",
            "attempt",
            "node",
            1,
            "noise",
            0,
            0,
            RetryRngPolicy::Advance,
            cpu,
            7,
            RngAlgorithm::Philox4x32_10,
        )?,
        checkpoint(
            "workflow",
            "attempt",
            "node",
            0,
            "noise-2",
            0,
            0,
            RetryRngPolicy::Advance,
            cpu,
            7,
            RngAlgorithm::Philox4x32_10,
        )?,
        checkpoint(
            "workflow",
            "attempt",
            "node",
            0,
            "noise",
            1,
            0,
            RetryRngPolicy::Advance,
            cpu,
            7,
            RngAlgorithm::Philox4x32_10,
        )?,
        checkpoint(
            "workflow",
            "attempt",
            "node",
            0,
            "noise",
            0,
            1,
            RetryRngPolicy::Advance,
            cpu,
            7,
            RngAlgorithm::Philox4x32_10,
        )?,
        checkpoint(
            "workflow",
            "attempt",
            "node",
            0,
            "noise",
            0,
            0,
            RetryRngPolicy::Replay,
            cpu,
            7,
            RngAlgorithm::Philox4x32_10,
        )?,
        checkpoint(
            "workflow",
            "attempt",
            "node",
            0,
            "noise",
            0,
            0,
            RetryRngPolicy::Advance,
            DeviceId::new(DeviceKind::Cuda, 0),
            7,
            RngAlgorithm::Philox4x32_10,
        )?,
        checkpoint(
            "workflow",
            "attempt",
            "node",
            0,
            "noise",
            0,
            0,
            RetryRngPolicy::Advance,
            cpu,
            8,
            RngAlgorithm::Philox4x32_10,
        )?,
        checkpoint(
            "workflow",
            "attempt",
            "node",
            0,
            "noise",
            0,
            0,
            RetryRngPolicy::Advance,
            cpu,
            7,
            RngAlgorithm::Mt19937,
        )?,
    ];
    assert!(variants.iter().all(|checkpoint| checkpoint != &base));

    let stream = RngStream::new(
        RngProfileVersion::V2,
        RngAlgorithm::Philox4x32_10,
        7,
        RngStreamAddress::for_device(
            "workflow",
            "attempt",
            "node",
            0,
            "noise",
            0,
            0,
            RetryRngPolicy::Advance,
            cpu,
        )?,
    )?;
    let mut transaction = stream.begin(None)?;
    let initial = transaction.checkpoint();
    transaction.next_u32(&CancellationToken::default())?;
    assert_ne!(transaction.checkpoint(), initial);
    Ok(())
}

#[test]
fn checkpoints_resume_exactly_and_fail_closed_on_wrong_phase() -> Result<(), Box<dyn Error>> {
    let token = CancellationToken::default();
    let mut first = open_cpu("COMFY-RNG-8A604BD4AC80", 123)?;
    let first_half = first.draw_uniform(7, &token)?;
    let checkpoint = first.checkpoint();
    let serialized = serde_json::to_vec(&checkpoint)?;
    let checkpoint: RngCheckpoint = serde_json::from_slice(&serialized)?;
    let mut resumed = CompatibilityRngTransaction::open(
        "COMFY-RNG-8A604BD4AC80",
        cpu_request(123),
        Some(checkpoint.clone()),
        &token,
    )?;
    let second_half = resumed.draw_uniform(9, &token)?;

    let mut continuous = open_cpu("COMFY-RNG-8A604BD4AC80", 123)?;
    let all = continuous.draw_uniform(16, &token)?;
    assert_eq!(first_half.as_slice(), &all[..7]);
    assert_eq!(second_half.as_slice(), &all[7..]);
    assert!(matches!(
        CompatibilityRngTransaction::open(
            "COMFY-RNG-8D517574907F",
            cpu_request(123),
            Some(checkpoint),
            &token,
        ),
        Err(RngCompatibilityError::Canonical(
            sim_tensor::RngError::CheckpointMismatch
        ))
    ));
    Ok(())
}

fn terminal_philox_checkpoint(
    stream: &RngStream,
    block_index: usize,
) -> Result<RngCheckpoint, Box<dyn Error>> {
    let key = [0x1020_3040, 0x5060_7080];
    let block = Philox4x32::generate([u32::MAX; 4], key);
    let mut value = serde_json::to_value(stream.begin(None)?.commit())?;
    let state = value
        .pointer_mut("/generator/state")
        .ok_or("Philox checkpoint has no serialized generator state")?;
    *state = serde_json::json!({
        "counter": [u32::MAX, u32::MAX, u32::MAX, u32::MAX],
        "key": key,
        "block": block,
        "block_index": block_index,
        "counter_exhausted": true,
    });
    Ok(serde_json::from_value(value)?)
}

#[test]
fn forged_philox_blocks_are_rejected_and_multiword_faults_do_not_advance()
-> Result<(), Box<dyn Error>> {
    let device = DeviceId::new(DeviceKind::Cuda, 0);
    let stream = RngStream::new(
        RngProfileVersion::V2,
        RngAlgorithm::Philox4x32_10,
        0x1234_5678_9abc_def0,
        RngStreamAddress::for_device(
            "workflow",
            "attempt",
            "node",
            0,
            "sampling-noise-and-solver",
            0,
            0,
            RetryRngPolicy::Replay,
            device,
        )?,
    )?;

    let partial = terminal_philox_checkpoint(&stream, 3)?;
    let mut forged = serde_json::to_value(&partial)?;
    let first_word = forged
        .pointer_mut("/generator/state/block/0")
        .ok_or("Philox checkpoint has no first block word")?;
    *first_word = serde_json::json!(0);
    assert!(serde_json::from_value::<RngCheckpoint>(forged).is_err());

    let mut unconsumed = serde_json::to_value(&partial)?;
    let block_index = unconsumed
        .pointer_mut("/generator/state/block_index")
        .ok_or("Philox checkpoint has no block index")?;
    *block_index = serde_json::json!(0);
    assert!(serde_json::from_value::<RngCheckpoint>(unconsumed).is_err());

    let cancellation = CancellationToken::default();
    let mut unit = stream.begin(Some(partial.clone()))?;
    assert_eq!(
        unit.next_unit_f64(&cancellation),
        Err(RngError::CounterOverflow)
    );
    assert_eq!(unit.checkpoint(), partial);

    let mut normal = stream.begin(Some(partial.clone()))?;
    assert_eq!(
        normal.next_standard_normal_pair(&cancellation),
        Err(RngError::CounterOverflow)
    );
    assert_eq!(normal.checkpoint(), partial);

    let mut bounded = stream.begin(Some(partial.clone()))?;
    assert_eq!(
        bounded.next_bounded_u64(17, &cancellation),
        Err(RngError::CounterOverflow)
    );
    assert_eq!(bounded.checkpoint(), partial);
    Ok(())
}

#[test]
fn native_compatibility_operations_are_deterministic_and_transactional()
-> Result<(), Box<dyn Error>> {
    let token = CancellationToken::default();

    let mut normal = open_cpu("COMFY-RNG-49AB7DF1BB2A", 1)?;
    let normal_values = normal.draw_normal(7, &token)?;
    assert_eq!(normal_values.len(), 7);
    assert!(normal_values.iter().all(|value| value.is_finite()));

    let mut integers = open_cpu("COMFY-RNG-DD4FB4404AA8", 2)?;
    let integer_values = integers.draw_integers(-9, 13, 128, &token)?;
    assert!(integer_values.iter().all(|value| (-9..13).contains(value)));

    let mut permutation = open_cpu("COMFY-RNG-9D90BF16BDD9", 3)?;
    let mut permutation_values = permutation.draw_permutation(64, &token)?;
    permutation_values.sort_unstable();
    assert_eq!(permutation_values, (0..64).collect::<Vec<_>>());

    let mut choice = open_cpu("COMFY-RNG-A87E838B0B91", 4)?;
    let choices = choice.draw_choice(&['a', 'b', 'c', 'd'], 4, false, &token)?;
    assert_eq!(choices.len(), 4);
    let mut sorted = choices;
    sorted.sort_unstable();
    assert_eq!(sorted, vec!['a', 'b', 'c', 'd']);

    let mut multinomial = open_cpu("COMFY-RNG-07843F80B32F", 5)?;
    let samples = multinomial.draw_multinomial(&[0.0, 1.0, 0.0], 8, true, &token)?;
    assert_eq!(samples, vec![1; 8]);

    let mut rounding = open_cpu("COMFY-RNG-BAEBF34A762E", 6)?;
    assert_eq!(
        rounding.stochastic_round_up(&[0.0, 1.0, 0.0, 1.0], &token)?,
        vec![false, true, false, true]
    );

    let mut sobol = CompatibilityRngTransaction::open(
        "COMFY-RNG-D24115262D6C",
        request_with_transform(
            0,
            0,
            0,
            RetryRngPolicy::Replay,
            0,
            RngSeedTransform::Fixed(123),
            RngGenerationPlacement::Native(DeviceId::CPU),
            RngExecutionScope::Production,
        ),
        None,
        &token,
    )?;
    let mut engine = sobol.sobol_engine(3, true, &token)?;
    let serialized_engine = serde_json::to_vec(&engine)?;
    engine = serde_json::from_slice(&serialized_engine)?;
    let sobol_draw = open_cpu("COMFY-RNG-E38F1D9F896D", 123)?;
    let sobol_values = sobol_draw.sobol_draw(&mut engine, 8, &token)?;
    assert_eq!(sobol_values.len(), 24);
    assert!(sobol_values.iter().all(|value| (0.0..1.0).contains(value)));

    let mut brownian = open_cpu("COMFY-RNG-DED616CC3432", 7)?;
    let mut tree: BrownianTree = brownian.brownian_tree(0.0, vec![0.0, 1.0], 1.0, &token)?;
    let serialized_tree = serde_json::to_vec(&tree)?;
    tree = serde_json::from_slice(&serialized_tree)?;
    let increment = tree.increment(0.25, 0.75, &token)?;
    assert_eq!(increment.len(), 2);
    assert!(increment.iter().all(|value| value.is_finite()));

    let mut cancelled = open_cpu("COMFY-RNG-8A604BD4AC80", 8)?;
    let before = cancelled.checkpoint();
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    assert_eq!(
        cancelled.draw_uniform(64, &cancellation),
        Err(RngCompatibilityError::Cancelled)
    );
    assert_eq!(cancelled.checkpoint(), before);
    Ok(())
}

#[test]
fn seed_device_scope_and_operation_boundaries_fail_closed() -> Result<(), Box<dyn Error>> {
    assert_eq!(RngSeedTransform::TorchSigned64.apply(-1)?, u64::MAX - 1);
    assert_eq!(
        RngSeedTransform::NumpyModulo32MinusOne.apply(-1)?,
        u64::from(u32::MAX - 1)
    );
    assert_eq!(RngSeedTransform::Add(-10).apply(20)?, 10);
    assert_eq!(
        RngSeedTransform::BlockOffset {
            block_index: 3,
            item_index: 7,
            block_stride: 1_000,
        }
        .apply(11)?,
        3_018
    );
    assert!(
        RngSeedTransform::TorchSigned64
            .apply(i128::from(u64::MAX) + 1)
            .is_err()
    );
    assert!(matches!(
        CompatibilityRngTransaction::open(
            "COMFY-RNG-7D7369F996F5",
            request_with_transform(
                1,
                0,
                0,
                RetryRngPolicy::Replay,
                0,
                RngSeedTransform::TorchSigned64,
                RngGenerationPlacement::Native(DeviceId::CPU),
                RngExecutionScope::ProductionWithConditionalCapability,
            ),
            None,
            &CancellationToken::default(),
        ),
        Err(RngCompatibilityError::SeedTransformMismatch { .. })
    ));
    assert!(matches!(
        CompatibilityRngTransaction::open(
            "COMFY-RNG-9523F0E94398",
            cpu_request(1),
            None,
            &CancellationToken::default(),
        ),
        Err(RngCompatibilityError::ConditionalContractUnavailable { .. })
    ));

    let cuda = DeviceId::new(DeviceKind::Cuda, 0);
    assert!(matches!(
        CompatibilityRngTransaction::open(
            "COMFY-RNG-E30A300B3733",
            request(
                1,
                0,
                0,
                RetryRngPolicy::Replay,
                0,
                RngGenerationPlacement::Native(cuda),
                RngExecutionScope::Production,
            ),
            None,
            &CancellationToken::default(),
        ),
        Err(RngCompatibilityError::UnsupportedDevice { .. })
    ));
    for cpu_only_id in [
        "COMFY-RNG-A87E838B0B91",
        "COMFY-RNG-854EBB64647D",
        "COMFY-RNG-D48F4F28D2AA",
    ] {
        assert!(matches!(
            CompatibilityRngTransaction::open(
                cpu_only_id,
                request(
                    1,
                    0,
                    0,
                    RetryRngPolicy::Replay,
                    0,
                    RngGenerationPlacement::Native(cuda),
                    RngExecutionScope::Production,
                ),
                None,
                &CancellationToken::default(),
            ),
            Err(RngCompatibilityError::UnsupportedDevice { .. })
        ));
    }
    assert!(
        CompatibilityRngTransaction::open(
            "COMFY-RNG-D48F4F28D2AA",
            request(
                1,
                0,
                0,
                RetryRngPolicy::Replay,
                0,
                RngGenerationPlacement::CpuSeededTransfer {
                    output_device: cuda,
                },
                RngExecutionScope::Production,
            ),
            None,
            &CancellationToken::default(),
        )
        .is_ok()
    );
    assert!(matches!(
        CompatibilityRngTransaction::open(
            "COMFY-RNG-3C7295ABF6D6",
            cpu_request(1),
            None,
            &CancellationToken::default(),
        ),
        Err(RngCompatibilityError::DeveloperOnlyContract { .. })
    ));
    assert!(
        CompatibilityRngTransaction::open(
            "COMFY-RNG-3C7295ABF6D6",
            request(
                1,
                0,
                0,
                RetryRngPolicy::Replay,
                0,
                RngGenerationPlacement::Native(DeviceId::CPU),
                RngExecutionScope::DeveloperValidation,
            ),
            None,
            &CancellationToken::default(),
        )
        .is_ok()
    );
    assert!(matches!(
        CompatibilityRngTransaction::open(
            "COMFY-RNG-NOT-CATALOGED",
            cpu_request(1),
            None,
            &CancellationToken::default(),
        ),
        Err(RngCompatibilityError::UnknownContract(_))
    ));

    let mut normal = open_cpu("COMFY-RNG-49AB7DF1BB2A", 9)?;
    assert!(matches!(
        normal.draw_permutation(4, &CancellationToken::default()),
        Err(RngCompatibilityError::UnsupportedOperation { .. })
    ));
    let mut multinomial = open_cpu("COMFY-RNG-07843F80B32F", 10)?;
    assert_eq!(
        multinomial.draw_multinomial(&[0.0, f64::NAN], 1, true, &CancellationToken::default()),
        Err(RngCompatibilityError::InvalidWeights)
    );
    Ok(())
}

#[test]
fn every_catalog_contract_opens_in_its_declared_scope() -> Result<(), Box<dyn Error>> {
    for contract in RNG_COMPATIBILITY_CONTRACTS {
        let scope = match contract.availability() {
            RngContractAvailability::DeveloperOnly => RngExecutionScope::DeveloperValidation,
            RngContractAvailability::Conditional => {
                RngExecutionScope::ProductionWithConditionalCapability
            }
            RngContractAvailability::Active => RngExecutionScope::Production,
        };
        let seed_transform = match contract.seed_expressions() {
            "123" => RngSeedTransform::Fixed(123),
            "seed % (2 ** 32 - 1)" => RngSeedTransform::NumpyModulo32MinusOne,
            _ => RngSeedTransform::TorchSigned64,
        };
        let mut transaction = CompatibilityRngTransaction::open(
            contract.rng_id(),
            request_with_transform(
                123,
                2,
                1,
                RetryRngPolicy::Advance,
                4,
                seed_transform,
                RngGenerationPlacement::Native(DeviceId::CPU),
                scope,
            ),
            None,
            &CancellationToken::default(),
        )?;
        assert_eq!(transaction.contract(), *contract);
        assert_eq!(
            transaction.profile(),
            NativeRngExecutionProfile::CpuMt19937V1
        );
        assert_eq!(
            transaction.checkpoint().algorithm,
            sim_tensor::RngAlgorithm::Mt19937
        );
        assert_ne!(contract.symbol(), "");
        assert!(matches!(
            contract.operation(),
            RngCompatibilityOperation::Generator
                | RngCompatibilityOperation::ManualSeed
                | RngCompatibilityOperation::Choice
                | RngCompatibilityOperation::Uniform
                | RngCompatibilityOperation::UniformInitializer
                | RngCompatibilityOperation::SobolEngine
                | RngCompatibilityOperation::SobolDraw
                | RngCompatibilityOperation::Multinomial
                | RngCompatibilityOperation::Integer
                | RngCompatibilityOperation::Normal
                | RngCompatibilityOperation::NormalLike
                | RngCompatibilityOperation::NormalInitializer
                | RngCompatibilityOperation::Permutation
                | RngCompatibilityOperation::BrownianTree
        ));
        let before = transaction.checkpoint();
        let cancellation = CancellationToken::default();
        match contract.operation() {
            RngCompatibilityOperation::Generator | RngCompatibilityOperation::ManualSeed => {}
            RngCompatibilityOperation::Choice => {
                assert_eq!(
                    transaction
                        .draw_choice(&[11, 13, 17], 2, false, &cancellation)?
                        .len(),
                    2
                );
            }
            RngCompatibilityOperation::Uniform | RngCompatibilityOperation::UniformInitializer => {
                assert_eq!(transaction.draw_uniform(3, &cancellation)?.len(), 3);
            }
            RngCompatibilityOperation::SobolEngine => {
                assert_eq!(
                    transaction
                        .sobol_engine(3, true, &cancellation)?
                        .dimension(),
                    3
                );
            }
            RngCompatibilityOperation::SobolDraw => {
                let mut engine = sim_tensor::SobolEngine::new(3, false, 123)?;
                assert_eq!(
                    transaction.sobol_draw(&mut engine, 2, &cancellation)?.len(),
                    6
                );
                assert_eq!(engine.generated(), 2);
            }
            RngCompatibilityOperation::Multinomial => {
                assert_eq!(
                    transaction
                        .draw_multinomial(&[1.0, 2.0, 3.0], 2, false, &cancellation)?
                        .len(),
                    2
                );
            }
            RngCompatibilityOperation::Integer => {
                let values = transaction.draw_integers(-3, 5, 4, &cancellation)?;
                assert!(values.iter().all(|value| (-3..5).contains(value)));
            }
            RngCompatibilityOperation::Normal
            | RngCompatibilityOperation::NormalLike
            | RngCompatibilityOperation::NormalInitializer => {
                let values = transaction.draw_normal(3, &cancellation)?;
                assert!(values.iter().all(|value| value.is_finite()));
            }
            RngCompatibilityOperation::Permutation => {
                let mut values = transaction.draw_permutation(4, &cancellation)?;
                values.sort_unstable();
                assert_eq!(values, vec![0, 1, 2, 3]);
            }
            RngCompatibilityOperation::BrownianTree => {
                assert_eq!(
                    transaction
                        .brownian_tree(0.0, vec![0.0, 1.0], 1.0, &cancellation)?
                        .dimension(),
                    2
                );
            }
        }
        if !matches!(
            contract.operation(),
            RngCompatibilityOperation::Generator
                | RngCompatibilityOperation::ManualSeed
                | RngCompatibilityOperation::SobolDraw
        ) {
            assert_ne!(transaction.checkpoint(), before);
        }
    }
    Ok(())
}
