use super::{
    BrownianTree, CancellationToken, DeviceId, RetryRngPolicy, RngAlgorithm, RngCheckpoint,
    RngError, RngProfileVersion, RngStream, RngStreamAddress, RngTransaction, SobolEngine,
};
use serde::{Deserialize, Serialize};
use sim_types::DeviceKind;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RngCompatibilityPhase {
    ContextWindowSelection,
    ModelInternalStochasticity,
    NodeLevelNoise,
    RuntimeUtility,
    SamplingNoiseAndSolver,
    StochasticQuantization,
    TemporaryOutputNaming,
    TestFixture,
    TrainingAndDataOrder,
}

impl RngCompatibilityPhase {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ContextWindowSelection => "context-window-selection",
            Self::ModelInternalStochasticity => "model-internal-stochasticity",
            Self::NodeLevelNoise => "node-level-noise",
            Self::RuntimeUtility => "runtime-utility",
            Self::SamplingNoiseAndSolver => "sampling-noise-and-solver",
            Self::StochasticQuantization => "stochastic-quantization",
            Self::TemporaryOutputNaming => "temporary-output-naming",
            Self::TestFixture => "test-fixture",
            Self::TrainingAndDataOrder => "training-and-data-order",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RngCompatibilityOperation {
    Generator,
    ManualSeed,
    Choice,
    Uniform,
    UniformInitializer,
    SobolEngine,
    SobolDraw,
    Multinomial,
    Integer,
    Normal,
    NormalLike,
    NormalInitializer,
    Permutation,
    BrownianTree,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RngSeededness {
    StateConstructorOrSnapshot,
    StateMutator,
    ExplicitSeedOrGenerator,
    ImplicitObjectState,
    ExplicitOrObjectState,
    EntropyDefault,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RngContractAvailability {
    Active,
    Conditional,
    DeveloperOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RngCompatibilityContract {
    rng_id: &'static str,
    phase: RngCompatibilityPhase,
    operation: RngCompatibilityOperation,
    seededness: RngSeededness,
    availability: RngContractAvailability,
    symbol: &'static str,
    seed_expressions: &'static str,
    generator_expressions: &'static str,
    device_expressions: &'static str,
}

impl RngCompatibilityContract {
    pub const fn rng_id(self) -> &'static str {
        self.rng_id
    }

    pub const fn phase(self) -> RngCompatibilityPhase {
        self.phase
    }

    pub const fn operation(self) -> RngCompatibilityOperation {
        self.operation
    }

    pub const fn seededness(self) -> RngSeededness {
        self.seededness
    }

    pub const fn availability(self) -> RngContractAvailability {
        self.availability
    }

    pub const fn symbol(self) -> &'static str {
        self.symbol
    }

    pub const fn seed_expressions(self) -> &'static str {
        self.seed_expressions
    }

    pub const fn generator_expressions(self) -> &'static str {
        self.generator_expressions
    }

    pub const fn device_expressions(self) -> &'static str {
        self.device_expressions
    }

    fn requires_cpu(self) -> bool {
        self.device_expressions == "'cpu'"
            || self.symbol.starts_with("numpy.random.")
            || self.symbol.starts_with("random.")
            || self.symbol == "generator.choice"
            || self.symbol.starts_with("torch.quasirandom.")
    }
}

macro_rules! contract {
    ($id:literal, $phase:ident, $operation:ident, $seededness:ident, $availability:ident, $symbol:literal, $seeds:literal, $generators:literal, $devices:literal) => {
        RngCompatibilityContract {
            rng_id: $id,
            phase: RngCompatibilityPhase::$phase,
            operation: RngCompatibilityOperation::$operation,
            seededness: RngSeededness::$seededness,
            availability: RngContractAvailability::$availability,
            symbol: $symbol,
            seed_expressions: $seeds,
            generator_expressions: $generators,
            device_expressions: $devices,
        }
    };
}

pub const RNG_COMPATIBILITY_CONTRACTS: &[RngCompatibilityContract] = &[
    contract!(
        "COMFY-RNG-E30A300B3733",
        ContextWindowSelection,
        Generator,
        StateConstructorOrSnapshot,
        Active,
        "torch.Generator",
        "",
        "",
        "'cpu'"
    ),
    contract!(
        "COMFY-RNG-BC5120977B61",
        ContextWindowSelection,
        ManualSeed,
        StateMutator,
        Active,
        "torch.Generator.manual_seed",
        "seed",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-9D90BF16BDD9",
        ContextWindowSelection,
        Permutation,
        ExplicitSeedOrGenerator,
        Active,
        "torch.randperm",
        "",
        "generator",
        "'cpu'"
    ),
    contract!(
        "COMFY-RNG-A87E838B0B91",
        ModelInternalStochasticity,
        Choice,
        ImplicitObjectState,
        Active,
        "generator.choice",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-669933ED7F3B",
        ModelInternalStochasticity,
        ManualSeed,
        StateMutator,
        Active,
        "generator.manual_seed",
        "kwargs.get('seed', 0) - 10 | seed",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-854EBB64647D",
        ModelInternalStochasticity,
        Generator,
        EntropyDefault,
        Active,
        "numpy.random.default_rng",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-8A604BD4AC80",
        ModelInternalStochasticity,
        Uniform,
        ImplicitObjectState,
        Active,
        "numpy.random.rand",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-19548855F372",
        ModelInternalStochasticity,
        Generator,
        StateConstructorOrSnapshot,
        Active,
        "torch.Generator",
        "",
        "",
        "'cpu' | device"
    ),
    contract!(
        "COMFY-RNG-BED75D487793",
        ModelInternalStochasticity,
        ManualSeed,
        StateMutator,
        Active,
        "torch.Generator.manual_seed",
        "seed",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-B93ACA286660",
        ModelInternalStochasticity,
        ManualSeed,
        StateMutator,
        Active,
        "torch.manual_seed",
        "seed",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-07843F80B32F",
        ModelInternalStochasticity,
        Multinomial,
        ExplicitSeedOrGenerator,
        Active,
        "torch.multinomial",
        "",
        "generator",
        ""
    ),
    contract!(
        "COMFY-RNG-F630E85820A8",
        ModelInternalStochasticity,
        UniformInitializer,
        ImplicitObjectState,
        Active,
        "torch.nn.init.uniform_",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-D24115262D6C",
        ModelInternalStochasticity,
        SobolEngine,
        ExplicitSeedOrGenerator,
        Active,
        "torch.quasirandom.SobolEngine",
        "123",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-E38F1D9F896D",
        ModelInternalStochasticity,
        SobolDraw,
        ImplicitObjectState,
        Active,
        "torch.quasirandom.SobolEngine().draw",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-1D16866E414E",
        ModelInternalStochasticity,
        Uniform,
        ExplicitOrObjectState,
        Active,
        "torch.rand",
        "",
        "generator",
        ""
    ),
    contract!(
        "COMFY-RNG-630ED2FD1166",
        ModelInternalStochasticity,
        Integer,
        ImplicitObjectState,
        Active,
        "torch.randint",
        "",
        "",
        "non_sky_depth.device | src.device | valid.device | x.device"
    ),
    contract!(
        "COMFY-RNG-CE4D123C8056",
        ModelInternalStochasticity,
        Normal,
        ExplicitOrObjectState,
        Active,
        "torch.randn",
        "",
        "generator | torch.manual_seed(seed)",
        "'cpu' | device | self.mean.device | self.parameters.device | x.device"
    ),
    contract!(
        "COMFY-RNG-CF0765383ED1",
        ModelInternalStochasticity,
        NormalLike,
        ImplicitObjectState,
        Active,
        "torch.randn_like",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-208DBEFC42E6",
        ModelInternalStochasticity,
        Permutation,
        ImplicitObjectState,
        Active,
        "torch.randperm",
        "",
        "",
        "device | pc.device"
    ),
    contract!(
        "COMFY-RNG-5F27E6E9CF62",
        NodeLevelNoise,
        ManualSeed,
        StateMutator,
        Conditional,
        "torch.manual_seed",
        "seed",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-7E8226FD6698",
        NodeLevelNoise,
        Normal,
        ExplicitOrObjectState,
        Conditional,
        "torch.randn",
        "",
        "generator",
        "'cpu' | device"
    ),
    contract!(
        "COMFY-RNG-023278562D68",
        NodeLevelNoise,
        NormalLike,
        ImplicitObjectState,
        Conditional,
        "torch.randn_like",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-9523F0E94398",
        RuntimeUtility,
        Integer,
        ImplicitObjectState,
        Conditional,
        "numpy.random.randint",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-96011D5BEC0E",
        RuntimeUtility,
        Generator,
        StateConstructorOrSnapshot,
        Conditional,
        "torch.Generator",
        "",
        "",
        "'cpu'"
    ),
    contract!(
        "COMFY-RNG-0B344C03A275",
        RuntimeUtility,
        ManualSeed,
        StateMutator,
        Conditional,
        "torch.Generator.manual_seed",
        "seed",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-E95CB6472219",
        RuntimeUtility,
        NormalInitializer,
        ImplicitObjectState,
        Active,
        "torch.nn.init.normal_",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-F5171D570EB5",
        RuntimeUtility,
        Integer,
        ImplicitObjectState,
        Conditional,
        "torch.randint",
        "",
        "",
        "metric.device"
    ),
    contract!(
        "COMFY-RNG-49AB7DF1BB2A",
        RuntimeUtility,
        Normal,
        ImplicitObjectState,
        Active,
        "torch.randn",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-91EC48410531",
        RuntimeUtility,
        NormalLike,
        ImplicitObjectState,
        Conditional,
        "torch.randn_like",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-39C5A625240B",
        RuntimeUtility,
        Permutation,
        ImplicitObjectState,
        Conditional,
        "torch.randperm",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-E726D5E6319D",
        SamplingNoiseAndSolver,
        ManualSeed,
        StateMutator,
        Active,
        "generator.manual_seed",
        "seed",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-0E78C95BEFF9",
        SamplingNoiseAndSolver,
        Generator,
        StateConstructorOrSnapshot,
        Active,
        "torch.Generator",
        "",
        "",
        "x.device"
    ),
    contract!(
        "COMFY-RNG-FBE61A38BE1E",
        SamplingNoiseAndSolver,
        ManualSeed,
        StateMutator,
        Active,
        "torch.manual_seed",
        "extra_args.get('seed', 41) + 1 | seed | seed + block_idx * 1000 + i",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-8D517574907F",
        SamplingNoiseAndSolver,
        Uniform,
        ImplicitObjectState,
        Active,
        "torch.rand",
        "",
        "",
        "device"
    ),
    contract!(
        "COMFY-RNG-DD4FB4404AA8",
        SamplingNoiseAndSolver,
        Integer,
        ImplicitObjectState,
        Active,
        "torch.randint",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-B35F0F617BFA",
        SamplingNoiseAndSolver,
        Normal,
        ExplicitOrObjectState,
        Active,
        "torch.randn",
        "",
        "generator",
        "'cpu' | device | x.device"
    ),
    contract!(
        "COMFY-RNG-D68A0DD3FBE1",
        SamplingNoiseAndSolver,
        NormalLike,
        ImplicitObjectState,
        Active,
        "torch.randn_like",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-DED616CC3432",
        SamplingNoiseAndSolver,
        BrownianTree,
        ExplicitSeedOrGenerator,
        Active,
        "torchsde.BrownianTree",
        "s",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-89DB49186517",
        StochasticQuantization,
        ManualSeed,
        StateMutator,
        Active,
        "generator.manual_seed",
        "seed",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-2415186A60B6",
        StochasticQuantization,
        Generator,
        StateConstructorOrSnapshot,
        Active,
        "torch.Generator",
        "",
        "",
        "value.device | x.device"
    ),
    contract!(
        "COMFY-RNG-BAEBF34A762E",
        StochasticQuantization,
        Uniform,
        ExplicitSeedOrGenerator,
        Active,
        "torch.rand",
        "",
        "generator",
        "mantissa_scaled.device | x.device"
    ),
    contract!(
        "COMFY-RNG-B6A95015CE1D",
        StochasticQuantization,
        Integer,
        ExplicitSeedOrGenerator,
        Active,
        "torch.randint",
        "",
        "generator",
        "value.device"
    ),
    contract!(
        "COMFY-RNG-D48F4F28D2AA",
        TemporaryOutputNaming,
        Choice,
        ImplicitObjectState,
        Active,
        "random.choice",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-3C7295ABF6D6",
        TestFixture,
        Integer,
        ImplicitObjectState,
        DeveloperOnly,
        "random.randint",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-074906879767",
        TestFixture,
        ManualSeed,
        StateMutator,
        DeveloperOnly,
        "torch.manual_seed",
        "123",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-32EEEB26D2EE",
        TestFixture,
        Uniform,
        ImplicitObjectState,
        DeveloperOnly,
        "torch.rand",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-DF83DF84AB24",
        TestFixture,
        Normal,
        ImplicitObjectState,
        DeveloperOnly,
        "torch.randn",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-4394100BA02E",
        TestFixture,
        NormalLike,
        ImplicitObjectState,
        DeveloperOnly,
        "torch.randn_like",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-5FC8CA616A46",
        TrainingAndDataOrder,
        Permutation,
        ImplicitObjectState,
        Conditional,
        "numpy.random.permutation",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-3161695351F4",
        TrainingAndDataOrder,
        Integer,
        ImplicitObjectState,
        Conditional,
        "numpy.random.randint",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-7D7369F996F5",
        TrainingAndDataOrder,
        ManualSeed,
        StateMutator,
        Conditional,
        "numpy.random.seed",
        "seed % (2 ** 32 - 1)",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-8F0BFD8FEB1C",
        TrainingAndDataOrder,
        Multinomial,
        ImplicitObjectState,
        Conditional,
        "torch.multinomial",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-A3A664E9F5E6",
        TrainingAndDataOrder,
        Uniform,
        ImplicitObjectState,
        Conditional,
        "torch.rand",
        "",
        "",
        ""
    ),
    contract!(
        "COMFY-RNG-5A94B27C5DCC",
        TrainingAndDataOrder,
        Permutation,
        ImplicitObjectState,
        Conditional,
        "torch.randperm",
        "",
        "",
        ""
    ),
];

pub fn rng_compatibility_contract(rng_id: &str) -> Option<RngCompatibilityContract> {
    RNG_COMPATIBILITY_CONTRACTS
        .iter()
        .copied()
        .find(|contract| contract.rng_id == rng_id)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeRngExecutionProfile {
    CpuMt19937V1,
    DevicePhilox4x32_10V1,
}

impl NativeRngExecutionProfile {
    pub const fn algorithm(self) -> RngAlgorithm {
        match self {
            Self::CpuMt19937V1 => RngAlgorithm::Mt19937,
            Self::DevicePhilox4x32_10V1 => RngAlgorithm::Philox4x32_10,
        }
    }

    pub const fn stream_profile(self) -> RngProfileVersion {
        match self {
            Self::CpuMt19937V1 => RngProfileVersion::V1,
            Self::DevicePhilox4x32_10V1 => RngProfileVersion::V2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RngExecutionScope {
    Production,
    ProductionWithConditionalCapability,
    DeveloperValidation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RngGenerationPlacement {
    Native(DeviceId),
    CpuSeededTransfer { output_device: DeviceId },
}

impl RngGenerationPlacement {
    pub const fn generation_device(self) -> DeviceId {
        match self {
            Self::Native(device) => device,
            Self::CpuSeededTransfer { .. } => DeviceId::CPU,
        }
    }

    pub const fn output_device(self) -> DeviceId {
        match self {
            Self::Native(device)
            | Self::CpuSeededTransfer {
                output_device: device,
            } => device,
        }
    }

    const fn identity(self) -> &'static str {
        match self {
            Self::Native(_) => "native",
            Self::CpuSeededTransfer { .. } => "cpu-seeded-transfer",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RngSeedTransform {
    TorchSigned64,
    WrapUnsigned64,
    NumpyModulo32MinusOne,
    Add(i128),
    BlockOffset {
        block_index: u64,
        item_index: u64,
        block_stride: u64,
    },
    Fixed(u64),
}

impl RngSeedTransform {
    pub fn apply(self, base_seed: i128) -> Result<u64, RngCompatibilityError> {
        match self {
            Self::TorchSigned64 => torch_signed_seed(base_seed),
            Self::WrapUnsigned64 => {
                let modulus = i128::from(u64::MAX) + 1;
                u64::try_from(base_seed.rem_euclid(modulus)).map_err(|_| {
                    RngCompatibilityError::InvalidSeed {
                        reason: "wrapped seed did not fit u64".to_owned(),
                    }
                })
            }
            Self::NumpyModulo32MinusOne => {
                let modulus = i128::from(u32::MAX);
                u64::try_from(base_seed.rem_euclid(modulus)).map_err(|_| {
                    RngCompatibilityError::InvalidSeed {
                        reason: "NumPy seed did not fit its 32-bit-minus-one modulus".to_owned(),
                    }
                })
            }
            Self::Add(offset) => {
                let adjusted = base_seed.checked_add(offset).ok_or_else(|| {
                    RngCompatibilityError::InvalidSeed {
                        reason: "seed offset overflowed i128".to_owned(),
                    }
                })?;
                torch_signed_seed(adjusted)
            }
            Self::BlockOffset {
                block_index,
                item_index,
                block_stride,
            } => {
                let offset = block_index
                    .checked_mul(block_stride)
                    .and_then(|value| value.checked_add(item_index))
                    .ok_or_else(|| RngCompatibilityError::InvalidSeed {
                        reason: "block seed offset overflowed u64".to_owned(),
                    })?;
                let adjusted = base_seed.checked_add(i128::from(offset)).ok_or_else(|| {
                    RngCompatibilityError::InvalidSeed {
                        reason: "block seed overflowed i128".to_owned(),
                    }
                })?;
                torch_signed_seed(adjusted)
            }
            Self::Fixed(seed) => Ok(seed),
        }
    }
}

fn torch_signed_seed(seed: i128) -> Result<u64, RngCompatibilityError> {
    if seed < i128::from(i64::MIN) || seed > i128::from(u64::MAX) {
        return Err(RngCompatibilityError::InvalidSeed {
            reason: "seed is outside PyTorch's signed-to-unsigned 64-bit range".to_owned(),
        });
    }
    if seed < 0 {
        let remapped = i128::from(u64::MAX).checked_add(seed).ok_or_else(|| {
            RngCompatibilityError::InvalidSeed {
                reason: "negative PyTorch seed remapping overflowed".to_owned(),
            }
        })?;
        u64::try_from(remapped).map_err(|_| RngCompatibilityError::InvalidSeed {
            reason: "negative PyTorch seed remapping did not fit u64".to_owned(),
        })
    } else {
        u64::try_from(seed).map_err(|_| RngCompatibilityError::InvalidSeed {
            reason: "positive PyTorch seed did not fit u64".to_owned(),
        })
    }
}

const fn device_kind_name(kind: DeviceKind) -> &'static str {
    match kind {
        DeviceKind::Cpu => "cpu",
        DeviceKind::Cuda => "cuda",
        DeviceKind::Rocm => "rocm",
        DeviceKind::Metal => "metal",
        DeviceKind::DirectMl => "directml",
        DeviceKind::Xpu => "xpu",
        DeviceKind::Npu => "npu",
        DeviceKind::Mlu => "mlu",
        DeviceKind::CoreX => "corex",
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RngCompatibilityRequest {
    workflow: String,
    attempt: String,
    node: String,
    output: u32,
    execution_ordinal: u64,
    batch: u64,
    retry: u32,
    retry_policy: RetryRngPolicy,
    base_seed: i128,
    seed_transform: RngSeedTransform,
    placement: RngGenerationPlacement,
    scope: RngExecutionScope,
}

impl RngCompatibilityRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        workflow: impl Into<String>,
        attempt: impl Into<String>,
        node: impl Into<String>,
        output: u32,
        execution_ordinal: u64,
        batch: u64,
        retry: u32,
        retry_policy: RetryRngPolicy,
        base_seed: i128,
        seed_transform: RngSeedTransform,
        placement: RngGenerationPlacement,
        scope: RngExecutionScope,
    ) -> Self {
        Self {
            workflow: workflow.into(),
            attempt: attempt.into(),
            node: node.into(),
            output,
            execution_ordinal,
            batch,
            retry,
            retry_policy,
            base_seed,
            seed_transform,
            placement,
            scope,
        }
    }
}

pub struct CompatibilityRngTransaction {
    contract: RngCompatibilityContract,
    profile: NativeRngExecutionProfile,
    placement: RngGenerationPlacement,
    transaction: RngTransaction,
}

impl CompatibilityRngTransaction {
    pub fn open(
        rng_id: &str,
        request: RngCompatibilityRequest,
        checkpoint: Option<RngCheckpoint>,
        cancellation: &CancellationToken,
    ) -> Result<Self, RngCompatibilityError> {
        cancellation
            .check()
            .map_err(|_| RngCompatibilityError::Cancelled)?;
        let contract = rng_compatibility_contract(rng_id)
            .ok_or_else(|| RngCompatibilityError::UnknownContract(rng_id.to_owned()))?;
        if contract.availability == RngContractAvailability::DeveloperOnly
            && request.scope != RngExecutionScope::DeveloperValidation
        {
            return Err(RngCompatibilityError::DeveloperOnlyContract {
                rng_id: contract.rng_id,
            });
        }
        if contract.availability == RngContractAvailability::Conditional
            && request.scope == RngExecutionScope::Production
        {
            return Err(RngCompatibilityError::ConditionalContractUnavailable {
                rng_id: contract.rng_id,
            });
        }
        let generation_device = request.placement.generation_device();
        if contract.requires_cpu() && generation_device != DeviceId::CPU {
            return Err(RngCompatibilityError::UnsupportedDevice {
                rng_id: contract.rng_id,
                device: generation_device,
            });
        }
        let profile = if generation_device == DeviceId::CPU {
            NativeRngExecutionProfile::CpuMt19937V1
        } else {
            NativeRngExecutionProfile::DevicePhilox4x32_10V1
        };
        validate_seed_transform(contract, request.seed_transform)?;
        let seed = request.seed_transform.apply(request.base_seed)?;
        let output_device = request.placement.output_device();
        let node = format!(
            "{}@{}:{}:{}:{}:{}",
            request.node,
            request.execution_ordinal,
            contract.rng_id,
            request.placement.identity(),
            device_kind_name(output_device.kind()),
            output_device.ordinal()
        );
        let address = RngStreamAddress::for_device(
            request.workflow,
            request.attempt,
            node,
            request.output,
            contract.phase.as_str(),
            request.batch,
            request.retry,
            request.retry_policy,
            generation_device,
        )?;
        let stream = RngStream::new(profile.stream_profile(), profile.algorithm(), seed, address)?;
        let transaction = stream.begin(checkpoint)?;
        Ok(Self {
            contract,
            profile,
            placement: request.placement,
            transaction,
        })
    }

    pub const fn contract(&self) -> RngCompatibilityContract {
        self.contract
    }

    pub const fn profile(&self) -> NativeRngExecutionProfile {
        self.profile
    }

    pub const fn generation_device(&self) -> DeviceId {
        self.placement.generation_device()
    }

    pub const fn output_device(&self) -> DeviceId {
        self.placement.output_device()
    }

    pub fn checkpoint(&self) -> RngCheckpoint {
        self.transaction.checkpoint()
    }

    pub fn commit(self) -> RngCheckpoint {
        self.transaction.commit()
    }

    pub fn draw_uniform(
        &mut self,
        count: usize,
        cancellation: &CancellationToken,
    ) -> Result<Vec<f64>, RngCompatibilityError> {
        self.require_operation(&[
            RngCompatibilityOperation::Uniform,
            RngCompatibilityOperation::UniformInitializer,
        ])?;
        self.transactional_values(count, cancellation, |transaction, cancellation| {
            transaction.next_unit_f64(cancellation).map_err(Into::into)
        })
    }

    pub fn draw_normal(
        &mut self,
        count: usize,
        cancellation: &CancellationToken,
    ) -> Result<Vec<f64>, RngCompatibilityError> {
        self.require_operation(&[
            RngCompatibilityOperation::Normal,
            RngCompatibilityOperation::NormalLike,
            RngCompatibilityOperation::NormalInitializer,
        ])?;
        let mut candidate = self.transaction.clone();
        let mut output = allocate(count, "normal output")?;
        while output.len() < count {
            cancellation
                .check()
                .map_err(|_| RngCompatibilityError::Cancelled)?;
            let pair = candidate.next_standard_normal_pair(cancellation)?;
            output.push(pair[0]);
            if output.len() < count {
                output.push(pair[1]);
            }
        }
        cancellation
            .check()
            .map_err(|_| RngCompatibilityError::Cancelled)?;
        self.transaction = candidate;
        Ok(output)
    }

    pub fn draw_integers(
        &mut self,
        low: i64,
        high: i64,
        count: usize,
        cancellation: &CancellationToken,
    ) -> Result<Vec<i64>, RngCompatibilityError> {
        self.require_operation(&[RngCompatibilityOperation::Integer])?;
        let width = high
            .checked_sub(low)
            .ok_or(RngCompatibilityError::InvalidRange)?;
        let width = u64::try_from(width).map_err(|_| RngCompatibilityError::InvalidRange)?;
        if width == 0 {
            return Err(RngCompatibilityError::InvalidRange);
        }
        let mut candidate = self.transaction.clone();
        let mut output = allocate(count, "integer output")?;
        for _ in 0..count {
            cancellation
                .check()
                .map_err(|_| RngCompatibilityError::Cancelled)?;
            let offset = candidate.next_bounded_u64(width, cancellation)?;
            let offset = i64::try_from(offset).map_err(|_| RngCompatibilityError::InvalidRange)?;
            output.push(
                low.checked_add(offset)
                    .ok_or(RngCompatibilityError::InvalidRange)?,
            );
        }
        cancellation
            .check()
            .map_err(|_| RngCompatibilityError::Cancelled)?;
        self.transaction = candidate;
        Ok(output)
    }

    pub fn draw_permutation(
        &mut self,
        count: usize,
        cancellation: &CancellationToken,
    ) -> Result<Vec<usize>, RngCompatibilityError> {
        self.require_operation(&[RngCompatibilityOperation::Permutation])?;
        let mut candidate = self.transaction.clone();
        let mut output = Vec::new();
        output
            .try_reserve_exact(count)
            .map_err(|_| RngCompatibilityError::AllocationFailed {
                allocation: "permutation",
                count,
            })?;
        output.extend(0..count);
        for upper in (1..count).rev() {
            cancellation
                .check()
                .map_err(|_| RngCompatibilityError::Cancelled)?;
            let bound =
                u64::try_from(upper + 1).map_err(|_| RngCompatibilityError::InvalidCount)?;
            let selected = candidate.next_bounded_u64(bound, cancellation)?;
            let selected =
                usize::try_from(selected).map_err(|_| RngCompatibilityError::InvalidCount)?;
            output.swap(upper, selected);
        }
        cancellation
            .check()
            .map_err(|_| RngCompatibilityError::Cancelled)?;
        self.transaction = candidate;
        Ok(output)
    }

    pub fn draw_choice<T: Clone>(
        &mut self,
        values: &[T],
        count: usize,
        replacement: bool,
        cancellation: &CancellationToken,
    ) -> Result<Vec<T>, RngCompatibilityError> {
        self.require_operation(&[RngCompatibilityOperation::Choice])?;
        if values.is_empty() || (!replacement && count > values.len()) {
            return Err(RngCompatibilityError::InvalidChoice);
        }
        let mut candidate = self.transaction.clone();
        let mut available = values.to_vec();
        let mut output = allocate(count, "choice output")?;
        for _ in 0..count {
            cancellation
                .check()
                .map_err(|_| RngCompatibilityError::Cancelled)?;
            let bound =
                u64::try_from(available.len()).map_err(|_| RngCompatibilityError::InvalidCount)?;
            let index = candidate.next_bounded_u64(bound, cancellation)?;
            let index = usize::try_from(index).map_err(|_| RngCompatibilityError::InvalidCount)?;
            let selected = available
                .get(index)
                .cloned()
                .ok_or(RngCompatibilityError::InvalidChoice)?;
            output.push(selected);
            if !replacement {
                available.remove(index);
            }
        }
        cancellation
            .check()
            .map_err(|_| RngCompatibilityError::Cancelled)?;
        self.transaction = candidate;
        Ok(output)
    }

    pub fn draw_multinomial(
        &mut self,
        weights: &[f64],
        count: usize,
        replacement: bool,
        cancellation: &CancellationToken,
    ) -> Result<Vec<usize>, RngCompatibilityError> {
        self.require_operation(&[RngCompatibilityOperation::Multinomial])?;
        validate_weights(weights, count, replacement)?;
        let mut candidate = self.transaction.clone();
        let mut available = weights.to_vec();
        let mut output = allocate(count, "multinomial output")?;
        for _ in 0..count {
            cancellation
                .check()
                .map_err(|_| RngCompatibilityError::Cancelled)?;
            let sum = available.iter().sum::<f64>();
            let target = candidate.next_unit_f64(cancellation)? * sum;
            let mut cumulative = 0.0;
            let mut fallback = None;
            let mut selected = None;
            for (index, weight) in available.iter().copied().enumerate() {
                if weight > 0.0 {
                    fallback = Some(index);
                    cumulative += weight;
                    if target < cumulative {
                        selected = Some(index);
                        break;
                    }
                }
            }
            let selected = selected
                .or(fallback)
                .ok_or(RngCompatibilityError::InvalidWeights)?;
            output.push(selected);
            if !replacement {
                let weight = available
                    .get_mut(selected)
                    .ok_or(RngCompatibilityError::InvalidWeights)?;
                *weight = 0.0;
            }
        }
        cancellation
            .check()
            .map_err(|_| RngCompatibilityError::Cancelled)?;
        self.transaction = candidate;
        Ok(output)
    }

    pub fn stochastic_round_up(
        &mut self,
        fractional_parts: &[f64],
        cancellation: &CancellationToken,
    ) -> Result<Vec<bool>, RngCompatibilityError> {
        if self.contract.phase != RngCompatibilityPhase::StochasticQuantization
            || self.contract.operation != RngCompatibilityOperation::Uniform
        {
            return Err(self.operation_error(&[RngCompatibilityOperation::Uniform]));
        }
        if fractional_parts
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(RngCompatibilityError::InvalidProbability);
        }
        let mut candidate = self.transaction.clone();
        let mut output = allocate(fractional_parts.len(), "stochastic rounding output")?;
        for probability in fractional_parts {
            cancellation
                .check()
                .map_err(|_| RngCompatibilityError::Cancelled)?;
            output.push(candidate.next_unit_f64(cancellation)? < *probability);
        }
        cancellation
            .check()
            .map_err(|_| RngCompatibilityError::Cancelled)?;
        self.transaction = candidate;
        Ok(output)
    }

    pub fn sobol_engine(
        &mut self,
        dimension: usize,
        scramble: bool,
        cancellation: &CancellationToken,
    ) -> Result<SobolEngine, RngCompatibilityError> {
        self.require_operation(&[RngCompatibilityOperation::SobolEngine])?;
        let mut candidate = self.transaction.clone();
        let high = u64::from(candidate.next_u32(cancellation)?);
        let low = u64::from(candidate.next_u32(cancellation)?);
        let engine = SobolEngine::new(dimension, scramble, (high << 32) | low)?;
        cancellation
            .check()
            .map_err(|_| RngCompatibilityError::Cancelled)?;
        self.transaction = candidate;
        Ok(engine)
    }

    pub fn sobol_draw(
        &self,
        engine: &mut SobolEngine,
        count: usize,
        cancellation: &CancellationToken,
    ) -> Result<Vec<f32>, RngCompatibilityError> {
        self.require_operation(&[RngCompatibilityOperation::SobolDraw])?;
        let mut candidate = engine.clone();
        let output = candidate.draw(count, cancellation)?;
        cancellation
            .check()
            .map_err(|_| RngCompatibilityError::Cancelled)?;
        *engine = candidate;
        Ok(output)
    }

    pub fn brownian_tree(
        &mut self,
        start: f64,
        initial: Vec<f64>,
        end: f64,
        cancellation: &CancellationToken,
    ) -> Result<BrownianTree, RngCompatibilityError> {
        self.require_operation(&[RngCompatibilityOperation::BrownianTree])?;
        let mut candidate = self.transaction.clone();
        let high = u64::from(candidate.next_u32(cancellation)?);
        let low = u64::from(candidate.next_u32(cancellation)?);
        let tree = BrownianTree::new(start, initial, end, (high << 32) | low)?;
        cancellation
            .check()
            .map_err(|_| RngCompatibilityError::Cancelled)?;
        self.transaction = candidate;
        Ok(tree)
    }

    fn transactional_values<T>(
        &mut self,
        count: usize,
        cancellation: &CancellationToken,
        mut next: impl FnMut(
            &mut RngTransaction,
            &CancellationToken,
        ) -> Result<T, RngCompatibilityError>,
    ) -> Result<Vec<T>, RngCompatibilityError> {
        let mut candidate = self.transaction.clone();
        let mut output = allocate(count, "RNG output")?;
        for _ in 0..count {
            cancellation
                .check()
                .map_err(|_| RngCompatibilityError::Cancelled)?;
            output.push(next(&mut candidate, cancellation)?);
        }
        cancellation
            .check()
            .map_err(|_| RngCompatibilityError::Cancelled)?;
        self.transaction = candidate;
        Ok(output)
    }

    fn require_operation(
        &self,
        expected: &'static [RngCompatibilityOperation],
    ) -> Result<(), RngCompatibilityError> {
        if expected.contains(&self.contract.operation) {
            Ok(())
        } else {
            Err(self.operation_error(expected))
        }
    }

    fn operation_error(
        &self,
        expected: &'static [RngCompatibilityOperation],
    ) -> RngCompatibilityError {
        RngCompatibilityError::UnsupportedOperation {
            rng_id: self.contract.rng_id,
            actual: self.contract.operation,
            expected,
        }
    }
}

fn allocate<T>(count: usize, allocation: &'static str) -> Result<Vec<T>, RngCompatibilityError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(|_| RngCompatibilityError::AllocationFailed { allocation, count })?;
    Ok(output)
}

fn validate_weights(
    weights: &[f64],
    count: usize,
    replacement: bool,
) -> Result<(), RngCompatibilityError> {
    if count == 0
        || weights.is_empty()
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        || weights.iter().all(|weight| *weight == 0.0)
        || (!replacement && count > weights.iter().filter(|weight| **weight > 0.0).count())
    {
        return Err(RngCompatibilityError::InvalidWeights);
    }
    Ok(())
}

fn validate_seed_transform(
    contract: RngCompatibilityContract,
    transform: RngSeedTransform,
) -> Result<(), RngCompatibilityError> {
    let expressions = contract.seed_expressions;
    let accepted = if expressions == "123" {
        transform == RngSeedTransform::Fixed(123)
    } else if expressions == "seed % (2 ** 32 - 1)" {
        transform == RngSeedTransform::NumpyModulo32MinusOne
    } else if expressions.contains(" - 10") {
        matches!(
            transform,
            RngSeedTransform::TorchSigned64 | RngSeedTransform::Add(-10)
        )
    } else if expressions.contains("block_idx * 1000") {
        matches!(
            transform,
            RngSeedTransform::TorchSigned64
                | RngSeedTransform::Add(1)
                | RngSeedTransform::BlockOffset {
                    block_stride: 1_000,
                    ..
                }
        )
    } else if expressions.is_empty() {
        true
    } else {
        matches!(
            transform,
            RngSeedTransform::TorchSigned64 | RngSeedTransform::WrapUnsigned64
        )
    };
    if !accepted {
        return Err(RngCompatibilityError::SeedTransformMismatch {
            rng_id: contract.rng_id,
            expressions,
            transform,
        });
    }
    Ok(())
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum RngCompatibilityError {
    #[error("unknown RNG compatibility contract {0}")]
    UnknownContract(String),
    #[error("RNG compatibility contract {rng_id} is developer-only")]
    DeveloperOnlyContract { rng_id: &'static str },
    #[error("RNG compatibility contract {rng_id} requires an enabled conditional capability")]
    ConditionalContractUnavailable { rng_id: &'static str },
    #[error("RNG compatibility contract {rng_id} requires CPU generation, not {device:?}")]
    UnsupportedDevice {
        rng_id: &'static str,
        device: DeviceId,
    },
    #[error(
        "RNG compatibility contract {rng_id} operation {actual:?} cannot perform any of {expected:?}"
    )]
    UnsupportedOperation {
        rng_id: &'static str,
        actual: RngCompatibilityOperation,
        expected: &'static [RngCompatibilityOperation],
    },
    #[error("invalid compatibility seed: {reason}")]
    InvalidSeed { reason: String },
    #[error(
        "RNG compatibility contract {rng_id} seed expressions {expressions:?} reject transform {transform:?}"
    )]
    SeedTransformMismatch {
        rng_id: &'static str,
        expressions: &'static str,
        transform: RngSeedTransform,
    },
    #[error("RNG operation was cancelled before state publication")]
    Cancelled,
    #[error("invalid RNG element count")]
    InvalidCount,
    #[error("integer RNG range must satisfy low < high without overflow")]
    InvalidRange,
    #[error("choice input must be nonempty and large enough when sampling without replacement")]
    InvalidChoice,
    #[error("multinomial weights or requested sample count are invalid")]
    InvalidWeights,
    #[error("stochastic-rounding probabilities must be finite and in the inclusive range 0..=1")]
    InvalidProbability,
    #[error("could not allocate {count} elements for {allocation}")]
    AllocationFailed {
        allocation: &'static str,
        count: usize,
    },
    #[error(transparent)]
    Canonical(#[from] RngError),
}
