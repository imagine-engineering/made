use crate::{DType, Layout};
#[path = "../operation_contract_evidence.rs"]
mod operation_contract_evidence;
use operation_contract_evidence::{
    ResolutionExpectation, valid_evidence_fixture_path, valid_module_name,
    validate_resolution_evidence, validate_resolution_semantics,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::OnceLock,
};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractInventoryKind {
    CallableOperation,
    ReclassifiedExternalOperation,
    NamespaceValueReference,
    TypeReference,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractResolutionState {
    ResolvedCallable,
    ResolvedReference,
    ReclassifiedExternalOperation,
    BlockedReceiverUnverified,
    BlockedMissingSemanticsProfile,
    BlockedMissingOracleDependency,
}

impl ContractResolutionState {
    pub const fn is_blocked(self) -> bool {
        matches!(
            self,
            Self::BlockedReceiverUnverified
                | Self::BlockedMissingSemanticsProfile
                | Self::BlockedMissingOracleDependency
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OperationContractRecord {
    pub(crate) operation_id: &'static str,
    pub(crate) overload_id: &'static str,
    pub(crate) inventory_kind: ContractInventoryKind,
    pub(crate) canonical_target: &'static str,
    pub(crate) resolution_state: ContractResolutionState,
    pub(crate) blocker_reason: &'static str,
    pub(crate) call_style: &'static str,
    pub(crate) ordered_parameters_json: &'static str,
    pub(crate) output_arity: &'static str,
    pub(crate) output_types_json: &'static str,
    pub(crate) rust_signature: &'static str,
    pub(crate) reference_semantic: &'static str,
    pub(crate) resolution_owner_task_id: &'static str,
    pub(crate) expected_resolution_module: &'static str,
    pub(crate) release_closure_required: bool,
    pub(crate) mutation_rule: &'static str,
    pub(crate) alias_rule: &'static str,
    pub(crate) shape_rule: &'static str,
    pub(crate) dtype_rule: &'static str,
    pub(crate) accumulation_dtype: &'static str,
    pub(crate) layout_rule: &'static str,
    pub(crate) device_rule: &'static str,
    pub(crate) numeric_rule: &'static str,
    pub(crate) tolerance: &'static str,
    pub(crate) determinism: &'static str,
    pub(crate) cancellation_points: &'static str,
    pub(crate) vjp_rule: &'static str,
    pub(crate) jvp_rule: &'static str,
    pub(crate) source_call_sites: &'static str,
    pub(crate) oracle_fixture: &'static str,
    pub(crate) oracle_fixture_sha256: &'static str,
    pub(crate) evidence: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedOperationContract {
    pub resolution_module: &'static str,
    pub operation_id: &'static str,
    pub baseline_overload_id: &'static str,
    pub baseline_fixture_sha256: &'static str,
    pub overload_id: &'static str,
    pub ordered_parameters_json: &'static str,
    pub output_arity: &'static str,
    pub output_types_json: &'static str,
    pub rust_signature: &'static str,
    pub mutation_rule: &'static str,
    pub alias_rule: &'static str,
    pub shape_rule: &'static str,
    pub dtype_rule: &'static str,
    pub accumulation_dtype: &'static str,
    pub layout_rule: &'static str,
    pub device_rule: &'static str,
    pub numeric_rule: &'static str,
    pub tolerance: &'static str,
    pub determinism: &'static str,
    pub cancellation_points: &'static str,
    pub vjp_rule: &'static str,
    pub jvp_rule: &'static str,
    pub owner_task_id: &'static str,
    pub evidence_fixture: &'static str,
    pub evidence_fixture_sha256: &'static str,
    build_seal: OperationResolutionBuildSeal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OperationResolutionBuildSeal;

const OPERATION_RESOLUTION_BUILD_SEAL: OperationResolutionBuildSeal = OperationResolutionBuildSeal;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TypedReferenceContract {
    operation_id: &'static str,
    canonical_target: &'static str,
    inventory_kind: ContractInventoryKind,
    semantic: CanonicalReference,
}

impl TypedReferenceContract {
    pub const fn operation_id(self) -> &'static str {
        self.operation_id
    }

    pub const fn canonical_target(self) -> &'static str {
        self.canonical_target
    }

    pub const fn inventory_kind(self) -> ContractInventoryKind {
        self.inventory_kind
    }

    pub const fn semantic(self) -> CanonicalReference {
        self.semantic
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MemoryFormatReference {
    Layout(Layout),
    PreserveFormat,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BooleanCapabilityReference {
    CudaMatmulAllowFp16Accumulation,
    CudaMatmulAllowTf32,
    CudnnAllowTf32,
    CudnnBenchmark,
    CudnnEnabled,
    XpuHasFp16,
    XformersHasCppLibrary,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NumericConstantReference {
    FloatInfoBits,
    FloatInfoEpsilon,
    FloatInfoMaximum,
    FloatInfoMinimum,
    Infinity,
    Pi,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FunctionReference {
    AutogradOnceDifferentiable,
    Log10,
    Hardswish,
    Hardtanh,
    Mish,
    Selu,
    Softsign,
    XpuStream,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TypeMarkerReference {
    ComfyCastWeightBiasOp,
    ComfyDisableWeightInit,
    ComfyDisableWeightInitRmsNorm,
    ComfyManualCast,
    AcceleratorError,
    LongTensor,
    AutogradFunction,
    CudaOutOfMemoryError,
    DType,
    EmptyTensorDevice,
    JitFinal,
    ConvTranspose1d,
    ConvTranspose2d,
    RmsNorm,
    Optimizer,
    LearningRateScheduler,
    Dataset,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NamespaceReference {
    ComfyOps,
    TorchPackagePath,
    TorchNeuralNetwork,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TensorPropertyReference {
    InverseFftReal,
    MedianValues,
    UniqueShape,
    VandermondeTranspose,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DevicePropertyReference {
    CudaGcnArchitectureName,
    XpuTotalMemory,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum EnumVariantReference {
    SdpCudnnAttention,
    SdpEfficientAttention,
    SdpFlashAttention,
    SdpMath,
    InterpolationNearest,
    FunctionalInterpolationBicubic,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VersionValueReference {
    Torch,
    Cuda,
    Hip,
    Xformers,
    XformersModule,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CanonicalReference {
    DType(DType),
    MemoryFormat(MemoryFormatReference),
    BooleanCapability(BooleanCapabilityReference),
    NumericConstant(NumericConstantReference),
    Function(FunctionReference),
    TypeMarker(TypeMarkerReference),
    Namespace(NamespaceReference),
    TensorProperty(TensorPropertyReference),
    DeviceProperty(DevicePropertyReference),
    EnumVariant(EnumVariantReference),
    VersionValue(VersionValueReference),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeneratedOperationResolutionSlice {
    pub module_name: &'static str,
    pub contracts: &'static [ResolvedOperationContract],
}

impl GeneratedOperationResolutionSlice {
    pub fn iter(&self) -> std::slice::Iter<'static, ResolvedOperationContract> {
        self.contracts.iter()
    }

    pub const fn len(&self) -> usize {
        self.contracts.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.contracts.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ReferenceSemanticCategory {
    NotApplicable,
    DType,
    LayoutOrMemoryFormat,
    BooleanCapability,
    NumericConstant,
    FunctionReference,
    TypeMarker,
    Namespace,
    TensorProperty,
    DeviceProperty,
    EnumVariant,
    VersionValue,
}

impl ReferenceSemanticCategory {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "not-applicable" => Some(Self::NotApplicable),
            "dtype" => Some(Self::DType),
            "layout-or-memory-format" => Some(Self::LayoutOrMemoryFormat),
            "boolean-capability" => Some(Self::BooleanCapability),
            "numeric-constant" => Some(Self::NumericConstant),
            "function-reference" => Some(Self::FunctionReference),
            "type-marker" => Some(Self::TypeMarker),
            "namespace" => Some(Self::Namespace),
            "tensor-property" => Some(Self::TensorProperty),
            "device-property" => Some(Self::DeviceProperty),
            "enum-variant" => Some(Self::EnumVariant),
            "version-value" => Some(Self::VersionValue),
            _ => None,
        }
    }
}

impl OperationContractRecord {
    pub fn typed_reference(self) -> Option<TypedReferenceContract> {
        match self.inventory_kind {
            ContractInventoryKind::CallableOperation
            | ContractInventoryKind::ReclassifiedExternalOperation => None,
            ContractInventoryKind::NamespaceValueReference
            | ContractInventoryKind::TypeReference => {
                let semantic = self.canonical_reference()?;
                Some(TypedReferenceContract {
                    operation_id: self.operation_id,
                    canonical_target: self.canonical_target,
                    inventory_kind: self.inventory_kind,
                    semantic,
                })
            }
        }
    }

    pub fn reference_semantic_category(self) -> Option<ReferenceSemanticCategory> {
        let (category, value) = parse_reference_semantic(self.reference_semantic)?;
        (value == self.canonical_target).then_some(category)
    }

    fn canonical_reference(self) -> Option<CanonicalReference> {
        let category = self.reference_semantic_category()?;
        let target = self.canonical_target;
        match category {
            ReferenceSemanticCategory::NotApplicable => None,
            ReferenceSemanticCategory::DType => Some(match target {
                "torch.bfloat16" => CanonicalReference::DType(DType::Bf16),
                "torch.bool" => CanonicalReference::DType(DType::Bool),
                "torch.complex64" => CanonicalReference::DType(DType::Complex64),
                "torch.float" | "torch.float32" => CanonicalReference::DType(DType::F32),
                "torch.float16" => CanonicalReference::DType(DType::F16),
                "torch.float64" => CanonicalReference::DType(DType::F64),
                "torch.float8_e4m3fn" => CanonicalReference::DType(DType::Float8E4m3Fn),
                "torch.float8_e5m2" => CanonicalReference::DType(DType::Float8E5m2),
                "torch.float8_e4m3fnuz" => CanonicalReference::DType(DType::Float8E4m3Fnuz),
                "torch.float8_e5m2fnuz" => CanonicalReference::DType(DType::Float8E5m2Fnuz),
                "torch.float8_e8m0fnu" => CanonicalReference::DType(DType::Float8E8m0Fnu),
                "torch.int" | "torch.int32" => CanonicalReference::DType(DType::I32),
                "torch.int16" => CanonicalReference::DType(DType::I16),
                "torch.int64" | "torch.long" => CanonicalReference::DType(DType::I64),
                "torch.int8" => CanonicalReference::DType(DType::I8),
                "torch.uint16" => CanonicalReference::DType(DType::U16),
                "torch.uint32" => CanonicalReference::DType(DType::U32),
                "torch.uint64" => CanonicalReference::DType(DType::U64),
                "torch.uint8" => CanonicalReference::DType(DType::U8),
                _ => return None,
            }),
            ReferenceSemanticCategory::LayoutOrMemoryFormat => Some(match target {
                "torch.channels_last" => CanonicalReference::MemoryFormat(
                    MemoryFormatReference::Layout(Layout::ChannelsLast),
                ),
                "torch.preserve_format" => {
                    CanonicalReference::MemoryFormat(MemoryFormatReference::PreserveFormat)
                }
                _ => return None,
            }),
            ReferenceSemanticCategory::BooleanCapability => {
                Some(CanonicalReference::BooleanCapability(match target {
                    "torch.backends.cuda.matmul.allow_fp16_accumulation" => {
                        BooleanCapabilityReference::CudaMatmulAllowFp16Accumulation
                    }
                    "torch.backends.cuda.matmul.allow_tf32" => {
                        BooleanCapabilityReference::CudaMatmulAllowTf32
                    }
                    "torch.backends.cudnn.allow_tf32" => BooleanCapabilityReference::CudnnAllowTf32,
                    "torch.backends.cudnn.benchmark" => BooleanCapabilityReference::CudnnBenchmark,
                    "torch.backends.cudnn.enabled" => BooleanCapabilityReference::CudnnEnabled,
                    "torch.xpu.get_device_properties().has_fp16" => {
                        BooleanCapabilityReference::XpuHasFp16
                    }
                    "xformers._has_cpp_library" => {
                        BooleanCapabilityReference::XformersHasCppLibrary
                    }
                    _ => return None,
                }))
            }
            ReferenceSemanticCategory::NumericConstant => {
                Some(CanonicalReference::NumericConstant(match target {
                    "torch.finfo().bits" => NumericConstantReference::FloatInfoBits,
                    "torch.finfo().eps" => NumericConstantReference::FloatInfoEpsilon,
                    "torch.finfo().max" => NumericConstantReference::FloatInfoMaximum,
                    "torch.finfo().min" => NumericConstantReference::FloatInfoMinimum,
                    "torch.inf" => NumericConstantReference::Infinity,
                    "torch.pi" => NumericConstantReference::Pi,
                    _ => return None,
                }))
            }
            ReferenceSemanticCategory::FunctionReference => {
                Some(CanonicalReference::Function(match target {
                    "torch.autograd.function.once_differentiable" => {
                        FunctionReference::AutogradOnceDifferentiable
                    }
                    "torch.log10" => FunctionReference::Log10,
                    "torch.nn.Hardswish" => FunctionReference::Hardswish,
                    "torch.nn.Hardtanh" => FunctionReference::Hardtanh,
                    "torch.nn.Mish" => FunctionReference::Mish,
                    "torch.nn.SELU" => FunctionReference::Selu,
                    "torch.nn.Softsign" => FunctionReference::Softsign,
                    "torch.xpu.stream" => FunctionReference::XpuStream,
                    _ => return None,
                }))
            }
            ReferenceSemanticCategory::TypeMarker => {
                Some(CanonicalReference::TypeMarker(match target {
                    "comfy.ops.CastWeightBiasOp" => TypeMarkerReference::ComfyCastWeightBiasOp,
                    "comfy.ops.disable_weight_init" => TypeMarkerReference::ComfyDisableWeightInit,
                    "comfy.ops.disable_weight_init.RMSNorm" => {
                        TypeMarkerReference::ComfyDisableWeightInitRmsNorm
                    }
                    "comfy.ops.manual_cast" => TypeMarkerReference::ComfyManualCast,
                    "torch.AcceleratorError" => TypeMarkerReference::AcceleratorError,
                    "torch.LongTensor" => TypeMarkerReference::LongTensor,
                    "torch.autograd.Function" => TypeMarkerReference::AutogradFunction,
                    "torch.cuda.OutOfMemoryError" => TypeMarkerReference::CudaOutOfMemoryError,
                    "torch.dtype" => TypeMarkerReference::DType,
                    "torch.empty().device" => TypeMarkerReference::EmptyTensorDevice,
                    "torch.jit.Final" => TypeMarkerReference::JitFinal,
                    "torch.nn.ConvTranspose1d" => TypeMarkerReference::ConvTranspose1d,
                    "torch.nn.ConvTranspose2d" => TypeMarkerReference::ConvTranspose2d,
                    "torch.nn.RMSNorm" => TypeMarkerReference::RmsNorm,
                    "torch.optim.Optimizer" => TypeMarkerReference::Optimizer,
                    "torch.optim.lr_scheduler._LRScheduler" => {
                        TypeMarkerReference::LearningRateScheduler
                    }
                    "torch.utils.data.Dataset" => TypeMarkerReference::Dataset,
                    _ => return None,
                }))
            }
            ReferenceSemanticCategory::Namespace => {
                Some(CanonicalReference::Namespace(match target {
                    "comfy.ops" => NamespaceReference::ComfyOps,
                    "torch.__path__" => NamespaceReference::TorchPackagePath,
                    "torch.nn" => NamespaceReference::TorchNeuralNetwork,
                    _ => return None,
                }))
            }
            ReferenceSemanticCategory::TensorProperty => {
                Some(CanonicalReference::TensorProperty(match target {
                    "torch.fft.ifftn().real" => TensorPropertyReference::InverseFftReal,
                    "torch.median().values" => TensorPropertyReference::MedianValues,
                    "torch.unique().shape" => TensorPropertyReference::UniqueShape,
                    "torch.vander().T" => TensorPropertyReference::VandermondeTranspose,
                    _ => return None,
                }))
            }
            ReferenceSemanticCategory::DeviceProperty => {
                Some(CanonicalReference::DeviceProperty(match target {
                    "torch.cuda.get_device_properties().gcnArchName" => {
                        DevicePropertyReference::CudaGcnArchitectureName
                    }
                    "torch.xpu.get_device_properties().total_memory" => {
                        DevicePropertyReference::XpuTotalMemory
                    }
                    _ => return None,
                }))
            }
            ReferenceSemanticCategory::EnumVariant => {
                Some(CanonicalReference::EnumVariant(match target {
                    "torch.nn.attention.SDPBackend.CUDNN_ATTENTION" => {
                        EnumVariantReference::SdpCudnnAttention
                    }
                    "torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION" => {
                        EnumVariantReference::SdpEfficientAttention
                    }
                    "torch.nn.attention.SDPBackend.FLASH_ATTENTION" => {
                        EnumVariantReference::SdpFlashAttention
                    }
                    "torch.nn.attention.SDPBackend.MATH" => EnumVariantReference::SdpMath,
                    "torchvision.transforms.InterpolationMode.NEAREST" => {
                        EnumVariantReference::InterpolationNearest
                    }
                    "torchvision.transforms.functional.InterpolationMode.BICUBIC" => {
                        EnumVariantReference::FunctionalInterpolationBicubic
                    }
                    _ => return None,
                }))
            }
            ReferenceSemanticCategory::VersionValue => {
                Some(CanonicalReference::VersionValue(match target {
                    "torch.version.__version__" => VersionValueReference::Torch,
                    "torch.version.cuda" => VersionValueReference::Cuda,
                    "torch.version.hip" => VersionValueReference::Hip,
                    "xformers.__version__" => VersionValueReference::Xformers,
                    "xformers.version.__version__" => VersionValueReference::XformersModule,
                    _ => return None,
                }))
            }
        }
    }
}

fn parse_reference_semantic(value: &str) -> Option<(ReferenceSemanticCategory, &str)> {
    let value = value.strip_prefix("{\"category\":\"")?;
    let (category, value) = value.split_once("\",\"value\":\"")?;
    let value = value.strip_suffix("\"}")?;
    Some((ReferenceSemanticCategory::parse(category)?, value))
}

pub(crate) fn compiled_resolution_by_identifier(
    identifier: &str,
) -> Option<&'static ResolvedOperationContract> {
    static REGISTRY_IS_VALID: OnceLock<bool> = OnceLock::new();
    if !*REGISTRY_IS_VALID.get_or_init(|| {
        validate_operation_resolution_iter(
            OPERATION_CONTRACTS,
            GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
                .iter()
                .flat_map(|slice| {
                    slice
                        .iter()
                        .map(move |resolution| (slice.module_name, resolution))
                }),
        )
        .is_ok()
            && GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
                .iter()
                .flat_map(GeneratedOperationResolutionSlice::iter)
                .all(is_build_sealed_resolution)
    }) {
        return None;
    }
    for resolution in GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
        .iter()
        .flat_map(|slice| slice.iter())
    {
        let baseline = OPERATION_CONTRACTS
            .iter()
            .find(|baseline| baseline.operation_id == resolution.operation_id)?;
        if identifier == baseline.operation_id || identifier == resolution.overload_id {
            return Some(resolution);
        }
    }
    None
}

fn is_build_sealed_resolution(resolution: &ResolvedOperationContract) -> bool {
    resolution.build_seal == OPERATION_RESOLUTION_BUILD_SEAL
        && GENERATED_BUILD_SEALED_OPERATION_RESOLUTIONS.iter().any(
            |(operation_id, overload_id, module_name, evidence_digest)| {
                *operation_id == resolution.operation_id
                    && *overload_id == resolution.overload_id
                    && *module_name == resolution.resolution_module
                    && *evidence_digest == resolution.evidence_fixture_sha256
            },
        )
}

pub fn resolve_operation_identifier<'a>(
    contracts: &[OperationContractRecord],
    resolutions: &'a [ResolvedOperationContract],
    identifier: &str,
) -> Result<Option<&'a ResolvedOperationContract>, OperationContractTableError> {
    validate_operation_resolution_iter(
        contracts,
        resolutions
            .iter()
            .map(|resolution| (resolution.resolution_module, resolution)),
    )?;
    for resolution in resolutions {
        let Some(baseline) = contracts
            .iter()
            .find(|baseline| baseline.operation_id == resolution.operation_id)
        else {
            return Err(OperationContractTableError::UnknownCompiledResolution(
                resolution.operation_id.to_owned(),
            ));
        };
        if identifier == baseline.operation_id || identifier == resolution.overload_id {
            return Ok(Some(resolution));
        }
    }
    Ok(None)
}

fn resolution_discharges_baseline(
    resolution: &ResolvedOperationContract,
    baseline: &OperationContractRecord,
) -> bool {
    baseline.inventory_kind == ContractInventoryKind::CallableOperation
        && baseline.resolution_state.is_blocked()
        && baseline.release_closure_required
        && resolution.operation_id == baseline.operation_id
        && resolution.baseline_overload_id == baseline.overload_id
        && resolution.baseline_fixture_sha256 == baseline.oracle_fixture_sha256
        && resolution.overload_id != baseline.overload_id
        && !resolution
            .overload_id
            .to_ascii_lowercase()
            .contains("blocked")
        && resolution.owner_task_id == baseline.resolution_owner_task_id
        && resolution.resolution_module == baseline.expected_resolution_module
        && valid_evidence_fixture_path(resolution.evidence_fixture, resolution.resolution_module)
        && is_lowercase_sha256(resolution.baseline_fixture_sha256)
        && is_lowercase_sha256(resolution.evidence_fixture_sha256)
        && resolution.evidence_fixture_sha256 != resolution.baseline_fixture_sha256
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn resolution_expectation(resolution: &ResolvedOperationContract) -> ResolutionExpectation<'_> {
    ResolutionExpectation {
        resolution_module: resolution.resolution_module,
        operation_id: resolution.operation_id,
        baseline_overload_id: resolution.baseline_overload_id,
        baseline_fixture_sha256: resolution.baseline_fixture_sha256,
        overload_id: resolution.overload_id,
        ordered_parameters_json: resolution.ordered_parameters_json,
        output_arity: resolution.output_arity,
        output_types_json: resolution.output_types_json,
        rust_signature: resolution.rust_signature,
        mutation_rule: resolution.mutation_rule,
        alias_rule: resolution.alias_rule,
        shape_rule: resolution.shape_rule,
        dtype_rule: resolution.dtype_rule,
        accumulation_dtype: resolution.accumulation_dtype,
        layout_rule: resolution.layout_rule,
        device_rule: resolution.device_rule,
        numeric_rule: resolution.numeric_rule,
        tolerance: resolution.tolerance,
        determinism: resolution.determinism,
        cancellation_points: resolution.cancellation_points,
        vjp_rule: resolution.vjp_rule,
        jvp_rule: resolution.jvp_rule,
        owner_task_id: resolution.owner_task_id,
        evidence_fixture: resolution.evidence_fixture,
        evidence_fixture_sha256: resolution.evidence_fixture_sha256,
    }
}

pub fn validate_operation_resolution_evidence(
    workspace_root: &Path,
    resolution: &ResolvedOperationContract,
) -> Result<(), OperationContractTableError> {
    validate_resolution_evidence(workspace_root, &resolution_expectation(resolution)).map_err(
        |_| {
            OperationContractTableError::InvalidResolutionEvidence(
                resolution.operation_id.to_owned(),
            )
        },
    )
}

pub fn validate_operation_contracts(
    contracts: &[OperationContractRecord],
) -> Result<(), OperationContractTableError> {
    let mut operation_ids = HashSet::with_capacity(contracts.len());
    let mut overload_ids = HashSet::with_capacity(contracts.len());
    let mut owner_modules = HashMap::new();
    let mut module_owners = HashMap::new();
    for contract in contracts {
        if !operation_ids.insert(contract.operation_id) {
            return Err(OperationContractTableError::DuplicateOperationId(
                contract.operation_id.to_owned(),
            ));
        }
        if !overload_ids.insert(contract.overload_id) {
            return Err(OperationContractTableError::DuplicateOverloadId(
                contract.overload_id.to_owned(),
            ));
        }
        for (field, value) in [
            ("operation_id", contract.operation_id),
            ("overload_id", contract.overload_id),
            ("canonical_target", contract.canonical_target),
            ("call_style", contract.call_style),
            ("ordered_parameters_json", contract.ordered_parameters_json),
            ("output_arity", contract.output_arity),
            ("output_types_json", contract.output_types_json),
            ("reference_semantic", contract.reference_semantic),
            (
                "resolution_owner_task_id",
                contract.resolution_owner_task_id,
            ),
            (
                "expected_resolution_module",
                contract.expected_resolution_module,
            ),
            ("mutation_rule", contract.mutation_rule),
            ("alias_rule", contract.alias_rule),
            ("shape_rule", contract.shape_rule),
            ("dtype_rule", contract.dtype_rule),
            ("accumulation_dtype", contract.accumulation_dtype),
            ("layout_rule", contract.layout_rule),
            ("device_rule", contract.device_rule),
            ("numeric_rule", contract.numeric_rule),
            ("tolerance", contract.tolerance),
            ("determinism", contract.determinism),
            ("cancellation_points", contract.cancellation_points),
            ("vjp_rule", contract.vjp_rule),
            ("jvp_rule", contract.jvp_rule),
            ("source_call_sites", contract.source_call_sites),
            ("oracle_fixture", contract.oracle_fixture),
            ("evidence", contract.evidence),
        ] {
            if value.is_empty() {
                return Err(OperationContractTableError::MissingField {
                    operation_id: contract.operation_id.to_owned(),
                    field: field.to_owned(),
                });
            }
        }
        if !is_lowercase_sha256(contract.oracle_fixture_sha256) {
            return Err(OperationContractTableError::InvalidFixtureDigest(
                contract.operation_id.to_owned(),
            ));
        }
        if !valid_module_name(contract.expected_resolution_module)
            || owner_modules
                .insert(
                    contract.resolution_owner_task_id,
                    contract.expected_resolution_module,
                )
                .is_some_and(|module| module != contract.expected_resolution_module)
            || module_owners
                .insert(
                    contract.expected_resolution_module,
                    contract.resolution_owner_task_id,
                )
                .is_some_and(|owner| owner != contract.resolution_owner_task_id)
        {
            return Err(OperationContractTableError::InvalidResolutionOwnership(
                contract.operation_id.to_owned(),
            ));
        }
        match contract.inventory_kind {
            ContractInventoryKind::CallableOperation => {
                if contract.reference_semantic_category()
                    != Some(ReferenceSemanticCategory::NotApplicable)
                {
                    return Err(OperationContractTableError::InvalidReferenceSemantic(
                        contract.operation_id.to_owned(),
                    ));
                }
                if !contract.resolution_state.is_blocked()
                    || contract.blocker_reason.is_empty()
                    || !contract.release_closure_required
                {
                    return Err(OperationContractTableError::UnclassifiedCallable(
                        contract.operation_id.to_owned(),
                    ));
                }
            }
            ContractInventoryKind::ReclassifiedExternalOperation => {
                if contract.reference_semantic_category()
                    != Some(ReferenceSemanticCategory::NotApplicable)
                    || contract.resolution_state
                        != ContractResolutionState::ReclassifiedExternalOperation
                    || contract.blocker_reason.is_empty()
                    || contract.rust_signature != "ExternalOperationDisposition"
                    || contract.release_closure_required
                    || contract.typed_reference().is_some()
                {
                    return Err(OperationContractTableError::InvalidExternalDisposition(
                        contract.operation_id.to_owned(),
                    ));
                }
            }
            ContractInventoryKind::NamespaceValueReference
            | ContractInventoryKind::TypeReference => {
                let Some(_) = contract.reference_semantic_category() else {
                    return Err(OperationContractTableError::InvalidReferenceSemantic(
                        contract.operation_id.to_owned(),
                    ));
                };
                if contract.resolution_state != ContractResolutionState::ResolvedReference
                    || contract.rust_signature != "TypedReferenceContract"
                    || contract.release_closure_required
                    || contract.typed_reference().is_none()
                {
                    return Err(OperationContractTableError::IncompleteReference(
                        contract.operation_id.to_owned(),
                    ));
                }
            }
        }
    }

    validate_operation_resolution_iter(
        contracts,
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .flat_map(|slice| {
                slice
                    .iter()
                    .map(move |resolution| (slice.module_name, resolution))
            }),
    )?;
    Ok(())
}

pub fn validate_operation_resolutions(
    contracts: &[OperationContractRecord],
    resolutions: &[ResolvedOperationContract],
) -> Result<(), OperationContractTableError> {
    validate_operation_resolution_iter(
        contracts,
        resolutions
            .iter()
            .map(|resolution| (resolution.resolution_module, resolution)),
    )?;
    Ok(())
}

pub fn validate_operation_resolution_slice(
    contracts: &[OperationContractRecord],
    module_name: &str,
    resolutions: &[ResolvedOperationContract],
) -> Result<(), OperationContractTableError> {
    if !valid_module_name(module_name) {
        return Err(OperationContractTableError::InvalidResolutionModule(
            module_name.to_owned(),
        ));
    }
    validate_operation_resolution_iter(
        contracts,
        resolutions
            .iter()
            .map(|resolution| (module_name, resolution)),
    )?;
    Ok(())
}

pub fn validate_operation_release_closure(
    contracts: &[OperationContractRecord],
    resolutions: &[ResolvedOperationContract],
) -> Result<(), OperationContractTableError> {
    let resolution_operation_ids = validate_operation_resolution_iter(
        contracts,
        resolutions
            .iter()
            .map(|resolution| (resolution.resolution_module, resolution)),
    )?;
    require_release_closure(contracts, &resolution_operation_ids)
}

pub fn validate_generated_operation_release_closure(
    contracts: &[OperationContractRecord],
) -> Result<(), OperationContractTableError> {
    let resolution_operation_ids = validate_operation_resolution_iter(
        contracts,
        GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .flat_map(|slice| {
                slice
                    .iter()
                    .map(move |resolution| (slice.module_name, resolution))
            }),
    )?;
    require_release_closure(contracts, &resolution_operation_ids)
}

fn validate_operation_resolution_iter<'a>(
    contracts: &[OperationContractRecord],
    resolutions: impl IntoIterator<Item = (&'a str, &'a ResolvedOperationContract)>,
) -> Result<HashSet<&'a str>, OperationContractTableError> {
    let mut resolution_operation_ids = HashSet::new();
    let mut resolution_overload_ids = HashSet::new();
    for (expected_module, resolution) in resolutions {
        if !valid_module_name(expected_module)
            || resolution.resolution_module != expected_module
            || !valid_module_name(resolution.resolution_module)
        {
            return Err(OperationContractTableError::InvalidResolutionModule(
                resolution.operation_id.to_owned(),
            ));
        }
        validate_resolution_semantics(&resolution_expectation(resolution)).map_err(|_| {
            OperationContractTableError::InvalidResolutionSemantics(
                resolution.operation_id.to_owned(),
            )
        })?;
        if !resolution_operation_ids.insert(resolution.operation_id)
            || !resolution_overload_ids.insert(resolution.overload_id)
        {
            return Err(OperationContractTableError::DuplicateCompiledResolution(
                resolution.operation_id.to_owned(),
            ));
        }
        if contracts.iter().any(|contract| {
            resolution.overload_id == contract.operation_id
                || resolution.overload_id == contract.overload_id
        }) {
            return Err(OperationContractTableError::ResolutionIdentifierCollision(
                resolution.overload_id.to_owned(),
            ));
        }
        let Some(baseline) = contracts
            .iter()
            .find(|contract| contract.operation_id == resolution.operation_id)
        else {
            return Err(OperationContractTableError::UnknownCompiledResolution(
                resolution.operation_id.to_owned(),
            ));
        };
        if !resolution_discharges_baseline(resolution, baseline) {
            return Err(OperationContractTableError::MismatchedCompiledResolution(
                resolution.operation_id.to_owned(),
            ));
        }
    }
    Ok(resolution_operation_ids)
}

fn require_release_closure(
    contracts: &[OperationContractRecord],
    resolution_operation_ids: &HashSet<&str>,
) -> Result<(), OperationContractTableError> {
    if let Some(contract) = contracts.iter().find(|contract| {
        contract.inventory_kind == ContractInventoryKind::CallableOperation
            && !resolution_operation_ids.contains(contract.operation_id)
    }) {
        return Err(OperationContractTableError::MissingCompiledResolution(
            contract.operation_id.to_owned(),
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum OperationContractTableError {
    #[error("duplicate operation contract ID: {0}")]
    DuplicateOperationId(String),
    #[error("duplicate overload contract ID: {0}")]
    DuplicateOverloadId(String),
    #[error("callable contract has neither a resolved signature nor an explicit blocker: {0}")]
    UnclassifiedCallable(String),
    #[error("typed reference contract is incomplete: {0}")]
    IncompleteReference(String),
    #[error("operation contract has an invalid typed reference semantic: {0}")]
    InvalidReferenceSemantic(String),
    #[error("operation contract {operation_id} is missing required field {field}")]
    MissingField { operation_id: String, field: String },
    #[error("operation contract has an invalid oracle fixture digest: {0}")]
    InvalidFixtureDigest(String),
    #[error("compiled operation resolution is duplicated: {0}")]
    DuplicateCompiledResolution(String),
    #[error("compiled operation resolution has an invalid or mismatched module: {0}")]
    InvalidResolutionModule(String),
    #[error("operation resolution owner and module mapping is invalid: {0}")]
    InvalidResolutionOwnership(String),
    #[error("compiled operation resolution has invalid structured semantics: {0}")]
    InvalidResolutionSemantics(String),
    #[error("compiled operation resolution has invalid evidence: {0}")]
    InvalidResolutionEvidence(String),
    #[error("compiled operation overload collides with a discovery identifier: {0}")]
    ResolutionIdentifierCollision(String),
    #[error("compiled operation resolution has no baseline record: {0}")]
    UnknownCompiledResolution(String),
    #[error("compiled operation resolution does not validly discharge its blocked baseline: {0}")]
    MismatchedCompiledResolution(String),
    #[error("blocked callable has no compiled operation resolution: {0}")]
    MissingCompiledResolution(String),
    #[error("reclassified external operation has an invalid disposition: {0}")]
    InvalidExternalDisposition(String),
}

include!(concat!(
    env!("OUT_DIR"),
    "/generated_operation_resolutions.rs"
));
include!("operation_contract_records.rs");

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, io, path::Path};
    #[test]
    fn generated_contract_table_is_complete_and_valid() {
        assert_eq!(OPERATION_CONTRACTS.len(), 600);
        assert_eq!(validate_operation_contracts(OPERATION_CONTRACTS), Ok(()));
        assert_eq!(
            validate_generated_operation_release_closure(OPERATION_CONTRACTS),
            Ok(())
        );
        assert_eq!(
            OPERATION_CONTRACTS
                .iter()
                .filter(|contract| {
                    contract.inventory_kind == ContractInventoryKind::CallableOperation
                })
                .count(),
            511
        );
        assert_eq!(
            OPERATION_CONTRACTS
                .iter()
                .filter(|contract| contract.typed_reference().is_some())
                .count(),
            82
        );
        assert_eq!(
            OPERATION_CONTRACTS
                .iter()
                .filter(|contract| contract.resolution_state.is_blocked())
                .count(),
            511
        );
        assert_eq!(
            OPERATION_CONTRACTS
                .iter()
                .filter(|contract| {
                    contract.resolution_state == ContractResolutionState::BlockedReceiverUnverified
                })
                .count(),
            94
        );
        assert_eq!(
            OPERATION_CONTRACTS
                .iter()
                .filter(|contract| {
                    contract.inventory_kind == ContractInventoryKind::ReclassifiedExternalOperation
                        && contract.resolution_state
                            == ContractResolutionState::ReclassifiedExternalOperation
                })
                .count(),
            7
        );
        assert!(OPERATION_CONTRACTS.iter().all(|contract| {
            !contract.resolution_owner_task_id.is_empty()
                && (contract.release_closure_required == contract.resolution_state.is_blocked())
        }));
        for slice in GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES {
            assert!(
                validate_operation_resolution_slice(
                    OPERATION_CONTRACTS,
                    slice.module_name,
                    slice.contracts,
                )
                .is_ok()
            );
            for resolution in slice.contracts {
                assert!(
                    validate_operation_resolution_evidence(workspace_root(), resolution).is_ok()
                );
            }
        }
        let categories = OPERATION_CONTRACTS
            .iter()
            .filter_map(|contract| contract.typed_reference())
            .filter_map(|contract| {
                OPERATION_CONTRACTS
                    .iter()
                    .find(|record| record.operation_id == contract.operation_id)
                    .and_then(|record| record.reference_semantic_category())
            })
            .collect::<HashSet<_>>();
        assert_eq!(categories.len(), 11);

        for (target, expected) in [
            ("torch.float8_e4m3fnuz", DType::Float8E4m3Fnuz),
            ("torch.float8_e5m2fnuz", DType::Float8E5m2Fnuz),
            ("torch.float8_e8m0fnu", DType::Float8E8m0Fnu),
        ] {
            let semantic = OPERATION_CONTRACTS
                .iter()
                .find(|contract| contract.canonical_target == target)
                .and_then(|contract| contract.typed_reference())
                .map(|contract| contract.semantic);
            assert_eq!(semantic, Some(CanonicalReference::DType(expected)));
        }

        let reference_categories = OPERATION_CONTRACTS
            .iter()
            .filter_map(|contract| contract.typed_reference())
            .map(|contract| contract.semantic)
            .collect::<HashSet<_>>();
        assert_eq!(reference_categories.len(), 79);

        let reference_template = *OPERATION_CONTRACTS
            .iter()
            .find(|contract| contract.typed_reference().is_some())
            .expect("the discovery ledger has typed references");
        for reference_semantic in [
            "{\"category\":\"dtype\",\"value\":\"unknown.target\"}",
            "{\"category\":\"layout-or-memory-format\",\"value\":\"unknown.target\"}",
            "{\"category\":\"boolean-capability\",\"value\":\"unknown.target\"}",
            "{\"category\":\"numeric-constant\",\"value\":\"unknown.target\"}",
            "{\"category\":\"function-reference\",\"value\":\"unknown.target\"}",
            "{\"category\":\"type-marker\",\"value\":\"unknown.target\"}",
            "{\"category\":\"namespace\",\"value\":\"unknown.target\"}",
            "{\"category\":\"tensor-property\",\"value\":\"unknown.target\"}",
            "{\"category\":\"device-property\",\"value\":\"unknown.target\"}",
            "{\"category\":\"enum-variant\",\"value\":\"unknown.target\"}",
            "{\"category\":\"version-value\",\"value\":\"unknown.target\"}",
        ] {
            let unknown = OperationContractRecord {
                canonical_target: "unknown.target",
                reference_semantic,
                ..reference_template
            };
            assert!(unknown.typed_reference().is_none());
            assert!(matches!(
                validate_operation_contracts(&[unknown]),
                Err(OperationContractTableError::IncompleteReference(_))
            ));
        }
    }

    #[test]
    fn compiled_resolutions_discharge_only_the_exact_owned_callable_baseline() {
        let baseline = OPERATION_CONTRACTS
            .iter()
            .find(|contract| contract.operation_id == "COMFY-TENSOR-OP-DDAAD49116D0")
            .expect("the discovery ledger has the contract-validation baseline");
        let valid = complete_resolution(baseline);
        assert!(!GENERATED_BUILD_SEALED_OPERATION_RESOLUTIONS.is_empty());
        assert!(
            GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
                .iter()
                .flat_map(GeneratedOperationResolutionSlice::iter)
                .all(is_build_sealed_resolution)
        );
        assert!(is_build_sealed_resolution(&valid));
        assert!(compiled_resolution_by_identifier(baseline.operation_id).is_some());
        assert_eq!(
            validate_operation_resolutions(OPERATION_CONTRACTS, &[valid]),
            Ok(())
        );
        assert_eq!(
            validate_operation_resolution_slice(
                OPERATION_CONTRACTS,
                baseline.expected_resolution_module,
                &[valid]
            ),
            Ok(())
        );
        assert_eq!(
            validate_operation_resolution_evidence(workspace_root(), &valid),
            Ok(())
        );
        assert!(matches!(
            resolve_operation_identifier(
                OPERATION_CONTRACTS,
                &[valid],
                baseline.operation_id
            ),
            Ok(Some(resolution)) if resolution.operation_id == baseline.operation_id
        ));
        assert!(matches!(
            resolve_operation_identifier(
                OPERATION_CONTRACTS,
                &[valid],
                valid.overload_id
            ),
            Ok(Some(resolution)) if resolution.operation_id == baseline.operation_id
        ));
        assert_eq!(
            resolve_operation_identifier(OPERATION_CONTRACTS, &[valid], baseline.overload_id),
            Ok(None)
        );
        assert!(matches!(
            validate_operation_release_closure(OPERATION_CONTRACTS, &[valid]),
            Err(OperationContractTableError::MissingCompiledResolution(_))
        ));
        assert!(matches!(
            validate_operation_resolutions(OPERATION_CONTRACTS, &[valid, valid]),
            Err(OperationContractTableError::DuplicateCompiledResolution(_))
        ));
        assert!(matches!(
            validate_operation_resolution_slice(OPERATION_CONTRACTS, "wrong_module", &[valid]),
            Err(OperationContractTableError::InvalidResolutionModule(_))
        ));

        let other_owner = OPERATION_CONTRACTS
            .iter()
            .find(|contract| {
                contract.inventory_kind == ContractInventoryKind::CallableOperation
                    && contract.expected_resolution_module != baseline.expected_resolution_module
            })
            .expect("the discovery ledger has a callable assigned to another module");
        let cross_leaf = ResolvedOperationContract {
            resolution_module: other_owner.expected_resolution_module,
            owner_task_id: other_owner.resolution_owner_task_id,
            ..valid
        };
        assert!(matches!(
            validate_operation_resolution_slice(
                OPERATION_CONTRACTS,
                other_owner.expected_resolution_module,
                &[cross_leaf]
            ),
            Err(OperationContractTableError::MismatchedCompiledResolution(_))
        ));

        let unknown = ResolvedOperationContract {
            operation_id: "COMFY-TENSOR-OP-UNKNOWN",
            overload_id: "native:absent:resolved:v1",
            ..valid
        };
        assert!(matches!(
            validate_operation_resolutions(OPERATION_CONTRACTS, &[unknown]),
            Err(OperationContractTableError::UnknownCompiledResolution(_))
        ));

        let other_baseline = OPERATION_CONTRACTS
            .iter()
            .find(|contract| contract.operation_id != baseline.operation_id)
            .expect("the discovery ledger has another identifier");
        let colliding = ResolvedOperationContract {
            overload_id: other_baseline.operation_id,
            ..valid
        };
        assert!(matches!(
            validate_operation_resolutions(OPERATION_CONTRACTS, &[colliding]),
            Err(OperationContractTableError::ResolutionIdentifierCollision(
                _
            ))
        ));
        assert!(matches!(
            resolve_operation_identifier(
                OPERATION_CONTRACTS,
                &[valid, unknown],
                baseline.operation_id
            ),
            Err(OperationContractTableError::UnknownCompiledResolution(_))
        ));

        for invalid in [
            ResolvedOperationContract {
                owner_task_id: "wrong-owner",
                ..valid
            },
            ResolvedOperationContract {
                baseline_fixture_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
                ..valid
            },
            ResolvedOperationContract {
                evidence_fixture: "/absolute/evidence.json",
                ..valid
            },
            ResolvedOperationContract {
                evidence_fixture: "crates/sim_test_support/fixtures/tensor_operations/../evidence.json",
                ..valid
            },
            ResolvedOperationContract {
                evidence_fixture: "crates/sim_test_support/fixtures/tensor_operations/wrong_module/evidence.json",
                ..valid
            },
            ResolvedOperationContract {
                evidence_fixture_sha256: baseline.oracle_fixture_sha256,
                ..valid
            },
            ResolvedOperationContract {
                baseline_overload_id: "wrong:blocked",
                ..valid
            },
        ] {
            assert!(matches!(
                validate_operation_resolutions(OPERATION_CONTRACTS, &[invalid]),
                Err(OperationContractTableError::MismatchedCompiledResolution(_))
            ));
        }

        for invalid in [
            ResolvedOperationContract {
                rust_signature: "unresolved",
                ..valid
            },
            ResolvedOperationContract {
                dtype_rule: " ",
                ..valid
            },
            ResolvedOperationContract {
                ordered_parameters_json: "{}",
                ..valid
            },
            ResolvedOperationContract {
                ordered_parameters_json: "[{\"default\":null,\"keyword_only\":false,\"kind\":\"observed_keyword\",\"name\":\"input\",\"type\":\"Tensor\"}]",
                ..valid
            },
            ResolvedOperationContract {
                ordered_parameters_json: "[{\"default\":null,\"keyword_only\":false,\"kind\":\"positional_or_keyword\",\"name\":\"input\",\"type\":\"Tensor\"},{\"default\":null,\"keyword_only\":false,\"kind\":\"positional_or_keyword\",\"name\":\"input\",\"type\":\"Tensor\"}]",
                ..valid
            },
            ResolvedOperationContract {
                ordered_parameters_json: "[{\"default\":null,\"extra\":true,\"keyword_only\":false,\"kind\":\"positional_or_keyword\",\"name\":\"input\",\"type\":\"Tensor\"}]",
                ..valid
            },
            ResolvedOperationContract {
                ordered_parameters_json: "[{\"default\":\"TODO\",\"keyword_only\":false,\"kind\":\"positional_or_keyword\",\"name\":\"input\",\"type\":\"Tensor\"}]",
                ..valid
            },
            ResolvedOperationContract {
                ordered_parameters_json: "[{\"default\":{\"nested\":\"not implemented\"},\"keyword_only\":false,\"kind\":\"positional_or_keyword\",\"name\":\"input\",\"type\":\"Tensor\"}]",
                ..valid
            },
            ResolvedOperationContract {
                output_arity: "2",
                ..valid
            },
            ResolvedOperationContract {
                output_types_json: "[null]",
                ..valid
            },
            ResolvedOperationContract {
                shape_rule: "unknown",
                ..valid
            },
            ResolvedOperationContract {
                dtype_rule: "TODO",
                ..valid
            },
            ResolvedOperationContract {
                numeric_rule: "placeholder",
                ..valid
            },
            ResolvedOperationContract {
                alias_rule: "not implemented",
                ..valid
            },
        ] {
            assert!(matches!(
                validate_operation_resolutions(OPERATION_CONTRACTS, &[invalid]),
                Err(OperationContractTableError::InvalidResolutionSemantics(_))
            ));
        }

        for invalid in [
            ResolvedOperationContract {
                evidence_fixture_sha256: "2222222222222222222222222222222222222222222222222222222222222222",
                ..valid
            },
            ResolvedOperationContract {
                evidence_fixture: "crates/sim_test_support/fixtures/tensor_operations/elementwise_or_runtime_operation_20/missing.json",
                ..valid
            },
            ResolvedOperationContract {
                numeric_rule: "approximately exact",
                ..valid
            },
        ] {
            assert!(matches!(
                validate_operation_resolution_evidence(workspace_root(), &invalid),
                Err(OperationContractTableError::InvalidResolutionEvidence(_))
            ));
        }

        let reference = OPERATION_CONTRACTS
            .iter()
            .find(|contract| contract.typed_reference().is_some())
            .expect("the discovery ledger has references");
        assert_eq!(
            resolve_operation_identifier(OPERATION_CONTRACTS, &[valid], reference.operation_id),
            Ok(None)
        );
        let invalid_reference_resolution = ResolvedOperationContract {
            operation_id: reference.operation_id,
            baseline_overload_id: reference.overload_id,
            baseline_fixture_sha256: reference.oracle_fixture_sha256,
            owner_task_id: reference.resolution_owner_task_id,
            ..valid
        };
        assert!(matches!(
            validate_operation_resolutions(OPERATION_CONTRACTS, &[invalid_reference_resolution]),
            Err(OperationContractTableError::MismatchedCompiledResolution(_))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn resolution_evidence_rejects_symlinked_parent_directories()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::fs::symlink;

        let baseline = OPERATION_CONTRACTS
            .iter()
            .find(|contract| contract.operation_id == "COMFY-TENSOR-OP-DDAAD49116D0")
            .ok_or("the discovery ledger has no contract-validation baseline")?;
        let resolution = complete_resolution(baseline);
        let source_fixture = workspace_root().join(resolution.evidence_fixture);
        let target_root = workspace_root().join("target/sim-parity");
        fs::create_dir_all(&target_root)?;

        let module_case = target_root.join(format!(
            "operation-contract-module-symlink-{}",
            std::process::id()
        ));
        remove_directory_if_exists(&module_case)?;
        let tensor_operations =
            module_case.join("crates/sim_test_support/fixtures/tensor_operations");
        let real_module = module_case.join("real-module");
        fs::create_dir_all(&tensor_operations)?;
        fs::create_dir_all(&real_module)?;
        fs::copy(&source_fixture, real_module.join("activation_1d.json"))?;
        symlink(
            &real_module,
            tensor_operations.join(resolution.resolution_module),
        )?;
        assert!(validate_operation_resolution_evidence(&module_case, &resolution).is_err());
        remove_directory_if_exists(&module_case)?;

        let parent_case = target_root.join(format!(
            "operation-contract-parent-symlink-{}",
            std::process::id()
        ));
        remove_directory_if_exists(&parent_case)?;
        let fixtures = parent_case.join("crates/sim_test_support/fixtures");
        let real_tensor_operations = parent_case.join("real-tensor-operations");
        let real_parent_module = real_tensor_operations.join(resolution.resolution_module);
        fs::create_dir_all(&fixtures)?;
        fs::create_dir_all(&real_parent_module)?;
        fs::copy(
            source_fixture,
            real_parent_module.join("activation_1d.json"),
        )?;
        symlink(&real_tensor_operations, fixtures.join("tensor_operations"))?;
        assert!(validate_operation_resolution_evidence(&parent_case, &resolution).is_err());
        remove_directory_if_exists(&parent_case)?;
        Ok(())
    }

    fn remove_directory_if_exists(path: &Path) -> io::Result<()> {
        match fs::remove_dir_all(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }

    fn complete_resolution(baseline: &OperationContractRecord) -> ResolvedOperationContract {
        *GENERATED_RESOLVED_OPERATION_CONTRACT_SLICES
            .iter()
            .find_map(|slice| {
                slice
                    .contracts
                    .iter()
                    .find(|resolution| resolution.operation_id == baseline.operation_id)
            })
            .expect("the generated resolution table has the contract-validation baseline")
    }

    fn workspace_root() -> &'static Path {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("sim_tensor is nested directly below the workspace crates directory")
    }
}
