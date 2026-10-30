use crate::{
    ArtifactIndex, LoadedModel, ModelStore, ModelStoreError, NativeModelResourceIdentity,
    NativeModelResourceRole, NativeModule, NativeOpsError,
};
use sha2::{Digest, Sha256};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DeviceId, ExecutionContext, StorageId,
    Tensor, TensorDescriptor, TensorError,
    generated_activation_normalization_functional_01::{
        FunctionalError, batch_norm_with_context_exact_native,
        group_norm_with_context_exact_native, relu_with_context_exact_native_in_place,
        silu_with_context_exact_native_in_place,
    },
    generated_external_tensor_kernel_01::{
        ExternalTensorKernelPartOneError, NativeBilinearBoundary, checked_bilinear_weights,
    },
    generated_sim_operator_indirection_01::{ConvolutionGeometry, OperatorIndirectionError},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    mem,
};
use thiserror::Error;

pub const EFFICIENTNET_V2_S_OPERATION_ID: &str = "COMFY-TENSOR-OP-638DE6179D46";
pub const RAFT_LARGE_OPERATION_ID: &str = "COMFY-TENSOR-OP-852D8E9DBC9C";
pub const RAFT_LARGE_SOURCE_TYPE_ID: &str = "OPTICAL_FLOW";
pub const RAFT_LARGE_RESOURCE_ROLE: &str = "optical_flow";
const EFFICIENTNET_V2_S_FEATURE_MODULE: &str = "efficientnet_v2_s.features";
const RAFT_LARGE_ARCHITECTURE_ID: &str = "torchvision.models.optical_flow.raft_large";
const RAFT_LARGE_RESOURCE_FORMAT: &str = "zed-native-torchvision-raft-large-v1";

#[derive(Clone, Debug, Error)]
pub enum NativeVisionModelError {
    #[error(transparent)]
    Module(#[from] NativeOpsError),
    #[error(transparent)]
    Tensor(#[from] OperatorIndirectionError),
    #[error(transparent)]
    TensorStorage(#[from] TensorError),
    #[error(transparent)]
    ModelStore(#[from] ModelStoreError),
    #[error(transparent)]
    Functional(#[from] FunctionalError),
    #[error("native vision model bilinear sampling failed: {0}")]
    ExternalKernel(String),
    #[error("native vision model configuration is invalid: {0}")]
    Invalid(String),
    #[error("state dictionary is missing parameter or buffer {0}")]
    MissingState(String),
    #[error("state dictionary contains unexpected parameter or buffer {0}")]
    UnexpectedState(String),
    #[error("state dictionary tensor {name} has shape {actual:?}; expected {expected:?}")]
    StateShape {
        name: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("state dictionary tensor {name} has dtype {actual:?}; expected {expected:?}")]
    StateDType {
        name: String,
        expected: DType,
        actual: DType,
    },
    #[error("native vision model parameters have not been loaded")]
    ParametersNotLoaded,
    #[error("{0} supports production inference only; call eval() before forward")]
    EvaluationRequired(&'static str),
    #[error("native vision model execution was cancelled")]
    Cancelled,
    #[error("native vision model shape arithmetic overflowed")]
    ShapeOverflow,
    #[error("native RAFT-large semantic identity changed")]
    SemanticIdentityChanged,
}

impl From<sim_types::CancellationError> for NativeVisionModelError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<ExternalTensorKernelPartOneError> for NativeVisionModelError {
    fn from(error: ExternalTensorKernelPartOneError) -> Self {
        Self::ExternalKernel(error.to_string())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeVisionStateKind {
    Parameter,
    Buffer,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeVisionStateSpec {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub kind: NativeVisionStateKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeEfficientNetBlockKind {
    FusedMbConv,
    MbConv,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeEfficientNetV2SFeatureSource {
    StableCascadeEncoder,
    StableCascadeCombined,
}

impl NativeEfficientNetV2SFeatureSource {
    const fn prefix(self) -> &'static str {
        match self {
            Self::StableCascadeEncoder => "backbone.",
            Self::StableCascadeCombined => "encoder.backbone.",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeEfficientNetStage {
    pub block: NativeEfficientNetBlockKind,
    pub expand_ratio: usize,
    pub kernel: usize,
    pub stride: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub layers: usize,
}

const EFFICIENTNET_V2_S_STAGES: [NativeEfficientNetStage; 6] = [
    NativeEfficientNetStage {
        block: NativeEfficientNetBlockKind::FusedMbConv,
        expand_ratio: 1,
        kernel: 3,
        stride: 1,
        input_channels: 24,
        output_channels: 24,
        layers: 2,
    },
    NativeEfficientNetStage {
        block: NativeEfficientNetBlockKind::FusedMbConv,
        expand_ratio: 4,
        kernel: 3,
        stride: 2,
        input_channels: 24,
        output_channels: 48,
        layers: 4,
    },
    NativeEfficientNetStage {
        block: NativeEfficientNetBlockKind::FusedMbConv,
        expand_ratio: 4,
        kernel: 3,
        stride: 2,
        input_channels: 48,
        output_channels: 64,
        layers: 4,
    },
    NativeEfficientNetStage {
        block: NativeEfficientNetBlockKind::MbConv,
        expand_ratio: 4,
        kernel: 3,
        stride: 2,
        input_channels: 64,
        output_channels: 128,
        layers: 6,
    },
    NativeEfficientNetStage {
        block: NativeEfficientNetBlockKind::MbConv,
        expand_ratio: 6,
        kernel: 3,
        stride: 1,
        input_channels: 128,
        output_channels: 160,
        layers: 9,
    },
    NativeEfficientNetStage {
        block: NativeEfficientNetBlockKind::MbConv,
        expand_ratio: 6,
        kernel: 3,
        stride: 2,
        input_channels: 160,
        output_channels: 256,
        layers: 15,
    },
];

#[derive(Clone, Debug)]
struct NativeModuleSlot {
    weight_name: String,
    bias_name: Option<String>,
    module: NativeModule,
}

#[derive(Debug)]
struct NativeValues {
    shape: Vec<usize>,
    values: VisionValues,
}

type VisionValues = CpuWorkspaceVec<f32>;

struct VisionExecution<'a, 'context> {
    backend: &'a CpuBackend,
    context: &'a ExecutionContext<'context>,
}

impl<'a, 'context> VisionExecution<'a, 'context> {
    fn canonical(backend: &'a CpuBackend, context: &'a ExecutionContext<'context>) -> Self {
        Self { backend, context }
    }

    fn values(&self, capacity: usize) -> Result<VisionValues, NativeVisionModelError> {
        self.backend
            .workspace_vec(self.context, capacity)
            .map_err(Into::into)
    }

    fn zeroed(&self, length: usize) -> Result<VisionValues, NativeVisionModelError> {
        let mut values = self.values(length)?;
        for _ in 0..length {
            push_value(&mut values, 0.0)?;
        }
        Ok(values)
    }

    fn copy(&self, source: &[f32]) -> Result<VisionValues, NativeVisionModelError> {
        let mut values = self.values(source.len())?;
        extend_values(&mut values, source)?;
        Ok(values)
    }
}

impl NativeValues {
    fn from_tensor(
        execution: &VisionExecution<'_, '_>,
        tensor: &Tensor,
    ) -> Result<Self, NativeVisionModelError> {
        execution.context.cancellation.check()?;
        if tensor.descriptor().dtype() != DType::F32
            || tensor.descriptor().device() != DeviceId::CPU
        {
            return Err(NativeVisionModelError::Invalid(
                "native vision staging requires a CPU F32 tensor".into(),
            ));
        }
        if tensor.descriptor().stream() != execution.context.stream {
            return Err(NativeVisionModelError::TensorStorage(
                TensorError::StreamMismatch {
                    expected: execution.context.stream,
                    actual: tensor.descriptor().stream(),
                },
            ));
        }
        let element_count = usize::try_from(tensor.descriptor().element_count()?)
            .map_err(|_| NativeVisionModelError::ShapeOverflow)?;
        let mut values = execution.values(element_count)?;
        for index in 0..element_count {
            if index.is_multiple_of(4096) {
                execution.context.cancellation.check()?;
            }
            let encoded: [u8; 4] = tensor
                .linear_element_bytes(
                    u64::try_from(index).map_err(|_| NativeVisionModelError::ShapeOverflow)?,
                )?
                .try_into()
                .map_err(|_| NativeVisionModelError::Invalid("unaligned F32 tensor".into()))?;
            push_value(&mut values, f32::from_ne_bytes(encoded))?;
        }
        execution.context.cancellation.check()?;
        Ok(Self {
            shape: tensor
                .descriptor()
                .shape()
                .iter()
                .map(|dimension| {
                    usize::try_from(*dimension).map_err(|_| NativeVisionModelError::ShapeOverflow)
                })
                .collect::<Result<_, _>>()?,
            values,
        })
    }

    fn to_tensor(
        &self,
        execution: &VisionExecution<'_, '_>,
    ) -> Result<Tensor, NativeVisionModelError> {
        let shape = self
            .shape
            .iter()
            .map(|dimension| {
                u64::try_from(*dimension).map_err(|_| NativeVisionModelError::ShapeOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let descriptor = TensorDescriptor::contiguous(
            shape,
            DType::F32,
            DeviceId::CPU,
            execution.context.stream,
        )?;
        Ok(execution
            .backend
            .upload_f32(descriptor, &self.values, execution.context)?
            .0)
    }

    fn try_clone(
        &self,
        execution: &VisionExecution<'_, '_>,
    ) -> Result<Self, NativeVisionModelError> {
        Ok(Self {
            shape: self.shape.clone(),
            values: execution.copy(&self.values)?,
        })
    }
}

fn push_value(values: &mut VisionValues, value: f32) -> Result<(), NativeVisionModelError> {
    values.try_push(value)?;
    Ok(())
}

fn extend_values(values: &mut VisionValues, source: &[f32]) -> Result<(), NativeVisionModelError> {
    for value in source {
        push_value(values, *value)?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct NativeEfficientNetV2S {
    root: NativeModule,
    modules: Vec<NativeModuleSlot>,
    residual_state: BTreeMap<String, Tensor>,
    schema: Vec<NativeVisionStateSpec>,
    feature_module_count: usize,
    feature_schema_count: usize,
    training: bool,
    feature_parameters_loaded: bool,
    classifier_parameters_loaded: bool,
}

impl NativeEfficientNetV2S {
    pub const fn stages(&self) -> &[NativeEfficientNetStage; 6] {
        &EFFICIENTNET_V2_S_STAGES
    }

    pub fn root(&self) -> &NativeModule {
        &self.root
    }

    pub fn state_schema(&self) -> &[NativeVisionStateSpec] {
        &self.schema
    }

    pub fn feature_state_schema(&self) -> Result<&[NativeVisionStateSpec], NativeVisionModelError> {
        self.schema.get(..self.feature_schema_count).ok_or_else(|| {
            NativeVisionModelError::Invalid("invalid EfficientNet feature schema boundary".into())
        })
    }

    pub fn parameter_count(&self) -> Result<u64, NativeVisionModelError> {
        parameter_count(&self.schema)
    }

    pub fn feature_parameter_count(&self) -> Result<u64, NativeVisionModelError> {
        parameter_count(self.feature_state_schema()?)
    }

    pub const fn block_count(&self) -> usize {
        40
    }

    pub const fn is_training(&self) -> bool {
        self.training
    }

    pub const fn parameters_loaded(&self) -> bool {
        self.feature_parameters_loaded && self.classifier_parameters_loaded
    }

    pub const fn feature_parameters_loaded(&self) -> bool {
        self.feature_parameters_loaded
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    pub fn load_state_dict(
        &mut self,
        state: BTreeMap<String, Tensor>,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeVisionModelError> {
        validate_state_dictionary(&self.schema, &state, cancellation)?;
        let loaded = load_state_dictionary(&self.schema, &self.modules, state, cancellation)?;
        self.modules = loaded.modules;
        self.residual_state = loaded.residual_state;
        self.feature_parameters_loaded = true;
        self.classifier_parameters_loaded = true;
        Ok(())
    }

    pub fn load_feature_state_dict(
        &mut self,
        state: BTreeMap<String, Tensor>,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeVisionModelError> {
        let schema = self
            .schema
            .get(..self.feature_schema_count)
            .ok_or_else(|| {
                NativeVisionModelError::Invalid(
                    "invalid EfficientNet feature schema boundary".into(),
                )
            })?;
        let module_templates = self
            .modules
            .get(..self.feature_module_count)
            .ok_or_else(|| {
                NativeVisionModelError::Invalid(
                    "invalid EfficientNet feature module boundary".into(),
                )
            })?;
        let loaded = load_state_dictionary(schema, module_templates, state, cancellation)?;
        let feature_modules = self
            .modules
            .get_mut(..self.feature_module_count)
            .ok_or_else(|| {
                NativeVisionModelError::Invalid(
                    "invalid EfficientNet feature module boundary".into(),
                )
            })?;
        if feature_modules.len() != loaded.modules.len() {
            return Err(NativeVisionModelError::Invalid(
                "loaded EfficientNet feature module count does not match the architecture".into(),
            ));
        }
        feature_modules.clone_from_slice(&loaded.modules);
        self.residual_state = loaded.residual_state;
        self.feature_parameters_loaded = true;
        self.classifier_parameters_loaded = false;
        Ok(())
    }

    pub fn load_stage_c_features_from_model_store_with_context(
        &mut self,
        backend: &CpuBackend,
        store: &ModelStore,
        index: &ArtifactIndex,
        model: &LoadedModel,
        source: NativeEfficientNetV2SFeatureSource,
        context: &ExecutionContext<'_>,
    ) -> Result<(), NativeVisionModelError> {
        let state = load_efficientnet_feature_state_from_model_store_with_context(
            backend,
            store,
            index,
            model,
            self.feature_state_schema()?,
            source,
            context,
        )?;
        self.load_feature_state_dict(state, context.cancellation)
    }

    pub fn feature_execution_module(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<NativeModule, NativeVisionModelError> {
        cancellation.check()?;
        if !self.feature_parameters_loaded {
            return Err(NativeVisionModelError::ParametersNotLoaded);
        }
        let feature_modules = self
            .modules
            .get(..self.feature_module_count)
            .ok_or_else(|| {
                NativeVisionModelError::Invalid(
                    "invalid EfficientNet feature module boundary".into(),
                )
            })?;
        let feature_schema = self.feature_state_schema()?;
        let residual_names = feature_schema
            .iter()
            .filter(|spec| {
                !feature_modules.iter().any(|slot| {
                    slot.weight_name == spec.name
                        || slot.bias_name.as_deref() == Some(spec.name.as_str())
                })
            })
            .map(|spec| spec.name.as_str())
            .collect::<BTreeSet<_>>();
        let mut children = Vec::with_capacity(
            feature_modules
                .len()
                .checked_add(residual_names.len())
                .ok_or(NativeVisionModelError::ShapeOverflow)?,
        );
        for slot in feature_modules {
            cancellation.check()?;
            children.push(slot.module.clone());
        }
        for name in residual_names {
            cancellation.check()?;
            let tensor = self
                .residual_state
                .get(name)
                .ok_or_else(|| NativeVisionModelError::MissingState(name.to_owned()))?;
            children.push(NativeModule::buffer(name, tensor.clone())?);
        }
        cancellation.check()?;
        NativeModule::module_dict(EFFICIENTNET_V2_S_FEATURE_MODULE, children).map_err(Into::into)
    }

    fn load_feature_execution_module(
        &mut self,
        module: &NativeModule,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeVisionModelError> {
        cancellation.check()?;
        if module.layer_name() != EFFICIENTNET_V2_S_FEATURE_MODULE {
            return Err(NativeVisionModelError::Invalid(format!(
                "expected EfficientNet feature module {EFFICIENTNET_V2_S_FEATURE_MODULE}, got {}",
                module.layer_name()
            )));
        }
        let feature_modules = self
            .modules
            .get(..self.feature_module_count)
            .ok_or_else(|| {
                NativeVisionModelError::Invalid(
                    "invalid EfficientNet feature module boundary".into(),
                )
            })?;
        let feature_schema = self.feature_state_schema()?;
        let expected_children = feature_modules
            .iter()
            .map(|slot| slot.module.layer_name())
            .chain(feature_schema.iter().filter_map(|spec| {
                (!feature_modules.iter().any(|slot| {
                    slot.weight_name == spec.name
                        || slot.bias_name.as_deref() == Some(spec.name.as_str())
                }))
                .then_some(spec.name.as_str())
            }))
            .collect::<BTreeSet<_>>();
        let mut actual_children = BTreeSet::new();
        for child in module.children() {
            cancellation.check()?;
            if !actual_children.insert(child.layer_name())
                || !expected_children.contains(child.layer_name())
            {
                return Err(NativeVisionModelError::UnexpectedState(
                    child.layer_name().to_owned(),
                ));
            }
        }
        if actual_children != expected_children {
            let missing = expected_children
                .difference(&actual_children)
                .next()
                .copied()
                .unwrap_or(EFFICIENTNET_V2_S_FEATURE_MODULE);
            return Err(NativeVisionModelError::MissingState(missing.to_owned()));
        }

        let mut state = BTreeMap::new();
        for slot in feature_modules {
            cancellation.check()?;
            let child = module
                .children()
                .iter()
                .find(|child| child.layer_name() == slot.module.layer_name())
                .ok_or_else(|| {
                    NativeVisionModelError::MissingState(slot.module.layer_name().to_owned())
                })?;
            let (weight, bias) = child.dense_parameters()?;
            state.insert(slot.weight_name.clone(), weight.clone());
            match (&slot.bias_name, bias) {
                (Some(name), Some(bias)) => {
                    state.insert(name.clone(), bias.clone());
                }
                (Some(name), None) => {
                    return Err(NativeVisionModelError::MissingState(name.clone()));
                }
                (None, Some(_)) => {
                    return Err(NativeVisionModelError::UnexpectedState(format!(
                        "{}.bias",
                        slot.module.layer_name()
                    )));
                }
                (None, None) => {}
            }
        }
        for spec in feature_schema {
            if state.contains_key(&spec.name) {
                continue;
            }
            cancellation.check()?;
            let tensor = module
                .children()
                .iter()
                .find(|child| child.layer_name() == spec.name)
                .and_then(NativeModule::registered_buffer)
                .ok_or_else(|| NativeVisionModelError::MissingState(spec.name.clone()))?;
            state.insert(spec.name.clone(), tensor.clone());
        }
        self.load_feature_state_dict(state, cancellation)
    }

    pub fn load_from_model_store_with_context(
        &mut self,
        backend: &CpuBackend,
        store: &ModelStore,
        index: &ArtifactIndex,
        model: &LoadedModel,
        context: &ExecutionContext<'_>,
    ) -> Result<(), NativeVisionModelError> {
        let state = load_vision_state_from_model_store_with_context(
            backend,
            store,
            index,
            model,
            &self.schema,
            context,
        )?;
        self.load_state_dict(state, context.cancellation)
    }

    pub fn forward_features_with_context(
        &mut self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeVisionModelError> {
        self.forward_features_impl(backend, input, context)
    }

    fn forward_features_impl(
        &mut self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeVisionModelError> {
        context.cancellation.check()?;
        require_nchw_f32(input, 3)?;
        if !self.feature_parameters_loaded {
            return Err(NativeVisionModelError::ParametersNotLoaded);
        }
        if self.training {
            return Err(NativeVisionModelError::EvaluationRequired(
                "EfficientNet-V2-S",
            ));
        }
        let execution = VisionExecution::canonical(backend, context);
        let input = NativeValues::from_tensor(&execution, input)?;
        efficientnet_features_full(
            &mut self.modules,
            &mut self.residual_state,
            input,
            false,
            &execution,
        )?
        .to_tensor(&execution)
    }

    pub fn forward_with_context(
        &mut self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeVisionModelError> {
        self.forward_impl(backend, input, context)
    }

    fn forward_impl(
        &mut self,
        backend: &CpuBackend,
        input: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, NativeVisionModelError> {
        context.cancellation.check()?;
        if !self.parameters_loaded() {
            return Err(NativeVisionModelError::ParametersNotLoaded);
        }
        let features = self.forward_features_impl(backend, input, context)?;
        let execution = VisionExecution::canonical(backend, context);
        let features = NativeValues::from_tensor(&execution, &features)?;
        let pooled = adaptive_average_pool(&features, &execution)?;
        run_module(&mut self.modules, "classifier.1", &pooled, &execution)?.to_tensor(&execution)
    }
}

pub fn efficientnet_v2_s_exact_native(
    cancellation: &CancellationToken,
) -> Result<NativeEfficientNetV2S, NativeVisionModelError> {
    cancellation.check()?;
    let root = NativeModule::container("torchvision.models.efficientnet_v2_s")?;
    let mut builder = SchemaBuilder::default();
    builder.conv_bn("features.0", 3, 24, 3, 2, 1, false)?;
    for (stage_index, stage) in EFFICIENTNET_V2_S_STAGES.iter().enumerate() {
        let mut input_channels = stage.input_channels;
        for block_index in 0..stage.layers {
            cancellation.check()?;
            let stride = if block_index == 0 { stage.stride } else { 1 };
            let prefix = format!("features.{}.{}.block", stage_index + 1, block_index);
            let expanded_channels = make_divisible(input_channels * stage.expand_ratio, 8);
            match stage.block {
                NativeEfficientNetBlockKind::FusedMbConv if expanded_channels == input_channels => {
                    builder.conv_bn(
                        &format!("{prefix}.0"),
                        input_channels,
                        stage.output_channels,
                        stage.kernel,
                        stride,
                        1,
                        false,
                    )?;
                }
                NativeEfficientNetBlockKind::FusedMbConv => {
                    builder.conv_bn(
                        &format!("{prefix}.0"),
                        input_channels,
                        expanded_channels,
                        stage.kernel,
                        stride,
                        1,
                        false,
                    )?;
                    builder.conv_bn(
                        &format!("{prefix}.1"),
                        expanded_channels,
                        stage.output_channels,
                        1,
                        1,
                        1,
                        false,
                    )?;
                }
                NativeEfficientNetBlockKind::MbConv => {
                    builder.conv_bn(
                        &format!("{prefix}.0"),
                        input_channels,
                        expanded_channels,
                        1,
                        1,
                        1,
                        false,
                    )?;
                    builder.conv_bn(
                        &format!("{prefix}.1"),
                        expanded_channels,
                        expanded_channels,
                        stage.kernel,
                        stride,
                        expanded_channels,
                        false,
                    )?;
                    let squeeze_channels = (input_channels / 4).max(1);
                    builder.conv(
                        &format!("{prefix}.2.fc1"),
                        expanded_channels,
                        squeeze_channels,
                        1,
                        1,
                        1,
                        true,
                    )?;
                    builder.conv(
                        &format!("{prefix}.2.fc2"),
                        squeeze_channels,
                        expanded_channels,
                        1,
                        1,
                        1,
                        true,
                    )?;
                    builder.conv_bn(
                        &format!("{prefix}.3"),
                        expanded_channels,
                        stage.output_channels,
                        1,
                        1,
                        1,
                        false,
                    )?;
                }
            }
            input_channels = stage.output_channels;
        }
    }
    builder.conv_bn("features.7", 256, 1280, 1, 1, 1, false)?;
    let feature_module_count = builder.modules.len();
    let feature_schema_count = builder.schema.len();
    builder.linear("classifier.1", 1280, 1000, true)?;
    let model = NativeEfficientNetV2S {
        root,
        modules: builder.modules,
        residual_state: BTreeMap::new(),
        schema: builder.schema,
        feature_module_count,
        feature_schema_count,
        training: true,
        feature_parameters_loaded: false,
        classifier_parameters_loaded: false,
    };
    if model.block_count() != 40 || model.parameter_count()? != 21_458_488 {
        return Err(NativeVisionModelError::Invalid(
            "EfficientNet-V2-S architecture does not match the canonical 40-block schema".into(),
        ));
    }
    cancellation.check()?;
    Ok(model)
}

pub fn load_stage_c_efficientnet_feature_module_from_model_store_with_context(
    backend: &CpuBackend,
    store: &ModelStore,
    index: &ArtifactIndex,
    model: &LoadedModel,
    source: NativeEfficientNetV2SFeatureSource,
    context: &ExecutionContext<'_>,
) -> Result<NativeModule, NativeVisionModelError> {
    context.cancellation.check()?;
    let mut efficientnet = efficientnet_v2_s_exact_native(context.cancellation)?;
    efficientnet.load_stage_c_features_from_model_store_with_context(
        backend, store, index, model, source, context,
    )?;
    efficientnet.eval();
    efficientnet.feature_execution_module(context.cancellation)
}

pub fn efficientnet_v2_s_features_from_module_with_context(
    module: &NativeModule,
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, NativeVisionModelError> {
    context.cancellation.check()?;
    let feature_module = find_unique_feature_execution_module(module)?;
    let mut efficientnet = efficientnet_v2_s_exact_native(context.cancellation)?;
    efficientnet.load_feature_execution_module(feature_module, context.cancellation)?;
    efficientnet.eval();
    efficientnet.forward_features_with_context(backend, input, context)
}

fn find_unique_feature_execution_module(
    module: &NativeModule,
) -> Result<&NativeModule, NativeVisionModelError> {
    fn collect<'a>(module: &'a NativeModule, found: &mut Vec<&'a NativeModule>) {
        if module.layer_name() == EFFICIENTNET_V2_S_FEATURE_MODULE {
            found.push(module);
        }
        for child in module.children() {
            collect(child, found);
        }
    }

    let mut found = Vec::new();
    collect(module, &mut found);
    match found.as_slice() {
        [feature_module] => Ok(*feature_module),
        [] => Err(NativeVisionModelError::MissingState(
            EFFICIENTNET_V2_S_FEATURE_MODULE.to_owned(),
        )),
        _ => Err(NativeVisionModelError::UnexpectedState(
            EFFICIENTNET_V2_S_FEATURE_MODULE.to_owned(),
        )),
    }
}

struct RaftSemanticHasher(Sha256);

impl RaftSemanticHasher {
    fn new(domain: &[u8]) -> Result<Self, NativeVisionModelError> {
        let mut hasher = Self(Sha256::new());
        hasher.field(domain)?;
        Ok(hasher)
    }

    fn field(&mut self, value: &[u8]) -> Result<(), NativeVisionModelError> {
        let length =
            u64::try_from(value.len()).map_err(|_| NativeVisionModelError::ShapeOverflow)?;
        self.0.update(length.to_le_bytes());
        self.0.update(value);
        Ok(())
    }

    fn u64(&mut self, value: u64) {
        self.0.update(value.to_le_bytes());
    }

    fn finish(self) -> String {
        format!("{:x}", self.0.finalize())
    }
}

fn raft_architecture_digest(
    schema: &[NativeVisionStateSpec],
) -> Result<String, NativeVisionModelError> {
    let mut digest = RaftSemanticHasher::new(b"zed.sim.model.raft-large-architecture.v1")?;
    digest.field(RAFT_LARGE_ARCHITECTURE_ID.as_bytes())?;
    digest.field(RAFT_LARGE_OPERATION_ID.as_bytes())?;
    digest.field(RAFT_LARGE_SOURCE_TYPE_ID.as_bytes())?;
    digest.field(RAFT_LARGE_RESOURCE_ROLE.as_bytes())?;
    digest.u64(parameter_count(schema)?);
    digest.u64(u64::try_from(schema.len()).map_err(|_| NativeVisionModelError::ShapeOverflow)?);
    for spec in schema {
        digest.field(spec.name.as_bytes())?;
        digest.field(&[raft_dtype_tag(spec.dtype)])?;
        digest.field(&[match spec.kind {
            NativeVisionStateKind::Parameter => 1,
            NativeVisionStateKind::Buffer => 2,
        }])?;
        digest.u64(
            u64::try_from(spec.shape.len()).map_err(|_| NativeVisionModelError::ShapeOverflow)?,
        );
        for dimension in &spec.shape {
            digest.u64(*dimension);
        }
    }
    Ok(digest.finish())
}

fn raft_state_digest(
    state: &BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<String, NativeVisionModelError> {
    cancellation.check()?;
    let mut digest = RaftSemanticHasher::new(b"zed.sim.model.raft-large-state.v1")?;
    digest.u64(u64::try_from(state.len()).map_err(|_| NativeVisionModelError::ShapeOverflow)?);
    for (name, tensor) in state {
        cancellation.check()?;
        let descriptor = tensor.descriptor();
        digest.field(name.as_bytes())?;
        digest.field(&[raft_dtype_tag(descriptor.dtype())])?;
        digest.u64(
            u64::try_from(descriptor.shape().len())
                .map_err(|_| NativeVisionModelError::ShapeOverflow)?,
        );
        for dimension in descriptor.shape() {
            digest.u64(*dimension);
        }
        let bytes = tensor.contiguous_bytes()?;
        for chunk in bytes.chunks(64 * 1024) {
            cancellation.check()?;
            digest.field(chunk)?;
        }
    }
    cancellation.check()?;
    Ok(digest.finish())
}

fn raft_module_state_digest(
    modules: &[NativeModuleSlot],
    residual_state: &BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<String, NativeVisionModelError> {
    cancellation.check()?;
    let mut digest = RaftSemanticHasher::new(b"zed.sim.model.raft-large-modules.v1")?;
    digest.u64(u64::try_from(modules.len()).map_err(|_| NativeVisionModelError::ShapeOverflow)?);
    for slot in modules {
        cancellation.check()?;
        digest.field(slot.weight_name.as_bytes())?;
        match &slot.bias_name {
            Some(name) => digest.field(name.as_bytes())?,
            None => digest.field(b"")?,
        }
        digest.field(slot.module.semantic_state_digest(cancellation)?.as_bytes())?;
    }
    digest.field(raft_state_digest(residual_state, cancellation)?.as_bytes())?;
    cancellation.check()?;
    Ok(digest.finish())
}

fn raft_dtype_tag(dtype: DType) -> u8 {
    match dtype {
        DType::F64 => 1,
        DType::F32 => 2,
        DType::F16 => 3,
        DType::Bf16 => 4,
        DType::I64 => 5,
        DType::I32 => 6,
        DType::I16 => 7,
        DType::I8 => 8,
        DType::U64 => 9,
        DType::U32 => 10,
        DType::U16 => 11,
        DType::U8 => 12,
        DType::Bool => 13,
        DType::Complex64 => 14,
        DType::Complex128 => 15,
        DType::Float8E4m3Fn => 16,
        DType::Float8E5m2 => 17,
        DType::Float8E4m3Fnuz => 18,
        DType::Float8E5m2Fnuz => 19,
        DType::Float8E8m0Fnu => 20,
    }
}

fn checked_resident_add(
    total: u64,
    bytes: usize,
    _subject: &'static str,
) -> Result<u64, NativeVisionModelError> {
    total
        .checked_add(u64::try_from(bytes).map_err(|_| NativeVisionModelError::ShapeOverflow)?)
        .ok_or(NativeVisionModelError::ShapeOverflow)
}

fn raft_state_map_resident_bytes(
    mut total: u64,
    state: &BTreeMap<String, Tensor>,
) -> Result<u64, NativeVisionModelError> {
    total = checked_resident_add(
        total,
        state
            .len()
            .checked_mul(mem::size_of::<(String, Tensor)>())
            .ok_or(NativeVisionModelError::ShapeOverflow)?,
        "RAFT state entries",
    )?;
    for name in state.keys() {
        total = checked_resident_add(total, name.capacity(), "RAFT state names")?;
    }
    Ok(total)
}

fn raft_semantic_identity(
    architecture_digest_sha256: String,
    state_digest_sha256: String,
) -> Result<NativeModelResourceIdentity, NativeVisionModelError> {
    NativeModelResourceIdentity::checked(
        NativeModelResourceRole::OpticalFlow,
        RAFT_LARGE_ARCHITECTURE_ID,
        RAFT_LARGE_RESOURCE_FORMAT,
        state_digest_sha256,
        architecture_digest_sha256,
    )
    .map_err(|error| NativeVisionModelError::Invalid(error.to_string()))
}

#[derive(Clone, Debug)]
pub struct NativeRaftLarge {
    root: NativeModule,
    modules: Vec<NativeModuleSlot>,
    residual_state: BTreeMap<String, Tensor>,
    schema: Vec<NativeVisionStateSpec>,
    training: bool,
    parameters_loaded: bool,
    canonical_state: BTreeMap<String, Tensor>,
    semantic_identity: Option<NativeModelResourceIdentity>,
    module_state_digest_sha256: Option<String>,
}

#[derive(Clone, Debug)]
pub struct NativeRaftLargeExecutionSession {
    model: NativeRaftLarge,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeRaftTensorResidentAllocation {
    storage_id: StorageId,
    resident_bytes: u64,
}

impl NativeRaftTensorResidentAllocation {
    pub const fn storage_id(&self) -> StorageId {
        self.storage_id
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeRaftResidentParts {
    owned_bytes: u64,
    tensor_allocations: Vec<NativeRaftTensorResidentAllocation>,
}

impl NativeRaftResidentParts {
    pub const fn owned_bytes(&self) -> u64 {
        self.owned_bytes
    }

    pub fn tensor_allocations(&self) -> &[NativeRaftTensorResidentAllocation] {
        &self.tensor_allocations
    }

    pub fn resident_bytes(&self) -> Result<u64, NativeVisionModelError> {
        self.tensor_allocations
            .iter()
            .try_fold(self.owned_bytes, |bytes, allocation| {
                bytes
                    .checked_add(allocation.resident_bytes)
                    .ok_or(NativeVisionModelError::ShapeOverflow)
            })
    }
}

impl NativeRaftLargeExecutionSession {
    pub fn semantic_identity(
        &self,
    ) -> Result<&NativeModelResourceIdentity, NativeVisionModelError> {
        self.model
            .semantic_identity
            .as_ref()
            .ok_or(NativeVisionModelError::ParametersNotLoaded)
    }

    pub fn forward_with_context(
        &mut self,
        backend: &CpuBackend,
        image1: &Tensor,
        image2: &Tensor,
        number_of_flow_updates: usize,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, NativeVisionModelError> {
        self.model
            .forward_impl(backend, image1, image2, number_of_flow_updates, context)
    }
}

impl NativeRaftLarge {
    pub fn root(&self) -> &NativeModule {
        &self.root
    }

    pub fn state_schema(&self) -> &[NativeVisionStateSpec] {
        &self.schema
    }

    pub fn parameter_count(&self) -> Result<u64, NativeVisionModelError> {
        parameter_count(&self.schema)
    }

    pub const fn default_flow_updates(&self) -> usize {
        12
    }

    pub const fn is_training(&self) -> bool {
        self.training
    }

    pub fn semantic_identity(
        &self,
    ) -> Result<&NativeModelResourceIdentity, NativeVisionModelError> {
        self.semantic_identity
            .as_ref()
            .ok_or(NativeVisionModelError::ParametersNotLoaded)
    }

    pub fn semantic_digest_sha256(&self) -> Result<&str, NativeVisionModelError> {
        Ok(self.semantic_identity()?.digest_sha256())
    }

    pub fn resident_storage_bytes(&self) -> Result<u64, NativeVisionModelError> {
        self.resident_parts()?
            .tensor_allocations()
            .iter()
            .try_fold(0_u64, |bytes, allocation| {
                bytes
                    .checked_add(allocation.resident_bytes())
                    .ok_or(NativeVisionModelError::ShapeOverflow)
            })
    }

    fn resident_owned_bytes(&self) -> Result<u64, NativeVisionModelError> {
        let identity = self.semantic_identity()?;
        let mut bytes = u64::try_from(mem::size_of::<Self>())
            .map_err(|_| NativeVisionModelError::ShapeOverflow)?;
        bytes = checked_resident_add(
            bytes,
            self.modules
                .capacity()
                .checked_mul(mem::size_of::<NativeModuleSlot>())
                .ok_or(NativeVisionModelError::ShapeOverflow)?,
            "RAFT module slots",
        )?;
        for slot in &self.modules {
            bytes = checked_resident_add(bytes, slot.weight_name.capacity(), "RAFT module names")?;
            if let Some(name) = &slot.bias_name {
                bytes = checked_resident_add(bytes, name.capacity(), "RAFT module names")?;
            }
        }
        bytes = checked_resident_add(
            bytes,
            self.schema
                .capacity()
                .checked_mul(mem::size_of::<NativeVisionStateSpec>())
                .ok_or(NativeVisionModelError::ShapeOverflow)?,
            "RAFT schema entries",
        )?;
        for spec in &self.schema {
            bytes = checked_resident_add(bytes, spec.name.capacity(), "RAFT schema names")?;
            bytes = checked_resident_add(
                bytes,
                spec.shape
                    .capacity()
                    .checked_mul(mem::size_of::<u64>())
                    .ok_or(NativeVisionModelError::ShapeOverflow)?,
                "RAFT schema shapes",
            )?;
        }
        bytes = raft_state_map_resident_bytes(bytes, &self.canonical_state)?;
        bytes = raft_state_map_resident_bytes(bytes, &self.residual_state)?;
        if let Some(digest) = &self.module_state_digest_sha256 {
            bytes = checked_resident_add(bytes, digest.capacity(), "RAFT module digest")?;
        }
        bytes = bytes
            .checked_add(
                identity
                    .resident_owned_bytes()
                    .map_err(|_| NativeVisionModelError::ShapeOverflow)?,
            )
            .ok_or(NativeVisionModelError::ShapeOverflow)?;
        Ok(bytes)
    }

    pub fn resident_parts(&self) -> Result<NativeRaftResidentParts, NativeVisionModelError> {
        if !self.parameters_loaded {
            return Err(NativeVisionModelError::ParametersNotLoaded);
        }
        let mut storages = BTreeMap::new();
        for tensor in self.canonical_state.values() {
            let storage_id = tensor.storage_id();
            let resident_bytes = tensor.storage_byte_len();
            if let Some(existing) = storages.insert(storage_id.get(), (storage_id, resident_bytes))
                && existing.1 != resident_bytes
            {
                return Err(NativeVisionModelError::ShapeOverflow);
            }
        }
        let parts = NativeRaftResidentParts {
            owned_bytes: self.resident_owned_bytes()?,
            tensor_allocations: storages
                .into_values()
                .map(
                    |(storage_id, resident_bytes)| NativeRaftTensorResidentAllocation {
                        storage_id,
                        resident_bytes,
                    },
                )
                .collect(),
        };
        parts.resident_bytes()?;
        Ok(parts)
    }

    pub fn resident_bytes(&self) -> Result<u64, NativeVisionModelError> {
        self.resident_parts()?.resident_bytes()
    }

    pub fn validate(&self, cancellation: &CancellationToken) -> Result<(), NativeVisionModelError> {
        cancellation.check()?;
        if !self.parameters_loaded {
            return Err(NativeVisionModelError::ParametersNotLoaded);
        }
        validate_state_dictionary(&self.schema, &self.canonical_state, cancellation)?;
        let expected_identity = raft_semantic_identity(
            raft_architecture_digest(&self.schema)?,
            raft_state_digest(&self.canonical_state, cancellation)?,
        )?;
        let identity = self.semantic_identity()?;
        identity
            .validate()
            .map_err(|error| NativeVisionModelError::Invalid(error.to_string()))?;
        if identity != &expected_identity {
            return Err(NativeVisionModelError::SemanticIdentityChanged);
        }
        let module_state_digest =
            raft_module_state_digest(&self.modules, &self.residual_state, cancellation)?;
        if self.module_state_digest_sha256.as_deref() != Some(module_state_digest.as_str()) {
            return Err(NativeVisionModelError::SemanticIdentityChanged);
        }
        self.resident_bytes()?;
        cancellation.check()?;
        Ok(())
    }

    pub fn execution_session(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<NativeRaftLargeExecutionSession, NativeVisionModelError> {
        self.validate(cancellation)?;
        if self.training {
            return Err(NativeVisionModelError::EvaluationRequired("RAFT-large"));
        }
        cancellation.check()?;
        Ok(NativeRaftLargeExecutionSession {
            model: self.clone(),
        })
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    pub fn from_recipe_state(
        ordered_state: Vec<(String, Tensor)>,
        cancellation: &CancellationToken,
    ) -> Result<Self, NativeVisionModelError> {
        cancellation.check()?;
        let mut state = BTreeMap::new();
        for (index, (name, tensor)) in ordered_state.into_iter().enumerate() {
            if index.is_multiple_of(32) {
                cancellation.check()?;
            }
            if state.insert(name.clone(), tensor).is_some() {
                return Err(NativeVisionModelError::Invalid(format!(
                    "RAFT state contains duplicate tensor {name}"
                )));
            }
        }
        let mut model = raft_large_exact_native(false, false, cancellation)?;
        model.load_state_dict(state, cancellation)?;
        model.eval();
        model.validate(cancellation)?;
        Ok(model)
    }

    pub fn load_state_dict(
        &mut self,
        state: BTreeMap<String, Tensor>,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeVisionModelError> {
        validate_state_dictionary(&self.schema, &state, cancellation)?;
        let canonical_state = state.clone();
        let loaded = load_state_dictionary(&self.schema, &self.modules, state, cancellation)?;
        let semantic_identity = raft_semantic_identity(
            raft_architecture_digest(&self.schema)?,
            raft_state_digest(&canonical_state, cancellation)?,
        )?;
        let module_state_digest_sha256 =
            raft_module_state_digest(&loaded.modules, &loaded.residual_state, cancellation)?;
        cancellation.check()?;
        self.modules = loaded.modules;
        self.residual_state = loaded.residual_state;
        self.canonical_state = canonical_state;
        self.semantic_identity = Some(semantic_identity);
        self.module_state_digest_sha256 = Some(module_state_digest_sha256);
        self.parameters_loaded = true;
        Ok(())
    }

    pub fn load_from_model_store_with_context(
        &mut self,
        backend: &CpuBackend,
        store: &ModelStore,
        index: &ArtifactIndex,
        model: &LoadedModel,
        context: &ExecutionContext<'_>,
    ) -> Result<(), NativeVisionModelError> {
        let state = load_vision_state_from_model_store_with_context(
            backend,
            store,
            index,
            model,
            &self.schema,
            context,
        )?;
        self.load_state_dict(state, context.cancellation)
    }

    pub fn forward_with_context(
        &mut self,
        backend: &CpuBackend,
        image1: &Tensor,
        image2: &Tensor,
        number_of_flow_updates: usize,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, NativeVisionModelError> {
        self.forward_impl(backend, image1, image2, number_of_flow_updates, context)
    }

    fn forward_impl(
        &mut self,
        backend: &CpuBackend,
        image1: &Tensor,
        image2: &Tensor,
        number_of_flow_updates: usize,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<Tensor>, NativeVisionModelError> {
        context.cancellation.check()?;
        require_nchw_f32(image1, 3)?;
        require_nchw_f32(image2, 3)?;
        if image1.descriptor().shape() != image2.descriptor().shape() {
            return Err(NativeVisionModelError::Invalid(
                "RAFT input images must have identical shapes".into(),
            ));
        }
        let shape = image1.descriptor().shape();
        if !shape[2].is_multiple_of(8) || !shape[3].is_multiple_of(8) {
            return Err(NativeVisionModelError::Invalid(
                "RAFT input height and width must be divisible by eight".into(),
            ));
        }
        if shape[2] < 128 || shape[3] < 128 {
            return Err(NativeVisionModelError::Invalid(
                "RAFT-large correlation pyramid requires image dimensions of at least 128".into(),
            ));
        }
        if !self.parameters_loaded {
            return Err(NativeVisionModelError::ParametersNotLoaded);
        }
        if self.training {
            return Err(NativeVisionModelError::EvaluationRequired("RAFT-large"));
        }
        let execution = VisionExecution::canonical(backend, context);
        let image1 = NativeValues::from_tensor(&execution, image1)?;
        let image2 = NativeValues::from_tensor(&execution, image2)?;
        raft_forward_full(
            &mut self.modules,
            &self.residual_state,
            image1,
            image2,
            number_of_flow_updates,
            false,
            &execution,
        )
    }
}

pub fn raft_large_exact_native(
    weights_requested: bool,
    _progress: bool,
    cancellation: &CancellationToken,
) -> Result<NativeRaftLarge, NativeVisionModelError> {
    cancellation.check()?;
    if weights_requested {
        return Err(NativeVisionModelError::Invalid(
            "production construction never downloads torchvision weights; load a verified state dictionary through ModelStore".into(),
        ));
    }
    let root = NativeModule::container("torchvision.models.optical_flow.raft_large")?;
    let mut builder = SchemaBuilder::default();
    raft_encoder(&mut builder, "feature_encoder", false)?;
    raft_encoder(&mut builder, "context_encoder", true)?;
    builder.conv(
        "update_block.motion_encoder.convcorr1.0",
        324,
        256,
        1,
        1,
        1,
        true,
    )?;
    builder.conv(
        "update_block.motion_encoder.convcorr2.0",
        256,
        192,
        3,
        1,
        1,
        true,
    )?;
    builder.conv(
        "update_block.motion_encoder.convflow1.0",
        2,
        128,
        7,
        1,
        1,
        true,
    )?;
    builder.conv(
        "update_block.motion_encoder.convflow2.0",
        128,
        64,
        3,
        1,
        1,
        true,
    )?;
    builder.conv(
        "update_block.motion_encoder.conv.0",
        256,
        126,
        3,
        1,
        1,
        true,
    )?;
    for (gru, kernel_height, kernel_width) in [("convgru1", 1, 5), ("convgru2", 5, 1)] {
        for gate in ["convz", "convr", "convq"] {
            builder.conv_rect(
                &format!("update_block.recurrent_block.{gru}.{gate}"),
                384,
                128,
                kernel_height,
                kernel_width,
                1,
                1,
                true,
            )?;
        }
    }
    builder.conv("update_block.flow_head.conv1", 128, 256, 3, 1, 1, true)?;
    builder.conv("update_block.flow_head.conv2", 256, 2, 3, 1, 1, true)?;
    builder.conv("mask_predictor.convrelu.0", 128, 256, 3, 1, 1, true)?;
    builder.conv("mask_predictor.conv", 256, 576, 1, 1, 1, true)?;
    let model = NativeRaftLarge {
        root,
        modules: builder.modules,
        residual_state: BTreeMap::new(),
        schema: builder.schema,
        training: true,
        parameters_loaded: false,
        canonical_state: BTreeMap::new(),
        semantic_identity: None,
        module_state_digest_sha256: None,
    };
    if model.parameter_count()? != 5_257_536 {
        return Err(NativeVisionModelError::Invalid(
            "RAFT-large architecture does not match torchvision's 5,257,536-parameter schema"
                .into(),
        ));
    }
    cancellation.check()?;
    Ok(model)
}

fn raft_encoder(
    builder: &mut SchemaBuilder,
    prefix: &str,
    batch_norm: bool,
) -> Result<(), NativeVisionModelError> {
    if batch_norm {
        builder.conv_bn(&format!("{prefix}.convnormrelu"), 3, 64, 7, 2, 1, true)?;
    } else {
        builder.conv(&format!("{prefix}.convnormrelu.0"), 3, 64, 7, 2, 1, true)?;
    }
    for (layer, input_channels, output_channels, stride) in
        [(1, 64, 64, 1), (2, 64, 96, 2), (3, 96, 128, 2)]
    {
        for block in 0..2 {
            let block_input = if block == 0 {
                input_channels
            } else {
                output_channels
            };
            let block_stride = if block == 0 { stride } else { 1 };
            let block_prefix = format!("{prefix}.layer{layer}.{block}");
            if batch_norm {
                builder.conv_bn(
                    &format!("{block_prefix}.convnormrelu1"),
                    block_input,
                    output_channels,
                    3,
                    block_stride,
                    1,
                    true,
                )?;
                builder.conv_bn(
                    &format!("{block_prefix}.convnormrelu2"),
                    output_channels,
                    output_channels,
                    3,
                    1,
                    1,
                    true,
                )?;
            } else {
                builder.conv(
                    &format!("{block_prefix}.convnormrelu1.0"),
                    block_input,
                    output_channels,
                    3,
                    block_stride,
                    1,
                    true,
                )?;
                builder.conv(
                    &format!("{block_prefix}.convnormrelu2.0"),
                    output_channels,
                    output_channels,
                    3,
                    1,
                    1,
                    true,
                )?;
            }
            if block == 0 && (block_stride != 1 || block_input != output_channels) {
                if batch_norm {
                    builder.conv_bn(
                        &format!("{block_prefix}.downsample"),
                        block_input,
                        output_channels,
                        1,
                        block_stride,
                        1,
                        true,
                    )?;
                } else {
                    builder.conv(
                        &format!("{block_prefix}.downsample.0"),
                        block_input,
                        output_channels,
                        1,
                        block_stride,
                        1,
                        true,
                    )?;
                }
            }
        }
    }
    builder.conv(&format!("{prefix}.conv"), 128, 256, 1, 1, 1, true)?;
    Ok(())
}

#[derive(Default)]
struct SchemaBuilder {
    schema: Vec<NativeVisionStateSpec>,
    modules: Vec<NativeModuleSlot>,
}

impl SchemaBuilder {
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
        groups: usize,
        bias: bool,
    ) -> Result<(), NativeVisionModelError> {
        self.conv_rect(
            prefix,
            input_channels,
            output_channels,
            kernel,
            kernel,
            stride,
            groups,
            bias,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_rect(
        &mut self,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel_height: usize,
        kernel_width: usize,
        stride: usize,
        groups: usize,
        bias: bool,
    ) -> Result<(), NativeVisionModelError> {
        let geometry = ConvolutionGeometry::new(
            2,
            vec![stride, stride],
            vec![kernel_height / 2, kernel_width / 2],
            vec![1, 1],
            groups,
            false,
            vec![0, 0],
        )?;
        let module = NativeModule::convolution(
            prefix,
            input_channels,
            output_channels,
            vec![kernel_height, kernel_width],
            bias,
            geometry,
            false,
        )?;
        let weight_name = format!("{prefix}.weight");
        let bias_name = bias.then(|| format!("{prefix}.bias"));
        self.parameter(
            &weight_name,
            vec![
                output_channels as u64,
                (input_channels / groups) as u64,
                kernel_height as u64,
                kernel_width as u64,
            ],
        );
        if let Some(name) = &bias_name {
            self.parameter(name, vec![output_channels as u64]);
        }
        self.modules.push(NativeModuleSlot {
            weight_name,
            bias_name,
            module,
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_bn(
        &mut self,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
        groups: usize,
        conv_bias: bool,
    ) -> Result<(), NativeVisionModelError> {
        self.conv(
            &format!("{prefix}.0"),
            input_channels,
            output_channels,
            kernel,
            stride,
            groups,
            conv_bias,
        )?;
        self.batch_norm(&format!("{prefix}.1"), output_channels);
        Ok(())
    }

    fn linear(
        &mut self,
        prefix: &str,
        input_features: usize,
        output_features: usize,
        bias: bool,
    ) -> Result<(), NativeVisionModelError> {
        let module = NativeModule::linear(prefix, input_features, output_features, bias, false)?;
        let weight_name = format!("{prefix}.weight");
        let bias_name = bias.then(|| format!("{prefix}.bias"));
        self.parameter(
            &weight_name,
            vec![output_features as u64, input_features as u64],
        );
        if let Some(name) = &bias_name {
            self.parameter(name, vec![output_features as u64]);
        }
        self.modules.push(NativeModuleSlot {
            weight_name,
            bias_name,
            module,
        });
        Ok(())
    }

    fn batch_norm(&mut self, prefix: &str, channels: usize) {
        for suffix in ["weight", "bias"] {
            self.parameter(&format!("{prefix}.{suffix}"), vec![channels as u64]);
        }
        for suffix in ["running_mean", "running_var"] {
            self.buffer(
                &format!("{prefix}.{suffix}"),
                vec![channels as u64],
                DType::F32,
            );
        }
        self.buffer(
            &format!("{prefix}.num_batches_tracked"),
            Vec::new(),
            DType::I64,
        );
    }

    fn parameter(&mut self, name: &str, shape: Vec<u64>) {
        self.schema.push(NativeVisionStateSpec {
            name: name.into(),
            shape,
            dtype: DType::F32,
            kind: NativeVisionStateKind::Parameter,
        });
    }

    fn buffer(&mut self, name: &str, shape: Vec<u64>, dtype: DType) {
        self.schema.push(NativeVisionStateSpec {
            name: name.into(),
            shape,
            dtype,
            kind: NativeVisionStateKind::Buffer,
        });
    }
}

struct LoadedState {
    modules: Vec<NativeModuleSlot>,
    residual_state: BTreeMap<String, Tensor>,
}

pub fn load_vision_state_from_model_store_with_context(
    backend: &CpuBackend,
    store: &ModelStore,
    index: &ArtifactIndex,
    model: &LoadedModel,
    schema: &[NativeVisionStateSpec],
    context: &ExecutionContext<'_>,
) -> Result<BTreeMap<String, Tensor>, NativeVisionModelError> {
    load_vision_state_from_model_store_impl(backend, store, index, model, schema, &[], context)
}

pub(crate) fn load_projected_vision_state_from_model_store_with_context(
    backend: &CpuBackend,
    store: &ModelStore,
    index: &ArtifactIndex,
    model: &LoadedModel,
    projection: &[(String, NativeVisionStateSpec)],
    context: &ExecutionContext<'_>,
) -> Result<BTreeMap<String, Tensor>, NativeVisionModelError> {
    let source_names = projection
        .iter()
        .map(|(source, _)| source)
        .collect::<BTreeSet<_>>();
    let target_names = projection
        .iter()
        .map(|(_, target)| &target.name)
        .collect::<BTreeSet<_>>();
    if source_names.len() != projection.len() || target_names.len() != projection.len() {
        return Err(NativeVisionModelError::Invalid(
            "vision state projection endpoints must be unique".into(),
        ));
    }
    let projection = projection
        .iter()
        .map(|(source, target)| (source.clone(), target))
        .collect::<Vec<_>>();
    load_projected_vision_state_from_model_store_impl(
        backend,
        store,
        index,
        model,
        &projection,
        None,
        &[],
        context,
    )
}

pub fn load_vision_state_with_sibling_namespaces_from_model_store_with_context(
    backend: &CpuBackend,
    store: &ModelStore,
    index: &ArtifactIndex,
    model: &LoadedModel,
    schema: &[NativeVisionStateSpec],
    sibling_namespaces: &[&str],
    context: &ExecutionContext<'_>,
) -> Result<BTreeMap<String, Tensor>, NativeVisionModelError> {
    if sibling_namespaces
        .iter()
        .any(|prefix| prefix.is_empty() || !prefix.ends_with('.'))
    {
        return Err(NativeVisionModelError::Invalid(
            "vision state sibling namespaces must be nonempty dotted prefixes".into(),
        ));
    }
    let unique = sibling_namespaces.iter().copied().collect::<BTreeSet<_>>();
    if unique.len() != sibling_namespaces.len() {
        return Err(NativeVisionModelError::Invalid(
            "vision state sibling namespaces must be unique".into(),
        ));
    }
    load_vision_state_from_model_store_impl(
        backend,
        store,
        index,
        model,
        schema,
        sibling_namespaces,
        context,
    )
}

fn load_efficientnet_feature_state_from_model_store_with_context(
    backend: &CpuBackend,
    store: &ModelStore,
    index: &ArtifactIndex,
    model: &LoadedModel,
    schema: &[NativeVisionStateSpec],
    source: NativeEfficientNetV2SFeatureSource,
    context: &ExecutionContext<'_>,
) -> Result<BTreeMap<String, Tensor>, NativeVisionModelError> {
    let prefix = source.prefix();
    let projection = schema
        .iter()
        .map(|spec| {
            let suffix = spec.name.strip_prefix("features.").ok_or_else(|| {
                NativeVisionModelError::Invalid(format!(
                    "EfficientNet feature state {} is outside the canonical features namespace",
                    spec.name
                ))
            })?;
            Ok((format!("{prefix}{suffix}"), spec))
        })
        .collect::<Result<Vec<_>, NativeVisionModelError>>()?;
    load_projected_vision_state_from_model_store_impl(
        backend,
        store,
        index,
        model,
        &projection,
        Some(prefix),
        &[],
        context,
    )
}

fn load_vision_state_from_model_store_impl(
    backend: &CpuBackend,
    store: &ModelStore,
    index: &ArtifactIndex,
    model: &LoadedModel,
    schema: &[NativeVisionStateSpec],
    sibling_namespaces: &[&str],
    context: &ExecutionContext<'_>,
) -> Result<BTreeMap<String, Tensor>, NativeVisionModelError> {
    let projection = schema
        .iter()
        .map(|spec| (spec.name.clone(), spec))
        .collect::<Vec<_>>();
    load_projected_vision_state_from_model_store_impl(
        backend,
        store,
        index,
        model,
        &projection,
        None,
        sibling_namespaces,
        context,
    )
}

fn load_projected_vision_state_from_model_store_impl(
    backend: &CpuBackend,
    store: &ModelStore,
    index: &ArtifactIndex,
    model: &LoadedModel,
    projection: &[(String, &NativeVisionStateSpec)],
    strict_namespace: Option<&str>,
    sibling_namespaces: &[&str],
    context: &ExecutionContext<'_>,
) -> Result<BTreeMap<String, Tensor>, NativeVisionModelError> {
    context.cancellation.check()?;
    let expected_names = projection
        .iter()
        .map(|(source_name, _)| source_name.as_str())
        .collect::<BTreeSet<_>>();
    if let Some(name) = model.tensors().keys().find(|name| match strict_namespace {
        Some(prefix) => name.starts_with(prefix) && !expected_names.contains(name.as_str()),
        None => {
            !expected_names.contains(name.as_str())
                && !sibling_namespaces
                    .iter()
                    .any(|prefix| name.starts_with(prefix))
        }
    }) {
        return Err(NativeVisionModelError::UnexpectedState(name.clone()));
    }
    for (source_name, spec) in projection {
        context.cancellation.check()?;
        let metadata = model
            .tensors()
            .get(source_name)
            .ok_or_else(|| NativeVisionModelError::MissingState(source_name.clone()))?;
        if metadata.shape != spec.shape {
            return Err(NativeVisionModelError::StateShape {
                name: source_name.clone(),
                expected: spec.shape.clone(),
                actual: metadata.shape.clone(),
            });
        }
        let dtype = canonical_vision_model_store_dtype(&metadata.data_type).ok_or_else(|| {
            NativeVisionModelError::Invalid(format!(
                "ModelStore tensor {} has unsupported dtype {}",
                source_name, metadata.data_type
            ))
        })?;
        if dtype != spec.dtype {
            return Err(NativeVisionModelError::StateDType {
                name: source_name.clone(),
                expected: spec.dtype,
                actual: dtype,
            });
        }
    }
    let names = projection
        .iter()
        .map(|(source_name, _)| source_name.as_str())
        .collect::<Vec<_>>();
    let mut encoded = store.read_tensors(index, model, names, context.cancellation)?;
    let mut state = BTreeMap::new();
    for (source_name, spec) in projection {
        context.cancellation.check()?;
        let source = encoded
            .remove(source_name)
            .ok_or_else(|| NativeVisionModelError::MissingState(source_name.clone()))?;
        let expected_bytes = spec
            .shape
            .iter()
            .try_fold(1_u64, |count, dimension| {
                count
                    .checked_mul(*dimension)
                    .ok_or(NativeVisionModelError::ShapeOverflow)
            })?
            .checked_mul(spec.dtype.byte_width())
            .ok_or(NativeVisionModelError::ShapeOverflow)?;
        if u64::try_from(source.len()).map_err(|_| NativeVisionModelError::ShapeOverflow)?
            != expected_bytes
        {
            return Err(NativeVisionModelError::Invalid(format!(
                "ModelStore tensor {} has {} encoded bytes; expected {expected_bytes}",
                source_name,
                source.len()
            )));
        }
        let bytes =
            vision_model_store_native_bytes(backend, context, &source, spec.dtype, source_name)?;
        let descriptor = TensorDescriptor::contiguous(
            spec.shape.clone(),
            spec.dtype,
            DeviceId::CPU,
            context.stream,
        )?;
        let (tensor, _) = backend.upload_bytes(descriptor, &bytes, context)?;
        state.insert(spec.name.clone(), tensor);
    }
    context.cancellation.check()?;
    Ok(state)
}

pub(crate) fn canonical_vision_model_store_dtype(value: &str) -> Option<DType> {
    match value {
        "F32" | "float32" | "Float" => Some(DType::F32),
        "F16" | "float16" | "Half" => Some(DType::F16),
        "BF16" | "bfloat16" | "BFloat16" => Some(DType::Bf16),
        "I64" | "int64" | "Long" => Some(DType::I64),
        _ => None,
    }
}

fn vision_model_store_native_bytes(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    source: &[u8],
    dtype: DType,
    name: &str,
) -> Result<VisionBytes, NativeVisionModelError> {
    let width = match dtype {
        DType::F32 => 4,
        DType::F16 | DType::Bf16 => 2,
        DType::I64 => 8,
        _ => {
            return Err(NativeVisionModelError::Invalid(format!(
                "ModelStore tensor {name} has unsupported vision-model dtype {dtype:?}"
            )));
        }
    };
    if !source.len().is_multiple_of(width) {
        return Err(NativeVisionModelError::Invalid(format!(
            "ModelStore tensor {name} byte length is not aligned to {dtype:?}"
        )));
    }
    let mut bytes = VisionBytes(backend.workspace_vec(context, source.len())?);
    match dtype {
        DType::F32 => {
            for encoded in source.chunks_exact(4) {
                let encoded: [u8; 4] = encoded.try_into().map_err(|_| {
                    NativeVisionModelError::Invalid(format!(
                        "ModelStore tensor {name} contains an incomplete F32 value"
                    ))
                })?;
                bytes.extend(&f32::from_le_bytes(encoded).to_ne_bytes())?;
            }
        }
        DType::F16 | DType::Bf16 => {
            for encoded in source.chunks_exact(2) {
                let encoded: [u8; 2] = encoded.try_into().map_err(|_| {
                    NativeVisionModelError::Invalid(format!(
                        "ModelStore tensor {name} contains an incomplete {dtype:?} value"
                    ))
                })?;
                bytes.extend(&u16::from_le_bytes(encoded).to_ne_bytes())?;
            }
        }
        DType::I64 => {
            for encoded in source.chunks_exact(8) {
                let encoded: [u8; 8] = encoded.try_into().map_err(|_| {
                    NativeVisionModelError::Invalid(format!(
                        "ModelStore tensor {name} contains an incomplete I64 value"
                    ))
                })?;
                bytes.extend(&i64::from_le_bytes(encoded).to_ne_bytes())?;
            }
        }
        _ => {
            return Err(NativeVisionModelError::Invalid(format!(
                "ModelStore tensor {name} has unsupported vision-model dtype {dtype:?}"
            )));
        }
    }
    Ok(bytes)
}

#[derive(Debug)]
struct VisionBytes(CpuWorkspaceVec<u8>);

impl VisionBytes {
    fn extend(&mut self, source: &[u8]) -> Result<(), NativeVisionModelError> {
        for byte in source {
            self.0.try_push(*byte)?;
        }
        Ok(())
    }
}

impl std::ops::Deref for VisionBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn load_state_dictionary(
    schema: &[NativeVisionStateSpec],
    module_templates: &[NativeModuleSlot],
    mut state: BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<LoadedState, NativeVisionModelError> {
    validate_state_dictionary(schema, &state, cancellation)?;
    let mut modules = module_templates.to_vec();
    for slot in &mut modules {
        cancellation.check()?;
        let weight = state
            .remove(&slot.weight_name)
            .ok_or_else(|| NativeVisionModelError::MissingState(slot.weight_name.clone()))?;
        let bias = match &slot.bias_name {
            Some(name) => Some(
                state
                    .remove(name)
                    .ok_or_else(|| NativeVisionModelError::MissingState(name.clone()))?,
            ),
            None => None,
        };
        slot.module.load_dense_parameters(weight, bias)?;
    }
    cancellation.check()?;
    Ok(LoadedState {
        modules,
        residual_state: state,
    })
}

fn validate_state_dictionary(
    schema: &[NativeVisionStateSpec],
    state: &BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<(), NativeVisionModelError> {
    cancellation.check()?;
    let expected_names: BTreeSet<&str> = schema.iter().map(|spec| spec.name.as_str()).collect();
    if let Some(name) = state
        .keys()
        .find(|name| !expected_names.contains(name.as_str()))
    {
        return Err(NativeVisionModelError::UnexpectedState(name.clone()));
    }
    for spec in schema {
        cancellation.check()?;
        let tensor = state
            .get(&spec.name)
            .ok_or_else(|| NativeVisionModelError::MissingState(spec.name.clone()))?;
        if tensor.descriptor().shape() != spec.shape {
            return Err(NativeVisionModelError::StateShape {
                name: spec.name.clone(),
                expected: spec.shape.clone(),
                actual: tensor.descriptor().shape().to_vec(),
            });
        }
        if tensor.descriptor().dtype() != spec.dtype {
            return Err(NativeVisionModelError::StateDType {
                name: spec.name.clone(),
                expected: spec.dtype,
                actual: tensor.descriptor().dtype(),
            });
        }
        if tensor.descriptor().device() != DeviceId::CPU || !tensor.descriptor().is_contiguous()? {
            return Err(NativeVisionModelError::Invalid(format!(
                "state dictionary tensor {} must be contiguous on the canonical CPU backend",
                spec.name
            )));
        }
    }
    cancellation.check()?;
    Ok(())
}

fn f32_state_values(
    state: &BTreeMap<String, Tensor>,
    name: &str,
    execution: &VisionExecution<'_, '_>,
) -> Result<VisionValues, NativeVisionModelError> {
    let tensor = state
        .get(name)
        .ok_or_else(|| NativeVisionModelError::MissingState(name.into()))?;
    if tensor.descriptor().dtype() != DType::F32 || !tensor.descriptor().is_contiguous()? {
        return Err(NativeVisionModelError::Invalid(format!(
            "state dictionary tensor {name} must be contiguous F32"
        )));
    }
    let bytes = tensor.contiguous_bytes()?;
    let mut values = execution.values(bytes.len() / 4)?;
    for (index, encoded) in bytes.chunks_exact(4).enumerate() {
        if index.is_multiple_of(4096) {
            execution.context.cancellation.check()?;
        }
        let encoded: [u8; 4] = encoded.try_into().map_err(|_| {
            NativeVisionModelError::Invalid(format!("state tensor {name} is unaligned"))
        })?;
        push_value(&mut values, f32::from_ne_bytes(encoded))?;
    }
    Ok(values)
}

fn run_module(
    modules: &mut [NativeModuleSlot],
    prefix: &str,
    input: &NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    execution.context.cancellation.check()?;
    let module = modules
        .iter_mut()
        .find(|slot| slot.module.layer_name() == prefix)
        .ok_or_else(|| {
            NativeVisionModelError::Invalid(format!("missing native module {prefix}"))
        })?;
    let input = input.to_tensor(execution)?;
    let zero_output = module.module.forward_if_dense_weight_is_zero_with_context(
        execution.backend,
        &input,
        execution.context,
    )?;
    if let Some(output) = zero_output {
        return NativeValues::from_tensor(execution, &output);
    }
    let output =
        module
            .module
            .forward_with_context(execution.backend, &input, execution.context)?;
    NativeValues::from_tensor(execution, &output)
}

fn activate_silu(
    mut input: NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    silu_with_context_exact_native_in_place(
        execution.backend,
        &mut input.values,
        DeviceId::CPU,
        execution.context,
    )?;
    Ok(input)
}

fn activate_relu(
    mut input: NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    relu_with_context_exact_native_in_place(
        execution.backend,
        &mut input.values,
        DeviceId::CPU,
        execution.context,
    )?;
    Ok(input)
}

fn activate_tanh(
    mut input: NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    for (index, value) in input.values.iter_mut().enumerate() {
        if index.is_multiple_of(4096) {
            execution.context.cancellation.check()?;
        }
        *value = value.tanh();
    }
    Ok(input)
}

fn activate_sigmoid(
    mut input: NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    for (index, value) in input.values.iter_mut().enumerate() {
        if index.is_multiple_of(4096) {
            execution.context.cancellation.check()?;
        }
        *value = 1.0 / (1.0 + (-*value).exp());
    }
    Ok(input)
}

fn add_values(
    left: &NativeValues,
    right: &NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    if left.shape != right.shape || left.values.len() != right.values.len() {
        return Err(NativeVisionModelError::Invalid(
            "native model residual operands must have identical shapes".into(),
        ));
    }
    let mut values = execution.values(left.values.len())?;
    for (index, (left, right)) in left.values.iter().zip(right.values.iter()).enumerate() {
        if index.is_multiple_of(4096) {
            execution.context.cancellation.check()?;
        }
        push_value(&mut values, left + right)?;
    }
    Ok(NativeValues {
        shape: left.shape.clone(),
        values,
    })
}

fn multiply_values(
    left: &NativeValues,
    right: &NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    if left.shape != right.shape || left.values.len() != right.values.len() {
        return Err(NativeVisionModelError::Invalid(
            "native model product operands must have identical shapes".into(),
        ));
    }
    let mut values = execution.values(left.values.len())?;
    for (index, (left, right)) in left.values.iter().zip(right.values.iter()).enumerate() {
        if index.is_multiple_of(4096) {
            execution.context.cancellation.check()?;
        }
        push_value(&mut values, left * right)?;
    }
    Ok(NativeValues {
        shape: left.shape.clone(),
        values,
    })
}

fn batch_normalize(
    state: &BTreeMap<String, Tensor>,
    prefix: &str,
    input: NativeValues,
    training: bool,
    epsilon: f32,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    let [_batch, channels, _height, _width]: [usize; 4] =
        input.shape.as_slice().try_into().map_err(|_| {
            NativeVisionModelError::Invalid("batch normalization requires NCHW input".into())
        })?;
    if training {
        return Err(NativeVisionModelError::EvaluationRequired(
            "vision-model batch normalization",
        ));
    }
    let weight = f32_state_values(state, &format!("{prefix}.weight"), execution)?;
    let bias = f32_state_values(state, &format!("{prefix}.bias"), execution)?;
    let mut means = f32_state_values(state, &format!("{prefix}.running_mean"), execution)?;
    let mut variances = f32_state_values(state, &format!("{prefix}.running_var"), execution)?;
    if weight.len() != channels
        || bias.len() != channels
        || means.len() != channels
        || variances.len() != channels
    {
        return Err(NativeVisionModelError::Invalid(format!(
            "batch-normalization channel mismatch for {prefix}"
        )));
    }
    let output = batch_norm_with_context_exact_native(
        execution.backend,
        &input.values,
        &input.shape,
        Some(&mut means),
        Some(&mut variances),
        Some(&weight),
        Some(&bias),
        false,
        0.1,
        epsilon,
        DeviceId::CPU,
        execution.context,
    )?;
    let values = execution.copy(&output)?;
    Ok(NativeValues {
        shape: input.shape,
        values,
    })
}

fn instance_normalize(
    input: NativeValues,
    epsilon: f32,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    let [_batch, channels, _height, _width]: [usize; 4] =
        input.shape.as_slice().try_into().map_err(|_| {
            NativeVisionModelError::Invalid("instance normalization requires NCHW input".into())
        })?;
    let output = group_norm_with_context_exact_native(
        execution.backend,
        &input.values,
        &input.shape,
        channels,
        None,
        None,
        epsilon,
        DeviceId::CPU,
        execution.context,
    )?;
    let values = execution.copy(&output)?;
    Ok(NativeValues {
        shape: input.shape,
        values,
    })
}

fn adaptive_average_pool(
    input: &NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    let [batch, channels, height, width]: [usize; 4] =
        input.shape.as_slice().try_into().map_err(|_| {
            NativeVisionModelError::Invalid("adaptive average pooling requires NCHW input".into())
        })?;
    let spatial = height
        .checked_mul(width)
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    let output_count = batch
        .checked_mul(channels)
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    let mut values = execution.values(output_count)?;
    for batch_index in 0..batch {
        for channel in 0..channels {
            execution.context.cancellation.check()?;
            let start = (batch_index * channels + channel) * spatial;
            push_value(
                &mut values,
                input.values[start..start + spatial]
                    .iter()
                    .copied()
                    .sum::<f32>()
                    / spatial as f32,
            )?;
        }
    }
    Ok(NativeValues {
        shape: vec![batch, channels],
        values,
    })
}

fn efficientnet_conv_bn(
    modules: &mut [NativeModuleSlot],
    state: &BTreeMap<String, Tensor>,
    prefix: &str,
    input: &NativeValues,
    activation: bool,
    training: bool,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    let output = run_module(modules, &format!("{prefix}.0"), input, execution)?;
    let output = batch_normalize(
        state,
        &format!("{prefix}.1"),
        output,
        training,
        1e-3,
        execution,
    )?;
    if activation {
        activate_silu(output, execution)
    } else {
        Ok(output)
    }
}

fn efficientnet_se(
    modules: &mut [NativeModuleSlot],
    prefix: &str,
    input: &NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    let [batch, channels, height, width]: [usize; 4] =
        input.shape.as_slice().try_into().map_err(|_| {
            NativeVisionModelError::Invalid("squeeze excitation requires NCHW input".into())
        })?;
    let pooled = adaptive_average_pool(input, execution)?;
    let pooled = NativeValues {
        shape: vec![batch, channels, 1, 1],
        values: pooled.values,
    };
    let hidden = activate_silu(
        run_module(modules, &format!("{prefix}.fc1"), &pooled, execution)?,
        execution,
    )?;
    let gates = activate_sigmoid(
        run_module(modules, &format!("{prefix}.fc2"), &hidden, execution)?,
        execution,
    )?;
    let spatial = height
        .checked_mul(width)
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    let mut expanded = execution.values(input.values.len())?;
    for value in gates.values.iter().copied() {
        for _ in 0..spatial {
            push_value(&mut expanded, value)?;
        }
    }
    multiply_values(
        input,
        &NativeValues {
            shape: input.shape.clone(),
            values: expanded,
        },
        execution,
    )
}

fn efficientnet_features_full(
    modules: &mut [NativeModuleSlot],
    state: &mut BTreeMap<String, Tensor>,
    input: NativeValues,
    training: bool,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    let mut output = efficientnet_conv_bn(
        modules,
        state,
        "features.0",
        &input,
        true,
        training,
        execution,
    )?;
    drop(input);
    for (stage_index, stage) in EFFICIENTNET_V2_S_STAGES.iter().enumerate() {
        let mut input_channels = stage.input_channels;
        for block_index in 0..stage.layers {
            execution.context.cancellation.check()?;
            let stride = if block_index == 0 { stage.stride } else { 1 };
            let prefix = format!("features.{}.{}.block", stage_index + 1, block_index);
            let expanded_channels = make_divisible(input_channels * stage.expand_ratio, 8);
            let residual = output.try_clone(execution)?;
            output = match stage.block {
                NativeEfficientNetBlockKind::FusedMbConv if expanded_channels == input_channels => {
                    efficientnet_conv_bn(
                        modules,
                        state,
                        &format!("{prefix}.0"),
                        &output,
                        true,
                        training,
                        execution,
                    )?
                }
                NativeEfficientNetBlockKind::FusedMbConv => {
                    let expanded = efficientnet_conv_bn(
                        modules,
                        state,
                        &format!("{prefix}.0"),
                        &output,
                        true,
                        training,
                        execution,
                    )?;
                    efficientnet_conv_bn(
                        modules,
                        state,
                        &format!("{prefix}.1"),
                        &expanded,
                        false,
                        training,
                        execution,
                    )?
                }
                NativeEfficientNetBlockKind::MbConv => {
                    let expanded = efficientnet_conv_bn(
                        modules,
                        state,
                        &format!("{prefix}.0"),
                        &output,
                        true,
                        training,
                        execution,
                    )?;
                    let depthwise = efficientnet_conv_bn(
                        modules,
                        state,
                        &format!("{prefix}.1"),
                        &expanded,
                        true,
                        training,
                        execution,
                    )?;
                    let excited =
                        efficientnet_se(modules, &format!("{prefix}.2"), &depthwise, execution)?;
                    efficientnet_conv_bn(
                        modules,
                        state,
                        &format!("{prefix}.3"),
                        &excited,
                        false,
                        training,
                        execution,
                    )?
                }
            };
            if stride == 1 && input_channels == stage.output_channels {
                output = add_values(&output, &residual, execution)?;
            }
            input_channels = stage.output_channels;
        }
    }
    efficientnet_conv_bn(
        modules,
        state,
        "features.7",
        &output,
        true,
        training,
        execution,
    )
}

fn concatenate_batch(
    first: &NativeValues,
    second: &NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    let [first_batch, channels, height, width]: [usize; 4] =
        first.shape.as_slice().try_into().map_err(|_| {
            NativeVisionModelError::Invalid("batch concatenation requires NCHW input".into())
        })?;
    let [second_batch, second_channels, second_height, second_width]: [usize; 4] =
        second.shape.as_slice().try_into().map_err(|_| {
            NativeVisionModelError::Invalid("batch concatenation requires NCHW input".into())
        })?;
    if (channels, height, width) != (second_channels, second_height, second_width) {
        return Err(NativeVisionModelError::Invalid(
            "batch concatenation shape mismatch".into(),
        ));
    }
    let capacity = first
        .values
        .len()
        .checked_add(second.values.len())
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    let mut values = execution.values(capacity)?;
    extend_values(&mut values, &first.values)?;
    extend_values(&mut values, &second.values)?;
    let combined_batch = first_batch
        .checked_add(second_batch)
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    Ok(NativeValues {
        shape: vec![combined_batch, channels, height, width],
        values,
    })
}

fn concatenate_channels(
    first: &NativeValues,
    second: &NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    let [batch, first_channels, height, width]: [usize; 4] =
        first.shape.as_slice().try_into().map_err(|_| {
            NativeVisionModelError::Invalid("channel concatenation requires NCHW input".into())
        })?;
    let [second_batch, second_channels, second_height, second_width]: [usize; 4] =
        second.shape.as_slice().try_into().map_err(|_| {
            NativeVisionModelError::Invalid("channel concatenation requires NCHW input".into())
        })?;
    if (batch, height, width) != (second_batch, second_height, second_width) {
        return Err(NativeVisionModelError::Invalid(
            "channel concatenation shape mismatch".into(),
        ));
    }
    let spatial = height
        .checked_mul(width)
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    let capacity = first
        .values
        .len()
        .checked_add(second.values.len())
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    let mut values = execution.values(capacity)?;
    for batch_index in 0..batch {
        execution.context.cancellation.check()?;
        let first_start = batch_index * first_channels * spatial;
        extend_values(
            &mut values,
            &first.values[first_start..first_start + first_channels * spatial],
        )?;
        let second_start = batch_index * second_channels * spatial;
        extend_values(
            &mut values,
            &second.values[second_start..second_start + second_channels * spatial],
        )?;
    }
    let combined_channels = first_channels
        .checked_add(second_channels)
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    Ok(NativeValues {
        shape: vec![batch, combined_channels, height, width],
        values,
    })
}

fn split_channels(
    input: &NativeValues,
    first_channels: usize,
    execution: &VisionExecution<'_, '_>,
) -> Result<(NativeValues, NativeValues), NativeVisionModelError> {
    let [batch, channels, height, width]: [usize; 4] =
        input.shape.as_slice().try_into().map_err(|_| {
            NativeVisionModelError::Invalid("channel split requires NCHW input".into())
        })?;
    if first_channels > channels {
        return Err(NativeVisionModelError::Invalid(
            "channel split exceeds input channels".into(),
        ));
    }
    let second_channels = channels - first_channels;
    let spatial = height
        .checked_mul(width)
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    let first_count = batch
        .checked_mul(first_channels)
        .and_then(|value| value.checked_mul(spatial))
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    let second_count = batch
        .checked_mul(second_channels)
        .and_then(|value| value.checked_mul(spatial))
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    let mut first = execution.values(first_count)?;
    let mut second = execution.values(second_count)?;
    for batch_index in 0..batch {
        let start = batch_index * channels * spatial;
        extend_values(
            &mut first,
            &input.values[start..start + first_channels * spatial],
        )?;
        extend_values(
            &mut second,
            &input.values[start + first_channels * spatial..start + channels * spatial],
        )?;
    }
    Ok((
        NativeValues {
            shape: vec![batch, first_channels, height, width],
            values: first,
        },
        NativeValues {
            shape: vec![batch, second_channels, height, width],
            values: second,
        },
    ))
}

fn split_batch(
    input: &NativeValues,
    first_batch: usize,
    execution: &VisionExecution<'_, '_>,
) -> Result<(NativeValues, NativeValues), NativeVisionModelError> {
    let [batch, channels, height, width]: [usize; 4] =
        input.shape.as_slice().try_into().map_err(|_| {
            NativeVisionModelError::Invalid("batch split requires NCHW input".into())
        })?;
    if first_batch > batch {
        return Err(NativeVisionModelError::Invalid(
            "batch split exceeds input".into(),
        ));
    }
    let per_batch = channels
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    let split = first_batch
        .checked_mul(per_batch)
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    Ok((
        NativeValues {
            shape: vec![first_batch, channels, height, width],
            values: execution.copy(&input.values[..split])?,
        },
        NativeValues {
            shape: vec![batch - first_batch, channels, height, width],
            values: execution.copy(&input.values[split..])?,
        },
    ))
}

fn raft_normalize(
    state: &BTreeMap<String, Tensor>,
    prefix: &str,
    input: NativeValues,
    batch_norm: bool,
    training: bool,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    if batch_norm {
        batch_normalize(state, prefix, input, training, 1e-5, execution)
    } else {
        instance_normalize(input, 1e-5, execution)
    }
}

#[allow(clippy::too_many_arguments)]
fn raft_conv_norm(
    modules: &mut [NativeModuleSlot],
    state: &BTreeMap<String, Tensor>,
    prefix: &str,
    input: &NativeValues,
    batch_norm: bool,
    training: bool,
    activation: bool,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    let output = run_module(modules, &format!("{prefix}.0"), input, execution)?;
    let output = raft_normalize(
        state,
        &format!("{prefix}.1"),
        output,
        batch_norm,
        training,
        execution,
    )?;
    if activation {
        activate_relu(output, execution)
    } else {
        Ok(output)
    }
}

#[allow(clippy::too_many_arguments)]
fn raft_encoder_full(
    modules: &mut [NativeModuleSlot],
    state: &BTreeMap<String, Tensor>,
    prefix: &str,
    input: NativeValues,
    batch_norm: bool,
    training: bool,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    let first = run_module(
        modules,
        &format!("{prefix}.convnormrelu.0"),
        &input,
        execution,
    )?;
    let mut output = activate_relu(
        raft_normalize(
            state,
            &format!("{prefix}.convnormrelu.1"),
            first,
            batch_norm,
            training,
            execution,
        )?,
        execution,
    )?;
    for (layer, input_channels, output_channels, stride) in
        [(1, 64, 64, 1), (2, 64, 96, 2), (3, 96, 128, 2)]
    {
        for block in 0..2 {
            execution.context.cancellation.check()?;
            let block_input_channels = if block == 0 {
                input_channels
            } else {
                output_channels
            };
            let block_stride = if block == 0 { stride } else { 1 };
            let block_prefix = format!("{prefix}.layer{layer}.{block}");
            let residual = output.try_clone(execution)?;
            let first = raft_conv_norm(
                modules,
                state,
                &format!("{block_prefix}.convnormrelu1"),
                &output,
                batch_norm,
                training,
                true,
                execution,
            )?;
            let second = raft_conv_norm(
                modules,
                state,
                &format!("{block_prefix}.convnormrelu2"),
                &first,
                batch_norm,
                training,
                false,
                execution,
            )?;
            let residual = if block_stride != 1 || block_input_channels != output_channels {
                raft_conv_norm(
                    modules,
                    state,
                    &format!("{block_prefix}.downsample"),
                    &residual,
                    batch_norm,
                    training,
                    false,
                    execution,
                )?
            } else {
                residual
            };
            output = activate_relu(add_values(&residual, &second, execution)?, execution)?;
        }
    }
    run_module(modules, &format!("{prefix}.conv"), &output, execution)
}

fn average_pool_two(
    input: &NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    let [batch, channels, height, width]: [usize; 4] =
        input.shape.as_slice().try_into().map_err(|_| {
            NativeVisionModelError::Invalid("average pooling requires NCHW input".into())
        })?;
    let (output_height, output_width) = (height / 2, width / 2);
    let output_count = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(output_height))
        .and_then(|value| value.checked_mul(output_width))
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    let mut values = execution.zeroed(output_count)?;
    for batch_index in 0..batch {
        for channel in 0..channels {
            for y in 0..output_height {
                for x in 0..output_width {
                    let mut sum = 0.0;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            sum += input.values[((batch_index * channels + channel) * height
                                + y * 2
                                + dy)
                                * width
                                + x * 2
                                + dx];
                        }
                    }
                    values[((batch_index * channels + channel) * output_height + y)
                        * output_width
                        + x] = sum * 0.25;
                }
            }
        }
    }
    Ok(NativeValues {
        shape: vec![batch, channels, output_height, output_width],
        values,
    })
}

fn build_correlation_pyramid(
    first: &NativeValues,
    second: &NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<Vec<NativeValues>, NativeVisionModelError> {
    if first.shape != second.shape {
        return Err(NativeVisionModelError::Invalid(
            "RAFT feature maps must match".into(),
        ));
    }
    let [batch, channels, height, width]: [usize; 4] =
        first.shape.as_slice().try_into().map_err(|_| {
            NativeVisionModelError::Invalid("RAFT correlation requires NCHW features".into())
        })?;
    let rows = batch
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    let value_count = rows
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    let mut values = execution.zeroed(value_count)?;
    if first.values.iter().all(|value| *value == 0.0)
        || second.values.iter().all(|value| *value == 0.0)
    {
        let mut pyramid = vec![NativeValues {
            shape: vec![rows, 1, height, width],
            values,
        }];
        for _ in 1..4 {
            let next = average_pool_two(
                pyramid.last().ok_or_else(|| {
                    NativeVisionModelError::Invalid("empty correlation pyramid".into())
                })?,
                execution,
            )?;
            pyramid.push(next);
        }
        return Ok(pyramid);
    }
    let scale = (channels as f32).sqrt().recip();
    for batch_index in 0..batch {
        for first_y in 0..height {
            for first_x in 0..width {
                execution.context.cancellation.check()?;
                let row = (batch_index * height + first_y) * width + first_x;
                for second_y in 0..height {
                    for second_x in 0..width {
                        let mut dot = 0.0;
                        for channel in 0..channels {
                            let first_index =
                                ((batch_index * channels + channel) * height + first_y) * width
                                    + first_x;
                            let second_index =
                                ((batch_index * channels + channel) * height + second_y) * width
                                    + second_x;
                            dot =
                                first.values[first_index].mul_add(second.values[second_index], dot);
                        }
                        values[(row * height + second_y) * width + second_x] = dot * scale;
                    }
                }
            }
        }
    }
    let mut pyramid = vec![NativeValues {
        shape: vec![rows, 1, height, width],
        values,
    }];
    for _ in 1..4 {
        let next = average_pool_two(
            pyramid.last().ok_or_else(|| {
                NativeVisionModelError::Invalid("empty correlation pyramid".into())
            })?,
            execution,
        )?;
        pyramid.push(next);
    }
    Ok(pyramid)
}

fn sample_bilinear(
    input: &NativeValues,
    batch: usize,
    y: f32,
    x: f32,
) -> Result<f32, NativeVisionModelError> {
    let height =
        u64::try_from(input.shape[2]).map_err(|_| NativeVisionModelError::ShapeOverflow)?;
    let width = u64::try_from(input.shape[3]).map_err(|_| NativeVisionModelError::ShapeOverflow)?;
    let mut result = 0.0;
    for sample in checked_bilinear_weights(
        height,
        width,
        y,
        x,
        NativeBilinearBoundary::ZeroPadding,
        RAFT_LARGE_OPERATION_ID,
    )? {
        let source_y =
            usize::try_from(sample.source_y).map_err(|_| NativeVisionModelError::ShapeOverflow)?;
        let source_x =
            usize::try_from(sample.source_x).map_err(|_| NativeVisionModelError::ShapeOverflow)?;
        result += input.values[(batch * input.shape[2] + source_y) * input.shape[3] + source_x]
            * sample.weight;
    }
    Ok(result)
}

fn index_correlation_pyramid(
    pyramid: &[NativeValues],
    flow: &NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    let [batch, channels, height, width]: [usize; 4] = flow
        .shape
        .as_slice()
        .try_into()
        .map_err(|_| NativeVisionModelError::Invalid("RAFT flow must be NCHW".into()))?;
    if channels != 2 || pyramid.len() != 4 {
        return Err(NativeVisionModelError::Invalid(
            "RAFT correlation indexing configuration mismatch".into(),
        ));
    }
    let value_count = batch
        .checked_mul(324)
        .and_then(|value| value.checked_mul(height))
        .and_then(|value| value.checked_mul(width))
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    let mut values = execution.zeroed(value_count)?;
    for batch_index in 0..batch {
        for y in 0..height {
            for x in 0..width {
                execution.context.cancellation.check()?;
                let flow_x = flow.values[((batch_index * 2) * height + y) * width + x];
                let flow_y = flow.values[((batch_index * 2 + 1) * height + y) * width + x];
                let row = (batch_index * height + y) * width + x;
                let mut output_channel = 0;
                for (level, volume) in pyramid.iter().enumerate() {
                    let scale = (1_usize << level) as f32;
                    for dy in -4..=4 {
                        for dx in -4..=4 {
                            values
                                [((batch_index * 324 + output_channel) * height + y) * width + x] =
                                sample_bilinear(
                                    volume,
                                    row,
                                    (y as f32 + flow_y) / scale + dy as f32,
                                    (x as f32 + flow_x) / scale + dx as f32,
                                )?;
                            output_channel += 1;
                        }
                    }
                }
            }
        }
    }
    Ok(NativeValues {
        shape: vec![batch, 324, height, width],
        values,
    })
}

fn raft_motion_encoder(
    modules: &mut [NativeModuleSlot],
    flow: &NativeValues,
    correlation: &NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    let correlation = activate_relu(
        run_module(
            modules,
            "update_block.motion_encoder.convcorr1.0",
            correlation,
            execution,
        )?,
        execution,
    )?;
    let correlation = activate_relu(
        run_module(
            modules,
            "update_block.motion_encoder.convcorr2.0",
            &correlation,
            execution,
        )?,
        execution,
    )?;
    let encoded_flow = activate_relu(
        run_module(
            modules,
            "update_block.motion_encoder.convflow1.0",
            flow,
            execution,
        )?,
        execution,
    )?;
    let encoded_flow = activate_relu(
        run_module(
            modules,
            "update_block.motion_encoder.convflow2.0",
            &encoded_flow,
            execution,
        )?,
        execution,
    )?;
    let joined = concatenate_channels(&correlation, &encoded_flow, execution)?;
    let joined = activate_relu(
        run_module(
            modules,
            "update_block.motion_encoder.conv.0",
            &joined,
            execution,
        )?,
        execution,
    )?;
    concatenate_channels(&joined, flow, execution)
}

fn raft_gru(
    modules: &mut [NativeModuleSlot],
    prefix: &str,
    hidden: NativeValues,
    input: &NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    let joined = concatenate_channels(&hidden, input, execution)?;
    let update = activate_sigmoid(
        run_module(modules, &format!("{prefix}.convz"), &joined, execution)?,
        execution,
    )?;
    let reset = activate_sigmoid(
        run_module(modules, &format!("{prefix}.convr"), &joined, execution)?,
        execution,
    )?;
    let reset_hidden = multiply_values(&reset, &hidden, execution)?;
    let candidate_input = concatenate_channels(&reset_hidden, input, execution)?;
    let candidate = activate_tanh(
        run_module(
            modules,
            &format!("{prefix}.convq"),
            &candidate_input,
            execution,
        )?,
        execution,
    )?;
    let mut values = execution.values(hidden.values.len())?;
    for index in 0..hidden.values.len() {
        if index.is_multiple_of(4096) {
            execution.context.cancellation.check()?;
        }
        push_value(
            &mut values,
            (1.0 - update.values[index]) * hidden.values[index]
                + update.values[index] * candidate.values[index],
        )?;
    }
    Ok(NativeValues {
        shape: hidden.shape,
        values,
    })
}

fn raft_update(
    modules: &mut [NativeModuleSlot],
    hidden: NativeValues,
    context: &NativeValues,
    correlation: &NativeValues,
    flow: &NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<(NativeValues, NativeValues), NativeVisionModelError> {
    let motion = raft_motion_encoder(modules, flow, correlation, execution)?;
    let input = concatenate_channels(context, &motion, execution)?;
    let hidden = raft_gru(
        modules,
        "update_block.recurrent_block.convgru1",
        hidden,
        &input,
        execution,
    )?;
    let hidden = raft_gru(
        modules,
        "update_block.recurrent_block.convgru2",
        hidden,
        &input,
        execution,
    )?;
    let delta = activate_relu(
        run_module(modules, "update_block.flow_head.conv1", &hidden, execution)?,
        execution,
    )?;
    let delta = run_module(modules, "update_block.flow_head.conv2", &delta, execution)?;
    Ok((hidden, delta))
}

fn raft_mask(
    modules: &mut [NativeModuleSlot],
    hidden: &NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    let mask = activate_relu(
        run_module(modules, "mask_predictor.convrelu.0", hidden, execution)?,
        execution,
    )?;
    let mut mask = run_module(modules, "mask_predictor.conv", &mask, execution)?;
    for value in mask.values.iter_mut() {
        *value *= 0.25;
    }
    Ok(mask)
}

fn convex_upsample(
    flow: &NativeValues,
    mask: &NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    let [batch, flow_channels, height, width]: [usize; 4] = flow
        .shape
        .as_slice()
        .try_into()
        .map_err(|_| NativeVisionModelError::Invalid("RAFT flow must be NCHW".into()))?;
    if flow_channels != 2 || mask.shape != [batch, 576, height, width] {
        return Err(NativeVisionModelError::Invalid(
            "RAFT upsampling mask shape mismatch".into(),
        ));
    }
    let output_height = height
        .checked_mul(8)
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    let output_width = width
        .checked_mul(8)
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    let output_count = batch
        .checked_mul(2)
        .and_then(|value| value.checked_mul(output_height))
        .and_then(|value| value.checked_mul(output_width))
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    let mut output = execution.zeroed(output_count)?;
    for batch_index in 0..batch {
        for y in 0..height {
            for x in 0..width {
                for sub_y in 0..8 {
                    for sub_x in 0..8 {
                        execution.context.cancellation.check()?;
                        let mut logits = [0.0; 9];
                        let mut maximum = f32::NEG_INFINITY;
                        for neighbor in 0..9 {
                            let channel = (neighbor * 8 + sub_y) * 8 + sub_x;
                            logits[neighbor] = mask.values
                                [((batch_index * 576 + channel) * height + y) * width + x];
                            maximum = maximum.max(logits[neighbor]);
                        }
                        let mut total = 0.0;
                        for logit in &mut logits {
                            *logit = (*logit - maximum).exp();
                            total += *logit;
                        }
                        for channel in 0..2 {
                            let mut value = 0.0;
                            for (neighbor, weight) in logits.iter().enumerate() {
                                let neighbor_y = y as isize + neighbor as isize / 3 - 1;
                                let neighbor_x = x as isize + neighbor as isize % 3 - 1;
                                if neighbor_y >= 0
                                    && neighbor_y < height as isize
                                    && neighbor_x >= 0
                                    && neighbor_x < width as isize
                                {
                                    let index = ((batch_index * 2 + channel) * height
                                        + neighbor_y as usize)
                                        * width
                                        + neighbor_x as usize;
                                    value += *weight / total * flow.values[index] * 8.0;
                                }
                            }
                            output[((batch_index * 2 + channel) * output_height + y * 8 + sub_y)
                                * output_width
                                + x * 8
                                + sub_x] = value;
                        }
                    }
                }
            }
        }
    }
    Ok(NativeValues {
        shape: vec![batch, 2, output_height, output_width],
        values: output,
    })
}

#[allow(clippy::too_many_arguments)]
fn raft_forward_full(
    modules: &mut [NativeModuleSlot],
    state: &BTreeMap<String, Tensor>,
    image1: NativeValues,
    image2: NativeValues,
    updates: usize,
    training: bool,
    execution: &VisionExecution<'_, '_>,
) -> Result<Vec<Tensor>, NativeVisionModelError> {
    let first_batch = image1.shape[0];
    let image1 = normalize_raft_image(image1, execution)?;
    let image2 = normalize_raft_image(image2, execution)?;
    let joined = concatenate_batch(&image1, &image2, execution)?;
    drop(image2);
    let features = raft_encoder_full(
        modules,
        state,
        "feature_encoder",
        joined,
        false,
        training,
        execution,
    )?;
    let (feature1, feature2) = split_batch(&features, first_batch, execution)?;
    drop(features);
    let pyramid = build_correlation_pyramid(&feature1, &feature2, execution)?;
    let feature_height = feature1.shape[2];
    let feature_width = feature1.shape[3];
    drop(feature1);
    drop(feature2);
    let encoded_context = raft_encoder_full(
        modules,
        state,
        "context_encoder",
        image1,
        true,
        training,
        execution,
    )?;
    let (hidden, context) = split_channels(&encoded_context, 128, execution)?;
    drop(encoded_context);
    let mut hidden = activate_tanh(hidden, execution)?;
    let context = activate_relu(context, execution)?;
    let flow_count = first_batch
        .checked_mul(2)
        .and_then(|value| value.checked_mul(feature_height))
        .and_then(|value| value.checked_mul(feature_width))
        .ok_or(NativeVisionModelError::ShapeOverflow)?;
    let mut flow = NativeValues {
        shape: vec![first_batch, 2, feature_height, feature_width],
        values: execution.zeroed(flow_count)?,
    };
    let mut predictions = Vec::new();
    predictions
        .try_reserve_exact(updates)
        .map_err(|_| NativeVisionModelError::ShapeOverflow)?;
    for _ in 0..updates {
        let correlation = index_correlation_pyramid(&pyramid, &flow, execution)?;
        let (next_hidden, delta) =
            raft_update(modules, hidden, &context, &correlation, &flow, execution)?;
        hidden = next_hidden;
        flow = add_values(&flow, &delta, execution)?;
        let mask = raft_mask(modules, &hidden, execution)?;
        let prediction = convex_upsample(&flow, &mask, execution)?;
        predictions.push(prediction.to_tensor(execution)?);
    }
    Ok(predictions)
}

fn normalize_raft_image(
    mut image: NativeValues,
    execution: &VisionExecution<'_, '_>,
) -> Result<NativeValues, NativeVisionModelError> {
    for (index, value) in image.values.iter_mut().enumerate() {
        if index.is_multiple_of(4096) {
            execution.context.cancellation.check()?;
        }
        *value = 2.0f32.mul_add(*value, -1.0);
    }
    execution.context.cancellation.check()?;
    Ok(image)
}

fn parameter_count(schema: &[NativeVisionStateSpec]) -> Result<u64, NativeVisionModelError> {
    schema
        .iter()
        .filter(|spec| spec.kind == NativeVisionStateKind::Parameter)
        .try_fold(0_u64, |total, spec| {
            let count = spec.shape.iter().try_fold(1_u64, |count, dimension| {
                count
                    .checked_mul(*dimension)
                    .ok_or(NativeVisionModelError::ShapeOverflow)
            })?;
            total
                .checked_add(count)
                .ok_or(NativeVisionModelError::ShapeOverflow)
        })
}

fn require_nchw_f32(input: &Tensor, channels: u64) -> Result<(), NativeVisionModelError> {
    if input.descriptor().device() != DeviceId::CPU
        || input.descriptor().dtype() != DType::F32
        || input.descriptor().shape().len() != 4
        || input.descriptor().shape()[1] != channels
    {
        return Err(NativeVisionModelError::Invalid(format!(
            "input must be a rank-four CPU F32 tensor with {channels} channels"
        )));
    }
    Ok(())
}

const fn make_divisible(value: usize, divisor: usize) -> usize {
    let rounded = (value + divisor / 2) / divisor * divisor;
    if rounded * 10 < value * 9 {
        rounded + divisor
    } else {
        rounded
    }
}

#[cfg(test)]
mod tests {
    use super::{
        NativeModelResourceRole, NativeRaftLarge, NativeRaftResidentParts,
        NativeRaftTensorResidentAllocation, NativeValues, NativeVisionModelError,
        RAFT_LARGE_ARCHITECTURE_ID, RAFT_LARGE_OPERATION_ID, RAFT_LARGE_RESOURCE_FORMAT,
        RAFT_LARGE_RESOURCE_ROLE, RAFT_LARGE_SOURCE_TYPE_ID, VisionExecution, normalize_raft_image,
        raft_large_exact_native,
    };
    use sim_tensor::{
        CancellationToken, CpuBackend, CpuWorkspaceAuthority, DType, DeviceId, ExecutionContext,
        StreamId, Tensor, TensorDescriptor,
    };
    use std::{
        collections::{BTreeMap, BTreeSet},
        mem,
    };

    fn loaded_zero_raft(
        backend: &CpuBackend,
        workspace_authority: &CpuWorkspaceAuthority,
        cancellation: &CancellationToken,
    ) -> Result<NativeRaftLarge, NativeVisionModelError> {
        let mut model = raft_large_exact_native(false, false, cancellation)?;
        let mut shared_tensors: Vec<(Vec<u64>, DType, Tensor)> = Vec::new();
        let mut state = BTreeMap::new();
        for spec in model.state_schema() {
            cancellation.check()?;
            let tensor = match shared_tensors
                .iter()
                .find(|(shape, dtype, _)| shape == &spec.shape && *dtype == spec.dtype)
            {
                Some((_, _, tensor)) => tensor.clone(),
                None => {
                    let encoded_bytes = spec
                        .shape
                        .iter()
                        .try_fold(spec.dtype.byte_width(), |bytes, dimension| {
                            bytes.checked_mul(*dimension)
                        })
                        .ok_or(NativeVisionModelError::ShapeOverflow)?;
                    let encoded_bytes = usize::try_from(encoded_bytes)
                        .map_err(|_| NativeVisionModelError::ShapeOverflow)?;
                    let descriptor = TensorDescriptor::contiguous(
                        spec.shape.clone(),
                        spec.dtype,
                        DeviceId::CPU,
                        StreamId::DEFAULT,
                    )?;
                    let context = ExecutionContext {
                        stream: StreamId::DEFAULT,
                        scratch: workspace_authority.authorize_workspace(
                            u64::try_from(encoded_bytes.max(1))
                                .map_err(|_| NativeVisionModelError::ShapeOverflow)?,
                        )?,
                        rng_phase: None,
                        cancellation,
                    };
                    let bytes = vec![0_u8; encoded_bytes];
                    let tensor = backend.upload_bytes(descriptor, &bytes, &context)?.0;
                    shared_tensors.push((spec.shape.clone(), spec.dtype, tensor.clone()));
                    tensor
                }
            };
            state.insert(spec.name.clone(), tensor);
        }
        model.load_state_dict(state, cancellation)?;
        model.eval();
        Ok(model)
    }

    fn test_image(
        backend: &CpuBackend,
        workspace_authority: &CpuWorkspaceAuthority,
        cancellation: &CancellationToken,
        side: u64,
    ) -> Result<Tensor, NativeVisionModelError> {
        let elements = side
            .checked_mul(side)
            .and_then(|value| value.checked_mul(3))
            .ok_or(NativeVisionModelError::ShapeOverflow)?;
        let elements =
            usize::try_from(elements).map_err(|_| NativeVisionModelError::ShapeOverflow)?;
        let descriptor = TensorDescriptor::contiguous(
            vec![1, 3, side, side],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: workspace_authority.authorize_workspace(
                u64::try_from(
                    elements
                        .checked_mul(mem::size_of::<f32>())
                        .ok_or(NativeVisionModelError::ShapeOverflow)?,
                )
                .map_err(|_| NativeVisionModelError::ShapeOverflow)?,
            )?,
            rng_phase: None,
            cancellation,
        };
        Ok(backend
            .upload_f32(descriptor, &vec![0.0; elements], &context)?
            .0)
    }

    #[test]
    fn raft_resource_identity_and_residency_are_owner_derived_and_alias_aware()
    -> Result<(), NativeVisionModelError> {
        let cancellation = CancellationToken::default();
        let (backend, workspace_authority) =
            CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
        let model = loaded_zero_raft(&backend, &workspace_authority, &cancellation)?;
        model.validate(&cancellation)?;

        let identity = model.semantic_identity()?;
        assert_eq!(identity.role(), NativeModelResourceRole::OpticalFlow);
        assert_eq!(identity.role().source_type_id(), RAFT_LARGE_SOURCE_TYPE_ID);
        assert_eq!(RAFT_LARGE_RESOURCE_ROLE, "optical_flow");
        assert_eq!(RAFT_LARGE_OPERATION_ID, "COMFY-TENSOR-OP-852D8E9DBC9C");
        assert_eq!(identity.identifier(), RAFT_LARGE_ARCHITECTURE_ID);
        assert_eq!(identity.format(), RAFT_LARGE_RESOURCE_FORMAT);
        assert_eq!(identity.execution_sha256().len(), 64);
        assert_eq!(identity.artifact_sha256().len(), 64);
        assert_eq!(identity.digest_sha256().len(), 64);

        let unique_storage =
            model
                .canonical_state
                .values()
                .fold(BTreeMap::new(), |mut storages, tensor| {
                    storages
                        .entry(tensor.storage_id().get())
                        .or_insert(tensor.storage_byte_len());
                    storages
                });
        let expected_storage = unique_storage.values().try_fold(0_u64, |total, bytes| {
            total
                .checked_add(*bytes)
                .ok_or(NativeVisionModelError::ShapeOverflow)
        })?;
        let unaliased_storage =
            model
                .canonical_state
                .values()
                .try_fold(0_u64, |total, tensor| {
                    total
                        .checked_add(tensor.storage_byte_len())
                        .ok_or(NativeVisionModelError::ShapeOverflow)
                })?;
        assert_eq!(model.resident_storage_bytes()?, expected_storage);
        assert!(expected_storage < unaliased_storage);
        assert!(model.resident_bytes()? > expected_storage);
        let parts = model.resident_parts()?;
        assert_eq!(parts.resident_bytes()?, model.resident_bytes()?);
        assert_eq!(parts.tensor_allocations().len(), unique_storage.len());
        assert_eq!(
            parts
                .tensor_allocations()
                .iter()
                .map(|allocation| (allocation.storage_id().get(), allocation.resident_bytes()))
                .collect::<BTreeMap<_, _>>(),
            unique_storage
        );

        let shared_storage = model.clone();
        assert_eq!(
            parts.tensor_allocations(),
            shared_storage.resident_parts()?.tensor_allocations()
        );
        let (distinct_backend, distinct_authority) =
            CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
        let distinct_storage =
            loaded_zero_raft(&distinct_backend, &distinct_authority, &cancellation)?;
        assert_eq!(
            model.semantic_digest_sha256()?,
            distinct_storage.semantic_digest_sha256()?
        );
        let distinct_ids = distinct_storage
            .resident_parts()?
            .tensor_allocations()
            .iter()
            .map(|allocation| allocation.storage_id().get())
            .collect::<BTreeSet<_>>();
        assert!(
            unique_storage
                .keys()
                .copied()
                .collect::<BTreeSet<_>>()
                .is_disjoint(&distinct_ids)
        );

        let overflow = NativeRaftResidentParts {
            owned_bytes: u64::MAX,
            tensor_allocations: vec![NativeRaftTensorResidentAllocation {
                storage_id: parts
                    .tensor_allocations()
                    .first()
                    .ok_or(NativeVisionModelError::ParametersNotLoaded)?
                    .storage_id(),
                resident_bytes: 1,
            }],
        };
        assert!(matches!(
            overflow.resident_bytes(),
            Err(NativeVisionModelError::ShapeOverflow)
        ));

        let mut drifted = model.clone();
        let removed_name = drifted
            .canonical_state
            .keys()
            .next()
            .cloned()
            .ok_or(NativeVisionModelError::ParametersNotLoaded)?;
        if drifted.canonical_state.remove(&removed_name).is_none() {
            return Err(NativeVisionModelError::MissingState(removed_name));
        }
        assert!(matches!(
            drifted.validate(&cancellation),
            Err(NativeVisionModelError::MissingState(name)) if name == removed_name
        ));

        let mut drifted = model.clone();
        drifted.module_state_digest_sha256 = Some("0".repeat(64));
        assert!(matches!(
            drifted.validate(&cancellation),
            Err(NativeVisionModelError::SemanticIdentityChanged)
        ));
        Ok(())
    }

    #[test]
    fn raft_execution_sessions_are_fresh_deterministic_and_do_not_mutate_semantic_state()
    -> Result<(), NativeVisionModelError> {
        let cancellation = CancellationToken::default();
        let (backend, workspace_authority) =
            CpuWorkspaceAuthority::create_backend(64 * 1024 * 1024)?;
        let model = loaded_zero_raft(&backend, &workspace_authority, &cancellation)?;
        let source_digest = model.semantic_digest_sha256()?.to_owned();
        let image = test_image(&backend, &workspace_authority, &cancellation, 64)?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: workspace_authority.authorize_workspace(4 * 1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let mut first = model.execution_session(&cancellation)?;
        let mut second = model.execution_session(&cancellation)?;
        assert_eq!(first.semantic_identity()?, second.semantic_identity()?);
        let first_failure = match first.forward_with_context(&backend, &image, &image, 1, &context)
        {
            Err(error) => error,
            Ok(_) => {
                return Err(NativeVisionModelError::Invalid(
                    "undersized RAFT input was accepted".to_owned(),
                ));
            }
        };
        let second_failure =
            match second.forward_with_context(&backend, &image, &image, 1, &context) {
                Err(error) => error,
                Ok(_) => {
                    return Err(NativeVisionModelError::Invalid(
                        "undersized RAFT input was accepted".to_owned(),
                    ));
                }
            };
        assert_eq!(first_failure.to_string(), second_failure.to_string());
        assert_eq!(first.semantic_identity()?.digest_sha256(), source_digest);
        assert_eq!(second.semantic_identity()?.digest_sha256(), source_digest);
        assert_eq!(model.semantic_digest_sha256()?, source_digest.as_str());
        model.validate(&cancellation)?;

        let mut cancelled_session = model.execution_session(&cancellation)?;
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: workspace_authority.authorize_workspace(4 * 1024 * 1024)?,
            rng_phase: None,
            cancellation: &cancelled,
        };
        assert!(matches!(
            cancelled_session.forward_with_context(&backend, &image, &image, 1, &cancelled_context),
            Err(NativeVisionModelError::Cancelled)
        ));
        assert_eq!(
            cancelled_session.semantic_identity()?.digest_sha256(),
            source_digest
        );
        assert!(matches!(
            model.execution_session(&cancelled),
            Err(NativeVisionModelError::Cancelled)
        ));
        assert_eq!(model.semantic_digest_sha256()?, source_digest.as_str());
        model.validate(&cancellation)?;
        Ok(())
    }

    #[test]
    fn raft_image_preprocessing_maps_unit_interval_to_signed_interval()
    -> Result<(), super::NativeVisionModelError> {
        let cancellation = CancellationToken::default();
        let (backend, workspace_authority) =
            sim_tensor::CpuWorkspaceAuthority::create_backend(1024)?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: workspace_authority.authorize_workspace(1024)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let execution = VisionExecution::canonical(&backend, &context);
        let mut values = execution.values(3)?;
        values.try_push(0.0)?;
        values.try_push(0.5)?;
        values.try_push(1.0)?;
        let normalized = normalize_raft_image(
            NativeValues {
                shape: vec![1, 3, 1, 1],
                values,
            },
            &execution,
        )?;
        assert_eq!(&*normalized.values, &[-1.0, 0.0, 1.0]);
        Ok(())
    }

    #[test]
    fn canonical_vision_staging_uses_exact_caller_workspace_and_converges()
    -> Result<(), NativeVisionModelError> {
        let cancellation = CancellationToken::default();
        let (backend, workspace_authority) =
            sim_tensor::CpuWorkspaceAuthority::create_backend(4096)?;
        let descriptor = TensorDescriptor::contiguous(
            vec![1, 3, 1, 1],
            sim_tensor::DType::F32,
            DeviceId::CPU,
            sim_tensor::StreamId::DEFAULT,
        )?;
        let upload_context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: workspace_authority.authorize_workspace(12)?,
            rng_phase: None,
            cancellation: &cancellation,
        };
        let tensor = backend
            .upload_f32(descriptor, &[0.0, 0.5, 1.0], &upload_context)?
            .0;

        let scratch = workspace_authority.authorize_workspace(12)?;
        let context = sim_tensor::ExecutionContext {
            stream: sim_tensor::StreamId::DEFAULT,
            scratch: scratch.clone(),
            rng_phase: None,
            cancellation: &cancellation,
        };
        let execution = VisionExecution::canonical(&backend, &context);
        let values = NativeValues::from_tensor(&execution, &tensor)?;
        assert_eq!(&*values.values, &[0.0, 0.5, 1.0]);
        assert_eq!(scratch.in_use_bytes(), 12);
        assert_eq!(scratch.peak_bytes(), 12);
        drop(values);
        assert_eq!(scratch.in_use_bytes(), 0);

        let insufficient = workspace_authority.authorize_workspace(11)?;
        let insufficient_context = sim_tensor::ExecutionContext {
            stream: sim_tensor::StreamId::DEFAULT,
            scratch: insufficient.clone(),
            rng_phase: None,
            cancellation: &cancellation,
        };
        let insufficient_execution = VisionExecution::canonical(&backend, &insufficient_context);
        assert!(matches!(
            NativeValues::from_tensor(&insufficient_execution, &tensor),
            Err(NativeVisionModelError::TensorStorage(
                sim_tensor::TensorError::WorkspaceAuthorizationExceeded { .. }
            ))
        ));
        assert_eq!(insufficient.in_use_bytes(), 0);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_scratch = workspace_authority.authorize_workspace(12)?;
        let cancelled_context = sim_tensor::ExecutionContext {
            stream: sim_tensor::StreamId::DEFAULT,
            scratch: cancelled_scratch.clone(),
            rng_phase: None,
            cancellation: &cancelled,
        };
        let cancelled_execution = VisionExecution::canonical(&backend, &cancelled_context);
        assert!(matches!(
            NativeValues::from_tensor(&cancelled_execution, &tensor),
            Err(NativeVisionModelError::Cancelled)
        ));
        assert_eq!(cancelled_scratch.in_use_bytes(), 0);
        Ok(())
    }
}
