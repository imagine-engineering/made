use crate::{ModelFamilyIdentity, NativeVae, PatchGraphIdentity, VaeError};
use sha2::{Digest, Sha256};
use sim_tensor::{
    BinaryOperation, CpuBackend, DType, DeviceId, ExecutionContext, NumericClass, ResizeMode,
    Scalar, ScalarSide, StorageId, Tensor, TensorBackend, TensorDescriptor, TensorError,
    UnaryOperation, binary_broadcast_shape,
    generated_external_tensor_kernel_01::resize_with_context_exact_native,
    generated_indexing_masking_01::narrow_method_exact_native,
    generated_reduction_01::tensor_mean_with_context_exact_native,
    generated_shape_layout_transform_01::{
        tensor_movedim_exact_native, tensor_unsqueeze_exact_native,
    },
    generated_shape_layout_transform_02::{
        tensor_repeat_with_context_exact_native, torch_cat_with_context_exact_native,
    },
    generated_sim_operator_indirection_01::cast_to_with_context_exact_native,
};
use std::{
    collections::{BTreeMap, HashMap},
    mem,
};
use thiserror::Error;

pub const CONTROLNET_SCHEMA_VERSION: u16 = 1;
const MAX_CHAIN_LENGTH: usize = 64;
const MAX_SLOT_LENGTH: usize = 256;
const MAX_EXTRA_CONDITIONING: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StrengthType {
    Constant,
    LinearUp,
}

impl StrengthType {
    fn multiplier(
        self,
        strength: f32,
        length: usize,
        index: usize,
    ) -> Result<f32, ControlNetError> {
        let multiplier = match self {
            Self::Constant => strength,
            Self::LinearUp => {
                let exponent = length.checked_sub(index).ok_or_else(|| {
                    ControlNetError::Invalid("control slot index exceeds its list".into())
                })?;
                strength.powf(exponent as f32)
            }
        };
        if !multiplier.is_finite() {
            return Err(ControlNetError::Invalid(
                "control strength produced a non-finite multiplier".into(),
            ));
        }
        Ok(multiplier)
    }

    fn identity_tag(self) -> &'static [u8] {
        match self {
            Self::Constant => b"constant",
            Self::LinearUp => b"linear-up",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlIsolation {
    CompleteChain,
    CurrentControlOnly,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlPercentWindow {
    start: f32,
    end: f32,
}

impl ControlPercentWindow {
    pub fn checked(start: f32, end: f32) -> Result<Self, ControlNetError> {
        if !start.is_finite()
            || !end.is_finite()
            || !(0.0..=1.0).contains(&start)
            || !(0.0..=1.0).contains(&end)
            || start > end
        {
            return Err(ControlNetError::Invalid(
                "control percent window must be finite, ordered, and within [0, 1]".into(),
            ));
        }
        Ok(Self { start, end })
    }

    pub const fn start(self) -> f32 {
        self.start
    }

    pub const fn end(self) -> f32 {
        self.end
    }

    pub fn contains(self, percent: f32) -> Result<bool, ControlNetError> {
        if !percent.is_finite() || !(0.0..=1.0).contains(&percent) {
            return Err(ControlNetError::Invalid(
                "resolved control percent must be finite and within [0, 1]".into(),
            ));
        }
        Ok(percent >= self.start && percent <= self.end)
    }
}

#[derive(Clone, Debug)]
pub struct ControlBase {
    strength: f32,
    strength_type: StrengthType,
    percent_window: ControlPercentWindow,
    global_average_pooling: bool,
    output_dtype: Option<DType>,
}

impl ControlBase {
    pub fn checked(
        strength: f32,
        strength_type: StrengthType,
        percent_window: ControlPercentWindow,
        global_average_pooling: bool,
        output_dtype: Option<DType>,
    ) -> Result<Self, ControlNetError> {
        if !strength.is_finite() {
            return Err(ControlNetError::Invalid(
                "control strength must be finite".into(),
            ));
        }
        if output_dtype.is_some_and(|dtype| dtype.class() != NumericClass::FloatingPoint) {
            return Err(ControlNetError::Invalid(
                "control output dtype must be floating point".into(),
            ));
        }
        Ok(Self {
            strength,
            strength_type,
            percent_window,
            global_average_pooling,
            output_dtype,
        })
    }

    pub const fn strength(&self) -> f32 {
        self.strength
    }

    pub const fn strength_type(&self) -> StrengthType {
        self.strength_type
    }

    pub const fn percent_window(&self) -> ControlPercentWindow {
        self.percent_window
    }

    pub const fn global_average_pooling(&self) -> bool {
        self.global_average_pooling
    }

    pub const fn output_dtype(&self) -> Option<DType> {
        self.output_dtype
    }
}

#[derive(Clone, Debug)]
pub struct ControlTensorBinding {
    tensor: Tensor,
    content_sha256: String,
}

impl ControlTensorBinding {
    pub fn checked(
        tensor: Tensor,
        content_sha256: impl Into<String>,
    ) -> Result<Self, ControlNetError> {
        let content_sha256 = content_sha256.into();
        validate_sha256("control tensor content", &content_sha256)?;
        Ok(Self {
            tensor,
            content_sha256,
        })
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn content_sha256(&self) -> &str {
        &self.content_sha256
    }
}

#[derive(Clone, Debug)]
pub struct ControlModelBinding {
    model_family: ModelFamilyIdentity,
    patch: PatchGraphIdentity,
    model_state_sha256: String,
    executor_sha256: String,
    dtype: DType,
    device: DeviceId,
}

impl ControlModelBinding {
    pub fn checked(
        model_family: ModelFamilyIdentity,
        patch: PatchGraphIdentity,
        model_state_sha256: impl Into<String>,
        executor_sha256: impl Into<String>,
        dtype: DType,
        device: DeviceId,
    ) -> Result<Self, ControlNetError> {
        let model_state_sha256 = model_state_sha256.into();
        let executor_sha256 = executor_sha256.into();
        validate_sha256("control model state", &model_state_sha256)?;
        validate_sha256("control model executor", &executor_sha256)?;
        patch
            .validate_for_base(&model_state_sha256)
            .map_err(|error| ControlNetError::Invalid(error.to_string()))?;
        if dtype.class() != NumericClass::FloatingPoint {
            return Err(ControlNetError::Invalid(
                "control model execution dtype must be floating point".into(),
            ));
        }
        Ok(Self {
            model_family,
            patch,
            model_state_sha256,
            executor_sha256,
            dtype,
            device,
        })
    }

    pub fn model_family(&self) -> &ModelFamilyIdentity {
        &self.model_family
    }

    pub fn patch(&self) -> &PatchGraphIdentity {
        &self.patch
    }

    pub fn model_state_sha256(&self) -> &str {
        &self.model_state_sha256
    }

    pub fn executor_sha256(&self) -> &str {
        &self.executor_sha256
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub const fn device(&self) -> DeviceId {
        self.device
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlHintPreprocess {
    Identity,
    Sd35Canny,
    Sd35Depth,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlHintLayout {
    Nchw,
    Nhwc,
}

#[derive(Clone, Debug)]
pub struct ControlNetRebindRequest {
    hint: ControlTensorBinding,
    hint_layout: ControlHintLayout,
    strength: f32,
    percent_window: ControlPercentWindow,
    expected_vae_sha256: Option<String>,
    extra_concat: Vec<ControlTensorBinding>,
    concat_mask: bool,
    union_control_type: Option<u32>,
}

impl ControlNetRebindRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn checked(
        hint: ControlTensorBinding,
        hint_layout: ControlHintLayout,
        strength: f32,
        percent_window: ControlPercentWindow,
        expected_vae_sha256: Option<String>,
        extra_concat: Vec<ControlTensorBinding>,
        concat_mask: bool,
        union_control_type: Option<u32>,
    ) -> Result<Self, ControlNetError> {
        if hint.tensor().descriptor().rank() != 4 || hint.tensor().descriptor().shape().contains(&0)
        {
            return Err(ControlNetError::Invalid(
                "control rebind hint must have non-empty rank-four shape".to_owned(),
            ));
        }
        if !strength.is_finite() {
            return Err(ControlNetError::Invalid(
                "control rebind strength must be finite".to_owned(),
            ));
        }
        if let Some(digest) = &expected_vae_sha256 {
            validate_sha256("control VAE execution", digest)?;
        }
        if union_control_type.is_some_and(|value| value > 7) {
            return Err(ControlNetError::Invalid(
                "union ControlNet type must be in the closed 0..=7 source range".to_owned(),
            ));
        }
        if extra_concat.len() > MAX_EXTRA_CONDITIONING {
            return Err(ControlNetError::Invalid(
                "control rebind extra concat exceeds the supported bound".to_owned(),
            ));
        }
        if extra_concat.iter().any(|binding| {
            binding.tensor().descriptor().rank() != 4
                || binding.tensor().descriptor().shape().contains(&0)
        }) {
            return Err(ControlNetError::Invalid(
                "control rebind extra concat tensors must have non-empty rank-four shape"
                    .to_owned(),
            ));
        }
        Ok(Self {
            hint,
            hint_layout,
            strength,
            percent_window,
            expected_vae_sha256,
            extra_concat,
            concat_mask,
            union_control_type,
        })
    }

    pub fn expected_vae_sha256(&self) -> Option<&str> {
        self.expected_vae_sha256.as_deref()
    }

    pub fn hint(&self) -> &ControlTensorBinding {
        &self.hint
    }

    pub const fn hint_layout(&self) -> ControlHintLayout {
        self.hint_layout
    }

    pub const fn strength(&self) -> f32 {
        self.strength
    }

    pub const fn percent_window(&self) -> ControlPercentWindow {
        self.percent_window
    }

    pub fn extra_concat(&self) -> &[ControlTensorBinding] {
        &self.extra_concat
    }

    pub const fn concat_mask(&self) -> bool {
        self.concat_mask
    }

    pub const fn union_control_type(&self) -> Option<u32> {
        self.union_control_type
    }
}

impl ControlHintPreprocess {
    fn identity_tag(self) -> &'static [u8] {
        match self {
            Self::Identity => b"identity",
            Self::Sd35Canny => b"sd35-canny-127.5x-plus-0.5",
            Self::Sd35Depth => b"sd35-depth-one-minus-x",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ControlNet {
    base: ControlBase,
    model: ControlModelBinding,
    hint: ControlTensorBinding,
    compression_ratio: u64,
    resize_mode: ResizeMode,
    preprocess: ControlHintPreprocess,
    expected_vae_sha256: Option<String>,
    extra_concat: Vec<ControlTensorBinding>,
    concat_mask: bool,
    union_control_type: Option<u32>,
    extra_conditioning_names: Vec<String>,
}

impl ControlNet {
    #[allow(clippy::too_many_arguments)]
    pub fn checked(
        base: ControlBase,
        model: ControlModelBinding,
        hint: ControlTensorBinding,
        compression_ratio: u64,
        resize_mode: ResizeMode,
        preprocess: ControlHintPreprocess,
        expected_vae_sha256: Option<String>,
        extra_concat: Vec<ControlTensorBinding>,
        concat_mask: bool,
        extra_conditioning_names: Vec<String>,
    ) -> Result<Self, ControlNetError> {
        Self::checked_with_union_control_type(
            base,
            model,
            hint,
            compression_ratio,
            resize_mode,
            preprocess,
            expected_vae_sha256,
            extra_concat,
            concat_mask,
            None,
            extra_conditioning_names,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn checked_with_union_control_type(
        base: ControlBase,
        model: ControlModelBinding,
        hint: ControlTensorBinding,
        compression_ratio: u64,
        resize_mode: ResizeMode,
        preprocess: ControlHintPreprocess,
        expected_vae_sha256: Option<String>,
        extra_concat: Vec<ControlTensorBinding>,
        concat_mask: bool,
        union_control_type: Option<u32>,
        extra_conditioning_names: Vec<String>,
    ) -> Result<Self, ControlNetError> {
        if compression_ratio == 0 {
            return Err(ControlNetError::Invalid(
                "control compression ratio must be positive".into(),
            ));
        }
        if hint.tensor().descriptor().rank() != 4 || hint.tensor().descriptor().shape().contains(&0)
        {
            return Err(ControlNetError::Invalid(
                "control hint must have non-empty NCHW rank-four shape".into(),
            ));
        }
        if let Some(digest) = &expected_vae_sha256 {
            validate_sha256("control VAE execution", digest)?;
        }
        if union_control_type.is_some_and(|value| value > 7) {
            return Err(ControlNetError::Invalid(
                "union ControlNet type must be in the closed 0..=7 source range".to_owned(),
            ));
        }
        if extra_concat.len() > MAX_EXTRA_CONDITIONING
            || extra_conditioning_names.len() > MAX_EXTRA_CONDITIONING
        {
            return Err(ControlNetError::Invalid(
                "control extra conditioning exceeds the supported bound".into(),
            ));
        }
        validate_unique_names(&extra_conditioning_names)?;
        Ok(Self {
            base,
            model,
            hint,
            compression_ratio,
            resize_mode,
            preprocess,
            expected_vae_sha256,
            extra_concat,
            concat_mask,
            union_control_type,
            extra_conditioning_names,
        })
    }

    pub fn base(&self) -> &ControlBase {
        &self.base
    }

    pub fn model(&self) -> &ControlModelBinding {
        &self.model
    }

    pub fn hint(&self) -> &ControlTensorBinding {
        &self.hint
    }

    pub const fn union_control_type(&self) -> Option<u32> {
        self.union_control_type
    }

    pub fn expected_vae_sha256(&self) -> Option<&str> {
        self.expected_vae_sha256.as_deref()
    }

    pub fn extra_concat(&self) -> &[ControlTensorBinding] {
        &self.extra_concat
    }

    pub const fn concat_mask(&self) -> bool {
        self.concat_mask
    }

    pub fn rebound(
        &self,
        request: &ControlNetRebindRequest,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, ControlNetError> {
        cancellation
            .check()
            .map_err(|_| ControlNetError::Cancelled)?;
        let hint = match request.hint_layout {
            ControlHintLayout::Nchw => request.hint.clone(),
            ControlHintLayout::Nhwc => {
                let tensor =
                    tensor_movedim_exact_native(request.hint.tensor(), &[-1], &[1], cancellation)
                        .map_err(|error| ControlNetError::CanonicalTensor(Box::new(error)))?;
                ControlTensorBinding::checked(tensor, request.hint.content_sha256.clone())?
            }
        };
        let base = ControlBase::checked(
            request.strength,
            self.base.strength_type,
            request.percent_window,
            self.base.global_average_pooling,
            self.base.output_dtype,
        )?;
        Self::checked_with_union_control_type(
            base,
            self.model.clone(),
            hint,
            self.compression_ratio,
            self.resize_mode,
            self.preprocess,
            request.expected_vae_sha256.clone(),
            request.extra_concat.clone(),
            request.concat_mask,
            request.union_control_type,
            self.extra_conditioning_names.clone(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct ControlLoraOps {
    patch: PatchGraphIdentity,
}

impl ControlLoraOps {
    pub fn checked(
        patch: PatchGraphIdentity,
        model_state_sha256: &str,
    ) -> Result<Self, ControlNetError> {
        patch
            .validate_for_base(model_state_sha256)
            .map_err(|error| ControlNetError::Invalid(error.to_string()))?;
        Ok(Self { patch })
    }

    pub fn patch(&self) -> &PatchGraphIdentity {
        &self.patch
    }
}

#[derive(Clone, Debug)]
pub struct ControlLora {
    control: ControlNet,
    operations: ControlLoraOps,
}

impl ControlLora {
    pub fn checked(
        control: ControlNet,
        operations: ControlLoraOps,
    ) -> Result<Self, ControlNetError> {
        if control.model.patch != operations.patch {
            return Err(ControlNetError::Invalid(
                "ControlLoRA operation graph does not match its loaded model binding".into(),
            ));
        }
        Ok(Self {
            control,
            operations,
        })
    }

    pub fn control(&self) -> &ControlNet {
        &self.control
    }

    pub fn operations(&self) -> &ControlLoraOps {
        &self.operations
    }
}

#[derive(Clone, Debug)]
pub struct ControlNetSD35 {
    control: ControlNet,
}

impl ControlNetSD35 {
    pub fn checked(control: ControlNet) -> Result<Self, ControlNetError> {
        if !matches!(
            control.preprocess,
            ControlHintPreprocess::Sd35Canny | ControlHintPreprocess::Sd35Depth
        ) {
            return Err(ControlNetError::Invalid(
                "SD3.5 ControlNet requires canny or depth preprocessing".into(),
            ));
        }
        Ok(Self { control })
    }

    pub fn control(&self) -> &ControlNet {
        &self.control
    }
}

#[derive(Clone, Debug)]
pub struct T2IAdapter {
    base: ControlBase,
    model: ControlModelBinding,
    hint: ControlTensorBinding,
    channels_in: u64,
    compression_ratio: u64,
    unshuffle_amount: u64,
    resize_mode: ResizeMode,
}

impl T2IAdapter {
    pub fn checked(
        base: ControlBase,
        model: ControlModelBinding,
        hint: ControlTensorBinding,
        channels_in: u64,
        compression_ratio: u64,
        unshuffle_amount: u64,
        resize_mode: ResizeMode,
    ) -> Result<Self, ControlNetError> {
        if channels_in == 0 || compression_ratio == 0 || unshuffle_amount == 0 {
            return Err(ControlNetError::Invalid(
                "T2I channel, compression, and unshuffle values must be positive".into(),
            ));
        }
        if hint.tensor().descriptor().rank() != 4 {
            return Err(ControlNetError::Invalid(
                "T2I hint must have NCHW rank four".into(),
            ));
        }
        Ok(Self {
            base,
            model,
            hint,
            channels_in,
            compression_ratio,
            unshuffle_amount,
            resize_mode,
        })
    }
}

#[derive(Clone, Debug)]
pub enum ControlNode {
    ControlNet(ControlNet),
    ControlLora(ControlLora),
    ControlNetSD35(ControlNetSD35),
    T2IAdapter(T2IAdapter),
}

impl ControlNode {
    fn base(&self) -> &ControlBase {
        match self {
            Self::ControlNet(control) => &control.base,
            Self::ControlLora(control) => &control.control.base,
            Self::ControlNetSD35(control) => &control.control.base,
            Self::T2IAdapter(adapter) => &adapter.base,
        }
    }

    fn rebound(
        &self,
        request: &ControlNetRebindRequest,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, ControlNetError> {
        match self {
            Self::ControlNet(control) => {
                Ok(Self::ControlNet(control.rebound(request, cancellation)?))
            }
            Self::ControlLora(control) => Ok(Self::ControlLora(ControlLora::checked(
                control.control.rebound(request, cancellation)?,
                control.operations.clone(),
            )?)),
            Self::ControlNetSD35(control) => Ok(Self::ControlNetSD35(ControlNetSD35::checked(
                control.control.rebound(request, cancellation)?,
            )?)),
            Self::T2IAdapter(_) => Err(ControlNetError::Invalid(
                "T2I adapters do not support ControlNet rebinding".to_owned(),
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ControlChain {
    nodes: Vec<ControlNode>,
    identity: ControlExecutionIdentity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ControlChainTensorResidentAllocation {
    storage_id: StorageId,
    resident_bytes: u64,
}

impl ControlChainTensorResidentAllocation {
    pub const fn storage_id(&self) -> StorageId {
        self.storage_id
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ControlChainResidentParts {
    owned_bytes: u64,
    tensor_allocations: Vec<ControlChainTensorResidentAllocation>,
}

impl ControlChainResidentParts {
    pub const fn owned_bytes(&self) -> u64 {
        self.owned_bytes
    }

    pub fn tensor_allocations(&self) -> &[ControlChainTensorResidentAllocation] {
        &self.tensor_allocations
    }

    pub fn resident_bytes(&self) -> Result<u64, ControlNetError> {
        self.tensor_allocations
            .iter()
            .try_fold(self.owned_bytes, |bytes, allocation| {
                bytes
                    .checked_add(allocation.resident_bytes)
                    .ok_or(ControlNetError::ResidentBytesOverflow)
            })
    }
}

impl ControlChain {
    pub fn checked(nodes: Vec<ControlNode>) -> Result<Self, ControlNetError> {
        if nodes.is_empty() || nodes.len() > MAX_CHAIN_LENGTH {
            return Err(ControlNetError::Invalid(format!(
                "control chain length must be between 1 and {MAX_CHAIN_LENGTH}"
            )));
        }
        let identity = compute_chain_identity(&nodes)?;
        Ok(Self { nodes, identity })
    }

    pub fn nodes(&self) -> &[ControlNode] {
        &self.nodes
    }

    pub fn identity(&self) -> &ControlExecutionIdentity {
        &self.identity
    }

    pub fn rebind_current(
        &self,
        request: &ControlNetRebindRequest,
        predecessor: Option<&ControlChain>,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, ControlNetError> {
        cancellation
            .check()
            .map_err(|_| ControlNetError::Cancelled)?;
        let current = self.nodes.last().ok_or_else(|| {
            ControlNetError::Invalid("control rebind source chain is empty".to_owned())
        })?;
        let predecessor_length = predecessor.map_or(0, |chain| chain.nodes.len());
        if predecessor_length >= MAX_CHAIN_LENGTH {
            return Err(ControlNetError::Invalid(
                "control predecessor chain leaves no room for the rebound control".to_owned(),
            ));
        }
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(predecessor_length + 1)
            .map_err(|_| {
                ControlNetError::Invalid("control chain allocation overflow".to_owned())
            })?;
        if let Some(predecessor) = predecessor {
            nodes.extend(predecessor.nodes.iter().cloned());
        }
        nodes.push(current.rebound(request, cancellation)?);
        cancellation
            .check()
            .map_err(|_| ControlNetError::Cancelled)?;
        Self::checked(nodes)
    }

    pub fn resident_bytes(&self) -> Result<u64, ControlNetError> {
        self.resident_parts()?.resident_bytes()
    }

    pub fn resident_parts(&self) -> Result<ControlChainResidentParts, ControlNetError> {
        let mut accounting = ControlChainResidentAccounting::default();
        accounting.add_owned_bytes(mem::size_of::<Self>())?;
        accounting.add_owned_bytes(
            self.nodes
                .capacity()
                .checked_mul(mem::size_of::<ControlNode>())
                .ok_or(ControlNetError::ResidentBytesOverflow)?,
        )?;
        accounting.add_owned_bytes(self.identity.0.capacity())?;
        for node in &self.nodes {
            control_node_resident_parts(node, &mut accounting)?;
        }
        accounting.into_parts()
    }

    pub fn require_executor_digest(&self, digest: &str) -> Result<(), ControlNetError> {
        validate_sha256("control executor implementation", digest)?;
        for node in &self.nodes {
            let binding = match node {
                ControlNode::ControlNet(control) => &control.model,
                ControlNode::ControlLora(control) => &control.control.model,
                ControlNode::ControlNetSD35(control) => &control.control.model,
                ControlNode::T2IAdapter(adapter) => &adapter.model,
            };
            if binding.executor_sha256() != digest {
                return Err(ControlNetError::Invalid(
                    "control executor identity does not match its model binding".to_owned(),
                ));
            }
        }
        Ok(())
    }

    pub fn require_vae_execution_digest(
        &self,
        digest: Option<&str>,
    ) -> Result<(), ControlNetError> {
        if let Some(digest) = digest {
            validate_sha256("control VAE execution", digest)?;
        }
        for node in &self.nodes {
            let expected = match node {
                ControlNode::ControlNet(control) => control.expected_vae_sha256.as_deref(),
                ControlNode::ControlLora(control) => control.control.expected_vae_sha256.as_deref(),
                ControlNode::ControlNetSD35(control) => {
                    control.control.expected_vae_sha256.as_deref()
                }
                ControlNode::T2IAdapter(_) => None,
            };
            if expected != digest {
                return Err(ControlNetError::Invalid(
                    "control VAE execution identity does not match every chain node".to_owned(),
                ));
            }
        }
        Ok(())
    }

    pub fn execution_identity(
        &self,
        conditioning: &ControlConditioning,
    ) -> Result<ControlExecutionIdentity, ControlNetError> {
        let mut hasher = Sha256::new();
        hash_field(&mut hasher, b"zed-controlnet-execution-v1");
        hash_field(&mut hasher, self.identity.digest().as_bytes());
        hash_tensor_binding(&mut hasher, &conditioning.noisy);
        hash_tensor_binding(&mut hasher, &conditioning.timestep);
        hash_tensor_binding(&mut hasher, &conditioning.cross_attention);
        hash_optional_tensor_binding(&mut hasher, conditioning.control_cross_attention.as_ref());
        hash_tensor_bindings(&mut hasher, &conditioning.concat)?;
        hash_optional_tensor_binding(&mut hasher, conditioning.mask.as_ref());
        hash_u64(
            &mut hasher,
            u64::try_from(conditioning.extra.len()).map_err(|_| {
                ControlNetError::Invalid("extra conditioning length overflow".into())
            })?,
        );
        for (name, binding) in &conditioning.extra {
            hash_field(&mut hasher, name.as_bytes());
            hash_tensor_binding(&mut hasher, binding);
        }
        hash_field(
            &mut hasher,
            &conditioning.resolved_percent.to_bits().to_le_bytes(),
        );
        hash_u64(&mut hasher, conditioning.batched_number);
        Ok(ControlExecutionIdentity(format!("{:x}", hasher.finalize())))
    }
}

#[derive(Default)]
struct ControlChainResidentAccounting {
    owned_bytes: u64,
    tensor_allocations: BTreeMap<u64, ControlChainTensorResidentAllocation>,
}

impl ControlChainResidentAccounting {
    fn add_owned_bytes(&mut self, bytes: usize) -> Result<(), ControlNetError> {
        self.owned_bytes = self
            .owned_bytes
            .checked_add(u64::try_from(bytes).map_err(|_| ControlNetError::ResidentBytesOverflow)?)
            .ok_or(ControlNetError::ResidentBytesOverflow)?;
        Ok(())
    }

    fn add_owned_u64(&mut self, bytes: u64) -> Result<(), ControlNetError> {
        self.owned_bytes = self
            .owned_bytes
            .checked_add(bytes)
            .ok_or(ControlNetError::ResidentBytesOverflow)?;
        Ok(())
    }

    fn add_tensor_binding(
        &mut self,
        binding: &ControlTensorBinding,
    ) -> Result<(), ControlNetError> {
        self.add_owned_bytes(binding.content_sha256.capacity())?;
        let allocation = ControlChainTensorResidentAllocation {
            storage_id: binding.tensor.storage_id(),
            resident_bytes: binding.tensor.storage_byte_len(),
        };
        if let Some(existing) = self
            .tensor_allocations
            .insert(allocation.storage_id.get(), allocation)
            && existing.resident_bytes != allocation.resident_bytes
        {
            return Err(ControlNetError::ResidentBytesOverflow);
        }
        Ok(())
    }

    fn into_parts(self) -> Result<ControlChainResidentParts, ControlNetError> {
        let parts = ControlChainResidentParts {
            owned_bytes: self.owned_bytes,
            tensor_allocations: self.tensor_allocations.into_values().collect(),
        };
        parts.resident_bytes()?;
        Ok(parts)
    }
}

fn add_patch_identity_resident_bytes(
    identity: &PatchGraphIdentity,
    accounting: &mut ControlChainResidentAccounting,
) -> Result<(), ControlNetError> {
    accounting.add_owned_bytes(identity.base_artifact_digest.capacity())?;
    accounting.add_owned_bytes(identity.ordered_digest.capacity())
}

fn add_model_binding_resident_bytes(
    binding: &ControlModelBinding,
    accounting: &mut ControlChainResidentAccounting,
) -> Result<(), ControlNetError> {
    accounting.add_owned_u64(
        binding
            .model_family
            .owned_resident_bytes()
            .ok_or(ControlNetError::ResidentBytesOverflow)?,
    )?;
    add_patch_identity_resident_bytes(&binding.patch, accounting)?;
    accounting.add_owned_bytes(binding.model_state_sha256.capacity())?;
    accounting.add_owned_bytes(binding.executor_sha256.capacity())
}

fn control_net_resident_parts(
    control: &ControlNet,
    accounting: &mut ControlChainResidentAccounting,
) -> Result<(), ControlNetError> {
    add_model_binding_resident_bytes(&control.model, accounting)?;
    accounting.add_tensor_binding(&control.hint)?;
    if let Some(digest) = &control.expected_vae_sha256 {
        accounting.add_owned_bytes(digest.capacity())?;
    }
    accounting.add_owned_bytes(
        control
            .extra_concat
            .capacity()
            .checked_mul(mem::size_of::<ControlTensorBinding>())
            .ok_or(ControlNetError::ResidentBytesOverflow)?,
    )?;
    for binding in &control.extra_concat {
        accounting.add_tensor_binding(binding)?;
    }
    accounting.add_owned_bytes(
        control
            .extra_conditioning_names
            .capacity()
            .checked_mul(mem::size_of::<String>())
            .ok_or(ControlNetError::ResidentBytesOverflow)?,
    )?;
    for name in &control.extra_conditioning_names {
        accounting.add_owned_bytes(name.capacity())?;
    }
    Ok(())
}

fn control_node_resident_parts(
    node: &ControlNode,
    accounting: &mut ControlChainResidentAccounting,
) -> Result<(), ControlNetError> {
    match node {
        ControlNode::ControlNet(control) => control_net_resident_parts(control, accounting),
        ControlNode::ControlLora(control) => {
            control_net_resident_parts(&control.control, accounting)?;
            add_patch_identity_resident_bytes(&control.operations.patch, accounting)
        }
        ControlNode::ControlNetSD35(control) => {
            control_net_resident_parts(&control.control, accounting)
        }
        ControlNode::T2IAdapter(adapter) => {
            add_model_binding_resident_bytes(&adapter.model, accounting)?;
            accounting.add_tensor_binding(&adapter.hint)
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ControlExecutionIdentity(String);

impl ControlExecutionIdentity {
    pub fn digest(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub struct ControlConditioning {
    noisy: ControlTensorBinding,
    timestep: ControlTensorBinding,
    cross_attention: ControlTensorBinding,
    control_cross_attention: Option<ControlTensorBinding>,
    concat: Vec<ControlTensorBinding>,
    mask: Option<ControlTensorBinding>,
    extra: BTreeMap<String, ControlTensorBinding>,
    resolved_percent: f32,
    batched_number: u64,
}

impl ControlConditioning {
    #[allow(clippy::too_many_arguments)]
    pub fn checked(
        noisy: ControlTensorBinding,
        timestep: ControlTensorBinding,
        cross_attention: ControlTensorBinding,
        control_cross_attention: Option<ControlTensorBinding>,
        concat: Vec<ControlTensorBinding>,
        mask: Option<ControlTensorBinding>,
        extra: BTreeMap<String, ControlTensorBinding>,
        resolved_percent: f32,
        batched_number: u64,
    ) -> Result<Self, ControlNetError> {
        let noisy_shape = noisy.tensor().descriptor().shape();
        if noisy_shape.len() != 4 || noisy_shape.contains(&0) {
            return Err(ControlNetError::Invalid(
                "control noisy input must have non-empty NCHW shape".into(),
            ));
        }
        if !resolved_percent.is_finite() || !(0.0..=1.0).contains(&resolved_percent) {
            return Err(ControlNetError::Invalid(
                "resolved control percent must be finite and within [0, 1]".into(),
            ));
        }
        let target_batch = noisy_shape[0];
        if batched_number == 0
            || batched_number > target_batch
            || !target_batch.is_multiple_of(batched_number)
        {
            return Err(ControlNetError::Invalid(
                "batched number must be positive and divide the noisy batch".into(),
            ));
        }
        if extra.len() > MAX_EXTRA_CONDITIONING || concat.len() > MAX_EXTRA_CONDITIONING {
            return Err(ControlNetError::Invalid(
                "control conditioning exceeds the supported bound".into(),
            ));
        }
        Ok(Self {
            noisy,
            timestep,
            cross_attention,
            control_cross_attention,
            concat,
            mask,
            extra,
            resolved_percent,
            batched_number,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ControlModelInput {
    pub noisy: Tensor,
    pub timestep: Tensor,
    pub hint: Tensor,
    pub context: Tensor,
    pub concat: Vec<Tensor>,
    pub extra: BTreeMap<String, Tensor>,
}

#[derive(Clone, Debug, Default)]
pub struct ControlResult {
    input: Vec<Option<Tensor>>,
    middle: Vec<Option<Tensor>>,
    output: Vec<Option<Tensor>>,
}

impl ControlResult {
    pub fn checked(
        input: Vec<Option<Tensor>>,
        middle: Vec<Option<Tensor>>,
        output: Vec<Option<Tensor>>,
    ) -> Result<Self, ControlNetError> {
        if input.len() > MAX_SLOT_LENGTH
            || middle.len() > MAX_SLOT_LENGTH
            || output.len() > MAX_SLOT_LENGTH
        {
            return Err(ControlNetError::Invalid(
                "control result exceeds the fixed-slot bound".into(),
            ));
        }
        Ok(Self {
            input,
            middle,
            output,
        })
    }

    pub fn input(&self) -> &[Option<Tensor>] {
        &self.input
    }

    pub fn middle(&self) -> &[Option<Tensor>] {
        &self.middle
    }

    pub fn output(&self) -> &[Option<Tensor>] {
        &self.output
    }
}

pub trait ControlModelExecutor: Send + Sync {
    fn execution_digest(&self) -> &str;

    fn resident_bytes(&self) -> Result<u64, ControlNetError>;

    fn execute_controlnet(
        &self,
        binding: &ControlModelBinding,
        input: &ControlModelInput,
        context: &ExecutionContext<'_>,
    ) -> Result<ControlResult, ControlNetError>;

    fn execute_t2i_adapter(
        &self,
        binding: &ControlModelBinding,
        hint: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<ControlResult, ControlNetError>;
}

#[derive(Debug, Error)]
pub enum ControlNetError {
    #[error("control execution was cancelled")]
    Cancelled,
    #[error("invalid control graph: {0}")]
    Invalid(String),
    #[error("control graph resident byte accounting overflowed")]
    ResidentBytesOverflow,
    #[error(transparent)]
    Tensor(TensorError),
    #[error("canonical tensor operation failed: {0}")]
    CanonicalTensor(#[source] Box<dyn std::error::Error + Send + Sync + 'static>),
    #[error(transparent)]
    Vae(#[from] VaeError),
    #[error(
        "control execution leaked workspace: started at {before} bytes and ended at {after} bytes"
    )]
    WorkspaceLeak { before: u64, after: u64 },
}

pub struct ControlRuntime<'a> {
    backend: &'a CpuBackend,
    executor: &'a dyn ControlModelExecutor,
}

impl<'a> ControlRuntime<'a> {
    pub const fn new(backend: &'a CpuBackend, executor: &'a dyn ControlModelExecutor) -> Self {
        Self { backend, executor }
    }

    pub fn execute(
        &self,
        chain: &ControlChain,
        isolation: ControlIsolation,
        conditioning: &ControlConditioning,
        vae: Option<&NativeVae>,
        context: &ExecutionContext<'_>,
    ) -> Result<Option<ControlResult>, ControlNetError> {
        check_context(context)?;
        let vae_execution_digest = vae.map(NativeVae::execution_digest);
        chain.require_vae_execution_digest(vae_execution_digest.as_deref())?;
        let workspace_before = context.scratch.in_use_bytes();
        let result = self.execute_atomic(chain, isolation, conditioning, vae, context);
        let workspace_after = context.scratch.in_use_bytes();
        if workspace_before != workspace_after {
            return Err(ControlNetError::WorkspaceLeak {
                before: workspace_before,
                after: workspace_after,
            });
        }
        result
    }

    fn execute_atomic(
        &self,
        chain: &ControlChain,
        isolation: ControlIsolation,
        conditioning: &ControlConditioning,
        vae: Option<&NativeVae>,
        context: &ExecutionContext<'_>,
    ) -> Result<Option<ControlResult>, ControlNetError> {
        check_context(context)?;
        let nodes = match isolation {
            ControlIsolation::CompleteChain => chain.nodes(),
            ControlIsolation::CurrentControlOnly => {
                let (current, _) = chain.nodes().split_last().ok_or_else(|| {
                    ControlNetError::Invalid("control chain has no current node".into())
                })?;
                std::slice::from_ref(current)
            }
        };
        let mut merged = None;
        for node in nodes {
            check_context(context)?;
            if !node
                .base()
                .percent_window
                .contains(conditioning.resolved_percent)?
            {
                continue;
            }
            let raw = match node {
                ControlNode::ControlNet(control) => {
                    self.execute_controlnet(control, conditioning, vae, context)?
                }
                ControlNode::ControlLora(control) => {
                    self.execute_controlnet(&control.control, conditioning, vae, context)?
                }
                ControlNode::ControlNetSD35(control) => {
                    self.execute_controlnet(&control.control, conditioning, vae, context)?
                }
                ControlNode::T2IAdapter(adapter) => {
                    self.execute_t2i(adapter, conditioning, context)?
                }
            };
            let strengthened = transform_result(self.backend, raw, node.base(), context)?;
            merged = Some(match merged {
                Some(previous) => merge_results(self.backend, strengthened, previous, context)?,
                None => strengthened,
            });
        }
        check_context(context)?;
        Ok(merged)
    }

    fn execute_controlnet(
        &self,
        control: &ControlNet,
        conditioning: &ControlConditioning,
        vae: Option<&NativeVae>,
        context: &ExecutionContext<'_>,
    ) -> Result<ControlResult, ControlNetError> {
        validate_backend_binding(self.backend, &control.model, conditioning)?;
        let hint = prepare_control_hint(self.backend, control, conditioning, vae, context)?;
        let noisy = cast_tensor(
            self.backend,
            conditioning.noisy.tensor(),
            control.model.dtype,
            control.model.device,
            context,
        )?;
        let timestep = cast_tensor(
            self.backend,
            conditioning.timestep.tensor(),
            control.model.dtype,
            control.model.device,
            context,
        )?;
        let attention = conditioning
            .control_cross_attention
            .as_ref()
            .unwrap_or(&conditioning.cross_attention);
        let attention = cast_tensor(
            self.backend,
            attention.tensor(),
            control.model.dtype,
            control.model.device,
            context,
        )?;
        let mut concat = Vec::with_capacity(conditioning.concat.len());
        for tensor in &conditioning.concat {
            concat.push(cast_tensor(
                self.backend,
                tensor.tensor(),
                control.model.dtype,
                control.model.device,
                context,
            )?);
        }
        let mut extra = BTreeMap::new();
        for name in &control.extra_conditioning_names {
            if let Some(tensor) = conditioning.extra.get(name) {
                extra.insert(
                    name.clone(),
                    cast_tensor(
                        self.backend,
                        tensor.tensor(),
                        control.model.dtype,
                        control.model.device,
                        context,
                    )?,
                );
            }
        }
        let input = ControlModelInput {
            noisy,
            timestep,
            hint,
            context: attention,
            concat,
            extra,
        };
        self.executor
            .execute_controlnet(&control.model, &input, context)
    }

    fn execute_t2i(
        &self,
        adapter: &T2IAdapter,
        conditioning: &ControlConditioning,
        context: &ExecutionContext<'_>,
    ) -> Result<ControlResult, ControlNetError> {
        validate_backend_binding(self.backend, &adapter.model, conditioning)?;
        let noisy_shape = conditioning.noisy.tensor().descriptor().shape();
        let target_height = rounded_extent(
            noisy_shape[2]
                .checked_mul(adapter.compression_ratio)
                .ok_or_else(|| ControlNetError::Invalid("T2I height overflow".into()))?,
            adapter.unshuffle_amount,
        )?;
        let target_width = rounded_extent(
            noisy_shape[3]
                .checked_mul(adapter.compression_ratio)
                .ok_or_else(|| ControlNetError::Invalid("T2I width overflow".into()))?,
            adapter.unshuffle_amount,
        )?;
        let mut hint = resize_hint(
            self.backend,
            adapter.hint.tensor(),
            target_height,
            target_width,
            adapter.resize_mode,
            context,
        )?;
        hint = cast_tensor(
            self.backend,
            &hint,
            DType::F32,
            adapter.model.device,
            context,
        )?;
        if adapter.channels_in == 1 && hint.descriptor().shape()[1] > 1 {
            hint = tensor_mean_with_context_exact_native(
                self.backend,
                &hint,
                Some(&[1]),
                true,
                None,
                context,
            )
            .map_err(|error| canonical_error(context, error))?;
        }
        hint = broadcast_image_to(
            self.backend,
            &hint,
            noisy_shape[0],
            conditioning.batched_number,
            context,
        )?;
        hint = cast_tensor(
            self.backend,
            &hint,
            adapter.model.dtype,
            adapter.model.device,
            context,
        )?;
        self.executor
            .execute_t2i_adapter(&adapter.model, &hint, context)
    }
}

fn prepare_control_hint(
    backend: &CpuBackend,
    control: &ControlNet,
    conditioning: &ControlConditioning,
    vae: Option<&NativeVae>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ControlNetError> {
    let noisy_shape = conditioning.noisy.tensor().descriptor().shape();
    let vae_ratio = match (&control.expected_vae_sha256, vae) {
        (Some(_), Some(vae)) => vae.descriptor().latent_format().spatial_downscale_ratio,
        (Some(_), None) => {
            return Err(ControlNetError::Invalid(
                "validated control execution lost its bound NativeVae".into(),
            ));
        }
        (None, Some(_)) => {
            return Err(ControlNetError::Invalid(
                "validated unbound control execution gained a NativeVae".into(),
            ));
        }
        (None, None) => 1,
    };
    let spatial_ratio = control
        .compression_ratio
        .checked_mul(vae_ratio)
        .ok_or_else(|| ControlNetError::Invalid("control hint ratio overflow".into()))?;
    let target_height = noisy_shape[2]
        .checked_mul(spatial_ratio)
        .ok_or_else(|| ControlNetError::Invalid("control hint height overflow".into()))?;
    let target_width = noisy_shape[3]
        .checked_mul(spatial_ratio)
        .ok_or_else(|| ControlNetError::Invalid("control hint width overflow".into()))?;
    let mut hint = resize_hint(
        backend,
        control.hint.tensor(),
        target_height,
        target_width,
        control.resize_mode,
        context,
    )?;
    hint = preprocess_hint(backend, &hint, control.preprocess, context)?;
    if let Some(vae) = vae {
        hint = vae.encode(backend, &hint, context)?;
    }
    let hint_shape = hint.descriptor().shape().to_vec();
    let mut extra_concat = control.extra_concat.clone();
    if control.concat_mask {
        let mask = conditioning.mask.clone().ok_or_else(|| {
            ControlNetError::Invalid("control graph requires a concatenated mask".into())
        })?;
        extra_concat.push(mask);
    }
    if !extra_concat.is_empty() {
        let mut tensors = Vec::with_capacity(extra_concat.len() + 1);
        tensors.push(hint);
        for extra in extra_concat {
            let mut tensor = extra.tensor().clone();
            while tensor.descriptor().rank() < hint_shape.len() {
                tensor = tensor_unsqueeze_exact_native(&tensor, 2, context.cancellation)
                    .map_err(|error| canonical_error(context, error))?;
            }
            if tensor.descriptor().rank() != hint_shape.len() {
                return Err(ControlNetError::Invalid(
                    "extra control concatenation rank exceeds hint rank".into(),
                ));
            }
            tensor = resize_hint(
                backend,
                &tensor,
                hint_shape[hint_shape.len() - 2],
                hint_shape[hint_shape.len() - 1],
                control.resize_mode,
                context,
            )?;
            tensor = broadcast_image_to(backend, &tensor, hint_shape[0], 1, context)?;
            tensors.push(tensor);
        }
        hint = torch_cat_with_context_exact_native(backend, &tensors, 1, context)
            .map_err(|error| canonical_error(context, error))?;
    }
    hint = cast_tensor(
        backend,
        &hint,
        control.model.dtype,
        control.model.device,
        context,
    )?;
    broadcast_image_to(
        backend,
        &hint,
        noisy_shape[0],
        conditioning.batched_number,
        context,
    )
}

fn resize_hint(
    backend: &CpuBackend,
    input: &Tensor,
    height: u64,
    width: u64,
    mode: ResizeMode,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ControlNetError> {
    if input.descriptor().rank() != 4 || height == 0 || width == 0 {
        return Err(ControlNetError::Invalid(
            "hint resize requires NCHW rank four and non-zero output dimensions".into(),
        ));
    }
    resize_with_context_exact_native(backend, input, height, width, mode, false, context)
        .map_err(|error| canonical_error(context, error))
}

fn broadcast_image_to(
    backend: &CpuBackend,
    input: &Tensor,
    target_batch_size: u64,
    batched_number: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ControlNetError> {
    let current_batch_size =
        *input.descriptor().shape().first().ok_or_else(|| {
            ControlNetError::Invalid("broadcast input must have a batch axis".into())
        })?;
    if current_batch_size == 0 || target_batch_size == 0 || batched_number == 0 {
        return Err(ControlNetError::Invalid(
            "broadcast batch values must be positive".into(),
        ));
    }
    if current_batch_size == 1 {
        return Ok(input.clone());
    }
    if batched_number > target_batch_size || !target_batch_size.is_multiple_of(batched_number) {
        return Err(ControlNetError::Invalid(
            "batched number must divide the target batch".into(),
        ));
    }
    let per_batch = target_batch_size / batched_number;
    let cropped_batch = current_batch_size.min(per_batch);
    let mut tensor = narrow_method_exact_native(input, 0, 0, cropped_batch, context.cancellation)
        .map_err(|error| canonical_error(context, error))?;
    if per_batch > cropped_batch {
        let quotient = per_batch / cropped_batch;
        let remainder = per_batch % cropped_batch;
        let capacity = usize::try_from(quotient)
            .ok()
            .and_then(|value| value.checked_add(usize::from(remainder > 0)))
            .ok_or_else(|| ControlNetError::Invalid("broadcast list length overflow".into()))?;
        let mut repeated = Vec::with_capacity(capacity);
        for _ in 0..quotient {
            repeated.push(tensor.clone());
        }
        if remainder > 0 {
            repeated.push(
                narrow_method_exact_native(&tensor, 0, 0, remainder, context.cancellation)
                    .map_err(|error| canonical_error(context, error))?,
            );
        }
        tensor = torch_cat_with_context_exact_native(backend, &repeated, 0, context)
            .map_err(|error| canonical_error(context, error))?;
    }
    if tensor.descriptor().shape()[0] == target_batch_size {
        return Ok(tensor);
    }
    let repeated = (0..batched_number)
        .map(|_| tensor.clone())
        .collect::<Vec<_>>();
    torch_cat_with_context_exact_native(backend, &repeated, 0, context)
        .map_err(|error| canonical_error(context, error))
}

fn preprocess_hint(
    backend: &CpuBackend,
    input: &Tensor,
    preprocess: ControlHintPreprocess,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ControlNetError> {
    match preprocess {
        ControlHintPreprocess::Identity => Ok(input.clone()),
        ControlHintPreprocess::Sd35Canny => {
            let scaled = binary_scalar(
                backend,
                BinaryOperation::Multiply,
                input,
                Scalar::Float(127.5),
                context,
            )?;
            binary_scalar(
                backend,
                BinaryOperation::Add,
                &scaled,
                Scalar::Float(0.5),
                context,
            )
        }
        ControlHintPreprocess::Sd35Depth => {
            let descriptor = contiguous_descriptor(
                input.descriptor().shape(),
                input.descriptor().dtype(),
                input,
            )?;
            backend
                .unary(
                    UnaryOperation::InvertUnitInterval,
                    input,
                    descriptor,
                    context,
                )
                .map(|(tensor, _)| tensor)
                .map_err(ControlNetError::from)
        }
    }
}

fn transform_result(
    backend: &CpuBackend,
    result: ControlResult,
    base: &ControlBase,
    context: &ExecutionContext<'_>,
) -> Result<ControlResult, ControlNetError> {
    Ok(ControlResult {
        input: transform_slots(backend, result.input, base, context)?,
        middle: transform_slots(backend, result.middle, base, context)?,
        output: transform_slots(backend, result.output, base, context)?,
    })
}

fn transform_slots(
    backend: &CpuBackend,
    slots: Vec<Option<Tensor>>,
    base: &ControlBase,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Option<Tensor>>, ControlNetError> {
    let length = slots.len();
    let mut transformed_by_tensor: HashMap<u64, Tensor> = HashMap::new();
    let mut output = Vec::with_capacity(length);
    for (index, tensor) in slots.into_iter().enumerate() {
        let Some(tensor) = tensor else {
            output.push(None);
            continue;
        };
        if let Some(transformed) = transformed_by_tensor.get(&tensor.tensor_id().get()) {
            output.push(Some(transformed.clone()));
            continue;
        }
        let mut transformed = if base.global_average_pooling {
            global_average_pool(backend, &tensor, context)?
        } else {
            tensor.clone()
        };
        let multiplier = base
            .strength_type
            .multiplier(base.strength, length, index)?;
        transformed = binary_scalar(
            backend,
            BinaryOperation::Multiply,
            &transformed,
            Scalar::Float(f64::from(multiplier)),
            context,
        )?;
        if let Some(dtype) = base.output_dtype {
            transformed = cast_tensor(
                backend,
                &transformed,
                dtype,
                transformed.descriptor().device(),
                context,
            )?;
        }
        transformed_by_tensor.insert(tensor.tensor_id().get(), transformed.clone());
        output.push(Some(transformed));
    }
    Ok(output)
}

fn global_average_pool(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ControlNetError> {
    let shape = input.descriptor().shape();
    if shape.len() != 4 {
        return Err(ControlNetError::Invalid(
            "global average pooling requires NCHW rank four".into(),
        ));
    }
    let pooled =
        tensor_mean_with_context_exact_native(backend, input, Some(&[2, 3]), true, None, context)
            .map_err(|error| canonical_error(context, error))?;
    let height = i64::try_from(shape[2])
        .map_err(|_| ControlNetError::Invalid("pool height exceeds repeat range".into()))?;
    let width = i64::try_from(shape[3])
        .map_err(|_| ControlNetError::Invalid("pool width exceeds repeat range".into()))?;
    tensor_repeat_with_context_exact_native(backend, &pooled, &[1, 1, height, width], context)
        .map_err(|error| canonical_error(context, error))
}

fn merge_results(
    backend: &CpuBackend,
    current: ControlResult,
    previous: ControlResult,
    context: &ExecutionContext<'_>,
) -> Result<ControlResult, ControlNetError> {
    Ok(ControlResult {
        input: merge_slots(backend, current.input, previous.input, context)?,
        middle: merge_slots(backend, current.middle, previous.middle, context)?,
        output: merge_slots(backend, current.output, previous.output, context)?,
    })
}

fn merge_slots(
    backend: &CpuBackend,
    mut current: Vec<Option<Tensor>>,
    previous: Vec<Option<Tensor>>,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Option<Tensor>>, ControlNetError> {
    for (index, previous) in previous.into_iter().enumerate() {
        if index >= current.len() {
            current.push(previous);
            continue;
        }
        let current_slot = current
            .get_mut(index)
            .ok_or_else(|| ControlNetError::Invalid("control slot merge index overflow".into()))?;
        *current_slot = match (current_slot.take(), previous) {
            (None, previous) => previous,
            (current, None) => current,
            (Some(current), Some(previous)) => {
                Some(add_tensors(backend, &previous, &current, context)?)
            }
        };
    }
    Ok(current)
}

fn add_tensors(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ControlNetError> {
    if left.descriptor().dtype() != right.descriptor().dtype()
        || left.descriptor().device() != right.descriptor().device()
        || left.descriptor().stream() != right.descriptor().stream()
    {
        return Err(ControlNetError::Invalid(
            "merged control tensors must share dtype, device, and stream".into(),
        ));
    }
    let shape = binary_broadcast_shape(left.descriptor().shape(), right.descriptor().shape())?;
    let descriptor = contiguous_descriptor(&shape, left.descriptor().dtype(), left)?;
    backend
        .binary(BinaryOperation::Add, left, right, descriptor, context)
        .map(|(tensor, _)| tensor)
        .map_err(ControlNetError::from)
}

fn binary_scalar(
    backend: &CpuBackend,
    operation: BinaryOperation,
    input: &Tensor,
    scalar: Scalar,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ControlNetError> {
    let descriptor = contiguous_descriptor(
        input.descriptor().shape(),
        input.descriptor().dtype(),
        input,
    )?;
    backend
        .binary_scalar(
            operation,
            input,
            scalar,
            ScalarSide::Right,
            descriptor,
            context,
        )
        .map(|(tensor, _)| tensor)
        .map_err(ControlNetError::from)
}

fn cast_tensor(
    backend: &CpuBackend,
    input: &Tensor,
    dtype: DType,
    device: DeviceId,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ControlNetError> {
    cast_to_with_context_exact_native(backend, input, dtype, device, false, false, context)
        .map_err(|error| canonical_error(context, error))
}

fn contiguous_descriptor(
    shape: &[u64],
    dtype: DType,
    input: &Tensor,
) -> Result<TensorDescriptor, ControlNetError> {
    Ok(TensorDescriptor::contiguous(
        shape.to_vec(),
        dtype,
        input.descriptor().device(),
        input.descriptor().stream(),
    )?)
}

fn validate_backend_binding(
    backend: &CpuBackend,
    model: &ControlModelBinding,
    conditioning: &ControlConditioning,
) -> Result<(), ControlNetError> {
    if backend.device() != model.device {
        return Err(ControlNetError::Invalid(format!(
            "control backend {:?} does not match model device {:?}",
            backend.device(),
            model.device
        )));
    }
    if conditioning.noisy.tensor().descriptor().device() != model.device {
        return Err(ControlNetError::Invalid(
            "control noisy input is not resident on the model device".into(),
        ));
    }
    Ok(())
}

fn rounded_extent(value: u64, multiple: u64) -> Result<u64, ControlNetError> {
    value
        .checked_add(multiple - 1)
        .map(|rounded| rounded / multiple * multiple)
        .ok_or_else(|| ControlNetError::Invalid("T2I extent rounding overflow".into()))
}

fn compute_chain_identity(
    nodes: &[ControlNode],
) -> Result<ControlExecutionIdentity, ControlNetError> {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, b"zed-controlnet-chain-v1");
    hash_u64(
        &mut hasher,
        u64::try_from(nodes.len())
            .map_err(|_| ControlNetError::Invalid("chain length overflow".into()))?,
    );
    for node in nodes {
        match node {
            ControlNode::ControlNet(control) => {
                hash_field(&mut hasher, b"ControlNet");
                hash_controlnet(&mut hasher, control)?;
            }
            ControlNode::ControlLora(control) => {
                hash_field(&mut hasher, b"ControlLora");
                hash_controlnet(&mut hasher, &control.control)?;
                hash_patch(&mut hasher, &control.operations.patch);
            }
            ControlNode::ControlNetSD35(control) => {
                hash_field(&mut hasher, b"ControlNetSD35");
                hash_controlnet(&mut hasher, &control.control)?;
            }
            ControlNode::T2IAdapter(adapter) => {
                hash_field(&mut hasher, b"T2IAdapter");
                hash_base(&mut hasher, &adapter.base);
                hash_model(&mut hasher, &adapter.model);
                hash_tensor_binding(&mut hasher, &adapter.hint);
                hash_u64(&mut hasher, adapter.channels_in);
                hash_u64(&mut hasher, adapter.compression_ratio);
                hash_u64(&mut hasher, adapter.unshuffle_amount);
                hash_field(&mut hasher, resize_mode_tag(adapter.resize_mode));
            }
        }
    }
    Ok(ControlExecutionIdentity(format!("{:x}", hasher.finalize())))
}

fn hash_controlnet(hasher: &mut Sha256, control: &ControlNet) -> Result<(), ControlNetError> {
    hash_base(hasher, &control.base);
    hash_model(hasher, &control.model);
    hash_tensor_binding(hasher, &control.hint);
    hash_u64(hasher, control.compression_ratio);
    hash_field(hasher, resize_mode_tag(control.resize_mode));
    hash_field(hasher, control.preprocess.identity_tag());
    hash_optional_field(hasher, control.expected_vae_sha256.as_deref());
    hash_u64(
        hasher,
        u64::try_from(control.extra_concat.len())
            .map_err(|_| ControlNetError::Invalid("extra concat length overflow".into()))?,
    );
    for binding in &control.extra_concat {
        hash_tensor_binding(hasher, binding);
    }
    hash_bool(hasher, control.concat_mask);
    if let Some(control_type) = control.union_control_type {
        hash_field(hasher, b"zed.sim.controlnet.union-control-type.v1");
        hash_u64(hasher, u64::from(control_type));
    }
    hash_u64(
        hasher,
        u64::try_from(control.extra_conditioning_names.len())
            .map_err(|_| ControlNetError::Invalid("extra conditioning length overflow".into()))?,
    );
    for name in &control.extra_conditioning_names {
        hash_field(hasher, name.as_bytes());
    }
    Ok(())
}

fn hash_base(hasher: &mut Sha256, base: &ControlBase) {
    hash_field(hasher, &base.strength.to_bits().to_le_bytes());
    hash_field(hasher, base.strength_type.identity_tag());
    hash_field(hasher, &base.percent_window.start.to_bits().to_le_bytes());
    hash_field(hasher, &base.percent_window.end.to_bits().to_le_bytes());
    hash_bool(hasher, base.global_average_pooling);
    hash_optional_field(hasher, base.output_dtype.map(DType::catalog_name));
}

fn hash_model(hasher: &mut Sha256, model: &ControlModelBinding) {
    hash_field(hasher, model.model_family.feature_id().as_bytes());
    hash_field(hasher, model.model_family.identifier().as_bytes());
    hash_field(hasher, model.model_family.architecture_version().as_bytes());
    hash_patch(hasher, &model.patch);
    hash_field(hasher, model.model_state_sha256.as_bytes());
    hash_field(hasher, model.executor_sha256.as_bytes());
    hash_field(hasher, model.dtype.catalog_name().as_bytes());
    hash_field(hasher, format!("{:?}", model.device).as_bytes());
}

fn hash_patch(hasher: &mut Sha256, patch: &PatchGraphIdentity) {
    hash_u64(hasher, u64::from(patch.schema_version));
    hash_field(hasher, patch.base_artifact_digest.as_bytes());
    hash_field(hasher, patch.ordered_digest.as_bytes());
}

fn hash_tensor_binding(hasher: &mut Sha256, binding: &ControlTensorBinding) {
    let descriptor = binding.tensor.descriptor();
    hash_field(hasher, binding.content_sha256.as_bytes());
    hash_u64(
        hasher,
        u64::try_from(descriptor.shape().len()).unwrap_or(u64::MAX),
    );
    for dimension in descriptor.shape() {
        hash_u64(hasher, *dimension);
    }
    hash_field(hasher, descriptor.dtype().catalog_name().as_bytes());
    hash_field(hasher, format!("{:?}", descriptor.device()).as_bytes());
    hash_u64(hasher, descriptor.stream().get());
}

fn hash_tensor_bindings(
    hasher: &mut Sha256,
    bindings: &[ControlTensorBinding],
) -> Result<(), ControlNetError> {
    hash_u64(
        hasher,
        u64::try_from(bindings.len())
            .map_err(|_| ControlNetError::Invalid("tensor binding length overflow".into()))?,
    );
    for binding in bindings {
        hash_tensor_binding(hasher, binding);
    }
    Ok(())
}

fn hash_optional_tensor_binding(hasher: &mut Sha256, binding: Option<&ControlTensorBinding>) {
    hash_bool(hasher, binding.is_some());
    if let Some(binding) = binding {
        hash_tensor_binding(hasher, binding);
    }
}

fn resize_mode_tag(mode: ResizeMode) -> &'static [u8] {
    match mode {
        ResizeMode::NearestExact => b"nearest-exact",
        ResizeMode::Bilinear => b"bilinear",
        ResizeMode::Area => b"area",
        ResizeMode::Bicubic => b"bicubic",
        ResizeMode::Lanczos => b"lanczos",
    }
}

fn validate_unique_names(names: &[String]) -> Result<(), ControlNetError> {
    let mut previous = None;
    let mut ordered = names.iter().collect::<Vec<_>>();
    ordered.sort();
    for name in ordered {
        if name.trim().is_empty() || name.len() > 256 {
            return Err(ControlNetError::Invalid(
                "control extra conditioning name is invalid".into(),
            ));
        }
        if previous == Some(name.as_str()) {
            return Err(ControlNetError::Invalid(format!(
                "duplicate control extra conditioning name: {name}"
            )));
        }
        previous = Some(name);
    }
    Ok(())
}

fn validate_sha256(label: &str, value: &str) -> Result<(), ControlNetError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(ControlNetError::Invalid(format!(
            "{label} digest must be lowercase SHA-256"
        )));
    }
    Ok(())
}

fn check_context(context: &ExecutionContext<'_>) -> Result<(), ControlNetError> {
    context.check().map_err(ControlNetError::from)
}

fn canonical_error<E>(context: &ExecutionContext<'_>, error: E) -> ControlNetError
where
    E: std::error::Error + Send + Sync + 'static,
{
    if context.cancellation.is_cancelled() {
        ControlNetError::Cancelled
    } else {
        ControlNetError::CanonicalTensor(Box::new(error))
    }
}

fn hash_field(hasher: &mut Sha256, bytes: &[u8]) {
    hash_u64(hasher, u64::try_from(bytes.len()).unwrap_or(u64::MAX));
    hasher.update(bytes);
}

fn hash_optional_field(hasher: &mut Sha256, value: Option<&str>) {
    hash_bool(hasher, value.is_some());
    if let Some(value) = value {
        hash_field(hasher, value.as_bytes());
    }
}

fn hash_bool(hasher: &mut Sha256, value: bool) {
    hasher.update([u8::from(value)]);
}

fn hash_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_le_bytes());
}

impl From<TensorError> for ControlNetError {
    fn from(error: TensorError) -> Self {
        match error {
            TensorError::Cancelled => Self::Cancelled,
            error => Self::Tensor(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{
        CancellationToken, CpuWorkspaceAuthority, StreamId,
        generated_native_diffusion::{tensor_from_f32, tensor_to_f32},
    };
    use std::{
        collections::BTreeSet,
        error::Error,
        path::Path,
        sync::{Arc, Mutex},
    };

    const MEMORY_BYTES: u64 = 16 * 1_048_576;

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn backend_and_context<'a>(
        cancellation: &'a CancellationToken,
        memory: u64,
    ) -> Result<
        (
            CpuBackend,
            sim_tensor::BackendWorkspaceAuthority,
            ExecutionContext<'a>,
        ),
        TensorError,
    > {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(memory)?;
        let context = ExecutionContext {
            stream: StreamId::DEFAULT,
            scratch: authority.authorize_workspace(memory)?,
            rng_phase: None,
            cancellation,
        };
        Ok((backend, authority, context))
    }

    fn tensor(
        backend: &CpuBackend,
        shape: &[u64],
        values: &[f32],
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, Box<dyn Error>> {
        Ok(tensor_from_f32(backend, shape, values, context)?)
    }

    fn values(
        backend: &CpuBackend,
        tensor: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        Ok(tensor_to_f32(backend, tensor, context)?.to_vec())
    }

    fn binding(tensor: Tensor, character: char) -> Result<ControlTensorBinding, ControlNetError> {
        ControlTensorBinding::checked(tensor, digest(character))
    }

    fn base(
        strength: f32,
        strength_type: StrengthType,
        global_average_pooling: bool,
        output_dtype: Option<DType>,
    ) -> Result<ControlBase, ControlNetError> {
        ControlBase::checked(
            strength,
            strength_type,
            ControlPercentWindow::checked(0.0, 1.0)?,
            global_average_pooling,
            output_dtype,
        )
    }

    fn model_binding() -> Result<ControlModelBinding, Box<dyn Error>> {
        let model_state_sha256 = digest('a');
        Ok(ControlModelBinding::checked(
            ModelFamilyIdentity::new("COMFY-MODEL-0001", "test-control", "v1")?,
            PatchGraphIdentity {
                schema_version: crate::PATCH_GRAPH_SCHEMA_VERSION,
                base_artifact_digest: model_state_sha256.clone(),
                ordered_digest: digest('b'),
            },
            model_state_sha256,
            digest('c'),
            DType::F32,
            DeviceId::CPU,
        )?)
    }

    fn controlnet(
        hint: ControlTensorBinding,
        strength: f32,
        strength_type: StrengthType,
        preprocess: ControlHintPreprocess,
    ) -> Result<ControlNet, Box<dyn Error>> {
        controlnet_with_vae(hint, strength, strength_type, preprocess, None)
    }

    fn controlnet_with_vae(
        hint: ControlTensorBinding,
        strength: f32,
        strength_type: StrengthType,
        preprocess: ControlHintPreprocess,
        expected_vae_sha256: Option<String>,
    ) -> Result<ControlNet, Box<dyn Error>> {
        Ok(ControlNet::checked(
            base(strength, strength_type, false, None)?,
            model_binding()?,
            hint,
            1,
            ResizeMode::NearestExact,
            preprocess,
            expected_vae_sha256,
            Vec::new(),
            false,
            vec!["y".into()],
        )?)
    }

    #[test]
    fn canonical_tensor_errors_preserve_typed_resource_and_cancellation_sources()
    -> Result<(), Box<dyn Error>> {
        let active = CancellationToken::default();
        let (_backend, _authority, context) = backend_and_context(&active, MEMORY_BYTES)?;
        let resource = canonical_error(
            &context,
            TensorError::WorkspaceAuthorizationExceeded {
                requested: 128,
                authorized: 64,
                in_use: 0,
            },
        );
        let ControlNetError::CanonicalTensor(resource) = resource else {
            return Err("canonical tensor resource failure lost its wrapper".into());
        };
        assert!(matches!(
            resource.downcast_ref::<TensorError>(),
            Some(TensorError::WorkspaceAuthorizationExceeded { .. })
        ));

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let (_backend, _authority, context) = backend_and_context(&cancelled, MEMORY_BYTES)?;
        assert!(matches!(
            canonical_error(&context, TensorError::Cancelled),
            ControlNetError::Cancelled
        ));
        Ok(())
    }

    #[test]
    fn chain_vae_binding_is_uniform_exact_and_checked_before_execution()
    -> Result<(), Box<dyn Error>> {
        let cancellation: CancellationToken = Default::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, MEMORY_BYTES)?;
        let vae_digest = digest('d');
        let other_vae_digest = digest('e');
        let bound_hint = tensor(&backend, &[1, 1, 1, 1], &[1.0], &context)?;
        let bound = controlnet_with_vae(
            binding(bound_hint, 'f')?,
            1.0,
            StrengthType::Constant,
            ControlHintPreprocess::Identity,
            Some(vae_digest.clone()),
        )?;
        let bound_chain = ControlChain::checked(vec![ControlNode::ControlNet(bound.clone())])?;
        bound_chain.require_vae_execution_digest(Some(&vae_digest))?;
        assert!(matches!(
            bound_chain.require_vae_execution_digest(None),
            Err(ControlNetError::Invalid(_))
        ));
        assert!(matches!(
            bound_chain.require_vae_execution_digest(Some(&other_vae_digest)),
            Err(ControlNetError::Invalid(_))
        ));

        let unbound_hint = tensor(&backend, &[1, 1, 1, 1], &[1.0], &context)?;
        let unbound = controlnet(
            binding(unbound_hint.clone(), '1')?,
            1.0,
            StrengthType::Constant,
            ControlHintPreprocess::Identity,
        )?;
        let unbound_chain = ControlChain::checked(vec![ControlNode::ControlNet(unbound.clone())])?;
        unbound_chain.require_vae_execution_digest(None)?;
        assert!(matches!(
            unbound_chain.require_vae_execution_digest(Some(&vae_digest)),
            Err(ControlNetError::Invalid(_))
        ));

        let adapter = T2IAdapter::checked(
            base(1.0, StrengthType::Constant, false, None)?,
            model_binding()?,
            binding(unbound_hint, '2')?,
            1,
            1,
            1,
            ResizeMode::NearestExact,
        )?;
        let adapter_chain = ControlChain::checked(vec![ControlNode::T2IAdapter(adapter)])?;
        adapter_chain.require_vae_execution_digest(None)?;
        assert!(matches!(
            adapter_chain.require_vae_execution_digest(Some(&vae_digest)),
            Err(ControlNetError::Invalid(_))
        ));

        let mixed_chain = ControlChain::checked(vec![
            ControlNode::ControlNet(bound),
            ControlNode::ControlNet(unbound),
        ])?;
        assert!(matches!(
            mixed_chain.require_vae_execution_digest(None),
            Err(ControlNetError::Invalid(_))
        ));
        assert!(matches!(
            mixed_chain.require_vae_execution_digest(Some(&vae_digest)),
            Err(ControlNetError::Invalid(_))
        ));
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn control_chain_resident_bytes_deduplicate_aliased_tensor_storage()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, MEMORY_BYTES)?;
        let shared = tensor(&backend, &[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0], &context)?;
        let distinct_hint = tensor(&backend, &[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0], &context)?;
        let distinct_extra = tensor(&backend, &[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0], &context)?;
        let control = |hint: Tensor, extra: Tensor| -> Result<ControlNet, Box<dyn Error>> {
            Ok(ControlNet::checked(
                base(1.0, StrengthType::Constant, false, None)?,
                model_binding()?,
                binding(hint, 'a')?,
                1,
                ResizeMode::NearestExact,
                ControlHintPreprocess::Identity,
                None,
                vec![binding(extra, 'b')?],
                false,
                vec!["y".into()],
            )?)
        };
        let aliased = ControlChain::checked(vec![ControlNode::ControlNet(control(
            shared.clone(),
            shared.clone(),
        )?)])?;
        let independent = ControlChain::checked(vec![ControlNode::ControlNet(control(
            distinct_hint,
            distinct_extra,
        )?)])?;
        assert_eq!(aliased.identity(), independent.identity());
        let aliased_parts = aliased.resident_parts()?;
        let independent_parts = independent.resident_parts()?;
        assert_eq!(aliased_parts.tensor_allocations().len(), 1);
        assert_eq!(independent_parts.tensor_allocations().len(), 2);
        assert_eq!(aliased_parts.resident_bytes()?, aliased.resident_bytes()?);
        assert_eq!(
            independent_parts.resident_bytes()?,
            independent.resident_bytes()?
        );
        assert_eq!(
            independent.resident_bytes()?,
            aliased
                .resident_bytes()?
                .checked_add(shared.storage_byte_len())
                .ok_or("resident byte comparison overflow")?
        );

        let first_outer = Arc::new(aliased.clone());
        let second_outer = Arc::new(aliased);
        assert!(!Arc::ptr_eq(&first_outer, &second_outer));
        assert_eq!(
            first_outer.resident_parts()?.tensor_allocations(),
            second_outer.resident_parts()?.tensor_allocations()
        );

        let overflow = ControlChainResidentParts {
            owned_bytes: u64::MAX,
            tensor_allocations: vec![
                *aliased_parts
                    .tensor_allocations()
                    .first()
                    .ok_or("aliased control storage projection is empty")?,
            ],
        };
        assert!(matches!(
            overflow.resident_bytes(),
            Err(ControlNetError::ResidentBytesOverflow)
        ));
        Ok(())
    }

    fn conditioning(
        noisy: ControlTensorBinding,
        timestep: ControlTensorBinding,
        attention: ControlTensorBinding,
        resolved_percent: f32,
        batched_number: u64,
    ) -> Result<ControlConditioning, ControlNetError> {
        ControlConditioning::checked(
            noisy,
            timestep,
            attention,
            None,
            Vec::new(),
            None,
            BTreeMap::new(),
            resolved_percent,
            batched_number,
        )
    }

    #[derive(Default)]
    struct RecordingExecutor {
        calls: Mutex<Vec<(Vec<u64>, Vec<u64>)>>,
    }

    impl ControlModelExecutor for RecordingExecutor {
        fn execution_digest(&self) -> &str {
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
        }

        fn resident_bytes(&self) -> Result<u64, ControlNetError> {
            u64::try_from(mem::size_of::<Self>())
                .map_err(|_| ControlNetError::ResidentBytesOverflow)
        }

        fn execute_controlnet(
            &self,
            _binding: &ControlModelBinding,
            input: &ControlModelInput,
            context: &ExecutionContext<'_>,
        ) -> Result<ControlResult, ControlNetError> {
            check_context(context)?;
            self.calls
                .lock()
                .map_err(|_| ControlNetError::Invalid("recording executor lock poisoned".into()))?
                .push((
                    input.noisy.descriptor().shape().to_vec(),
                    input.hint.descriptor().shape().to_vec(),
                ));
            ControlResult::checked(
                vec![Some(input.hint.clone()), Some(input.hint.clone())],
                vec![None],
                vec![Some(input.noisy.clone())],
            )
        }

        fn execute_t2i_adapter(
            &self,
            _binding: &ControlModelBinding,
            hint: &Tensor,
            context: &ExecutionContext<'_>,
        ) -> Result<ControlResult, ControlNetError> {
            check_context(context)?;
            self.calls
                .lock()
                .map_err(|_| ControlNetError::Invalid("recording executor lock poisoned".into()))?
                .push((Vec::new(), hint.descriptor().shape().to_vec()));
            ControlResult::checked(vec![Some(hint.clone())], Vec::new(), Vec::new())
        }
    }

    #[test]
    fn strength_and_percent_window_bounds_are_checked() -> Result<(), Box<dyn Error>> {
        assert!(ControlPercentWindow::checked(0.25, 0.75)?.contains(0.25)?);
        assert!(ControlPercentWindow::checked(0.25, 0.75)?.contains(0.75)?);
        assert!(!ControlPercentWindow::checked(0.25, 0.75)?.contains(0.1)?);
        assert!(ControlPercentWindow::checked(-0.1, 0.5).is_err());
        assert!(ControlPercentWindow::checked(0.8, 0.2).is_err());
        assert!(ControlPercentWindow::checked(0.0, f32::NAN).is_err());
        assert_eq!(StrengthType::Constant.multiplier(0.5, 4, 1)?, 0.5);
        assert_eq!(StrengthType::LinearUp.multiplier(2.0, 4, 1)?, 8.0);
        assert!(base(f32::INFINITY, StrengthType::Constant, false, None).is_err());
        Ok(())
    }

    #[test]
    fn quotient_remainder_broadcast_matches_source_order() -> Result<(), Box<dyn Error>> {
        let cancellation: CancellationToken = Default::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, MEMORY_BYTES)?;
        let input = tensor(&backend, &[3, 1, 1, 1], &[1.0, 2.0, 3.0], &context)?;
        let output = broadcast_image_to(&backend, &input, 10, 2, &context)?;
        assert_eq!(output.descriptor().shape(), &[10, 1, 1, 1]);
        assert_eq!(
            values(&backend, &output, &context)?,
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0]
        );
        let singleton = tensor(&backend, &[1, 1, 1, 1], &[9.0], &context)?;
        assert_eq!(
            broadcast_image_to(&backend, &singleton, 10, 2, &context)?.tensor_id(),
            singleton.tensor_id()
        );
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn shared_tensors_receive_linear_strength_once_and_slots_stay_ordered()
    -> Result<(), Box<dyn Error>> {
        let cancellation: CancellationToken = Default::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, MEMORY_BYTES)?;
        let shared = tensor(&backend, &[1, 1, 1, 1], &[1.0], &context)?;
        let final_slot = tensor(&backend, &[1, 1, 1, 1], &[1.0], &context)?;
        let result = ControlResult::checked(
            vec![Some(shared.clone()), Some(shared), Some(final_slot)],
            vec![None],
            vec![],
        )?;
        let transformed = transform_result(
            &backend,
            result,
            &base(2.0, StrengthType::LinearUp, false, None)?,
            &context,
        )?;
        assert_eq!(
            values(
                &backend,
                transformed.input()[0].as_ref().ok_or("slot 0")?,
                &context
            )?,
            vec![8.0]
        );
        assert_eq!(
            values(
                &backend,
                transformed.input()[1].as_ref().ok_or("slot 1")?,
                &context
            )?,
            vec![8.0]
        );
        assert_eq!(
            values(
                &backend,
                transformed.input()[2].as_ref().ok_or("slot 2")?,
                &context
            )?,
            vec![2.0]
        );
        assert!(transformed.middle()[0].is_none());
        assert!(transformed.output().is_empty());
        Ok(())
    }

    #[test]
    fn previous_chain_merge_fills_adds_and_appends_fixed_slots() -> Result<(), Box<dyn Error>> {
        let cancellation: CancellationToken = Default::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, MEMORY_BYTES)?;
        let one = tensor(&backend, &[1, 1, 1, 1], &[1.0], &context)?;
        let two = tensor(&backend, &[1, 1, 1, 1], &[2.0], &context)?;
        let three = tensor(&backend, &[1, 1, 1, 1], &[3.0], &context)?;
        let current =
            ControlResult::checked(vec![Some(one.clone()), None], vec![], vec![Some(one)])?;
        let previous = ControlResult::checked(
            vec![Some(two.clone()), Some(two), Some(three.clone())],
            vec![Some(three)],
            vec![None],
        )?;
        let merged = merge_results(&backend, current, previous, &context)?;
        assert_eq!(
            values(
                &backend,
                merged.input()[0].as_ref().ok_or("added")?,
                &context
            )?,
            vec![3.0]
        );
        assert_eq!(
            values(
                &backend,
                merged.input()[1].as_ref().ok_or("filled")?,
                &context
            )?,
            vec![2.0]
        );
        assert_eq!(
            values(
                &backend,
                merged.input()[2].as_ref().ok_or("appended")?,
                &context
            )?,
            vec![3.0]
        );
        assert_eq!(merged.middle().len(), 1);
        assert_eq!(merged.output().len(), 1);
        Ok(())
    }

    #[test]
    fn global_average_pool_sd35_and_t2i_equations_are_exact() -> Result<(), Box<dyn Error>> {
        let cancellation: CancellationToken = Default::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, MEMORY_BYTES)?;
        let input = tensor(&backend, &[1, 1, 2, 2], &[1.0, 3.0, 5.0, 7.0], &context)?;
        assert_eq!(
            values(
                &backend,
                &global_average_pool(&backend, &input, &context)?,
                &context
            )?,
            vec![4.0; 4]
        );
        assert_eq!(
            values(
                &backend,
                &preprocess_hint(&backend, &input, ControlHintPreprocess::Sd35Canny, &context)?,
                &context,
            )?,
            vec![128.0, 383.0, 638.0, 893.0]
        );
        assert_eq!(
            values(
                &backend,
                &preprocess_hint(&backend, &input, ControlHintPreprocess::Sd35Depth, &context)?,
                &context,
            )?,
            vec![0.0, -2.0, -4.0, -6.0]
        );
        assert_eq!(rounded_extent(17, 8)?, 24);
        assert_eq!(rounded_extent(16, 8)?, 16);
        Ok(())
    }

    #[test]
    fn identity_binds_hint_predecessor_and_patch_without_tensor_allocation_identity()
    -> Result<(), Box<dyn Error>> {
        let cancellation: CancellationToken = Default::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, MEMORY_BYTES)?;
        let first_tensor = tensor(&backend, &[1, 1, 2, 2], &[0.0; 4], &context)?;
        let second_tensor = tensor(&backend, &[1, 1, 2, 2], &[0.0; 4], &context)?;
        let first = controlnet(
            binding(first_tensor, 'd')?,
            1.0,
            StrengthType::Constant,
            ControlHintPreprocess::Identity,
        )?;
        let equivalent = controlnet(
            binding(second_tensor.clone(), 'd')?,
            1.0,
            StrengthType::Constant,
            ControlHintPreprocess::Identity,
        )?;
        let changed_hint = controlnet(
            binding(second_tensor, 'e')?,
            1.0,
            StrengthType::Constant,
            ControlHintPreprocess::Identity,
        )?;
        let first_chain = ControlChain::checked(vec![ControlNode::ControlNet(first.clone())])?;
        let equivalent_chain = ControlChain::checked(vec![ControlNode::ControlNet(equivalent)])?;
        let changed_chain = ControlChain::checked(vec![ControlNode::ControlNet(changed_hint)])?;
        let predecessor_chain = ControlChain::checked(vec![
            ControlNode::ControlNet(first.clone()),
            ControlNode::ControlNet(first),
        ])?;
        assert_eq!(first_chain.identity(), equivalent_chain.identity());
        assert_ne!(first_chain.identity(), changed_chain.identity());
        assert_ne!(first_chain.identity(), predecessor_chain.identity());
        assert_eq!(first_chain.identity().digest().len(), 64);
        let noisy = tensor(&backend, &[1, 1, 1, 1], &[0.0], &context)?;
        let timestep = tensor(&backend, &[1], &[1.0], &context)?;
        let attention = tensor(&backend, &[1, 1, 1], &[0.0], &context)?;
        let first_conditioning = conditioning(
            binding(noisy.clone(), '4')?,
            binding(timestep.clone(), '5')?,
            binding(attention.clone(), '6')?,
            0.25,
            1,
        )?;
        let changed_conditioning = conditioning(
            binding(noisy, '7')?,
            binding(timestep, '5')?,
            binding(attention, '6')?,
            0.25,
            1,
        )?;
        assert_ne!(
            first_chain.execution_identity(&first_conditioning)?,
            first_chain.execution_identity(&changed_conditioning)?
        );
        Ok(())
    }

    #[test]
    fn runtime_resizes_executes_chain_and_converges_workspace() -> Result<(), Box<dyn Error>> {
        let cancellation: CancellationToken = Default::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, MEMORY_BYTES)?;
        let executor = RecordingExecutor::default();
        let hint = tensor(&backend, &[2, 1, 1, 1], &[1.0, 2.0], &context)?;
        let control = controlnet(
            binding(hint, 'd')?,
            0.5,
            StrengthType::Constant,
            ControlHintPreprocess::Identity,
        )?;
        let chain = ControlChain::checked(vec![ControlNode::ControlNet(control)])?;
        let noisy = tensor(&backend, &[4, 1, 2, 3], &[0.0; 24], &context)?;
        let timestep = tensor(&backend, &[1], &[1.0], &context)?;
        let attention = tensor(&backend, &[1, 1, 1], &[0.0], &context)?;
        let conditioning = conditioning(
            binding(noisy, 'e')?,
            binding(timestep, 'f')?,
            binding(attention, '1')?,
            0.5,
            2,
        )?;
        let result = ControlRuntime::new(&backend, &executor)
            .execute(
                &chain,
                ControlIsolation::CompleteChain,
                &conditioning,
                None,
                &context,
            )?
            .ok_or("control result")?;
        assert_eq!(result.input().len(), 2);
        assert_eq!(result.middle().len(), 1);
        assert_eq!(result.output().len(), 1);
        assert_eq!(
            *executor.calls.lock().map_err(|_| "calls lock")?,
            vec![(vec![4, 1, 2, 3], vec![4, 1, 2, 3])]
        );
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn cancellation_and_oom_publish_no_partial_result() -> Result<(), Box<dyn Error>> {
        let cancellation: CancellationToken = Default::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, MEMORY_BYTES)?;
        let executor = RecordingExecutor::default();
        let hint = tensor(&backend, &[1, 1, 1, 1], &[1.0], &context)?;
        let chain = ControlChain::checked(vec![ControlNode::ControlNet(controlnet(
            binding(hint, 'd')?,
            1.0,
            StrengthType::Constant,
            ControlHintPreprocess::Identity,
        )?)])?;
        let noisy = tensor(&backend, &[1, 1, 1, 1], &[0.0], &context)?;
        let timestep = tensor(&backend, &[1], &[1.0], &context)?;
        let attention = tensor(&backend, &[1, 1, 1], &[0.0], &context)?;
        let conditioning = conditioning(
            binding(noisy, 'e')?,
            binding(timestep, 'f')?,
            binding(attention, '1')?,
            0.5,
            1,
        )?;
        cancellation.cancel();
        assert!(matches!(
            ControlRuntime::new(&backend, &executor).execute(
                &chain,
                ControlIsolation::CompleteChain,
                &conditioning,
                None,
                &context,
            ),
            Err(ControlNetError::Cancelled)
        ));
        assert!(executor.calls.lock().map_err(|_| "calls lock")?.is_empty());
        assert_eq!(context.scratch.in_use_bytes(), 0);

        let oom_cancellation: CancellationToken = Default::default();
        let (oom_backend, _oom_authority, oom_context) =
            backend_and_context(&oom_cancellation, 96)?;
        let large = tensor(&oom_backend, &[1, 1, 4, 4], &[1.0; 16], &oom_context)?;
        let oom_result = transform_result(
            &oom_backend,
            ControlResult::checked(vec![Some(large)], Vec::new(), Vec::new())?,
            &base(2.0, StrengthType::Constant, false, None)?,
            &oom_context,
        );
        assert!(oom_result.is_err());
        assert_eq!(oom_context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn control_lora_delegates_patch_math_and_sd35_requires_preprocess() -> Result<(), Box<dyn Error>>
    {
        let cancellation: CancellationToken = Default::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, MEMORY_BYTES)?;
        let hint = tensor(&backend, &[1, 1, 1, 1], &[0.0], &context)?;
        let standard = controlnet(
            binding(hint.clone(), 'd')?,
            1.0,
            StrengthType::Constant,
            ControlHintPreprocess::Identity,
        )?;
        assert!(ControlNetSD35::checked(standard.clone()).is_err());
        let sd35 = controlnet(
            binding(hint, 'd')?,
            1.0,
            StrengthType::Constant,
            ControlHintPreprocess::Sd35Canny,
        )?;
        assert!(ControlNetSD35::checked(sd35).is_ok());
        let operations = ControlLoraOps::checked(
            standard.model.patch.clone(),
            standard.model.model_state_sha256(),
        )?;
        let lora = ControlLora::checked(standard, operations)?;
        assert_eq!(lora.operations().patch(), lora.control().model().patch());
        Ok(())
    }

    fn python_symbol_sha256(source: &[u8], symbol: &str) -> Result<String, Box<dyn Error>> {
        let source = std::str::from_utf8(source)?;
        let lines = source.split_inclusive('\n').collect::<Vec<_>>();
        let class_parenthesized = format!("class {symbol}(");
        let class_plain = format!("class {symbol}:");
        let matches = lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| {
                let trimmed = line.trim_start_matches([' ', '\t']);
                (trimmed.starts_with(&class_parenthesized) || trimmed.starts_with(&class_plain))
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        let [start] = matches.as_slice() else {
            return Err(format!(
                "expected exactly one Python class for {symbol}, found {}",
                matches.len()
            )
            .into());
        };
        let indentation = lines[*start].len() - lines[*start].trim_start_matches([' ', '\t']).len();
        let mut header_complete = lines[*start].trim_end().ends_with(':');
        let mut body_seen = false;
        let mut end = *start + 1;
        while let Some(line) = lines.get(end) {
            let trimmed = line.trim_start_matches([' ', '\t']);
            let content = trimmed.trim_end_matches(['\r', '\n']);
            if content.is_empty() || content.starts_with('#') {
                end += 1;
                continue;
            }
            let line_indentation = line.len() - trimmed.len();
            if !header_complete {
                header_complete = line_indentation == indentation && content.ends_with(':');
                end += 1;
                continue;
            }
            if body_seen && line_indentation <= indentation {
                break;
            }
            if line_indentation > indentation {
                body_seen = true;
            }
            end += 1;
        }
        if !body_seen {
            return Err(format!("Python class {symbol} has no indented body").into());
        }
        while end > *start + 1 {
            let content = lines[end - 1].trim();
            if content.is_empty() || content.starts_with('#') {
                end -= 1;
            } else {
                break;
            }
        }
        let mut hasher = Sha256::new();
        for line in &lines[*start..end] {
            hasher.update(line.as_bytes());
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    fn execute_catalog_contract(
        symbol: &str,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<&'static str>, Box<dyn Error>> {
        let hint = || -> Result<ControlTensorBinding, Box<dyn Error>> {
            Ok(binding(
                tensor(backend, &[2, 1, 1, 1], &[1.0, 2.0], context)?,
                'd',
            )?)
        };
        let cases = match symbol {
            "StrengthType" => {
                assert_eq!(StrengthType::Constant.multiplier(0.5, 3, 1)?, 0.5);
                assert_eq!(StrengthType::LinearUp.multiplier(2.0, 3, 1)?, 4.0);
                vec!["constant", "linear-up"]
            }
            "ControlIsolation" => {
                assert_ne!(
                    ControlIsolation::CompleteChain,
                    ControlIsolation::CurrentControlOnly
                );
                vec!["complete-chain", "current-only"]
            }
            "ControlBase" => {
                let base = base(1.25, StrengthType::LinearUp, true, Some(DType::F32))?;
                assert_eq!(base.strength(), 1.25);
                assert!(base.global_average_pooling());
                assert_eq!(base.output_dtype(), Some(DType::F32));
                vec!["checked-strength-window-pooling-dtype"]
            }
            "ControlNet" => {
                let control = controlnet(
                    hint()?,
                    1.0,
                    StrengthType::Constant,
                    ControlHintPreprocess::Identity,
                )?;
                assert_eq!(control.model().dtype(), DType::F32);
                assert_eq!(control.hint().tensor().descriptor().shape(), &[2, 1, 1, 1]);
                vec!["hint-binding", "model-boundary"]
            }
            "ControlLoraOps" => {
                let model = model_binding()?;
                let operations =
                    ControlLoraOps::checked(model.patch().clone(), model.model_state_sha256())?;
                assert_eq!(operations.patch(), model.patch());
                vec!["canonical-patch-graph-delegation"]
            }
            "ControlLora" => {
                let control = controlnet(
                    hint()?,
                    1.0,
                    StrengthType::Constant,
                    ControlHintPreprocess::Identity,
                )?;
                let operations = ControlLoraOps::checked(
                    control.model().patch().clone(),
                    control.model().model_state_sha256(),
                )?;
                let lora = ControlLora::checked(control, operations)?;
                assert_eq!(lora.operations().patch(), lora.control().model().patch());
                vec!["model-patch-identity-binding"]
            }
            "ControlNetSD35" => {
                let control = controlnet(
                    hint()?,
                    1.0,
                    StrengthType::Constant,
                    ControlHintPreprocess::Sd35Depth,
                )?;
                assert!(ControlNetSD35::checked(control).is_ok());
                vec!["canny-depth-preprocess-binding"]
            }
            "T2IAdapter" => {
                let adapter = T2IAdapter::checked(
                    base(1.0, StrengthType::Constant, false, Some(DType::F32))?,
                    model_binding()?,
                    hint()?,
                    1,
                    8,
                    8,
                    ResizeMode::NearestExact,
                )?;
                assert_eq!(adapter.channels_in, 1);
                assert_eq!(rounded_extent(17, adapter.unshuffle_amount)?, 24);
                vec!["ceil-unshuffle", "grayscale-input"]
            }
            _ => return Err(format!("unaccounted ControlNet catalog symbol {symbol}").into()),
        };
        Ok(cases)
    }

    #[test]
    fn val_controlnet_001_catalog_manifest_is_exact_digest_bound_and_executable()
    -> Result<(), Box<dyn Error>> {
        const TASK: &str = "comfy-parity-controlnet-chain-foundation";
        const EXPECTED: [(&str, &str, &str, &str); 8] = [
            (
                "conditioning-controlnet-controlnet-strengthtype-77c5b612",
                "17",
                "efa2dec0eed7ed04da4b91cce0358267afeec9513af02b063a122d3ece093e57",
                "StrengthType",
            ),
            (
                "conditioning-controlnet-controlnet-controlisolation-68347c8b",
                "18",
                "efa2dec0eed7ed04da4b91cce0358267afeec9513af02b063a122d3ece093e57",
                "ControlIsolation",
            ),
            (
                "conditioning-controlnet-controlnet-controlbase-9b5fbfe8",
                "19",
                "efa2dec0eed7ed04da4b91cce0358267afeec9513af02b063a122d3ece093e57",
                "ControlBase",
            ),
            (
                "conditioning-controlnet-controlnet-controlnet-118682de",
                "20",
                "efa2dec0eed7ed04da4b91cce0358267afeec9513af02b063a122d3ece093e57",
                "ControlNet",
            ),
            (
                "conditioning-controlnet-controlnet-controlloraops-35e74d65",
                "21",
                "efa2dec0eed7ed04da4b91cce0358267afeec9513af02b063a122d3ece093e57",
                "ControlLoraOps",
            ),
            (
                "conditioning-controlnet-controlnet-controllora-ef66a4b4",
                "22",
                "efa2dec0eed7ed04da4b91cce0358267afeec9513af02b063a122d3ece093e57",
                "ControlLora",
            ),
            (
                "conditioning-controlnet-controlnet-controlnetsd35-8d4f077e",
                "23",
                "efa2dec0eed7ed04da4b91cce0358267afeec9513af02b063a122d3ece093e57",
                "ControlNetSD35",
            ),
            (
                "conditioning-controlnet-controlnet-t2iadapter-97dfdd8b",
                "24",
                "efa2dec0eed7ed04da4b91cce0358267afeec9513af02b063a122d3ece093e57",
                "T2IAdapter",
            ),
        ];
        const REQUIRED_CASE_IDS: [&str; 5] = [
            "controlnet:all-eight-contracts",
            "controlnet:strength-and-slot-merge",
            "controlnet:hint-preprocessing-and-batching",
            "controlnet:vae-latent-and-chain-delegation",
            "controlnet:cancellation-oom-workspace-ownership",
        ];
        let expected = EXPECTED
            .into_iter()
            .map(|entry| (entry.0, entry))
            .collect::<BTreeMap<_, _>>();
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let catalog = std::fs::read_to_string(
            repository
                .join(".agents/specs/comfy-parity/catalogs/backend-conditioning-contracts.csv"),
        )?;
        let cancellation: CancellationToken = Default::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, MEMORY_BYTES)?;
        let mut seen = BTreeSet::new();
        let mut contracts = Vec::new();
        for line in catalog.lines().skip(1) {
            let columns = line.split(',').collect::<Vec<_>>();
            if columns.get(8).copied() != Some(TASK) {
                continue;
            }
            assert_eq!(columns.len(), 15, "malformed ControlNet catalog row");
            let contract_id = columns[0];
            let expected_row = expected
                .get(contract_id)
                .ok_or_else(|| format!("unexpected ControlNet contract {contract_id}"))?;
            assert!(seen.insert(contract_id));
            assert_eq!(columns[1], "controlnet");
            assert_eq!(columns[2], "projects/comfy/ComfyUI/sim/controlnet.py");
            assert_eq!(columns[3], expected_row.3);
            assert_eq!(columns[4], expected_row.1);
            assert_eq!(columns[5], expected_row.2);
            assert_eq!(columns[7], "sim_model::controlnet");
            assert_eq!(columns[9], "sim_model::controlnet::tests");
            assert_eq!(columns[10], "native_rust");
            validate_sha256("catalog source", columns[5])?;
            validate_sha256("catalog symbol", columns[6])?;
            let source = std::fs::read(repository.join(columns[2]))?;
            assert_eq!(format!("{:x}", Sha256::digest(&source)), columns[5]);
            assert_eq!(python_symbol_sha256(&source, columns[3])?, columns[6]);
            let case_ids = execute_catalog_contract(columns[3], &backend, &context)?
                .into_iter()
                .map(|case_id| format!("{contract_id}:{case_id}"))
                .collect::<Vec<_>>();
            contracts.push(serde_json::json!({
                "contract_id": contract_id,
                "task_id": TASK,
                "source_sha256": columns[5],
                "symbol_sha256": columns[6],
                "status": "passed",
                "case_ids": case_ids,
            }));
        }
        assert_eq!(seen, expected.keys().copied().collect());
        assert_eq!(contracts.len(), 8);
        let implementation_path = "crates/sim_model/src/controlnet.rs";
        let implementation = std::fs::read(repository.join(implementation_path))?;
        let implementation_sha256 = format!("{:x}", Sha256::digest(implementation));
        let task_implementations = [
            "crates/sim_model/src/sim_model.rs",
            "crates/sim_model/src/controlnet.rs",
        ]
        .into_iter()
        .map(|path| {
            let bytes = std::fs::read(repository.join(path))?;
            Ok(serde_json::json!({
                "path": path,
                "sha256": format!("{:x}", Sha256::digest(bytes)),
            }))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        let task_results = BTreeMap::from([(
            TASK,
            serde_json::json!({
                "status": "passed",
                "passed": contracts.len(),
                "failed": 0,
                "skipped": 0,
                "case_ids": REQUIRED_CASE_IDS,
                "implementations": task_implementations,
            }),
        )]);
        let artifact = serde_json::json!({
            "schema_version": 1,
            "validation_id": "VAL-CONTROLNET-001",
            "overall_status": "passed",
            "environment": {
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
                "backend": "sim_tensor::CpuBackend",
                "device": "cpu",
                "dtype": "f32",
            },
            "summary": {
                "passed": contracts.len(),
                "failed": 0,
                "skipped": 0,
            },
            "implementation": {
                "path": implementation_path,
                "sha256": implementation_sha256,
            },
            "task_results": task_results,
            "contracts": contracts,
        });
        let artifact_directory = repository.join("target/sim-parity");
        std::fs::create_dir_all(&artifact_directory)?;
        std::fs::write(
            artifact_directory.join("val-controlnet-001.json"),
            serde_json::to_vec_pretty(&artifact)?,
        )?;
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(())
    }

    #[test]
    fn task403_controlnet_rebind_is_immutable_layout_exact_and_chain_ordered()
    -> Result<(), Box<dyn Error>> {
        let cancellation = CancellationToken::default();
        let (backend, _authority, context) = backend_and_context(&cancellation, MEMORY_BYTES)?;
        let original_hint = binding(tensor(&backend, &[1, 3, 2, 2], &[0.0; 12], &context)?, 'd')?;
        let original = ControlChain::checked(vec![ControlNode::ControlNet(controlnet(
            original_hint,
            1.0,
            StrengthType::Constant,
            ControlHintPreprocess::Identity,
        )?)])?;
        let predecessor = ControlChain::checked(vec![ControlNode::ControlNet(controlnet(
            binding(tensor(&backend, &[1, 3, 1, 1], &[2.0; 3], &context)?, 'e')?,
            0.5,
            StrengthType::Constant,
            ControlHintPreprocess::Identity,
        )?)])?;
        let request = ControlNetRebindRequest::checked(
            binding(
                tensor(
                    &backend,
                    &[1, 2, 2, 3],
                    &(0..12).map(|value| value as f32).collect::<Vec<_>>(),
                    &context,
                )?,
                'f',
            )?,
            ControlHintLayout::Nhwc,
            0.75,
            ControlPercentWindow::checked(0.2, 0.8)?,
            Some(digest('9')),
            vec![binding(
                tensor(&backend, &[1, 1, 2, 2], &[1.0; 4], &context)?,
                '8',
            )?],
            true,
            Some(6),
        )?;
        let original_identity = original.identity().digest().to_owned();
        let rebound = original.rebind_current(&request, Some(&predecessor), &cancellation)?;
        assert_eq!(original.identity().digest(), original_identity);
        assert_eq!(rebound.nodes().len(), 2);
        let ControlNode::ControlNet(control) =
            rebound.nodes().last().ok_or("missing rebound control")?
        else {
            return Err("unexpected rebound node".into());
        };
        assert_eq!(control.hint().tensor().descriptor().shape(), &[1, 3, 2, 2]);
        assert_eq!(control.base().strength(), 0.75);
        assert_eq!(
            control.base().percent_window(),
            ControlPercentWindow::checked(0.2, 0.8)?
        );
        assert_eq!(control.expected_vae_sha256(), Some(digest('9').as_str()));
        assert_eq!(control.extra_concat().len(), 1);
        assert!(control.concat_mask());
        assert_eq!(control.union_control_type(), Some(6));
        assert_ne!(rebound.identity(), original.identity());

        let cancelled = CancellationToken::default();
        assert!(cancelled.cancel());
        assert!(matches!(
            original.rebind_current(&request, None, &cancelled),
            Err(ControlNetError::Cancelled)
        ));
        assert!(
            ControlNetRebindRequest::checked(
                binding(tensor(&backend, &[1, 2, 2, 3], &[0.0; 12], &context)?, '7')?,
                ControlHintLayout::Nhwc,
                f32::NAN,
                ControlPercentWindow::checked(0.0, 1.0)?,
                None,
                Vec::new(),
                false,
                None,
            )
            .is_err()
        );
        assert!(
            ControlNetRebindRequest::checked(
                binding(tensor(&backend, &[1, 2, 2, 3], &[0.0; 12], &context)?, '6')?,
                ControlHintLayout::Nhwc,
                1.0,
                ControlPercentWindow::checked(0.0, 1.0)?,
                None,
                Vec::new(),
                false,
                Some(8),
            )
            .is_err()
        );
        Ok(())
    }
}
