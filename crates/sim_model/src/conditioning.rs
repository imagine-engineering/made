use crate::{ClipVisionOutput, LatentFormatIdentity, ModelFamilyIdentity};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_tensor::{
    CpuBackend, DType, DeviceId, ExecutionContext, NativeLatentBundle, ResizeMode, StorageId,
    Tensor, TensorDescriptor, TensorError,
    generated_external_tensor_kernel_01::{
        ExternalTensorKernelPartOneError, resize_with_context_exact_native,
    },
    generated_indexing_masking_01::{IndexingMaskingPartOneError, narrow_method_exact_native},
    generated_shape_layout_transform_01::{
        ShapeLayoutTransformPartOneError, tensor_unsqueeze_exact_native,
    },
    generated_shape_layout_transform_02::{
        ShapeLayoutTransformPartTwoError, tensor_repeat_with_context_exact_native,
        torch_cat_with_context_exact_native,
    },
    generated_sim_operator_indirection_01::{
        OperatorIndirectionError, cast_to_with_context_exact_native,
    },
};
use sim_types::CancellationError;
use std::{
    collections::{BTreeMap, BTreeSet},
    mem,
    sync::Arc,
};
use thiserror::Error;

pub const CONDITIONING_SCHEMA_VERSION: u16 = 1;
const CROSS_ATTENTION_MAX_REPEAT_RATIO: u64 = 4;
const MAX_CONDITIONING_CONSTANT_DEPTH: usize = 64;
const MAX_CONDITIONING_CONSTANT_ITEMS: usize = 4_096;
const MAX_CONDITIONING_CONSTANT_BYTES: usize = 16 * 1_048_576;
const MAX_CONDITIONING_LIST_ITEMS: usize = 1_024;
const MAX_CONDITIONING_ENTRIES: usize = 4_096;
const MAX_CONDITIONING_REGION_RANK: usize = 8;
const MAX_CONDITIONING_HOOK_REFERENCES: usize = 64;
const MAX_CONDITIONING_METADATA_ENTRIES: usize = 28;
const MAX_CONDITIONING_METADATA_LIST_ITEMS: usize = 64;
const MAX_CONDITIONING_METADATA_TENSOR_BYTES: u64 = 256 * 1_048_576;
const SOURCE_AREA_FEATHER: u64 = 8;

#[derive(Debug, Error)]
pub enum ConditioningError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Narrow(#[from] IndexingMaskingPartOneError),
    #[error(transparent)]
    ShapeView(#[from] ShapeLayoutTransformPartOneError),
    #[error(transparent)]
    ShapeOperation(#[from] ShapeLayoutTransformPartTwoError),
    #[error(transparent)]
    TensorCast(#[from] OperatorIndirectionError),
    #[error(transparent)]
    TensorResize(#[from] ExternalTensorKernelPartOneError),
    #[error("conditioning operation was cancelled")]
    Cancelled,
    #[error("conditioning input is invalid: {0}")]
    Invalid(String),
    #[error("conditioning shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
    #[error("conditioning resident byte accounting overflowed")]
    ResidentBytesOverflow,
    #[error("conditioning tensor content cannot be read on device {0:?}")]
    UnreadableDevice(DeviceId),
    #[error("conditioning descriptor encoding failed: {0}")]
    Encoding(String),
}

impl From<CancellationError> for ConditioningError {
    fn from(_: CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ConditioningIdentity {
    namespace: String,
    model_family: ModelFamilyIdentity,
    latent_format: LatentFormatIdentity,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ConditioningIdentityWire {
    schema_version: u16,
    namespace: String,
    model_family: ModelFamilyIdentity,
    latent_format: LatentFormatIdentity,
}

impl ConditioningIdentity {
    pub fn new(
        namespace: impl Into<String>,
        model_family: ModelFamilyIdentity,
        latent_format: LatentFormatIdentity,
    ) -> Result<Self, ConditioningError> {
        let namespace = namespace.into();
        validate_identifier("conditioning namespace", &namespace)?;
        Ok(Self {
            namespace,
            model_family,
            latent_format,
        })
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn model_family(&self) -> &ModelFamilyIdentity {
        &self.model_family
    }

    pub fn latent_format(&self) -> &LatentFormatIdentity {
        &self.latent_format
    }

    pub fn digest(&self) -> Result<String, ConditioningError> {
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.conditioning-identity.v1");
        hasher.update(CONDITIONING_SCHEMA_VERSION.to_le_bytes());
        hash_string(&mut hasher, &self.namespace)?;
        hash_string(&mut hasher, self.model_family.feature_id())?;
        hash_string(&mut hasher, self.model_family.identifier())?;
        hash_string(&mut hasher, self.model_family.architecture_version())?;
        hash_string(&mut hasher, self.latent_format.feature_id())?;
        hash_string(&mut hasher, self.latent_format.identifier())?;
        Ok(format!("{:x}", hasher.finalize()))
    }

    pub fn resident_bytes(&self) -> Result<u64, ConditioningError> {
        let mut bytes = u64::try_from(std::mem::size_of::<Self>())
            .map_err(|_| ConditioningError::ResidentBytesOverflow)?;
        bytes = bytes
            .checked_add(
                u64::try_from(self.namespace.capacity())
                    .map_err(|_| ConditioningError::ResidentBytesOverflow)?,
            )
            .and_then(|bytes| bytes.checked_add(self.model_family.owned_resident_bytes()?))
            .and_then(|bytes| bytes.checked_add(self.latent_format.owned_resident_bytes()?))
            .ok_or(ConditioningError::ResidentBytesOverflow)?;
        Ok(bytes)
    }

    fn validate(&self) -> Result<(), ConditioningError> {
        validate_identifier("conditioning namespace", &self.namespace)?;
        self.digest()?;
        self.resident_bytes()?;
        Ok(())
    }
}

impl Serialize for ConditioningIdentity {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        ConditioningIdentityWire {
            schema_version: CONDITIONING_SCHEMA_VERSION,
            namespace: self.namespace.clone(),
            model_family: self.model_family.clone(),
            latent_format: self.latent_format.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ConditioningIdentity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ConditioningIdentityWire::deserialize(deserializer)?;
        if wire.schema_version != CONDITIONING_SCHEMA_VERSION {
            return Err(serde::de::Error::custom(format!(
                "unsupported conditioning identity schema version: {}",
                wire.schema_version
            )));
        }
        Self::new(wire.namespace, wire.model_family, wire.latent_format)
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ConditioningControlReference(String);

impl ConditioningControlReference {
    pub fn checked(identifier: impl Into<String>) -> Result<Self, ConditioningError> {
        let identifier = identifier.into();
        validate_identifier("conditioning control reference", &identifier)?;
        Ok(Self(identifier))
    }

    pub fn identifier(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ConditioningHookReference(String);

impl ConditioningHookReference {
    pub fn checked(identifier: impl Into<String>) -> Result<Self, ConditioningError> {
        let identifier = identifier.into();
        validate_identifier("conditioning hook reference", &identifier)?;
        Ok(Self(identifier))
    }

    pub fn identifier(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ConditioningReferences {
    control: Option<ConditioningControlReference>,
    hooks: Vec<ConditioningHookReference>,
}

impl ConditioningReferences {
    pub fn checked(
        control: Option<ConditioningControlReference>,
        hooks: Vec<ConditioningHookReference>,
    ) -> Result<Self, ConditioningError> {
        let references = Self { control, hooks };
        references.validate()?;
        Ok(references)
    }

    fn validate(&self) -> Result<(), ConditioningError> {
        if self.hooks.len() > MAX_CONDITIONING_HOOK_REFERENCES {
            return Err(ConditioningError::Invalid(format!(
                "conditioning entry contains more than {MAX_CONDITIONING_HOOK_REFERENCES} hook references"
            )));
        }
        let mut identifiers = BTreeSet::new();
        for hook in &self.hooks {
            validate_identifier("conditioning hook reference", hook.identifier())?;
            if !identifiers.insert(hook.identifier()) {
                return Err(ConditioningError::Invalid(format!(
                    "conditioning hook reference is duplicated: {}",
                    hook.identifier()
                )));
            }
        }
        if let Some(control) = &self.control {
            validate_identifier("conditioning control reference", control.identifier())?;
        }
        Ok(())
    }

    pub fn control(&self) -> Option<&ConditioningControlReference> {
        self.control.as_ref()
    }

    pub fn hooks(&self) -> &[ConditioningHookReference] {
        &self.hooks
    }
}

#[derive(Clone, Debug)]
pub enum ConditioningConstant {
    Null,
    Boolean(bool),
    Signed(i64),
    Unsigned(u64),
    FiniteF64Bits(u64),
    Text(String),
    Bytes(Vec<u8>),
    Tensor(Tensor),
    List(Vec<ConditioningConstant>),
    Tuple(Vec<ConditioningConstant>),
    Map(BTreeMap<String, ConditioningConstant>),
}

impl PartialEq for ConditioningConstant {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Boolean(left), Self::Boolean(right)) => left == right,
            (Self::Signed(left), Self::Signed(right)) => left == right,
            (Self::Unsigned(left), Self::Unsigned(right)) => left == right,
            (Self::FiniteF64Bits(left), Self::FiniteF64Bits(right)) => left == right,
            (Self::Text(left), Self::Text(right)) => left == right,
            (Self::Bytes(left), Self::Bytes(right)) => left == right,
            (Self::Tensor(left), Self::Tensor(right)) => tensor_constants_equal(left, right),
            (Self::List(left), Self::List(right)) | (Self::Tuple(left), Self::Tuple(right)) => {
                left == right
            }
            (Self::Map(left), Self::Map(right)) => left == right,
            _ => false,
        }
    }
}

impl ConditioningConstant {
    pub fn finite_f64(value: f64) -> Result<Self, ConditioningError> {
        if !value.is_finite() {
            return Err(ConditioningError::Invalid(
                "constant floating-point value must be finite".to_owned(),
            ));
        }
        Ok(Self::FiniteF64Bits(value.to_bits()))
    }

    fn validate(&self) -> Result<(), ConditioningError> {
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(MAX_CONDITIONING_CONSTANT_DEPTH)
            .map_err(|_| ConditioningError::ShapeOverflow("conditioning constant validation"))?;
        pending.push((self, 1_usize));
        let mut item_count = 0_usize;
        let mut byte_count = 0_usize;
        while let Some((value, depth)) = pending.pop() {
            if depth > MAX_CONDITIONING_CONSTANT_DEPTH {
                return Err(ConditioningError::Invalid(format!(
                    "conditioning constant nesting exceeds {MAX_CONDITIONING_CONSTANT_DEPTH}"
                )));
            }
            item_count = item_count
                .checked_add(1)
                .ok_or(ConditioningError::ShapeOverflow(
                    "conditioning constant items",
                ))?;
            if item_count > MAX_CONDITIONING_CONSTANT_ITEMS {
                return Err(ConditioningError::Invalid(format!(
                    "conditioning constant contains more than {MAX_CONDITIONING_CONSTANT_ITEMS} items"
                )));
            }
            match value {
                Self::FiniteF64Bits(bits) if !f64::from_bits(*bits).is_finite() => {
                    return Err(ConditioningError::Invalid(
                        "constant floating-point value must be finite".to_owned(),
                    ));
                }
                Self::Text(value) => {
                    byte_count = byte_count.checked_add(value.len()).ok_or(
                        ConditioningError::ShapeOverflow("conditioning constant bytes"),
                    )?;
                }
                Self::Bytes(value) => {
                    byte_count = byte_count.checked_add(value.len()).ok_or(
                        ConditioningError::ShapeOverflow("conditioning constant bytes"),
                    )?;
                }
                Self::Tensor(tensor) => {
                    validate_constant_tensor(tensor)?;
                    byte_count = byte_count
                        .checked_add(usize::try_from(tensor.storage_byte_len()).map_err(|_| {
                            ConditioningError::ShapeOverflow("conditioning constant bytes")
                        })?)
                        .ok_or(ConditioningError::ShapeOverflow(
                            "conditioning constant bytes",
                        ))?;
                }
                Self::List(values) | Self::Tuple(values) => {
                    if values.len() > MAX_CONDITIONING_CONSTANT_ITEMS {
                        return Err(ConditioningError::Invalid(
                            "conditioning constant list exceeds the item limit".to_owned(),
                        ));
                    }
                    pending.try_reserve(values.len()).map_err(|_| {
                        ConditioningError::ShapeOverflow("conditioning constant validation")
                    })?;
                    for child in values.iter().rev() {
                        pending.push((child, depth.saturating_add(1)));
                    }
                }
                Self::Map(values) => {
                    if values.len() > MAX_CONDITIONING_CONSTANT_ITEMS {
                        return Err(ConditioningError::Invalid(
                            "conditioning constant map exceeds the item limit".to_owned(),
                        ));
                    }
                    pending.try_reserve(values.len()).map_err(|_| {
                        ConditioningError::ShapeOverflow("conditioning constant validation")
                    })?;
                    for (key, child) in values.iter().rev() {
                        byte_count = byte_count.checked_add(key.len()).ok_or(
                            ConditioningError::ShapeOverflow("conditioning constant bytes"),
                        )?;
                        pending.push((child, depth.saturating_add(1)));
                    }
                }
                _ => {}
            }
            if byte_count > MAX_CONDITIONING_CONSTANT_BYTES {
                return Err(ConditioningError::Invalid(format!(
                    "conditioning constant payload exceeds {MAX_CONDITIONING_CONSTANT_BYTES} bytes"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum ConditioningValue {
    Regular(Tensor),
    NoiseShape(Tensor),
    CrossAttention(Tensor),
    Constant(ConditioningConstant),
    List(Vec<Tensor>),
}

impl ConditioningValue {
    pub fn regular(tensor: Tensor) -> Result<Self, ConditioningError> {
        validate_batched_tensor(&tensor, "regular conditioning")?;
        Ok(Self::Regular(tensor))
    }

    pub fn noise_shape(tensor: Tensor) -> Result<Self, ConditioningError> {
        validate_batched_tensor(&tensor, "noise-shape conditioning")?;
        if tensor.descriptor().rank() < 3 {
            return Err(ConditioningError::Invalid(
                "noise-shape conditioning must have batch, channel, and spatial dimensions"
                    .to_owned(),
            ));
        }
        Ok(Self::NoiseShape(tensor))
    }

    pub fn cross_attention(tensor: Tensor) -> Result<Self, ConditioningError> {
        validate_batched_tensor(&tensor, "cross-attention conditioning")?;
        if tensor.descriptor().rank() != 3 {
            return Err(ConditioningError::Invalid(
                "cross-attention conditioning must have [batch, tokens, channels] shape".to_owned(),
            ));
        }
        if tensor.descriptor().shape().get(1).copied() == Some(0) {
            return Err(ConditioningError::Invalid(
                "cross-attention token count must be nonzero".to_owned(),
            ));
        }
        Ok(Self::CrossAttention(tensor))
    }

    pub fn constant(value: ConditioningConstant) -> Result<Self, ConditioningError> {
        value.validate()?;
        Ok(Self::Constant(value))
    }

    pub fn list(tensors: Vec<Tensor>) -> Result<Self, ConditioningError> {
        if tensors.is_empty() {
            return Err(ConditioningError::Invalid(
                "conditioning tensor list must be nonempty".to_owned(),
            ));
        }
        if tensors.len() > MAX_CONDITIONING_LIST_ITEMS {
            return Err(ConditioningError::Invalid(format!(
                "conditioning tensor list contains more than {MAX_CONDITIONING_LIST_ITEMS} items"
            )));
        }
        for tensor in &tensors {
            validate_batched_tensor(tensor, "conditioning list item")?;
        }
        Ok(Self::List(tensors))
    }

    pub fn size(&self) -> Result<Vec<u64>, ConditioningError> {
        self.validate()?;
        match self {
            Self::Regular(tensor) | Self::NoiseShape(tensor) | Self::CrossAttention(tensor) => {
                Ok(tensor.descriptor().shape().to_vec())
            }
            Self::Constant(_) => Ok(vec![1]),
            Self::List(tensors) => {
                let mut elements = 0_u64;
                let mut token_count = 1_u64;
                for tensor in tensors {
                    elements = elements
                        .checked_add(tensor.descriptor().element_count()?)
                        .ok_or(ConditioningError::ShapeOverflow("conditioning list size"))?;
                    if let Some(tokens) = tensor.descriptor().shape().get(1) {
                        token_count = *tokens;
                    }
                }
                if token_count == 0 || !elements.is_multiple_of(token_count) {
                    return Err(ConditioningError::Invalid(
                        "conditioning list size is not token aligned".to_owned(),
                    ));
                }
                Ok(vec![1, token_count, elements / token_count])
            }
        }
    }

    pub fn process(
        &self,
        batch_size: u64,
        region: Option<&ResolvedConditioningRegion>,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ConditioningError> {
        context.cancellation.check()?;
        self.validate()?;
        if batch_size == 0 {
            return Err(ConditioningError::Invalid(
                "conditioning batch size must be nonzero".to_owned(),
            ));
        }
        let output = match self {
            Self::Regular(tensor) => {
                Self::Regular(repeat_to_batch_size(tensor, batch_size, backend, context)?)
            }
            Self::NoiseShape(tensor) => {
                let tensor = match region {
                    Some(region) => narrow_spatial(tensor, region, context)?,
                    None => tensor.clone(),
                };
                Self::NoiseShape(repeat_to_batch_size(&tensor, batch_size, backend, context)?)
            }
            Self::CrossAttention(tensor) => {
                Self::CrossAttention(repeat_to_batch_size(tensor, batch_size, backend, context)?)
            }
            Self::Constant(value) => Self::Constant(value.clone()),
            Self::List(tensors) => {
                let mut output = Vec::new();
                output
                    .try_reserve_exact(tensors.len())
                    .map_err(|_| ConditioningError::ShapeOverflow("conditioning list output"))?;
                for tensor in tensors {
                    context.cancellation.check()?;
                    output.push(repeat_to_batch_size(tensor, batch_size, backend, context)?);
                }
                Self::List(output)
            }
        };
        context.cancellation.check()?;
        Ok(output)
    }

    pub fn can_concat(&self, other: &Self) -> bool {
        if self.validate().is_err() || other.validate().is_err() {
            return false;
        }
        match (self, other) {
            (Self::Regular(left), Self::Regular(right))
            | (Self::NoiseShape(left), Self::NoiseShape(right)) => {
                tensor_concat_compatible(left, right, None)
            }
            (Self::CrossAttention(left), Self::CrossAttention(right)) => {
                cross_attention_pair_compatible(left, right)
            }
            (Self::Constant(left), Self::Constant(right)) => left == right,
            (Self::List(left), Self::List(right)) => {
                left.len() == right.len()
                    && left
                        .iter()
                        .zip(right)
                        .all(|(left, right)| tensor_concat_compatible(left, right, None))
            }
            _ => false,
        }
    }

    pub fn concat(
        &self,
        others: &[Self],
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ConditioningError> {
        context.cancellation.check()?;
        self.validate()?;
        for other in others {
            other.validate()?;
        }
        match self {
            Self::Regular(first) => Ok(Self::Regular(concat_regular(
                first,
                others,
                |value| match value {
                    Self::Regular(tensor) => Some(tensor),
                    _ => None,
                },
                backend,
                context,
            )?)),
            Self::NoiseShape(first) => Ok(Self::NoiseShape(concat_regular(
                first,
                others,
                |value| match value {
                    Self::NoiseShape(tensor) => Some(tensor),
                    _ => None,
                },
                backend,
                context,
            )?)),
            Self::CrossAttention(first) => Ok(Self::CrossAttention(concat_cross_attention(
                first, others, backend, context,
            )?)),
            Self::Constant(value) => {
                if others
                    .iter()
                    .all(|other| matches!(other, Self::Constant(other) if other == value))
                {
                    Ok(Self::Constant(value.clone()))
                } else {
                    Err(ConditioningError::Invalid(
                        "constant conditioning values must be equal to concatenate".to_owned(),
                    ))
                }
            }
            Self::List(first) => concat_lists(first, others, backend, context),
        }
    }

    pub fn deterministic_digest(
        &self,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<String, ConditioningError> {
        let mut checkpoint = || cancellation.check().map_err(ConditioningError::from);
        self.deterministic_digest_with_checkpoint(&mut checkpoint)
    }

    fn deterministic_digest_with_checkpoint<Checkpoint>(
        &self,
        checkpoint: &mut Checkpoint,
    ) -> Result<String, ConditioningError>
    where
        Checkpoint: FnMut() -> Result<(), ConditioningError>,
    {
        checkpoint()?;
        self.validate()?;
        let mut hasher = Sha256::new();
        match self {
            Self::Regular(tensor) => {
                hasher.update([1]);
                hash_tensor(&mut hasher, tensor, checkpoint)?;
            }
            Self::NoiseShape(tensor) => {
                hasher.update([2]);
                hash_tensor(&mut hasher, tensor, checkpoint)?;
            }
            Self::CrossAttention(tensor) => {
                hasher.update([3]);
                hash_tensor(&mut hasher, tensor, checkpoint)?;
            }
            Self::Constant(value) => {
                hasher.update([4]);
                hash_conditioning_constant(&mut hasher, value, checkpoint)?;
            }
            Self::List(tensors) => {
                hasher.update([5]);
                hash_u64(&mut hasher, tensors.len())?;
                for tensor in tensors {
                    hash_tensor(&mut hasher, tensor, checkpoint)?;
                }
            }
        }
        checkpoint()?;
        Ok(format!("{:x}", hasher.finalize()))
    }

    fn validate(&self) -> Result<(), ConditioningError> {
        match self {
            Self::Regular(tensor) => validate_batched_tensor(tensor, "regular conditioning"),
            Self::NoiseShape(tensor) => {
                validate_batched_tensor(tensor, "noise-shape conditioning")?;
                if tensor.descriptor().rank() < 3 {
                    return Err(ConditioningError::Invalid(
                        "noise-shape conditioning must have batch, channel, and spatial dimensions"
                            .to_owned(),
                    ));
                }
                Ok(())
            }
            Self::CrossAttention(tensor) => {
                validate_batched_tensor(tensor, "cross-attention conditioning")?;
                if tensor.descriptor().rank() != 3
                    || tensor.descriptor().shape().get(1).copied() == Some(0)
                {
                    return Err(ConditioningError::Invalid(
                        "cross-attention conditioning must have nonzero [batch, tokens, channels] shape"
                            .to_owned(),
                    ));
                }
                Ok(())
            }
            Self::Constant(value) => value.validate(),
            Self::List(tensors) => {
                if tensors.is_empty() {
                    return Err(ConditioningError::Invalid(
                        "conditioning tensor list must be nonempty".to_owned(),
                    ));
                }
                if tensors.len() > MAX_CONDITIONING_LIST_ITEMS {
                    return Err(ConditioningError::Invalid(format!(
                        "conditioning tensor list contains more than {MAX_CONDITIONING_LIST_ITEMS} items"
                    )));
                }
                for tensor in tensors {
                    validate_batched_tensor(tensor, "conditioning list item")?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConditioningRegion {
    Absolute { sizes: Vec<u64>, offsets: Vec<u64> },
    Percentage { sizes: Vec<f64>, offsets: Vec<f64> },
}

impl ConditioningRegion {
    pub fn absolute(sizes: Vec<u64>, offsets: Vec<u64>) -> Result<Self, ConditioningError> {
        validate_region_rank(sizes.len(), offsets.len())?;
        if sizes.contains(&0) {
            return Err(ConditioningError::Invalid(
                "absolute conditioning region sizes must be nonzero".to_owned(),
            ));
        }
        Ok(Self::Absolute { sizes, offsets })
    }

    pub fn percentage(sizes: Vec<f64>, offsets: Vec<f64>) -> Result<Self, ConditioningError> {
        validate_region_rank(sizes.len(), offsets.len())?;
        for size in &sizes {
            if !size.is_finite() || *size < 0.0 || *size > 1.0 {
                return Err(ConditioningError::Invalid(
                    "percentage conditioning region sizes must be finite and in [0, 1]".to_owned(),
                ));
            }
        }
        for offset in &offsets {
            if !offset.is_finite() || *offset < 0.0 || *offset > 1.0 {
                return Err(ConditioningError::Invalid(
                    "percentage conditioning region offsets must be finite and in [0, 1]"
                        .to_owned(),
                ));
            }
        }
        Ok(Self::Percentage { sizes, offsets })
    }

    pub fn rank(&self) -> usize {
        match self {
            Self::Absolute { sizes, .. } => sizes.len(),
            Self::Percentage { sizes, .. } => sizes.len(),
        }
    }

    pub fn resolve(
        &self,
        dimensions: &[u64],
    ) -> Result<ResolvedConditioningRegion, ConditioningError> {
        self.validate()?;
        if dimensions.len() != self.rank() || dimensions.contains(&0) {
            return Err(ConditioningError::Invalid(
                "conditioning region rank must match nonzero target dimensions".to_owned(),
            ));
        }
        let (requested_sizes, offsets) = match self {
            Self::Absolute { sizes, offsets } => (sizes.clone(), offsets.clone()),
            Self::Percentage { sizes, offsets } => {
                if sizes.contains(&0.0) {
                    return Err(ConditioningError::Invalid(
                        "zero-size source percentage regions cannot be consumed by the sampler"
                            .to_owned(),
                    ));
                }
                let mut resolved_sizes = Vec::new();
                let mut resolved_offsets = Vec::new();
                resolved_sizes
                    .try_reserve_exact(sizes.len())
                    .map_err(|_| ConditioningError::ShapeOverflow("percentage region sizes"))?;
                resolved_offsets
                    .try_reserve_exact(offsets.len())
                    .map_err(|_| ConditioningError::ShapeOverflow("percentage region offsets"))?;
                for ((size, offset), dimension) in sizes.iter().zip(offsets).zip(dimensions) {
                    let resolved_size = (*size * *dimension as f64).round_ties_even().max(1.0);
                    let resolved_offset = (*offset * *dimension as f64).round_ties_even();
                    if resolved_size > u64::MAX as f64 || resolved_offset > u64::MAX as f64 {
                        return Err(ConditioningError::ShapeOverflow("percentage region"));
                    }
                    resolved_sizes.push(resolved_size as u64);
                    resolved_offsets.push(resolved_offset as u64);
                }
                (resolved_sizes, resolved_offsets)
            }
        };
        let mut sizes = Vec::new();
        sizes
            .try_reserve_exact(dimensions.len())
            .map_err(|_| ConditioningError::ShapeOverflow("resolved region sizes"))?;
        for ((requested, offset), dimension) in requested_sizes.iter().zip(&offsets).zip(dimensions)
        {
            if *offset >= *dimension {
                return Err(ConditioningError::Invalid(
                    "conditioning region offset lies outside the target".to_owned(),
                ));
            }
            sizes.push((*requested).min(*dimension - *offset));
        }
        ResolvedConditioningRegion::checked(sizes, offsets, dimensions.to_vec())
    }

    fn validate(&self) -> Result<(), ConditioningError> {
        match self {
            Self::Absolute { sizes, offsets } => {
                validate_region_rank(sizes.len(), offsets.len())?;
                if sizes.contains(&0) {
                    return Err(ConditioningError::Invalid(
                        "absolute conditioning region sizes must be nonzero".to_owned(),
                    ));
                }
            }
            Self::Percentage { sizes, offsets } => {
                validate_region_rank(sizes.len(), offsets.len())?;
                if sizes
                    .iter()
                    .any(|size| !size.is_finite() || *size < 0.0 || *size > 1.0)
                    || offsets
                        .iter()
                        .any(|offset| !offset.is_finite() || *offset < 0.0 || *offset > 1.0)
                {
                    return Err(ConditioningError::Invalid(
                        "percentage conditioning region values are outside their checked ranges"
                            .to_owned(),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedConditioningRegion {
    sizes: Vec<u64>,
    offsets: Vec<u64>,
    target_dimensions: Vec<u64>,
}

impl ResolvedConditioningRegion {
    fn checked(
        sizes: Vec<u64>,
        offsets: Vec<u64>,
        target_dimensions: Vec<u64>,
    ) -> Result<Self, ConditioningError> {
        validate_region_rank(sizes.len(), offsets.len())?;
        if target_dimensions.len() != sizes.len() {
            return Err(ConditioningError::Invalid(
                "resolved region rank does not match target rank".to_owned(),
            ));
        }
        for ((size, offset), dimension) in sizes.iter().zip(&offsets).zip(&target_dimensions) {
            let end = offset
                .checked_add(*size)
                .ok_or(ConditioningError::ShapeOverflow("resolved region end"))?;
            if *size == 0 || end > *dimension {
                return Err(ConditioningError::Invalid(
                    "resolved region is outside the target".to_owned(),
                ));
            }
        }
        Ok(Self {
            sizes,
            offsets,
            target_dimensions,
        })
    }

    pub fn full(dimensions: Vec<u64>) -> Result<Self, ConditioningError> {
        if dimensions.is_empty() || dimensions.contains(&0) {
            return Err(ConditioningError::Invalid(
                "full conditioning region requires nonzero dimensions".to_owned(),
            ));
        }
        Self::checked(dimensions.clone(), vec![0; dimensions.len()], dimensions)
    }

    pub fn sizes(&self) -> &[u64] {
        &self.sizes
    }

    pub fn offsets(&self) -> &[u64] {
        &self.offsets
    }

    pub fn target_dimensions(&self) -> &[u64] {
        &self.target_dimensions
    }

    pub fn is_full(&self) -> bool {
        self.offsets.iter().all(|offset| *offset == 0) && self.sizes == self.target_dimensions
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConditioningWindow {
    start_percent: f64,
    end_percent: f64,
}

impl ConditioningWindow {
    pub fn new(start_percent: f64, end_percent: f64) -> Result<Self, ConditioningError> {
        let window = Self {
            start_percent,
            end_percent,
        };
        window.validate()?;
        Ok(window)
    }

    pub fn full() -> Self {
        Self {
            start_percent: 0.0,
            end_percent: 1.0,
        }
    }

    pub fn start_percent(self) -> f64 {
        self.start_percent
    }

    pub fn end_percent(self) -> f64 {
        self.end_percent
    }

    pub fn contains(self, percent: f64) -> bool {
        percent.is_finite() && percent >= self.start_percent && percent <= self.end_percent
    }

    fn validate(self) -> Result<(), ConditioningError> {
        if !self.start_percent.is_finite()
            || !self.end_percent.is_finite()
            || !(0.0..=1.0).contains(&self.start_percent)
            || !(0.0..=1.0).contains(&self.end_percent)
            || self.start_percent > self.end_percent
        {
            return Err(ConditioningError::Invalid(
                "conditioning window must be a finite ordered range within [0, 1]".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ConditioningMetadataKey {
    Region,
    Mask,
    MaskStrength,
    SetAreaToBounds,
    TimestepRange,
    ConcatLatentImage,
    ConcatMask,
    ConcatMaskIndex,
    Control,
    ControlApplyToUnconditioned,
    UnclipConditioning,
    ReferenceLatents,
    ClipVisionOutput,
    VaceFrames,
    VaceMask,
    VaceStrength,
    CameraConditions,
    TimeDimensionConcat,
    AudioEmbedding,
    ReferenceMotion,
    ControlVideo,
    PoseVideoLatent,
    FaceVideoPixels,
    DrivingMask28Channels,
    ReferenceMask28Channels,
    ReferenceMaskFlag,
    SecondsStart,
    SecondsTotal,
    CrossAttentionControlnet,
    PooledOutputControlnet,
    PooledOutput,
    FrameRate = 31,
    ConditioningLyrics = 32,
    ReferenceAudioTimbreLatents = 33,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConditioningTimestepRange {
    start_percent: f64,
    end_percent: f64,
}

impl ConditioningTimestepRange {
    pub fn source(start_percent: f64, end_percent: f64) -> Result<Self, ConditioningError> {
        if !start_percent.is_finite()
            || !end_percent.is_finite()
            || !(0.0..=1.0).contains(&start_percent)
            || !(0.0..=1.0).contains(&end_percent)
        {
            return Err(ConditioningError::Invalid(
                "conditioning source timestep values must be finite and within [0, 1]".to_owned(),
            ));
        }
        Ok(Self {
            start_percent,
            end_percent,
        })
    }

    pub fn start_percent(self) -> f64 {
        self.start_percent
    }

    pub fn end_percent(self) -> f64 {
        self.end_percent
    }

    pub fn sampler_window(self) -> Result<ConditioningWindow, ConditioningError> {
        ConditioningWindow::new(self.start_percent, self.end_percent)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConditioningControlBinding {
    execution_digest: String,
    apply_to_unconditioned: bool,
}

impl ConditioningControlBinding {
    pub fn checked(
        execution_digest: impl Into<String>,
        apply_to_unconditioned: bool,
    ) -> Result<Self, ConditioningError> {
        let execution_digest = execution_digest.into();
        if execution_digest.len() != 64
            || !execution_digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(ConditioningError::Invalid(
                "conditioning control binding must use a lowercase SHA-256 execution identity"
                    .to_owned(),
            ));
        }
        Ok(Self {
            execution_digest,
            apply_to_unconditioned,
        })
    }

    pub fn execution_digest(&self) -> &str {
        &self.execution_digest
    }

    pub fn apply_to_unconditioned(&self) -> bool {
        self.apply_to_unconditioned
    }
}

#[derive(Clone, Debug)]
pub struct ClipVisionConditioning {
    output: Arc<ClipVisionOutput>,
    strength: f64,
    noise_augmentation: f64,
}

impl ClipVisionConditioning {
    pub fn checked(
        output: Arc<ClipVisionOutput>,
        strength: f64,
        noise_augmentation: f64,
    ) -> Result<Self, ConditioningError> {
        if !strength.is_finite()
            || !noise_augmentation.is_finite()
            || !(0.0..=1.0).contains(&noise_augmentation)
        {
            return Err(ConditioningError::Invalid(
                "CLIP vision conditioning scalars must be finite and noise augmentation must be within [0, 1]"
                    .to_owned(),
            ));
        }
        output
            .validate()
            .map_err(|error| ConditioningError::Invalid(error.to_string()))?;
        Ok(Self {
            output,
            strength,
            noise_augmentation,
        })
    }

    pub fn output(&self) -> &Arc<ClipVisionOutput> {
        &self.output
    }

    pub fn strength(&self) -> f64 {
        self.strength
    }

    pub fn noise_augmentation(&self) -> f64 {
        self.noise_augmentation
    }
}

#[derive(Clone, Debug)]
pub enum ConditioningMetadataValue {
    Tensor(Tensor),
    TensorList(Vec<Tensor>),
    LatentBundle(Arc<NativeLatentBundle>),
    ClipVisionOutput(Arc<ClipVisionOutput>),
    ClipVisionConditioning(Vec<ClipVisionConditioning>),
    FiniteF64(f64),
    FiniteF64List(Vec<f64>),
    Unsigned(u64),
    Boolean(bool),
    Region(ConditioningRegion),
    TimestepRange(ConditioningTimestepRange),
    Control(ConditioningControlBinding),
    ControlnetPooledOutput(Option<Tensor>),
    PooledOutput(Option<Tensor>),
    ConditioningLyrics(Option<Tensor>),
    ReferenceAudioTimbreLatents(Option<Vec<Tensor>>),
}

#[derive(Clone, Debug, Default)]
pub struct ConditioningMetadataMap {
    values: BTreeMap<ConditioningMetadataKey, ConditioningMetadataValue>,
}

impl ConditioningMetadataMap {
    pub fn checked(
        values: impl IntoIterator<Item = (ConditioningMetadataKey, ConditioningMetadataValue)>,
    ) -> Result<Self, ConditioningError> {
        let mut map = BTreeMap::new();
        for (key, value) in values {
            if map.len() >= MAX_CONDITIONING_METADATA_ENTRIES && !map.contains_key(&key) {
                return Err(ConditioningError::Invalid(format!(
                    "conditioning metadata contains more than {MAX_CONDITIONING_METADATA_ENTRIES} entries"
                )));
            }
            validate_metadata_value(key, &value)?;
            if map.insert(key, value).is_some() {
                return Err(ConditioningError::Invalid(
                    "conditioning metadata contains a duplicate key".to_owned(),
                ));
            }
        }
        validate_metadata_values(&map)?;
        Ok(Self { values: map })
    }

    pub fn insert(
        &mut self,
        key: ConditioningMetadataKey,
        value: ConditioningMetadataValue,
    ) -> Result<(), ConditioningError> {
        if self.values.len() >= MAX_CONDITIONING_METADATA_ENTRIES && !self.values.contains_key(&key)
        {
            return Err(ConditioningError::Invalid(format!(
                "conditioning metadata contains more than {MAX_CONDITIONING_METADATA_ENTRIES} entries"
            )));
        }
        validate_metadata_value(key, &value)?;
        let mut values = self.values.clone();
        values.insert(key, value);
        validate_metadata_values(&values)?;
        self.values = values;
        Ok(())
    }

    pub fn get(&self, key: ConditioningMetadataKey) -> Option<&ConditioningMetadataValue> {
        self.values.get(&key)
    }

    pub fn iter(
        &self,
    ) -> impl Iterator<Item = (&ConditioningMetadataKey, &ConditioningMetadataValue)> {
        self.values.iter()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn validate(&self) -> Result<(), ConditioningError> {
        if self.values.len() > MAX_CONDITIONING_METADATA_ENTRIES {
            return Err(ConditioningError::Invalid(
                "conditioning metadata exceeds the supported bound".to_owned(),
            ));
        }
        validate_metadata_values(&self.values)
    }

    pub fn sampler_window(&self) -> Result<Option<ConditioningWindow>, ConditioningError> {
        match self.values.get(&ConditioningMetadataKey::TimestepRange) {
            Some(ConditioningMetadataValue::TimestepRange(range)) => {
                Ok(Some(range.sampler_window()?))
            }
            Some(_) => Err(ConditioningError::Invalid(
                "conditioning timestep metadata has an incompatible value".to_owned(),
            )),
            None => Ok(None),
        }
    }

    pub fn resident_tensor_bytes(&self) -> Result<u64, ConditioningError> {
        metadata_resident_bytes(&self.values)
    }
}

fn validate_metadata_values(
    values: &BTreeMap<ConditioningMetadataKey, ConditioningMetadataValue>,
) -> Result<(), ConditioningError> {
    for (key, value) in values {
        validate_metadata_value(*key, value)?;
    }
    use ConditioningMetadataKey as Key;
    use ConditioningMetadataValue as Value;
    if let Some(Value::FiniteF64(strength)) = values.get(&Key::MaskStrength)
        && *strength < 0.0
    {
        return Err(ConditioningError::Invalid(
            "conditioning mask strength must be nonnegative".to_owned(),
        ));
    }
    if values.contains_key(&Key::MaskStrength) && !values.contains_key(&Key::Mask) {
        return Err(ConditioningError::Invalid(
            "conditioning mask strength requires mask metadata".to_owned(),
        ));
    }
    if values.contains_key(&Key::SetAreaToBounds) && !values.contains_key(&Key::Region) {
        return Err(ConditioningError::Invalid(
            "set-area-to-bounds metadata requires region metadata".to_owned(),
        ));
    }
    if let Some(Value::Control(control)) = values.get(&Key::Control) {
        if let Some(Value::Boolean(apply)) = values.get(&Key::ControlApplyToUnconditioned)
            && *apply != control.apply_to_unconditioned()
        {
            return Err(ConditioningError::Invalid(
                "ControlNet unconditional-application metadata disagrees with its binding"
                    .to_owned(),
            ));
        }
    } else if values.contains_key(&Key::ControlApplyToUnconditioned) {
        return Err(ConditioningError::Invalid(
            "ControlNet unconditional-application metadata requires a control binding".to_owned(),
        ));
    }
    if let Some(Value::Unsigned(index)) = values.get(&Key::ConcatMaskIndex) {
        let count = match values.get(&Key::ConcatLatentImage) {
            Some(Value::Tensor(_)) | Some(Value::LatentBundle(_)) => 1,
            _ => 0,
        };
        if !values.contains_key(&Key::ConcatMask) || usize::try_from(*index).ok() >= Some(count) {
            return Err(ConditioningError::Invalid(
                "concat-mask index requires and must address concat latent and mask metadata"
                    .to_owned(),
            ));
        }
    }
    if let (Some(latent), Some(Value::Tensor(mask))) = (
        values.get(&Key::ConcatLatentImage),
        values.get(&Key::ConcatMask),
    ) {
        let latent = match latent {
            Value::Tensor(tensor) => Some(tensor),
            Value::LatentBundle(bundle) => bundle.samples().tensor(),
            _ => None,
        };
        if latent.is_none_or(|latent| !metadata_batch_spatial_compatible(latent, mask)) {
            return Err(ConditioningError::Invalid(
                "concat latent and mask metadata have incompatible batch or spatial shape"
                    .to_owned(),
            ));
        }
    }
    match (
        values.get(&Key::VaceFrames),
        values.get(&Key::VaceMask),
        values.get(&Key::VaceStrength),
    ) {
        (None, None, None) => {}
        (
            Some(Value::TensorList(frames)),
            Some(Value::TensorList(masks)),
            Some(Value::FiniteF64List(strengths)),
        ) if frames.len() == masks.len()
            && frames.len() == strengths.len()
            && frames
                .iter()
                .zip(masks)
                .all(|(frame, mask)| metadata_batch_spatial_compatible(frame, mask))
            && strengths.iter().all(|strength| *strength >= 0.0) => {}
        _ => {
            return Err(ConditioningError::Invalid(
                "VACE frame, mask, and strength metadata must have identical nonzero cardinality"
                    .to_owned(),
            ));
        }
    }
    if values.contains_key(&Key::ReferenceMaskFlag)
        && !values.contains_key(&Key::ReferenceMask28Channels)
    {
        return Err(ConditioningError::Invalid(
            "reference-mask flag requires reference-mask metadata".to_owned(),
        ));
    }
    if let (Some(Value::Tensor(driving)), Some(Value::Tensor(reference))) = (
        values.get(&Key::DrivingMask28Channels),
        values.get(&Key::ReferenceMask28Channels),
    ) && driving.descriptor().shape() != reference.descriptor().shape()
    {
        return Err(ConditioningError::Invalid(
            "driving and reference 28-channel masks must have identical shape".to_owned(),
        ));
    }
    match (
        values.get(&Key::SecondsStart),
        values.get(&Key::SecondsTotal),
    ) {
        (None, None) => {}
        (Some(Value::FiniteF64(start)), Some(Value::FiniteF64(total)))
            if *start >= 0.0 && *total >= 0.0 => {}
        _ => {
            return Err(ConditioningError::Invalid(
                "StableAudio timing requires paired finite nonnegative seconds_start and seconds_total"
                    .to_owned(),
            ));
        }
    }
    let bytes = metadata_resident_bytes(values)?;
    if bytes > MAX_CONDITIONING_METADATA_TENSOR_BYTES {
        return Err(ConditioningError::Invalid(
            "conditioning metadata map exceeds the aggregate byte bound".to_owned(),
        ));
    }
    Ok(())
}

fn metadata_batch_spatial_compatible(left: &Tensor, right: &Tensor) -> bool {
    let left = left.descriptor().shape();
    let right = right.descriptor().shape();
    left.first() == right.first() && left.len() == right.len() && left.get(2..) == right.get(2..)
}

fn metadata_resident_bytes(
    values: &BTreeMap<ConditioningMetadataKey, ConditioningMetadataValue>,
) -> Result<u64, ConditioningError> {
    let mut bytes = 0_u64;
    let mut storages = BTreeSet::new();
    for value in values.values() {
        match value {
            ConditioningMetadataValue::ControlnetPooledOutput(Some(tensor))
            | ConditioningMetadataValue::PooledOutput(Some(tensor))
            | ConditioningMetadataValue::ConditioningLyrics(Some(tensor))
            | ConditioningMetadataValue::Tensor(tensor) => {
                add_metadata_tensor_bytes(tensor, &mut storages, &mut bytes)?
            }
            ConditioningMetadataValue::TensorList(tensors)
            | ConditioningMetadataValue::ReferenceAudioTimbreLatents(Some(tensors)) => {
                for tensor in tensors {
                    add_metadata_tensor_bytes(tensor, &mut storages, &mut bytes)?;
                }
            }
            ConditioningMetadataValue::LatentBundle(bundle) => {
                bytes = bytes
                    .checked_add(bundle.projection().resident_bytes())
                    .ok_or(ConditioningError::ResidentBytesOverflow)?;
            }
            ConditioningMetadataValue::ClipVisionOutput(output) => {
                bytes = bytes
                    .checked_add(output.resident_bytes())
                    .ok_or(ConditioningError::ResidentBytesOverflow)?;
            }
            ConditioningMetadataValue::ClipVisionConditioning(items) => {
                for item in items {
                    bytes = bytes
                        .checked_add(item.output.resident_bytes())
                        .ok_or(ConditioningError::ResidentBytesOverflow)?;
                }
            }
            _ => {}
        }
    }
    Ok(bytes)
}

fn add_metadata_tensor_bytes(
    tensor: &Tensor,
    storages: &mut BTreeSet<u64>,
    bytes: &mut u64,
) -> Result<(), ConditioningError> {
    if storages.insert(tensor.storage_id().get()) {
        *bytes = bytes
            .checked_add(tensor.storage_byte_len())
            .ok_or(ConditioningError::ResidentBytesOverflow)?;
    }
    Ok(())
}

fn validate_metadata_value(
    key: ConditioningMetadataKey,
    value: &ConditioningMetadataValue,
) -> Result<(), ConditioningError> {
    use ConditioningMetadataKey as Key;
    use ConditioningMetadataValue as Value;
    let compatible = matches!(
        (key, value),
        (Key::Region, Value::Region(_))
            | (Key::Mask, Value::Tensor(_))
            | (Key::MaskStrength, Value::FiniteF64(_))
            | (Key::SetAreaToBounds, Value::Boolean(_))
            | (Key::TimestepRange, Value::TimestepRange(_))
            | (
                Key::ConcatLatentImage,
                Value::Tensor(_) | Value::LatentBundle(_)
            )
            | (Key::ConcatMask, Value::Tensor(_))
            | (Key::ConcatMaskIndex, Value::Unsigned(_))
            | (Key::Control, Value::Control(_))
            | (Key::ControlApplyToUnconditioned, Value::Boolean(_))
            | (Key::UnclipConditioning, Value::ClipVisionConditioning(_))
            | (Key::ReferenceLatents, Value::TensorList(_))
            | (Key::ClipVisionOutput, Value::ClipVisionOutput(_))
            | (Key::VaceFrames, Value::TensorList(_))
            | (Key::VaceMask, Value::TensorList(_))
            | (Key::VaceStrength, Value::FiniteF64List(_))
            | (Key::CameraConditions, Value::Tensor(_))
            | (Key::TimeDimensionConcat, Value::Tensor(_))
            | (Key::AudioEmbedding, Value::Tensor(_))
            | (Key::ReferenceMotion, Value::Tensor(_))
            | (Key::ControlVideo, Value::Tensor(_))
            | (Key::PoseVideoLatent, Value::Tensor(_))
            | (Key::FaceVideoPixels, Value::Tensor(_))
            | (Key::DrivingMask28Channels, Value::Tensor(_))
            | (Key::ReferenceMask28Channels, Value::Tensor(_))
            | (Key::ReferenceMaskFlag, Value::Boolean(_))
            | (Key::SecondsStart, Value::FiniteF64(_))
            | (Key::SecondsTotal, Value::FiniteF64(_))
            | (Key::FrameRate, Value::FiniteF64(_))
            | (Key::CrossAttentionControlnet, Value::Tensor(_))
            | (
                Key::PooledOutputControlnet,
                Value::ControlnetPooledOutput(_)
            )
            | (Key::PooledOutput, Value::PooledOutput(_))
            | (Key::ConditioningLyrics, Value::ConditioningLyrics(_))
            | (
                Key::ReferenceAudioTimbreLatents,
                Value::ReferenceAudioTimbreLatents(_)
            )
    );
    if !compatible {
        return Err(ConditioningError::Invalid(format!(
            "conditioning metadata value is incompatible with key {key:?}"
        )));
    }
    match (key, value) {
        (Key::CrossAttentionControlnet, Value::Tensor(tensor)) => {
            validate_metadata_tensor_contract(
                tensor,
                "Controlnet text cross-attention",
                &[3],
                None,
                false,
            )?;
        }
        (Key::PooledOutputControlnet, Value::ControlnetPooledOutput(Some(tensor))) => {
            validate_metadata_tensor_contract(
                tensor,
                "Controlnet text pooled output",
                &[2],
                None,
                false,
            )?;
        }
        (Key::Mask, Value::Tensor(tensor)) => {
            validate_metadata_tensor_contract(tensor, "mask", &[2, 3, 4], None, true)?;
        }
        (Key::ConcatLatentImage, Value::Tensor(tensor)) => {
            validate_metadata_tensor_contract(tensor, "concat latent", &[4, 5], None, false)?;
        }
        (Key::ConcatLatentImage, Value::LatentBundle(bundle)) => {
            if bundle.samples().tensor().is_none() {
                return Err(ConditioningError::Invalid(
                    "concat latent metadata requires a single-tensor latent bundle".to_owned(),
                ));
            }
        }
        (Key::ConcatMask, Value::Tensor(tensor)) => {
            validate_metadata_tensor_contract(tensor, "concat mask", &[4, 5], Some((1, 1)), true)?;
        }
        (Key::ReferenceLatents, Value::TensorList(tensors)) => {
            validate_metadata_tensor_list_contract(
                tensors,
                "reference latents",
                &[4, 5],
                None,
                false,
            )?;
        }
        (Key::VaceFrames, Value::TensorList(tensors)) => {
            validate_metadata_tensor_list_contract(tensors, "VACE frames", &[4, 5], None, false)?;
        }
        (Key::VaceMask, Value::TensorList(tensors)) => {
            validate_metadata_tensor_list_contract(
                tensors,
                "VACE masks",
                &[4, 5],
                Some((1, 1)),
                true,
            )?;
        }
        (Key::CameraConditions, Value::Tensor(tensor)) => {
            validate_metadata_tensor_contract(
                tensor,
                "camera conditions",
                &[2, 3, 4],
                None,
                false,
            )?;
        }
        (Key::TimeDimensionConcat, Value::Tensor(tensor)) => {
            validate_metadata_tensor_contract(tensor, "time-dimension concat", &[2], None, false)?;
        }
        (Key::AudioEmbedding, Value::Tensor(tensor)) => {
            validate_metadata_tensor_contract(tensor, "audio embedding", &[3, 4], None, false)?;
        }
        (Key::ReferenceMotion, Value::Tensor(tensor)) => {
            validate_metadata_tensor_contract(tensor, "reference motion", &[4, 5], None, false)?;
        }
        (Key::ControlVideo | Key::PoseVideoLatent, Value::Tensor(tensor)) => {
            validate_metadata_tensor_contract(tensor, "video latent", &[5], None, false)?;
        }
        (Key::FaceVideoPixels, Value::Tensor(tensor)) => {
            validate_metadata_tensor_contract(
                tensor,
                "face video pixels",
                &[5],
                Some((4, 3)),
                false,
            )?;
        }
        (Key::DrivingMask28Channels | Key::ReferenceMask28Channels, Value::Tensor(tensor)) => {
            validate_metadata_tensor_contract(
                tensor,
                "28-channel video mask",
                &[5],
                Some((4, 28)),
                true,
            )?;
        }
        _ => {}
    }
    match value {
        Value::ReferenceAudioTimbreLatents(Some(tensors)) => {
            if tensors.len() > MAX_CONDITIONING_METADATA_LIST_ITEMS {
                return Err(ConditioningError::Invalid(format!(
                    "reference audio timbre metadata exceeds {MAX_CONDITIONING_METADATA_LIST_ITEMS} items"
                )));
            }
            for tensor in tensors {
                validate_metadata_tensor(tensor)?;
            }
            Ok(())
        }
        Value::ReferenceAudioTimbreLatents(None) => Ok(()),
        Value::Tensor(tensor) => validate_metadata_tensor(tensor),
        Value::TensorList(tensors) => {
            if tensors.is_empty() || tensors.len() > MAX_CONDITIONING_METADATA_LIST_ITEMS {
                return Err(ConditioningError::Invalid(format!(
                    "conditioning metadata tensor lists must contain 1..={MAX_CONDITIONING_METADATA_LIST_ITEMS} items"
                )));
            }
            let mut total_bytes = 0_u64;
            for tensor in tensors {
                validate_metadata_tensor(tensor)?;
                total_bytes = total_bytes
                    .checked_add(tensor.storage_byte_len())
                    .ok_or(ConditioningError::ResidentBytesOverflow)?;
            }
            if total_bytes > MAX_CONDITIONING_METADATA_TENSOR_BYTES {
                return Err(ConditioningError::Invalid(
                    "conditioning metadata tensor list exceeds the supported byte bound".to_owned(),
                ));
            }
            Ok(())
        }
        Value::LatentBundle(bundle) => {
            bundle
                .validate_retained()
                .map_err(|error| ConditioningError::Invalid(error.to_string()))?;
            if bundle.projection().resident_bytes() > MAX_CONDITIONING_METADATA_TENSOR_BYTES {
                return Err(ConditioningError::Invalid(
                    "conditioning metadata latent bundle exceeds the supported byte bound"
                        .to_owned(),
                ));
            }
            Ok(())
        }
        Value::ClipVisionOutput(output) => {
            output
                .validate()
                .map_err(|error| ConditioningError::Invalid(error.to_string()))?;
            if output.resident_bytes() > MAX_CONDITIONING_METADATA_TENSOR_BYTES {
                return Err(ConditioningError::Invalid(
                    "conditioning metadata CLIP vision output exceeds the supported byte bound"
                        .to_owned(),
                ));
            }
            Ok(())
        }
        Value::ClipVisionConditioning(values) => {
            if values.is_empty() || values.len() > MAX_CONDITIONING_METADATA_LIST_ITEMS {
                return Err(ConditioningError::Invalid(
                    "CLIP vision conditioning list is empty or oversized".to_owned(),
                ));
            }
            let mut total_bytes = 0_u64;
            for item in values {
                item.output
                    .validate()
                    .map_err(|error| ConditioningError::Invalid(error.to_string()))?;
                total_bytes = total_bytes
                    .checked_add(item.output.resident_bytes())
                    .ok_or(ConditioningError::ResidentBytesOverflow)?;
            }
            if total_bytes > MAX_CONDITIONING_METADATA_TENSOR_BYTES {
                return Err(ConditioningError::Invalid(
                    "CLIP vision conditioning list exceeds the supported byte bound".to_owned(),
                ));
            }
            Ok(())
        }
        Value::FiniteF64(value) => {
            if value.is_finite() {
                Ok(())
            } else {
                Err(ConditioningError::Invalid(
                    "conditioning metadata scalar must be finite".to_owned(),
                ))
            }
        }
        Value::FiniteF64List(values) => {
            if values.is_empty()
                || values.len() > MAX_CONDITIONING_METADATA_LIST_ITEMS
                || values.iter().any(|value| !value.is_finite())
            {
                return Err(ConditioningError::Invalid(
                    "conditioning metadata scalar list is empty, oversized, or non-finite"
                        .to_owned(),
                ));
            }
            Ok(())
        }
        Value::Region(region) => region.validate(),
        Value::TimestepRange(range) => {
            ConditioningTimestepRange::source(range.start_percent, range.end_percent).map(drop)
        }
        Value::Control(binding) => ConditioningControlBinding::checked(
            binding.execution_digest.clone(),
            binding.apply_to_unconditioned,
        )
        .map(drop),
        Value::ControlnetPooledOutput(Some(tensor))
        | Value::PooledOutput(Some(tensor))
        | Value::ConditioningLyrics(Some(tensor)) => validate_metadata_tensor(tensor),
        Value::ControlnetPooledOutput(None)
        | Value::PooledOutput(None)
        | Value::ConditioningLyrics(None)
        | Value::Unsigned(_)
        | Value::Boolean(_) => Ok(()),
    }
}

fn validate_metadata_tensor_list_contract(
    tensors: &[Tensor],
    subject: &str,
    ranks: &[usize],
    channel: Option<(usize, u64)>,
    allow_mask_dtype: bool,
) -> Result<(), ConditioningError> {
    for tensor in tensors {
        validate_metadata_tensor_contract(tensor, subject, ranks, channel, allow_mask_dtype)?;
    }
    if let Some(first) = tensors.first() {
        let first = first.descriptor();
        if tensors.iter().skip(1).any(|tensor| {
            tensor.descriptor().rank() != first.rank()
                || tensor.descriptor().dtype() != first.dtype()
                || tensor.descriptor().shape().first() != first.shape().first()
        }) {
            return Err(ConditioningError::Invalid(format!(
                "{subject} list items must have matching rank, dtype, and batch cardinality"
            )));
        }
    }
    Ok(())
}

fn validate_metadata_tensor_contract(
    tensor: &Tensor,
    subject: &str,
    ranks: &[usize],
    channel: Option<(usize, u64)>,
    allow_mask_dtype: bool,
) -> Result<(), ConditioningError> {
    validate_metadata_tensor(tensor)?;
    let descriptor = tensor.descriptor();
    if !ranks.contains(&descriptor.rank()) {
        return Err(ConditioningError::Invalid(format!(
            "{subject} metadata has an unsupported rank"
        )));
    }
    let floating = matches!(
        descriptor.dtype(),
        DType::F64 | DType::F32 | DType::F16 | DType::Bf16
    );
    let mask = matches!(descriptor.dtype(), DType::Bool | DType::U8);
    if !floating && !(allow_mask_dtype && mask) {
        return Err(ConditioningError::Invalid(format!(
            "{subject} metadata has an unsupported dtype"
        )));
    }
    if let Some((axis, expected)) = channel
        && descriptor.shape().get(axis).copied() != Some(expected)
    {
        return Err(ConditioningError::Invalid(format!(
            "{subject} metadata has an incompatible channel layout"
        )));
    }
    Ok(())
}

fn validate_metadata_tensor(tensor: &Tensor) -> Result<(), ConditioningError> {
    if tensor.descriptor().rank() == 0 || tensor.descriptor().shape().contains(&0) {
        return Err(ConditioningError::Invalid(
            "conditioning metadata tensors must have nonzero shape".to_owned(),
        ));
    }
    if tensor.storage_byte_len() > MAX_CONDITIONING_METADATA_TENSOR_BYTES {
        return Err(ConditioningError::Invalid(format!(
            "conditioning metadata tensor exceeds {MAX_CONDITIONING_METADATA_TENSOR_BYTES} bytes"
        )));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct ConditioningMask {
    tensor: Tensor,
    strength: f32,
    feather: Vec<u64>,
    set_region_to_nonzero_bounds: bool,
}

impl ConditioningMask {
    pub fn new(
        tensor: Tensor,
        strength: f32,
        feather: Vec<u64>,
        set_region_to_nonzero_bounds: bool,
    ) -> Result<Self, ConditioningError> {
        let mask = Self {
            tensor,
            strength,
            feather,
            set_region_to_nonzero_bounds,
        };
        mask.validate()?;
        Ok(mask)
    }

    fn validate(&self) -> Result<(), ConditioningError> {
        if !matches!(
            self.tensor.descriptor().dtype(),
            DType::F32 | DType::F16 | DType::Bf16
        ) {
            return Err(ConditioningError::Invalid(
                "conditioning mask must use a floating-point dtype".to_owned(),
            ));
        }
        if self.tensor.descriptor().rank() == 0 || self.tensor.descriptor().shape().contains(&0) {
            return Err(ConditioningError::Invalid(
                "conditioning mask must have nonzero spatial dimensions".to_owned(),
            ));
        }
        if !self.strength.is_finite() {
            return Err(ConditioningError::Invalid(
                "conditioning mask strength must be finite".to_owned(),
            ));
        }
        if self.feather.is_empty() {
            return Err(ConditioningError::Invalid(
                "conditioning mask feather rank must be nonzero".to_owned(),
            ));
        }
        if self.feather.iter().any(|value| *value != 0) {
            return Err(ConditioningError::Invalid(
                "conditioning mask feather must be zero because source feathering applies only to unmasked areas"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn strength(&self) -> f32 {
        self.strength
    }

    pub fn feather(&self) -> &[u64] {
        &self.feather
    }

    pub fn set_region_to_nonzero_bounds(&self) -> bool {
        self.set_region_to_nonzero_bounds
    }

    fn normalized_to_target(
        &self,
        target: &TensorDescriptor,
        target_dimensions: &[u64],
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Self, ConditioningError> {
        context.cancellation.check()?;
        if self.tensor.descriptor().device() != target.device()
            || self.tensor.descriptor().stream() != target.stream()
        {
            return Err(ConditioningError::Invalid(
                "conditioning mask must use the target device and stream".to_owned(),
            ));
        }
        let spatial_rank = target_dimensions.len();
        let mask_rank = self.tensor.descriptor().rank();
        let mut tensor = if mask_rank == spatial_rank {
            tensor_unsqueeze_exact_native(&self.tensor, 0, context.cancellation)?
        } else if mask_rank == spatial_rank + 1 {
            self.tensor.clone()
        } else {
            return Err(ConditioningError::Invalid(
                "conditioning mask must be spatial-only or [batch, spatial...]".to_owned(),
            ));
        };
        if tensor.descriptor().shape()[1..] == *target_dimensions {
            return Ok(Self {
                tensor,
                strength: self.strength,
                feather: self.feather.clone(),
                set_region_to_nonzero_bounds: self.set_region_to_nonzero_bounds,
            });
        }
        if spatial_rank < 2 {
            return Err(ConditioningError::Invalid(
                "one-dimensional conditioning masks cannot be bilinearly resized".to_owned(),
            ));
        }
        let dtype = tensor.descriptor().dtype();
        let resized_input = if dtype == DType::F32 {
            tensor
        } else {
            cast_to_with_context_exact_native(
                backend,
                &tensor,
                DType::F32,
                target.device(),
                false,
                false,
                context,
            )?
        };
        tensor = resize_with_context_exact_native(
            backend,
            &resized_input,
            target_dimensions[spatial_rank - 2],
            target_dimensions[spatial_rank - 1],
            ResizeMode::Bilinear,
            false,
            context,
        )?;
        if dtype != DType::F32 {
            tensor = cast_to_with_context_exact_native(
                backend,
                &tensor,
                dtype,
                target.device(),
                false,
                false,
                context,
            )?;
        }
        if tensor.descriptor().shape()[1..] != *target_dimensions {
            return Err(ConditioningError::Invalid(
                "conditioning mask non-spatial dimensions do not match the target".to_owned(),
            ));
        }
        context.cancellation.check()?;
        Ok(Self {
            tensor,
            strength: self.strength,
            feather: self.feather.clone(),
            set_region_to_nonzero_bounds: self.set_region_to_nonzero_bounds,
        })
    }

    fn nonzero_region(
        &self,
        target_dimensions: &[u64],
        cancellation: &sim_types::CancellationToken,
    ) -> Result<ResolvedConditioningRegion, ConditioningError> {
        cancellation.check()?;
        if self.tensor.descriptor().device() != DeviceId::CPU {
            return Err(ConditioningError::UnreadableDevice(
                self.tensor.descriptor().device(),
            ));
        }
        let rank = self.tensor.descriptor().rank();
        let spatial_rank = target_dimensions.len();
        let spatial_shape = if rank == spatial_rank {
            self.tensor.descriptor().shape()
        } else if rank == spatial_rank + 1 {
            self.tensor
                .descriptor()
                .shape()
                .get(1..)
                .ok_or(ConditioningError::ShapeOverflow("mask spatial shape"))?
        } else {
            return Err(ConditioningError::Invalid(
                "conditioning mask must be spatial-only or [batch, spatial...]".to_owned(),
            ));
        };
        if spatial_shape != target_dimensions {
            return Err(ConditioningError::Invalid(
                "conditioning mask spatial dimensions must match the target".to_owned(),
            ));
        }
        let mut minimum = vec![u64::MAX; spatial_rank];
        let mut maximum = vec![0_u64; spatial_rank];
        let mut any_nonzero = false;
        let element_count = self.tensor.descriptor().element_count()?;
        for linear_index in 0..element_count {
            if linear_index.is_multiple_of(1024) {
                cancellation.check()?;
            }
            let value = self
                .tensor
                .descriptor()
                .dtype()
                .decode_scalar(self.tensor.linear_element_bytes(linear_index)?)?;
            if !value.is_nonzero() {
                continue;
            }
            any_nonzero = true;
            let mut remainder = linear_index;
            for axis in (0..rank).rev() {
                let dimension = self.tensor.descriptor().shape()[axis];
                let coordinate = remainder % dimension;
                remainder /= dimension;
                if rank == spatial_rank || axis > 0 {
                    let spatial_axis = if rank == spatial_rank { axis } else { axis - 1 };
                    minimum[spatial_axis] = minimum[spatial_axis].min(coordinate);
                    maximum[spatial_axis] = maximum[spatial_axis].max(coordinate);
                }
            }
        }
        cancellation.check()?;
        if !any_nonzero {
            let sizes = target_dimensions
                .iter()
                .map(|dimension| (*dimension).min(8))
                .collect::<Vec<_>>();
            return ResolvedConditioningRegion::checked(
                sizes,
                vec![0; spatial_rank],
                target_dimensions.to_vec(),
            );
        }
        let sizes = minimum
            .iter()
            .zip(&maximum)
            .map(|(minimum, maximum)| maximum - minimum + 1)
            .collect::<Vec<_>>();
        ResolvedConditioningRegion::checked(sizes, minimum, target_dimensions.to_vec())
    }

    pub fn resolve(
        &self,
        target: &TensorDescriptor,
        region: &ResolvedConditioningRegion,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<ResolvedConditioningMask, ConditioningError> {
        context.cancellation.check()?;
        if target.rank() < 3 || target.shape()[2..] != *region.target_dimensions() {
            return Err(ConditioningError::Invalid(
                "conditioning mask target and region dimensions do not match".to_owned(),
            ));
        }
        let spatial_rank = region.sizes().len();
        let normalized =
            self.normalized_to_target(target, region.target_dimensions(), backend, context)?;
        if self.feather.len() != spatial_rank {
            return Err(ConditioningError::Invalid(
                "conditioning mask feather rank must match the spatial rank".to_owned(),
            ));
        }
        let mut mask = normalized.tensor;
        for (dimension, (&offset, &size)) in region.offsets().iter().zip(region.sizes()).enumerate()
        {
            let axis = i64::try_from(dimension + 1)
                .map_err(|_| ConditioningError::ShapeOverflow("mask crop axis"))?;
            let offset = i64::try_from(offset)
                .map_err(|_| ConditioningError::ShapeOverflow("mask crop offset"))?;
            mask = narrow_method_exact_native(&mask, axis, offset, size, context.cancellation)?;
        }
        let batch_size = target.shape()[0];
        let channel_count = target.shape()[1];
        mask = repeat_to_batch_size(&mask, batch_size, backend, context)?;
        mask = tensor_unsqueeze_exact_native(&mask, 1, context.cancellation)?;
        if channel_count != 1 {
            let mut repeats = vec![1_i64; mask.descriptor().rank()];
            *repeats
                .get_mut(1)
                .ok_or(ConditioningError::ShapeOverflow("mask channel repeat"))? =
                i64::try_from(channel_count)
                    .map_err(|_| ConditioningError::ShapeOverflow("mask channel count"))?;
            mask = tensor_repeat_with_context_exact_native(backend, &mask, &repeats, context)?;
        }
        for (&feather, &size) in self.feather.iter().zip(region.sizes()) {
            if feather > size.saturating_sub(1) / 2 {
                return Err(ConditioningError::Invalid(
                    "conditioning mask feather exceeds half the resolved region".to_owned(),
                ));
            }
        }
        context.cancellation.check()?;
        Ok(ResolvedConditioningMask {
            tensor: mask,
            strength: self.strength,
            feather: self.feather.clone(),
            region: region.clone(),
            set_region_to_nonzero_bounds: self.set_region_to_nonzero_bounds,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ResolvedConditioningMask {
    tensor: Tensor,
    strength: f32,
    feather: Vec<u64>,
    region: ResolvedConditioningRegion,
    set_region_to_nonzero_bounds: bool,
}

impl ResolvedConditioningMask {
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn strength(&self) -> f32 {
        self.strength
    }

    pub fn set_region_to_nonzero_bounds(&self) -> bool {
        self.set_region_to_nonzero_bounds
    }

    pub fn feather_weight(&self, spatial_indices: &[u64]) -> Result<f32, ConditioningError> {
        if spatial_indices.len() != self.region.sizes().len() {
            return Err(ConditioningError::Invalid(
                "feather index rank must match the region rank".to_owned(),
            ));
        }
        let mut weight = self.strength;
        for ((index, size), feather) in spatial_indices
            .iter()
            .zip(self.region.sizes())
            .zip(&self.feather)
        {
            if *index >= *size {
                return Err(ConditioningError::Invalid(
                    "feather index lies outside the resolved region".to_owned(),
                ));
            }
            if *feather == 0 {
                continue;
            }
            let divisor = (*feather + 1) as f32;
            let leading = (*index + 1) as f32 / divisor;
            let trailing = (*size - *index) as f32 / divisor;
            weight *= leading.min(trailing).min(1.0);
        }
        Ok(weight)
    }
}

#[derive(Clone, Debug)]
pub struct ConditioningEntryOptions {
    pub strength: f64,
    pub region: Option<ConditioningRegion>,
    pub mask: Option<ConditioningMask>,
    pub window: ConditioningWindow,
    pub default_region: bool,
}

impl Default for ConditioningEntryOptions {
    fn default() -> Self {
        Self {
            strength: 1.0,
            region: None,
            mask: None,
            window: ConditioningWindow::full(),
            default_region: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConditioningEntry {
    identifier: String,
    value: ConditioningValue,
    options: ConditioningEntryOptions,
    references: ConditioningReferences,
    metadata: ConditioningMetadataMap,
}

impl ConditioningEntry {
    pub fn checked(
        identifier: impl Into<String>,
        value: ConditioningValue,
        options: ConditioningEntryOptions,
    ) -> Result<Self, ConditioningError> {
        Self::checked_with_references(
            identifier,
            value,
            options,
            ConditioningReferences::default(),
        )
    }

    pub fn checked_with_references(
        identifier: impl Into<String>,
        value: ConditioningValue,
        options: ConditioningEntryOptions,
        references: ConditioningReferences,
    ) -> Result<Self, ConditioningError> {
        Self::checked_with_metadata(identifier, value, options, references, Default::default())
    }

    pub fn checked_with_metadata(
        identifier: impl Into<String>,
        value: ConditioningValue,
        options: ConditioningEntryOptions,
        references: ConditioningReferences,
        metadata: ConditioningMetadataMap,
    ) -> Result<Self, ConditioningError> {
        let identifier = identifier.into();
        let entry = Self {
            identifier,
            value,
            options,
            references,
            metadata,
        };
        entry.validate()?;
        Ok(entry)
    }

    fn validate(&self) -> Result<(), ConditioningError> {
        validate_identifier("conditioning entry", &self.identifier)?;
        if !self.options.strength.is_finite() {
            return Err(ConditioningError::Invalid(
                "conditioning entry strength must be finite".to_owned(),
            ));
        }
        self.value.validate()?;
        if let Some(region) = &self.options.region {
            region.validate()?;
        }
        if let Some(mask) = &self.options.mask {
            mask.validate()?;
        }
        self.options.window.validate()?;
        self.references.validate()?;
        self.metadata.validate()?;
        if self.options.default_region
            && (self.options.region.is_some() || self.options.mask.is_some())
        {
            return Err(ConditioningError::Invalid(
                "default-region conditioning cannot own an explicit region or mask".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    pub fn value(&self) -> &ConditioningValue {
        &self.value
    }

    pub fn options(&self) -> &ConditioningEntryOptions {
        &self.options
    }

    pub fn references(&self) -> &ConditioningReferences {
        &self.references
    }

    pub fn metadata(&self) -> &ConditioningMetadataMap {
        &self.metadata
    }

    pub fn with_metadata(
        &self,
        metadata: ConditioningMetadataMap,
    ) -> Result<Self, ConditioningError> {
        Self::checked_with_metadata(
            self.identifier.clone(),
            self.value.clone(),
            self.options.clone(),
            self.references.clone(),
            metadata,
        )
    }

    pub fn resolve(
        &self,
        target: &TensorDescriptor,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<ResolvedConditioningEntry, ConditioningError> {
        context.cancellation.check()?;
        if target.rank() < 3 || target.shape()[0] == 0 || target.shape()[1] == 0 {
            return Err(ConditioningError::Invalid(
                "conditioning target must have nonzero batch, channel, and spatial dimensions"
                    .to_owned(),
            ));
        }
        let target_dimensions = target.shape()[2..].to_vec();
        let mask = self
            .options
            .mask
            .as_ref()
            .map(|mask| mask.normalized_to_target(target, &target_dimensions, backend, context))
            .transpose()?;
        let mut region = match &self.options.region {
            Some(region) => region.resolve(&target_dimensions)?,
            None => ResolvedConditioningRegion::full(target_dimensions.clone())?,
        };
        if let Some(mask) = &mask
            && mask.set_region_to_nonzero_bounds()
        {
            let mask_region = mask.nonzero_region(&target_dimensions, context.cancellation)?;
            region = intersect_regions(&region, &mask_region)?;
        }
        let crop_value = self.options.region.is_some()
            || self
                .options
                .mask
                .as_ref()
                .is_some_and(ConditioningMask::set_region_to_nonzero_bounds);
        let value = self.value.process(
            target.shape()[0],
            crop_value.then_some(&region),
            backend,
            context,
        )?;
        let mask = mask
            .as_ref()
            .map(|mask| mask.resolve(target, &region, backend, context))
            .transpose()?;
        context.cancellation.check()?;
        Ok(ResolvedConditioningEntry {
            identifier: self.identifier.clone(),
            value,
            strength: self.options.strength,
            region,
            mask,
            window: self.options.window,
            default_region: self.options.default_region,
            references: self.references.clone(),
            metadata: self.metadata.clone(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct ResolvedConditioningEntry {
    identifier: String,
    value: ConditioningValue,
    strength: f64,
    region: ResolvedConditioningRegion,
    mask: Option<ResolvedConditioningMask>,
    window: ConditioningWindow,
    default_region: bool,
    references: ConditioningReferences,
    metadata: ConditioningMetadataMap,
}

impl ResolvedConditioningEntry {
    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    pub fn value(&self) -> &ConditioningValue {
        &self.value
    }

    pub fn strength(&self) -> f64 {
        self.strength
    }

    pub fn strength_for_cpu_f32(&self) -> Result<f32, ConditioningError> {
        let strength = self.strength as f32;
        if !strength.is_finite() {
            return Err(ConditioningError::Invalid(
                "conditioning strength cannot be represented as a finite CPU F32 scalar".to_owned(),
            ));
        }
        Ok(strength)
    }

    pub fn region(&self) -> &ResolvedConditioningRegion {
        &self.region
    }

    pub fn mask(&self) -> Option<&ResolvedConditioningMask> {
        self.mask.as_ref()
    }

    pub fn window(&self) -> ConditioningWindow {
        self.window
    }

    pub fn is_default_region(&self) -> bool {
        self.default_region
    }

    pub fn references(&self) -> &ConditioningReferences {
        &self.references
    }

    pub fn metadata(&self) -> &ConditioningMetadataMap {
        &self.metadata
    }

    pub fn contribution_weight(
        &self,
        spatial_indices: &[u64],
        mask_value: Option<f32>,
    ) -> Result<f32, ConditioningError> {
        if spatial_indices.len() != self.region.sizes().len() {
            return Err(ConditioningError::Invalid(
                "conditioning weight index rank must match the region rank".to_owned(),
            ));
        }
        for (index, size) in spatial_indices.iter().zip(self.region.sizes()) {
            if index >= size {
                return Err(ConditioningError::Invalid(
                    "conditioning weight index lies outside the resolved region".to_owned(),
                ));
            }
        }
        let local_weight = match (&self.mask, mask_value) {
            (Some(mask), Some(value)) if value.is_finite() => {
                mask.feather_weight(spatial_indices)? * value
            }
            (None, None) => area_only_feather_weight(&self.region, spatial_indices)?,
            (Some(_), Some(_)) => {
                return Err(ConditioningError::Invalid(
                    "conditioning mask value must be finite".to_owned(),
                ));
            }
            _ => {
                return Err(ConditioningError::Invalid(
                    "conditioning mask value presence does not match the resolved entry".to_owned(),
                ));
            }
        };
        let weight = self.strength_for_cpu_f32()? * local_weight;
        if !weight.is_finite() {
            return Err(ConditioningError::Invalid(
                "conditioning contribution weight must be finite".to_owned(),
            ));
        }
        Ok(weight)
    }
}

pub fn default_region_residual_weight(covered_weight: f32) -> Result<f32, ConditioningError> {
    if !covered_weight.is_finite() {
        return Err(ConditioningError::Invalid(
            "default-region covered weight must be finite".to_owned(),
        ));
    }
    Ok((1.0 - covered_weight).max(0.0))
}

#[derive(Clone, Debug)]
pub struct ConditioningSet {
    identity: ConditioningIdentity,
    entries: Vec<ConditioningEntry>,
    digest: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConditioningTensorResidentAllocation {
    storage_id: StorageId,
    resident_bytes: u64,
}

impl ConditioningTensorResidentAllocation {
    pub const fn storage_id(&self) -> StorageId {
        self.storage_id
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConditioningResidentParts {
    owned_bytes: u64,
    tensor_allocations: Vec<ConditioningTensorResidentAllocation>,
}

impl ConditioningResidentParts {
    pub const fn owned_bytes(&self) -> u64 {
        self.owned_bytes
    }

    pub fn tensor_allocations(&self) -> &[ConditioningTensorResidentAllocation] {
        &self.tensor_allocations
    }

    pub fn resident_bytes(&self) -> Result<u64, ConditioningError> {
        self.tensor_allocations
            .iter()
            .try_fold(self.owned_bytes, |bytes, allocation| {
                bytes
                    .checked_add(allocation.resident_bytes)
                    .ok_or(ConditioningError::ResidentBytesOverflow)
            })
    }
}

impl ConditioningSet {
    pub fn checked(
        identity: ConditioningIdentity,
        entries: Vec<ConditioningEntry>,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, ConditioningError> {
        cancellation.check()?;
        validate_conditioning_set(&identity, &entries, || {
            cancellation.check().map_err(ConditioningError::from)
        })?;
        let digest = hash_set(&identity, &entries, cancellation)?;
        Ok(Self {
            identity,
            entries,
            digest,
        })
    }

    pub fn identity(&self) -> &ConditioningIdentity {
        &self.identity
    }

    pub fn entries(&self) -> &[ConditioningEntry] {
        &self.entries
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    pub fn validate(&self) -> Result<(), ConditioningError> {
        validate_conditioning_set(&self.identity, &self.entries, || Ok(()))?;
        let mut checkpoint = || Ok(());
        if hash_set_with_checkpoint(&self.identity, &self.entries, &mut checkpoint)? != self.digest
        {
            return Err(ConditioningError::Invalid(
                "conditioning set digest no longer matches its contents".to_owned(),
            ));
        }
        self.resident_bytes()?;
        Ok(())
    }

    pub fn resident_bytes(&self) -> Result<u64, ConditioningError> {
        self.resident_parts()?.resident_bytes()
    }

    pub fn resident_parts(&self) -> Result<ConditioningResidentParts, ConditioningError> {
        let mut accounting = ConditioningResidentAccounting::default();
        accounting.add_set(self)?;
        accounting.resident_bytes()?;
        Ok(accounting.into_parts())
    }

    pub fn combined_resident_bytes(sets: &[&Self]) -> Result<u64, ConditioningError> {
        let mut accounting = ConditioningResidentAccounting::default();
        for set in sets {
            accounting.add_set(set)?;
        }
        accounting.resident_bytes()
    }

    pub fn with_controlnet_text_encoding(
        &self,
        cross_attention: &Tensor,
        pooled: Option<&Tensor>,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, ConditioningError> {
        cancellation.check()?;
        self.validate()?;
        validate_metadata_value(
            ConditioningMetadataKey::CrossAttentionControlnet,
            &ConditioningMetadataValue::Tensor(cross_attention.clone()),
        )?;
        let pooled = ConditioningMetadataValue::ControlnetPooledOutput(pooled.cloned());
        validate_metadata_value(ConditioningMetadataKey::PooledOutputControlnet, &pooled)?;
        let mut metadata = Vec::new();
        metadata
            .try_reserve_exact(self.entries.len())
            .map_err(|_| ConditioningError::ShapeOverflow("Controlnet text metadata entries"))?;
        for entry in &self.entries {
            cancellation.check()?;
            let mut values = entry.metadata().clone();
            values.insert(
                ConditioningMetadataKey::CrossAttentionControlnet,
                ConditioningMetadataValue::Tensor(cross_attention.clone()),
            )?;
            values.insert(
                ConditioningMetadataKey::PooledOutputControlnet,
                pooled.clone(),
            )?;
            metadata.push(values);
        }
        self.with_entry_metadata(metadata, cancellation)
    }

    pub fn with_entry_metadata(
        &self,
        metadata: Vec<ConditioningMetadataMap>,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<Self, ConditioningError> {
        cancellation.check()?;
        if metadata.len() != self.entries.len() {
            return Err(ConditioningError::Invalid(
                "conditioning metadata cardinality must match the entry cardinality".to_owned(),
            ));
        }
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(self.entries.len())
            .map_err(|_| ConditioningError::ShapeOverflow("conditioning metadata entries"))?;
        for (entry, metadata) in self.entries.iter().zip(metadata) {
            cancellation.check()?;
            entries.push(entry.with_metadata(metadata)?);
        }
        Self::checked(self.identity.clone(), entries, cancellation)
    }

    pub fn resolve(
        &self,
        target: &TensorDescriptor,
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<ResolvedConditioningEntry>, ConditioningError> {
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(self.entries.len())
            .map_err(|_| ConditioningError::ShapeOverflow("resolved conditioning entries"))?;
        for entry in &self.entries {
            context.cancellation.check()?;
            entries.push(entry.resolve(target, backend, context)?);
        }
        context.cancellation.check()?;
        Ok(entries)
    }
}

fn validate_conditioning_set<Checkpoint>(
    identity: &ConditioningIdentity,
    entries: &[ConditioningEntry],
    mut checkpoint: Checkpoint,
) -> Result<(), ConditioningError>
where
    Checkpoint: FnMut() -> Result<(), ConditioningError>,
{
    checkpoint()?;
    identity.validate()?;
    if entries.len() > MAX_CONDITIONING_ENTRIES {
        return Err(ConditioningError::Invalid(format!(
            "conditioning set contains more than {MAX_CONDITIONING_ENTRIES} entries"
        )));
    }
    let mut identifiers = BTreeSet::new();
    for entry in entries {
        checkpoint()?;
        entry.validate()?;
        if !identifiers.insert(entry.identifier()) {
            return Err(ConditioningError::Invalid(format!(
                "conditioning entry identifier is duplicated: {}",
                entry.identifier()
            )));
        }
    }
    checkpoint()?;
    Ok(())
}

#[derive(Default)]
struct ConditioningResidentAccounting {
    owned_bytes: u64,
    sets: BTreeSet<usize>,
    metadata_resources: BTreeSet<usize>,
    storages: BTreeMap<u64, (StorageId, u64)>,
}

impl ConditioningResidentAccounting {
    fn add_set(&mut self, set: &ConditioningSet) -> Result<(), ConditioningError> {
        if !self.sets.insert(std::ptr::from_ref(set) as usize) {
            return Ok(());
        }
        self.add_inline::<ConditioningSet>()?;
        self.add_bytes(set.identity.namespace.capacity())?;
        self.add_u64(
            set.identity
                .model_family
                .owned_resident_bytes()
                .ok_or(ConditioningError::ResidentBytesOverflow)?,
        )?;
        self.add_u64(
            set.identity
                .latent_format
                .owned_resident_bytes()
                .ok_or(ConditioningError::ResidentBytesOverflow)?,
        )?;
        self.add_allocation::<ConditioningEntry>(set.entries.capacity())?;
        self.add_bytes(set.digest.capacity())?;
        for entry in &set.entries {
            self.add_entry_allocations(entry)?;
        }
        Ok(())
    }

    fn add_entry_allocations(
        &mut self,
        entry: &ConditioningEntry,
    ) -> Result<(), ConditioningError> {
        self.add_bytes(entry.identifier.capacity())?;
        self.add_value_allocations(&entry.value)?;
        if let Some(region) = &entry.options.region {
            self.add_region_allocations(region)?;
        }
        if let Some(mask) = &entry.options.mask {
            self.add_tensor_storage(mask.tensor())?;
            self.add_allocation::<u64>(mask.feather.capacity())?;
        }
        if let Some(control) = &entry.references.control {
            self.add_bytes(control.0.capacity())?;
        }
        self.add_allocation::<ConditioningHookReference>(entry.references.hooks.capacity())?;
        for hook in &entry.references.hooks {
            self.add_bytes(hook.0.capacity())?;
        }
        self.add_allocation::<(ConditioningMetadataKey, ConditioningMetadataValue)>(
            entry.metadata.values.len(),
        )?;
        for value in entry.metadata.values.values() {
            self.add_metadata_value_allocations(value)?;
        }
        Ok(())
    }

    fn add_metadata_value_allocations(
        &mut self,
        value: &ConditioningMetadataValue,
    ) -> Result<(), ConditioningError> {
        match value {
            ConditioningMetadataValue::ControlnetPooledOutput(Some(tensor))
            | ConditioningMetadataValue::PooledOutput(Some(tensor))
            | ConditioningMetadataValue::ConditioningLyrics(Some(tensor))
            | ConditioningMetadataValue::Tensor(tensor) => self.add_tensor_storage(tensor),
            ConditioningMetadataValue::TensorList(tensors)
            | ConditioningMetadataValue::ReferenceAudioTimbreLatents(Some(tensors)) => {
                self.add_allocation::<Tensor>(tensors.capacity())?;
                for tensor in tensors {
                    self.add_tensor_storage(tensor)?;
                }
                Ok(())
            }
            ConditioningMetadataValue::LatentBundle(bundle) => {
                if !self.metadata_resources.insert(Arc::as_ptr(bundle) as usize) {
                    return Ok(());
                }
                let parts = bundle
                    .resident_parts()
                    .map_err(|error| ConditioningError::Invalid(error.to_string()))?;
                self.add_u64(parts.owned_bytes())?;
                for allocation in parts.tensor_allocations() {
                    self.add_storage_allocation(
                        allocation.storage_id(),
                        allocation.resident_bytes(),
                    )?;
                }
                Ok(())
            }
            ConditioningMetadataValue::ClipVisionOutput(output) => {
                self.add_clip_vision_output(output)
            }
            ConditioningMetadataValue::ClipVisionConditioning(values) => {
                self.add_allocation::<ClipVisionConditioning>(values.capacity())?;
                for value in values {
                    self.add_clip_vision_output(&value.output)?;
                }
                Ok(())
            }
            ConditioningMetadataValue::FiniteF64List(values) => {
                self.add_allocation::<f64>(values.capacity())
            }
            ConditioningMetadataValue::Region(region) => self.add_region_allocations(region),
            ConditioningMetadataValue::Control(binding) => {
                self.add_bytes(binding.execution_digest.capacity())
            }
            ConditioningMetadataValue::ControlnetPooledOutput(None)
            | ConditioningMetadataValue::PooledOutput(None)
            | ConditioningMetadataValue::ConditioningLyrics(None)
            | ConditioningMetadataValue::ReferenceAudioTimbreLatents(None)
            | ConditioningMetadataValue::FiniteF64(_)
            | ConditioningMetadataValue::Unsigned(_)
            | ConditioningMetadataValue::Boolean(_)
            | ConditioningMetadataValue::TimestepRange(_) => Ok(()),
        }
    }

    fn add_value_allocations(
        &mut self,
        value: &ConditioningValue,
    ) -> Result<(), ConditioningError> {
        match value {
            ConditioningValue::Regular(tensor)
            | ConditioningValue::NoiseShape(tensor)
            | ConditioningValue::CrossAttention(tensor) => self.add_tensor_storage(tensor),
            ConditioningValue::Constant(value) => self.add_constant_allocations(value),
            ConditioningValue::List(tensors) => {
                self.add_allocation::<Tensor>(tensors.capacity())?;
                for tensor in tensors {
                    self.add_tensor_storage(tensor)?;
                }
                Ok(())
            }
        }
    }

    fn add_constant_allocations(
        &mut self,
        value: &ConditioningConstant,
    ) -> Result<(), ConditioningError> {
        match value {
            ConditioningConstant::Null
            | ConditioningConstant::Boolean(_)
            | ConditioningConstant::Signed(_)
            | ConditioningConstant::Unsigned(_)
            | ConditioningConstant::FiniteF64Bits(_) => Ok(()),
            ConditioningConstant::Text(value) => self.add_bytes(value.capacity()),
            ConditioningConstant::Bytes(value) => self.add_bytes(value.capacity()),
            ConditioningConstant::Tensor(tensor) => self.add_tensor_storage(tensor),
            ConditioningConstant::List(values) | ConditioningConstant::Tuple(values) => {
                self.add_allocation::<ConditioningConstant>(values.capacity())?;
                for value in values {
                    self.add_constant_allocations(value)?;
                }
                Ok(())
            }
            ConditioningConstant::Map(values) => {
                self.add_allocation::<(String, ConditioningConstant)>(values.len())?;
                for (key, value) in values {
                    self.add_bytes(key.capacity())?;
                    self.add_constant_allocations(value)?;
                }
                Ok(())
            }
        }
    }

    fn add_region_allocations(
        &mut self,
        region: &ConditioningRegion,
    ) -> Result<(), ConditioningError> {
        match region {
            ConditioningRegion::Absolute { sizes, offsets } => {
                self.add_allocation::<u64>(sizes.capacity())?;
                self.add_allocation::<u64>(offsets.capacity())
            }
            ConditioningRegion::Percentage { sizes, offsets } => {
                self.add_allocation::<f64>(sizes.capacity())?;
                self.add_allocation::<f64>(offsets.capacity())
            }
        }
    }

    fn add_tensor_storage(&mut self, tensor: &Tensor) -> Result<(), ConditioningError> {
        self.add_storage_allocation(tensor.storage_id(), tensor.storage_byte_len())
    }

    fn add_clip_vision_output(
        &mut self,
        output: &Arc<ClipVisionOutput>,
    ) -> Result<(), ConditioningError> {
        if !self.metadata_resources.insert(Arc::as_ptr(output) as usize) {
            return Ok(());
        }
        let parts = output
            .resident_parts()
            .map_err(|error| ConditioningError::Invalid(error.to_string()))?;
        self.add_u64(parts.owned_bytes())?;
        for allocation in parts.tensor_allocations() {
            self.add_storage_allocation(allocation.storage_id(), allocation.resident_bytes())?;
        }
        Ok(())
    }

    fn add_storage_allocation(
        &mut self,
        storage_id: StorageId,
        resident_bytes: u64,
    ) -> Result<(), ConditioningError> {
        if let Some((_, existing_bytes)) = self.storages.get(&storage_id.get()) {
            if *existing_bytes != resident_bytes {
                return Err(ConditioningError::Invalid(
                    "conditioning tensors disagree about shared storage residency".to_owned(),
                ));
            }
            return Ok(());
        }
        self.storages
            .insert(storage_id.get(), (storage_id, resident_bytes));
        Ok(())
    }

    fn add_inline<Value>(&mut self) -> Result<(), ConditioningError> {
        self.add_bytes(mem::size_of::<Value>())
    }

    fn add_allocation<Value>(&mut self, capacity: usize) -> Result<(), ConditioningError> {
        let bytes = capacity
            .checked_mul(mem::size_of::<Value>())
            .ok_or(ConditioningError::ResidentBytesOverflow)?;
        self.add_bytes(bytes)
    }

    fn add_bytes(&mut self, bytes: usize) -> Result<(), ConditioningError> {
        let bytes = u64::try_from(bytes).map_err(|_| ConditioningError::ResidentBytesOverflow)?;
        self.add_u64(bytes)
    }

    fn add_u64(&mut self, bytes: u64) -> Result<(), ConditioningError> {
        self.owned_bytes = self
            .owned_bytes
            .checked_add(bytes)
            .ok_or(ConditioningError::ResidentBytesOverflow)?;
        Ok(())
    }

    fn resident_bytes(&self) -> Result<u64, ConditioningError> {
        self.storages
            .values()
            .try_fold(self.owned_bytes, |bytes, (_, resident_bytes)| {
                bytes
                    .checked_add(*resident_bytes)
                    .ok_or(ConditioningError::ResidentBytesOverflow)
            })
    }

    fn into_parts(self) -> ConditioningResidentParts {
        ConditioningResidentParts {
            owned_bytes: self.owned_bytes,
            tensor_allocations: self
                .storages
                .into_values()
                .map(
                    |(storage_id, resident_bytes)| ConditioningTensorResidentAllocation {
                        storage_id,
                        resident_bytes,
                    },
                )
                .collect(),
        }
    }
}

fn validate_identifier(subject: &str, value: &str) -> Result<(), ConditioningError> {
    if value.is_empty()
        || value.len() > 256
        || value.starts_with('.')
        || value.ends_with('.')
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(ConditioningError::Invalid(format!(
            "{subject} must be 1..=256 ASCII identifier characters"
        )));
    }
    Ok(())
}

fn validate_batched_tensor(tensor: &Tensor, subject: &str) -> Result<(), ConditioningError> {
    if tensor.descriptor().rank() == 0 || tensor.descriptor().shape()[0] == 0 {
        return Err(ConditioningError::Invalid(format!(
            "{subject} must have a nonzero batch dimension"
        )));
    }
    Ok(())
}

fn validate_region_rank(sizes: usize, offsets: usize) -> Result<(), ConditioningError> {
    if sizes == 0 || sizes != offsets || sizes > MAX_CONDITIONING_REGION_RANK {
        return Err(ConditioningError::Invalid(format!(
            "conditioning region sizes and offsets must have the same rank in 1..={MAX_CONDITIONING_REGION_RANK}"
        )));
    }
    Ok(())
}

fn intersect_regions(
    left: &ResolvedConditioningRegion,
    right: &ResolvedConditioningRegion,
) -> Result<ResolvedConditioningRegion, ConditioningError> {
    if left.target_dimensions() != right.target_dimensions() {
        return Err(ConditioningError::Invalid(
            "conditioning region intersection targets differ".to_owned(),
        ));
    }
    let mut sizes = Vec::new();
    let mut offsets = Vec::new();
    sizes
        .try_reserve_exact(left.sizes().len())
        .map_err(|_| ConditioningError::ShapeOverflow("region intersection sizes"))?;
    offsets
        .try_reserve_exact(left.sizes().len())
        .map_err(|_| ConditioningError::ShapeOverflow("region intersection offsets"))?;
    for (((left_size, left_offset), right_size), right_offset) in left
        .sizes()
        .iter()
        .zip(left.offsets())
        .zip(right.sizes())
        .zip(right.offsets())
    {
        let start = (*left_offset).max(*right_offset);
        let left_end = left_offset
            .checked_add(*left_size)
            .ok_or(ConditioningError::ShapeOverflow("left region end"))?;
        let right_end = right_offset
            .checked_add(*right_size)
            .ok_or(ConditioningError::ShapeOverflow("right region end"))?;
        let end = left_end.min(right_end);
        if end <= start {
            return Err(ConditioningError::Invalid(
                "conditioning mask bounds do not intersect the explicit region".to_owned(),
            ));
        }
        offsets.push(start);
        sizes.push(end - start);
    }
    ResolvedConditioningRegion::checked(sizes, offsets, left.target_dimensions().to_vec())
}

fn area_only_feather_weight(
    region: &ResolvedConditioningRegion,
    spatial_indices: &[u64],
) -> Result<f32, ConditioningError> {
    let mut weight = 1.0_f32;
    for (((index, size), offset), target_dimension) in spatial_indices
        .iter()
        .zip(region.sizes())
        .zip(region.offsets())
        .zip(region.target_dimensions())
    {
        let feather = SOURCE_AREA_FEATHER.min(*size / 4);
        if feather == 0 {
            continue;
        }
        if *offset != 0 && *index < feather {
            weight *= (*index + 1) as f32 / feather as f32;
        }
        let end = offset
            .checked_add(*size)
            .ok_or(ConditioningError::ShapeOverflow(
                "conditioning area feather end",
            ))?;
        let trailing_start = size
            .checked_sub(feather)
            .ok_or(ConditioningError::ShapeOverflow(
                "conditioning area feather start",
            ))?;
        if end < *target_dimension && *index >= trailing_start {
            weight *= (*size - *index) as f32 / feather as f32;
        }
    }
    Ok(weight)
}

fn validate_constant_tensor(tensor: &Tensor) -> Result<(), ConditioningError> {
    tensor.descriptor().element_count()?;
    if tensor.storage_byte_len() > MAX_CONDITIONING_CONSTANT_BYTES as u64 {
        return Err(ConditioningError::Invalid(format!(
            "conditioning constant tensor exceeds {MAX_CONDITIONING_CONSTANT_BYTES} bytes"
        )));
    }
    Ok(())
}

fn tensor_constants_equal(left: &Tensor, right: &Tensor) -> bool {
    if left.descriptor() != right.descriptor() {
        return false;
    }
    let Ok(elements) = left.descriptor().element_count() else {
        return false;
    };
    for index in 0..elements {
        let Ok(left) = left.linear_element_bytes(index) else {
            return false;
        };
        let Ok(right) = right.linear_element_bytes(index) else {
            return false;
        };
        if left != right {
            return false;
        }
    }
    true
}

fn tensor_concat_compatible(left: &Tensor, right: &Tensor, ignored_axis: Option<usize>) -> bool {
    let left_descriptor = left.descriptor();
    let right_descriptor = right.descriptor();
    if left_descriptor.rank() != right_descriptor.rank()
        || left_descriptor.dtype() != right_descriptor.dtype()
        || left_descriptor.device() != right_descriptor.device()
        || left_descriptor.stream() != right_descriptor.stream()
    {
        return false;
    }
    left_descriptor
        .shape()
        .iter()
        .zip(right_descriptor.shape())
        .enumerate()
        .all(|(axis, (left, right))| ignored_axis == Some(axis) || left == right)
}

fn cross_attention_pair_compatible(left: &Tensor, right: &Tensor) -> bool {
    if !tensor_concat_compatible(left, right, Some(1)) {
        return false;
    }
    let left_tokens = left.descriptor().shape()[1];
    let right_tokens = right.descriptor().shape()[1];
    checked_lcm(left_tokens, right_tokens)
        .is_some_and(|tokens| tokens / left_tokens <= 4 && tokens / right_tokens <= 4)
}

fn concat_regular<Extract>(
    first: &Tensor,
    others: &[ConditioningValue],
    extract: Extract,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ConditioningError>
where
    Extract: Fn(&ConditioningValue) -> Option<&Tensor>,
{
    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(others.len() + 1)
        .map_err(|_| ConditioningError::ShapeOverflow("conditioning concatenation inputs"))?;
    tensors.push(first.clone());
    for other in others {
        context.cancellation.check()?;
        let tensor = extract(other).ok_or_else(|| {
            ConditioningError::Invalid("conditioning value kinds cannot be concatenated".to_owned())
        })?;
        if !tensor_concat_compatible(first, tensor, None) {
            return Err(ConditioningError::Invalid(
                "regular conditioning shapes, dtypes, devices, and streams must match".to_owned(),
            ));
        }
        tensors.push(tensor.clone());
    }
    Ok(torch_cat_with_context_exact_native(
        backend, &tensors, 0, context,
    )?)
}

fn concat_cross_attention(
    first: &Tensor,
    others: &[ConditioningValue],
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ConditioningError> {
    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(others.len() + 1)
        .map_err(|_| ConditioningError::ShapeOverflow("cross-attention inputs"))?;
    tensors.push(first.clone());
    for other in others {
        let ConditioningValue::CrossAttention(tensor) = other else {
            return Err(ConditioningError::Invalid(
                "cross-attention values can only concatenate with cross-attention values"
                    .to_owned(),
            ));
        };
        if !tensor_concat_compatible(first, tensor, Some(1)) {
            return Err(ConditioningError::Invalid(
                "cross-attention batch, channel, dtype, device, and stream contracts differ"
                    .to_owned(),
            ));
        }
        tensors.push(tensor.clone());
    }
    let token_count = tensors.iter().try_fold(1_u64, |token_count, tensor| {
        checked_lcm(token_count, tensor.descriptor().shape()[1]).ok_or(
            ConditioningError::ShapeOverflow("cross-attention token LCM"),
        )
    })?;
    if tensors.iter().any(|tensor| {
        token_count / tensor.descriptor().shape()[1] > CROSS_ATTENTION_MAX_REPEAT_RATIO
    }) {
        return Err(ConditioningError::Invalid(
            "cross-attention token padding exceeds the maximum source repeat ratio of four"
                .to_owned(),
        ));
    }
    let mut padded = Vec::new();
    padded
        .try_reserve_exact(tensors.len())
        .map_err(|_| ConditioningError::ShapeOverflow("padded cross-attention inputs"))?;
    for tensor in &tensors {
        context.cancellation.check()?;
        let ratio = token_count / tensor.descriptor().shape()[1];
        if ratio == 1 {
            padded.push(tensor.clone());
        } else {
            let mut repeats = vec![1_i64; 3];
            repeats[1] = i64::try_from(ratio)
                .map_err(|_| ConditioningError::ShapeOverflow("cross-attention repeat"))?;
            padded.push(tensor_repeat_with_context_exact_native(
                backend, tensor, &repeats, context,
            )?);
        }
    }
    Ok(torch_cat_with_context_exact_native(
        backend, &padded, 0, context,
    )?)
}

fn concat_lists(
    first: &[Tensor],
    others: &[ConditioningValue],
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<ConditioningValue, ConditioningError> {
    let mut lists = Vec::new();
    lists
        .try_reserve_exact(others.len())
        .map_err(|_| ConditioningError::ShapeOverflow("conditioning lists"))?;
    for other in others {
        let ConditioningValue::List(tensors) = other else {
            return Err(ConditioningError::Invalid(
                "conditioning lists can only concatenate with lists".to_owned(),
            ));
        };
        if tensors.len() != first.len() {
            return Err(ConditioningError::Invalid(
                "conditioning lists must have equal lengths".to_owned(),
            ));
        }
        lists.push(tensors);
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(first.len())
        .map_err(|_| ConditioningError::ShapeOverflow("concatenated conditioning list"))?;
    for (index, first_tensor) in first.iter().enumerate() {
        let mut tensors = Vec::new();
        tensors
            .try_reserve_exact(lists.len() + 1)
            .map_err(|_| ConditioningError::ShapeOverflow("conditioning list item inputs"))?;
        tensors.push(first_tensor.clone());
        for list in &lists {
            let tensor = list
                .get(index)
                .ok_or(ConditioningError::ShapeOverflow("conditioning list item"))?;
            if !tensor_concat_compatible(first_tensor, tensor, None) {
                return Err(ConditioningError::Invalid(
                    "conditioning list item shapes, dtypes, devices, and streams must match"
                        .to_owned(),
                ));
            }
            tensors.push(tensor.clone());
        }
        output.push(torch_cat_with_context_exact_native(
            backend, &tensors, 0, context,
        )?);
    }
    Ok(ConditioningValue::List(output))
}

fn repeat_to_batch_size(
    tensor: &Tensor,
    batch_size: u64,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ConditioningError> {
    context.cancellation.check()?;
    validate_batched_tensor(tensor, "repeated conditioning")?;
    let source_batch = tensor.descriptor().shape()[0];
    if source_batch == batch_size {
        return Ok(tensor.clone());
    }
    if source_batch > batch_size {
        return Ok(narrow_method_exact_native(
            tensor,
            0,
            0,
            batch_size,
            context.cancellation,
        )?);
    }
    let repeat_count =
        batch_size
            .checked_add(source_batch - 1)
            .ok_or(ConditioningError::ShapeOverflow(
                "conditioning batch repeat",
            ))?
            / source_batch;
    let mut repeats = vec![1_i64; tensor.descriptor().rank()];
    repeats[0] = i64::try_from(repeat_count)
        .map_err(|_| ConditioningError::ShapeOverflow("conditioning batch repeat"))?;
    let repeated = tensor_repeat_with_context_exact_native(backend, tensor, &repeats, context)?;
    Ok(narrow_method_exact_native(
        &repeated,
        0,
        0,
        batch_size,
        context.cancellation,
    )?)
}

fn narrow_spatial(
    tensor: &Tensor,
    region: &ResolvedConditioningRegion,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ConditioningError> {
    if tensor.descriptor().rank() != region.sizes().len() + 2 {
        return Err(ConditioningError::Invalid(
            "noise-shape conditioning and region spatial ranks differ".to_owned(),
        ));
    }
    let mut output = tensor.clone();
    for (dimension, (&offset, &size)) in region.offsets().iter().zip(region.sizes()).enumerate() {
        context.cancellation.check()?;
        let axis = i64::try_from(dimension + 2)
            .map_err(|_| ConditioningError::ShapeOverflow("noise-shape crop axis"))?;
        let offset = i64::try_from(offset)
            .map_err(|_| ConditioningError::ShapeOverflow("noise-shape crop offset"))?;
        output = narrow_method_exact_native(&output, axis, offset, size, context.cancellation)?;
    }
    Ok(output)
}

fn checked_lcm(left: u64, right: u64) -> Option<u64> {
    if left == 0 || right == 0 {
        return None;
    }
    left.checked_div(greatest_common_divisor(left, right))?
        .checked_mul(right)
}

fn greatest_common_divisor(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn hash_set(
    identity: &ConditioningIdentity,
    entries: &[ConditioningEntry],
    cancellation: &sim_types::CancellationToken,
) -> Result<String, ConditioningError> {
    let mut checkpoint = || cancellation.check().map_err(ConditioningError::from);
    hash_set_with_checkpoint(identity, entries, &mut checkpoint)
}

fn hash_set_with_checkpoint<Checkpoint>(
    identity: &ConditioningIdentity,
    entries: &[ConditioningEntry],
    checkpoint: &mut Checkpoint,
) -> Result<String, ConditioningError>
where
    Checkpoint: FnMut() -> Result<(), ConditioningError>,
{
    let mut hasher = Sha256::new();
    hasher.update(b"zed.sim.conditioning-set-strength-f64.v2");
    hasher.update(CONDITIONING_SCHEMA_VERSION.to_le_bytes());
    hash_string(&mut hasher, identity.namespace())?;
    hash_string(&mut hasher, identity.model_family().feature_id())?;
    hash_string(&mut hasher, identity.model_family().identifier())?;
    hash_string(&mut hasher, identity.model_family().architecture_version())?;
    hash_string(&mut hasher, identity.latent_format().feature_id())?;
    hash_string(&mut hasher, identity.latent_format().identifier())?;
    hash_u64(&mut hasher, entries.len())?;
    for entry in entries {
        checkpoint()?;
        hash_string(&mut hasher, entry.identifier())?;
        hash_string(
            &mut hasher,
            &entry
                .value()
                .deterministic_digest_with_checkpoint(checkpoint)?,
        )?;
        hasher.update(entry.options().strength.to_bits().to_le_bytes());
        hash_region(&mut hasher, entry.options().region.as_ref())?;
        hasher.update(entry.options().window.start_percent.to_bits().to_le_bytes());
        hasher.update(entry.options().window.end_percent.to_bits().to_le_bytes());
        hasher.update([u8::from(entry.options().default_region)]);
        match entry.references().control() {
            Some(control) => {
                hasher.update([1]);
                hash_string(&mut hasher, control.identifier())?;
            }
            None => hasher.update([0]),
        }
        hash_u64(&mut hasher, entry.references().hooks().len())?;
        for hook in entry.references().hooks() {
            hash_string(&mut hasher, hook.identifier())?;
        }
        match &entry.options().mask {
            Some(mask) => {
                hasher.update([1]);
                hash_tensor(&mut hasher, mask.tensor(), checkpoint)?;
                hasher.update(mask.strength().to_bits().to_le_bytes());
                hash_u64(&mut hasher, mask.feather().len())?;
                for feather in mask.feather() {
                    hasher.update(feather.to_le_bytes());
                }
                hasher.update([u8::from(mask.set_region_to_nonzero_bounds())]);
            }
            None => hasher.update([0]),
        }
        if !entry.metadata().is_empty() {
            hasher.update(b"zed.sim.conditioning-metadata.v1");
            hash_metadata_map(&mut hasher, entry.metadata(), checkpoint)?;
        }
    }
    checkpoint()?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_metadata_map<Checkpoint>(
    hasher: &mut Sha256,
    metadata: &ConditioningMetadataMap,
    checkpoint: &mut Checkpoint,
) -> Result<(), ConditioningError>
where
    Checkpoint: FnMut() -> Result<(), ConditioningError>,
{
    hash_u64(hasher, metadata.len())?;
    for (key, value) in metadata.iter() {
        checkpoint()?;
        hasher.update((*key as u8).to_le_bytes());
        match value {
            ConditioningMetadataValue::Tensor(tensor) => {
                hasher.update([0]);
                hash_tensor(hasher, tensor, checkpoint)?;
            }
            ConditioningMetadataValue::TensorList(tensors) => {
                hasher.update([1]);
                hash_u64(hasher, tensors.len())?;
                for tensor in tensors {
                    hash_tensor(hasher, tensor, checkpoint)?;
                }
            }
            ConditioningMetadataValue::LatentBundle(bundle) => {
                hasher.update([2]);
                hash_string(hasher, bundle.semantic_digest_sha256())?;
            }
            ConditioningMetadataValue::ClipVisionOutput(output) => {
                hasher.update([3]);
                hasher.update(output.semantic_digest_sha256());
            }
            ConditioningMetadataValue::ClipVisionConditioning(values) => {
                hasher.update([4]);
                hash_u64(hasher, values.len())?;
                for value in values {
                    hasher.update(value.output.semantic_digest_sha256());
                    hasher.update(value.strength.to_bits().to_le_bytes());
                    hasher.update(value.noise_augmentation.to_bits().to_le_bytes());
                }
            }
            ConditioningMetadataValue::FiniteF64(value) => {
                hasher.update([5]);
                hasher.update(value.to_bits().to_le_bytes());
            }
            ConditioningMetadataValue::FiniteF64List(values) => {
                hasher.update([6]);
                hash_u64(hasher, values.len())?;
                for value in values {
                    hasher.update(value.to_bits().to_le_bytes());
                }
            }
            ConditioningMetadataValue::Unsigned(value) => {
                hasher.update([7]);
                hasher.update(value.to_le_bytes());
            }
            ConditioningMetadataValue::Boolean(value) => {
                hasher.update([8, u8::from(*value)]);
            }
            ConditioningMetadataValue::Region(region) => {
                hasher.update([9]);
                hash_region(hasher, Some(region))?;
            }
            ConditioningMetadataValue::TimestepRange(range) => {
                hasher.update([10]);
                hasher.update(range.start_percent.to_bits().to_le_bytes());
                hasher.update(range.end_percent.to_bits().to_le_bytes());
            }
            ConditioningMetadataValue::Control(binding) => {
                hasher.update([11]);
                hash_string(hasher, &binding.execution_digest)?;
                hasher.update([u8::from(binding.apply_to_unconditioned)]);
            }
            ConditioningMetadataValue::ControlnetPooledOutput(pooled) => {
                hasher.update([12, u8::from(pooled.is_some())]);
                if let Some(tensor) = pooled {
                    hash_tensor(hasher, tensor, checkpoint)?;
                }
            }
            ConditioningMetadataValue::PooledOutput(pooled) => {
                hasher.update([13, u8::from(pooled.is_some())]);
                if let Some(tensor) = pooled {
                    hash_tensor(hasher, tensor, checkpoint)?;
                }
            }
            ConditioningMetadataValue::ConditioningLyrics(lyrics) => {
                hasher.update([14, u8::from(lyrics.is_some())]);
                if let Some(tensor) = lyrics {
                    hash_tensor(hasher, tensor, checkpoint)?;
                }
            }
            ConditioningMetadataValue::ReferenceAudioTimbreLatents(tensors) => {
                hasher.update([15, u8::from(tensors.is_some())]);
                if let Some(tensors) = tensors {
                    hash_u64(hasher, tensors.len())?;
                    for tensor in tensors {
                        hash_tensor(hasher, tensor, checkpoint)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn hash_conditioning_constant<Checkpoint>(
    hasher: &mut Sha256,
    value: &ConditioningConstant,
    checkpoint: &mut Checkpoint,
) -> Result<(), ConditioningError>
where
    Checkpoint: FnMut() -> Result<(), ConditioningError>,
{
    checkpoint()?;
    match value {
        ConditioningConstant::Null => hasher.update([0]),
        ConditioningConstant::Boolean(value) => hasher.update([1, u8::from(*value)]),
        ConditioningConstant::Signed(value) => {
            hasher.update([2]);
            hasher.update(value.to_le_bytes());
        }
        ConditioningConstant::Unsigned(value) => {
            hasher.update([3]);
            hasher.update(value.to_le_bytes());
        }
        ConditioningConstant::FiniteF64Bits(value) => {
            hasher.update([4]);
            hasher.update(value.to_le_bytes());
        }
        ConditioningConstant::Text(value) => {
            hasher.update([5]);
            hash_string(hasher, value)?;
        }
        ConditioningConstant::Bytes(value) => {
            hasher.update([6]);
            hash_u64(hasher, value.len())?;
            hasher.update(value);
        }
        ConditioningConstant::Tensor(tensor) => {
            hasher.update([7]);
            hash_tensor(hasher, tensor, checkpoint)?;
        }
        ConditioningConstant::List(values) | ConditioningConstant::Tuple(values) => {
            hasher.update([if matches!(value, ConditioningConstant::List(_)) {
                8
            } else {
                9
            }]);
            hash_u64(hasher, values.len())?;
            for value in values {
                hash_conditioning_constant(hasher, value, checkpoint)?;
            }
        }
        ConditioningConstant::Map(values) => {
            hasher.update([10]);
            hash_u64(hasher, values.len())?;
            for (key, value) in values {
                hash_string(hasher, key)?;
                hash_conditioning_constant(hasher, value, checkpoint)?;
            }
        }
    }
    Ok(())
}

fn hash_region(
    hasher: &mut Sha256,
    region: Option<&ConditioningRegion>,
) -> Result<(), ConditioningError> {
    match region {
        None => hasher.update([0]),
        Some(ConditioningRegion::Absolute { sizes, offsets }) => {
            hasher.update([1]);
            hash_u64(hasher, sizes.len())?;
            for value in sizes.iter().chain(offsets) {
                hasher.update(value.to_le_bytes());
            }
        }
        Some(ConditioningRegion::Percentage { sizes, offsets }) => {
            hasher.update([2]);
            hash_u64(hasher, sizes.len())?;
            for value in sizes.iter().chain(offsets) {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
    }
    Ok(())
}

fn hash_tensor<Checkpoint>(
    hasher: &mut Sha256,
    tensor: &Tensor,
    checkpoint: &mut Checkpoint,
) -> Result<(), ConditioningError>
where
    Checkpoint: FnMut() -> Result<(), ConditioningError>,
{
    checkpoint()?;
    if tensor.descriptor().device() != DeviceId::CPU {
        return Err(ConditioningError::UnreadableDevice(
            tensor.descriptor().device(),
        ));
    }
    hasher.update(
        serde_json::to_vec(&(
            tensor.descriptor().shape(),
            tensor.descriptor().dtype(),
            tensor.descriptor().device(),
            tensor.descriptor().stream(),
        ))
        .map_err(|error| ConditioningError::Encoding(error.to_string()))?,
    );
    let elements = tensor.descriptor().element_count()?;
    for index in 0..elements {
        if index.is_multiple_of(1024) {
            checkpoint()?;
        }
        hasher.update(tensor.linear_element_bytes(index)?);
    }
    checkpoint()?;
    Ok(())
}

fn hash_string(hasher: &mut Sha256, value: &str) -> Result<(), ConditioningError> {
    hash_u64(hasher, value.len())?;
    hasher.update(value.as_bytes());
    Ok(())
}

fn hash_u64(hasher: &mut Sha256, value: usize) -> Result<(), ConditioningError> {
    let value = u64::try_from(value)
        .map_err(|_| ConditioningError::ShapeOverflow("conditioning digest length"))?;
    hasher.update(value.to_le_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{CpuWorkspaceAuthority, DecodedScalar, DeviceId, StreamId};
    use std::error::Error;

    type TestResult = Result<(), Box<dyn Error>>;

    fn runtime() -> Result<(CpuBackend, CpuWorkspaceAuthority), TensorError> {
        CpuWorkspaceAuthority::create_backend(16 * 1_048_576)
    }

    fn upload_f32(
        backend: &CpuBackend,
        authority: &CpuWorkspaceAuthority,
        shape: Vec<u64>,
        values: &[f32],
    ) -> Result<Tensor, Box<dyn Error>> {
        let cancellation = sim_types::CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(0)?,
            &cancellation,
        );
        let descriptor =
            TensorDescriptor::contiguous(shape, DType::F32, DeviceId::CPU, StreamId::DEFAULT)?;
        Ok(backend.upload_f32(descriptor, values, &context)?.0)
    }

    fn tensor_values(tensor: &Tensor) -> Result<Vec<f64>, Box<dyn Error>> {
        let count = tensor.descriptor().element_count()?;
        let mut values = Vec::new();
        values.try_reserve_exact(usize::try_from(count)?)?;
        for index in 0..count {
            let value = tensor
                .descriptor()
                .dtype()
                .decode_scalar(tensor.linear_element_bytes(index)?)?;
            match value {
                DecodedScalar::Real(value) => values.push(value),
                _ => return Err("test tensor is not real-valued".into()),
            }
        }
        Ok(values)
    }

    fn identity() -> Result<ConditioningIdentity, Box<dyn Error>> {
        Ok(ConditioningIdentity::new(
            "native.conditioning.test",
            ModelFamilyIdentity::new("COMFY-MODEL-0999", "conditioning_test", "v1")?,
            LatentFormatIdentity::new("COMFY-MODEL-0045", "SD15")?,
        )?)
    }

    fn structured_conditioning_set(
        tensors: Vec<Tensor>,
        cancellation: &sim_types::CancellationToken,
    ) -> Result<ConditioningSet, Box<dyn Error>> {
        let mut tensors = tensors.into_iter();
        let list_first = tensors.next().ok_or("missing first list tensor")?;
        let list_second = tensors.next().ok_or("missing second list tensor")?;
        let constant_tensor = tensors.next().ok_or("missing constant tensor")?;
        let cross_attention = tensors.next().ok_or("missing cross-attention tensor")?;
        let mask_tensor = tensors.next().ok_or("missing mask tensor")?;
        if tensors.next().is_some() {
            return Err("too many structured conditioning tensors".into());
        }

        let listed = ConditioningEntry::checked_with_references(
            "listed",
            ConditioningValue::list(vec![list_first, list_second])?,
            ConditioningEntryOptions {
                region: Some(ConditioningRegion::absolute(vec![1, 2], vec![0, 0])?),
                window: ConditioningWindow::new(0.1, 0.9)?,
                ..ConditioningEntryOptions::default()
            },
            ConditioningReferences::checked(
                Some(ConditioningControlReference::checked("control.primary")?),
                vec![
                    ConditioningHookReference::checked("hook.first")?,
                    ConditioningHookReference::checked("hook.second")?,
                ],
            )?,
        )?;
        let constant = ConditioningEntry::checked(
            "constant",
            ConditioningValue::constant(ConditioningConstant::Map(BTreeMap::from([
                (
                    "tensor".to_owned(),
                    ConditioningConstant::Tensor(constant_tensor),
                ),
                (
                    "text".to_owned(),
                    ConditioningConstant::Text("metadata".to_owned()),
                ),
            ])))?,
            ConditioningEntryOptions::default(),
        )?;
        let masked = ConditioningEntry::checked(
            "masked",
            ConditioningValue::cross_attention(cross_attention)?,
            ConditioningEntryOptions {
                mask: Some(ConditioningMask::new(mask_tensor, 0.75, vec![0, 0], false)?),
                ..ConditioningEntryOptions::default()
            },
        )?;
        Ok(ConditioningSet::checked(
            identity()?,
            vec![listed, constant, masked],
            cancellation,
        )?)
    }

    #[test]
    fn identities_and_windows_reuse_canonical_owners_and_reject_invalid_inputs() -> TestResult {
        let identity = identity()?;
        assert_eq!(identity.namespace(), "native.conditioning.test");
        assert_eq!(identity.model_family().identifier(), "conditioning_test");
        assert_eq!(identity.latent_format().identifier(), "SD15");
        assert!(
            ConditioningIdentity::new(
                "bad namespace",
                identity.model_family().clone(),
                identity.latent_format().clone(),
            )
            .is_err()
        );

        let window = ConditioningWindow::new(0.2, 0.8)?;
        assert!(window.contains(0.2));
        assert!(window.contains(0.8));
        assert!(!window.contains(0.81));
        assert!(ConditioningWindow::new(-0.1, 0.8).is_err());
        assert!(ConditioningWindow::new(0.9, 0.8).is_err());
        assert!(ConditioningWindow::new(f64::NAN, 0.8).is_err());

        let encoded = serde_json::to_string(&identity)?;
        let decoded: ConditioningIdentity = serde_json::from_str(&encoded)?;
        assert_eq!(decoded, identity);
        assert_eq!(decoded.digest()?, identity.digest()?);
        assert_eq!(decoded.resident_bytes()?, identity.resident_bytes()?);
        let mut unsupported: serde_json::Value = serde_json::from_str(&encoded)?;
        let object = unsupported
            .as_object_mut()
            .ok_or("conditioning identity did not encode as an object")?;
        object.insert("schema_version".to_owned(), serde_json::Value::from(2));
        assert!(serde_json::from_value::<ConditioningIdentity>(unsupported).is_err());
        let mut unknown: serde_json::Value = serde_json::from_str(&encoded)?;
        unknown
            .as_object_mut()
            .ok_or("conditioning identity did not encode as an object")?
            .insert("unknown".to_owned(), serde_json::Value::Null);
        assert!(serde_json::from_value::<ConditioningIdentity>(unknown).is_err());

        let references = ConditioningReferences::checked(
            Some(ConditioningControlReference::checked("control.primary")?),
            vec![
                ConditioningHookReference::checked("hook.first")?,
                ConditioningHookReference::checked("hook.second")?,
            ],
        )?;
        assert_eq!(
            references.control().map(|reference| reference.identifier()),
            Some("control.primary")
        );
        assert_eq!(references.hooks()[1].identifier(), "hook.second");
        assert!(
            ConditioningReferences::checked(
                None,
                vec![
                    ConditioningHookReference::checked("hook.duplicate")?,
                    ConditioningHookReference::checked("hook.duplicate")?,
                ],
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn regular_repeat_and_concat_delegate_to_canonical_tensor_operations() -> TestResult {
        let (backend, authority) = runtime()?;
        let tensor = upload_f32(&backend, &authority, vec![2, 1], &[1.0, 2.0])?;
        let original_storage = tensor.storage_id();
        let value = ConditioningValue::regular(tensor.clone())?;
        let cancellation = sim_types::CancellationToken::default();
        let scratch = authority.authorize_workspace(6 * 4)?;
        let context = backend.execution_context(StreamId::DEFAULT, scratch.clone(), &cancellation);
        let processed = value.process(5, None, &backend, &context)?;
        let ConditioningValue::Regular(processed_tensor) = processed else {
            return Err("processed regular value changed kind".into());
        };
        assert_eq!(processed_tensor.descriptor().shape(), &[5, 1]);
        assert_eq!(tensor_values(&processed_tensor)?, [1.0, 2.0, 1.0, 2.0, 1.0]);
        assert_eq!(tensor.storage_id(), original_storage);
        assert_eq!(tensor_values(&tensor)?, [1.0, 2.0]);
        assert_eq!(scratch.in_use_bytes(), 0);

        let other =
            ConditioningValue::regular(upload_f32(&backend, &authority, vec![2, 1], &[3.0, 4.0])?)?;
        assert!(value.can_concat(&other));
        let concatenated = value.concat(&[other], &backend, &context)?;
        let ConditioningValue::Regular(concatenated) = concatenated else {
            return Err("concatenated regular value changed kind".into());
        };
        assert_eq!(concatenated.descriptor().shape(), &[4, 1]);
        assert_eq!(tensor_values(&concatenated)?, [1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn noise_shape_narrows_every_spatial_axis_before_batch_repeat() -> TestResult {
        let (backend, authority) = runtime()?;
        let values = (0..20).map(|value| value as f32).collect::<Vec<_>>();
        let tensor = upload_f32(&backend, &authority, vec![1, 1, 4, 5], &values)?;
        let value = ConditioningValue::noise_shape(tensor)?;
        let region = ConditioningRegion::absolute(vec![2, 3], vec![1, 1])?.resolve(&[4, 5])?;
        let cancellation = sim_types::CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(2 * 2 * 3 * 4)?,
            &cancellation,
        );
        let processed = value.process(2, Some(&region), &backend, &context)?;
        let ConditioningValue::NoiseShape(processed) = processed else {
            return Err("noise-shape value changed kind".into());
        };
        assert_eq!(processed.descriptor().shape(), &[2, 1, 2, 3]);
        assert_eq!(
            tensor_values(&processed)?,
            [
                6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0
            ]
        );
        Ok(())
    }

    #[test]
    fn cross_attention_uses_checked_lcm_padding_and_caps_every_source_ratio() -> TestResult {
        let (backend, authority) = runtime()?;
        let left = ConditioningValue::cross_attention(upload_f32(
            &backend,
            &authority,
            vec![1, 2, 1],
            &[1.0, 2.0],
        )?)?;
        let right = ConditioningValue::cross_attention(upload_f32(
            &backend,
            &authority,
            vec![1, 3, 1],
            &[3.0, 4.0, 5.0],
        )?)?;
        assert!(left.can_concat(&right));
        let cancellation = sim_types::CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(6 * 4)?,
            &cancellation,
        );
        let output = left.concat(&[right], &backend, &context)?;
        let ConditioningValue::CrossAttention(output) = output else {
            return Err("cross-attention value changed kind".into());
        };
        assert_eq!(output.descriptor().shape(), &[2, 6, 1]);
        assert_eq!(
            tensor_values(&output)?,
            [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0]
        );

        let too_long = ConditioningValue::cross_attention(upload_f32(
            &backend,
            &authority,
            vec![1, 10, 1],
            &[0.0; 10],
        )?)?;
        assert!(!left.can_concat(&too_long));
        assert!(left.concat(&[too_long], &backend, &context).is_err());

        let three = ConditioningValue::cross_attention(upload_f32(
            &backend,
            &authority,
            vec![1, 3, 1],
            &[0.0; 3],
        )?)?;
        let four = ConditioningValue::cross_attention(upload_f32(
            &backend,
            &authority,
            vec![1, 4, 1],
            &[0.0; 4],
        )?)?;
        assert!(left.can_concat(&three));
        assert!(left.can_concat(&four));
        assert!(left.concat(&[three, four], &backend, &context).is_err());
        Ok(())
    }

    #[test]
    fn constants_and_lists_preserve_sim_equality_and_itemwise_concat_contracts() -> TestResult {
        let (backend, authority) = runtime()?;
        assert!(ConditioningConstant::finite_f64(f64::INFINITY).is_err());
        let nullable = ConditioningValue::constant(ConditioningConstant::Null)?;
        assert!(nullable.can_concat(&ConditioningValue::constant(ConditioningConstant::Null)?));
        let mut too_deep = ConditioningConstant::Null;
        for _ in 0..MAX_CONDITIONING_CONSTANT_DEPTH {
            too_deep = ConditioningConstant::List(vec![too_deep]);
        }
        assert!(ConditioningValue::constant(too_deep).is_err());
        assert!(
            ConditioningValue::constant(ConditioningConstant::List(vec![
                ConditioningConstant::Null;
                MAX_CONDITIONING_CONSTANT_ITEMS
                    + 1
            ]))
            .is_err()
        );
        let constant = ConditioningValue::constant(ConditioningConstant::finite_f64(0.5)?)?;
        let same = ConditioningValue::constant(ConditioningConstant::finite_f64(0.5)?)?;
        let different = ConditioningValue::constant(ConditioningConstant::finite_f64(0.25)?)?;
        assert!(constant.can_concat(&same));
        assert!(!constant.can_concat(&different));
        let tensor_constant = ConditioningValue::constant(ConditioningConstant::Tensor(
            upload_f32(&backend, &authority, vec![1, 2], &[1.0, 2.0])?,
        ))?;
        let same_tensor_constant = ConditioningValue::constant(ConditioningConstant::Tensor(
            upload_f32(&backend, &authority, vec![1, 2], &[1.0, 2.0])?,
        ))?;
        let different_tensor_constant = ConditioningValue::constant(ConditioningConstant::Tensor(
            upload_f32(&backend, &authority, vec![1, 2], &[1.0, 3.0])?,
        ))?;
        assert!(tensor_constant.can_concat(&same_tensor_constant));
        assert!(!tensor_constant.can_concat(&different_tensor_constant));
        assert_eq!(
            tensor_constant.deterministic_digest(&sim_types::CancellationToken::default())?,
            same_tensor_constant.deterministic_digest(&sim_types::CancellationToken::default())?
        );
        let list_constant = ConditioningValue::constant(ConditioningConstant::List(vec![
            ConditioningConstant::Signed(1),
        ]))?;
        let tuple_constant = ConditioningValue::constant(ConditioningConstant::Tuple(vec![
            ConditioningConstant::Signed(1),
        ]))?;
        assert!(!list_constant.can_concat(&tuple_constant));
        assert_ne!(
            list_constant.deterministic_digest(&sim_types::CancellationToken::default())?,
            tuple_constant.deterministic_digest(&sim_types::CancellationToken::default())?
        );
        assert!(
            ConditioningValue::constant(ConditioningConstant::Map(BTreeMap::from([(
                "arbitrary map key with spaces".into(),
                ConditioningConstant::Null,
            ),])))
            .is_ok()
        );
        let cancellation = sim_types::CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(32)?,
            &cancellation,
        );
        assert!(matches!(
            constant.concat(&[same], &backend, &context)?,
            ConditioningValue::Constant(_)
        ));
        assert!(constant.concat(&[different], &backend, &context).is_err());

        let list = ConditioningValue::list(vec![
            upload_f32(&backend, &authority, vec![1, 2], &[1.0, 2.0])?,
            upload_f32(&backend, &authority, vec![1, 1], &[3.0])?,
        ])?;
        let other = ConditioningValue::list(vec![
            upload_f32(&backend, &authority, vec![1, 2], &[4.0, 5.0])?,
            upload_f32(&backend, &authority, vec![1, 1], &[6.0])?,
        ])?;
        assert!(list.can_concat(&other));
        let concatenated = list.concat(&[other], &backend, &context)?;
        let ConditioningValue::List(items) = concatenated else {
            return Err("conditioning list changed kind".into());
        };
        assert_eq!(items[0].descriptor().shape(), &[2, 2]);
        assert_eq!(tensor_values(&items[0])?, [1.0, 2.0, 4.0, 5.0]);
        assert_eq!(tensor_values(&items[1])?, [3.0, 6.0]);
        Ok(())
    }

    #[test]
    fn multidimensional_regions_resolve_percentage_rounding_clamping_and_bounds() -> TestResult {
        let percentage = ConditioningRegion::percentage(vec![0.5, 0.3, 1.0], vec![0.25, 0.8, 0.0])?;
        let resolved = percentage.resolve(&[8, 10, 3])?;
        assert_eq!(resolved.sizes(), &[4, 2, 3]);
        assert_eq!(resolved.offsets(), &[2, 8, 0]);
        assert!(!resolved.is_full());

        let ties_to_even =
            ConditioningRegion::percentage(vec![0.25], vec![0.25])?.resolve(&[10])?;
        assert_eq!(ties_to_even.sizes(), &[2]);
        assert_eq!(ties_to_even.offsets(), &[2]);

        let absolute = ConditioningRegion::absolute(vec![100, 2], vec![3, 2])?;
        let resolved = absolute.resolve(&[5, 5])?;
        assert_eq!(resolved.sizes(), &[2, 2]);
        assert_eq!(resolved.offsets(), &[3, 2]);
        assert!(ConditioningRegion::absolute(vec![0], vec![0]).is_err());
        assert!(ConditioningRegion::percentage(vec![1.1], vec![0.0]).is_err());
        let source_edge = ConditioningRegion::percentage(vec![0.5], vec![1.0])?;
        assert!(source_edge.resolve(&[5]).is_err());
        assert!(absolute.resolve(&[5]).is_err());
        let unchecked_invalid = ConditioningRegion::Percentage {
            sizes: vec![f64::NAN],
            offsets: vec![0.0],
        };
        assert!(unchecked_invalid.resolve(&[5]).is_err());
        Ok(())
    }

    #[test]
    fn masks_resize_crop_broadcast_channels_and_do_not_apply_area_feathering() -> TestResult {
        let (backend, authority) = runtime()?;
        let values = (0..25).map(|value| value as f32).collect::<Vec<_>>();
        let mask_tensor = upload_f32(&backend, &authority, vec![5, 5], &values)?;
        assert!(ConditioningMask::new(mask_tensor.clone(), 0.8, vec![1, 1], false).is_err());
        let mask = ConditioningMask::new(mask_tensor, 0.8, vec![0, 0], false)?;
        let target = TensorDescriptor::contiguous(
            vec![2, 3, 5, 5],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let region = ConditioningRegion::absolute(vec![3, 3], vec![1, 1])?.resolve(&[5, 5])?;
        let cancellation = sim_types::CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(2 * 3 * 3 * 3 * 4)?,
            &cancellation,
        );
        let resolved = mask.resolve(&target, &region, &backend, &context)?;
        assert_eq!(resolved.tensor().descriptor().shape(), &[2, 3, 3, 3]);
        assert_eq!(resolved.feather_weight(&[0, 0])?, 0.8);
        assert_eq!(resolved.feather_weight(&[1, 1])?, 0.8);
        assert!(resolved.feather_weight(&[3, 0]).is_err());
        let values = tensor_values(resolved.tensor())?;
        assert_eq!(
            &values[..9],
            &[6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 16.0, 17.0, 18.0]
        );
        assert_eq!(&values[9..18], &values[..9]);
        assert_eq!(&values[27..36], &values[..9]);
        Ok(())
    }

    #[test]
    fn masks_use_source_bilinear_resize_before_crop_and_broadcast() -> TestResult {
        let (backend, authority) = runtime()?;
        let mask = ConditioningMask::new(
            upload_f32(&backend, &authority, vec![1, 2, 2], &[0.0, 1.0, 2.0, 3.0])?,
            1.0,
            vec![0, 0],
            false,
        )?;
        let target = TensorDescriptor::contiguous(
            vec![1, 1, 4, 4],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let region = ResolvedConditioningRegion::full(vec![4, 4])?;
        let cancellation = sim_types::CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1 << 16)?,
            &cancellation,
        );
        let resolved = mask.resolve(&target, &region, &backend, &context)?;
        assert_eq!(resolved.tensor().descriptor().shape(), &[1, 1, 4, 4]);
        assert_eq!(
            tensor_values(resolved.tensor())?,
            [
                0.0, 0.25, 0.75, 1.0, 0.5, 0.75, 1.25, 1.5, 1.5, 1.75, 2.25, 2.5, 2.0, 2.25, 2.75,
                3.0,
            ]
        );
        let f16_source = upload_f32(&backend, &authority, vec![1, 2, 2], &[0.0, 1.0, 2.0, 3.0])?;
        let f16_source = cast_to_with_context_exact_native(
            &backend,
            &f16_source,
            DType::F16,
            DeviceId::CPU,
            false,
            false,
            &context,
        )?;
        let f16_mask = ConditioningMask::new(f16_source, 1.0, vec![0, 0], false)?;
        let f16_resolved = f16_mask.resolve(&target, &region, &backend, &context)?;
        assert_eq!(f16_resolved.tensor().descriptor().dtype(), DType::F16);
        assert_eq!(
            tensor_values(f16_resolved.tensor())?,
            tensor_values(resolved.tensor())?
        );
        Ok(())
    }

    #[test]
    fn area_only_feather_and_default_subtraction_match_source_weights() -> TestResult {
        let (backend, authority) = runtime()?;
        let entry = ConditioningEntry::checked_with_references(
            "area",
            ConditioningValue::constant(ConditioningConstant::Null)?,
            ConditioningEntryOptions {
                region: Some(ConditioningRegion::absolute(vec![8], vec![2])?),
                ..ConditioningEntryOptions::default()
            },
            ConditioningReferences::checked(
                Some(ConditioningControlReference::checked("control.area")?),
                vec![ConditioningHookReference::checked("hook.area")?],
            )?,
        )?;
        let target = TensorDescriptor::contiguous(
            vec![1, 1, 12],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let cancellation = sim_types::CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(0)?,
            &cancellation,
        );
        let resolved = entry.resolve(&target, &backend, &context)?;
        assert_eq!(resolved.contribution_weight(&[0], None)?, 0.5);
        assert_eq!(resolved.contribution_weight(&[1], None)?, 1.0);
        assert_eq!(resolved.contribution_weight(&[6], None)?, 1.0);
        assert_eq!(resolved.contribution_weight(&[7], None)?, 0.5);
        assert_eq!(
            resolved
                .references()
                .control()
                .map(|reference| reference.identifier()),
            Some("control.area")
        );
        assert_eq!(default_region_residual_weight(0.0)?, 1.0);
        assert_eq!(default_region_residual_weight(0.25)?, 0.75);
        assert_eq!(default_region_residual_weight(1.0)?, 0.0);
        assert_eq!(default_region_residual_weight(2.0)?, 0.0);
        assert_eq!(default_region_residual_weight(-0.5)?, 1.5);
        assert!(default_region_residual_weight(f32::NAN).is_err());
        Ok(())
    }

    #[test]
    fn nonzero_mask_bounds_drive_region_and_default_regions_are_explicit_markers() -> TestResult {
        let (backend, authority) = runtime()?;
        let mut mask_values = vec![0.0_f32; 20];
        mask_values[7] = 1.0;
        mask_values[13] = 1.0;
        let mask = ConditioningMask::new(
            upload_f32(&backend, &authority, vec![1, 4, 5], &mask_values)?,
            1.0,
            vec![0, 0],
            true,
        )?;
        let noise = ConditioningValue::noise_shape(upload_f32(
            &backend,
            &authority,
            vec![1, 1, 4, 5],
            &(0..20).map(|value| value as f32).collect::<Vec<_>>(),
        )?)?;
        let entry = ConditioningEntry::checked(
            "bounded",
            noise,
            ConditioningEntryOptions {
                mask: Some(mask),
                ..ConditioningEntryOptions::default()
            },
        )?;
        let target = TensorDescriptor::contiguous(
            vec![1, 1, 4, 5],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let cancellation = sim_types::CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(32)?,
            &cancellation,
        );
        let resolved = entry.resolve(&target, &backend, &context)?;
        assert_eq!(resolved.region().sizes(), &[2, 2]);
        assert_eq!(resolved.region().offsets(), &[1, 2]);
        let ConditioningValue::NoiseShape(value) = resolved.value() else {
            return Err("resolved bounded entry changed kind".into());
        };
        assert_eq!(value.descriptor().shape(), &[1, 1, 2, 2]);
        assert_eq!(tensor_values(value)?, [7.0, 8.0, 12.0, 13.0]);

        let invalid_default = ConditioningEntry::checked(
            "invalid-default",
            ConditioningValue::constant(ConditioningConstant::Boolean(true))?,
            ConditioningEntryOptions {
                region: Some(ConditioningRegion::absolute(vec![1], vec![0])?),
                default_region: true,
                ..ConditioningEntryOptions::default()
            },
        );
        assert!(invalid_default.is_err());
        let default = ConditioningEntry::checked(
            "default",
            ConditioningValue::constant(ConditioningConstant::Boolean(true))?,
            ConditioningEntryOptions {
                default_region: true,
                ..ConditioningEntryOptions::default()
            },
        )?;
        let resolved_default = default.resolve(&target, &backend, &context)?;
        assert!(resolved_default.is_default_region());
        assert!(resolved_default.region().is_full());
        Ok(())
    }

    #[test]
    fn conditioning_sets_have_content_digests_duplicate_guards_and_cancellation() -> TestResult {
        let (backend, authority) = runtime()?;
        let tensor = upload_f32(&backend, &authority, vec![1, 2], &[1.0, 2.0])?;
        let entry = ConditioningEntry::checked(
            "positive",
            ConditioningValue::regular(tensor)?,
            ConditioningEntryOptions::default(),
        )?;
        let cancellation = sim_types::CancellationToken::default();
        let first = ConditioningSet::checked(identity()?, vec![entry.clone()], &cancellation)?;
        let second = ConditioningSet::checked(identity()?, vec![entry.clone()], &cancellation)?;
        assert_eq!(first.digest(), second.digest());
        assert_eq!(first.digest().len(), 64);

        let source = upload_f32(&backend, &authority, vec![2, 1], &[0.0, 2.0])?;
        let narrowed = narrow_method_exact_native(&source, 0, 1, 1, &cancellation)?;
        let fresh = upload_f32(&backend, &authority, vec![1, 1], &[2.0])?;
        assert_ne!(narrowed.descriptor(), fresh.descriptor());
        assert_eq!(
            ConditioningValue::regular(narrowed)?.deterministic_digest(&cancellation)?,
            ConditioningValue::regular(fresh)?.deterministic_digest(&cancellation)?,
        );
        assert!(
            ConditioningSet::checked(identity()?, vec![entry.clone(), entry], &cancellation,)
                .is_err()
        );

        let cancelled = sim_types::CancellationToken::default();
        assert!(cancelled.cancel());
        assert!(matches!(
            ConditioningSet::checked(
                identity()?,
                vec![ConditioningEntry::checked(
                    "negative",
                    ConditioningValue::constant(ConditioningConstant::Boolean(false))?,
                    ConditioningEntryOptions::default(),
                )?],
                &cancelled,
            ),
            Err(ConditioningError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn conditioning_set_validation_and_residency_cover_metadata_and_unique_tensor_aliases()
    -> TestResult {
        let (backend, authority) = runtime()?;
        let cancellation = sim_types::CancellationToken::default();
        let aliased_tensor = upload_f32(&backend, &authority, vec![1, 1, 2], &[0.25, 0.75])?;
        let aliased = structured_conditioning_set(vec![aliased_tensor.clone(); 5], &cancellation)?;
        let mut distinct_tensors = Vec::new();
        for _ in 0..5 {
            distinct_tensors.push(upload_f32(
                &backend,
                &authority,
                vec![1, 1, 2],
                &[0.25, 0.75],
            )?);
        }
        let distinct = structured_conditioning_set(distinct_tensors, &cancellation)?;

        aliased.validate()?;
        distinct.validate()?;
        let aliased_parts = aliased.resident_parts()?;
        assert!(aliased_parts.owned_bytes() > 0);
        assert_eq!(aliased_parts.resident_bytes()?, aliased.resident_bytes()?);
        assert_eq!(aliased_parts.tensor_allocations().len(), 1);
        let aliased_allocation = aliased_parts
            .tensor_allocations()
            .first()
            .ok_or("aliased conditioning did not expose its tensor allocation")?;
        assert_eq!(aliased_allocation.storage_id(), aliased_tensor.storage_id());
        assert_eq!(
            aliased_allocation.resident_bytes(),
            aliased_tensor.storage_byte_len()
        );
        let distinct_parts = distinct.resident_parts()?;
        assert_eq!(distinct_parts.resident_bytes()?, distinct.resident_bytes()?);
        assert_eq!(distinct_parts.tensor_allocations().len(), 5);
        assert!(
            distinct_parts
                .tensor_allocations()
                .windows(2)
                .all(|pair| pair[0].storage_id().get() < pair[1].storage_id().get())
        );
        assert_eq!(aliased.digest(), distinct.digest());
        assert_eq!(
            distinct
                .resident_bytes()?
                .checked_sub(aliased.resident_bytes()?)
                .ok_or("aliased conditioning residency exceeded distinct residency")?,
            aliased_tensor
                .storage_byte_len()
                .checked_mul(4)
                .ok_or("conditioning test storage byte count overflowed")?
        );
        assert_eq!(
            ConditioningSet::combined_resident_bytes(&[&aliased, &aliased])?,
            aliased.resident_bytes()?
        );
        let aliased_clone = aliased.clone();
        assert_eq!(
            aliased_clone.resident_parts()?.tensor_allocations(),
            aliased_parts.tensor_allocations()
        );
        assert_eq!(
            ConditioningSet::combined_resident_bytes(&[&aliased, &aliased_clone])?,
            aliased
                .resident_bytes()?
                .checked_add(aliased_clone.resident_bytes()?)
                .and_then(|bytes| bytes.checked_sub(aliased_tensor.storage_byte_len()))
                .ok_or("combined conditioning resident byte count overflowed")?
        );

        let mut forged_digest = aliased.clone();
        forged_digest.digest = "0".repeat(64);
        assert!(matches!(
            forged_digest.validate(),
            Err(ConditioningError::Invalid(message))
                if message.contains("digest no longer matches")
        ));
        let mut duplicated_entry = aliased;
        let duplicate = duplicated_entry
            .entries
            .first()
            .cloned()
            .ok_or("conditioning set unexpectedly contained no entry")?;
        duplicated_entry.entries.push(duplicate);
        assert!(matches!(
            duplicated_entry.validate(),
            Err(ConditioningError::Invalid(message)) if message.contains("duplicated")
        ));
        Ok(())
    }

    #[test]
    fn cancellation_and_workspace_failures_propagate_without_private_fallbacks() -> TestResult {
        let (backend, authority) = runtime()?;
        let tensor = upload_f32(&backend, &authority, vec![1, 2], &[1.0, 2.0])?;
        let value = ConditioningValue::regular(tensor.clone())?;

        let cancelled = sim_types::CancellationToken::default();
        assert!(cancelled.cancel());
        let cancelled_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(40)?,
            &cancelled,
        );
        assert!(matches!(
            value.process(5, None, &backend, &cancelled_context),
            Err(ConditioningError::Cancelled)
        ));

        let cancellation = sim_types::CancellationToken::default();
        let scratch = authority.authorize_workspace(4)?;
        let context = backend.execution_context(StreamId::DEFAULT, scratch.clone(), &cancellation);
        assert!(matches!(
            value.process(5, None, &backend, &context),
            Err(ConditioningError::ShapeOperation(_))
        ));
        assert_eq!(scratch.in_use_bytes(), 0);
        assert_eq!(tensor_values(&tensor)?, [1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn mask_dtype_and_cross_attention_device_dtype_contracts_fail_closed() -> TestResult {
        let (backend, authority) = runtime()?;
        let cancellation = sim_types::CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(0)?,
            &cancellation,
        );
        let integer_descriptor =
            TensorDescriptor::contiguous(vec![2, 2], DType::I32, DeviceId::CPU, StreamId::DEFAULT)?;
        let integer_tensor = backend
            .upload_bytes(integer_descriptor, &[0; 16], &context)?
            .0;
        assert!(ConditioningMask::new(integer_tensor, 1.0, vec![0, 0], false).is_err());

        let f32_value = ConditioningValue::cross_attention(upload_f32(
            &backend,
            &authority,
            vec![1, 2, 1],
            &[0.0, 1.0],
        )?)?;
        let f16_descriptor = TensorDescriptor::contiguous(
            vec![1, 2, 1],
            DType::F16,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let f16_tensor = backend.upload_bytes(f16_descriptor, &[0; 4], &context)?.0;
        let f16_value = ConditioningValue::cross_attention(f16_tensor)?;
        assert!(!f32_value.can_concat(&f16_value));
        Ok(())
    }

    #[test]
    fn task403_typed_metadata_is_sealed_bounded_and_sampler_strict() -> TestResult {
        let (backend, authority) = runtime()?;
        let tensor = upload_f32(&backend, &authority, vec![1, 1, 2, 2], &[1.0; 4])?;
        let source_region = ConditioningRegion::percentage(vec![0.0, 0.5], vec![0.0, 0.25])?;
        let source_range = ConditioningTimestepRange::source(0.9, 0.1)?;
        let control = ConditioningControlBinding::checked(digest_hex('a'), false)?;
        let metadata = ConditioningMetadataMap::checked([
            (
                ConditioningMetadataKey::Region,
                ConditioningMetadataValue::Region(source_region.clone()),
            ),
            (
                ConditioningMetadataKey::Mask,
                ConditioningMetadataValue::Tensor(tensor.clone()),
            ),
            (
                ConditioningMetadataKey::TimestepRange,
                ConditioningMetadataValue::TimestepRange(source_range),
            ),
            (
                ConditioningMetadataKey::Control,
                ConditioningMetadataValue::Control(control),
            ),
            (
                ConditioningMetadataKey::SecondsStart,
                ConditioningMetadataValue::FiniteF64(0.25),
            ),
            (
                ConditioningMetadataKey::SecondsTotal,
                ConditioningMetadataValue::FiniteF64(5.0),
            ),
        ])?;
        assert!(metadata.sampler_window().is_err());
        assert!(source_region.resolve(&[8, 8]).is_err());
        assert!(
            ConditioningMetadataMap::checked([(
                ConditioningMetadataKey::Mask,
                ConditioningMetadataValue::Boolean(true),
            )])
            .is_err()
        );
        assert!(
            ConditioningMetadataMap::checked([(
                ConditioningMetadataKey::ReferenceLatents,
                ConditioningMetadataValue::TensorList(vec![tensor.clone(); 65]),
            )])
            .is_err()
        );
        assert!(
            ConditioningMetadataMap::checked([(
                ConditioningMetadataKey::SecondsStart,
                ConditioningMetadataValue::FiniteF64(0.25),
            )])
            .is_err()
        );
        assert!(
            ConditioningMetadataMap::checked([
                (
                    ConditioningMetadataKey::Control,
                    ConditioningMetadataValue::Control(ConditioningControlBinding::checked(
                        digest_hex('b'),
                        false,
                    )?),
                ),
                (
                    ConditioningMetadataKey::ControlApplyToUnconditioned,
                    ConditioningMetadataValue::Boolean(true),
                ),
            ])
            .is_err()
        );
        assert!(
            ConditioningMetadataMap::checked([(
                ConditioningMetadataKey::DrivingMask28Channels,
                ConditioningMetadataValue::Tensor(tensor.clone()),
            )])
            .is_err()
        );
        assert!(
            ConditioningMetadataMap::checked([
                (
                    ConditioningMetadataKey::VaceFrames,
                    ConditioningMetadataValue::TensorList(vec![tensor.clone()]),
                ),
                (
                    ConditioningMetadataKey::VaceMask,
                    ConditioningMetadataValue::TensorList(vec![tensor.clone(), tensor]),
                ),
                (
                    ConditioningMetadataKey::VaceStrength,
                    ConditioningMetadataValue::FiniteF64List(vec![1.0]),
                ),
            ])
            .is_err()
        );

        let entry = ConditioningEntry::checked(
            "entry",
            ConditioningValue::regular(upload_f32(&backend, &authority, vec![1, 2], &[1.0, 2.0])?)?,
            ConditioningEntryOptions::default(),
        )?;
        let cancellation = sim_types::CancellationToken::default();
        let set = ConditioningSet::checked(identity()?, vec![entry], &cancellation)?;
        let rebound = set.with_entry_metadata(vec![metadata], &cancellation)?;
        assert_eq!(rebound.entries().len(), set.entries().len());
        assert_ne!(rebound.digest(), set.digest());
        rebound.validate()?;
        Ok(())
    }

    fn digest_hex(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }
    #[test]
    fn controlnet_text_metadata_preserves_source_aliases_overwrite_none_and_digest_accounting()
    -> TestResult {
        let (backend, authority) = runtime()?;
        let cancellation = sim_types::CancellationToken::default();
        let original = upload_f32(&backend, &authority, vec![1, 2, 2], &[1.0, 2.0, 3.0, 4.0])?;
        let encoded = upload_f32(&backend, &authority, vec![1, 2, 2], &[5.0, 6.0, 7.0, 8.0])?;
        let pooled = upload_f32(&backend, &authority, vec![1, 2], &[9.0, 10.0])?;
        let mut entries = Vec::new();
        for identifier in ["positive", "secondary"] {
            entries.push(ConditioningEntry::checked_with_metadata(
                identifier,
                ConditioningValue::cross_attention(original.clone())?,
                ConditioningEntryOptions {
                    strength: 0.5,
                    ..Default::default()
                },
                ConditioningReferences::default(),
                ConditioningMetadataMap::checked([
                    (
                        ConditioningMetadataKey::TimestepRange,
                        ConditioningMetadataValue::TimestepRange(
                            ConditioningTimestepRange::source(0.2, 0.8)?,
                        ),
                    ),
                    (
                        ConditioningMetadataKey::CrossAttentionControlnet,
                        ConditioningMetadataValue::Tensor(original.clone()),
                    ),
                    (
                        ConditioningMetadataKey::PooledOutputControlnet,
                        ConditioningMetadataValue::ControlnetPooledOutput(None),
                    ),
                ])?,
            )?);
        }
        let input = ConditioningSet::checked(identity()?, entries, &cancellation)?;
        let before = input.digest().to_owned();
        let output = input.with_controlnet_text_encoding(&encoded, Some(&pooled), &cancellation)?;
        assert_eq!(input.digest(), before);
        assert_eq!(output.identity(), input.identity());
        assert_ne!(output.digest(), input.digest());
        for (before, after) in input.entries().iter().zip(output.entries()) {
            assert_eq!(before.identifier(), after.identifier());
            assert_eq!(before.options().strength, after.options().strength);
            let ConditioningValue::CrossAttention(value) = after.value() else {
                return Err("Primary conditioning lost".into());
            };
            assert_eq!(value.storage_id(), original.storage_id());
            let Some(ConditioningMetadataValue::Tensor(value)) = after
                .metadata()
                .get(ConditioningMetadataKey::CrossAttentionControlnet)
            else {
                return Err("Source cross-attention metadata missing".into());
            };
            assert_eq!(value.storage_id(), encoded.storage_id());
            let Some(ConditioningMetadataValue::ControlnetPooledOutput(Some(value))) = after
                .metadata()
                .get(ConditioningMetadataKey::PooledOutputControlnet)
            else {
                return Err("Source pooled metadata missing".into());
            };
            assert_eq!(value.storage_id(), pooled.storage_id());
            assert!(matches!(
                before
                    .metadata()
                    .get(ConditioningMetadataKey::PooledOutputControlnet),
                Some(ConditioningMetadataValue::ControlnetPooledOutput(None))
            ));
            assert!(
                matches!(after.metadata().get(ConditioningMetadataKey::TimestepRange), Some(ConditioningMetadataValue::TimestepRange(range)) if *range == ConditioningTimestepRange::source(0.2, 0.8)?)
            );
        }
        let parts = output.resident_parts()?;
        assert_eq!(parts.tensor_allocations().len(), 3);
        for tensor in [&original, &encoded, &pooled] {
            let matches = parts
                .tensor_allocations()
                .iter()
                .filter(|allocation| allocation.storage_id() == tensor.storage_id())
                .collect::<Vec<_>>();
            let [allocation] = matches.as_slice() else {
                return Err("Shared storage charged more than once".into());
            };
            assert_eq!(allocation.resident_bytes(), tensor.storage_byte_len());
        }
        let no_pooled = output.with_controlnet_text_encoding(&encoded, None, &cancellation)?;
        let without_key = no_pooled.with_entry_metadata(
            no_pooled
                .entries()
                .iter()
                .map(|entry| {
                    ConditioningMetadataMap::checked(
                        entry
                            .metadata()
                            .iter()
                            .filter(|(key, _)| {
                                **key != ConditioningMetadataKey::PooledOutputControlnet
                            })
                            .map(|(key, value)| (*key, value.clone())),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
            &cancellation,
        )?;
        assert_ne!(no_pooled.digest(), without_key.digest());
        assert_ne!(output.digest(), no_pooled.digest());
        assert_eq!(no_pooled.resident_parts()?.tensor_allocations().len(), 2);
        let repeated =
            no_pooled.with_controlnet_text_encoding(&encoded, Some(&pooled), &cancellation)?;
        assert_eq!(repeated.digest(), output.digest());
        repeated.validate()?;
        Ok(())
    }

    #[test]
    fn controlnet_text_metadata_rejects_wrong_types_shapes_and_cancellation_then_recovers()
    -> TestResult {
        let (backend, authority) = runtime()?;
        let cancellation = sim_types::CancellationToken::default();
        let tensor = upload_f32(&backend, &authority, vec![1, 1, 2], &[1.0, 2.0])?;
        let pooled = upload_f32(&backend, &authority, vec![1, 2], &[3.0, 4.0])?;
        let set = ConditioningSet::checked(
            identity()?,
            vec![ConditioningEntry::checked(
                "primary",
                ConditioningValue::cross_attention(tensor.clone())?,
                Default::default(),
            )?],
            &cancellation,
        )?;
        let before = set.digest().to_owned();
        assert!(
            ConditioningMetadataMap::checked([(
                ConditioningMetadataKey::CrossAttentionControlnet,
                ConditioningMetadataValue::ControlnetPooledOutput(None)
            )])
            .is_err()
        );
        assert!(
            ConditioningMetadataMap::checked([(
                ConditioningMetadataKey::SecondsTotal,
                ConditioningMetadataValue::ControlnetPooledOutput(None)
            )])
            .is_err()
        );
        assert!(
            set.with_controlnet_text_encoding(&pooled, None, &cancellation)
                .is_err()
        );
        assert!(
            set.with_controlnet_text_encoding(&tensor, Some(&tensor), &cancellation)
                .is_err()
        );
        let cancelled = sim_types::CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            set.with_controlnet_text_encoding(&tensor, Some(&pooled), &cancelled),
            Err(ConditioningError::Cancelled)
        ));
        assert_eq!(set.digest(), before);
        let recovered = set.with_controlnet_text_encoding(&tensor, Some(&pooled), &cancellation)?;
        recovered.validate()?;
        assert_eq!(recovered.resident_parts()?.tensor_allocations().len(), 2);
        Ok(())
    }

    #[test]
    fn ordinary_pooled_metadata_distinguishes_absent_none_and_tensor_content() -> TestResult {
        let (backend, authority) = runtime()?;
        let cancellation = sim_types::CancellationToken::default();
        let primary = upload_f32(&backend, &authority, vec![1, 1, 2], &[1.0, 2.0])?;
        let pooled = upload_f32(&backend, &authority, vec![2], &[3.0, 4.0])?;
        let equal_pooled = upload_f32(&backend, &authority, vec![2], &[3.0, 4.0])?;
        let set = ConditioningSet::checked(
            identity()?,
            vec![ConditioningEntry::checked(
                "primary",
                ConditioningValue::cross_attention(primary)?,
                Default::default(),
            )?],
            &cancellation,
        )?;
        let absent_digest = set.digest().to_owned();
        let none = set.with_entry_metadata(
            vec![ConditioningMetadataMap::checked([(
                ConditioningMetadataKey::PooledOutput,
                ConditioningMetadataValue::PooledOutput(None),
            )])?],
            &cancellation,
        )?;
        let tensor = set.with_entry_metadata(
            vec![ConditioningMetadataMap::checked([(
                ConditioningMetadataKey::PooledOutput,
                ConditioningMetadataValue::PooledOutput(Some(pooled.clone())),
            )])?],
            &cancellation,
        )?;
        let equal_tensor = set.with_entry_metadata(
            vec![ConditioningMetadataMap::checked([(
                ConditioningMetadataKey::PooledOutput,
                ConditioningMetadataValue::PooledOutput(Some(equal_pooled)),
            )])?],
            &cancellation,
        )?;
        assert_ne!(none.digest(), absent_digest);
        assert_ne!(tensor.digest(), none.digest());
        assert_ne!(tensor.digest(), absent_digest);
        assert_eq!(equal_tensor.digest(), tensor.digest());
        assert_eq!(tensor.clone().digest(), tensor.digest());
        assert_eq!(set.digest(), absent_digest);
        assert!(
            set.entries()
                .first()
                .ok_or("missing entry")?
                .metadata()
                .get(ConditioningMetadataKey::PooledOutput)
                .is_none()
        );
        assert!(matches!(
            none.entries()
                .first()
                .ok_or("missing entry")?
                .metadata()
                .get(ConditioningMetadataKey::PooledOutput),
            Some(ConditioningMetadataValue::PooledOutput(None))
        ));
        let Some(ConditioningMetadataValue::PooledOutput(Some(retained))) = tensor
            .entries()
            .first()
            .ok_or("missing entry")?
            .metadata()
            .get(ConditioningMetadataKey::PooledOutput)
        else {
            return Err("missing pooled tensor".into());
        };
        assert_eq!(retained.storage_id(), pooled.storage_id());
        Ok(())
    }

    #[test]
    fn ordinary_pooled_metadata_preserves_controlnet_separation_and_shared_storage_accounting()
    -> TestResult {
        let (backend, authority) = runtime()?;
        let cancellation = sim_types::CancellationToken::default();
        let tensor = upload_f32(&backend, &authority, vec![1, 2], &[3.0, 4.0])?;
        let metadata = ConditioningMetadataMap::checked([
            (
                ConditioningMetadataKey::PooledOutput,
                ConditioningMetadataValue::PooledOutput(Some(tensor.clone())),
            ),
            (
                ConditioningMetadataKey::PooledOutputControlnet,
                ConditioningMetadataValue::ControlnetPooledOutput(Some(tensor.clone())),
            ),
        ])?;
        assert_eq!(metadata.resident_tensor_bytes()?, tensor.storage_byte_len());
        let entries = ["first", "second"]
            .into_iter()
            .map(|identifier| {
                ConditioningEntry::checked_with_metadata(
                    identifier,
                    ConditioningValue::regular(tensor.clone())?,
                    Default::default(),
                    Default::default(),
                    metadata.clone(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let set = ConditioningSet::checked(identity()?, entries, &cancellation)?;
        let parts = set.resident_parts()?;
        let [allocation] = parts.tensor_allocations() else {
            return Err("aliased pooled storage charged more than once".into());
        };
        assert_eq!(allocation.storage_id(), tensor.storage_id());
        assert_eq!(allocation.resident_bytes(), tensor.storage_byte_len());
        let mut changed = metadata;
        changed.insert(
            ConditioningMetadataKey::PooledOutput,
            ConditioningMetadataValue::PooledOutput(None),
        )?;
        assert!(
            matches!(changed.get(ConditioningMetadataKey::PooledOutputControlnet), Some(ConditioningMetadataValue::ControlnetPooledOutput(Some(value))) if value.storage_id() == tensor.storage_id())
        );
        for (key, value) in [
            (
                ConditioningMetadataKey::PooledOutput,
                ConditioningMetadataValue::ControlnetPooledOutput(None),
            ),
            (
                ConditioningMetadataKey::PooledOutputControlnet,
                ConditioningMetadataValue::PooledOutput(None),
            ),
            (
                ConditioningMetadataKey::Mask,
                ConditioningMetadataValue::PooledOutput(Some(tensor)),
            ),
        ] {
            assert!(ConditioningMetadataMap::checked([(key, value)]).is_err());
        }
        Ok(())
    }

    #[test]
    fn ordinary_pooled_metadata_uses_generic_tensor_admission_without_batch_or_rank_two_policy()
    -> TestResult {
        let (backend, authority) = runtime()?;
        let cancellation = sim_types::CancellationToken::default();
        let primary = upload_f32(&backend, &authority, vec![1, 1, 2], &[1.0, 2.0])?;
        let set = ConditioningSet::checked(
            identity()?,
            vec![ConditioningEntry::checked(
                "primary",
                ConditioningValue::cross_attention(primary)?,
                Default::default(),
            )?],
            &cancellation,
        )?;
        for (shape, values) in [
            (vec![2], vec![1.0, 2.0]),
            (vec![3, 2], vec![1.0; 6]),
            (vec![2, 1, 3], vec![1.0; 6]),
        ] {
            let tensor = upload_f32(&backend, &authority, shape, &values)?;
            let output = set.with_entry_metadata(
                vec![ConditioningMetadataMap::checked([(
                    ConditioningMetadataKey::PooledOutput,
                    ConditioningMetadataValue::PooledOutput(Some(tensor)),
                )])?],
                &cancellation,
            )?;
            output.validate()?;
        }
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(0)?,
            &cancellation,
        );
        let integer = backend
            .upload_bytes(
                TensorDescriptor::contiguous(
                    vec![3],
                    DType::I32,
                    DeviceId::CPU,
                    StreamId::DEFAULT,
                )?,
                &[0; 12],
                &context,
            )?
            .0;
        let integer_output = set.with_entry_metadata(
            vec![ConditioningMetadataMap::checked([(
                ConditioningMetadataKey::PooledOutput,
                ConditioningMetadataValue::PooledOutput(Some(integer)),
            )])?],
            &cancellation,
        )?;
        integer_output.validate()?;
        for (shape, values) in [(vec![], vec![1.0]), (vec![0], vec![])] {
            let tensor = upload_f32(&backend, &authority, shape, &values)?;
            assert!(
                ConditioningMetadataMap::checked([(
                    ConditioningMetadataKey::PooledOutput,
                    ConditioningMetadataValue::PooledOutput(Some(tensor))
                )])
                .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn ordinary_pooled_metadata_does_not_increase_the_existing_entry_bound() -> TestResult {
        use ConditioningMetadataKey as Key;
        use ConditioningMetadataValue as Value;
        let (backend, authority) = runtime()?;
        let latent = upload_f32(&backend, &authority, vec![1, 1, 1, 1], &[1.0])?;
        let video = upload_f32(&backend, &authority, vec![1, 1, 1, 1, 1], &[1.0])?;
        let matrix = upload_f32(&backend, &authority, vec![1, 2], &[1.0, 2.0])?;
        let attention = upload_f32(&backend, &authority, vec![1, 1, 2], &[1.0, 2.0])?;
        let face = upload_f32(&backend, &authority, vec![1, 1, 1, 1, 3], &[1.0; 3])?;
        let mask = upload_f32(&backend, &authority, vec![1, 1, 1, 1, 28], &[1.0; 28])?;
        let values = vec![
            (
                Key::Region,
                Value::Region(ConditioningRegion::absolute(vec![1, 1], vec![0, 0])?),
            ),
            (Key::Mask, Value::Tensor(matrix.clone())),
            (Key::MaskStrength, Value::FiniteF64(1.0)),
            (Key::SetAreaToBounds, Value::Boolean(false)),
            (
                Key::TimestepRange,
                Value::TimestepRange(ConditioningTimestepRange::source(0.0, 1.0)?),
            ),
            (Key::ConcatLatentImage, Value::Tensor(latent.clone())),
            (Key::ConcatMask, Value::Tensor(latent.clone())),
            (Key::ConcatMaskIndex, Value::Unsigned(0)),
            (
                Key::Control,
                Value::Control(ConditioningControlBinding::checked("a".repeat(64), false)?),
            ),
            (Key::ControlApplyToUnconditioned, Value::Boolean(false)),
            (
                Key::ReferenceLatents,
                Value::TensorList(vec![latent.clone()]),
            ),
            (Key::VaceFrames, Value::TensorList(vec![latent.clone()])),
            (Key::VaceMask, Value::TensorList(vec![latent.clone()])),
            (Key::VaceStrength, Value::FiniteF64List(vec![1.0])),
            (Key::CameraConditions, Value::Tensor(matrix.clone())),
            (Key::TimeDimensionConcat, Value::Tensor(matrix.clone())),
            (Key::AudioEmbedding, Value::Tensor(attention.clone())),
            (Key::ReferenceMotion, Value::Tensor(latent)),
            (Key::ControlVideo, Value::Tensor(video.clone())),
            (Key::PoseVideoLatent, Value::Tensor(video)),
            (Key::FaceVideoPixels, Value::Tensor(face)),
            (Key::DrivingMask28Channels, Value::Tensor(mask.clone())),
            (Key::ReferenceMask28Channels, Value::Tensor(mask)),
            (Key::ReferenceMaskFlag, Value::Boolean(true)),
            (Key::SecondsStart, Value::FiniteF64(0.0)),
            (Key::SecondsTotal, Value::FiniteF64(1.0)),
            (Key::CrossAttentionControlnet, Value::Tensor(attention)),
            (
                Key::PooledOutputControlnet,
                Value::ControlnetPooledOutput(Some(matrix)),
            ),
        ];
        assert_eq!(values.len(), MAX_CONDITIONING_METADATA_ENTRIES);
        let mut metadata = ConditioningMetadataMap::checked(values.clone())?;
        metadata.validate()?;
        assert!(matches!(
            metadata.insert(Key::PooledOutput, Value::PooledOutput(None)),
            Err(ConditioningError::Invalid(_))
        ));
        assert_eq!(metadata.len(), 28);
        assert!(metadata.get(Key::PooledOutput).is_none());
        assert!(
            ConditioningMetadataMap::checked(
                values
                    .into_iter()
                    .chain([(Key::PooledOutput, Value::PooledOutput(None))])
            )
            .is_err()
        );
        metadata.insert(Key::SecondsTotal, Value::FiniteF64(2.0))?;
        assert_eq!(metadata.len(), 28);
        assert!(
            matches!(metadata.get(Key::SecondsTotal),Some(Value::FiniteF64(value)) if *value == 2.0)
        );
        Ok(())
    }

    #[test]
    fn ordinary_pooled_metadata_cancellation_preserves_input_then_recovers() -> TestResult {
        let (backend, authority) = runtime()?;
        let cancellation = sim_types::CancellationToken::default();
        let tensor = upload_f32(&backend, &authority, vec![1, 1, 2], &[1.0, 2.0])?;
        let set = ConditioningSet::checked(
            identity()?,
            vec![ConditioningEntry::checked(
                "primary",
                ConditioningValue::cross_attention(tensor.clone())?,
                Default::default(),
            )?],
            &cancellation,
        )?;
        let before = set.digest().to_owned();
        let metadata = ConditioningMetadataMap::checked([(
            ConditioningMetadataKey::PooledOutput,
            ConditioningMetadataValue::PooledOutput(Some(tensor)),
        )])?;
        let cancelled = sim_types::CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            set.with_entry_metadata(vec![metadata.clone()], &cancelled),
            Err(ConditioningError::Cancelled)
        ));
        assert_eq!(set.digest(), before);
        let recovered = set.with_entry_metadata(vec![metadata], &cancellation)?;
        assert_eq!(recovered.resident_parts()?.tensor_allocations().len(), 1);
        recovered.validate()?;
        Ok(())
    }

    #[test]
    fn source_percentage_endpoint_is_stored_without_inventing_sampler_consumption() -> TestResult {
        let (backend, authority) = runtime()?;
        let cancellation = sim_types::CancellationToken::default();
        let tensor = upload_f32(&backend, &authority, vec![1, 1, 2], &[1.0, 2.0])?;
        let region = ConditioningRegion::percentage(vec![0.0, 1.0], vec![1.0, -0.0])?;
        let metadata = ConditioningMetadataMap::checked([(
            ConditioningMetadataKey::Region,
            ConditioningMetadataValue::Region(region.clone()),
        )])?;
        metadata.validate()?;
        let set = ConditioningSet::checked(
            identity()?,
            vec![ConditioningEntry::checked_with_metadata(
                "source.edge",
                ConditioningValue::cross_attention(tensor.clone())?,
                ConditioningEntryOptions {
                    region: Some(region.clone()),
                    ..Default::default()
                },
                Default::default(),
                metadata,
            )?],
            &cancellation,
        )?;
        set.validate()?;
        assert_eq!(
            set.entries()
                .first()
                .ok_or("Missing entry")?
                .options()
                .region
                .as_ref(),
            Some(&region)
        );
        assert_eq!(set.resident_parts()?.tensor_allocations().len(), 1);
        assert!(region.resolve(&[8, 8]).is_err());
        let endpoint = ConditioningRegion::percentage(vec![1.0], vec![1.0])?;
        assert!(endpoint.resolve(&[8]).is_err());
        assert!(ConditioningRegion::percentage(vec![1.0], vec![1.00001]).is_err());
        assert!(ConditioningRegion::percentage(vec![1.0], vec![-0.00001]).is_err());
        assert!(ConditioningRegion::percentage(vec![1.0], vec![f64::NAN]).is_err());
        assert_eq!(tensor_values(&tensor)?, [1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn source_audio_nonnegative_timing_pairs_preserve_aliases_and_digest() -> TestResult {
        use ConditioningMetadataKey as Key;
        use ConditioningMetadataValue as Value;
        let (backend, authority) = runtime()?;
        let cancellation = sim_types::CancellationToken::default();
        let tensor = upload_f32(&backend, &authority, vec![1, 1, 2], &[2.0, 4.0])?;
        let set = ConditioningSet::checked(
            identity()?,
            vec![ConditioningEntry::checked(
                "timing",
                ConditioningValue::cross_attention(tensor.clone())?,
                Default::default(),
            )?],
            &cancellation,
        )?;
        let original = set.digest().to_owned();
        let mut digests = std::collections::BTreeSet::new();
        for (start, total) in [(0.0, 0.0), (-0.0, -0.0), (50.0, 47.125), (1000.0, 0.0)] {
            let metadata = ConditioningMetadataMap::checked([
                (Key::SecondsStart, Value::FiniteF64(start)),
                (Key::SecondsTotal, Value::FiniteF64(total)),
                (Key::PooledOutput, Value::PooledOutput(Some(tensor.clone()))),
            ])?;
            metadata.validate()?;
            let output = set.with_entry_metadata(vec![metadata.clone()], &cancellation)?;
            let replay = set.with_entry_metadata(vec![metadata], &cancellation)?;
            assert_eq!(output.digest(), replay.digest());
            assert!(digests.insert(output.digest().to_owned()));
            assert_eq!(output.resident_parts()?.tensor_allocations().len(), 1);
            assert!(
                matches!(output.entries().first().ok_or("Missing output entry")?.value(), ConditioningValue::CrossAttention(value) if value.storage_id() == tensor.storage_id())
            );
            assert_eq!(
                output
                    .entries()
                    .first()
                    .ok_or("Missing output entry")?
                    .identifier(),
                "timing"
            );
        }
        for values in [
            vec![(Key::SecondsStart, Value::FiniteF64(0.0))],
            vec![(Key::SecondsTotal, Value::FiniteF64(0.0))],
            vec![
                (Key::SecondsStart, Value::FiniteF64(-0.1)),
                (Key::SecondsTotal, Value::FiniteF64(1.0)),
            ],
            vec![
                (Key::SecondsStart, Value::FiniteF64(0.0)),
                (Key::SecondsTotal, Value::FiniteF64(f64::INFINITY)),
            ],
        ] {
            assert!(ConditioningMetadataMap::checked(values).is_err());
        }
        assert_eq!(set.digest(), original);
        assert_eq!(tensor_values(&tensor)?, [2.0, 4.0]);
        Ok(())
    }

    #[test]
    fn binary64_entry_strength_retains_source_bits_digest_and_aliases() -> TestResult {
        let (backend, authority) = runtime()?;
        let cancellation = sim_types::CancellationToken::default();
        let tensor = upload_f32(&backend, &authority, vec![1, 1, 2], &[2.0, 4.0])?;
        let make_set = |strength| -> Result<ConditioningSet, ConditioningError> {
            ConditioningSet::checked(
                identity().map_err(|error| ConditioningError::Invalid(error.to_string()))?,
                vec![ConditioningEntry::checked(
                    "source-strength",
                    ConditioningValue::cross_attention(tensor.clone())?,
                    ConditioningEntryOptions {
                        strength,
                        ..Default::default()
                    },
                )?],
                &cancellation,
            )
        };
        let source = make_set(0.1)?;
        let rounded = make_set(f64::from(0.1_f32))?;
        let adjacent = make_set(f64::from_bits(0.1_f64.to_bits() + 1))?;
        assert_ne!(source.digest(), rounded.digest());
        assert_ne!(source.digest(), adjacent.digest());
        assert_ne!(make_set(0.0)?.digest(), make_set(-0.0)?.digest());
        assert_eq!(
            source
                .entries()
                .first()
                .ok_or("Source entry")?
                .options()
                .strength
                .to_bits(),
            0.1_f64.to_bits()
        );
        let replay =
            source.with_entry_metadata(vec![ConditioningMetadataMap::default()], &cancellation)?;
        assert_eq!(source.digest(), replay.digest());
        assert_eq!(source.resident_parts()?.tensor_allocations().len(), 1);
        assert_eq!(replay.resident_parts()?.tensor_allocations().len(), 1);
        let ConditioningValue::CrossAttention(actual) =
            replay.entries().first().ok_or("Replay entry")?.value()
        else {
            return Err("Cross attention".into());
        };
        assert_eq!(actual.storage_id(), tensor.storage_id());
        source.validate()?;
        rounded.validate()?;
        adjacent.validate()?;
        assert!(make_set(f64::NAN).is_err());
        assert!(make_set(f64::INFINITY).is_err());
        Ok(())
    }

    #[test]
    fn binary64_resolved_strength_keeps_exact_scalar_until_cpu_f32_weight_boundary() -> TestResult {
        let (backend, authority) = runtime()?;
        let cancellation = sim_types::CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024)?,
            &cancellation,
        );
        let value = upload_f32(&backend, &authority, vec![1, 1], &[1.0])?;
        let target = TensorDescriptor::contiguous(
            vec![1, 1, 2],
            DType::F32,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        for strength in [
            0.1,
            f64::from_bits(0.1_f64.to_bits() + 1),
            -0.0,
            1e-46,
            10.0,
        ] {
            let entry = ConditioningEntry::checked(
                "source",
                ConditioningValue::regular(value.clone())?,
                ConditioningEntryOptions {
                    strength,
                    ..Default::default()
                },
            )?;
            let resolved = entry.resolve(&target, &backend, &context)?;
            assert_eq!(resolved.strength().to_bits(), strength.to_bits());
            assert_eq!(
                resolved.strength_for_cpu_f32()?.to_bits(),
                (strength as f32).to_bits()
            );
            assert_eq!(
                resolved.contribution_weight(&[0], None)?.to_bits(),
                (strength as f32).to_bits()
            );
        }
        let oversized = ConditioningEntry::checked(
            "large-source",
            ConditioningValue::regular(value)?,
            ConditioningEntryOptions {
                strength: 1e300,
                ..Default::default()
            },
        )?;
        let resolved = oversized.resolve(&target, &backend, &context)?;
        assert_eq!(resolved.strength(), 1e300);
        assert!(resolved.strength_for_cpu_f32().is_err());
        assert!(resolved.contribution_weight(&[0], None).is_err());
        assert!(resolved.contribution_weight(&[2], None).is_err());
        assert!(resolved.contribution_weight(&[0], Some(1.0)).is_err());
        Ok(())
    }

    #[test]
    fn binary64_strength_metadata_update_cancellation_preserves_source_and_recovers() -> TestResult
    {
        let (backend, authority) = runtime()?;
        let tensor = upload_f32(&backend, &authority, vec![1, 1, 2], &[1.0, 2.0])?;
        let cancellation = sim_types::CancellationToken::default();
        let source = ConditioningSet::checked(
            identity()?,
            vec![ConditioningEntry::checked(
                "source",
                ConditioningValue::cross_attention(tensor)?,
                ConditioningEntryOptions {
                    strength: 0.1,
                    ..Default::default()
                },
            )?],
            &cancellation,
        )?;
        let digest = source.digest().to_owned();
        let cancelled = sim_types::CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            source.with_entry_metadata(vec![ConditioningMetadataMap::default()], &cancelled),
            Err(ConditioningError::Cancelled)
        ));
        assert_eq!(source.digest(), digest);
        let output =
            source.with_entry_metadata(vec![ConditioningMetadataMap::default()], &cancellation)?;
        assert_eq!(output.digest(), digest);
        assert_eq!(
            output
                .entries()
                .first()
                .ok_or("Output")?
                .options()
                .strength
                .to_bits(),
            0.1_f64.to_bits()
        );
        Ok(())
    }
    #[test]
    fn empty_conditioning_preserves_identity_digest_and_owned_residency() -> TestResult {
        let cancellation = sim_types::CancellationToken::default();
        let empty = ConditioningSet::checked(identity()?, Vec::new(), &cancellation)?;
        empty.validate()?;
        assert!(empty.entries().is_empty());
        assert!(empty.resident_bytes()? > 0);
        assert!(empty.resident_parts()?.tensor_allocations().is_empty());
        assert_eq!(
            ConditioningSet::combined_resident_bytes(&[&empty, &empty])?,
            empty.resident_bytes()?
        );
        let replay = empty.with_entry_metadata(Vec::new(), &cancellation)?;
        assert_eq!(replay.identity(), empty.identity());
        assert_eq!(replay.digest(), empty.digest());
        assert!(
            empty
                .with_entry_metadata(vec![ConditioningMetadataMap::default()], &cancellation)
                .is_err()
        );
        let nullable = ConditioningSet::checked(
            identity()?,
            vec![ConditioningEntry::checked(
                "null",
                ConditioningValue::constant(ConditioningConstant::Null)?,
                ConditioningEntryOptions::default(),
            )?],
            &cancellation,
        )?;
        assert_ne!(nullable.digest(), empty.digest());
        let mut invalid_digest = empty;
        invalid_digest.digest = "0".repeat(64);
        assert!(invalid_digest.validate().is_err());
        Ok(())
    }

    #[test]
    fn empty_conditioning_retains_identity_and_cancellation_validation() -> TestResult {
        let cancellation = sim_types::CancellationToken::default();
        let mut invalid_identity = identity()?;
        invalid_identity.namespace.clear();
        assert!(ConditioningSet::checked(invalid_identity, Vec::new(), &cancellation).is_err());
        assert!(cancellation.cancel());
        assert!(matches!(
            ConditioningSet::checked(identity()?, Vec::new(), &cancellation),
            Err(ConditioningError::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn task618_exact_lyrics_metadata_distinguishes_absent_none_and_tensor_content() -> TestResult {
        let (backend, authority) = runtime()?;
        let cancellation = sim_types::CancellationToken::default();
        let primary = upload_f32(&backend, &authority, vec![1, 1, 2], &[1.0, 2.0])?;
        let pooled = upload_f32(&backend, &authority, vec![2], &[3.0, 4.0])?;
        let equal_pooled = upload_f32(&backend, &authority, vec![2], &[3.0, 4.0])?;
        let set = ConditioningSet::checked(
            identity()?,
            vec![ConditioningEntry::checked(
                "primary",
                ConditioningValue::cross_attention(primary)?,
                Default::default(),
            )?],
            &cancellation,
        )?;
        let absent_digest = set.digest().to_owned();
        let none = set.with_entry_metadata(
            vec![ConditioningMetadataMap::checked([(
                ConditioningMetadataKey::ConditioningLyrics,
                ConditioningMetadataValue::ConditioningLyrics(None),
            )])?],
            &cancellation,
        )?;
        let tensor = set.with_entry_metadata(
            vec![ConditioningMetadataMap::checked([(
                ConditioningMetadataKey::ConditioningLyrics,
                ConditioningMetadataValue::ConditioningLyrics(Some(pooled.clone())),
            )])?],
            &cancellation,
        )?;
        let equal_tensor = set.with_entry_metadata(
            vec![ConditioningMetadataMap::checked([(
                ConditioningMetadataKey::ConditioningLyrics,
                ConditioningMetadataValue::ConditioningLyrics(Some(equal_pooled)),
            )])?],
            &cancellation,
        )?;
        assert_ne!(none.digest(), absent_digest);
        assert_ne!(tensor.digest(), none.digest());
        assert_ne!(tensor.digest(), absent_digest);
        assert_eq!(equal_tensor.digest(), tensor.digest());
        assert_eq!(tensor.clone().digest(), tensor.digest());
        assert_eq!(set.digest(), absent_digest);
        assert!(
            set.entries()
                .first()
                .ok_or("missing entry")?
                .metadata()
                .get(ConditioningMetadataKey::ConditioningLyrics)
                .is_none()
        );
        assert!(matches!(
            none.entries()
                .first()
                .ok_or("missing entry")?
                .metadata()
                .get(ConditioningMetadataKey::ConditioningLyrics),
            Some(ConditioningMetadataValue::ConditioningLyrics(None))
        ));
        let Some(ConditioningMetadataValue::ConditioningLyrics(Some(retained))) = tensor
            .entries()
            .first()
            .ok_or("missing entry")?
            .metadata()
            .get(ConditioningMetadataKey::ConditioningLyrics)
        else {
            return Err("missing pooled tensor".into());
        };
        assert_eq!(retained.storage_id(), pooled.storage_id());
        Ok(())
    }

    #[test]
    fn task618_exact_lyrics_metadata_uses_generic_tensor_admission_without_batch_or_rank_two_policy()
    -> TestResult {
        let (backend, authority) = runtime()?;
        let cancellation = sim_types::CancellationToken::default();
        let primary = upload_f32(&backend, &authority, vec![1, 1, 2], &[1.0, 2.0])?;
        let set = ConditioningSet::checked(
            identity()?,
            vec![ConditioningEntry::checked(
                "primary",
                ConditioningValue::cross_attention(primary)?,
                Default::default(),
            )?],
            &cancellation,
        )?;
        for (shape, values) in [
            (vec![2], vec![1.0, 2.0]),
            (vec![3, 2], vec![1.0; 6]),
            (vec![2, 1, 3], vec![1.0; 6]),
        ] {
            let tensor = upload_f32(&backend, &authority, shape, &values)?;
            let output = set.with_entry_metadata(
                vec![ConditioningMetadataMap::checked([(
                    ConditioningMetadataKey::ConditioningLyrics,
                    ConditioningMetadataValue::ConditioningLyrics(Some(tensor)),
                )])?],
                &cancellation,
            )?;
            output.validate()?;
        }
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(0)?,
            &cancellation,
        );
        let integer = backend
            .upload_bytes(
                TensorDescriptor::contiguous(
                    vec![3],
                    DType::I64,
                    DeviceId::CPU,
                    StreamId::DEFAULT,
                )?,
                &[0; 24],
                &context,
            )?
            .0;
        let integer_output = set.with_entry_metadata(
            vec![ConditioningMetadataMap::checked([(
                ConditioningMetadataKey::ConditioningLyrics,
                ConditioningMetadataValue::ConditioningLyrics(Some(integer)),
            )])?],
            &cancellation,
        )?;
        integer_output.validate()?;
        for (shape, values) in [(vec![], vec![1.0]), (vec![0], vec![])] {
            let tensor = upload_f32(&backend, &authority, shape, &values)?;
            assert!(
                ConditioningMetadataMap::checked([(
                    ConditioningMetadataKey::ConditioningLyrics,
                    ConditioningMetadataValue::ConditioningLyrics(Some(tensor))
                )])
                .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn task618_exact_lyrics_metadata_separation_residency_and_hash_cancellation() -> TestResult {
        use ConditioningMetadataKey as Key;
        use ConditioningMetadataValue as Value;
        let (backend, authority) = runtime()?;
        let cancellation = sim_types::CancellationToken::default();
        let tensor = upload_f32(&backend, &authority, vec![1, 2], &[3.0, 4.0])?;
        let metadata = ConditioningMetadataMap::checked([
            (Key::PooledOutput, Value::PooledOutput(Some(tensor.clone()))),
            (
                Key::ConditioningLyrics,
                Value::ConditioningLyrics(Some(tensor.clone())),
            ),
        ])?;
        assert_eq!(Key::PooledOutput as u8, 30);
        assert_eq!(Key::ConditioningLyrics as u8, 32);
        assert_eq!(metadata.resident_tensor_bytes()?, tensor.storage_byte_len());
        let entries = ["first", "second"]
            .into_iter()
            .map(|identifier| {
                ConditioningEntry::checked_with_metadata(
                    identifier,
                    ConditioningValue::regular(tensor.clone())?,
                    Default::default(),
                    Default::default(),
                    metadata.clone(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let set = ConditioningSet::checked(identity()?, entries, &cancellation)?;
        let parts = set.resident_parts()?;
        let [allocation] = parts.tensor_allocations() else {
            return Err("shared lyrics storage charged more than once".into());
        };
        assert_eq!(allocation.storage_id(), tensor.storage_id());
        assert_eq!(allocation.resident_bytes(), tensor.storage_byte_len());
        for (key, value) in [
            (Key::ConditioningLyrics, Value::PooledOutput(None)),
            (Key::PooledOutput, Value::ConditioningLyrics(None)),
            (Key::ConditioningLyrics, Value::Tensor(tensor.clone())),
            (Key::Mask, Value::ConditioningLyrics(Some(tensor))),
        ] {
            assert!(ConditioningMetadataMap::checked([(key, value)]).is_err());
        }
        let lyrics_only = ConditioningMetadataMap::checked([(
            Key::ConditioningLyrics,
            metadata
                .get(Key::ConditioningLyrics)
                .ok_or("lyrics")?
                .clone(),
        )])?;
        let mut checks = 0;
        let mut checkpoint = || {
            checks += 1;
            if checks == 3 {
                Err(ConditioningError::Cancelled)
            } else {
                Ok(())
            }
        };
        let mut hasher = Sha256::new();
        assert!(matches!(
            hash_metadata_map(&mut hasher, &lyrics_only, &mut checkpoint),
            Err(ConditioningError::Cancelled)
        ));
        assert_eq!(checks, 3);
        let recovered = ConditioningSet::checked(
            set.identity().clone(),
            set.entries().to_vec(),
            &cancellation,
        )?;
        assert_eq!(recovered.digest(), set.digest());
        Ok(())
    }
    #[test]
    fn frame_rate_metadata_keeps_f64_words_and_existing_digest_tags() -> TestResult {
        use ConditioningMetadataKey as Key;
        use ConditioningMetadataValue as Value;
        assert_eq!(Key::Region as u8, 0);
        assert_eq!(Key::PooledOutput as u8, 30);
        assert_eq!(Key::FrameRate as u8, 31);
        assert_eq!(Key::ConditioningLyrics as u8, 32);
        for frame_rate in [-0.0_f64, 0.0, 25.0, 1000.0] {
            let metadata =
                ConditioningMetadataMap::checked([(Key::FrameRate, Value::FiniteF64(frame_rate))])?;
            let Some(Value::FiniteF64(stored)) = metadata.get(Key::FrameRate) else {
                return Err("Missing typed frame rate".into());
            };
            assert_eq!(stored.to_bits(), frame_rate.to_bits());
        }
        for frame_rate in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                ConditioningMetadataMap::checked([(Key::FrameRate, Value::FiniteF64(frame_rate))])
                    .is_err()
            );
        }
        assert!(
            ConditioningMetadataMap::checked([(Key::FrameRate, Value::Boolean(true))]).is_err()
        );
        assert!(ConditioningMetadataMap::checked([(Key::FrameRate, Value::Unsigned(25))]).is_err());
        Ok(())
    }

    #[test]
    fn frame_rate_entry_update_preserves_aliases_and_changes_only_canonical_metadata_identity()
    -> TestResult {
        use ConditioningMetadataKey as Key;
        use ConditioningMetadataValue as Value;
        let (backend, authority) = runtime()?;
        let tensor = upload_f32(&backend, &authority, vec![1, 1, 2], &[1.0, -0.0])?;
        let cancellation = sim_types::CancellationToken::default();
        let original = ConditioningSet::checked(
            identity()?,
            vec![ConditioningEntry::checked_with_metadata(
                "frame-rate-entry",
                ConditioningValue::regular(tensor.clone())?,
                ConditioningEntryOptions::default(),
                ConditioningReferences::default(),
                ConditioningMetadataMap::checked([(Key::PooledOutput, Value::PooledOutput(None))])?,
            )?],
            &cancellation,
        )?;
        let before_digest = original.digest().to_owned();
        let before_bytes = backend.memory_snapshot().current_bytes;
        let mut metadata = original
            .entries()
            .first()
            .ok_or("Missing original entry")?
            .metadata()
            .clone();
        metadata.insert(Key::FrameRate, Value::FiniteF64(25.0))?;
        let updated = original.with_entry_metadata(vec![metadata.clone()], &cancellation)?;
        assert_ne!(updated.digest(), original.digest());
        assert_eq!(updated.identity(), original.identity());
        let entry = updated.entries().first().ok_or("Missing updated entry")?;
        let ConditioningValue::Regular(updated_tensor) = entry.value() else {
            return Err("Regular tensor lost".into());
        };
        assert_eq!(updated_tensor.storage_id(), tensor.storage_id());
        assert!(matches!(
            entry.metadata().get(Key::PooledOutput),
            Some(Value::PooledOutput(None))
        ));
        assert!(
            original
                .entries()
                .first()
                .ok_or("Missing original entry")?
                .metadata()
                .get(Key::FrameRate)
                .is_none()
        );
        assert_eq!(original.digest(), before_digest);
        assert_eq!(backend.memory_snapshot().current_bytes, before_bytes);
        assert!(
            ConditioningSet::combined_resident_bytes(&[&original, &updated])?
                < original.resident_bytes()? + updated.resident_bytes()?
        );
        metadata.insert(Key::FrameRate, Value::FiniteF64(0.0))?;
        let overwritten = updated.with_entry_metadata(vec![metadata.clone()], &cancellation)?;
        assert_ne!(overwritten.digest(), updated.digest());
        let Some(Value::FiniteF64(stored)) = overwritten
            .entries()
            .first()
            .ok_or("Missing overwritten entry")?
            .metadata()
            .get(Key::FrameRate)
        else {
            return Err("Frame rate lost".into());
        };
        assert_eq!(stored.to_bits(), 0.0_f64.to_bits());
        let cancelled = sim_types::CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            original.with_entry_metadata(vec![metadata], &cancelled),
            Err(ConditioningError::Cancelled)
        ));
        assert_eq!(original.digest(), before_digest);
        assert_eq!(backend.memory_snapshot().current_bytes, before_bytes);
        Ok(())
    }
    #[test]
    fn frame_rate_preserves_exact_nullable_lyrics_empty_sets_and_signed_zero_digest() -> TestResult
    {
        use ConditioningMetadataKey as Key;
        use ConditioningMetadataValue as Value;
        let (backend, authority) = runtime()?;
        let cancellation = sim_types::CancellationToken::default();
        let tensor = upload_f32(&backend, &authority, vec![1, 1, 2], &[1.0, -0.0])?;
        for lyrics in [None, Some(tensor.clone())] {
            let source = ConditioningSet::checked(
                identity()?,
                vec![ConditioningEntry::checked_with_metadata(
                    "lyrics-and-rate",
                    ConditioningValue::regular(tensor.clone())?,
                    Default::default(),
                    Default::default(),
                    ConditioningMetadataMap::checked([(
                        Key::ConditioningLyrics,
                        Value::ConditioningLyrics(lyrics.clone()),
                    )])?,
                )?],
                &cancellation,
            )?;
            let before = backend.memory_snapshot().current_bytes;
            let mut metadata = source.entries().first().ok_or("source")?.metadata().clone();
            metadata.insert(Key::FrameRate, Value::FiniteF64(-0.0))?;
            let negative = source.with_entry_metadata(vec![metadata.clone()], &cancellation)?;
            metadata.insert(Key::FrameRate, Value::FiniteF64(0.0))?;
            let positive = source.with_entry_metadata(vec![metadata], &cancellation)?;
            assert_ne!(negative.digest(), positive.digest());
            for updated in [&negative, &positive] {
                match (
                    lyrics.as_ref(),
                    updated
                        .entries()
                        .first()
                        .ok_or("updated")?
                        .metadata()
                        .get(Key::ConditioningLyrics),
                ) {
                    (None, Some(Value::ConditioningLyrics(None))) => {}
                    (Some(original), Some(Value::ConditioningLyrics(Some(retained)))) => {
                        assert_eq!(original.storage_id(), retained.storage_id());
                        assert_eq!(original.contiguous_bytes()?, retained.contiguous_bytes()?);
                    }
                    _ => {
                        return Err("nullable exact lyrics changed while setting frame rate".into());
                    }
                }
            }
            assert_eq!(backend.memory_snapshot().current_bytes, before);
            assert!(
                source
                    .entries()
                    .first()
                    .ok_or("source")?
                    .metadata()
                    .get(Key::FrameRate)
                    .is_none()
            );
        }
        let empty = ConditioningSet::checked(identity()?, Vec::new(), &cancellation)?;
        let before = backend.memory_snapshot().current_bytes;
        let updated = empty.with_entry_metadata(Vec::new(), &cancellation)?;
        assert!(updated.entries().is_empty());
        assert_eq!(updated.identity(), empty.identity());
        assert_eq!(updated.digest(), empty.digest());
        assert_eq!(backend.memory_snapshot().current_bytes, before);
        Ok(())
    }

    #[test]
    fn reference_audio_timbre_metadata_presence_order_and_alias_identity() -> TestResult {
        use ConditioningMetadataKey as Key;
        use ConditioningMetadataValue as Value;
        let (backend, authority) = runtime()?;
        let cancellation = sim_types::CancellationToken::default();
        let first = upload_f32(&backend, &authority, vec![1, 64, 2], &[0.25; 128])?;
        let second = upload_f32(&backend, &authority, vec![2, 7, 3, 1], &[0.5; 42])?;
        let source = ConditioningSet::checked(
            identity()?,
            vec![ConditioningEntry::checked(
                "primary",
                ConditioningValue::regular(first.clone())?,
                Default::default(),
            )?],
            &cancellation,
        )?;
        let mut digests = BTreeSet::from([source.digest().to_owned()]);
        for references in [
            None,
            Some(Vec::new()),
            Some(vec![first.clone(), second.clone()]),
            Some(vec![second, first]),
        ] {
            let map = ConditioningMetadataMap::checked([(
                Key::ReferenceAudioTimbreLatents,
                Value::ReferenceAudioTimbreLatents(references),
            )])?;
            let output = source.with_entry_metadata(vec![map], &cancellation)?;
            assert!(digests.insert(output.digest().to_owned()));
            output.validate()?;
        }
        assert_eq!(Key::ReferenceAudioTimbreLatents as u8, 33);
        assert_eq!(Key::FrameRate as u8, 31);
        assert_eq!(Key::ConditioningLyrics as u8, 32);
        assert!(
            source
                .entries()
                .first()
                .ok_or("Primary")?
                .metadata()
                .get(Key::ReferenceAudioTimbreLatents)
                .is_none()
        );
        Ok(())
    }
    #[test]
    fn reference_audio_timbre_metadata_retains_strict_capacity_and_unique_storage_accounting()
    -> TestResult {
        use ConditioningMetadataKey as Key;
        use ConditioningMetadataValue as Value;
        let (backend, authority) = runtime()?;
        let tensor = upload_f32(&backend, &authority, vec![1, 64, 2], &[0.25; 128])?;
        let map = ConditioningMetadataMap::checked([(
            Key::ReferenceAudioTimbreLatents,
            Value::ReferenceAudioTimbreLatents(Some(vec![tensor.clone(); 64])),
        )])?;
        assert_eq!(map.resident_tensor_bytes()?, tensor.storage_byte_len());
        assert!(
            ConditioningMetadataMap::checked([(
                Key::ReferenceAudioTimbreLatents,
                Value::ReferenceAudioTimbreLatents(Some(vec![tensor.clone(); 65]))
            )])
            .is_err()
        );
        for (key, value) in [
            (
                Key::ReferenceLatents,
                Value::ReferenceAudioTimbreLatents(None),
            ),
            (
                Key::ReferenceAudioTimbreLatents,
                Value::TensorList(vec![tensor.clone()]),
            ),
        ] {
            assert!(ConditioningMetadataMap::checked([(key, value)]).is_err());
        }
        let cancellation = sim_types::CancellationToken::default();
        let set = ConditioningSet::checked(
            identity()?,
            vec![ConditioningEntry::checked_with_metadata(
                "primary",
                ConditioningValue::regular(tensor.clone())?,
                Default::default(),
                Default::default(),
                map,
            )?],
            &cancellation,
        )?;
        let parts = set.resident_parts()?;
        let [allocation] = parts.tensor_allocations() else {
            return Err("Audio aliases charged twice".into());
        };
        assert_eq!(allocation.storage_id(), tensor.storage_id());
        assert_eq!(allocation.resident_bytes(), tensor.storage_byte_len());
        assert_eq!(MAX_CONDITIONING_METADATA_ENTRIES, 28);
        Ok(())
    }
    #[test]
    fn reference_audio_timbre_metadata_hash_cancellation_preserves_source_and_recovers()
    -> TestResult {
        let (backend, authority) = runtime()?;
        let tensor = upload_f32(&backend, &authority, vec![1, 64, 2], &[0.25; 128])?;
        let map = ConditioningMetadataMap::checked([(
            ConditioningMetadataKey::ReferenceAudioTimbreLatents,
            ConditioningMetadataValue::ReferenceAudioTimbreLatents(Some(vec![tensor.clone()])),
        )])?;
        let mut checks = 0;
        let mut checkpoint = || {
            checks += 1;
            if checks == 3 {
                Err(ConditioningError::Cancelled)
            } else {
                Ok(())
            }
        };
        let mut hasher = Sha256::new();
        assert!(matches!(
            hash_metadata_map(&mut hasher, &map, &mut checkpoint),
            Err(ConditioningError::Cancelled)
        ));
        let set = ConditioningSet::checked(
            identity()?,
            vec![ConditioningEntry::checked_with_metadata(
                "primary",
                ConditioningValue::regular(tensor)?,
                Default::default(),
                Default::default(),
                map,
            )?],
            &sim_types::CancellationToken::default(),
        )?;
        set.validate()?;
        Ok(())
    }
    #[test]
    fn reference_audio_timbre_metadata_does_not_increase_the_existing_entry_bound() -> TestResult {
        use ConditioningMetadataKey as Key;
        use ConditioningMetadataValue as Value;
        let (backend, authority) = runtime()?;
        let latent = upload_f32(&backend, &authority, vec![1, 1, 1, 1], &[1.0])?;
        let video = upload_f32(&backend, &authority, vec![1, 1, 1, 1, 1], &[1.0])?;
        let matrix = upload_f32(&backend, &authority, vec![1, 2], &[1.0, 2.0])?;
        let attention = upload_f32(&backend, &authority, vec![1, 1, 2], &[1.0, 2.0])?;
        let face = upload_f32(&backend, &authority, vec![1, 1, 1, 1, 3], &[1.0; 3])?;
        let mask = upload_f32(&backend, &authority, vec![1, 1, 1, 1, 28], &[1.0; 28])?;
        let values = vec![
            (
                Key::Region,
                Value::Region(ConditioningRegion::absolute(vec![1, 1], vec![0, 0])?),
            ),
            (Key::Mask, Value::Tensor(matrix.clone())),
            (Key::MaskStrength, Value::FiniteF64(1.0)),
            (Key::SetAreaToBounds, Value::Boolean(false)),
            (
                Key::TimestepRange,
                Value::TimestepRange(ConditioningTimestepRange::source(0.0, 1.0)?),
            ),
            (Key::ConcatLatentImage, Value::Tensor(latent.clone())),
            (Key::ConcatMask, Value::Tensor(latent.clone())),
            (Key::ConcatMaskIndex, Value::Unsigned(0)),
            (
                Key::Control,
                Value::Control(ConditioningControlBinding::checked("a".repeat(64), false)?),
            ),
            (Key::ControlApplyToUnconditioned, Value::Boolean(false)),
            (
                Key::ReferenceLatents,
                Value::TensorList(vec![latent.clone()]),
            ),
            (Key::VaceFrames, Value::TensorList(vec![latent.clone()])),
            (Key::VaceMask, Value::TensorList(vec![latent.clone()])),
            (Key::VaceStrength, Value::FiniteF64List(vec![1.0])),
            (Key::CameraConditions, Value::Tensor(matrix.clone())),
            (Key::TimeDimensionConcat, Value::Tensor(matrix.clone())),
            (Key::AudioEmbedding, Value::Tensor(attention.clone())),
            (Key::ReferenceMotion, Value::Tensor(latent)),
            (Key::ControlVideo, Value::Tensor(video.clone())),
            (Key::PoseVideoLatent, Value::Tensor(video)),
            (Key::FaceVideoPixels, Value::Tensor(face)),
            (Key::DrivingMask28Channels, Value::Tensor(mask.clone())),
            (Key::ReferenceMask28Channels, Value::Tensor(mask)),
            (Key::ReferenceMaskFlag, Value::Boolean(true)),
            (Key::SecondsStart, Value::FiniteF64(0.0)),
            (Key::SecondsTotal, Value::FiniteF64(1.0)),
            (Key::CrossAttentionControlnet, Value::Tensor(attention)),
            (
                Key::PooledOutputControlnet,
                Value::ControlnetPooledOutput(Some(matrix)),
            ),
        ];
        assert_eq!(values.len(), MAX_CONDITIONING_METADATA_ENTRIES);
        let mut metadata = ConditioningMetadataMap::checked(values.clone())?;
        metadata.validate()?;
        assert!(matches!(
            metadata.insert(
                Key::ReferenceAudioTimbreLatents,
                Value::ReferenceAudioTimbreLatents(None)
            ),
            Err(ConditioningError::Invalid(_))
        ));
        assert_eq!(metadata.len(), 28);
        assert!(metadata.get(Key::ReferenceAudioTimbreLatents).is_none());
        assert!(
            ConditioningMetadataMap::checked(values.into_iter().chain([(
                Key::ReferenceAudioTimbreLatents,
                Value::ReferenceAudioTimbreLatents(None)
            )]))
            .is_err()
        );
        metadata.insert(Key::SecondsTotal, Value::FiniteF64(2.0))?;
        assert_eq!(metadata.len(), 28);
        assert!(
            matches!(metadata.get(Key::SecondsTotal),Some(Value::FiniteF64(value)) if *value == 2.0)
        );
        Ok(())
    }
}
