use crate::{MappedModelWeights, ModelFamilyError};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_tensor::{
    BinaryOperation, DType, ExecutionContext, Layout, LinearAlgebraOperation, Scalar, ScalarSide,
    Tensor, TensorBackend, TensorDescriptor, TensorError, ViewAccess,
    generated_sim_operator_indirection_01::OperatorIndirectionError,
};
use std::collections::{BTreeMap, BTreeSet};
use std::mem;
use thiserror::Error;

pub const PATCH_GRAPH_SCHEMA_VERSION: u16 = 2;
const MAX_PATCH_TEXT_BYTES: usize = 64 * 1024;
const MAX_SEMANTIC_PATCH_DEPTH: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatchKind {
    ScaleOnly,
    DenseDiff,
    Set,
    Lora,
    Loha,
    Lokr,
    Oft,
    Glora,
    Boft,
    Dora,
    Nested,
    ModelAsLora,
    ControlNet,
    Adapter,
    Replacement,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatchTensor {
    pub shape: Vec<u64>,
    pub values: Vec<f32>,
}

impl PatchTensor {
    pub fn checked(shape: Vec<u64>, values: Vec<f32>) -> Result<Self, PatchGraphError> {
        let tensor = Self { shape, values };
        validate_patch_tensor("payload", &tensor)?;
        Ok(tensor)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatchSlice {
    pub dimension: u64,
    pub start: u64,
    pub length: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatchValueTransform {
    pub scale: f32,
    pub bias: f32,
}

impl Default for PatchValueTransform {
    fn default() -> Self {
        Self {
            scale: 1.0,
            bias: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PatchComputeBoundary {
    Configured(DType),
    WeightDType,
}

impl PatchComputeBoundary {
    pub fn configured(dtype: DType) -> Result<Self, PatchGraphError> {
        validate_configured_patch_compute_dtype(dtype)?;
        Ok(Self::Configured(dtype))
    }

    pub const fn weight_dtype() -> Self {
        Self::WeightDType
    }

    pub fn intermediate_dtype(self, weight_dtype: DType) -> Result<DType, PatchGraphError> {
        let dtype = match self {
            Self::Configured(dtype) => {
                validate_configured_patch_compute_dtype(dtype)?;
                dtype
            }
            Self::WeightDType => {
                validate_patch_compute_dtype(weight_dtype)?;
                weight_dtype
            }
        };
        Ok(dtype)
    }
}

pub fn factorize_patch_dimension(
    dimension: u64,
    preferred_factor: Option<u64>,
) -> Result<(u64, u64), PatchGraphError> {
    if dimension == 0 {
        return Err(PatchGraphError::InvalidPayload(
            "patch factorization dimension must be positive".into(),
        ));
    }
    if preferred_factor == Some(0) {
        return Err(PatchGraphError::InvalidPayload(
            "patch factorization preference must be positive".into(),
        ));
    }
    if let Some(factor) = preferred_factor {
        let square = factor
            .checked_mul(factor)
            .ok_or(PatchGraphError::ShapeOverflow)?;
        if dimension.is_multiple_of(factor) && dimension >= square {
            let quotient = dimension / factor;
            return Ok(if factor <= quotient {
                (factor, quotient)
            } else {
                (quotient, factor)
            });
        }
    }

    let factor_limit = preferred_factor.unwrap_or(dimension);
    let mut left = 1_u64;
    let mut right = dimension;
    let initial_length = left
        .checked_add(right)
        .ok_or(PatchGraphError::ShapeOverflow)?;
    while left < right {
        let mut next_left = left.checked_add(1).ok_or(PatchGraphError::ShapeOverflow)?;
        while !dimension.is_multiple_of(next_left) {
            next_left = next_left
                .checked_add(1)
                .ok_or(PatchGraphError::ShapeOverflow)?;
        }
        let next_right = dimension / next_left;
        if next_left
            .checked_add(next_right)
            .ok_or(PatchGraphError::ShapeOverflow)?
            > initial_length
            || next_left > factor_limit
        {
            break;
        }
        left = next_left;
        right = next_right;
    }
    Ok(if left <= right {
        (left, right)
    } else {
        (right, left)
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum PatchPayload {
    ScaleOnly,
    DenseDiff {
        tensor: PatchTensor,
        pad_weight: bool,
    },
    Set {
        tensor: PatchTensor,
    },
    Lora {
        up: PatchTensor,
        down: PatchTensor,
        mid: Option<PatchTensor>,
        alpha: Option<f32>,
        dora_scale: Option<PatchTensor>,
        reshape: Option<Vec<u64>>,
    },
    Loha {
        first_up: PatchTensor,
        first_down: PatchTensor,
        second_up: PatchTensor,
        second_down: PatchTensor,
        first_tucker: Option<PatchTensor>,
        second_tucker: Option<PatchTensor>,
        alpha: Option<f32>,
        dora_scale: Option<PatchTensor>,
    },
    Lokr {
        first: Option<PatchTensor>,
        second: Option<PatchTensor>,
        first_up: Option<PatchTensor>,
        first_down: Option<PatchTensor>,
        second_up: Option<PatchTensor>,
        second_down: Option<PatchTensor>,
        second_tucker: Option<PatchTensor>,
        alpha: Option<f32>,
        dora_scale: Option<PatchTensor>,
    },
    Oft {
        blocks: PatchTensor,
        rescale: Option<PatchTensor>,
        constraint: Option<f32>,
        dora_scale: Option<PatchTensor>,
    },
    Glora {
        first_a: PatchTensor,
        second_a: PatchTensor,
        first_b: PatchTensor,
        second_b: PatchTensor,
        alpha: Option<f32>,
        dora_scale: Option<PatchTensor>,
    },
    Boft {
        blocks: PatchTensor,
        rescale: Option<PatchTensor>,
        constraint: Option<f32>,
        dora_scale: Option<PatchTensor>,
    },
    Dora {
        difference: PatchTensor,
        scale: PatchTensor,
        alpha: f32,
    },
    Nested {
        base: PatchTensor,
        #[serde(default)]
        base_transform: PatchValueTransform,
        patches: Vec<NestedPatch>,
    },
    ModelAsLora {
        target: PatchTensor,
    },
}

impl PatchPayload {
    pub fn kind(&self) -> PatchKind {
        match self {
            Self::ScaleOnly => PatchKind::ScaleOnly,
            Self::DenseDiff { .. } => PatchKind::DenseDiff,
            Self::Set { .. } => PatchKind::Set,
            Self::Lora {
                dora_scale: Some(_),
                ..
            }
            | Self::Loha {
                dora_scale: Some(_),
                ..
            }
            | Self::Lokr {
                dora_scale: Some(_),
                ..
            }
            | Self::Oft {
                dora_scale: Some(_),
                ..
            }
            | Self::Glora {
                dora_scale: Some(_),
                ..
            }
            | Self::Boft {
                dora_scale: Some(_),
                ..
            }
            | Self::Dora { .. } => PatchKind::Dora,
            Self::Lora { .. } => PatchKind::Lora,
            Self::Loha { .. } => PatchKind::Loha,
            Self::Lokr { .. } => PatchKind::Lokr,
            Self::Oft { .. } => PatchKind::Oft,
            Self::Glora { .. } => PatchKind::Glora,
            Self::Boft { .. } => PatchKind::Boft,
            Self::Nested { .. } => PatchKind::Nested,
            Self::ModelAsLora { .. } => PatchKind::ModelAsLora,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NestedPatch {
    pub strength: f32,
    pub strength_model: f32,
    pub transform: PatchValueTransform,
    pub payload: PatchPayload,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticPatchOperation {
    pub identifier: String,
    pub target_key: String,
    pub expected_shape: Vec<u64>,
    pub strength: f32,
    pub strength_model: f32,
    pub slices: Vec<PatchSlice>,
    pub transform: PatchValueTransform,
    pub payload: PatchPayload,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatchApplication {
    Add,
    Replace,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatchTarget {
    pub key: String,
    pub expected_shape: Vec<u64>,
    pub values: Vec<f32>,
    pub application: PatchApplication,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatchOperation {
    pub identifier: String,
    pub kind: PatchKind,
    pub scale: f32,
    pub targets: Vec<PatchTarget>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatchGraphIdentity {
    pub schema_version: u16,
    pub base_artifact_digest: String,
    pub ordered_digest: String,
}

impl PatchGraphIdentity {
    pub fn owned_resident_bytes(&self) -> Option<u64> {
        let bytes = self
            .base_artifact_digest
            .capacity()
            .checked_add(self.ordered_digest.capacity())?;
        u64::try_from(bytes).ok()
    }

    pub fn validate_for_base(
        &self,
        expected_base_artifact_digest: &str,
    ) -> Result<(), PatchGraphIdentityError> {
        if self.schema_version != PATCH_GRAPH_SCHEMA_VERSION {
            return Err(PatchGraphIdentityError::SchemaVersion {
                expected: PATCH_GRAPH_SCHEMA_VERSION,
                actual: self.schema_version,
            });
        }
        validate_identity_digest("expected base artifact", expected_base_artifact_digest)?;
        validate_identity_digest("base artifact", &self.base_artifact_digest)?;
        validate_identity_digest("ordered", &self.ordered_digest)?;
        if self.base_artifact_digest != expected_base_artifact_digest {
            return Err(PatchGraphIdentityError::BaseDigestMismatch {
                expected: expected_base_artifact_digest.to_owned(),
                actual: self.base_artifact_digest.clone(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PatchGraph {
    base_artifact_digest: String,
    semantic_operations: Vec<SemanticPatchOperation>,
    ordered_digest: String,
}

impl PatchGraph {
    pub fn checked(
        base_artifact_digest: impl Into<String>,
        operations: Vec<PatchOperation>,
    ) -> Result<Self, PatchGraphError> {
        let base_artifact_digest = base_artifact_digest.into();
        validate_digest(&base_artifact_digest)?;
        validate_legacy_operations(&operations)?;
        let semantic_operations = map_legacy_operations(&operations)?;
        let ordered_digest = ordered_digest(&base_artifact_digest, &operations)?;
        Ok(Self {
            base_artifact_digest,
            semantic_operations,
            ordered_digest,
        })
    }

    pub fn checked_semantic(
        base_artifact_digest: impl Into<String>,
        operations: Vec<SemanticPatchOperation>,
    ) -> Result<Self, PatchGraphError> {
        let base_artifact_digest = base_artifact_digest.into();
        validate_digest(&base_artifact_digest)?;
        validate_semantic_operations(&operations)?;
        let ordered_digest = semantic_ordered_digest(&base_artifact_digest, &operations)?;
        Ok(Self {
            base_artifact_digest,
            semantic_operations: operations,
            ordered_digest,
        })
    }

    pub fn identity(&self) -> PatchGraphIdentity {
        PatchGraphIdentity {
            schema_version: PATCH_GRAPH_SCHEMA_VERSION,
            base_artifact_digest: self.base_artifact_digest.clone(),
            ordered_digest: self.ordered_digest.clone(),
        }
    }

    pub fn semantic_operations(&self) -> &[SemanticPatchOperation] {
        &self.semantic_operations
    }

    pub fn resident_bytes(&self) -> Result<u64, PatchGraphError> {
        let mut bytes = resident_allocation(mem::size_of::<Self>())?;
        add_resident_bytes(&mut bytes, self.base_artifact_digest.capacity())?;
        add_resident_bytes(&mut bytes, self.ordered_digest.capacity())?;
        add_resident_bytes(
            &mut bytes,
            self.semantic_operations
                .capacity()
                .checked_mul(mem::size_of::<SemanticPatchOperation>())
                .ok_or(PatchGraphError::ResidentBytesOverflow)?,
        )?;
        for operation in &self.semantic_operations {
            add_resident_bytes(&mut bytes, operation.identifier.capacity())?;
            add_resident_bytes(&mut bytes, operation.target_key.capacity())?;
            add_resident_bytes(
                &mut bytes,
                operation
                    .expected_shape
                    .capacity()
                    .checked_mul(mem::size_of::<u64>())
                    .ok_or(PatchGraphError::ResidentBytesOverflow)?,
            )?;
            add_resident_bytes(
                &mut bytes,
                operation
                    .slices
                    .capacity()
                    .checked_mul(mem::size_of::<PatchSlice>())
                    .ok_or(PatchGraphError::ResidentBytesOverflow)?,
            )?;
            patch_payload_resident_bytes(&operation.payload, &mut bytes, 0)?;
        }
        Ok(bytes)
    }

    pub fn append_semantic(
        &self,
        operations: impl IntoIterator<Item = SemanticPatchOperation>,
    ) -> Result<Self, PatchGraphError> {
        let mut combined = self.semantic_operations.clone();
        combined.extend(operations);
        Self::checked_semantic(self.base_artifact_digest.clone(), combined)
    }

    pub fn operations_for_key(&self, target_key: &str) -> Vec<&SemanticPatchOperation> {
        self.semantic_operations
            .iter()
            .filter(|operation| operation.target_key == target_key)
            .collect()
    }

    pub fn apply(
        &self,
        backend: &dyn TensorBackend,
        source: &MappedModelWeights,
        context: &ExecutionContext<'_>,
    ) -> Result<MappedModelWeights, PatchGraphError> {
        self.apply_with_compute_boundary(
            backend,
            source,
            PatchComputeBoundary::Configured(DType::F32),
            context,
        )
    }

    pub fn apply_with_compute_boundary(
        &self,
        backend: &dyn TensorBackend,
        source: &MappedModelWeights,
        compute_boundary: PatchComputeBoundary,
        context: &ExecutionContext<'_>,
    ) -> Result<MappedModelWeights, PatchGraphError> {
        context.cancellation.check()?;
        if source.base_artifact_digest() != self.base_artifact_digest {
            return Err(PatchGraphError::BaseDigestMismatch {
                expected: self.base_artifact_digest.clone(),
                actual: source.base_artifact_digest().to_owned(),
            });
        }
        validate_semantic_operations(&self.semantic_operations)?;
        let mut staged = source.tensors().clone();
        let mut touched_dtypes = BTreeMap::new();
        for operation in &self.semantic_operations {
            context.cancellation.check()?;
            let current = staged
                .get(&operation.target_key)
                .ok_or_else(|| PatchGraphError::MissingTarget(operation.target_key.clone()))?;
            let immutable_original = source
                .unpatched_tensors()
                .get(&operation.target_key)
                .ok_or_else(|| PatchGraphError::MissingTarget(operation.target_key.clone()))?;
            if current.descriptor().shape() != operation.expected_shape {
                return Err(PatchGraphError::ShapeMismatch {
                    key: operation.target_key.clone(),
                    expected: operation.expected_shape.clone(),
                    actual: current.descriptor().shape().to_vec(),
                });
            }
            let output_dtype = source
                .tensors()
                .get(&operation.target_key)
                .ok_or_else(|| PatchGraphError::MissingTarget(operation.target_key.clone()))?
                .descriptor()
                .dtype();
            let compute_dtype = compute_boundary.intermediate_dtype(output_dtype)?;
            let current = if let Some((staged_dtype, staged_output_dtype)) =
                touched_dtypes.get(&operation.target_key)
            {
                if *staged_dtype != compute_dtype || *staged_output_dtype != output_dtype {
                    return Err(PatchGraphError::InvalidPayload(format!(
                        "patch compute dtype changed within target {}",
                        operation.target_key
                    )));
                }
                current.clone()
            } else {
                touched_dtypes.insert(operation.target_key.clone(), (compute_dtype, output_dtype));
                backend_cast_tensor(backend, current, compute_dtype, false, false, context)?
            };
            let immutable_original = backend_cast_tensor(
                backend,
                immutable_original,
                compute_dtype,
                false,
                false,
                context,
            )?;
            let replacement = apply_semantic_operation(
                backend,
                &current,
                &immutable_original,
                operation,
                context,
            )?;
            staged.insert(operation.target_key.clone(), replacement);
        }
        context.cancellation.check()?;
        for target_key in touched_dtypes.keys() {
            let output_dtype = source
                .tensors()
                .get(target_key)
                .ok_or_else(|| PatchGraphError::MissingTarget(target_key.clone()))?
                .descriptor()
                .dtype();
            let current = staged
                .get(target_key)
                .ok_or_else(|| PatchGraphError::MissingTarget(target_key.clone()))?;
            let committed =
                backend_cast_tensor(backend, current, output_dtype, false, false, context)?;
            staged.insert(target_key.clone(), committed);
        }
        context.cancellation.check()?;
        let applied_digest =
            applied_patch_digest(&self.ordered_digest, compute_boundary, &touched_dtypes)?;
        Ok(source
            .with_tensors_preserving_identity(staged)
            .with_patch_graph_identity(&applied_digest)?)
    }

    pub fn apply_single_tensor(
        &self,
        backend: &dyn TensorBackend,
        target_key: &str,
        source: &Tensor,
        compute_boundary: PatchComputeBoundary,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, PatchGraphError> {
        let graph = Self::checked_semantic(
            self.base_artifact_digest.clone(),
            self.semantic_operations
                .iter()
                .filter(|operation| operation.target_key == target_key)
                .cloned()
                .collect(),
        )?;
        let mapped = MappedModelWeights::from_parts(
            self.base_artifact_digest.clone(),
            BTreeMap::from([(target_key.to_owned(), source.clone())]),
            Vec::new(),
        );
        let applied =
            graph.apply_with_compute_boundary(backend, &mapped, compute_boundary, context)?;
        applied
            .tensors()
            .get(target_key)
            .cloned()
            .ok_or_else(|| PatchGraphError::MissingTarget(target_key.to_owned()))
    }
}

fn resident_allocation(bytes: usize) -> Result<u64, PatchGraphError> {
    u64::try_from(bytes).map_err(|_| PatchGraphError::ResidentBytesOverflow)
}

fn add_resident_bytes(total: &mut u64, bytes: usize) -> Result<(), PatchGraphError> {
    *total = total
        .checked_add(resident_allocation(bytes)?)
        .ok_or(PatchGraphError::ResidentBytesOverflow)?;
    Ok(())
}

fn patch_tensor_resident_bytes(
    tensor: &PatchTensor,
    total: &mut u64,
) -> Result<(), PatchGraphError> {
    add_resident_bytes(
        total,
        tensor
            .shape
            .capacity()
            .checked_mul(mem::size_of::<u64>())
            .ok_or(PatchGraphError::ResidentBytesOverflow)?,
    )?;
    add_resident_bytes(
        total,
        tensor
            .values
            .capacity()
            .checked_mul(mem::size_of::<f32>())
            .ok_or(PatchGraphError::ResidentBytesOverflow)?,
    )
}

fn optional_patch_tensor_resident_bytes(
    tensor: Option<&PatchTensor>,
    total: &mut u64,
) -> Result<(), PatchGraphError> {
    if let Some(tensor) = tensor {
        patch_tensor_resident_bytes(tensor, total)?;
    }
    Ok(())
}

fn patch_payload_resident_bytes(
    payload: &PatchPayload,
    total: &mut u64,
    depth: usize,
) -> Result<(), PatchGraphError> {
    if depth > MAX_SEMANTIC_PATCH_DEPTH {
        return Err(PatchGraphError::NestingDepth);
    }
    match payload {
        PatchPayload::ScaleOnly => {}
        PatchPayload::DenseDiff { tensor, .. }
        | PatchPayload::Set { tensor }
        | PatchPayload::ModelAsLora { target: tensor } => {
            patch_tensor_resident_bytes(tensor, total)?;
        }
        PatchPayload::Lora {
            up,
            down,
            mid,
            dora_scale,
            reshape,
            ..
        } => {
            patch_tensor_resident_bytes(up, total)?;
            patch_tensor_resident_bytes(down, total)?;
            optional_patch_tensor_resident_bytes(mid.as_ref(), total)?;
            optional_patch_tensor_resident_bytes(dora_scale.as_ref(), total)?;
            if let Some(reshape) = reshape {
                add_resident_bytes(
                    total,
                    reshape
                        .capacity()
                        .checked_mul(mem::size_of::<u64>())
                        .ok_or(PatchGraphError::ResidentBytesOverflow)?,
                )?;
            }
        }
        PatchPayload::Loha {
            first_up,
            first_down,
            second_up,
            second_down,
            first_tucker,
            second_tucker,
            dora_scale,
            ..
        } => {
            for tensor in [first_up, first_down, second_up, second_down] {
                patch_tensor_resident_bytes(tensor, total)?;
            }
            for tensor in [
                first_tucker.as_ref(),
                second_tucker.as_ref(),
                dora_scale.as_ref(),
            ] {
                optional_patch_tensor_resident_bytes(tensor, total)?;
            }
        }
        PatchPayload::Lokr {
            first,
            second,
            first_up,
            first_down,
            second_up,
            second_down,
            second_tucker,
            dora_scale,
            ..
        } => {
            for tensor in [
                first.as_ref(),
                second.as_ref(),
                first_up.as_ref(),
                first_down.as_ref(),
                second_up.as_ref(),
                second_down.as_ref(),
                second_tucker.as_ref(),
                dora_scale.as_ref(),
            ] {
                optional_patch_tensor_resident_bytes(tensor, total)?;
            }
        }
        PatchPayload::Oft {
            blocks,
            rescale,
            dora_scale,
            ..
        }
        | PatchPayload::Boft {
            blocks,
            rescale,
            dora_scale,
            ..
        } => {
            patch_tensor_resident_bytes(blocks, total)?;
            optional_patch_tensor_resident_bytes(rescale.as_ref(), total)?;
            optional_patch_tensor_resident_bytes(dora_scale.as_ref(), total)?;
        }
        PatchPayload::Glora {
            first_a,
            second_a,
            first_b,
            second_b,
            dora_scale,
            ..
        } => {
            for tensor in [first_a, second_a, first_b, second_b] {
                patch_tensor_resident_bytes(tensor, total)?;
            }
            optional_patch_tensor_resident_bytes(dora_scale.as_ref(), total)?;
        }
        PatchPayload::Dora {
            difference, scale, ..
        } => {
            patch_tensor_resident_bytes(difference, total)?;
            patch_tensor_resident_bytes(scale, total)?;
        }
        PatchPayload::Nested { base, patches, .. } => {
            patch_tensor_resident_bytes(base, total)?;
            add_resident_bytes(
                total,
                patches
                    .capacity()
                    .checked_mul(mem::size_of::<NestedPatch>())
                    .ok_or(PatchGraphError::ResidentBytesOverflow)?,
            )?;
            for patch in patches {
                patch_payload_resident_bytes(&patch.payload, total, depth + 1)?;
            }
        }
    }
    Ok(())
}

fn apply_semantic_operation(
    backend: &dyn TensorBackend,
    current: &Tensor,
    immutable_original: &Tensor,
    operation: &SemanticPatchOperation,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    let compute_dtype = current.descriptor().dtype();
    let original = immutable_original.clone();
    require_same_shape(
        "semantic original",
        current.descriptor().shape(),
        original.descriptor().shape(),
    )?;
    let selected = extract_slices(backend, &current, &operation.slices, context)?;
    let selected_original = extract_slices(backend, &original, &operation.slices, context)?;
    let selected = canonical_affine(backend, &selected, operation.strength_model, 0.0, context)?;
    let selected = round_tensor(backend, &selected, compute_dtype, context)?;
    let patched = apply_payload_with_original(
        backend,
        &selected,
        &selected_original,
        &operation.payload,
        operation.strength,
        operation.transform,
        compute_dtype,
        current_dtype_epsilon(compute_dtype)?,
        context,
        0,
    )?;
    let committed = if operation.slices.is_empty() {
        patched
    } else {
        scatter_slices(backend, &current, &patched, &operation.slices, context)?
    };
    validate_finite_result(backend, &committed, &operation.target_key, context)?;
    Ok(committed)
}

fn apply_payload_with_original(
    backend: &dyn TensorBackend,
    base: &Tensor,
    immutable_original: &Tensor,
    payload: &PatchPayload,
    strength: f32,
    transform: PatchValueTransform,
    target_dtype: DType,
    dtype_epsilon: f32,
    context: &ExecutionContext<'_>,
    depth: usize,
) -> Result<Tensor, PatchGraphError> {
    context.cancellation.check()?;
    if depth > MAX_SEMANTIC_PATCH_DEPTH {
        return Err(PatchGraphError::NestingDepth);
    }
    match payload {
        PatchPayload::ScaleOnly => Ok(base.clone()),
        PatchPayload::Set { tensor } => {
            require_same_shape("set", base.descriptor().shape(), &tensor.shape)?;
            let tensor = patch_tensor_to_tensor(backend, tensor, target_dtype, context)?;
            round_tensor(backend, &tensor, target_dtype, context)
        }
        PatchPayload::Nested {
            base: nested_base,
            base_transform,
            patches,
        } => {
            let staged = patch_tensor_to_tensor(backend, nested_base, target_dtype, context)?;
            let mut staged = canonical_affine(
                backend,
                &staged,
                base_transform.scale,
                base_transform.bias,
                context,
            )?;
            for patch in patches {
                staged = canonical_affine(backend, &staged, patch.strength_model, 0.0, context)?;
                staged = apply_payload_with_original(
                    backend,
                    &staged,
                    immutable_original,
                    &patch.payload,
                    patch.strength,
                    patch.transform,
                    target_dtype,
                    dtype_epsilon,
                    context,
                    depth + 1,
                )?;
            }
            add_difference(
                backend,
                base,
                &staged,
                strength,
                transform,
                None,
                target_dtype,
                dtype_epsilon,
                true,
                context,
            )
        }
        PatchPayload::DenseDiff { tensor, pad_weight } => {
            let base = if *pad_weight && base.descriptor().shape() != tensor.shape {
                pad_tensor(backend, base, &tensor.shape, context)?
            } else {
                base.clone()
            };
            require_same_shape("dense diff", base.descriptor().shape(), &tensor.shape)?;
            let tensor = patch_tensor_to_tensor(backend, tensor, target_dtype, context)?;
            add_difference(
                backend,
                &base,
                &tensor,
                strength,
                transform,
                None,
                target_dtype,
                dtype_epsilon,
                true,
                context,
            )
        }
        PatchPayload::ModelAsLora { target } => {
            require_same_shape(
                "model-as-LoRA target",
                base.descriptor().shape(),
                &target.shape,
            )?;
            require_same_shape(
                "model-as-LoRA immutable original",
                base.descriptor().shape(),
                immutable_original.descriptor().shape(),
            )?;
            let target = patch_tensor_to_tensor(backend, target, target_dtype, context)?;
            let difference = backend_binary(
                backend,
                &target,
                immutable_original,
                BinaryOperation::Subtract,
                context,
            )?;
            add_difference(
                backend,
                base,
                &difference,
                strength,
                transform,
                None,
                target_dtype,
                dtype_epsilon,
                true,
                context,
            )
        }
        PatchPayload::Dora {
            difference,
            scale,
            alpha,
        } => {
            let difference = patch_tensor_to_tensor(backend, difference, target_dtype, context)?;
            add_difference(
                backend,
                base,
                &difference,
                strength,
                transform,
                Some((scale, *alpha)),
                target_dtype,
                dtype_epsilon,
                false,
                context,
            )
        }
        _ => {
            let effective_base = match payload {
                PatchPayload::Lora {
                    reshape: Some(shape),
                    ..
                } if shape != base.descriptor().shape() => {
                    pad_tensor(backend, base, shape, context)?
                }
                _ => base.clone(),
            };
            let (difference, alpha, dora) = resolve_factor_payload(
                backend,
                &effective_base,
                payload,
                strength,
                target_dtype,
                context,
            )?;
            if let Some(scale) = dora {
                add_difference(
                    backend,
                    &effective_base,
                    &difference,
                    strength,
                    transform,
                    Some((scale, alpha)),
                    target_dtype,
                    dtype_epsilon,
                    false,
                    context,
                )
            } else {
                add_difference(
                    backend,
                    &effective_base,
                    &difference,
                    strength * alpha,
                    transform,
                    None,
                    target_dtype,
                    dtype_epsilon,
                    false,
                    context,
                )
            }
        }
    }
}

#[cfg(test)]
fn apply_payload(
    backend: &dyn TensorBackend,
    base: &PatchTensor,
    payload: &PatchPayload,
    strength: f32,
    transform: PatchValueTransform,
    context: &ExecutionContext<'_>,
    depth: usize,
) -> Result<PatchTensor, PatchGraphError> {
    let base = patch_tensor_to_tensor(backend, base, DType::F32, context)?;
    let output = apply_payload_with_original(
        backend,
        &base,
        &base,
        payload,
        strength,
        transform,
        DType::F32,
        f32::EPSILON,
        context,
        depth,
    )?;
    tensor_to_patch_tensor(backend, &output, context)
}

fn resolve_factor_payload<'a>(
    backend: &dyn TensorBackend,
    base: &Tensor,
    payload: &'a PatchPayload,
    adapter_strength: f32,
    target_dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<(Tensor, f32, Option<&'a PatchTensor>), PatchGraphError> {
    match payload {
        PatchPayload::ScaleOnly => Err(PatchGraphError::InvalidPayload(
            "scale-only operations have no factor payload".into(),
        )),
        PatchPayload::Lora {
            up,
            down,
            mid,
            alpha,
            dora_scale,
            reshape,
        } => {
            let effective_base = reshape.as_ref().map_or(Ok(base.clone()), |shape| {
                if shape == base.descriptor().shape() {
                    Ok(base.clone())
                } else {
                    pad_tensor(backend, base, shape, context)
                }
            })?;
            let difference = if let Some(mid) = mid {
                canonical_tucker(backend, up, down, mid, target_dtype, context)?
            } else {
                let up = patch_tensor_to_tensor(backend, up, target_dtype, context)?;
                let down = patch_tensor_to_tensor(backend, down, target_dtype, context)?;
                canonical_matrix_product(backend, &up, &down, context)?
            };
            require_element_count_tensor("LoRA", effective_base.descriptor().shape(), &difference)?;
            let difference =
                reshape_read_only(&difference, effective_base.descriptor().shape().to_vec())?;
            let rank = down
                .shape
                .first()
                .copied()
                .ok_or_else(|| PatchGraphError::InvalidPayload("LoRA down rank".into()))?;
            Ok((
                difference,
                alpha.unwrap_or(rank as f32) / rank as f32,
                dora_scale.as_ref(),
            ))
        }
        PatchPayload::Loha {
            first_up,
            first_down,
            second_up,
            second_down,
            first_tucker,
            second_tucker,
            alpha,
            dora_scale,
        } => {
            let first = match (first_tucker, second_tucker) {
                (Some(first_tucker), Some(_)) => canonical_tucker(
                    backend,
                    first_up,
                    first_down,
                    first_tucker,
                    target_dtype,
                    context,
                )?,
                (None, None) => {
                    let up = patch_tensor_to_tensor(backend, first_up, target_dtype, context)?;
                    let down = patch_tensor_to_tensor(backend, first_down, target_dtype, context)?;
                    canonical_matrix_product(backend, &up, &down, context)?
                }
                _ => {
                    return Err(PatchGraphError::InvalidPayload(
                        "LoHa requires both Tucker tensors".into(),
                    ));
                }
            };
            let second = match (first_tucker, second_tucker) {
                (Some(_), Some(second_tucker)) => canonical_tucker(
                    backend,
                    second_up,
                    second_down,
                    second_tucker,
                    target_dtype,
                    context,
                )?,
                (None, None) => {
                    let up = patch_tensor_to_tensor(backend, second_up, target_dtype, context)?;
                    let down = patch_tensor_to_tensor(backend, second_down, target_dtype, context)?;
                    canonical_matrix_product(backend, &up, &down, context)?
                }
                _ => {
                    return Err(PatchGraphError::InvalidPayload(
                        "LoHa requires both Tucker tensors".into(),
                    ));
                }
            };
            let difference =
                backend_binary(backend, &first, &second, BinaryOperation::Multiply, context)?;
            require_element_count_tensor("LoHa", base.descriptor().shape(), &difference)?;
            let difference = reshape_read_only(&difference, base.descriptor().shape().to_vec())?;
            let rank = first_down
                .shape
                .first()
                .copied()
                .ok_or_else(|| PatchGraphError::InvalidPayload("LoHa down rank".into()))?;
            Ok((
                difference,
                alpha.unwrap_or(rank as f32) / rank as f32,
                dora_scale.as_ref(),
            ))
        }
        PatchPayload::Lokr {
            first,
            second,
            first_up,
            first_down,
            second_up,
            second_down,
            second_tucker,
            alpha,
            dora_scale,
        } => {
            let (first, first_rank) = resolve_full_or_decomposed(
                backend,
                "LoKr first",
                first.as_ref(),
                first_up.as_ref(),
                first_down.as_ref(),
                None,
                target_dtype,
                context,
            )?;
            let (second, second_rank) = resolve_full_or_decomposed(
                backend,
                "LoKr second",
                second.as_ref(),
                second_up.as_ref(),
                second_down.as_ref(),
                second_tucker.as_ref(),
                target_dtype,
                context,
            )?;
            let first = if second.descriptor().rank() == 4 {
                if first.descriptor().rank() != 2 {
                    return Err(PatchGraphError::InvalidPayload(
                        "LoKr convolutional first factor must be rank two".into(),
                    ));
                }
                let mut shape = first.descriptor().shape().to_vec();
                shape.extend([1, 1]);
                reshape_read_only(&first, shape)?
            } else {
                first
            };
            let difference = canonical_kronecker(backend, &first, &second, context)?;
            require_element_count_tensor("LoKr", base.descriptor().shape(), &difference)?;
            let difference = reshape_read_only(&difference, base.descriptor().shape().to_vec())?;
            let rank = second_rank.or(first_rank);
            let alpha = match (alpha, rank) {
                (Some(alpha), Some(rank)) => *alpha / rank as f32,
                _ => 1.0,
            };
            Ok((difference, alpha, dora_scale.as_ref()))
        }
        PatchPayload::Glora {
            first_a,
            second_a,
            first_b,
            second_b,
            alpha,
            dora_scale,
        } => {
            let (old_layout, rank) = infer_glora_layout(
                first_a,
                second_a,
                first_b,
                second_b,
                base.descriptor().shape(),
            )?;
            let first_a = flatten_first_axis(&patch_tensor_to_tensor(
                backend,
                first_a,
                target_dtype,
                context,
            )?)?;
            let second_a = flatten_first_axis(&patch_tensor_to_tensor(
                backend,
                second_a,
                target_dtype,
                context,
            )?)?;
            let first_b = flatten_first_axis(&patch_tensor_to_tensor(
                backend,
                first_b,
                target_dtype,
                context,
            )?)?;
            let second_b = flatten_first_axis(&patch_tensor_to_tensor(
                backend,
                second_b,
                target_dtype,
                context,
            )?)?;
            let difference = if old_layout {
                let base_matrix = flatten_first_axis(base)?;
                let b = canonical_matrix_product(backend, &second_b, &first_b, context)?;
                let wa = canonical_matrix_product(backend, &base_matrix, &second_a, context)?;
                let wa = canonical_matrix_product(backend, &wa, &first_a, context)?;
                backend_binary(backend, &b, &wa, BinaryOperation::Add, context)?
            } else {
                let wa = if base.descriptor().rank() > 2 {
                    let wa = canonical_glora_input_axis(backend, base, &first_a, context)?;
                    canonical_glora_input_axis(backend, &wa, &second_a, context)?
                } else {
                    let wa = canonical_matrix_product(backend, base, &first_a, context)?;
                    canonical_matrix_product(backend, &wa, &second_a, context)?
                };
                let b = canonical_matrix_product(backend, &first_b, &second_b, context)?;
                require_element_count_tensor("GLoRA B path", base.descriptor().shape(), &b)?;
                let b = reshape_read_only(&b, base.descriptor().shape().to_vec())?;
                backend_binary(backend, &wa, &b, BinaryOperation::Add, context)?
            };
            require_element_count_tensor("GLoRA", base.descriptor().shape(), &difference)?;
            let difference = reshape_read_only(&difference, base.descriptor().shape().to_vec())?;
            Ok((
                difference,
                alpha.unwrap_or(rank as f32) / rank as f32,
                dora_scale.as_ref(),
            ))
        }
        PatchPayload::Oft {
            blocks,
            rescale,
            constraint,
            dora_scale,
        } => {
            let transformed = apply_oft(
                backend,
                base,
                blocks,
                rescale.as_ref(),
                *constraint,
                adapter_strength,
                target_dtype,
                context,
            )?;
            Ok((
                transformed,
                if dora_scale.is_some() {
                    constraint.unwrap_or(0.0)
                } else {
                    1.0
                },
                dora_scale.as_ref(),
            ))
        }
        PatchPayload::Boft {
            blocks,
            rescale,
            constraint,
            dora_scale,
        } => {
            if dora_scale.is_some() && constraint.is_none() {
                return Err(PatchGraphError::InvalidPayload(
                    "BOFT with DORA requires an explicit constraint alpha".into(),
                ));
            }
            let transformed = apply_boft(
                backend,
                base,
                blocks,
                rescale.as_ref(),
                *constraint,
                adapter_strength,
                target_dtype,
                context,
            )?;
            let difference = backend_binary(
                backend,
                &transformed,
                base,
                BinaryOperation::Subtract,
                context,
            )?;
            Ok((
                difference,
                if dora_scale.is_some() {
                    constraint.unwrap_or(0.0)
                } else {
                    1.0
                },
                dora_scale.as_ref(),
            ))
        }
        _ => Err(PatchGraphError::InvalidPayload(
            "payload is not factorized".into(),
        )),
    }
}

fn canonical_matrix_product(
    backend: &dyn TensorBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    let left = flatten_first_axis(left)?;
    let right = flatten_first_axis(right)?;
    if left.descriptor().shape()[1] != right.descriptor().shape()[0] {
        return Err(PatchGraphError::InvalidPayload(
            "matrix product contracted dimensions differ".into(),
        ));
    }
    let rows = left.descriptor().shape()[0];
    let columns = right.descriptor().shape()[1];
    let left = reshape_read_only(&left, vec![1, rows, left.descriptor().shape()[1]])?;
    let right = reshape_read_only(&right, vec![1, right.descriptor().shape()[0], columns])?;
    let output = backend_batch_matrix_product(backend, &left, &right, context)?;
    reshape_read_only(&output, vec![rows, columns])
}

fn backend_batch_matrix_product(
    backend: &dyn TensorBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    if left.descriptor().dtype() != right.descriptor().dtype() {
        return Err(PatchGraphError::InvalidPayload(
            "batch matrix product compute dtypes differ".into(),
        ));
    }
    let compute_dtype = left.descriptor().dtype();
    let [batch, rows, contracted] = left.descriptor().shape() else {
        return Err(PatchGraphError::InvalidPayload(
            "batch matrix product left operand must have rank three".into(),
        ));
    };
    let [right_batch, right_contracted, columns] = right.descriptor().shape() else {
        return Err(PatchGraphError::InvalidPayload(
            "batch matrix product right operand must have rank three".into(),
        ));
    };
    if batch != right_batch || contracted != right_contracted {
        return Err(PatchGraphError::InvalidPayload(
            "batch matrix product dimensions are incompatible".into(),
        ));
    }
    let left = backend_cast_tensor(backend, left, DType::F32, false, false, context)?;
    let right = backend_cast_tensor(backend, right, DType::F32, false, false, context)?;
    let descriptor = TensorDescriptor::contiguous(
        vec![*batch, *rows, *columns],
        DType::F32,
        backend.device(),
        context.stream,
    )?;
    let (output, event) = backend.linear_algebra(
        LinearAlgebraOperation::BatchMatrixMultiply,
        &[left, right],
        descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    round_tensor(backend, &output, compute_dtype, context)
}

fn backend_binary(
    backend: &dyn TensorBackend,
    left: &Tensor,
    right: &Tensor,
    operation: BinaryOperation,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    if left.descriptor().shape() != right.descriptor().shape() {
        return Err(PatchGraphError::InvalidPayload(
            "canonical binary tensor shapes differ".into(),
        ));
    }
    if left.descriptor().dtype() != right.descriptor().dtype() {
        return Err(PatchGraphError::InvalidPayload(
            "canonical binary tensor compute dtypes differ".into(),
        ));
    }
    let compute_dtype = left.descriptor().dtype();
    let left = backend_cast_tensor(backend, left, DType::F32, false, false, context)?;
    let right = backend_cast_tensor(backend, right, DType::F32, false, false, context)?;
    let descriptor = TensorDescriptor::contiguous(
        left.descriptor().shape().to_vec(),
        DType::F32,
        backend.device(),
        context.stream,
    )?;
    let (output, event) = backend.binary(operation, &left, &right, descriptor, context)?;
    backend.wait_event(event, context)?;
    round_tensor(backend, &output, compute_dtype, context)
}

fn backend_binary_broadcast(
    backend: &dyn TensorBackend,
    left: &Tensor,
    right: &Tensor,
    operation: BinaryOperation,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    if left.descriptor().dtype() != right.descriptor().dtype() {
        return Err(PatchGraphError::InvalidPayload(
            "canonical broadcast compute dtypes differ".into(),
        ));
    }
    let compute_dtype = left.descriptor().dtype();
    let left = backend_cast_tensor(backend, left, DType::F32, false, false, context)?;
    let right = backend_cast_tensor(backend, right, DType::F32, false, false, context)?;
    let shape =
        sim_tensor::binary_broadcast_shape(left.descriptor().shape(), right.descriptor().shape())?;
    let descriptor =
        TensorDescriptor::contiguous(shape, DType::F32, backend.device(), context.stream)?;
    let (output, event) = backend.binary(operation, &left, &right, descriptor, context)?;
    backend.wait_event(event, context)?;
    round_tensor(backend, &output, compute_dtype, context)
}

fn backend_binary_scalar(
    backend: &dyn TensorBackend,
    input: &Tensor,
    operation: BinaryOperation,
    scalar: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    let compute_dtype = input.descriptor().dtype();
    let input = backend_cast_tensor(backend, input, DType::F32, false, false, context)?;
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        backend.device(),
        context.stream,
    )?;
    let (output, event) = backend.binary_scalar(
        operation,
        &input,
        Scalar::Float(f64::from(scalar)),
        ScalarSide::Right,
        descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    round_tensor(backend, &output, compute_dtype, context)
}

fn backend_left_scalar_binary(
    backend: &dyn TensorBackend,
    input: &Tensor,
    operation: BinaryOperation,
    scalar: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    let compute_dtype = input.descriptor().dtype();
    let input = backend_cast_tensor(backend, input, DType::F32, false, false, context)?;
    let descriptor = TensorDescriptor::contiguous(
        input.descriptor().shape().to_vec(),
        DType::F32,
        backend.device(),
        context.stream,
    )?;
    let (output, event) = backend.binary_scalar(
        operation,
        &input,
        Scalar::Float(f64::from(scalar)),
        ScalarSide::Left,
        descriptor,
        context,
    )?;
    backend.wait_event(event, context)?;
    round_tensor(backend, &output, compute_dtype, context)
}

fn backend_cast_tensor(
    backend: &dyn TensorBackend,
    input: &Tensor,
    dtype: DType,
    non_blocking: bool,
    copy: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    let (output, event) = backend.cast_tensor(input, dtype, non_blocking, copy, context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn validate_finite_result(
    backend: &dyn TensorBackend,
    tensor: &Tensor,
    target_key: &str,
    context: &ExecutionContext<'_>,
) -> Result<(), PatchGraphError> {
    match backend.validate_finite(tensor, context) {
        Ok(()) => Ok(()),
        Err(TensorError::InvalidNumeric { .. }) => {
            Err(PatchGraphError::NonFiniteResult(target_key.to_owned()))
        }
        Err(error) => Err(PatchGraphError::Tensor(error)),
    }
}

fn transpose_matrix(
    backend: &dyn TensorBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    if tensor.descriptor().rank() != 2 {
        return Err(PatchGraphError::InvalidPayload(
            "matrix transpose requires rank two".into(),
        ));
    }
    permute_contiguous(backend, tensor, &[1, 0], context)
}

fn permute_contiguous(
    backend: &dyn TensorBackend,
    tensor: &Tensor,
    permutation: &[usize],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    context.check()?;
    let view = tensor.view(
        tensor.descriptor().permuted_view(permutation)?,
        ViewAccess::ReadOnly,
    )?;
    backend_cast_tensor(
        backend,
        &view,
        view.descriptor().dtype(),
        false,
        true,
        context,
    )
}

fn reshape_read_only(tensor: &Tensor, shape: Vec<u64>) -> Result<Tensor, PatchGraphError> {
    Ok(tensor.view(
        tensor.descriptor().reshaped_view(shape)?,
        ViewAccess::ReadOnly,
    )?)
}

fn canonical_kronecker(
    backend: &dyn TensorBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    if left.descriptor().dtype() != right.descriptor().dtype() {
        return Err(PatchGraphError::InvalidPayload(
            "Kronecker compute dtypes differ".into(),
        ));
    }
    let compute_dtype = left.descriptor().dtype();
    let rank = left.descriptor().rank().max(right.descriptor().rank());
    let mut left_shape = vec![1_u64; rank - left.descriptor().rank()];
    left_shape.extend_from_slice(left.descriptor().shape());
    let mut right_shape = vec![1_u64; rank - right.descriptor().rank()];
    right_shape.extend_from_slice(right.descriptor().shape());
    let output_shape = left_shape
        .iter()
        .zip(&right_shape)
        .map(|(left, right)| {
            left.checked_mul(*right)
                .ok_or(PatchGraphError::ShapeOverflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let left = backend_cast_tensor(backend, left, DType::F32, false, false, context)?;
    let right = backend_cast_tensor(backend, right, DType::F32, false, false, context)?;
    let (output, event) = backend.kronecker_product(&left, &right, context)?;
    backend.wait_event(event, context)?;
    if output.descriptor().shape() != output_shape {
        return Err(PatchGraphError::InvalidPayload(
            "canonical Kronecker owner returned an unexpected shape".into(),
        ));
    }
    round_tensor(backend, &output, compute_dtype, context)
}

fn canonical_tucker(
    backend: &dyn TensorBackend,
    up: &PatchTensor,
    down: &PatchTensor,
    core: &PatchTensor,
    compute_dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    if core.shape.len() < 2 {
        return Err(PatchGraphError::InvalidPayload(
            "Tucker core must have rank at least two".into(),
        ));
    }
    let up = orient_tucker_factor(backend, up, core.shape[0], compute_dtype, context)?;
    let down = orient_tucker_factor(backend, down, core.shape[1], compute_dtype, context)?;
    let trailing = u64::try_from(product_usize(&core.shape[2..])?)
        .map_err(|_| PatchGraphError::ShapeOverflow)?;
    let core_tensor = patch_tensor_to_tensor(backend, core, compute_dtype, context)?;
    let mut core_permutation = vec![0];
    core_permutation.extend(2..core.shape.len());
    core_permutation.push(1);
    let core_tensor = permute_contiguous(backend, &core_tensor, &core_permutation, context)?;
    let core_tensor = reshape_read_only(
        &core_tensor,
        vec![1, core.shape[0] * trailing, core.shape[1]],
    )?;
    let down_tensor = reshape_read_only(
        &down,
        vec![
            1,
            down.descriptor().shape()[0],
            down.descriptor().shape()[1],
        ],
    )?;
    let first = backend_batch_matrix_product(backend, &core_tensor, &down_tensor, context)?;
    let mut first_shape = vec![core.shape[0]];
    first_shape.extend_from_slice(&core.shape[2..]);
    first_shape.push(down.descriptor().shape()[1]);
    let first = reshape_read_only(&first, first_shape)?;
    let last = first.descriptor().rank() - 1;
    let mut first_permutation = vec![0, last];
    first_permutation.extend(1..last);
    let first = permute_contiguous(backend, &first, &first_permutation, context)?;
    let first = reshape_read_only(
        &first,
        vec![1, core.shape[0], down.descriptor().shape()[1] * trailing],
    )?;
    let up_tensor = permute_contiguous(backend, &up, &[1, 0], context)?;
    let up_tensor = reshape_read_only(
        &up_tensor,
        vec![1, up.descriptor().shape()[1], up.descriptor().shape()[0]],
    )?;
    let output = backend_batch_matrix_product(backend, &up_tensor, &first, context)?;
    let mut output_shape = vec![up.descriptor().shape()[1], down.descriptor().shape()[1]];
    output_shape.extend_from_slice(&core.shape[2..]);
    let output = reshape_read_only(&output, output_shape)?;
    Ok(output)
}

fn orient_tucker_factor(
    backend: &dyn TensorBackend,
    factor: &PatchTensor,
    contracted: u64,
    compute_dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    let factor = flatten_first_axis(&patch_tensor_to_tensor(
        backend,
        factor,
        compute_dtype,
        context,
    )?)?;
    if factor.descriptor().shape()[0] == contracted {
        return Ok(factor);
    }
    if factor.descriptor().shape()[1] != contracted {
        return Err(PatchGraphError::InvalidPayload(
            "Tucker factor does not match its core axis".into(),
        ));
    }
    transpose_matrix(backend, &factor, context)
}

fn resolve_full_or_decomposed(
    backend: &dyn TensorBackend,
    name: &'static str,
    full: Option<&PatchTensor>,
    up: Option<&PatchTensor>,
    down: Option<&PatchTensor>,
    tucker: Option<&PatchTensor>,
    compute_dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<(Tensor, Option<u64>), PatchGraphError> {
    match (full, up, down) {
        (Some(full), None, None) => Ok((
            patch_tensor_to_tensor(backend, full, compute_dtype, context)?,
            None,
        )),
        (None, Some(up), Some(down)) => {
            let rank = down.shape.first().copied();
            let value = if let Some(tucker) = tucker {
                canonical_tucker(backend, up, down, tucker, compute_dtype, context)?
            } else {
                let up = patch_tensor_to_tensor(backend, up, compute_dtype, context)?;
                let down = patch_tensor_to_tensor(backend, down, compute_dtype, context)?;
                canonical_matrix_product(backend, &up, &down, context)?
            };
            Ok((value, rank))
        }
        _ => Err(PatchGraphError::InvalidPayload(format!(
            "{name} requires exactly a full tensor or an up/down pair"
        ))),
    }
}

fn patch_tensor_to_tensor(
    backend: &dyn TensorBackend,
    tensor: &PatchTensor,
    dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    let (output, event) =
        backend.upload_f32_payload(&tensor.shape, &tensor.values, DType::F32, context)?;
    backend.wait_event(event, context)?;
    round_tensor(backend, &output, dtype, context)
}

#[cfg(test)]
fn tensor_to_patch_tensor(
    backend: &dyn TensorBackend,
    tensor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<PatchTensor, PatchGraphError> {
    Ok(PatchTensor {
        shape: tensor.descriptor().shape().to_vec(),
        values: sim_tensor::generated_sim_operator_indirection_01::tensor_to_f32_with_backend_exact_native(
            backend, tensor, context,
        )?,
    })
}

fn round_tensor(
    backend: &dyn TensorBackend,
    tensor: &Tensor,
    dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    if dtype == DType::F32 {
        return Ok(tensor.clone());
    }
    backend_cast_tensor(backend, tensor, dtype, false, false, context)
}

fn current_dtype_epsilon(dtype: DType) -> Result<f32, PatchGraphError> {
    let epsilon = dtype.floating_point_info()?.epsilon();
    let epsilon = epsilon as f32;
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(PatchGraphError::InvalidPayload(
            "patch target dtype has no finite positive epsilon".into(),
        ));
    }
    Ok(epsilon)
}

fn flatten_first_axis(tensor: &Tensor) -> Result<Tensor, PatchGraphError> {
    let rows =
        *tensor.descriptor().shape().first().ok_or_else(|| {
            PatchGraphError::InvalidPayload("cannot flatten a scalar payload".into())
        })?;
    let columns =
        tensor.descriptor().shape()[1..]
            .iter()
            .try_fold(1_u64, |product, dimension| {
                product
                    .checked_mul(*dimension)
                    .ok_or(PatchGraphError::ShapeOverflow)
            })?;
    reshape_read_only(tensor, vec![rows, columns])
}

fn infer_glora_layout(
    first_a: &PatchTensor,
    second_a: &PatchTensor,
    first_b: &PatchTensor,
    second_b: &PatchTensor,
    base_shape: &[u64],
) -> Result<(bool, u64), PatchGraphError> {
    let dimension = |tensor: &PatchTensor, axis: usize| {
        tensor.shape.get(axis).copied().ok_or_else(|| {
            PatchGraphError::InvalidPayload("GLoRA factors must have rank at least two".into())
        })
    };
    let old_layout = dimension(second_b, 1)? == dimension(first_b, 0)?
        && dimension(first_b, 0)? == dimension(first_a, 0)?
        && dimension(first_a, 0)? == dimension(second_a, 1)?;
    let new_layout = dimension(second_b, 0)? == dimension(first_b, 1)?
        && dimension(first_b, 1)? == dimension(first_a, 1)?
        && dimension(first_a, 1)? == dimension(second_a, 0)?;
    if old_layout {
        let rank = dimension(first_a, 0)?;
        if !new_layout
            || (dimension(second_a, 0)? == base_shape[0]
                && base_shape.get(1).copied() == Some(base_shape[0]))
        {
            return Ok((true, rank));
        }
    }
    if new_layout {
        return Ok((false, dimension(second_a, 0)?));
    }
    Err(PatchGraphError::InvalidPayload(
        "GLoRA factor shapes match neither the source old nor new layout".into(),
    ))
}

fn canonical_glora_input_axis(
    backend: &dyn TensorBackend,
    weight: &Tensor,
    factor: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    let weight_shape = weight.descriptor().shape();
    let factor_shape = factor.descriptor().shape();
    if weight_shape.len() < 3 || factor_shape.len() != 2 || weight_shape[1] != factor_shape[0] {
        return Err(PatchGraphError::InvalidPayload(
            "GLoRA convolution input axis does not match its factor".into(),
        ));
    }
    let trailing = u64::try_from(product_usize(&weight_shape[2..])?)
        .map_err(|_| PatchGraphError::ShapeOverflow)?;
    let mut permutation = vec![0];
    permutation.extend(2..weight_shape.len());
    permutation.push(1);
    let weight_tensor = permute_contiguous(backend, weight, &permutation, context)?;
    let weight_tensor = reshape_read_only(
        &weight_tensor,
        vec![1, weight_shape[0] * trailing, weight_shape[1]],
    )?;
    let factor_tensor = reshape_read_only(factor, vec![1, factor_shape[0], factor_shape[1]])?;
    let output = backend_batch_matrix_product(backend, &weight_tensor, &factor_tensor, context)?;
    let mut intermediate_shape = vec![weight_shape[0]];
    intermediate_shape.extend_from_slice(&weight_shape[2..]);
    intermediate_shape.push(factor_shape[1]);
    let output = reshape_read_only(&output, intermediate_shape)?;
    let last = output.descriptor().rank() - 1;
    let mut output_permutation = vec![0, last];
    output_permutation.extend(1..last);
    let output = permute_contiguous(backend, &output, &output_permutation, context)?;
    Ok(output)
}

fn canonical_affine(
    backend: &dyn TensorBackend,
    input: &Tensor,
    scale: f32,
    bias: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    let scaled = backend_binary_scalar(backend, input, BinaryOperation::Multiply, scale, context)?;
    let output = if bias == 0.0 {
        scaled
    } else {
        backend_binary_scalar(backend, &scaled, BinaryOperation::Add, bias, context)?
    };
    Ok(output)
}

fn canonical_target_affine(
    backend: &dyn TensorBackend,
    input: &Tensor,
    scale: f32,
    bias: f32,
    target_dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    let scaled = backend_binary_scalar(backend, input, BinaryOperation::Multiply, scale, context)?;
    let scaled = round_tensor(backend, &scaled, target_dtype, context)?;
    if bias == 0.0 {
        return Ok(scaled);
    }
    let output = backend_binary_scalar(backend, &scaled, BinaryOperation::Add, bias, context)?;
    round_tensor(backend, &output, target_dtype, context)
}

fn add_difference(
    backend: &dyn TensorBackend,
    base: &Tensor,
    difference: &Tensor,
    strength: f32,
    transform: PatchValueTransform,
    dora: Option<(&PatchTensor, f32)>,
    target_dtype: DType,
    dtype_epsilon: f32,
    cast_before_scaling: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    require_same_shape(
        "patch difference",
        base.descriptor().shape(),
        difference.descriptor().shape(),
    )?;
    if let Some((scale, alpha)) = dora {
        let transformed = canonical_affine(backend, difference, alpha, 0.0, context)?;
        let transformed = canonical_affine(
            backend,
            &transformed,
            transform.scale,
            transform.bias,
            context,
        )?;
        let transformed = round_tensor(backend, &transformed, target_dtype, context)?;
        return apply_dora(
            backend,
            base,
            &transformed,
            scale,
            1.0,
            strength,
            target_dtype,
            dtype_epsilon,
            context,
        );
    }
    let difference = if cast_before_scaling {
        round_tensor(backend, difference, target_dtype, context)?
    } else {
        difference.clone()
    };
    let transformed = canonical_affine(backend, &difference, strength, 0.0, context)?;
    let transformed = round_tensor(backend, &transformed, target_dtype, context)?;
    let transformed = canonical_target_affine(
        backend,
        &transformed,
        transform.scale,
        transform.bias,
        target_dtype,
        context,
    )?;
    let output = backend_binary(backend, base, &transformed, BinaryOperation::Add, context)?;
    round_tensor(backend, &output, target_dtype, context)
}

fn apply_dora(
    backend: &dyn TensorBackend,
    base: &Tensor,
    difference: &Tensor,
    scale: &PatchTensor,
    alpha: f32,
    strength: f32,
    target_dtype: DType,
    dtype_epsilon: f32,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    let base_shape = base.descriptor().shape();
    let output_extent =
        usize::try_from(base_shape[0]).map_err(|_| PatchGraphError::ShapeOverflow)?;
    let input_extent = usize::try_from(*base_shape.get(1).ok_or_else(|| {
        PatchGraphError::InvalidPayload("DORA requires rank at least two".into())
    })?)
    .map_err(|_| PatchGraphError::ShapeOverflow)?;
    let scale_leading = scale.shape.first().copied().ok_or_else(|| {
        PatchGraphError::InvalidPayload("DORA scale must have rank at least one".into())
    })?;
    let output_axis = scale_leading == base_shape[0];
    let input_axis = scale_leading == base_shape[1];
    if !output_axis && !input_axis {
        return Err(PatchGraphError::InvalidPayload(
            "DORA scale leading dimension must match the output or input axis".into(),
        ));
    }
    require_same_shape(
        "DORA difference",
        base_shape,
        difference.descriptor().shape(),
    )?;
    let difference = canonical_affine(backend, difference, alpha, 0.0, context)?;
    let calculated = backend_binary(backend, base, &difference, BinaryOperation::Add, context)?;
    let calculated = round_tensor(backend, &calculated, target_dtype, context)?;
    let norm_extent = if output_axis {
        output_extent
    } else {
        input_extent
    };
    if scale.values.len() != norm_extent {
        return Err(PatchGraphError::InvalidPayload(
            "DORA scale must contain one value for its selected axis".into(),
        ));
    }
    let norm_source = if output_axis { base } else { &calculated };
    let dimensions = if output_axis {
        (1..base_shape.len())
            .map(|dimension| i64::try_from(dimension).map_err(|_| PatchGraphError::ShapeOverflow))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        std::iter::once(0)
            .chain(2..base_shape.len())
            .map(|dimension| i64::try_from(dimension).map_err(|_| PatchGraphError::ShapeOverflow))
            .collect::<Result<Vec<_>, _>>()?
    };
    let (norm, event) = backend.vector_norm(
        norm_source,
        2.0,
        &dimensions,
        true,
        Some(DType::F32),
        context,
    )?;
    backend.wait_event(event, context)?;
    let norm = round_tensor(backend, &norm, target_dtype, context)?;
    let norm = backend_binary_scalar(backend, &norm, BinaryOperation::Add, dtype_epsilon, context)?;
    let norm = round_tensor(backend, &norm, target_dtype, context)?;
    let scale = patch_tensor_to_tensor(backend, scale, target_dtype, context)?;
    let ratio = backend_binary_broadcast(backend, &scale, &norm, BinaryOperation::Divide, context)?;
    let ratio = backend_cast_tensor(backend, &ratio, target_dtype, false, false, context)?;
    let normalized = backend_binary_broadcast(
        backend,
        &calculated,
        &ratio,
        BinaryOperation::Multiply,
        context,
    )?;
    if normalized.descriptor().shape() != base_shape {
        return Err(PatchGraphError::InvalidPayload(format!(
            "DORA broadcast produces shape {:?} instead of {:?}",
            normalized.descriptor().shape(),
            base_shape
        )));
    }
    let normalized = round_tensor(backend, &normalized, target_dtype, context)?;
    if strength == 1.0 {
        return Ok(normalized);
    }
    let difference = backend_binary(
        backend,
        &normalized,
        base,
        BinaryOperation::Subtract,
        context,
    )?;
    let difference = round_tensor(backend, &difference, target_dtype, context)?;
    let difference = canonical_affine(backend, &difference, strength, 0.0, context)?;
    let difference = round_tensor(backend, &difference, target_dtype, context)?;
    let output = backend_binary(backend, base, &difference, BinaryOperation::Add, context)?;
    round_tensor(backend, &output, target_dtype, context)
}

fn apply_oft(
    backend: &dyn TensorBackend,
    base: &Tensor,
    blocks: &PatchTensor,
    _rescale: Option<&PatchTensor>,
    constraint: Option<f32>,
    strength: f32,
    target_dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    if blocks.shape.len() != 3 || blocks.shape[1] != blocks.shape[2] {
        return Err(PatchGraphError::InvalidPayload(
            "OFT blocks must have shape [block_count, block_size, block_size]".into(),
        ));
    }
    let block_count = blocks.shape[0];
    let block_size = blocks.shape[1];
    if block_count.checked_mul(block_size) != base.descriptor().shape().first().copied() {
        return Err(PatchGraphError::InvalidPayload(
            "OFT blocks do not cover the output axis".into(),
        ));
    }
    let rotation = canonical_cayley_rotation(backend, blocks, constraint, target_dtype, context)?;
    let rotation = round_tensor(backend, &rotation, target_dtype, context)?;
    let rotation = backend_binary_scalar(
        backend,
        &rotation,
        BinaryOperation::Multiply,
        strength,
        context,
    )?;
    let rotation = round_tensor(backend, &rotation, target_dtype, context)?;
    let (identity, event) = backend.eye(
        block_size,
        Some(block_size),
        DType::F32,
        Layout::Contiguous,
        context,
    )?;
    backend.wait_event(event, context)?;
    let identity = round_tensor(backend, &identity, target_dtype, context)?;
    let identity = backend_binary_scalar(
        backend,
        &identity,
        BinaryOperation::Multiply,
        strength,
        context,
    )?;
    let identity = round_tensor(backend, &identity, target_dtype, context)?;
    let rotation = backend_binary_broadcast(
        backend,
        &rotation,
        &identity,
        BinaryOperation::Subtract,
        context,
    )?;
    let rotation = round_tensor(backend, &rotation, target_dtype, context)?;
    let trailing = u64::try_from(product_usize(&base.descriptor().shape()[1..])?)
        .map_err(|_| PatchGraphError::ShapeOverflow)?;
    let rotation = permute_contiguous(backend, &rotation, &[0, 2, 1], context)?;
    let blocked = reshape_read_only(base, vec![block_count, block_size, trailing])?;
    let output = backend_batch_matrix_product(backend, &rotation, &blocked, context)?;
    let output = round_tensor(backend, &output, target_dtype, context)?;
    reshape_read_only(&output, base.descriptor().shape().to_vec())
}

fn apply_boft(
    backend: &dyn TensorBackend,
    base: &Tensor,
    blocks: &PatchTensor,
    rescale: Option<&PatchTensor>,
    constraint: Option<f32>,
    strength: f32,
    target_dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    if blocks.shape.len() != 4
        || blocks.shape[2] != blocks.shape[3]
        || !blocks.shape[2].is_multiple_of(2)
    {
        return Err(PatchGraphError::InvalidPayload(
            "BOFT blocks must have shape [stages, block_count, even_block_size, even_block_size]"
                .into(),
        ));
    }
    let stages = usize::try_from(blocks.shape[0]).map_err(|_| PatchGraphError::ShapeOverflow)?;
    let block_count =
        usize::try_from(blocks.shape[1]).map_err(|_| PatchGraphError::ShapeOverflow)?;
    let block_size =
        usize::try_from(blocks.shape[2]).map_err(|_| PatchGraphError::ShapeOverflow)?;
    let output_extent = usize::try_from(base.descriptor().shape()[0])
        .map_err(|_| PatchGraphError::ShapeOverflow)?;
    if block_count
        .checked_mul(block_size)
        .ok_or(PatchGraphError::ShapeOverflow)?
        != output_extent
    {
        return Err(PatchGraphError::InvalidPayload(
            "BOFT blocks do not cover the output axis".into(),
        ));
    }
    let rotation = canonical_cayley_rotation(backend, blocks, constraint, target_dtype, context)?;
    let rotation = round_tensor(backend, &rotation, target_dtype, context)?;
    let row_width = u64::try_from(product_usize(&base.descriptor().shape()[1..])?)
        .map_err(|_| PatchGraphError::ShapeOverflow)?;
    let mut current = base.clone();
    let identity = if strength == 1.0 {
        None
    } else {
        let (identity, event) = backend.eye(
            blocks.shape[2],
            Some(blocks.shape[2]),
            DType::F32,
            Layout::Contiguous,
            context,
        )?;
        backend.wait_event(event, context)?;
        let identity = backend_binary_scalar(
            backend,
            &identity,
            BinaryOperation::Multiply,
            1.0 - strength,
            context,
        )?;
        Some(round_tensor(backend, &identity, target_dtype, context)?)
    };
    let half = block_size / 2;
    for stage in 0..stages {
        context.cancellation.check()?;
        let group_width = (1_usize
            .checked_shl(u32::try_from(stage).map_err(|_| PatchGraphError::ShapeOverflow)?)
            .ok_or(PatchGraphError::ShapeOverflow)?)
        .checked_mul(half)
        .ok_or(PatchGraphError::ShapeOverflow)?;
        if output_extent % (2 * group_width) != 0 {
            return Err(PatchGraphError::InvalidPayload(
                "BOFT butterfly stage is incompatible with the output axis".into(),
            ));
        }
        let outer = output_extent / (2 * group_width);
        let blocked = reshape_read_only(
            &current,
            vec![
                u64::try_from(outer).map_err(|_| PatchGraphError::ShapeOverflow)?,
                2,
                u64::try_from(group_width).map_err(|_| PatchGraphError::ShapeOverflow)?,
                row_width,
            ],
        )?;
        let blocked = permute_contiguous(backend, &blocked, &[0, 2, 1, 3], context)?;
        let blocked =
            reshape_read_only(&blocked, vec![blocks.shape[1], blocks.shape[2], row_width])?;
        let stage_rotation = rotation.narrow_read_only(
            0,
            i64::try_from(stage).map_err(|_| PatchGraphError::ShapeOverflow)?,
            1,
        )?;
        let stage_rotation = reshape_read_only(
            &stage_rotation,
            vec![blocks.shape[1], blocks.shape[2], blocks.shape[3]],
        )?;
        let stage_rotation = if let Some(identity) = &identity {
            let stage_rotation = backend_binary_scalar(
                backend,
                &stage_rotation,
                BinaryOperation::Multiply,
                strength,
                context,
            )?;
            let stage_rotation = round_tensor(backend, &stage_rotation, target_dtype, context)?;
            backend_binary_broadcast(
                backend,
                &stage_rotation,
                identity,
                BinaryOperation::Add,
                context,
            )?
        } else {
            stage_rotation
        };
        let next = backend_batch_matrix_product(backend, &stage_rotation, &blocked, context)?;
        let next = if identity.is_none() {
            round_tensor(backend, &next, target_dtype, context)?
        } else {
            next
        };
        let next = reshape_read_only(
            &next,
            vec![
                u64::try_from(outer).map_err(|_| PatchGraphError::ShapeOverflow)?,
                u64::try_from(group_width).map_err(|_| PatchGraphError::ShapeOverflow)?,
                2,
                row_width,
            ],
        )?;
        let next = permute_contiguous(backend, &next, &[0, 2, 1, 3], context)?;
        current = reshape_read_only(&next, base.descriptor().shape().to_vec())?;
    }
    apply_rescale(backend, &current, rescale, target_dtype, context)
}

fn canonical_cayley_rotation(
    backend: &dyn TensorBackend,
    blocks: &PatchTensor,
    constraint: Option<f32>,
    compute_dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    let matrix_size = *blocks
        .shape
        .last()
        .ok_or_else(|| PatchGraphError::InvalidPayload("rotation blocks are empty".into()))?;
    let matrix_elements = matrix_size
        .checked_mul(matrix_size)
        .ok_or(PatchGraphError::ShapeOverflow)?;
    let matrix_count = u64::try_from(blocks.values.len())
        .map_err(|_| PatchGraphError::ShapeOverflow)?
        / matrix_elements;
    let blocks_tensor = patch_tensor_to_tensor(backend, blocks, compute_dtype, context)?;
    let mut permutation = (0..blocks.shape.len()).collect::<Vec<_>>();
    let last = permutation
        .len()
        .checked_sub(1)
        .ok_or(PatchGraphError::ShapeOverflow)?;
    let penultimate = last.checked_sub(1).ok_or(PatchGraphError::ShapeOverflow)?;
    permutation.swap(penultimate, last);
    let transpose = permute_contiguous(backend, &blocks_tensor, &permutation, context)?;
    let blocks_tensor =
        reshape_read_only(&blocks_tensor, vec![matrix_count, matrix_size, matrix_size])?;
    let transpose = reshape_read_only(&transpose, vec![matrix_count, matrix_size, matrix_size])?;
    let skew = backend_binary(
        backend,
        &blocks_tensor,
        &transpose,
        BinaryOperation::Subtract,
        context,
    )?;
    let skew = if let Some(constraint) = constraint.filter(|value| *value > 0.0) {
        let dimensions = [0_i64, 1, 2];
        let (norm, event) =
            backend.vector_norm(&skew, 2.0, &dimensions, false, Some(DType::F32), context)?;
        backend.wait_event(event, context)?;
        let norm = round_tensor(backend, &norm, compute_dtype, context)?;
        let norm = backend_binary_scalar(backend, &norm, BinaryOperation::Add, 1.0e-8, context)?;
        let ratio = backend_left_scalar_binary(
            backend,
            &norm,
            BinaryOperation::Divide,
            constraint,
            context,
        )?;
        let factor =
            backend_binary_scalar(backend, &ratio, BinaryOperation::Minimum, 1.0, context)?;
        backend_binary_broadcast(backend, &skew, &factor, BinaryOperation::Multiply, context)?
    } else {
        skew
    };
    let (identity, event) = backend.eye(
        matrix_size,
        Some(matrix_size),
        compute_dtype,
        Layout::Contiguous,
        context,
    )?;
    backend.wait_event(event, context)?;
    let plus = backend_binary_broadcast(backend, &skew, &identity, BinaryOperation::Add, context)?;
    let minus = backend_binary_broadcast(
        backend,
        &identity,
        &skew,
        BinaryOperation::Subtract,
        context,
    )?;
    let minus_f32 = backend_cast_tensor(backend, &minus, DType::F32, false, false, context)?;
    let (inverse, event) = backend.matrix_inverse(&minus_f32, context)?;
    backend.wait_event(event, context)?;
    let inverse = round_tensor(backend, &inverse, compute_dtype, context)?;
    let rotation = backend_batch_matrix_product(backend, &plus, &inverse, context)?;
    let rotation = reshape_read_only(&rotation, blocks.shape.clone())?;
    Ok(rotation)
}

fn apply_rescale(
    backend: &dyn TensorBackend,
    tensor: &Tensor,
    rescale: Option<&PatchTensor>,
    compute_dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    let Some(rescale) = rescale else {
        return Ok(tensor.clone());
    };
    let scale = patch_tensor_to_tensor(backend, rescale, compute_dtype, context)?;
    let output =
        backend_binary_broadcast(backend, tensor, &scale, BinaryOperation::Multiply, context)?;
    if output.descriptor().shape() != tensor.descriptor().shape() {
        return Err(PatchGraphError::InvalidPayload(format!(
            "BOFT rescale broadcasts to {:?} instead of {:?}",
            output.descriptor().shape(),
            tensor.descriptor().shape()
        )));
    }
    Ok(output)
}

fn extract_slices(
    backend: &dyn TensorBackend,
    tensor: &Tensor,
    slices: &[PatchSlice],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    if slices.is_empty() {
        return Ok(tensor.clone());
    }
    let mut selected = tensor.clone();
    for slice in slices {
        let dimension =
            usize::try_from(slice.dimension).map_err(|_| PatchGraphError::ShapeOverflow)?;
        selected = selected.narrow_read_only(
            dimension,
            i64::try_from(slice.start).map_err(|_| PatchGraphError::ShapeOverflow)?,
            slice.length,
        )?;
    }
    let selected = backend_cast_tensor(
        backend,
        &selected,
        selected.descriptor().dtype(),
        false,
        true,
        context,
    )?;
    Ok(selected)
}

fn scatter_slices(
    backend: &dyn TensorBackend,
    tensor: &Tensor,
    patch: &Tensor,
    slices: &[PatchSlice],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    let expected = extract_slices(backend, tensor, slices, context)?;
    require_same_shape(
        "sliced patch",
        expected.descriptor().shape(),
        patch.descriptor().shape(),
    )?;
    let mut offsets = vec![0_u64; tensor.descriptor().rank()];
    for slice in slices {
        let dimension =
            usize::try_from(slice.dimension).map_err(|_| PatchGraphError::ShapeOverflow)?;
        offsets[dimension] = slice.start;
    }
    let (output, event) = backend.replace_rectangular_slice(tensor, patch, &offsets, context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn pad_tensor(
    backend: &dyn TensorBackend,
    tensor: &Tensor,
    shape: &[u64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, PatchGraphError> {
    if shape.len() != tensor.descriptor().rank()
        || shape
            .iter()
            .zip(tensor.descriptor().shape())
            .any(|(new, old)| new < old)
    {
        return Err(PatchGraphError::InvalidPayload(
            "padding cannot shrink or change tensor rank".into(),
        ));
    }
    let mut padding = Vec::with_capacity(shape.len() * 2);
    for (&new_extent, &old_extent) in shape.iter().zip(tensor.descriptor().shape()).rev() {
        padding.push(0);
        padding.push(
            i64::try_from(new_extent - old_extent).map_err(|_| PatchGraphError::ShapeOverflow)?,
        );
    }
    let (output, event) = backend.constant_pad(tensor, &padding, None, context)?;
    backend.wait_event(event, context)?;
    Ok(output)
}

fn product_usize(shape: &[u64]) -> Result<usize, PatchGraphError> {
    let product = shape.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or(PatchGraphError::ShapeOverflow)
    })?;
    usize::try_from(product).map_err(|_| PatchGraphError::ShapeOverflow)
}

fn require_same_shape(
    name: &'static str,
    expected: &[u64],
    actual: &[u64],
) -> Result<(), PatchGraphError> {
    if expected == actual {
        Ok(())
    } else {
        Err(PatchGraphError::InvalidPayload(format!(
            "{name} shape mismatch: expected {expected:?}, got {actual:?}"
        )))
    }
}

fn require_element_count_tensor(
    name: &'static str,
    shape: &[u64],
    tensor: &Tensor,
) -> Result<(), PatchGraphError> {
    let expected =
        u64::try_from(product_usize(shape)?).map_err(|_| PatchGraphError::ShapeOverflow)?;
    if expected == tensor.descriptor().element_count()? {
        Ok(())
    } else {
        Err(PatchGraphError::InvalidPayload(format!(
            "{name} result does not fit target shape"
        )))
    }
}

fn validate_legacy_operations(operations: &[PatchOperation]) -> Result<(), PatchGraphError> {
    let mut operation_ids = BTreeSet::new();
    for operation in operations {
        validate_text("operation identifier", &operation.identifier)?;
        if !operation_ids.insert(operation.identifier.as_str()) {
            return Err(PatchGraphError::DuplicateOperation(
                operation.identifier.clone(),
            ));
        }
        if !operation.scale.is_finite() {
            return Err(PatchGraphError::NonFiniteScale(
                operation.identifier.clone(),
            ));
        }
        if operation.targets.is_empty() {
            return Err(PatchGraphError::EmptyTargets(operation.identifier.clone()));
        }
        let mut keys = BTreeSet::new();
        for target in &operation.targets {
            validate_text("target key", &target.key)?;
            if !keys.insert(target.key.as_str()) {
                return Err(PatchGraphError::DuplicateTarget {
                    operation: operation.identifier.clone(),
                    key: target.key.clone(),
                });
            }
            if target.expected_shape.is_empty() || target.expected_shape.contains(&0) {
                return Err(PatchGraphError::InvalidShape(target.key.clone()));
            }
            let elements = target
                .expected_shape
                .iter()
                .try_fold(1_u64, |product, dimension| {
                    product
                        .checked_mul(*dimension)
                        .ok_or(PatchGraphError::ShapeOverflow)
                })?;
            let actual =
                u64::try_from(target.values.len()).map_err(|_| PatchGraphError::ShapeOverflow)?;
            if elements != actual {
                return Err(PatchGraphError::ValueCount {
                    key: target.key.clone(),
                    expected: elements,
                    actual,
                });
            }
            if target.values.iter().any(|value| !value.is_finite()) {
                return Err(PatchGraphError::NonFiniteValue(target.key.clone()));
            }
            if operation.kind == PatchKind::Replacement
                && target.application != PatchApplication::Replace
            {
                return Err(PatchGraphError::KindApplication {
                    operation: operation.identifier.clone(),
                    kind: operation.kind,
                    application: target.application,
                });
            }
        }
    }
    Ok(())
}

fn map_legacy_operations(
    operations: &[PatchOperation],
) -> Result<Vec<SemanticPatchOperation>, PatchGraphError> {
    let target_count = operations.iter().try_fold(0_usize, |count, operation| {
        count
            .checked_add(operation.targets.len())
            .ok_or(PatchGraphError::ShapeOverflow)
    })?;
    let mut mapped = Vec::with_capacity(target_count);
    for (operation_index, operation) in operations.iter().enumerate() {
        for (target_index, target) in operation.targets.iter().enumerate() {
            mapped.push(SemanticPatchOperation {
                identifier: format!("legacy:{operation_index}:{target_index}"),
                target_key: target.key.clone(),
                expected_shape: target.expected_shape.clone(),
                strength: operation.scale,
                strength_model: match target.application {
                    PatchApplication::Add => 1.0,
                    PatchApplication::Replace => 0.0,
                },
                slices: Vec::new(),
                transform: PatchValueTransform::default(),
                payload: PatchPayload::DenseDiff {
                    tensor: PatchTensor {
                        shape: target.expected_shape.clone(),
                        values: target.values.clone(),
                    },
                    pad_weight: false,
                },
            });
        }
    }
    validate_semantic_operations(&mapped)?;
    Ok(mapped)
}

fn validate_semantic_operations(
    operations: &[SemanticPatchOperation],
) -> Result<(), PatchGraphError> {
    let mut identifiers = BTreeSet::new();
    for operation in operations {
        validate_text("semantic operation identifier", &operation.identifier)?;
        validate_text("semantic target key", &operation.target_key)?;
        if !identifiers.insert(operation.identifier.as_str()) {
            return Err(PatchGraphError::DuplicateOperation(
                operation.identifier.clone(),
            ));
        }
        if operation.expected_shape.is_empty() || operation.expected_shape.contains(&0) {
            return Err(PatchGraphError::InvalidShape(operation.target_key.clone()));
        }
        product_usize(&operation.expected_shape)?;
        if !operation.strength.is_finite() || !operation.strength_model.is_finite() {
            return Err(PatchGraphError::NonFiniteScale(
                operation.identifier.clone(),
            ));
        }
        if matches!(operation.payload, PatchPayload::ScaleOnly) && operation.strength != 0.0 {
            return Err(PatchGraphError::InvalidPayload(
                "scale-only operation must have zero patch strength".into(),
            ));
        }
        validate_transform(operation.transform)?;
        let mut dimensions = BTreeSet::new();
        for slice in &operation.slices {
            let dimension =
                usize::try_from(slice.dimension).map_err(|_| PatchGraphError::ShapeOverflow)?;
            if dimension >= operation.expected_shape.len()
                || !dimensions.insert(dimension)
                || slice.length == 0
            {
                return Err(PatchGraphError::InvalidSlice(operation.identifier.clone()));
            }
            let end = slice
                .start
                .checked_add(slice.length)
                .ok_or(PatchGraphError::ShapeOverflow)?;
            if end > operation.expected_shape[dimension] {
                return Err(PatchGraphError::InvalidSlice(operation.identifier.clone()));
            }
        }
        validate_payload(&operation.payload, 0)?;
    }
    Ok(())
}

fn validate_payload(payload: &PatchPayload, depth: usize) -> Result<(), PatchGraphError> {
    if depth > MAX_SEMANTIC_PATCH_DEPTH {
        return Err(PatchGraphError::NestingDepth);
    }
    match payload {
        PatchPayload::ScaleOnly => Ok(()),
        PatchPayload::DenseDiff { tensor, .. } | PatchPayload::Set { tensor } => {
            validate_patch_tensor("payload", tensor)
        }
        PatchPayload::Lora {
            up,
            down,
            mid,
            alpha,
            dora_scale,
            reshape,
        } => {
            validate_patch_tensor("LoRA up", up)?;
            validate_patch_tensor("LoRA down", down)?;
            validate_optional_tensor("LoRA mid", mid.as_ref())?;
            validate_alpha(*alpha)?;
            validate_optional_dora(dora_scale.as_ref())?;
            if reshape
                .as_ref()
                .is_some_and(|shape| shape.is_empty() || shape.contains(&0))
            {
                return Err(PatchGraphError::InvalidPayload(
                    "LoRA reshape is invalid".into(),
                ));
            }
            Ok(())
        }
        PatchPayload::Loha {
            first_up,
            first_down,
            second_up,
            second_down,
            first_tucker,
            second_tucker,
            alpha,
            dora_scale,
        } => {
            for (name, tensor) in [
                ("LoHa first up", first_up),
                ("LoHa first down", first_down),
                ("LoHa second up", second_up),
                ("LoHa second down", second_down),
            ] {
                validate_patch_tensor(name, tensor)?;
            }
            if first_tucker.is_some() != second_tucker.is_some() {
                return Err(PatchGraphError::InvalidPayload(
                    "LoHa requires both Tucker tensors".into(),
                ));
            }
            validate_optional_tensor("LoHa first Tucker", first_tucker.as_ref())?;
            validate_optional_tensor("LoHa second Tucker", second_tucker.as_ref())?;
            validate_alpha(*alpha)?;
            validate_optional_dora(dora_scale.as_ref())
        }
        PatchPayload::Lokr {
            first,
            second,
            first_up,
            first_down,
            second_up,
            second_down,
            second_tucker,
            alpha,
            dora_scale,
        } => {
            validate_full_or_pair(
                "LoKr first",
                first.as_ref(),
                first_up.as_ref(),
                first_down.as_ref(),
            )?;
            validate_full_or_pair(
                "LoKr second",
                second.as_ref(),
                second_up.as_ref(),
                second_down.as_ref(),
            )?;
            if second_tucker.is_some() && second.is_some() {
                return Err(PatchGraphError::InvalidPayload(
                    "LoKr Tucker core requires decomposed second factor".into(),
                ));
            }
            validate_optional_tensor("LoKr second Tucker", second_tucker.as_ref())?;
            validate_alpha(*alpha)?;
            validate_optional_dora(dora_scale.as_ref())
        }
        PatchPayload::Oft {
            blocks,
            rescale,
            constraint,
            dora_scale,
        } => {
            validate_patch_tensor("OFT blocks", blocks)?;
            if blocks.shape.len() != 3 || blocks.shape[1] != blocks.shape[2] {
                return Err(PatchGraphError::InvalidPayload(
                    "OFT blocks must be square rank-three tensors".into(),
                ));
            }
            validate_optional_tensor("OFT rescale", rescale.as_ref())?;
            validate_constraint(*constraint)?;
            validate_optional_dora(dora_scale.as_ref())
        }
        PatchPayload::Boft {
            blocks,
            rescale,
            constraint,
            dora_scale,
        } => {
            validate_patch_tensor("BOFT blocks", blocks)?;
            if blocks.shape.len() != 4
                || blocks.shape[2] != blocks.shape[3]
                || !blocks.shape[2].is_multiple_of(2)
            {
                return Err(PatchGraphError::InvalidPayload(
                    "BOFT blocks must be rank-four with square even blocks".into(),
                ));
            }
            validate_optional_tensor("BOFT rescale", rescale.as_ref())?;
            validate_constraint(*constraint)?;
            if dora_scale.is_some() && constraint.is_none() {
                return Err(PatchGraphError::InvalidPayload(
                    "BOFT with DORA requires an explicit constraint alpha".into(),
                ));
            }
            validate_optional_dora(dora_scale.as_ref())
        }
        PatchPayload::Glora {
            first_a,
            second_a,
            first_b,
            second_b,
            alpha,
            dora_scale,
            ..
        } => {
            for (name, tensor) in [
                ("GLoRA first A", first_a),
                ("GLoRA second A", second_a),
                ("GLoRA first B", first_b),
                ("GLoRA second B", second_b),
            ] {
                validate_patch_tensor(name, tensor)?;
            }
            validate_alpha(*alpha)?;
            validate_optional_dora(dora_scale.as_ref())
        }
        PatchPayload::Dora {
            difference,
            scale,
            alpha,
        } => {
            validate_patch_tensor("DORA difference", difference)?;
            validate_dora_scale(scale)?;
            if !alpha.is_finite() {
                return Err(PatchGraphError::InvalidPayload(
                    "DORA alpha must be finite".into(),
                ));
            }
            Ok(())
        }
        PatchPayload::Nested {
            base,
            base_transform,
            patches,
        } => {
            validate_patch_tensor("nested base", base)?;
            validate_transform(*base_transform)?;
            if patches.is_empty() {
                return Err(PatchGraphError::InvalidPayload(
                    "nested patch list is empty".into(),
                ));
            }
            for patch in patches {
                if !patch.strength.is_finite() || !patch.strength_model.is_finite() {
                    return Err(PatchGraphError::InvalidPayload(
                        "nested strengths must be finite".into(),
                    ));
                }
                validate_transform(patch.transform)?;
                validate_payload(&patch.payload, depth + 1)?;
            }
            Ok(())
        }
        PatchPayload::ModelAsLora { target } => {
            validate_patch_tensor("model-as-LoRA target", target)?;
            Ok(())
        }
    }
}

fn validate_patch_tensor(name: &'static str, tensor: &PatchTensor) -> Result<(), PatchGraphError> {
    if tensor.shape.is_empty() || tensor.shape.contains(&0) {
        return Err(PatchGraphError::InvalidPayload(format!(
            "{name} has invalid shape"
        )));
    }
    let expected = product_usize(&tensor.shape)?;
    if expected != tensor.values.len() {
        return Err(PatchGraphError::InvalidPayload(format!(
            "{name} value count does not match shape"
        )));
    }
    if tensor.values.iter().any(|value| !value.is_finite()) {
        return Err(PatchGraphError::InvalidPayload(format!(
            "{name} contains a non-finite value"
        )));
    }
    Ok(())
}

fn validate_optional_tensor(
    name: &'static str,
    tensor: Option<&PatchTensor>,
) -> Result<(), PatchGraphError> {
    tensor.map_or(Ok(()), |tensor| validate_patch_tensor(name, tensor))
}

fn validate_dora_scale(tensor: &PatchTensor) -> Result<(), PatchGraphError> {
    validate_patch_tensor("DORA scale", tensor)
}

fn validate_optional_dora(tensor: Option<&PatchTensor>) -> Result<(), PatchGraphError> {
    tensor.map_or(Ok(()), validate_dora_scale)
}

fn validate_alpha(alpha: Option<f32>) -> Result<(), PatchGraphError> {
    if alpha.is_some_and(|alpha| !alpha.is_finite()) {
        Err(PatchGraphError::InvalidPayload(
            "patch alpha must be finite".into(),
        ))
    } else {
        Ok(())
    }
}

fn validate_constraint(constraint: Option<f32>) -> Result<(), PatchGraphError> {
    if constraint.is_some_and(|constraint| !constraint.is_finite()) {
        Err(PatchGraphError::InvalidPayload(
            "OFT/BOFT constraint must be finite".into(),
        ))
    } else {
        Ok(())
    }
}

fn validate_transform(transform: PatchValueTransform) -> Result<(), PatchGraphError> {
    if transform.scale.is_finite() && transform.bias.is_finite() {
        Ok(())
    } else {
        Err(PatchGraphError::InvalidPayload(
            "patch transform must be finite".into(),
        ))
    }
}

fn validate_patch_compute_dtype(dtype: DType) -> Result<(), PatchGraphError> {
    match dtype {
        DType::F32 | DType::F16 | DType::Bf16 => Ok(()),
        _ => Err(PatchGraphError::InvalidPayload(format!(
            "patch intermediate compute dtype {dtype:?} is unsupported"
        ))),
    }
}

fn validate_configured_patch_compute_dtype(dtype: DType) -> Result<(), PatchGraphError> {
    match dtype {
        DType::F32 | DType::F16 => Ok(()),
        _ => Err(PatchGraphError::InvalidPayload(format!(
            "configured LoRA compute dtype {dtype:?} is unsupported"
        ))),
    }
}

fn validate_full_or_pair(
    name: &'static str,
    full: Option<&PatchTensor>,
    up: Option<&PatchTensor>,
    down: Option<&PatchTensor>,
) -> Result<(), PatchGraphError> {
    match (full, up, down) {
        (Some(full), None, None) => validate_patch_tensor(name, full),
        (None, Some(up), Some(down)) => {
            validate_patch_tensor(name, up)?;
            validate_patch_tensor(name, down)
        }
        _ => Err(PatchGraphError::InvalidPayload(format!(
            "{name} requires exactly a full tensor or an up/down pair"
        ))),
    }
}

fn semantic_ordered_digest(
    base_digest: &str,
    operations: &[SemanticPatchOperation],
) -> Result<String, PatchGraphError> {
    let encoded = serde_json::to_vec(operations)
        .map_err(|error| PatchGraphError::Serialization(error.to_string()))?;
    let mut digest = Sha256::new();
    digest.update(b"zed-sim-patch-graph-semantic-v2\0");
    update_text(&mut digest, base_digest)?;
    update_len(&mut digest, encoded.len())?;
    digest.update(encoded);
    Ok(format!("{:x}", digest.finalize()))
}

fn applied_patch_digest(
    ordered_digest: &str,
    compute_boundary: PatchComputeBoundary,
    touched_dtypes: &BTreeMap<String, (DType, DType)>,
) -> Result<String, PatchGraphError> {
    if touched_dtypes.is_empty() {
        return Ok(ordered_digest.to_owned());
    }
    let mut digest = Sha256::new();
    digest.update(b"zed-sim-patch-applied-compute-v1\0");
    update_text(&mut digest, ordered_digest)?;
    update_text(
        &mut digest,
        match compute_boundary {
            PatchComputeBoundary::Configured(_) => "configured",
            PatchComputeBoundary::WeightDType => "weight_dtype",
        },
    )?;
    update_len(&mut digest, touched_dtypes.len())?;
    for (target_key, (compute_dtype, output_dtype)) in touched_dtypes {
        update_text(&mut digest, target_key)?;
        update_text(&mut digest, patch_compute_dtype_tag(*compute_dtype)?)?;
        update_text(&mut digest, patch_compute_dtype_tag(*output_dtype)?)?;
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn patch_compute_dtype_tag(dtype: DType) -> Result<&'static str, PatchGraphError> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::Bf16 => Ok("bf16"),
        _ => Err(PatchGraphError::InvalidPayload(format!(
            "patch intermediate compute dtype {dtype:?} is unsupported"
        ))),
    }
}

fn ordered_digest(
    base_digest: &str,
    operations: &[PatchOperation],
) -> Result<String, PatchGraphError> {
    let mut digest = Sha256::new();
    digest.update(b"zed-sim-patch-graph-v1\0");
    update_text(&mut digest, base_digest)?;
    update_len(&mut digest, operations.len())?;
    for operation in operations {
        update_text(&mut digest, &operation.identifier)?;
        update_text(&mut digest, patch_kind_tag(operation.kind))?;
        digest.update(operation.scale.to_bits().to_le_bytes());
        update_len(&mut digest, operation.targets.len())?;
        for target in &operation.targets {
            update_text(&mut digest, &target.key)?;
            update_text(&mut digest, patch_application_tag(target.application))?;
            update_len(&mut digest, target.expected_shape.len())?;
            for dimension in &target.expected_shape {
                digest.update(dimension.to_le_bytes());
            }
            update_len(&mut digest, target.values.len())?;
            for value in &target.values {
                digest.update(value.to_bits().to_le_bytes());
            }
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn update_text(digest: &mut Sha256, value: &str) -> Result<(), PatchGraphError> {
    update_len(digest, value.len())?;
    digest.update(value.as_bytes());
    Ok(())
}

fn update_len(digest: &mut Sha256, length: usize) -> Result<(), PatchGraphError> {
    digest.update(
        u64::try_from(length)
            .map_err(|_| PatchGraphError::ShapeOverflow)?
            .to_le_bytes(),
    );
    Ok(())
}

fn patch_kind_tag(kind: PatchKind) -> &'static str {
    match kind {
        PatchKind::ScaleOnly => "scale_only",
        PatchKind::DenseDiff => "dense_diff",
        PatchKind::Set => "set",
        PatchKind::Lora => "lora",
        PatchKind::Loha => "loha",
        PatchKind::Lokr => "lokr",
        PatchKind::Oft => "oft",
        PatchKind::Glora => "glora",
        PatchKind::Boft => "boft",
        PatchKind::Dora => "dora",
        PatchKind::Nested => "nested",
        PatchKind::ModelAsLora => "model_as_lora",
        PatchKind::ControlNet => "control_net",
        PatchKind::Adapter => "adapter",
        PatchKind::Replacement => "replacement",
    }
}

fn patch_application_tag(application: PatchApplication) -> &'static str {
    match application {
        PatchApplication::Add => "add",
        PatchApplication::Replace => "replace",
    }
}

fn validate_text(field: &'static str, value: &str) -> Result<(), PatchGraphError> {
    if value.is_empty() || value.len() > MAX_PATCH_TEXT_BYTES || value.chars().any(char::is_control)
    {
        return Err(PatchGraphError::InvalidText {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_digest(value: &str) -> Result<(), PatchGraphError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PatchGraphError::InvalidDigest(value.to_owned()));
    }
    Ok(())
}

fn validate_identity_digest(
    field: &'static str,
    value: &str,
) -> Result<(), PatchGraphIdentityError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PatchGraphIdentityError::InvalidDigest {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum PatchGraphIdentityError {
    #[error("unsupported patch graph schema version {actual}; expected {expected}")]
    SchemaVersion { expected: u16, actual: u16 },
    #[error("invalid patch graph {field} digest: {value}")]
    InvalidDigest { field: &'static str, value: String },
    #[error("patch graph expects base {expected} but consumer uses {actual}")]
    BaseDigestMismatch { expected: String, actual: String },
}

#[derive(Debug, Error)]
pub enum PatchGraphError {
    #[error("invalid patch graph digest: {0}")]
    InvalidDigest(String),
    #[error("invalid patch graph {field}: {value}")]
    InvalidText { field: &'static str, value: String },
    #[error("patch graph repeats operation {0}")]
    DuplicateOperation(String),
    #[error("patch operation {0} has no targets")]
    EmptyTargets(String),
    #[error("patch operation {operation} repeats target {key}")]
    DuplicateTarget { operation: String, key: String },
    #[error("patch operation {0} has a non-finite scale")]
    NonFiniteScale(String),
    #[error("patch target {0} has a non-finite value")]
    NonFiniteValue(String),
    #[error("patch computation for target {0} produced a non-finite value")]
    NonFiniteResult(String),
    #[error("patch target {0} has an invalid shape")]
    InvalidShape(String),
    #[error("patch target {key} expected {expected} values but received {actual}")]
    ValueCount {
        key: String,
        expected: u64,
        actual: u64,
    },
    #[error("patch shape arithmetic overflowed")]
    ShapeOverflow,
    #[error("patch graph resident byte accounting overflowed")]
    ResidentBytesOverflow,
    #[error("patch operation {operation} kind {kind:?} cannot use {application:?}")]
    KindApplication {
        operation: String,
        kind: PatchKind,
        application: PatchApplication,
    },
    #[error("patch graph expects base {expected} but model uses {actual}")]
    BaseDigestMismatch { expected: String, actual: String },
    #[error("patch graph target is missing: {0}")]
    MissingTarget(String),
    #[error("patch operation has an invalid slice: {0}")]
    InvalidSlice(String),
    #[error("invalid typed patch payload: {0}")]
    InvalidPayload(String),
    #[error("nested patch depth exceeds the supported bound")]
    NestingDepth,
    #[error("canonical tensor operation failed: {0}")]
    CanonicalTensorOperation(String),
    #[error("patch graph serialization failed: {0}")]
    Serialization(String),
    #[error("patch target {key} expected shape {expected:?} but received {actual:?}")]
    ShapeMismatch {
        key: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorOperation(#[from] OperatorIndirectionError),
    #[error(transparent)]
    Cancelled(#[from] sim_types::CancellationError),
    #[error(transparent)]
    ModelFamily(#[from] ModelFamilyError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{
        BackendCapabilityMatrix, BackendWorkspaceLease, CachedAllocationOwner, ConvolutionSpec,
        CpuBackend, CpuWorkspaceAuthority, CustomKernelId, DeviceId, EventFence, IndexSpec,
        ReductionSpec, ResizeSpec, StreamId, UnaryOperation,
        generated_sim_operator_indirection_01::{
            cast_to_with_backend_exact_native, tensor_from_f32_with_backend_exact_native,
            tensor_to_f32_with_backend_exact_native,
        },
    };
    use sim_types::{CancellationToken, DeviceKind};
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const BASE_DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn tensor(shape: &[u64], values: &[f32]) -> PatchTensor {
        PatchTensor::checked(shape.to_vec(), values.to_vec()).expect("valid test tensor")
    }

    fn operation(identifier: &str, shape: &[u64], payload: PatchPayload) -> SemanticPatchOperation {
        SemanticPatchOperation {
            identifier: identifier.to_owned(),
            target_key: "weight".to_owned(),
            expected_shape: shape.to_vec(),
            strength: 1.0,
            strength_model: 1.0,
            slices: Vec::new(),
            transform: PatchValueTransform::default(),
            payload,
        }
    }

    fn mapped(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        shape: &[u64],
        values: &[f32],
    ) -> Result<MappedModelWeights, PatchGraphError> {
        mapped_dtype(backend, context, shape, values, DType::F32)
    }

    fn mapped_dtype(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        shape: &[u64],
        values: &[f32],
        dtype: DType,
    ) -> Result<MappedModelWeights, PatchGraphError> {
        let weight = tensor_from_f32_with_backend_exact_native(
            backend,
            shape,
            values,
            dtype,
            sim_tensor::DeviceId::CPU,
            context,
        )?;
        Ok(MappedModelWeights::from_parts(
            BASE_DIGEST.to_owned(),
            BTreeMap::from([("weight".to_owned(), weight)]),
            Vec::new(),
        ))
    }

    fn values(
        backend: &CpuBackend,
        mapped: &MappedModelWeights,
        context: &ExecutionContext<'_>,
    ) -> Result<Vec<f32>, PatchGraphError> {
        Ok(tensor_to_f32_with_backend_exact_native(
            backend,
            mapped
                .tensors()
                .get("weight")
                .ok_or_else(|| PatchGraphError::MissingTarget("weight".into()))?,
            context,
        )?)
    }

    fn weight_bytes(mapped: &MappedModelWeights) -> Result<Vec<u8>, PatchGraphError> {
        Ok(mapped
            .tensors()
            .get("weight")
            .ok_or_else(|| PatchGraphError::MissingTarget("weight".into()))?
            .contiguous_bytes()?
            .to_vec())
    }

    struct DelegatingBackend<'a> {
        backend: &'a CpuBackend,
        cancellation_on_binary: Option<&'a CancellationToken>,
        reserve_calls: AtomicUsize,
        primitive_calls: AtomicUsize,
    }

    impl CachedAllocationOwner for DelegatingBackend<'_> {
        fn cache_device(&self) -> DeviceId {
            DeviceId::CPU
        }

        fn release_cached_allocations(
            &self,
            cancellation: &CancellationToken,
        ) -> Result<u64, TensorError> {
            self.backend.release_cached_allocations(cancellation)
        }
    }

    impl TensorBackend for DelegatingBackend<'_> {
        fn device(&self) -> DeviceId {
            DeviceId::CPU
        }

        fn capabilities(&self) -> &BackendCapabilityMatrix {
            self.backend.capabilities()
        }

        fn reserve_workspace(
            &self,
            context: &ExecutionContext<'_>,
            requested: u64,
        ) -> Result<BackendWorkspaceLease, TensorError> {
            self.reserve_calls.fetch_add(1, Ordering::AcqRel);
            self.backend.reserve_workspace(context, requested)
        }

        fn upload_f32_payload(
            &self,
            shape: &[u64],
            values: &[f32],
            dtype: DType,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.primitive_calls.fetch_add(1, Ordering::AcqRel);
            self.backend
                .upload_f32_payload(shape, values, dtype, context)
        }

        fn cast_tensor(
            &self,
            input: &Tensor,
            dtype: DType,
            non_blocking: bool,
            copy: bool,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.primitive_calls.fetch_add(1, Ordering::AcqRel);
            self.backend
                .cast_tensor(input, dtype, non_blocking, copy, context)
        }

        fn allocate(
            &self,
            descriptor: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.primitive_calls.fetch_add(1, Ordering::AcqRel);
            self.backend.allocate(descriptor, context)
        }

        fn copy(
            &self,
            source: &Tensor,
            destination: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.primitive_calls.fetch_add(1, Ordering::AcqRel);
            self.backend.copy(source, destination, context)
        }

        fn record_event(&self, context: &ExecutionContext<'_>) -> Result<EventFence, TensorError> {
            self.primitive_calls.fetch_add(1, Ordering::AcqRel);
            self.backend.record_event(context)
        }

        fn wait_event(
            &self,
            event: EventFence,
            context: &ExecutionContext<'_>,
        ) -> Result<(), TensorError> {
            self.primitive_calls.fetch_add(1, Ordering::AcqRel);
            self.backend.wait_event(event, context)
        }

        fn fill(
            &self,
            value: Scalar,
            output: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.primitive_calls.fetch_add(1, Ordering::AcqRel);
            self.backend.fill(value, output, context)
        }

        fn unary(
            &self,
            operation: UnaryOperation,
            input: &Tensor,
            output: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.primitive_calls.fetch_add(1, Ordering::AcqRel);
            self.backend.unary(operation, input, output, context)
        }

        fn binary(
            &self,
            operation: BinaryOperation,
            left: &Tensor,
            right: &Tensor,
            output: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.primitive_calls.fetch_add(1, Ordering::AcqRel);
            let result = self
                .backend
                .binary(operation, left, right, output, context)?;
            if let Some(cancellation) = self.cancellation_on_binary {
                cancellation.cancel();
            }
            Ok(result)
        }

        fn binary_scalar(
            &self,
            operation: BinaryOperation,
            input: &Tensor,
            scalar: Scalar,
            scalar_side: ScalarSide,
            output: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.primitive_calls.fetch_add(1, Ordering::AcqRel);
            let result = self.backend.binary_scalar(
                operation,
                input,
                scalar,
                scalar_side,
                output,
                context,
            )?;
            if let Some(cancellation) = self.cancellation_on_binary {
                cancellation.cancel();
            }
            Ok(result)
        }

        fn reduction(
            &self,
            operation: &ReductionSpec,
            input: &Tensor,
            output: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.primitive_calls.fetch_add(1, Ordering::AcqRel);
            self.backend.reduction(operation, input, output, context)
        }

        fn indexing(
            &self,
            operation: &IndexSpec,
            inputs: &[Tensor],
            output: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.primitive_calls.fetch_add(1, Ordering::AcqRel);
            self.backend.indexing(operation, inputs, output, context)
        }

        fn resize(
            &self,
            operation: ResizeSpec,
            input: &Tensor,
            output: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.primitive_calls.fetch_add(1, Ordering::AcqRel);
            self.backend.resize(operation, input, output, context)
        }

        fn convolution(
            &self,
            operation: &ConvolutionSpec,
            inputs: &[Tensor],
            output: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.primitive_calls.fetch_add(1, Ordering::AcqRel);
            self.backend.convolution(operation, inputs, output, context)
        }

        fn linear_algebra(
            &self,
            operation: LinearAlgebraOperation,
            inputs: &[Tensor],
            output: TensorDescriptor,
            context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.primitive_calls.fetch_add(1, Ordering::AcqRel);
            self.backend
                .linear_algebra(operation, inputs, output, context)
        }

        fn custom_kernel(
            &self,
            kernel: &CustomKernelId,
            inputs: &[Tensor],
            outputs: &[TensorDescriptor],
            context: &ExecutionContext<'_>,
        ) -> Result<(Vec<Tensor>, EventFence), TensorError> {
            self.primitive_calls.fetch_add(1, Ordering::AcqRel);
            self.backend.custom_kernel(kernel, inputs, outputs, context)
        }
    }

    struct IncompleteBackend {
        capabilities: BackendCapabilityMatrix,
        unexpected_calls: AtomicUsize,
    }

    impl IncompleteBackend {
        fn new() -> Result<Self, TensorError> {
            Ok(Self {
                capabilities: BackendCapabilityMatrix::new(
                    DeviceId::new(DeviceKind::Metal, 0),
                    Vec::new(),
                    Vec::new(),
                )?,
                unexpected_calls: AtomicUsize::new(0),
            })
        }

        fn unexpected<T>(&self, operation: &'static str) -> Result<T, TensorError> {
            self.unexpected_calls.fetch_add(1, Ordering::AcqRel);
            Err(TensorError::UnsupportedCapability {
                operation: operation.to_owned(),
                device: self.device(),
                reason: "incomplete test backend does not implement primitives".to_owned(),
            })
        }
    }

    impl CachedAllocationOwner for IncompleteBackend {
        fn cache_device(&self) -> DeviceId {
            self.device()
        }

        fn release_cached_allocations(
            &self,
            _cancellation: &CancellationToken,
        ) -> Result<u64, TensorError> {
            Ok(0)
        }
    }

    impl TensorBackend for IncompleteBackend {
        fn device(&self) -> DeviceId {
            self.capabilities.device()
        }

        fn capabilities(&self) -> &BackendCapabilityMatrix {
            &self.capabilities
        }

        fn reserve_workspace(
            &self,
            _context: &ExecutionContext<'_>,
            _requested: u64,
        ) -> Result<BackendWorkspaceLease, TensorError> {
            self.unexpected("incomplete.reserve_workspace")
        }

        fn allocate(
            &self,
            _descriptor: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unexpected("incomplete.allocate")
        }

        fn copy(
            &self,
            _source: &Tensor,
            _destination: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unexpected("incomplete.copy")
        }

        fn record_event(&self, _context: &ExecutionContext<'_>) -> Result<EventFence, TensorError> {
            self.unexpected("incomplete.record_event")
        }

        fn wait_event(
            &self,
            _event: EventFence,
            _context: &ExecutionContext<'_>,
        ) -> Result<(), TensorError> {
            self.unexpected("incomplete.wait_event")
        }

        fn fill(
            &self,
            _value: Scalar,
            _output: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unexpected("incomplete.fill")
        }

        fn unary(
            &self,
            _operation: UnaryOperation,
            _input: &Tensor,
            _output: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unexpected("incomplete.unary")
        }

        fn binary(
            &self,
            _operation: BinaryOperation,
            _left: &Tensor,
            _right: &Tensor,
            _output: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unexpected("incomplete.binary")
        }

        fn binary_scalar(
            &self,
            _operation: BinaryOperation,
            _input: &Tensor,
            _scalar: Scalar,
            _scalar_side: ScalarSide,
            _output: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unexpected("incomplete.binary_scalar")
        }

        fn reduction(
            &self,
            _operation: &ReductionSpec,
            _input: &Tensor,
            _output: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unexpected("incomplete.reduction")
        }

        fn indexing(
            &self,
            _operation: &IndexSpec,
            _inputs: &[Tensor],
            _output: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unexpected("incomplete.indexing")
        }

        fn resize(
            &self,
            _operation: ResizeSpec,
            _input: &Tensor,
            _output: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unexpected("incomplete.resize")
        }

        fn convolution(
            &self,
            _operation: &ConvolutionSpec,
            _inputs: &[Tensor],
            _output: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unexpected("incomplete.convolution")
        }

        fn linear_algebra(
            &self,
            _operation: LinearAlgebraOperation,
            _inputs: &[Tensor],
            _output: TensorDescriptor,
            _context: &ExecutionContext<'_>,
        ) -> Result<(Tensor, EventFence), TensorError> {
            self.unexpected("incomplete.linear_algebra")
        }

        fn custom_kernel(
            &self,
            _kernel: &CustomKernelId,
            _inputs: &[Tensor],
            _outputs: &[TensorDescriptor],
            _context: &ExecutionContext<'_>,
        ) -> Result<(Vec<Tensor>, EventFence), TensorError> {
            self.unexpected("incomplete.custom_kernel")
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "value {index}: expected {expected}, got {actual}"
            );
        }
    }

    #[derive(Clone)]
    enum JsonStep {
        Key(String),
        Index(usize),
    }

    fn collect_digest_leaf_paths(
        value: &serde_json::Value,
        path: &mut Vec<JsonStep>,
        output: &mut Vec<Vec<JsonStep>>,
    ) {
        match value {
            serde_json::Value::Object(fields) => {
                for (key, value) in fields {
                    if key == "family" {
                        continue;
                    }
                    path.push(JsonStep::Key(key.clone()));
                    collect_digest_leaf_paths(value, path, output);
                    path.pop();
                }
            }
            serde_json::Value::Array(values) => {
                for (index, value) in values.iter().enumerate() {
                    path.push(JsonStep::Index(index));
                    collect_digest_leaf_paths(value, path, output);
                    path.pop();
                }
            }
            serde_json::Value::Null => {}
            _ => output.push(path.clone()),
        }
    }

    fn mutate_digest_leaf(value: &mut serde_json::Value, path: &[JsonStep]) {
        let Some((step, remaining)) = path.split_first() else {
            match value {
                serde_json::Value::Bool(value) => *value = !*value,
                serde_json::Value::Number(value) if value.is_u64() => {
                    if let Some(number) = value.as_u64() {
                        *value = serde_json::Number::from(number.saturating_add(1));
                    }
                }
                serde_json::Value::Number(value) if value.is_i64() => {
                    if let Some(number) = value.as_i64() {
                        *value = serde_json::Number::from(number.saturating_add(1));
                    }
                }
                serde_json::Value::Number(value) => {
                    if let Some(number) = value
                        .as_f64()
                        .and_then(|number| serde_json::Number::from_f64(number + 0.25))
                    {
                        *value = number;
                    }
                }
                serde_json::Value::String(value) => value.push_str("-changed"),
                _ => {}
            }
            return;
        };
        match step {
            JsonStep::Key(key) => mutate_digest_leaf(&mut value[key], remaining),
            JsonStep::Index(index) => mutate_digest_leaf(&mut value[*index], remaining),
        }
    }

    fn assert_every_serialized_leaf_changes_digest(
        operation: &SemanticPatchOperation,
    ) -> Result<(), PatchGraphError> {
        let original = semantic_ordered_digest(BASE_DIGEST, std::slice::from_ref(operation))?;
        let encoded = serde_json::to_value(operation)
            .map_err(|error| PatchGraphError::Serialization(error.to_string()))?;
        let mut paths = Vec::new();
        collect_digest_leaf_paths(&encoded, &mut Vec::new(), &mut paths);
        for path in paths {
            let mut changed = encoded.clone();
            mutate_digest_leaf(&mut changed, &path);
            let changed: SemanticPatchOperation = serde_json::from_value(changed)
                .map_err(|error| PatchGraphError::Serialization(error.to_string()))?;
            assert_ne!(
                semantic_ordered_digest(BASE_DIGEST, &[changed])?,
                original,
                "serialized semantic patch leaf did not affect the digest"
            );
        }
        Ok(())
    }

    struct CatalogPatchCase {
        name: &'static str,
        base_shape: Vec<u64>,
        base_values: Vec<f32>,
        payload: PatchPayload,
        strength: f32,
        expected_shape: Vec<u64>,
        expected_values: Vec<f32>,
    }

    fn catalog_case(
        name: &'static str,
        base_shape: &[u64],
        base_values: &[f32],
        payload: PatchPayload,
        expected_shape: &[u64],
        expected_values: &[f32],
    ) -> CatalogPatchCase {
        CatalogPatchCase {
            name,
            base_shape: base_shape.to_vec(),
            base_values: base_values.to_vec(),
            payload,
            strength: 1.0,
            expected_shape: expected_shape.to_vec(),
            expected_values: expected_values.to_vec(),
        }
    }

    fn execute_catalog_patch_case(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        case: CatalogPatchCase,
    ) -> Result<String, PatchGraphError> {
        let source = mapped(backend, context, &case.base_shape, &case.base_values)?;
        let source_bytes = weight_bytes(&source)?;
        let source_identity = source.cache_identity().to_owned();
        let mut semantic = operation(case.name, &case.base_shape, case.payload);
        semantic.strength = case.strength;
        let graph = PatchGraph::checked_semantic(BASE_DIGEST, vec![semantic])?;
        let patched = graph.apply_with_compute_boundary(
            backend,
            &source,
            PatchComputeBoundary::configured(DType::F32)?,
            context,
        )?;
        let output = patched
            .tensors()
            .get("weight")
            .ok_or_else(|| PatchGraphError::MissingTarget("weight".into()))?;
        assert_eq!(output.descriptor().shape(), case.expected_shape);
        assert_eq!(output.descriptor().device(), DeviceId::CPU);
        assert_close(&values(backend, &patched, context)?, &case.expected_values);
        assert_eq!(weight_bytes(&source)?, source_bytes);
        assert_eq!(source.cache_identity(), source_identity);
        assert_ne!(patched.cache_identity(), source.cache_identity());
        let repeated = graph.apply_with_compute_boundary(
            backend,
            &source,
            PatchComputeBoundary::configured(DType::F32)?,
            context,
        )?;
        assert_eq!(weight_bytes(&patched)?, weight_bytes(&repeated)?);
        assert_eq!(patched.cache_identity(), repeated.cache_identity());
        assert_eq!(context.scratch.in_use_bytes(), 0);
        Ok(case.name.to_owned())
    }

    fn execute_catalog_patch_contract(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        source_path: &str,
        source_symbol: &str,
    ) -> Result<Vec<String>, PatchGraphError> {
        let identity = || tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let tucker = |convolution: bool| PatchPayload::Lora {
            up: tensor(&[2, 1], &[1.0, 2.0]),
            down: tensor(&[1, 2], &[3.0, 4.0]),
            mid: Some(tensor(
                if convolution { &[1, 1, 1, 1] } else { &[1, 1] },
                &[1.0],
            )),
            alpha: Some(1.0),
            dora_scale: None,
            reshape: Some(if convolution {
                vec![2, 2, 1, 1]
            } else {
                vec![2, 2]
            }),
        };
        let cases = match (source_path, source_symbol) {
            ("projects/comfy/ComfyUI/sim/weight_adapter/base.py", "weight_decompose") => {
                let dora = |shape: &[u64]| PatchPayload::Dora {
                    difference: tensor(shape, &[1.0; 4]),
                    scale: tensor(
                        if shape.len() == 2 {
                            &[2]
                        } else {
                            &[2, 1, 1, 1]
                        },
                        &[2.0, 3.0],
                    ),
                    alpha: 0.5,
                };
                let mut linear = catalog_case(
                    "linear-weight-decompose",
                    &[2, 2],
                    &[1.0, 0.0, 0.0, 1.0],
                    dora(&[2, 2]),
                    &[2, 2],
                    &[1.5, 0.375, 0.25, 1.875],
                );
                linear.strength = 0.25;
                let mut convolution = catalog_case(
                    "convolution-weight-decompose",
                    &[2, 2, 1, 1],
                    &[1.0, 0.0, 0.0, 1.0],
                    dora(&[2, 2, 1, 1]),
                    &[2, 2, 1, 1],
                    &[1.5, 0.25, 0.375, 1.875],
                );
                convolution.strength = 0.25;
                vec![linear, convolution]
            }
            ("projects/comfy/ComfyUI/sim/weight_adapter/base.py", "pad_tensor_to_shape")
            | ("projects/comfy/ComfyUI/sim/lora.py", "pad_tensor_to_shape")
            | ("projects/comfy/ComfyUI/sim/lora.py", "calculate_shape") => vec![
                catalog_case(
                    "linear-pad-and-shape",
                    &[1, 1],
                    &[2.0],
                    PatchPayload::DenseDiff {
                        tensor: tensor(&[2, 2], &[1.0; 4]),
                        pad_weight: true,
                    },
                    &[2, 2],
                    &[3.0, 1.0, 1.0, 1.0],
                ),
                catalog_case(
                    "convolution-pad-and-shape",
                    &[1, 1, 1, 1],
                    &[2.0],
                    PatchPayload::DenseDiff {
                        tensor: tensor(&[2, 2, 1, 1], &[1.0; 4]),
                        pad_weight: true,
                    },
                    &[2, 2, 1, 1],
                    &[3.0, 1.0, 1.0, 1.0],
                ),
            ],
            (
                "projects/comfy/ComfyUI/sim/weight_adapter/base.py",
                "tucker_weight_from_conv" | "tucker_weight",
            ) => vec![
                catalog_case(
                    "linear-tucker",
                    &[2, 2],
                    &[0.0; 4],
                    tucker(false),
                    &[2, 2],
                    &[3.0, 4.0, 6.0, 8.0],
                ),
                catalog_case(
                    "convolution-tucker",
                    &[2, 2, 1, 1],
                    &[0.0; 4],
                    tucker(true),
                    &[2, 2, 1, 1],
                    &[3.0, 4.0, 6.0, 8.0],
                ),
            ],
            ("projects/comfy/ComfyUI/sim/weight_adapter/base.py", "factorization") => {
                assert_eq!(factorize_patch_dimension(360, None)?, (18, 20));
                assert_eq!(factorize_patch_dimension(360, Some(16))?, (15, 24));
                assert!(factorize_patch_dimension(0, None).is_err());
                assert!(factorize_patch_dimension(4, Some(0)).is_err());
                let (left, right) = factorize_patch_dimension(4, None)?;
                let identity_values = vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ];
                let payload = |convolution: bool| {
                    let second_shape = if convolution {
                        vec![right, right, 1, 1]
                    } else {
                        vec![right, right]
                    };
                    PatchPayload::Lokr {
                        first: Some(tensor(&[left, left], &[1.0, 0.0, 0.0, 1.0])),
                        second: Some(tensor(&second_shape, &[1.0, 0.0, 0.0, 1.0])),
                        first_up: None,
                        first_down: None,
                        second_up: None,
                        second_down: None,
                        second_tucker: None,
                        alpha: None,
                        dora_scale: None,
                    }
                };
                vec![
                    catalog_case(
                        "linear-factorization-lokr",
                        &[4, 4],
                        &[0.0; 16],
                        payload(false),
                        &[4, 4],
                        &identity_values,
                    ),
                    catalog_case(
                        "convolution-factorization-lokr",
                        &[4, 4, 1, 1],
                        &[0.0; 16],
                        payload(true),
                        &[4, 4, 1, 1],
                        &identity_values,
                    ),
                ]
            }
            ("projects/comfy/ComfyUI/sim/lora.py", "calculate_weight") => {
                let nested = |shape: &[u64]| PatchPayload::Nested {
                    base: tensor(shape, &[1.0; 4]),
                    base_transform: PatchValueTransform::default(),
                    patches: vec![NestedPatch {
                        strength: 1.0,
                        strength_model: 1.0,
                        transform: PatchValueTransform::default(),
                        payload: PatchPayload::DenseDiff {
                            tensor: tensor(shape, &[1.0; 4]),
                            pad_weight: false,
                        },
                    }],
                };
                vec![
                    catalog_case(
                        "linear-ordered-calculate-weight",
                        &[2, 2],
                        &[0.0; 4],
                        nested(&[2, 2]),
                        &[2, 2],
                        &[2.0; 4],
                    ),
                    catalog_case(
                        "convolution-ordered-calculate-weight",
                        &[2, 2, 1, 1],
                        &[0.0; 4],
                        nested(&[2, 2, 1, 1]),
                        &[2, 2, 1, 1],
                        &[2.0; 4],
                    ),
                ]
            }
            ("projects/comfy/ComfyUI/sim/weight_adapter/boft.py", "calculate_weight") => {
                let boft = || PatchPayload::Boft {
                    blocks: tensor(&[1, 1, 2, 2], &[0.0, 0.25, -0.25, 0.0]),
                    rescale: Some(tensor(&[1], &[2.0])),
                    constraint: None,
                    dora_scale: None,
                };
                vec![
                    catalog_case(
                        "linear-boft",
                        &[2, 2],
                        &[1.0, 0.0, 0.0, 1.0],
                        boft(),
                        &[2, 2],
                        &[1.2, 1.6, -1.6, 1.2],
                    ),
                    catalog_case(
                        "convolution-boft",
                        &[2, 2, 1, 1],
                        &[1.0, 0.0, 0.0, 1.0],
                        boft(),
                        &[2, 2, 1, 1],
                        &[1.2, 1.6, -1.6, 1.2],
                    ),
                ]
            }
            ("projects/comfy/ComfyUI/sim/weight_adapter/glora.py", "calculate_weight") => vec![
                catalog_case(
                    "linear-glora",
                    &[2, 2],
                    &[1.0, 0.0, 0.0, 1.0],
                    PatchPayload::Glora {
                        first_a: tensor(&[1, 2], &[1.0, 0.0]),
                        second_a: tensor(&[2, 1], &[1.0, 0.0]),
                        first_b: tensor(&[1, 2], &[0.0, 0.0]),
                        second_b: tensor(&[2, 1], &[0.0, 0.0]),
                        alpha: None,
                        dora_scale: None,
                    },
                    &[2, 2],
                    &[2.0, 0.0, 0.0, 1.0],
                ),
                catalog_case(
                    "convolution-glora",
                    &[1, 2, 1, 1],
                    &[1.0, 2.0],
                    PatchPayload::Glora {
                        first_a: tensor(&[2, 1], &[1.0, 1.0]),
                        second_a: tensor(&[1, 2], &[2.0, 3.0]),
                        first_b: tensor(&[1, 1], &[0.0]),
                        second_b: tensor(&[1, 2], &[0.0, 0.0]),
                        alpha: None,
                        dora_scale: None,
                    },
                    &[1, 2, 1, 1],
                    &[7.0, 11.0],
                ),
            ],
            ("projects/comfy/ComfyUI/sim/weight_adapter/loha.py", "calculate_weight") => vec![
                catalog_case(
                    "linear-loha",
                    &[2, 2],
                    &[0.0; 4],
                    PatchPayload::Loha {
                        first_up: identity(),
                        first_down: identity(),
                        second_up: identity(),
                        second_down: identity(),
                        first_tucker: None,
                        second_tucker: None,
                        alpha: Some(2.0),
                        dora_scale: None,
                    },
                    &[2, 2],
                    &[1.0, 0.0, 0.0, 1.0],
                ),
                catalog_case(
                    "convolution-loha-tucker",
                    &[2, 2, 1, 1],
                    &[0.0; 4],
                    PatchPayload::Loha {
                        first_up: tensor(&[2, 1], &[1.0, 2.0]),
                        first_down: tensor(&[1, 2], &[3.0, 4.0]),
                        second_up: tensor(&[2, 1], &[1.0, 2.0]),
                        second_down: tensor(&[1, 2], &[3.0, 4.0]),
                        first_tucker: Some(tensor(&[1, 1, 1, 1], &[1.0])),
                        second_tucker: Some(tensor(&[1, 1, 1, 1], &[1.0])),
                        alpha: Some(1.0),
                        dora_scale: None,
                    },
                    &[2, 2, 1, 1],
                    &[9.0, 16.0, 36.0, 64.0],
                ),
            ],
            ("projects/comfy/ComfyUI/sim/weight_adapter/lokr.py", "calculate_weight") => vec![
                catalog_case(
                    "linear-lokr-decomposed",
                    &[2, 2],
                    &[0.0; 4],
                    PatchPayload::Lokr {
                        first: None,
                        second: None,
                        first_up: Some(tensor(&[1, 2], &[1.0, 0.0])),
                        first_down: Some(tensor(&[2, 1], &[1.0, 0.0])),
                        second_up: Some(tensor(&[2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])),
                        second_down: Some(tensor(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0])),
                        second_tucker: None,
                        alpha: Some(6.0),
                        dora_scale: None,
                    },
                    &[2, 2],
                    &[2.0, 0.0, 0.0, 2.0],
                ),
                catalog_case(
                    "convolution-lokr-tucker",
                    &[2, 2, 1, 1],
                    &[0.0; 4],
                    PatchPayload::Lokr {
                        first: Some(tensor(&[1, 1], &[1.0])),
                        second: None,
                        first_up: None,
                        first_down: None,
                        second_up: Some(tensor(&[2, 1], &[1.0, 2.0])),
                        second_down: Some(tensor(&[1, 2], &[3.0, 4.0])),
                        second_tucker: Some(tensor(&[1, 1, 1, 1], &[1.0])),
                        alpha: None,
                        dora_scale: None,
                    },
                    &[2, 2, 1, 1],
                    &[3.0, 4.0, 6.0, 8.0],
                ),
            ],
            ("projects/comfy/ComfyUI/sim/weight_adapter/lora.py", "calculate_weight") => vec![
                catalog_case(
                    "linear-lora",
                    &[2, 2],
                    &[0.0; 4],
                    PatchPayload::Lora {
                        up: identity(),
                        down: identity(),
                        mid: None,
                        alpha: Some(2.0),
                        dora_scale: None,
                        reshape: None,
                    },
                    &[2, 2],
                    &[1.0, 0.0, 0.0, 1.0],
                ),
                catalog_case(
                    "convolution-lora-tucker",
                    &[2, 2, 1, 1],
                    &[0.0; 4],
                    tucker(true),
                    &[2, 2, 1, 1],
                    &[3.0, 4.0, 6.0, 8.0],
                ),
            ],
            ("projects/comfy/ComfyUI/sim/weight_adapter/oft.py", "calculate_weight") => {
                let oft = || PatchPayload::Oft {
                    blocks: tensor(&[1, 2, 2], &[0.0, 0.25, -0.25, 0.0]),
                    rescale: None,
                    constraint: None,
                    dora_scale: None,
                };
                vec![
                    catalog_case(
                        "linear-oft",
                        &[2, 2],
                        &[1.0, 0.0, 0.0, 1.0],
                        oft(),
                        &[2, 2],
                        &[0.6, -0.8, 0.8, 0.6],
                    ),
                    catalog_case(
                        "convolution-oft",
                        &[2, 2, 1, 1],
                        &[1.0, 0.0, 0.0, 1.0],
                        oft(),
                        &[2, 2, 1, 1],
                        &[0.6, -0.8, 0.8, 0.6],
                    ),
                ]
            }
            _ => {
                return Err(PatchGraphError::InvalidPayload(format!(
                    "unaccounted PatchGraph catalog contract {source_path}::{source_symbol}"
                )));
            }
        };
        cases
            .into_iter()
            .map(|case| execute_catalog_patch_case(backend, context, case))
            .collect()
    }

    fn python_symbol_sha256(source: &[u8], symbol: &str) -> Result<String, PatchGraphError> {
        let source = std::str::from_utf8(source)
            .map_err(|error| PatchGraphError::Serialization(error.to_string()))?;
        let lines = source.split_inclusive('\n').collect::<Vec<_>>();
        let synchronous_signature = format!("def {symbol}(");
        let asynchronous_signature = format!("async def {symbol}(");
        let matches = lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| {
                let trimmed = line.trim_start_matches([' ', '\t']);
                (trimmed.starts_with(&synchronous_signature)
                    || trimmed.starts_with(&asynchronous_signature))
                .then_some(index)
            })
            .collect::<Vec<_>>();
        let [start] = matches.as_slice() else {
            return Err(PatchGraphError::Serialization(format!(
                "expected exactly one Python definition for {symbol}, found {}",
                matches.len()
            )));
        };
        let indentation = lines[*start].len() - lines[*start].trim_start_matches([' ', '\t']).len();
        let mut header_complete = lines[*start].trim_end().ends_with(':');
        let mut body_seen = false;
        let mut end = *start + 1;
        while let Some(line) = lines.get(end) {
            let trimmed = line.trim_start_matches([' ', '\t']);
            let trimmed_content = trimmed.trim_end_matches(['\r', '\n']);
            if trimmed_content.is_empty() || trimmed_content.starts_with('#') {
                end += 1;
                continue;
            }
            let line_indentation = line.len() - trimmed.len();
            if !header_complete {
                header_complete = line_indentation == indentation && trimmed_content.ends_with(':');
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
            return Err(PatchGraphError::Serialization(format!(
                "Python definition {symbol} has no indented body"
            )));
        }
        while end > *start + 1 {
            let content = lines[end - 1].trim();
            if content.is_empty() || content.starts_with('#') {
                end -= 1;
            } else {
                break;
            }
        }
        let mut digest = Sha256::new();
        for line in &lines[*start..end] {
            digest.update(line.as_bytes());
        }
        Ok(format!("{:x}", digest.finalize()))
    }

    #[test]
    fn val_patch_001_catalog_manifest_is_exact_digest_bound_and_executable()
    -> Result<(), PatchGraphError> {
        const TASK: &str = "comfy-parity-patch-graph-semantic-foundation";
        const EXPECTED: [(&str, &str, &str, &str); 14] = [
            (
                "conditioning-patch-payload-base-weight-decompose-fc84ed30",
                "134",
                "484f0d83a96e700f80c793e4bcc6a897d633233e51a59191d91da5e59da345c7",
                "weight_decompose",
            ),
            (
                "conditioning-patch-payload-base-pad-tensor-to-shape-e0f4f771",
                "135",
                "484f0d83a96e700f80c793e4bcc6a897d633233e51a59191d91da5e59da345c7",
                "pad_tensor_to_shape",
            ),
            (
                "conditioning-patch-payload-base-tucker-weight-from-conv-d8ae2891",
                "136",
                "484f0d83a96e700f80c793e4bcc6a897d633233e51a59191d91da5e59da345c7",
                "tucker_weight_from_conv",
            ),
            (
                "conditioning-patch-payload-base-tucker-weight-909e1055",
                "137",
                "484f0d83a96e700f80c793e4bcc6a897d633233e51a59191d91da5e59da345c7",
                "tucker_weight",
            ),
            (
                "conditioning-patch-payload-base-factorization-331c28b1",
                "138",
                "484f0d83a96e700f80c793e4bcc6a897d633233e51a59191d91da5e59da345c7",
                "factorization",
            ),
            (
                "conditioning-patch-semantics-lora-pad-tensor-to-shape-e0f4f771",
                "147",
                "8f75c95ddc8ab0144919fe5277c4e6b4fa4f4f45aa64aa3de5d2d3b1b4a927d8",
                "pad_tensor_to_shape",
            ),
            (
                "conditioning-patch-semantics-lora-calculate-shape-d168b9d7",
                "148",
                "8f75c95ddc8ab0144919fe5277c4e6b4fa4f4f45aa64aa3de5d2d3b1b4a927d8",
                "calculate_shape",
            ),
            (
                "conditioning-patch-semantics-lora-calculate-weight-5a305deb",
                "149",
                "8f75c95ddc8ab0144919fe5277c4e6b4fa4f4f45aa64aa3de5d2d3b1b4a927d8",
                "calculate_weight",
            ),
            (
                "conditioning-patch-family-equation-boft-calculate-weight-5a305deb",
                "158",
                "2850e0b4c2295cd87445415e287061fa3bfd69e88bd0aeb3eb16064864bd078d",
                "calculate_weight",
            ),
            (
                "conditioning-patch-family-equation-glora-calculate-weight-5a305deb",
                "160",
                "31cdd03f5b0beaa0df055512560128930f4f26b219ba57602d21abb086425b09",
                "calculate_weight",
            ),
            (
                "conditioning-patch-family-equation-loha-calculate-weight-5a305deb",
                "163",
                "579ca1e33e0d244e0d7eedd30fb727913341f8e7bfbd74b51221f567612286d5",
                "calculate_weight",
            ),
            (
                "conditioning-patch-family-equation-lokr-calculate-weight-5a305deb",
                "166",
                "b4763cc32215a47e4d906cfa6cbad9cf893f6a1329ada2225fda81fd99fcfeb4",
                "calculate_weight",
            ),
            (
                "conditioning-patch-family-equation-lora-calculate-weight-5a305deb",
                "169",
                "e506062b4eb189be4c36f88270e0fd4dcce038c79a98f7163604ed6b44efe4b5",
                "calculate_weight",
            ),
            (
                "conditioning-patch-family-equation-oft-calculate-weight-5a305deb",
                "172",
                "88be3c32f610478bc6900a10009eaadf6fe2af973ed4861731e4f21e1afacf89",
                "calculate_weight",
            ),
        ];
        let expected = EXPECTED
            .into_iter()
            .map(|entry| (entry.0, entry))
            .collect::<BTreeMap<_, _>>();
        let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let catalog_path = repository
            .join(".agents/specs/comfy-parity/catalogs/backend-conditioning-contracts.csv");
        let catalog = std::fs::read_to_string(&catalog_path)
            .map_err(|error| PatchGraphError::Serialization(error.to_string()))?;
        let mut seen = BTreeSet::new();
        let mut contracts = Vec::new();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        for line in catalog.lines().skip(1) {
            let columns = line.split(',').collect::<Vec<_>>();
            if columns.get(8).copied() != Some(TASK) {
                continue;
            }
            assert_eq!(columns.len(), 15, "malformed PatchGraph catalog row");
            let contract_id = columns[0];
            let expected_row = expected.get(contract_id).ok_or_else(|| {
                PatchGraphError::InvalidPayload(format!(
                    "unexpected PatchGraph catalog row {contract_id}"
                ))
            })?;
            assert!(seen.insert(contract_id));
            assert_eq!(columns[4], expected_row.1);
            assert_eq!(columns[5], expected_row.2);
            assert_eq!(columns[3], expected_row.3);
            assert!(matches!(
                columns[1],
                "patch_payload" | "patch_semantics" | "patch_family_equation"
            ));
            assert_eq!(columns[7], "sim_model::patch_graph");
            assert_eq!(columns[9], "sim_model::patch_graph::tests");
            assert_eq!(columns[10], "native_rust");
            assert_eq!(columns[14], "VAL-PATCH-001");
            for digest in [columns[5], columns[6]] {
                assert_eq!(digest.len(), 64);
                assert!(
                    digest
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                );
            }
            let source = std::fs::read(repository.join(columns[2]))
                .map_err(|error| PatchGraphError::Serialization(error.to_string()))?;
            assert_eq!(format!("{:x}", Sha256::digest(&source)), columns[5]);
            assert_eq!(python_symbol_sha256(&source, columns[3])?, columns[6]);
            let case_ids =
                execute_catalog_patch_contract(&backend, &context, columns[2], columns[3])?
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
        assert_eq!(seen.len(), EXPECTED.len());
        assert_eq!(seen, expected.keys().copied().collect());
        let implementation_path = "crates/sim_model/src/patch_graph.rs";
        let implementation = std::fs::read(repository.join(implementation_path))
            .map_err(|error| PatchGraphError::Serialization(error.to_string()))?;
        let implementation_sha256 = format!("{:x}", Sha256::digest(implementation));
        const TASK_IMPLEMENTATION_PATHS: [&str; 11] = [
            "crates/sim_model/src/sim_model.rs",
            "crates/sim_model/src/clip.rs",
            "crates/sim_model/src/model_family.rs",
            "crates/sim_model/src/patch_graph.rs",
            "crates/sim_model/src/vae.rs",
            "crates/sim_model/tests/model_family_foundation.rs",
            "crates/sim_tensor/src/cpu_backend.rs",
            "crates/sim_tensor/src/operation.rs",
            "crates/sim_test_support/tests/patch_compute_boundary.rs",
            "crates/sim_worker/src/memory_modes.rs",
            "crates/sim_worker/tests/memory_conformance.rs",
        ];
        let implementations = TASK_IMPLEMENTATION_PATHS
            .into_iter()
            .map(|path| {
                let bytes = std::fs::read(repository.join(path))
                    .map_err(|error| PatchGraphError::Serialization(error.to_string()))?;
                Ok(serde_json::json!({
                    "path": path,
                    "sha256": format!("{:x}", Sha256::digest(bytes)),
                }))
            })
            .collect::<Result<Vec<_>, PatchGraphError>>()?;
        let task_results = BTreeMap::from([(
            TASK,
            serde_json::json!({
                "status": "passed",
                "passed": contracts.len(),
                "failed": 0,
                "skipped": 0,
                "implementations": implementations,
            }),
        )]);
        let artifact = serde_json::json!({
            "schema_version": 1,
            "validation_id": "VAL-PATCH-001",
            "overall_status": "partial",
            "remaining_validations": [
                "ordering_and_dtype_boundary",
                "device_matrix",
                "cancellation_and_oom_rollback",
                "workspace_convergence",
                "authoritative_ownership",
            ],
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
        std::fs::create_dir_all(&artifact_directory)
            .map_err(|error| PatchGraphError::Serialization(error.to_string()))?;
        std::fs::write(
            artifact_directory.join("val-patch-001.json"),
            serde_json::to_vec_pretty(&artifact)
                .map_err(|error| PatchGraphError::Serialization(error.to_string()))?,
        )
        .map_err(|error| PatchGraphError::Serialization(error.to_string()))?;
        Ok(())
    }

    #[test]
    fn legacy_boundary_maps_totally_into_the_single_semantic_execution_owner()
    -> Result<(), PatchGraphError> {
        let operations = vec![
            PatchOperation {
                identifier: "legacy-add".to_owned(),
                kind: PatchKind::Adapter,
                scale: 0.5,
                targets: vec![PatchTarget {
                    key: "weight".to_owned(),
                    expected_shape: vec![2],
                    values: vec![2.0, 4.0],
                    application: PatchApplication::Add,
                }],
            },
            PatchOperation {
                identifier: "legacy-replace".to_owned(),
                kind: PatchKind::Replacement,
                scale: 2.0,
                targets: vec![PatchTarget {
                    key: "weight".to_owned(),
                    expected_shape: vec![2],
                    values: vec![3.0, 5.0],
                    application: PatchApplication::Replace,
                }],
            },
        ];
        let graph = PatchGraph::checked(BASE_DIGEST, operations.clone())?;
        assert_eq!(graph.semantic_operations().len(), 2);
        assert_eq!(
            graph.identity().ordered_digest,
            ordered_digest(BASE_DIGEST, &operations)?
        );
        for (index, semantic) in graph.semantic_operations().iter().enumerate() {
            assert_eq!(semantic.identifier, format!("legacy:{index}:0"));
            assert!(matches!(semantic.payload, PatchPayload::DenseDiff { .. }));
        }
        assert_eq!(graph.semantic_operations()[0].strength_model, 1.0);
        assert_eq!(graph.semantic_operations()[1].strength_model, 0.0);

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(256 * 1024)?,
            &cancellation,
        );
        let source = mapped(&backend, &context, &[2], &[1.0, 2.0])?;
        let source_bytes = weight_bytes(&source)?;
        let patched = graph.apply(&backend, &source, &context)?;
        assert_eq!(values(&backend, &patched, &context)?, [6.0, 10.0]);
        assert_eq!(weight_bytes(&source)?, source_bytes);
        Ok(())
    }

    #[test]
    fn production_patch_graph_has_one_validation_application_and_commit_path() {
        let source = include_str!("patch_graph.rs");
        let production = source
            .split_once("#[cfg(test)]")
            .map_or(source, |(production, _)| production);
        assert!(!production.contains("apply_target_with_canonical_tensor_operations"));
        assert!(!production.contains("patch_target_map"));
        assert!(!production.contains(
            "operations: Vec<PatchOperation>,\n    semantic_operations: Vec<SemanticPatchOperation>"
        ));
        assert_eq!(production.matches("staged.insert(").count(), 2);
        assert_eq!(production.matches(".with_patch_graph_identity(").count(), 1);
        assert_eq!(
            production.matches("fn apply_semantic_operation(").count(),
            1
        );
        let application_start = production
            .find("fn apply_semantic_operation(")
            .expect("semantic application owner");
        let application_end = production[application_start..]
            .find("\nfn apply_payload_with_original(")
            .map(|offset| application_start + offset)
            .expect("semantic application boundary");
        assert!(!production[application_start..application_end].contains("backend_cast_tensor("));
        assert_eq!(
            production
                .matches("backend_cast_tensor(backend, current, output_dtype")
                .count(),
            1
        );
    }

    #[test]
    fn patch_graph_identity_is_validated_only_through_the_canonical_base_binding()
    -> Result<(), PatchGraphError> {
        let identity = PatchGraph::checked_semantic(BASE_DIGEST, Vec::new())?.identity();
        identity
            .validate_for_base(BASE_DIGEST)
            .map_err(|error| PatchGraphError::InvalidPayload(error.to_string()))?;

        let mut invalid_schema = identity.clone();
        invalid_schema.schema_version += 1;
        assert!(matches!(
            invalid_schema.validate_for_base(BASE_DIGEST),
            Err(PatchGraphIdentityError::SchemaVersion { .. })
        ));
        assert!(matches!(
            identity.validate_for_base(&"b".repeat(64)),
            Err(PatchGraphIdentityError::BaseDigestMismatch { .. })
        ));
        let mut invalid_digest = identity;
        invalid_digest.ordered_digest = "not-a-digest".to_owned();
        assert!(matches!(
            invalid_digest.validate_for_base(BASE_DIGEST),
            Err(PatchGraphIdentityError::InvalidDigest {
                field: "ordered",
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn patch_graph_resident_bytes_track_owned_payload_without_changing_identity()
    -> Result<(), PatchGraphError> {
        let first = PatchGraph::checked_semantic(
            BASE_DIGEST,
            vec![operation(
                "resident",
                &[2],
                PatchPayload::Set {
                    tensor: tensor(&[2], &[1.0, 2.0]),
                },
            )],
        )?;
        let equivalent = first.clone();
        let larger = PatchGraph::checked_semantic(
            BASE_DIGEST,
            vec![operation(
                "resident",
                &[4],
                PatchPayload::Set {
                    tensor: tensor(&[4], &[1.0, 2.0, 3.0, 4.0]),
                },
            )],
        )?;
        assert_eq!(first.identity(), equivalent.identity());
        assert_eq!(first.resident_bytes()?, equivalent.resident_bytes()?);
        assert!(larger.resident_bytes()? > first.resident_bytes()?);
        assert_ne!(larger.identity(), first.identity());
        Ok(())
    }

    #[test]
    fn nested_base_convert_func_is_explicit_typed_transform_metadata() -> Result<(), PatchGraphError>
    {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(256 * 1024)?,
            &cancellation,
        );
        let payload = PatchPayload::Nested {
            base: tensor(&[1], &[2.0]),
            base_transform: PatchValueTransform {
                scale: 2.0,
                bias: 1.0,
            },
            patches: vec![NestedPatch {
                strength: 1.0,
                strength_model: 1.0,
                transform: PatchValueTransform::default(),
                payload: PatchPayload::DenseDiff {
                    tensor: tensor(&[1], &[0.0]),
                    pad_weight: false,
                },
            }],
        };
        let operation = operation("nested-base-transform", &[1], payload);
        let encoded = serde_json::to_value(&operation)
            .map_err(|error| PatchGraphError::Serialization(error.to_string()))?;
        assert_eq!(encoded["payload"]["base_transform"]["scale"], 2.0);
        assert_eq!(encoded["payload"]["base_transform"]["bias"], 1.0);
        let source = mapped(&backend, &context, &[1], &[1.0])?;
        let patched = PatchGraph::checked_semantic(BASE_DIGEST, vec![operation])?
            .apply(&backend, &source, &context)?;
        assert_eq!(values(&backend, &patched, &context)?, [6.0]);
        Ok(())
    }

    #[test]
    fn boft_dora_requires_source_exact_explicit_constraint_alpha() -> Result<(), PatchGraphError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(256 * 1024)?,
            &cancellation,
        );
        let source = mapped(&backend, &context, &[2, 2], &[1.0, 0.0, 0.0, 1.0])?;
        let source_bytes = weight_bytes(&source)?;
        let missing_alpha = operation(
            "boft-dora-missing-alpha",
            &[2, 2],
            PatchPayload::Boft {
                blocks: tensor(&[1, 1, 2, 2], &[0.0; 4]),
                rescale: None,
                constraint: None,
                dora_scale: Some(tensor(&[2], &[1.0, 1.0])),
            },
        );
        assert!(matches!(
            PatchGraph::checked_semantic(BASE_DIGEST, vec![missing_alpha]),
            Err(PatchGraphError::InvalidPayload(message))
                if message == "BOFT with DORA requires an explicit constraint alpha"
        ));
        assert_eq!(weight_bytes(&source)?, source_bytes);

        let explicit_zero = operation(
            "boft-dora-zero-alpha",
            &[2, 2],
            PatchPayload::Boft {
                blocks: tensor(&[1, 1, 2, 2], &[0.0; 4]),
                rescale: None,
                constraint: Some(0.0),
                dora_scale: Some(tensor(&[2], &[1.0, 1.0])),
            },
        );
        PatchGraph::checked_semantic(BASE_DIGEST, vec![explicit_zero])?;
        Ok(())
    }

    #[test]
    fn semantic_dense_set_slice_order_digest_and_copy_on_write() -> Result<(), PatchGraphError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(256 * 1024)?,
            &cancellation,
        );
        let source = mapped(&backend, &context, &[2, 2], &[1.0, 2.0, 3.0, 4.0])?;
        let mut sliced = operation(
            "slice",
            &[2, 2],
            PatchPayload::DenseDiff {
                tensor: tensor(&[1, 2], &[10.0, 20.0]),
                pad_weight: false,
            },
        );
        sliced.slices.push(PatchSlice {
            dimension: 0,
            start: 1,
            length: 1,
        });
        let set = operation(
            "set",
            &[2, 2],
            PatchPayload::Set {
                tensor: tensor(&[2, 2], &[7.0, 8.0, 9.0, 10.0]),
            },
        );
        let graph = PatchGraph::checked_semantic(BASE_DIGEST, vec![sliced.clone(), set.clone()])?;
        let reversed = PatchGraph::checked_semantic(BASE_DIGEST, vec![set, sliced])?;
        assert_ne!(
            graph.identity().ordered_digest,
            reversed.identity().ordered_digest
        );
        let patched = graph.apply(&backend, &source, &context)?;
        assert_eq!(values(&backend, &patched, &context)?, [7.0, 8.0, 9.0, 10.0]);
        assert_eq!(values(&backend, &source, &context)?, [1.0, 2.0, 3.0, 4.0]);
        assert_ne!(patched.cache_identity(), source.cache_identity());

        let immutable_source = mapped(&backend, &context, &[2, 2], &[1.0; 4])?;
        let add_first = operation(
            "add-before-model-as-lora",
            &[2, 2],
            PatchPayload::DenseDiff {
                tensor: tensor(&[2, 2], &[1.0; 4]),
                pad_weight: false,
            },
        );
        let model_as_lora = operation(
            "model-as-lora",
            &[2, 2],
            PatchPayload::ModelAsLora {
                target: tensor(&[2, 2], &[3.0; 4]),
            },
        );
        let immutable_graph =
            PatchGraph::checked_semantic(BASE_DIGEST, vec![add_first, model_as_lora])?;
        let immutable_result = immutable_graph.apply(&backend, &immutable_source, &context)?;
        assert_eq!(values(&backend, &immutable_result, &context)?, [4.0; 4]);
        Ok(())
    }

    #[test]
    fn lora_loha_lokr_and_convolution_shapes_are_exact() -> Result<(), PatchGraphError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(256 * 1024)?,
            &cancellation,
        );
        let identity = tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let base = tensor(&[2, 2], &[0.0; 4]);
        let convolution_base = tensor(&[2, 2, 1, 1], &[0.0; 4]);
        let lora = PatchPayload::Lora {
            up: identity.clone(),
            down: identity.clone(),
            mid: None,
            alpha: Some(2.0),
            dora_scale: None,
            reshape: None,
        };
        assert_eq!(
            apply_payload(
                &backend,
                &base,
                &lora,
                1.0,
                PatchValueTransform::default(),
                &context,
                0
            )?
            .values,
            identity.values
        );
        let decomposed_lokr = PatchPayload::Lokr {
            first: None,
            second: None,
            first_up: Some(tensor(&[1, 2], &[1.0, 0.0])),
            first_down: Some(tensor(&[2, 1], &[1.0, 0.0])),
            second_up: Some(tensor(&[2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0])),
            second_down: Some(tensor(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0])),
            second_tucker: None,
            alpha: Some(6.0),
            dora_scale: None,
        };
        assert_eq!(
            apply_payload(
                &backend,
                &base,
                &decomposed_lokr,
                1.0,
                PatchValueTransform::default(),
                &context,
                0,
            )?
            .values,
            [2.0, 0.0, 0.0, 2.0]
        );
        let convolution_lokr = PatchPayload::Lokr {
            first: Some(tensor(&[1, 1], &[1.0])),
            second: Some(tensor(&[2, 2, 1, 1], &[1.0, 0.0, 0.0, 1.0])),
            first_up: None,
            first_down: None,
            second_up: None,
            second_down: None,
            second_tucker: None,
            alpha: None,
            dora_scale: None,
        };
        assert_eq!(
            apply_payload(
                &backend,
                &convolution_base,
                &convolution_lokr,
                1.0,
                PatchValueTransform::default(),
                &context,
                0,
            )?,
            tensor(&[2, 2, 1, 1], &[1.0, 0.0, 0.0, 1.0])
        );
        let loha = PatchPayload::Loha {
            first_up: identity.clone(),
            first_down: identity.clone(),
            second_up: identity.clone(),
            second_down: identity.clone(),
            first_tucker: None,
            second_tucker: None,
            alpha: Some(2.0),
            dora_scale: None,
        };
        assert_eq!(
            apply_payload(
                &backend,
                &base,
                &loha,
                1.0,
                PatchValueTransform::default(),
                &context,
                0
            )?
            .values,
            identity.values
        );
        let lokr = PatchPayload::Lokr {
            first: Some(tensor(&[1, 1], &[1.0])),
            second: Some(identity.clone()),
            first_up: None,
            first_down: None,
            second_up: None,
            second_down: None,
            second_tucker: None,
            alpha: None,
            dora_scale: None,
        };
        assert_eq!(
            apply_payload(
                &backend,
                &base,
                &lokr,
                1.0,
                PatchValueTransform::default(),
                &context,
                0
            )?
            .values,
            identity.values
        );
        let convolution = PatchPayload::Lora {
            up: identity.clone(),
            down: identity,
            mid: None,
            alpha: Some(2.0),
            dora_scale: None,
            reshape: None,
        };
        assert_eq!(
            apply_payload(
                &backend,
                &convolution_base,
                &convolution,
                1.0,
                PatchValueTransform::default(),
                &context,
                0
            )?
            .shape,
            [2, 2, 1, 1]
        );
        Ok(())
    }

    #[test]
    fn tucker_glora_oft_boft_dora_nested_and_model_as_lora_validate() -> Result<(), PatchGraphError>
    {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(512 * 1024)?,
            &cancellation,
        );
        let zero = tensor(&[2, 2], &[0.0; 4]);
        let tucker = PatchPayload::Lora {
            up: tensor(&[2, 1], &[1.0, 2.0]),
            down: tensor(&[1, 2], &[3.0, 4.0]),
            mid: Some(tensor(&[1, 1, 1, 1], &[1.0])),
            alpha: Some(1.0),
            dora_scale: None,
            reshape: Some(vec![2, 2, 1, 1]),
        };
        let tucker_result = apply_payload(
            &backend,
            &tensor(&[2, 2, 1, 1], &[0.0; 4]),
            &tucker,
            1.0,
            PatchValueTransform::default(),
            &context,
            0,
        )?;
        assert_eq!(tucker_result.values, [3.0, 4.0, 6.0, 8.0]);
        let oft = PatchPayload::Oft {
            blocks: tensor(&[1, 2, 2], &[0.0; 4]),
            rescale: None,
            constraint: Some(1.0),
            dora_scale: None,
        };
        assert_eq!(
            apply_payload(
                &backend,
                &zero,
                &oft,
                1.0,
                PatchValueTransform::default(),
                &context,
                0
            )?
            .values,
            zero.values
        );
        let glora = PatchPayload::Glora {
            first_a: tensor(&[2, 1], &[1.0, 0.0]),
            second_a: tensor(&[1, 2], &[1.0, 0.0]),
            first_b: tensor(&[2, 1], &[1.0, 0.0]),
            second_b: tensor(&[1, 2], &[1.0, 0.0]),
            alpha: Some(1.0),
            dora_scale: None,
        };
        assert_eq!(
            apply_payload(
                &backend,
                &zero,
                &glora,
                1.0,
                PatchValueTransform::default(),
                &context,
                0
            )?
            .values,
            [1.0, 0.0, 0.0, 0.0]
        );
        let boft = PatchPayload::Boft {
            blocks: tensor(&[1, 1, 2, 2], &[0.0; 4]),
            rescale: None,
            constraint: None,
            dora_scale: None,
        };
        assert_eq!(
            apply_payload(
                &backend,
                &zero,
                &boft,
                1.0,
                PatchValueTransform::default(),
                &context,
                0
            )?
            .values,
            zero.values
        );
        let dora = PatchPayload::Dora {
            difference: tensor(&[2, 2], &[0.0; 4]),
            scale: tensor(&[2], &[1.0, 1.0]),
            alpha: 1.0,
        };
        let dora_result = apply_payload(
            &backend,
            &tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]),
            &dora,
            1.0,
            PatchValueTransform::default(),
            &context,
            0,
        )?;
        assert!((dora_result.values[0] - 1.0).abs() < 1.0e-5);
        let nested = PatchPayload::Nested {
            base: zero.clone(),
            base_transform: PatchValueTransform::default(),
            patches: vec![NestedPatch {
                strength: 1.0,
                strength_model: 1.0,
                transform: PatchValueTransform::default(),
                payload: PatchPayload::DenseDiff {
                    tensor: tensor(&[2, 2], &[1.0; 4]),
                    pad_weight: false,
                },
            }],
        };
        assert_eq!(
            apply_payload(
                &backend,
                &zero,
                &nested,
                1.0,
                PatchValueTransform::default(),
                &context,
                0
            )?
            .values,
            [1.0; 4]
        );
        let model_as = PatchPayload::ModelAsLora {
            target: tensor(&[2, 2], &[2.0; 4]),
        };
        assert_eq!(
            apply_payload(
                &backend,
                &zero,
                &model_as,
                1.0,
                PatchValueTransform::default(),
                &context,
                0
            )?
            .values,
            [2.0; 4]
        );
        let padded = PatchPayload::DenseDiff {
            tensor: tensor(&[2, 2], &[1.0; 4]),
            pad_weight: true,
        };
        assert_eq!(
            apply_payload(
                &backend,
                &tensor(&[1, 1], &[2.0]),
                &padded,
                1.0,
                PatchValueTransform::default(),
                &context,
                0,
            )?,
            tensor(&[2, 2], &[3.0, 1.0, 1.0, 1.0])
        );
        Ok(())
    }

    #[test]
    fn nonzero_oft_boft_and_dora_match_pinned_weight_adapter_equations()
    -> Result<(), PatchGraphError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(512 * 1024)?,
            &cancellation,
        );
        let identity = tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let skew_source = tensor(&[1, 2, 2], &[0.0, 0.25, -0.25, 0.0]);
        let oft = PatchPayload::Oft {
            blocks: skew_source,
            rescale: None,
            constraint: None,
            dora_scale: None,
        };
        let oft = apply_payload(
            &backend,
            &identity,
            &oft,
            1.0,
            PatchValueTransform::default(),
            &context,
            0,
        )?;
        assert_close(&oft.values, &[0.6, -0.8, 0.8, 0.6]);

        let boft = PatchPayload::Boft {
            blocks: tensor(&[1, 1, 2, 2], &[0.0, 0.25, -0.25, 0.0]),
            rescale: Some(tensor(&[1], &[2.0])),
            constraint: None,
            dora_scale: None,
        };
        let boft = apply_payload(
            &backend,
            &identity,
            &boft,
            1.0,
            PatchValueTransform::default(),
            &context,
            0,
        )?;
        assert_close(&boft.values, &[1.2, 1.6, -1.6, 1.2]);

        let dora = PatchPayload::Dora {
            difference: tensor(&[2, 2], &[1.0; 4]),
            scale: tensor(&[2], &[2.0, 3.0]),
            alpha: 0.5,
        };
        let dora = apply_payload(
            &backend,
            &identity,
            &dora,
            0.25,
            PatchValueTransform::default(),
            &context,
            0,
        )?;
        assert_close(&dora.values, &[1.5, 0.375, 0.25, 1.875]);
        let shaped_dora = PatchPayload::Dora {
            difference: tensor(&[2, 2], &[1.0; 4]),
            scale: tensor(&[2, 1], &[2.0, 3.0]),
            alpha: 0.5,
        };
        let shaped_dora = apply_payload(
            &backend,
            &identity,
            &shaped_dora,
            0.25,
            PatchValueTransform::default(),
            &context,
            0,
        )?;
        assert_close(&shaped_dora.values, &[1.5, 0.25, 0.375, 1.875]);
        Ok(())
    }

    #[test]
    fn glora_layout_inference_and_convolution_preserve_source_axes() -> Result<(), PatchGraphError>
    {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(512 * 1024)?,
            &cancellation,
        );
        let convolution = PatchPayload::Glora {
            first_a: tensor(&[2, 1], &[1.0, 1.0]),
            second_a: tensor(&[1, 2], &[2.0, 3.0]),
            first_b: tensor(&[1, 1], &[0.0]),
            second_b: tensor(&[1, 2], &[0.0, 0.0]),
            alpha: None,
            dora_scale: None,
        };
        let convolution_result = apply_payload(
            &backend,
            &tensor(&[1, 2, 1, 1], &[1.0, 2.0]),
            &convolution,
            1.0,
            PatchValueTransform::default(),
            &context,
            0,
        )?;
        assert_eq!(convolution_result.shape, [1, 2, 1, 1]);
        assert_close(&convolution_result.values, &[7.0, 11.0]);

        let old_layout = PatchPayload::Glora {
            first_a: tensor(&[1, 2], &[1.0, 0.0]),
            second_a: tensor(&[2, 1], &[1.0, 0.0]),
            first_b: tensor(&[1, 2], &[0.0, 0.0]),
            second_b: tensor(&[2, 1], &[0.0, 0.0]),
            alpha: None,
            dora_scale: None,
        };
        let old_result = apply_payload(
            &backend,
            &tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]),
            &old_layout,
            1.0,
            PatchValueTransform::default(),
            &context,
            0,
        )?;
        assert_close(&old_result.values, &[2.0, 0.0, 0.0, 1.0]);
        Ok(())
    }

    #[test]
    fn multidimensional_offsets_strength_model_and_transforms_are_ordered()
    -> Result<(), PatchGraphError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(256 * 1024)?,
            &cancellation,
        );
        let source = mapped(
            &backend,
            &context,
            &[2, 3, 2],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        )?;
        let mut sliced = operation(
            "two-dimensional-slice",
            &[2, 3, 2],
            PatchPayload::DenseDiff {
                tensor: tensor(&[1, 2, 2], &[1.0; 4]),
                pad_weight: false,
            },
        );
        sliced.strength = 2.0;
        sliced.strength_model = 0.5;
        sliced.transform = PatchValueTransform {
            scale: 3.0,
            bias: 4.0,
        };
        sliced.slices = vec![
            PatchSlice {
                dimension: 0,
                start: 1,
                length: 1,
            },
            PatchSlice {
                dimension: 1,
                start: 1,
                length: 2,
            },
        ];
        let patched = PatchGraph::checked_semantic(BASE_DIGEST, vec![sliced])?
            .apply(&backend, &source, &context)?;
        assert_close(
            &values(&backend, &patched, &context)?,
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 14.0, 14.5, 15.0, 15.5,
            ],
        );
        Ok(())
    }

    #[test]
    fn validation_and_transaction_failures_publish_nothing() -> Result<(), PatchGraphError> {
        let invalid = operation(
            "bad",
            &[2, 2],
            PatchPayload::DenseDiff {
                tensor: PatchTensor {
                    shape: vec![2, 2],
                    values: vec![f32::NAN; 4],
                },
                pad_weight: false,
            },
        );
        assert!(matches!(
            PatchGraph::checked_semantic(BASE_DIGEST, vec![invalid]),
            Err(PatchGraphError::InvalidPayload(_))
        ));
        let missing = operation(
            "missing",
            &[2, 2],
            PatchPayload::DenseDiff {
                tensor: tensor(&[2, 2], &[1.0; 4]),
                pad_weight: false,
            },
        );
        let graph = PatchGraph::checked_semantic(BASE_DIGEST, vec![missing])?;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(128 * 1024)?,
            &cancellation,
        );
        let source =
            MappedModelWeights::from_parts(BASE_DIGEST.to_owned(), BTreeMap::new(), Vec::new());
        assert!(matches!(
            graph.apply(&backend, &source, &context),
            Err(PatchGraphError::MissingTarget(_))
        ));
        let source = mapped(&backend, &context, &[2, 2], &[1.0; 4])?;
        let original_values = values(&backend, &source, &context)?;
        let (constrained_backend, constrained_authority) =
            CpuWorkspaceAuthority::create_backend(32)?;
        let constrained_context = constrained_backend.execution_context(
            StreamId::DEFAULT,
            constrained_authority.authorize_workspace(16)?,
            &cancellation,
        );
        assert!(
            graph
                .apply(&constrained_backend, &source, &constrained_context)
                .is_err()
        );
        assert_eq!(values(&backend, &source, &context)?, original_values);
        cancellation.cancel();
        assert!(matches!(
            graph.apply(&backend, &source, &context),
            Err(PatchGraphError::Cancelled(_))
        ));
        assert_eq!(source.cache_identity(), BASE_DIGEST);
        Ok(())
    }

    #[test]
    fn source_pinned_oft_boft_constraint_rescale_and_strength_are_exact()
    -> Result<(), PatchGraphError> {
        const SOURCE_CONTRACTS: &str = include_str!(
            "../../../.agents/specs/comfy-parity/catalogs/backend-conditioning-contracts.csv"
        );
        assert!(SOURCE_CONTRACTS.contains(concat!(
            "projects/comfy/ComfyUI/sim/weight_adapter/oft.py,calculate_weight,172,",
            "88be3c32f610478bc6900a10009eaadf6fe2af973ed4861731e4f21e1afacf89,",
            "a01a25e6e53a021e86e4df70556eeef5fc8b33c597c72cbf33d6cc0703a12772"
        )));
        assert!(SOURCE_CONTRACTS.contains(concat!(
            "projects/comfy/ComfyUI/sim/weight_adapter/boft.py,calculate_weight,158,",
            "2850e0b4c2295cd87445415e287061fa3bfd69e88bd0aeb3eb16064864bd078d,",
            "94f4949f38d7f37f8267d68aac1f9e5389d8b3e1845cf7efd4f69357856d0c4d"
        )));

        let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let identity = tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let blocks = tensor(&[1, 2, 2], &[0.0, 0.25, -0.25, 0.0]);
        let half_strength = PatchPayload::Oft {
            blocks: blocks.clone(),
            rescale: None,
            constraint: None,
            dora_scale: None,
        };
        assert_close(
            &apply_payload(
                &backend,
                &identity,
                &half_strength,
                0.5,
                PatchValueTransform::default(),
                &context,
                0,
            )?
            .values,
            &[0.9, -0.2, 0.2, 0.9],
        );
        let ignored_rescale = PatchPayload::Oft {
            blocks: blocks.clone(),
            rescale: Some(tensor(&[1], &[99.0])),
            constraint: None,
            dora_scale: None,
        };
        assert_eq!(
            apply_payload(
                &backend,
                &identity,
                &ignored_rescale,
                1.0,
                PatchValueTransform::default(),
                &context,
                0,
            )?,
            apply_payload(
                &backend,
                &identity,
                &PatchPayload::Oft {
                    blocks: blocks.clone(),
                    rescale: None,
                    constraint: None,
                    dora_scale: None,
                },
                1.0,
                PatchValueTransform::default(),
                &context,
                0,
            )?,
        );
        let constrained = PatchPayload::Oft {
            blocks: tensor(&[1, 2, 2], &[0.0, 0.5, -0.5, 0.0]),
            rescale: None,
            constraint: Some(0.1),
            dora_scale: None,
        };
        assert_close(
            &apply_payload(
                &backend,
                &identity,
                &constrained,
                1.0,
                PatchValueTransform::default(),
                &context,
                0,
            )?
            .values,
            &[0.9900498, -0.1407178, 0.1407178, 0.9900498],
        );
        let negative_constraint = PatchPayload::Oft {
            blocks: blocks.clone(),
            rescale: None,
            constraint: Some(-1.0),
            dora_scale: None,
        };
        assert_eq!(
            apply_payload(
                &backend,
                &identity,
                &negative_constraint,
                1.0,
                PatchValueTransform::default(),
                &context,
                0,
            )?,
            apply_payload(
                &backend,
                &identity,
                &PatchPayload::Oft {
                    blocks,
                    rescale: None,
                    constraint: None,
                    dora_scale: None,
                },
                1.0,
                PatchValueTransform::default(),
                &context,
                0,
            )?,
        );
        let boft = PatchPayload::Boft {
            blocks: tensor(&[1, 1, 2, 2], &[0.0; 4]),
            rescale: Some(tensor(&[2], &[2.0, 3.0])),
            constraint: None,
            dora_scale: None,
        };
        assert_close(
            &apply_payload(
                &backend,
                &identity,
                &boft,
                1.0,
                PatchValueTransform::default(),
                &context,
                0,
            )?
            .values,
            &[2.0, 0.0, 0.0, 3.0],
        );
        let multi_stage = PatchPayload::Boft {
            blocks: tensor(&[2, 2, 2, 2], &[0.0; 16]),
            rescale: None,
            constraint: None,
            dora_scale: None,
        };
        let identity_four = tensor(
            &[4, 4],
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        );
        assert_eq!(
            apply_payload(
                &backend,
                &identity_four,
                &multi_stage,
                1.0,
                PatchValueTransform::default(),
                &context,
                0,
            )?,
            identity_four,
        );
        Ok(())
    }

    #[test]
    fn tucker_variants_non_square_dora_and_nested_depth_are_exact() -> Result<(), PatchGraphError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(2 * 1024 * 1024)?,
            &cancellation,
        );
        let up = tensor(&[2, 1], &[1.0, 2.0]);
        let down = tensor(&[1, 2], &[3.0, 4.0]);
        let core = tensor(&[1, 1, 1, 1], &[1.0]);
        let loha = PatchPayload::Loha {
            first_up: up.clone(),
            first_down: down.clone(),
            second_up: up.clone(),
            second_down: down.clone(),
            first_tucker: Some(core.clone()),
            second_tucker: Some(core.clone()),
            alpha: Some(1.0),
            dora_scale: None,
        };
        assert_close(
            &apply_payload(
                &backend,
                &tensor(&[2, 2, 1, 1], &[0.0; 4]),
                &loha,
                1.0,
                PatchValueTransform::default(),
                &context,
                0,
            )?
            .values,
            &[9.0, 16.0, 36.0, 64.0],
        );
        let lokr = PatchPayload::Lokr {
            first: Some(tensor(&[1, 1], &[1.0])),
            second: None,
            first_up: None,
            first_down: None,
            second_up: Some(up),
            second_down: Some(down),
            second_tucker: Some(core),
            alpha: None,
            dora_scale: None,
        };
        assert_close(
            &apply_payload(
                &backend,
                &tensor(&[2, 2, 1, 1], &[0.0; 4]),
                &lokr,
                1.0,
                PatchValueTransform::default(),
                &context,
                0,
            )?
            .values,
            &[3.0, 4.0, 6.0, 8.0],
        );
        let dora = PatchPayload::Dora {
            difference: tensor(&[2, 3], &[1.0; 6]),
            scale: tensor(
                &[3],
                &[2.0_f32.sqrt(), 2.0_f32.sqrt() * 2.0, 2.0_f32.sqrt() * 3.0],
            ),
            alpha: 1.0,
        };
        assert_close(
            &apply_payload(
                &backend,
                &tensor(&[2, 3], &[0.0; 6]),
                &dora,
                1.0,
                PatchValueTransform::default(),
                &context,
                0,
            )?
            .values,
            &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        );
        let signed_dora = PatchPayload::Dora {
            difference: tensor(&[2, 3], &[1.0; 6]),
            scale: tensor(
                &[3],
                &[
                    -2.0_f32.sqrt(),
                    -2.0_f32.sqrt() * 2.0,
                    -2.0_f32.sqrt() * 3.0,
                ],
            ),
            alpha: 1.0,
        };
        assert_close(
            &apply_payload(
                &backend,
                &tensor(&[2, 3], &[0.0; 6]),
                &signed_dora,
                1.0,
                PatchValueTransform::default(),
                &context,
                0,
            )?
            .values,
            &[-1.0, -2.0, -3.0, -1.0, -2.0, -3.0],
        );

        let mut nested = PatchPayload::DenseDiff {
            tensor: tensor(&[1], &[1.0]),
            pad_weight: false,
        };
        for _ in 0..=MAX_SEMANTIC_PATCH_DEPTH {
            nested = PatchPayload::Nested {
                base: tensor(&[1], &[0.0]),
                base_transform: PatchValueTransform::default(),
                patches: vec![NestedPatch {
                    strength: 1.0,
                    strength_model: 1.0,
                    transform: PatchValueTransform::default(),
                    payload: nested,
                }],
            };
        }
        assert!(matches!(
            PatchGraph::checked_semantic(BASE_DIGEST, vec![operation("too-deep", &[1], nested)]),
            Err(PatchGraphError::NestingDepth)
        ));
        Ok(())
    }

    #[test]
    fn sequential_graphs_bind_model_as_lora_to_raw_base_and_digest_every_field()
    -> Result<(), PatchGraphError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(512 * 1024)?,
            &cancellation,
        );
        let source = mapped(&backend, &context, &[1], &[1.0])?;
        let first = PatchGraph::checked_semantic(
            BASE_DIGEST,
            vec![operation(
                "first",
                &[1],
                PatchPayload::DenseDiff {
                    tensor: tensor(&[1], &[1.0]),
                    pad_weight: false,
                },
            )],
        )?
        .apply(&backend, &source, &context)?;
        let second_operation = operation(
            "model-as-lora",
            &[1],
            PatchPayload::ModelAsLora {
                target: tensor(&[1], &[3.0]),
            },
        );
        let second = PatchGraph::checked_semantic(BASE_DIGEST, vec![second_operation.clone()])?
            .apply(&backend, &first, &context)?;
        assert_close(&values(&backend, &second, &context)?, &[4.0]);
        assert_close(&values(&backend, &source, &context)?, &[1.0]);

        let digest = PatchGraph::checked_semantic(BASE_DIGEST, vec![second_operation.clone()])?
            .identity()
            .ordered_digest;
        let mut variants = Vec::new();
        let mut changed = second_operation.clone();
        changed.identifier = "changed-id".into();
        variants.push(changed);
        let mut changed = second_operation.clone();
        changed.target_key = "changed-key".into();
        variants.push(changed);
        let mut changed = second_operation.clone();
        changed.strength = 0.5;
        variants.push(changed);
        let mut changed = second_operation.clone();
        changed.strength_model = 0.5;
        variants.push(changed);
        let mut changed = second_operation;
        changed.transform.bias = 1.0;
        variants.push(changed);
        for changed in variants {
            assert_ne!(
                PatchGraph::checked_semantic(BASE_DIGEST, vec![changed])?
                    .identity()
                    .ordered_digest,
                digest
            );
        }
        let unit = tensor(&[1, 1], &[1.0]);
        let payloads = vec![
            PatchPayload::DenseDiff {
                tensor: unit.clone(),
                pad_weight: true,
            },
            PatchPayload::Set {
                tensor: unit.clone(),
            },
            PatchPayload::Lora {
                up: unit.clone(),
                down: unit.clone(),
                mid: Some(unit.clone()),
                alpha: Some(0.75),
                dora_scale: Some(unit.clone()),
                reshape: Some(vec![1, 1]),
            },
            PatchPayload::Loha {
                first_up: unit.clone(),
                first_down: unit.clone(),
                second_up: unit.clone(),
                second_down: unit.clone(),
                first_tucker: Some(unit.clone()),
                second_tucker: Some(unit.clone()),
                alpha: Some(0.75),
                dora_scale: Some(unit.clone()),
            },
            PatchPayload::Lokr {
                first: None,
                second: None,
                first_up: Some(unit.clone()),
                first_down: Some(unit.clone()),
                second_up: Some(unit.clone()),
                second_down: Some(unit.clone()),
                second_tucker: Some(unit.clone()),
                alpha: Some(0.75),
                dora_scale: Some(unit.clone()),
            },
            PatchPayload::Lokr {
                first: Some(unit.clone()),
                second: Some(unit.clone()),
                first_up: None,
                first_down: None,
                second_up: None,
                second_down: None,
                second_tucker: None,
                alpha: None,
                dora_scale: None,
            },
            PatchPayload::Oft {
                blocks: tensor(&[1, 1, 1], &[0.25]),
                rescale: Some(unit.clone()),
                constraint: Some(0.75),
                dora_scale: Some(unit.clone()),
            },
            PatchPayload::Glora {
                first_a: unit.clone(),
                second_a: unit.clone(),
                first_b: unit.clone(),
                second_b: unit.clone(),
                alpha: Some(0.75),
                dora_scale: Some(unit.clone()),
            },
            PatchPayload::Boft {
                blocks: tensor(&[1, 1, 2, 2], &[0.0, 0.25, -0.25, 0.0]),
                rescale: Some(unit.clone()),
                constraint: Some(0.75),
                dora_scale: Some(unit.clone()),
            },
            PatchPayload::Dora {
                difference: unit.clone(),
                scale: unit.clone(),
                alpha: 0.75,
            },
            PatchPayload::Nested {
                base: unit.clone(),
                base_transform: PatchValueTransform::default(),
                patches: vec![NestedPatch {
                    strength: 0.75,
                    strength_model: 0.5,
                    transform: PatchValueTransform {
                        scale: 1.25,
                        bias: 0.25,
                    },
                    payload: PatchPayload::Set {
                        tensor: unit.clone(),
                    },
                }],
            },
            PatchPayload::ModelAsLora { target: unit },
        ];
        for (index, payload) in payloads.into_iter().enumerate() {
            let mut operation = operation(&format!("digest-{index}"), &[1, 1], payload);
            operation.strength = 0.75;
            operation.strength_model = 0.5;
            operation.slices = vec![PatchSlice {
                dimension: 0,
                start: 0,
                length: 1,
            }];
            operation.transform = PatchValueTransform {
                scale: 1.25,
                bias: 0.25,
            };
            assert_every_serialized_leaf_changes_digest(&operation)?;
        }
        Ok(())
    }

    #[test]
    fn bf16_set_factor_model_padding_and_sliced_replacement_paths_cast_once_at_commit()
    -> Result<(), PatchGraphError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );

        let scalar_cases = [
            (
                "set-bf16",
                PatchPayload::Set {
                    tensor: tensor(&[1, 1], &[1.00390625]),
                },
                1.0,
            ),
            (
                "loha-bf16",
                PatchPayload::Loha {
                    first_up: tensor(&[1, 1], &[1.00390625]),
                    first_down: tensor(&[1, 1], &[1.0]),
                    second_up: tensor(&[1, 1], &[1.0]),
                    second_down: tensor(&[1, 1], &[1.0]),
                    first_tucker: None,
                    second_tucker: None,
                    alpha: Some(1.0),
                    dora_scale: None,
                },
                1.0,
            ),
            (
                "lokr-bf16",
                PatchPayload::Lokr {
                    first: Some(tensor(&[1, 1], &[1.0])),
                    second: Some(tensor(&[1, 1], &[1.00390625])),
                    first_up: None,
                    first_down: None,
                    second_up: None,
                    second_down: None,
                    second_tucker: None,
                    alpha: Some(1.0),
                    dora_scale: None,
                },
                1.0,
            ),
            (
                "glora-bf16",
                PatchPayload::Glora {
                    first_a: tensor(&[1, 1], &[0.0]),
                    second_a: tensor(&[1, 1], &[0.0]),
                    first_b: tensor(&[1, 1], &[1.00390625]),
                    second_b: tensor(&[1, 1], &[1.0]),
                    alpha: Some(1.0),
                    dora_scale: None,
                },
                1.0,
            ),
            (
                "model-as-lora-bf16",
                PatchPayload::ModelAsLora {
                    target: tensor(&[1, 1], &[1.00390625]),
                },
                1.0,
            ),
        ];
        for (identifier, payload, expected) in scalar_cases {
            let source = mapped_dtype(&backend, &context, &[1, 1], &[0.0], DType::Bf16)?;
            let patched = PatchGraph::checked_semantic(
                BASE_DIGEST,
                vec![operation(identifier, &[1, 1], payload)],
            )?
            .apply(&backend, &source, &context)?;
            let expected = mapped_dtype(&backend, &context, &[1, 1], &[expected], DType::Bf16)?;
            assert_eq!(weight_bytes(&patched)?, weight_bytes(&expected)?);
        }

        let padded_source = mapped_dtype(&backend, &context, &[1], &[1.0], DType::Bf16)?;
        let padded = PatchGraph::checked_semantic(
            BASE_DIGEST,
            vec![operation(
                "padding-bf16",
                &[1],
                PatchPayload::DenseDiff {
                    tensor: tensor(&[2], &[0.00390625, 1.00390625]),
                    pad_weight: true,
                },
            )],
        )?
        .apply(&backend, &padded_source, &context)?;
        let expected_padded = mapped_dtype(&backend, &context, &[2], &[1.0, 1.0], DType::Bf16)?;
        assert_eq!(weight_bytes(&padded)?, weight_bytes(&expected_padded)?);

        let sliced_source = mapped_dtype(
            &backend,
            &context,
            &[2, 2],
            &[1.00390625, 2.0078125, 3.00390625, 4.0078125],
            DType::Bf16,
        )?;
        let source_bytes = weight_bytes(&sliced_source)?;
        let mut sliced = operation(
            "sliced-set-bf16",
            &[2, 2],
            PatchPayload::Set {
                tensor: tensor(&[1, 2], &[9.00390625, 10.0078125]),
            },
        );
        sliced.slices.push(PatchSlice {
            dimension: 0,
            start: 1,
            length: 1,
        });
        let sliced = PatchGraph::checked_semantic(BASE_DIGEST, vec![sliced])?.apply(
            &backend,
            &sliced_source,
            &context,
        )?;
        let expected_sliced = mapped_dtype(
            &backend,
            &context,
            &[2, 2],
            &[1.0, 2.0, 9.0, 10.0],
            DType::Bf16,
        )?;
        let sliced_bytes = weight_bytes(&sliced)?;
        assert_eq!(sliced_bytes, weight_bytes(&expected_sliced)?);
        assert_eq!(sliced_bytes.get(..4), source_bytes.get(..4));

        Ok(())
    }

    #[test]
    fn checked_compute_boundary_distinguishes_normal_and_low_vram_across_ordered_bf16_patches()
    -> Result<(), PatchGraphError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let source = mapped_dtype(&backend, &context, &[2, 2], &[0.0; 4], DType::Bf16)?;
        let nonlinear = operation(
            "nonlinear-bf16",
            &[2, 2],
            PatchPayload::Loha {
                first_up: tensor(&[2, 2], &[1.00390625, 0.0, 0.0, 1.01171875]),
                first_down: tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]),
                second_up: tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]),
                second_down: tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]),
                first_tucker: None,
                second_tucker: None,
                alpha: Some(2.0),
                dora_scale: None,
            },
        );
        let repeated_target = operation(
            "ordered-bf16-followup",
            &[2, 2],
            PatchPayload::DenseDiff {
                tensor: tensor(&[2, 2], &[0.00390625, 0.0, 0.0, -0.00390625]),
                pad_weight: false,
            },
        );
        let graph = PatchGraph::checked_semantic(BASE_DIGEST, vec![nonlinear, repeated_target])?;
        let structural_identity = graph.identity();
        let normal_boundary = PatchComputeBoundary::configured(DType::F32)?;
        let low_vram_boundary = PatchComputeBoundary::weight_dtype();
        let normal =
            graph.apply_with_compute_boundary(&backend, &source, normal_boundary, &context)?;
        let low_vram =
            graph.apply_with_compute_boundary(&backend, &source, low_vram_boundary, &context)?;
        assert_eq!(graph.identity(), structural_identity);
        assert_ne!(weight_bytes(&normal)?, weight_bytes(&low_vram)?);
        assert_close(
            &values(&backend, &normal, &context)?,
            &[1.0078125, 0.0, 0.0, 1.0078125],
        );
        assert_close(
            &values(&backend, &low_vram, &context)?,
            &[1.0, 0.0, 0.0, 1.015625],
        );
        assert_ne!(normal.cache_identity(), low_vram.cache_identity());
        let repeated_normal =
            graph.apply_with_compute_boundary(&backend, &source, normal_boundary, &context)?;
        let repeated_low_vram =
            graph.apply_with_compute_boundary(&backend, &source, low_vram_boundary, &context)?;
        assert_eq!(normal.cache_identity(), repeated_normal.cache_identity());
        assert_eq!(
            low_vram.cache_identity(),
            repeated_low_vram.cache_identity()
        );

        assert!(PatchComputeBoundary::configured(DType::Bf16).is_err());
        assert!(PatchComputeBoundary::configured(DType::U8).is_err());
        assert!(matches!(
            PatchComputeBoundary::Configured(DType::Bf16).intermediate_dtype(DType::Bf16),
            Err(PatchGraphError::InvalidPayload(_))
        ));
        let compute_and_bf16 = BTreeMap::from([("weight".to_owned(), (DType::F32, DType::Bf16))]);
        let compute_and_f16 = BTreeMap::from([("weight".to_owned(), (DType::F32, DType::F16))]);
        assert_ne!(
            applied_patch_digest(
                &structural_identity.ordered_digest,
                normal_boundary,
                &compute_and_bf16,
            )?,
            applied_patch_digest(
                &structural_identity.ordered_digest,
                normal_boundary,
                &compute_and_f16,
            )?
        );
        assert_ne!(
            applied_patch_digest(
                &structural_identity.ordered_digest,
                normal_boundary,
                &BTreeMap::from([("weight".to_owned(), (DType::F32, DType::F32))]),
            )?,
            applied_patch_digest(
                &structural_identity.ordered_digest,
                low_vram_boundary,
                &BTreeMap::from([("weight".to_owned(), (DType::F32, DType::F32))]),
            )?
        );

        let f32_source = mapped(&backend, &context, &[2, 2], &[0.0; 4])?;
        let configured_f32 =
            graph.apply_with_compute_boundary(&backend, &f32_source, normal_boundary, &context)?;
        let weight_dtype_f32 = graph.apply_with_compute_boundary(
            &backend,
            &f32_source,
            low_vram_boundary,
            &context,
        )?;
        assert_eq!(
            weight_bytes(&configured_f32)?,
            weight_bytes(&weight_dtype_f32)?
        );
        assert_ne!(
            configured_f32.cache_identity(),
            weight_dtype_f32.cache_identity()
        );

        let incomplete = IncompleteBackend::new()?;
        let source_bytes = weight_bytes(&source)?;
        assert!(matches!(
            graph.apply_with_compute_boundary(
                &incomplete,
                &source,
                low_vram_boundary,
                &context,
            ),
            Err(PatchGraphError::Tensor(TensorError::UnsupportedCapability {
                operation,
                device,
                ..
            })) if operation == "COMFY-TENSOR-OP-56B106D5BEE7"
                && device == DeviceId::new(DeviceKind::Metal, 0)
        ));
        assert_eq!(incomplete.unexpected_calls.load(Ordering::Acquire), 0);
        assert_eq!(weight_bytes(&source)?, source_bytes);
        Ok(())
    }

    #[test]
    fn recursive_nested_payload_boundaries_preserve_the_selected_intermediate_dtype()
    -> Result<(), PatchGraphError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(2 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let source = mapped_dtype(&backend, &context, &[2], &[0.0; 2], DType::Bf16)?;
        let recursive_payload = PatchPayload::Nested {
            base: tensor(&[2], &[0.0; 2]),
            base_transform: PatchValueTransform::default(),
            patches: vec![NestedPatch {
                strength: 1.0,
                strength_model: 1.0,
                transform: PatchValueTransform::default(),
                payload: PatchPayload::Nested {
                    base: tensor(&[2], &[1.00390625, 1.01171875]),
                    base_transform: PatchValueTransform::default(),
                    patches: vec![NestedPatch {
                        strength: 1.0,
                        strength_model: 1.0,
                        transform: PatchValueTransform::default(),
                        payload: PatchPayload::DenseDiff {
                            tensor: tensor(&[2], &[0.00390625, -0.00390625]),
                            pad_weight: false,
                        },
                    }],
                },
            }],
        };
        let graph = PatchGraph::checked_semantic(
            BASE_DIGEST,
            vec![operation("recursive-nested-bf16", &[2], recursive_payload)],
        )?;
        let normal = graph.apply_with_compute_boundary(
            &backend,
            &source,
            PatchComputeBoundary::configured(DType::F32)?,
            &context,
        )?;
        let low_vram = graph.apply_with_compute_boundary(
            &backend,
            &source,
            PatchComputeBoundary::weight_dtype(),
            &context,
        )?;
        assert_ne!(weight_bytes(&normal)?, weight_bytes(&low_vram)?);
        assert_close(&values(&backend, &normal, &context)?, &[1.0078125; 2]);
        assert_close(&values(&backend, &low_vram, &context)?, &[1.0, 1.015625]);
        Ok(())
    }

    #[test]
    fn dtype_rounding_nonfinite_cancellation_oom_and_dyn_boundary_are_transactional()
    -> Result<(), PatchGraphError> {
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let mut rounded_operation = operation(
            "rounded",
            &[1],
            PatchPayload::DenseDiff {
                tensor: tensor(&[1], &[0.00390625]),
                pad_weight: false,
            },
        );
        rounded_operation.strength_model = 1.01;
        let graph = PatchGraph::checked_semantic(BASE_DIGEST, vec![rounded_operation])?;
        let bf16_source = mapped_dtype(&backend, &context, &[1], &[1.0], DType::Bf16)?;
        let rounded = graph.apply(&backend, &bf16_source, &context)?;
        assert_close(&values(&backend, &rounded, &context)?, &[1.015625]);

        let mut dense_order = operation(
            "dense-order",
            &[1],
            PatchPayload::DenseDiff {
                tensor: tensor(&[1], &[1.00390625]),
                pad_weight: false,
            },
        );
        dense_order.strength = 1.5;
        dense_order.transform.scale = 1.00390625;
        let dense_order = PatchGraph::checked_semantic(BASE_DIGEST, vec![dense_order])?.apply(
            &backend,
            &mapped_dtype(&backend, &context, &[1], &[0.0], DType::Bf16)?,
            &context,
        )?;
        let mut factor_order = operation(
            "factor-order",
            &[1],
            PatchPayload::Lora {
                up: tensor(&[1, 1], &[1.00390625]),
                down: tensor(&[1, 1], &[1.0]),
                mid: None,
                alpha: Some(1.0),
                dora_scale: None,
                reshape: None,
            },
        );
        factor_order.strength = 1.5;
        factor_order.transform.scale = 1.00390625;
        let factor_order = PatchGraph::checked_semantic(BASE_DIGEST, vec![factor_order])?.apply(
            &backend,
            &mapped_dtype(&backend, &context, &[1], &[0.0], DType::Bf16)?,
            &context,
        )?;
        assert_close(&values(&backend, &dense_order, &context)?, &[1.515625]);
        assert_close(&values(&backend, &factor_order, &context)?, &[1.515625]);
        assert_eq!(weight_bytes(&dense_order)?, weight_bytes(&factor_order)?);

        let mut nested_order = operation(
            "nested-order",
            &[1],
            PatchPayload::Nested {
                base: tensor(&[1], &[1.0009765625]),
                base_transform: PatchValueTransform::default(),
                patches: vec![NestedPatch {
                    strength: 1.0,
                    strength_model: 1.0,
                    transform: PatchValueTransform::default(),
                    payload: PatchPayload::DenseDiff {
                        tensor: tensor(&[1], &[0.0]),
                        pad_weight: false,
                    },
                }],
            },
        );
        nested_order.strength = 1.3;
        nested_order.transform.scale = 0.7;
        let nested_order = PatchGraph::checked_semantic(BASE_DIGEST, vec![nested_order])?.apply(
            &backend,
            &mapped_dtype(&backend, &context, &[1], &[0.0], DType::Bf16)?,
            &context,
        )?;
        assert_close(&values(&backend, &nested_order, &context)?, &[0.91015625]);

        let mut dora_order = operation(
            "dora-order",
            &[1, 2],
            PatchPayload::Dora {
                difference: tensor(&[1, 2], &[1.0, 0.0]),
                scale: tensor(&[1], &[1.0078125]),
                alpha: 1.1,
            },
        );
        dora_order.transform.scale = 1.00390625;
        let dora_order = PatchGraph::checked_semantic(BASE_DIGEST, vec![dora_order])?.apply(
            &backend,
            &mapped_dtype(&backend, &context, &[1, 2], &[1.0, 0.0], DType::Bf16)?,
            &context,
        )?;
        assert_close(&values(&backend, &dora_order, &context)?, &[2.125, 0.0]);

        let block_value = 0.12345_f32;
        let blocks = tensor(&[1, 2, 2], &[0.0, block_value, -block_value, 0.0]);
        let skew_value = block_value * 2.0;
        let denominator = 1.0 + skew_value * skew_value;
        let diagonal = (1.0 - skew_value * skew_value) / denominator;
        let off_diagonal = (2.0 * skew_value) / denominator;
        let expected_rotation = tensor_from_f32_with_backend_exact_native(
            &backend,
            &[1, 2, 2],
            &[diagonal, off_diagonal, -off_diagonal, diagonal],
            DType::Bf16,
            DeviceId::CPU,
            &context,
        )?;
        let rotation = canonical_cayley_rotation(&backend, &blocks, None, DType::F32, &context)?;
        let rounded_rotation = cast_to_with_backend_exact_native(
            &backend,
            &rotation,
            DType::Bf16,
            DeviceId::CPU,
            false,
            false,
            &context,
        )?;
        assert_eq!(
            rounded_rotation.contiguous_bytes()?,
            expected_rotation.contiguous_bytes()?
        );
        let rounded_rotation_values =
            tensor_to_f32_with_backend_exact_native(&backend, &expected_rotation, &context)?;
        let oft_order = PatchGraph::checked_semantic(
            BASE_DIGEST,
            vec![operation(
                "oft-order",
                &[2, 2],
                PatchPayload::Oft {
                    blocks: blocks.clone(),
                    rescale: None,
                    constraint: None,
                    dora_scale: None,
                },
            )],
        )?
        .apply(
            &backend,
            &mapped_dtype(
                &backend,
                &context,
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
                DType::Bf16,
            )?,
            &context,
        )?;
        let expected_oft = mapped_dtype(
            &backend,
            &context,
            &[2, 2],
            &[
                rounded_rotation_values[0],
                rounded_rotation_values[2],
                rounded_rotation_values[1],
                rounded_rotation_values[3],
            ],
            DType::Bf16,
        )?;
        assert_eq!(weight_bytes(&oft_order)?, weight_bytes(&expected_oft)?);

        let boft_strength = 0.73_f32;
        let mut boft_operation = operation(
            "boft-order",
            &[2, 2],
            PatchPayload::Boft {
                blocks: tensor(&[1, 1, 2, 2], &blocks.values),
                rescale: None,
                constraint: None,
                dora_scale: None,
            },
        );
        boft_operation.strength = boft_strength;
        let boft_order = PatchGraph::checked_semantic(BASE_DIGEST, vec![boft_operation])?.apply(
            &backend,
            &mapped_dtype(
                &backend,
                &context,
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
                DType::Bf16,
            )?,
            &context,
        )?;
        let squared_strength = boft_strength * boft_strength;
        let expected_boft = mapped_dtype(
            &backend,
            &context,
            &[2, 2],
            &[
                1.0 + squared_strength * (diagonal - 1.0),
                squared_strength * off_diagonal,
                -squared_strength * off_diagonal,
                1.0 + squared_strength * (diagonal - 1.0),
            ],
            DType::Bf16,
        )?;
        assert_eq!(weight_bytes(&boft_order)?, weight_bytes(&expected_boft)?);

        let overflow = PatchGraph::checked_semantic(
            BASE_DIGEST,
            vec![operation(
                "overflow",
                &[1],
                PatchPayload::DenseDiff {
                    tensor: tensor(&[1], &[f32::MAX]),
                    pad_weight: false,
                },
            )],
        )?;
        let mut overflow_operation = overflow.semantic_operations()[0].clone();
        overflow_operation.strength = 2.0;
        let overflow = PatchGraph::checked_semantic(BASE_DIGEST, vec![overflow_operation])?;
        let source = mapped(&backend, &context, &[1], &[1.0])?;
        assert!(matches!(
            overflow.apply(&backend, &source, &context),
            Err(PatchGraphError::NonFiniteResult(_))
        ));
        assert_close(&values(&backend, &source, &context)?, &[1.0]);

        let cancellation = CancellationToken::default();
        let cancelling_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &cancellation,
        );
        let cancelling_backend = DelegatingBackend {
            backend: &backend,
            cancellation_on_binary: Some(&cancellation),
            reserve_calls: AtomicUsize::new(0),
            primitive_calls: AtomicUsize::new(0),
        };
        let cancellation_graph = PatchGraph::checked_semantic(
            BASE_DIGEST,
            vec![operation(
                "cancel-mid-compute",
                &[1],
                PatchPayload::DenseDiff {
                    tensor: tensor(&[1], &[1.0]),
                    pad_weight: false,
                },
            )],
        )?;
        assert!(matches!(
            cancellation_graph.apply(&cancelling_backend, &source, &cancelling_context),
            Err(PatchGraphError::Tensor(TensorError::Cancelled))
                | Err(PatchGraphError::Cancelled(_))
                | Err(PatchGraphError::TensorOperation(_))
        ));
        assert_eq!(cancelling_context.scratch.in_use_bytes(), 0);
        assert_close(&values(&backend, &source, &context)?, &[1.0]);

        let oom_cancellation = CancellationToken::default();
        let oom_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(3)?,
            &oom_cancellation,
        );
        let mut oom_operation = operation(
            "oom",
            &[1],
            PatchPayload::DenseDiff {
                tensor: tensor(&[1], &[1.0]),
                pad_weight: false,
            },
        );
        oom_operation.slices.push(PatchSlice {
            dimension: 0,
            start: 0,
            length: 1,
        });
        let oom_graph = PatchGraph::checked_semantic(BASE_DIGEST, vec![oom_operation])?;
        assert!(oom_graph.apply(&backend, &source, &oom_context).is_err());
        assert_eq!(oom_context.scratch.in_use_bytes(), 0);

        let non_cpu = IncompleteBackend::new()?;
        let source_bytes = weight_bytes(&source)?;
        let graph_identity = cancellation_graph.identity();
        assert!(matches!(
            cancellation_graph.apply(&non_cpu, &source, &context),
            Err(PatchGraphError::Tensor(TensorError::UnsupportedCapability {
                operation,
                device,
                ..
            })) if operation == "COMFY-TENSOR-OP-56B106D5BEE7"
                && device == DeviceId::new(DeviceKind::Metal, 0)
        ));
        assert_eq!(non_cpu.unexpected_calls.load(Ordering::Acquire), 0);
        assert_eq!(weight_bytes(&source)?, source_bytes);
        assert_eq!(cancellation_graph.identity(), graph_identity);

        let convergence_cancellation = CancellationToken::default();
        let convergence_context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(1024 * 1024)?,
            &convergence_cancellation,
        );
        oom_graph.apply(&backend, &source, &convergence_context)?;
        let peak = convergence_context.scratch.peak_bytes();
        assert!(peak > 0);
        assert_eq!(convergence_context.scratch.in_use_bytes(), 0);
        oom_graph.apply(&backend, &source, &convergence_context)?;
        assert_eq!(convergence_context.scratch.peak_bytes(), peak);
        assert_eq!(convergence_context.scratch.in_use_bytes(), 0);
        Ok(())
    }
}
