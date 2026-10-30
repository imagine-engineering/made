use crate::{
    CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DecodedScalar, DeviceId,
    ExecutionContext, Layout, StreamId, Tensor, TensorDescriptor, TensorError,
    cpu_backend::{binary_broadcast_shape, broadcast_indices},
    generated_elementwise_or_runtime_operation_20::{
        ElementwiseRuntimePartTwentyError,
        cross_jvp_with_context_exact_native as canonical_cross_jvp_with_context,
        cross_vjp_with_context_exact_native as canonical_cross_vjp_with_context,
        cross_with_context_exact_native as canonical_cross_with_context,
    },
};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const EINSUM_OPERATION_ID: &str = "COMFY-TENSOR-OP-061170CBB6F7";
pub const LINALG_CROSS_OPERATION_ID: &str = "COMFY-TENSOR-OP-4444EA894499";
pub const DETERMINANT_OPERATION_ID: &str = "COMFY-TENSOR-OP-1D21A20B5805";
pub const EIGH_OPERATION_ID: &str = "COMFY-TENSOR-OP-1B84B4F50448";
pub const INVERSE_OPERATION_ID: &str = "COMFY-TENSOR-OP-7DD46810B2C2";
pub const QR_OPERATION_ID: &str = "COMFY-TENSOR-OP-3FB914121F89";
pub const SOLVE_OPERATION_ID: &str = "COMFY-TENSOR-OP-93065313ABB0";
pub const VECTOR_NORM_OPERATION_ID: &str = "COMFY-TENSOR-OP-8E3FD7459720";
pub const MATMUL_OPERATION_ID: &str = "COMFY-TENSOR-OP-98D79FD6A7D2";
pub const MM_OPERATION_ID: &str = "COMFY-TENSOR-OP-277D4AF43E05";
pub const TENSORDOT_OPERATION_ID: &str = "COMFY-TENSOR-OP-2F913F6635CB";

#[derive(Debug, Error)]
pub enum LinearAlgebraPartOneError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Cross(#[from] ElementwiseRuntimePartTwentyError),
    #[error("linear-algebra part-one execution was cancelled")]
    Cancelled,
    #[error("operation {operation} requires CPU ordinal zero, got {device:?}")]
    UnsupportedDevice {
        operation: &'static str,
        device: DeviceId,
    },
    #[error("operation {operation} requires F32 tensors, got {dtype:?}")]
    UnsupportedDType {
        operation: &'static str,
        dtype: DType,
    },
    #[error("operation {operation} received invalid input: {reason}")]
    Invalid {
        operation: &'static str,
        reason: String,
    },
    #[error("shape arithmetic overflowed while computing {0}")]
    ShapeOverflow(&'static str),
    #[error("operation {operation} received a singular matrix")]
    Singular { operation: &'static str },
    #[error("operation {operation} did not converge")]
    DidNotConverge { operation: &'static str },
}

impl From<sim_types::CancellationError> for LinearAlgebraPartOneError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Clone, Debug)]
pub struct MatmulGradients {
    pub input: Tensor,
    pub other: Tensor,
}

#[derive(Clone, Debug)]
pub struct EinsumGradients {
    pub operands: Vec<Tensor>,
}

#[derive(Clone, Debug)]
pub struct TensorDotGradients {
    pub input: Tensor,
    pub other: Tensor,
}

#[derive(Clone, Debug)]
pub struct SolveGradients {
    pub coefficient: Tensor,
    pub right_hand_side: Tensor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QrMode {
    Reduced,
    Complete,
    R,
}

#[derive(Clone, Debug)]
pub struct QrOutput {
    pub q: Option<Tensor>,
    pub r: Tensor,
}

#[derive(Clone, Debug)]
pub struct EighOutput {
    pub eigenvalues: Tensor,
    pub eigenvectors: Tensor,
}

#[derive(Clone, Debug)]
struct RectangularMatrixBatch {
    batch_shape_u64: Vec<u64>,
    batch_count: usize,
    rows: usize,
    columns: usize,
    matrix_elements: usize,
}

impl RectangularMatrixBatch {
    fn new(input: &Tensor, operation: &'static str) -> Result<Self, LinearAlgebraPartOneError> {
        require_f32_cpu(input, operation)?;
        let shape = usize_shape(input.descriptor().shape(), "matrix shape")?;
        if shape.len() < 2 {
            return invalid(operation, "matrix input must have rank at least two");
        }
        let rows = shape[shape.len() - 2];
        let columns = shape[shape.len() - 1];
        let batch_shape = &shape[..shape.len() - 2];
        let batch_count = checked_product(batch_shape, "matrix batch count")?;
        let matrix_elements = rows
            .checked_mul(columns)
            .ok_or(LinearAlgebraPartOneError::ShapeOverflow("matrix elements"))?;
        Ok(Self {
            batch_shape_u64: input.descriptor().shape()[..shape.len() - 2].to_vec(),
            batch_count,
            rows,
            columns,
            matrix_elements,
        })
    }

    fn batch<'a>(
        &self,
        values: &'a [f64],
        batch: usize,
    ) -> Result<&'a [f64], LinearAlgebraPartOneError> {
        let start = batch.checked_mul(self.matrix_elements).ok_or(
            LinearAlgebraPartOneError::ShapeOverflow("matrix batch offset"),
        )?;
        let end = start
            .checked_add(self.matrix_elements)
            .ok_or(LinearAlgebraPartOneError::ShapeOverflow("matrix batch end"))?;
        values.get(start..end).ok_or_else(|| {
            invalid_error(
                EIGH_OPERATION_ID,
                "matrix storage is shorter than its descriptor",
            )
        })
    }
}

#[derive(Clone, Debug)]
struct SquareMatrixBatch {
    matrix: RectangularMatrixBatch,
    batch_shape_u64: Vec<u64>,
    batch_count: usize,
    size: usize,
}

impl SquareMatrixBatch {
    fn new(input: &Tensor, operation: &'static str) -> Result<Self, LinearAlgebraPartOneError> {
        let matrix = RectangularMatrixBatch::new(input, operation)?;
        if matrix.rows != matrix.columns {
            return invalid(operation, "matrix input must be square");
        }
        Ok(Self {
            batch_shape_u64: matrix.batch_shape_u64.clone(),
            batch_count: matrix.batch_count,
            size: matrix.rows,
            matrix,
        })
    }

    fn batch<'a>(
        &self,
        values: &'a [f64],
        batch: usize,
    ) -> Result<&'a [f64], LinearAlgebraPartOneError> {
        self.matrix.batch(values, batch)
    }
}

#[derive(Clone, Debug)]
struct SolveGeometry {
    matrix: SquareMatrixBatch,
    right_batch_shape_u64: Vec<u64>,
    output_batch_shape: Vec<usize>,
    output_batch_count: usize,
    output_shape: Vec<u64>,
    columns: usize,
    right_elements: usize,
}

impl SolveGeometry {
    fn new(
        coefficient: &Tensor,
        right_hand_side: &Tensor,
    ) -> Result<Self, LinearAlgebraPartOneError> {
        let matrix = SquareMatrixBatch::new(coefficient, SOLVE_OPERATION_ID)?;
        require_pair(coefficient, right_hand_side, SOLVE_OPERATION_ID)?;
        let coefficient_shape = coefficient.descriptor().shape();
        let right_shape = usize_shape(
            right_hand_side.descriptor().shape(),
            "solve right-hand side",
        )?;
        let coefficient_rank = coefficient_shape.len();
        let batch_rank = coefficient_rank - 2;
        let (right_batch, right_rows, columns, vector) =
            if right_shape.len() == coefficient_rank - 1 {
                (
                    &right_shape[..right_shape.len() - 1],
                    right_shape[right_shape.len() - 1],
                    1,
                    true,
                )
            } else if right_shape.len() >= 2 {
                (
                    &right_shape[..right_shape.len() - 2],
                    right_shape[right_shape.len() - 2],
                    right_shape[right_shape.len() - 1],
                    false,
                )
            } else {
                return invalid(
                    SOLVE_OPERATION_ID,
                    "right-hand side must be shaped [..., n] or [..., n, k]",
                );
            };
        if right_rows != matrix.size {
            return invalid(
                SOLVE_OPERATION_ID,
                "right-hand side row dimension must match the coefficient matrix",
            );
        }
        let right_batch_shape_u64 = right_batch
            .iter()
            .copied()
            .map(|extent| {
                u64::try_from(extent)
                    .map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("solve batch"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_batch_shape_u64 =
            binary_broadcast_shape(&coefficient_shape[..batch_rank], &right_batch_shape_u64)?;
        let output_batch_shape = usize_shape(&output_batch_shape_u64, "solve output batch")?;
        let output_batch_count = checked_product(&output_batch_shape, "solve output batch")?;
        let mut output_shape = output_batch_shape_u64;
        output_shape.push(
            u64::try_from(matrix.size)
                .map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("solve rows"))?,
        );
        if !vector {
            output_shape.push(
                u64::try_from(columns)
                    .map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("solve columns"))?,
            );
        }
        let right_elements =
            matrix
                .size
                .checked_mul(columns)
                .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
                    "solve right-hand-side batch",
                ))?;
        Ok(Self {
            matrix,
            right_batch_shape_u64,
            output_batch_shape,
            output_batch_count,
            output_shape,
            columns,
            right_elements,
        })
    }

    fn source_batch_index(
        &self,
        output_batch: usize,
        source_shape: &[u64],
    ) -> Result<usize, LinearAlgebraPartOneError> {
        let output_indices = unravel_index(output_batch, &self.output_batch_shape)?
            .into_iter()
            .map(|index| {
                u64::try_from(index)
                    .map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("solve batch index"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let source_indices = broadcast_indices(&output_indices, source_shape)?;
        let source_indices = source_indices
            .into_iter()
            .map(|index| {
                usize::try_from(index)
                    .map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("solve batch index"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        linear_index(
            &source_indices,
            &usize_shape(source_shape, "solve source batch")?,
            "solve source batch index",
        )
    }

    fn coefficient_batch_index(
        &self,
        output_batch: usize,
    ) -> Result<usize, LinearAlgebraPartOneError> {
        self.source_batch_index(output_batch, &self.matrix.batch_shape_u64)
    }

    fn right_batch_index(&self, output_batch: usize) -> Result<usize, LinearAlgebraPartOneError> {
        self.source_batch_index(output_batch, &self.right_batch_shape_u64)
    }

    fn right_batch<'a>(
        &self,
        values: &'a [f64],
        source_batch: usize,
    ) -> Result<&'a [f64], LinearAlgebraPartOneError> {
        let start = source_batch.checked_mul(self.right_elements).ok_or(
            LinearAlgebraPartOneError::ShapeOverflow("solve batch offset"),
        )?;
        let end = start
            .checked_add(self.right_elements)
            .ok_or(LinearAlgebraPartOneError::ShapeOverflow("solve batch end"))?;
        values.get(start..end).ok_or_else(|| {
            invalid_error(SOLVE_OPERATION_ID, "right-hand-side storage is too short")
        })
    }

    fn output_batch<'a>(
        &self,
        values: &'a [f64],
        output_batch: usize,
    ) -> Result<&'a [f64], LinearAlgebraPartOneError> {
        let start = output_batch.checked_mul(self.right_elements).ok_or(
            LinearAlgebraPartOneError::ShapeOverflow("solve output batch offset"),
        )?;
        let end = start.checked_add(self.right_elements).ok_or(
            LinearAlgebraPartOneError::ShapeOverflow("solve output batch end"),
        )?;
        values
            .get(start..end)
            .ok_or_else(|| invalid_error(SOLVE_OPERATION_ID, "solve output storage is too short"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EquationToken {
    Label(usize),
    Ellipsis,
}

#[derive(Clone, Debug)]
struct EinsumOperand {
    labels: Vec<usize>,
    shape: Vec<usize>,
}

#[derive(Clone, Debug)]
struct EinsumPlan {
    operands: Vec<EinsumOperand>,
    output_labels: Vec<usize>,
    label_extents: BTreeMap<usize, usize>,
    iteration_labels: Vec<usize>,
    iteration_shape: Vec<usize>,
    stream: StreamId,
    operation: &'static str,
}

impl EinsumPlan {
    fn from_equation(
        equation: &str,
        tensors: &[Tensor],
    ) -> Result<Self, LinearAlgebraPartOneError> {
        if tensors.is_empty() {
            return invalid(EINSUM_OPERATION_ID, "einsum requires at least one operand");
        }
        validate_operand_descriptors(tensors, EINSUM_OPERATION_ID)?;
        let mut equation_parts = equation.split("->");
        let input = equation_parts
            .next()
            .ok_or_else(|| invalid_error(EINSUM_OPERATION_ID, "einsum equation is empty"))?;
        let explicit_output = equation_parts.next();
        if equation_parts.next().is_some() {
            return invalid(
                EINSUM_OPERATION_ID,
                "einsum equation contains multiple arrows",
            );
        }
        let raw_operands = input
            .split(',')
            .map(|term| parse_equation_term(term, EINSUM_OPERATION_ID))
            .collect::<Result<Vec<_>, _>>()?;
        if raw_operands.len() != tensors.len() {
            return invalid(
                EINSUM_OPERATION_ID,
                "einsum equation operand count does not match tensors",
            );
        }
        let mut maximum_ellipsis = 0usize;
        for (tokens, tensor) in raw_operands.iter().zip(tensors) {
            let named = tokens
                .iter()
                .filter(|token| matches!(token, EquationToken::Label(_)))
                .count();
            let ellipses = tokens
                .iter()
                .filter(|token| **token == EquationToken::Ellipsis)
                .count();
            if ellipses > 1 || (ellipses == 0 && named != tensor.descriptor().rank()) {
                return invalid(
                    EINSUM_OPERATION_ID,
                    "einsum term rank does not match its tensor",
                );
            }
            if named > tensor.descriptor().rank() {
                return invalid(
                    EINSUM_OPERATION_ID,
                    "einsum term names more dimensions than its tensor has",
                );
            }
            maximum_ellipsis = maximum_ellipsis.max(tensor.descriptor().rank() - named);
        }
        let mut operands = Vec::with_capacity(tensors.len());
        let mut counts = BTreeMap::<usize, usize>::new();
        for (tokens, tensor) in raw_operands.into_iter().zip(tensors) {
            let named = tokens
                .iter()
                .filter(|token| matches!(token, EquationToken::Label(_)))
                .count();
            let ellipsis_rank = tensor.descriptor().rank() - named;
            let labels = expand_tokens(&tokens, ellipsis_rank, maximum_ellipsis)?;
            for label in &labels {
                *counts.entry(*label).or_default() += 1;
            }
            operands.push(EinsumOperand {
                labels,
                shape: usize_shape(tensor.descriptor().shape(), "einsum operand")?,
            });
        }
        let output_labels = if let Some(output) = explicit_output {
            let tokens = parse_equation_term(output, EINSUM_OPERATION_ID)?;
            let ellipsis_count = tokens
                .iter()
                .filter(|token| **token == EquationToken::Ellipsis)
                .count();
            if ellipsis_count > 1 {
                return invalid(
                    EINSUM_OPERATION_ID,
                    "einsum output contains multiple ellipses",
                );
            }
            let labels = expand_tokens(&tokens, maximum_ellipsis, maximum_ellipsis)?;
            let mut unique = BTreeSet::new();
            if labels
                .iter()
                .any(|label| !unique.insert(*label) || !counts.contains_key(label))
            {
                return invalid(
                    EINSUM_OPERATION_ID,
                    "einsum output labels must be unique input labels",
                );
            }
            labels
        } else {
            let mut labels = (52..52 + maximum_ellipsis).collect::<Vec<_>>();
            labels.extend(
                counts
                    .iter()
                    .filter_map(|(label, count)| (*label < 52 && *count == 1).then_some(*label)),
            );
            labels
        };
        Self::finish(
            operands,
            output_labels,
            tensors[0].descriptor().stream(),
            EINSUM_OPERATION_ID,
        )
    }

    fn from_tensordot(
        input: &Tensor,
        other: &Tensor,
        input_dimensions: &[i64],
        other_dimensions: &[i64],
    ) -> Result<Self, LinearAlgebraPartOneError> {
        validate_operand_descriptors(&[input.clone(), other.clone()], TENSORDOT_OPERATION_ID)?;
        if input_dimensions.len() != other_dimensions.len() {
            return invalid(
                TENSORDOT_OPERATION_ID,
                "tensordot dimension lists must have equal length",
            );
        }
        let left_shape = usize_shape(input.descriptor().shape(), "tensordot input")?;
        let right_shape = usize_shape(other.descriptor().shape(), "tensordot other")?;
        let left_axes =
            normalized_axes_in_order(left_shape.len(), input_dimensions, TENSORDOT_OPERATION_ID)?;
        let right_axes =
            normalized_axes_in_order(right_shape.len(), other_dimensions, TENSORDOT_OPERATION_ID)?;
        for (left_axis, right_axis) in left_axes.iter().zip(&right_axes) {
            if left_shape[*left_axis] != right_shape[*right_axis] {
                return invalid(
                    TENSORDOT_OPERATION_ID,
                    "tensordot contraction extents do not match",
                );
            }
        }
        let mut next_label = 0usize;
        let mut left_labels = vec![usize::MAX; left_shape.len()];
        let mut right_labels = vec![usize::MAX; right_shape.len()];
        for (left_axis, right_axis) in left_axes.iter().zip(&right_axes) {
            left_labels[*left_axis] = next_label;
            right_labels[*right_axis] = next_label;
            next_label += 1;
        }
        let mut output_labels = Vec::new();
        for label in &mut left_labels {
            if *label == usize::MAX {
                *label = next_label;
                output_labels.push(next_label);
                next_label += 1;
            }
        }
        for label in &mut right_labels {
            if *label == usize::MAX {
                *label = next_label;
                output_labels.push(next_label);
                next_label += 1;
            }
        }
        Self::finish(
            vec![
                EinsumOperand {
                    labels: left_labels,
                    shape: left_shape,
                },
                EinsumOperand {
                    labels: right_labels,
                    shape: right_shape,
                },
            ],
            output_labels,
            input.descriptor().stream(),
            TENSORDOT_OPERATION_ID,
        )
    }

    fn from_matmul(
        input: &Tensor,
        other: &Tensor,
        operation: &'static str,
    ) -> Result<Self, LinearAlgebraPartOneError> {
        validate_operand_descriptors(&[input.clone(), other.clone()], operation)?;
        let left_shape = usize_shape(input.descriptor().shape(), "matmul input")?;
        let right_shape = usize_shape(other.descriptor().shape(), "matmul other")?;
        if left_shape.is_empty() || right_shape.is_empty() {
            return invalid(
                operation,
                "matmul inputs must each have at least one dimension",
            );
        }
        let left_vector = left_shape.len() == 1;
        let right_vector = right_shape.len() == 1;
        let left_contracted = left_shape[left_shape.len() - 1];
        let right_contracted = if right_vector {
            right_shape[0]
        } else {
            right_shape[right_shape.len() - 2]
        };
        if left_contracted != right_contracted {
            return invalid(
                operation,
                "matmul contraction dimensions must match exactly",
            );
        }
        let left_batch_rank = left_shape
            .len()
            .saturating_sub(if left_vector { 1 } else { 2 });
        let right_batch_rank = right_shape
            .len()
            .saturating_sub(if right_vector { 1 } else { 2 });
        let batch_rank = left_batch_rank.max(right_batch_rank);
        let contracted_label = batch_rank;
        let row_label = batch_rank + 1;
        let column_label = batch_rank + 2;
        let mut left_labels = ((batch_rank - left_batch_rank)..batch_rank).collect::<Vec<_>>();
        if left_vector {
            left_labels.push(contracted_label);
        } else {
            left_labels.extend([row_label, contracted_label]);
        }
        let mut right_labels = ((batch_rank - right_batch_rank)..batch_rank).collect::<Vec<_>>();
        if right_vector {
            right_labels.push(contracted_label);
        } else {
            right_labels.extend([contracted_label, column_label]);
        }
        let mut output_labels = (0..batch_rank).collect::<Vec<_>>();
        if !left_vector {
            output_labels.push(row_label);
        }
        if !right_vector {
            output_labels.push(column_label);
        }
        Self::finish(
            vec![
                EinsumOperand {
                    labels: left_labels,
                    shape: left_shape,
                },
                EinsumOperand {
                    labels: right_labels,
                    shape: right_shape,
                },
            ],
            output_labels,
            input.descriptor().stream(),
            operation,
        )
    }

    fn finish(
        operands: Vec<EinsumOperand>,
        output_labels: Vec<usize>,
        stream: StreamId,
        operation: &'static str,
    ) -> Result<Self, LinearAlgebraPartOneError> {
        let mut label_extents = BTreeMap::<usize, usize>::new();
        for operand in &operands {
            if operand.labels.len() != operand.shape.len() {
                return invalid(operation, "contraction labels do not match operand rank");
            }
            let mut local = BTreeMap::<usize, usize>::new();
            for (label, extent) in operand.labels.iter().copied().zip(&operand.shape) {
                if let Some(previous) = local.insert(label, *extent)
                    && previous != *extent
                {
                    return invalid(operation, "repeated operand labels require equal extents");
                }
                let stored = label_extents.entry(label).or_insert(*extent);
                *stored = match (*stored, *extent) {
                    (left, right) if left == right => left,
                    (1, right) => right,
                    (left, 1) => left,
                    _ => {
                        return invalid(operation, "contraction label extents do not broadcast");
                    }
                };
            }
        }
        let mut iteration_labels = output_labels.clone();
        iteration_labels.extend(
            label_extents
                .keys()
                .filter(|label| !output_labels.contains(label))
                .copied(),
        );
        if iteration_labels.len() != label_extents.len() {
            return invalid(operation, "contraction output labels are invalid");
        }
        let iteration_shape = iteration_labels
            .iter()
            .map(|label| label_extents[label])
            .collect::<Vec<_>>();
        Ok(Self {
            operands,
            output_labels,
            label_extents,
            iteration_labels,
            iteration_shape,
            stream,
            operation,
        })
    }

    fn output_shape(&self) -> Vec<usize> {
        self.output_labels
            .iter()
            .map(|label| self.label_extents[label])
            .collect()
    }

    fn output_shape_u64(&self) -> Result<Vec<u64>, LinearAlgebraPartOneError> {
        self.output_shape()
            .into_iter()
            .map(|extent| {
                u64::try_from(extent)
                    .map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("contraction output"))
            })
            .collect()
    }

    fn output_count(&self) -> Result<usize, LinearAlgebraPartOneError> {
        checked_product(&self.output_shape(), "contraction output")
    }

    fn evaluate_into(
        &self,
        values: &[&[f64]],
        output: &mut [f64],
        cancellation: &CancellationToken,
    ) -> Result<(), LinearAlgebraPartOneError> {
        self.validate_storage(values)?;
        if output.len() != self.output_count()? {
            return invalid(self.operation, "contraction output length is invalid");
        }
        output.fill(0.0);
        let iteration_count = checked_product(&self.iteration_shape, "contraction iteration")?;
        for linear in 0..iteration_count {
            check_periodically(linear, cancellation)?;
            let assignment = unravel_index(linear, &self.iteration_shape)?;
            let output_index = linear_index(
                &assignment[..self.output_labels.len()],
                &self.output_shape(),
                "contraction output index",
            )?;
            let mut product = 1.0;
            for (operand, values) in self.operands.iter().zip(values) {
                product *= values[self.operand_index(operand, &assignment)?];
            }
            output[output_index] += product;
        }
        cancellation.check()?;
        Ok(())
    }

    fn gradients_into(
        &self,
        values: &[&[f64]],
        upstream: &[f64],
        gradients: &mut [&mut [f64]],
        cancellation: &CancellationToken,
    ) -> Result<(), LinearAlgebraPartOneError> {
        self.validate_storage(values)?;
        if upstream.len() != self.output_count()? {
            return invalid(self.operation, "contraction upstream length is invalid");
        }
        if gradients.len() != values.len()
            || gradients
                .iter()
                .zip(values)
                .any(|(gradient, values)| gradient.len() != values.len())
        {
            return invalid(self.operation, "contraction gradient storage is invalid");
        }
        for gradient in gradients.iter_mut() {
            gradient.fill(0.0);
        }
        let iteration_count =
            checked_product(&self.iteration_shape, "contraction gradient iteration")?;
        for linear in 0..iteration_count {
            check_periodically(linear, cancellation)?;
            let assignment = unravel_index(linear, &self.iteration_shape)?;
            let output_index = linear_index(
                &assignment[..self.output_labels.len()],
                &self.output_shape(),
                "contraction gradient output index",
            )?;
            for differentiated in 0..self.operands.len() {
                let target = self.operand_index(&self.operands[differentiated], &assignment)?;
                let mut contribution = upstream[output_index];
                for (index, operand) in self.operands.iter().enumerate() {
                    if index != differentiated {
                        contribution *= values[index][self.operand_index(operand, &assignment)?];
                    }
                }
                gradients[differentiated][target] += contribution;
            }
        }
        cancellation.check()?;
        Ok(())
    }

    fn operand_index(
        &self,
        operand: &EinsumOperand,
        assignment: &[usize],
    ) -> Result<usize, LinearAlgebraPartOneError> {
        let indices = operand
            .labels
            .iter()
            .zip(&operand.shape)
            .map(|(label, extent)| {
                let position = self
                    .iteration_labels
                    .iter()
                    .position(|candidate| candidate == label)
                    .ok_or_else(|| invalid_error(self.operation, "contraction label is missing"))?;
                Ok(if *extent == 1 {
                    0
                } else {
                    assignment[position]
                })
            })
            .collect::<Result<Vec<_>, LinearAlgebraPartOneError>>()?;
        linear_index(&indices, &operand.shape, "contraction operand index")
    }

    fn validate_storage(&self, values: &[&[f64]]) -> Result<(), LinearAlgebraPartOneError> {
        if values.len() != self.operands.len() {
            return invalid(
                self.operation,
                "contraction value count does not match operands",
            );
        }
        for (operand, values) in self.operands.iter().zip(values) {
            if checked_product(&operand.shape, "contraction operand")? != values.len() {
                return invalid(
                    self.operation,
                    "contraction operand storage length is invalid",
                );
            }
        }
        Ok(())
    }
}

pub fn linalg_cross_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    Ok(canonical_cross_with_context(
        backend,
        input,
        other,
        Some(dimension),
        context,
    )?)
}

pub fn linalg_cross_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    output_gradient: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<(Tensor, Tensor), LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    Ok(canonical_cross_vjp_with_context(
        backend,
        input,
        other,
        output_gradient,
        Some(dimension),
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn linalg_cross_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    input_tangent: &Tensor,
    other_tangent: &Tensor,
    dimension: i64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    Ok(canonical_cross_jvp_with_context(
        backend,
        input,
        other,
        input_tangent,
        other_tangent,
        Some(dimension),
        context,
    )?)
}

pub fn einsum_with_context_exact_native(
    backend: &CpuBackend,
    equation: &str,
    operands: &[Tensor],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    let plan = EinsumPlan::from_equation(equation, operands)?;
    let operand_values =
        read_operands_with_context(backend, operands, EINSUM_OPERATION_ID, context)?;
    let slices = operand_values
        .iter()
        .map(|values| values.as_ref())
        .collect::<Vec<&[f64]>>();
    let mut output = workspace_filled(backend, context, plan.output_count()?, 0.0_f64)?;
    plan.evaluate_into(&slices, &mut output, context.cancellation)?;
    drop(slices);
    drop(operand_values);
    upload_f64_with_context(
        backend,
        &plan.output_shape_u64()?,
        plan.stream,
        &output,
        context,
    )
}

pub fn einsum_vjp_with_context_exact_native(
    backend: &CpuBackend,
    equation: &str,
    operands: &[Tensor],
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<EinsumGradients, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    let plan = EinsumPlan::from_equation(equation, operands)?;
    let gradients = contraction_gradients_with_context(
        backend,
        &plan,
        operands,
        output_gradient,
        EINSUM_OPERATION_ID,
        context,
    )?;
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(operands.len())
        .map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("einsum gradient outputs"))?;
    for (operand, gradient) in operands.iter().zip(gradients) {
        outputs.push(upload_f64_with_context(
            backend,
            operand.descriptor().shape(),
            operand.descriptor().stream(),
            &gradient,
            context,
        )?);
    }
    Ok(EinsumGradients { operands: outputs })
}

pub fn einsum_jvp_with_context_exact_native(
    backend: &CpuBackend,
    equation: &str,
    operands: &[Tensor],
    operand_tangents: &[Tensor],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    let plan = EinsumPlan::from_equation(equation, operands)?;
    let output = contraction_jvp_with_context(
        backend,
        &plan,
        operands,
        operand_tangents,
        EINSUM_OPERATION_ID,
        context,
    )?;
    upload_f64_with_context(
        backend,
        &plan.output_shape_u64()?,
        plan.stream,
        &output,
        context,
    )
}

pub fn tensordot_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    input_dimensions: &[i64],
    other_dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    let plan = EinsumPlan::from_tensordot(input, other, input_dimensions, other_dimensions)?;
    let operands = [input.clone(), other.clone()];
    let values = read_operands_with_context(backend, &operands, TENSORDOT_OPERATION_ID, context)?;
    let slices = values
        .iter()
        .map(|values| values.as_ref())
        .collect::<Vec<&[f64]>>();
    let mut output = workspace_filled(backend, context, plan.output_count()?, 0.0_f64)?;
    plan.evaluate_into(&slices, &mut output, context.cancellation)?;
    drop(slices);
    drop(values);
    upload_f64_with_context(
        backend,
        &plan.output_shape_u64()?,
        plan.stream,
        &output,
        context,
    )
}

pub fn tensordot_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    output_gradient: &Tensor,
    input_dimensions: &[i64],
    other_dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<TensorDotGradients, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    let plan = EinsumPlan::from_tensordot(input, other, input_dimensions, other_dimensions)?;
    let operands = [input.clone(), other.clone()];
    let mut gradients = contraction_gradients_with_context(
        backend,
        &plan,
        &operands,
        output_gradient,
        TENSORDOT_OPERATION_ID,
        context,
    )?
    .into_iter();
    let input_values = gradients
        .next()
        .ok_or_else(|| invalid_error(TENSORDOT_OPERATION_ID, "missing input gradient"))?;
    let other_values = gradients
        .next()
        .ok_or_else(|| invalid_error(TENSORDOT_OPERATION_ID, "missing other gradient"))?;
    Ok(TensorDotGradients {
        input: upload_f64_with_context(
            backend,
            input.descriptor().shape(),
            input.descriptor().stream(),
            &input_values,
            context,
        )?,
        other: upload_f64_with_context(
            backend,
            other.descriptor().shape(),
            other.descriptor().stream(),
            &other_values,
            context,
        )?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn tensordot_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    input_tangent: &Tensor,
    other_tangent: &Tensor,
    input_dimensions: &[i64],
    other_dimensions: &[i64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    let plan = EinsumPlan::from_tensordot(input, other, input_dimensions, other_dimensions)?;
    let operands = [input.clone(), other.clone()];
    let tangents = [input_tangent.clone(), other_tangent.clone()];
    let output = contraction_jvp_with_context(
        backend,
        &plan,
        &operands,
        &tangents,
        TENSORDOT_OPERATION_ID,
        context,
    )?;
    upload_f64_with_context(
        backend,
        &plan.output_shape_u64()?,
        plan.stream,
        &output,
        context,
    )
}

pub fn determinant_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    let geometry = SquareMatrixBatch::new(input, DETERMINANT_OPERATION_ID)?;
    let values = tensor_f64_with_context(backend, input, DETERMINANT_OPERATION_ID, context)?;
    let mut determinants = workspace_filled(backend, context, geometry.batch_count, 0.0_f64)?;
    for (batch, determinant) in determinants.iter_mut().enumerate() {
        check_periodically(batch, context.cancellation)?;
        *determinant = determinant_value_with_context(
            backend,
            context,
            geometry.batch(&values, batch)?,
            geometry.size,
            DETERMINANT_OPERATION_ID,
        )?;
    }
    drop(values);
    upload_f64_with_context(
        backend,
        &geometry.batch_shape_u64,
        input.descriptor().stream(),
        &determinants,
        context,
    )
}

pub fn determinant_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    let geometry = SquareMatrixBatch::new(input, DETERMINANT_OPERATION_ID)?;
    require_f32_cpu(output_gradient, DETERMINANT_OPERATION_ID)?;
    require_stream(input, output_gradient, DETERMINANT_OPERATION_ID)?;
    if output_gradient.descriptor().shape() != geometry.batch_shape_u64 {
        return invalid(
            DETERMINANT_OPERATION_ID,
            "determinant output gradient shape does not match",
        );
    }
    let values = tensor_f64_with_context(backend, input, DETERMINANT_OPERATION_ID, context)?;
    let upstream =
        tensor_f64_with_context(backend, output_gradient, DETERMINANT_OPERATION_ID, context)?;
    let mut gradients = workspace_filled(backend, context, values.len(), 0.0_f64)?;
    for batch in 0..geometry.batch_count {
        check_periodically(batch, context.cancellation)?;
        let mut batch_gradient = determinant_gradient_matrix_with_context(
            backend,
            context,
            geometry.batch(&values, batch)?,
            geometry.size,
        )?;
        let start = batch.checked_mul(geometry.matrix.matrix_elements).ok_or(
            LinearAlgebraPartOneError::ShapeOverflow("determinant VJP offset"),
        )?;
        let end = start.checked_add(geometry.matrix.matrix_elements).ok_or(
            LinearAlgebraPartOneError::ShapeOverflow("determinant VJP end"),
        )?;
        let output =
            gradients
                .get_mut(start..end)
                .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
                    "determinant VJP batch",
                ))?;
        for (output, value) in output.iter_mut().zip(batch_gradient.iter_mut()) {
            *output = *value * upstream[batch];
        }
    }
    drop(upstream);
    drop(values);
    upload_f64_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &gradients,
        context,
    )
}

pub fn determinant_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    require_matching(input, input_tangent, DETERMINANT_OPERATION_ID)?;
    let geometry = SquareMatrixBatch::new(input, DETERMINANT_OPERATION_ID)?;
    let values = tensor_f64_with_context(backend, input, DETERMINANT_OPERATION_ID, context)?;
    let tangents =
        tensor_f64_with_context(backend, input_tangent, DETERMINANT_OPERATION_ID, context)?;
    let mut output = workspace_filled(backend, context, geometry.batch_count, 0.0_f64)?;
    for (batch, output_value) in output.iter_mut().enumerate() {
        check_periodically(batch, context.cancellation)?;
        let gradient = determinant_gradient_matrix_with_context(
            backend,
            context,
            geometry.batch(&values, batch)?,
            geometry.size,
        )?;
        let tangent = geometry.batch(&tangents, batch)?;
        *output_value = gradient
            .iter()
            .zip(tangent)
            .map(|(left, right)| left * right)
            .sum();
    }
    drop(tangents);
    drop(values);
    upload_f64_with_context(
        backend,
        &geometry.batch_shape_u64,
        input.descriptor().stream(),
        &output,
        context,
    )
}

pub fn inverse_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    let geometry = SquareMatrixBatch::new(input, INVERSE_OPERATION_ID)?;
    let values = tensor_f64_with_context(backend, input, INVERSE_OPERATION_ID, context)?;
    let mut output = workspace_filled(backend, context, values.len(), 0.0_f64)?;
    for batch in 0..geometry.batch_count {
        check_periodically(batch, context.cancellation)?;
        let factor = lu_factor_with_context(
            backend,
            geometry.batch(&values, batch)?,
            geometry.size,
            INVERSE_OPERATION_ID,
            context,
        )?;
        let matrix_elements = geometry.matrix.matrix_elements;
        let mut identity = workspace_filled(backend, context, matrix_elements, 0.0_f64)?;
        for index in 0..geometry.size {
            identity[index * geometry.size + index] = 1.0;
        }
        let start =
            batch
                .checked_mul(matrix_elements)
                .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
                    "inverse output offset",
                ))?;
        let end =
            start
                .checked_add(matrix_elements)
                .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
                    "inverse output end",
                ))?;
        factor.solve_into(
            &identity,
            geometry.size,
            output
                .get_mut(start..end)
                .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
                    "inverse output batch",
                ))?,
            context.cancellation,
        )?;
    }
    drop(values);
    upload_f64_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &output,
        context,
    )
}

pub fn inverse_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    require_matching(input, input_tangent, INVERSE_OPERATION_ID)?;
    let inverse = inverse_with_context_exact_native(backend, input, context)?;
    let geometry = SquareMatrixBatch::new(input, INVERSE_OPERATION_ID)?;
    let inverse_values = tensor_f64_with_context(backend, &inverse, INVERSE_OPERATION_ID, context)?;
    let tangents = tensor_f64_with_context(backend, input_tangent, INVERSE_OPERATION_ID, context)?;
    let mut output = workspace_filled(backend, context, tangents.len(), 0.0_f64)?;
    let mut left = workspace_filled(backend, context, geometry.matrix.matrix_elements, 0.0_f64)?;
    for batch in 0..geometry.batch_count {
        check_periodically(batch, context.cancellation)?;
        let inverse = geometry.batch(&inverse_values, batch)?;
        let tangent = geometry.batch(&tangents, batch)?;
        matrix_product_into(
            inverse,
            tangent,
            geometry.size,
            geometry.size,
            geometry.size,
            &mut left,
            context.cancellation,
        )?;
        let start = batch.checked_mul(geometry.matrix.matrix_elements).ok_or(
            LinearAlgebraPartOneError::ShapeOverflow("inverse JVP offset"),
        )?;
        let end = start
            .checked_add(geometry.matrix.matrix_elements)
            .ok_or(LinearAlgebraPartOneError::ShapeOverflow("inverse JVP end"))?;
        let batch_output =
            output
                .get_mut(start..end)
                .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
                    "inverse JVP batch",
                ))?;
        matrix_product_into(
            &left,
            inverse,
            geometry.size,
            geometry.size,
            geometry.size,
            batch_output,
            context.cancellation,
        )?;
        for value in batch_output {
            *value = -*value;
        }
    }
    drop(left);
    drop(tangents);
    drop(inverse_values);
    upload_f64_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &output,
        context,
    )
}

pub fn inverse_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    require_matching(input, output_gradient, INVERSE_OPERATION_ID)?;
    let inverse = inverse_with_context_exact_native(backend, input, context)?;
    let geometry = SquareMatrixBatch::new(input, INVERSE_OPERATION_ID)?;
    let inverse_values = tensor_f64_with_context(backend, &inverse, INVERSE_OPERATION_ID, context)?;
    let upstream =
        tensor_f64_with_context(backend, output_gradient, INVERSE_OPERATION_ID, context)?;
    let mut output = workspace_filled(backend, context, upstream.len(), 0.0_f64)?;
    let mut inverse_transpose =
        workspace_filled(backend, context, geometry.matrix.matrix_elements, 0.0_f64)?;
    let mut left = workspace_filled(backend, context, geometry.matrix.matrix_elements, 0.0_f64)?;
    for batch in 0..geometry.batch_count {
        check_periodically(batch, context.cancellation)?;
        transpose_matrix_into(
            geometry.batch(&inverse_values, batch)?,
            geometry.size,
            geometry.size,
            &mut inverse_transpose,
            context.cancellation,
        )?;
        matrix_product_into(
            &inverse_transpose,
            geometry.batch(&upstream, batch)?,
            geometry.size,
            geometry.size,
            geometry.size,
            &mut left,
            context.cancellation,
        )?;
        let start = batch.checked_mul(geometry.matrix.matrix_elements).ok_or(
            LinearAlgebraPartOneError::ShapeOverflow("inverse VJP offset"),
        )?;
        let end = start
            .checked_add(geometry.matrix.matrix_elements)
            .ok_or(LinearAlgebraPartOneError::ShapeOverflow("inverse VJP end"))?;
        let batch_output =
            output
                .get_mut(start..end)
                .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
                    "inverse VJP batch",
                ))?;
        matrix_product_into(
            &left,
            &inverse_transpose,
            geometry.size,
            geometry.size,
            geometry.size,
            batch_output,
            context.cancellation,
        )?;
        for value in batch_output {
            *value = -*value;
        }
    }
    drop(left);
    drop(inverse_transpose);
    drop(upstream);
    drop(inverse_values);
    upload_f64_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &output,
        context,
    )
}

pub fn solve_with_context_exact_native(
    backend: &CpuBackend,
    coefficient: &Tensor,
    right_hand_side: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    if coefficient.descriptor().stream() != context.stream {
        return Err(TensorError::StreamMismatch {
            expected: context.stream,
            actual: coefficient.descriptor().stream(),
        }
        .into());
    }
    let geometry = SolveGeometry::new(coefficient, right_hand_side)?;
    let coefficients = tensor_f64_with_context(backend, coefficient, SOLVE_OPERATION_ID, context)?;
    let right = tensor_f64_with_context(backend, right_hand_side, SOLVE_OPERATION_ID, context)?;
    let output_count = geometry
        .output_batch_count
        .checked_mul(geometry.right_elements)
        .ok_or(LinearAlgebraPartOneError::ShapeOverflow("solve output"))?;
    let mut output = workspace_filled(backend, context, output_count, 0.0_f64)?;
    for output_batch in 0..geometry.output_batch_count {
        check_periodically(output_batch, context.cancellation)?;
        let coefficient_batch = geometry.coefficient_batch_index(output_batch)?;
        let right_batch = geometry.right_batch_index(output_batch)?;
        let factor = lu_factor_with_context(
            backend,
            geometry.matrix.batch(&coefficients, coefficient_batch)?,
            geometry.matrix.size,
            SOLVE_OPERATION_ID,
            context,
        )?;
        let output_start = output_batch.checked_mul(geometry.right_elements).ok_or(
            LinearAlgebraPartOneError::ShapeOverflow("solve output offset"),
        )?;
        let output_end = output_start
            .checked_add(geometry.right_elements)
            .ok_or(LinearAlgebraPartOneError::ShapeOverflow("solve output end"))?;
        let output_batch_values = output.get_mut(output_start..output_end).ok_or(
            LinearAlgebraPartOneError::ShapeOverflow("solve output batch"),
        )?;
        factor.solve_into(
            geometry.right_batch(&right, right_batch)?,
            geometry.columns,
            output_batch_values,
            context.cancellation,
        )?;
    }
    drop(right);
    drop(coefficients);
    upload_f64_with_context(
        backend,
        &geometry.output_shape,
        coefficient.descriptor().stream(),
        &output,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn solve_jvp_with_context_exact_native(
    backend: &CpuBackend,
    coefficient: &Tensor,
    right_hand_side: &Tensor,
    coefficient_tangent: &Tensor,
    right_hand_side_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    require_matching(coefficient, coefficient_tangent, SOLVE_OPERATION_ID)?;
    require_matching(right_hand_side, right_hand_side_tangent, SOLVE_OPERATION_ID)?;
    let geometry = SolveGeometry::new(coefficient, right_hand_side)?;
    let solution = solve_with_context_exact_native(backend, coefficient, right_hand_side, context)?;
    let coefficients = tensor_f64_with_context(backend, coefficient, SOLVE_OPERATION_ID, context)?;
    let coefficient_tangents =
        tensor_f64_with_context(backend, coefficient_tangent, SOLVE_OPERATION_ID, context)?;
    let right_tangents = tensor_f64_with_context(
        backend,
        right_hand_side_tangent,
        SOLVE_OPERATION_ID,
        context,
    )?;
    let solutions = tensor_f64_with_context(backend, &solution, SOLVE_OPERATION_ID, context)?;
    let output_count = geometry
        .output_batch_count
        .checked_mul(geometry.right_elements)
        .ok_or(LinearAlgebraPartOneError::ShapeOverflow("solve JVP output"))?;
    let mut output = workspace_filled(backend, context, output_count, 0.0_f64)?;
    for output_batch in 0..geometry.output_batch_count {
        check_periodically(output_batch, context.cancellation)?;
        let coefficient_batch = geometry.coefficient_batch_index(output_batch)?;
        let right_batch = geometry.right_batch_index(output_batch)?;
        let factor = lu_factor_with_context(
            backend,
            geometry.matrix.batch(&coefficients, coefficient_batch)?,
            geometry.matrix.size,
            SOLVE_OPERATION_ID,
            context,
        )?;
        let mut adjusted = workspace_filled(backend, context, geometry.right_elements, 0.0_f64)?;
        matrix_product_into(
            geometry
                .matrix
                .batch(&coefficient_tangents, coefficient_batch)?,
            geometry.output_batch(&solutions, output_batch)?,
            geometry.matrix.size,
            geometry.matrix.size,
            geometry.columns,
            &mut adjusted,
            context.cancellation,
        )?;
        for (adjusted, right) in adjusted
            .iter_mut()
            .zip(geometry.right_batch(&right_tangents, right_batch)?)
        {
            *adjusted = *right - *adjusted;
        }
        let output_start = output_batch
            .checked_mul(geometry.right_elements)
            .ok_or(LinearAlgebraPartOneError::ShapeOverflow("solve JVP offset"))?;
        let output_end = output_start
            .checked_add(geometry.right_elements)
            .ok_or(LinearAlgebraPartOneError::ShapeOverflow("solve JVP end"))?;
        factor.solve_into(
            &adjusted,
            geometry.columns,
            output.get_mut(output_start..output_end).ok_or(
                LinearAlgebraPartOneError::ShapeOverflow("solve JVP output batch"),
            )?,
            context.cancellation,
        )?;
    }
    drop(solutions);
    drop(right_tangents);
    drop(coefficient_tangents);
    drop(coefficients);
    upload_f64_with_context(
        backend,
        &geometry.output_shape,
        coefficient.descriptor().stream(),
        &output,
        context,
    )
}

pub fn solve_vjp_with_context_exact_native(
    backend: &CpuBackend,
    coefficient: &Tensor,
    right_hand_side: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<SolveGradients, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    let geometry = SolveGeometry::new(coefficient, right_hand_side)?;
    let solution = solve_with_context_exact_native(backend, coefficient, right_hand_side, context)?;
    require_matching(&solution, output_gradient, SOLVE_OPERATION_ID)?;
    let coefficients = tensor_f64_with_context(backend, coefficient, SOLVE_OPERATION_ID, context)?;
    let solutions = tensor_f64_with_context(backend, &solution, SOLVE_OPERATION_ID, context)?;
    let upstream = tensor_f64_with_context(backend, output_gradient, SOLVE_OPERATION_ID, context)?;
    let right_count = usize::try_from(right_hand_side.descriptor().element_count()?)
        .map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("solve VJP right gradient"))?;
    let mut coefficient_gradient = workspace_filled(backend, context, coefficients.len(), 0.0_f64)?;
    let mut right_gradient = workspace_filled(backend, context, right_count, 0.0_f64)?;
    for output_batch in 0..geometry.output_batch_count {
        check_periodically(output_batch, context.cancellation)?;
        let coefficient_batch = geometry.coefficient_batch_index(output_batch)?;
        let right_batch_index = geometry.right_batch_index(output_batch)?;
        let matrix_elements = geometry.matrix.matrix.matrix_elements;
        let mut coefficient_transpose =
            workspace_filled(backend, context, matrix_elements, 0.0_f64)?;
        transpose_matrix_into(
            geometry.matrix.batch(&coefficients, coefficient_batch)?,
            geometry.matrix.size,
            geometry.matrix.size,
            &mut coefficient_transpose,
            context.cancellation,
        )?;
        let factor = lu_factor_with_context(
            backend,
            &coefficient_transpose,
            geometry.matrix.size,
            SOLVE_OPERATION_ID,
            context,
        )?;
        let mut right_batch = workspace_filled(backend, context, geometry.right_elements, 0.0_f64)?;
        factor.solve_into(
            geometry.output_batch(&upstream, output_batch)?,
            geometry.columns,
            &mut right_batch,
            context.cancellation,
        )?;
        let mut solution_transpose =
            workspace_filled(backend, context, geometry.right_elements, 0.0_f64)?;
        transpose_matrix_into(
            geometry.output_batch(&solutions, output_batch)?,
            geometry.matrix.size,
            geometry.columns,
            &mut solution_transpose,
            context.cancellation,
        )?;
        let mut matrix_gradient = workspace_filled(backend, context, matrix_elements, 0.0_f64)?;
        matrix_product_into(
            &right_batch,
            &solution_transpose,
            geometry.matrix.size,
            geometry.columns,
            geometry.matrix.size,
            &mut matrix_gradient,
            context.cancellation,
        )?;
        let coefficient_offset = coefficient_batch.checked_mul(matrix_elements).ok_or(
            LinearAlgebraPartOneError::ShapeOverflow("solve coefficient gradient"),
        )?;
        for (offset, value) in matrix_gradient.iter().copied().enumerate() {
            coefficient_gradient[coefficient_offset + offset] -= value;
        }
        let right_offset = right_batch_index
            .checked_mul(geometry.right_elements)
            .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
                "solve right gradient",
            ))?;
        for (offset, value) in right_batch.iter().copied().enumerate() {
            right_gradient[right_offset + offset] += value;
        }
    }
    drop(upstream);
    drop(solutions);
    drop(coefficients);
    let coefficient_output = upload_f64_with_context(
        backend,
        coefficient.descriptor().shape(),
        coefficient.descriptor().stream(),
        &coefficient_gradient,
        context,
    )?;
    drop(coefficient_gradient);
    let right_output = upload_f64_with_context(
        backend,
        right_hand_side.descriptor().shape(),
        right_hand_side.descriptor().stream(),
        &right_gradient,
        context,
    )?;
    Ok(SolveGradients {
        coefficient: coefficient_output,
        right_hand_side: right_output,
    })
}

pub fn qr_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    mode: QrMode,
    context: &ExecutionContext<'_>,
) -> Result<QrOutput, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    require_f32_cpu(input, QR_OPERATION_ID)?;
    let geometry = RectangularMatrixBatch::new(input, QR_OPERATION_ID)?;
    let values = tensor_f64_with_context(backend, input, QR_OPERATION_ID, context)?;
    let reduced = geometry.rows.min(geometry.columns);
    let q_columns = if mode == QrMode::Complete {
        geometry.rows
    } else {
        reduced
    };
    let r_rows = if mode == QrMode::Complete {
        geometry.rows
    } else {
        reduced
    };
    let q_batch_elements = geometry
        .rows
        .checked_mul(q_columns)
        .ok_or(LinearAlgebraPartOneError::ShapeOverflow("QR Q matrix"))?;
    let r_batch_elements = r_rows
        .checked_mul(geometry.columns)
        .ok_or(LinearAlgebraPartOneError::ShapeOverflow("QR R matrix"))?;
    let q_count = geometry
        .batch_count
        .checked_mul(q_batch_elements)
        .ok_or(LinearAlgebraPartOneError::ShapeOverflow("QR Q batches"))?;
    let r_count = geometry
        .batch_count
        .checked_mul(r_batch_elements)
        .ok_or(LinearAlgebraPartOneError::ShapeOverflow("QR R batches"))?;
    let mut q_values = workspace_filled(backend, context, q_count, 0.0_f64)?;
    let mut r_values = workspace_filled(backend, context, r_count, 0.0_f64)?;
    for batch in 0..geometry.batch_count {
        check_periodically(batch, context.cancellation)?;
        let mut candidate = workspace_filled(backend, context, geometry.rows, 0.0_f64)?;
        let mut basis = workspace_filled(backend, context, geometry.rows, 0.0_f64)?;
        let q_start = batch
            .checked_mul(q_batch_elements)
            .ok_or(LinearAlgebraPartOneError::ShapeOverflow("QR Q offset"))?;
        let q_end = q_start
            .checked_add(q_batch_elements)
            .ok_or(LinearAlgebraPartOneError::ShapeOverflow("QR Q end"))?;
        let r_start = batch
            .checked_mul(r_batch_elements)
            .ok_or(LinearAlgebraPartOneError::ShapeOverflow("QR R offset"))?;
        let r_end = r_start
            .checked_add(r_batch_elements)
            .ok_or(LinearAlgebraPartOneError::ShapeOverflow("QR R end"))?;
        modified_gram_schmidt_into(
            geometry.batch(&values, batch)?,
            geometry.rows,
            geometry.columns,
            q_columns,
            r_rows,
            q_values
                .get_mut(q_start..q_end)
                .ok_or(LinearAlgebraPartOneError::ShapeOverflow("QR Q batch"))?,
            r_values
                .get_mut(r_start..r_end)
                .ok_or(LinearAlgebraPartOneError::ShapeOverflow("QR R batch"))?,
            &mut candidate,
            &mut basis,
            context.cancellation,
        )?;
    }
    drop(values);
    let mut q_shape = geometry.batch_shape_u64.clone();
    q_shape.extend([
        u64::try_from(geometry.rows)
            .map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("QR Q rows"))?,
        u64::try_from(q_columns)
            .map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("QR Q columns"))?,
    ]);
    let mut r_shape = geometry.batch_shape_u64.clone();
    r_shape.extend([
        u64::try_from(r_rows).map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("QR R rows"))?,
        u64::try_from(geometry.columns)
            .map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("QR R columns"))?,
    ]);
    let q = if mode == QrMode::R {
        drop(q_values);
        None
    } else {
        let output = upload_f64_with_context(
            backend,
            &q_shape,
            input.descriptor().stream(),
            &q_values,
            context,
        )?;
        drop(q_values);
        Some(output)
    };
    let r = upload_f64_with_context(
        backend,
        &r_shape,
        input.descriptor().stream(),
        &r_values,
        context,
    )?;
    Ok(QrOutput { q, r })
}

pub fn eigh_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    use_upper_triangle: bool,
    context: &ExecutionContext<'_>,
) -> Result<EighOutput, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    let geometry = SquareMatrixBatch::new(input, EIGH_OPERATION_ID)?;
    let values = tensor_f64_with_context(backend, input, EIGH_OPERATION_ID, context)?;
    let eigenvalue_count = geometry
        .batch_count
        .checked_mul(geometry.size)
        .ok_or(LinearAlgebraPartOneError::ShapeOverflow("eigh eigenvalues"))?;
    let mut eigenvalues = workspace_filled(backend, context, eigenvalue_count, 0.0_f64)?;
    let mut eigenvectors = workspace_filled(backend, context, values.len(), 0.0_f64)?;
    let matrix_elements = geometry.matrix.matrix_elements;
    for batch in 0..geometry.batch_count {
        check_periodically(batch, context.cancellation)?;
        let mut symmetric = workspace_filled(backend, context, matrix_elements, 0.0_f64)?;
        triangle_symmetric_into(
            geometry.batch(&values, batch)?,
            geometry.size,
            use_upper_triangle,
            &mut symmetric,
            context.cancellation,
        )?;
        let mut work_values = workspace_filled(backend, context, matrix_elements, 0.0_f64)?;
        let mut work_vectors = workspace_filled(backend, context, matrix_elements, 0.0_f64)?;
        let mut order = workspace_filled(backend, context, geometry.size, 0_usize)?;
        let eigenvalue_start =
            batch
                .checked_mul(geometry.size)
                .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
                    "eigh eigenvalue offset",
                ))?;
        let eigenvalue_end = eigenvalue_start.checked_add(geometry.size).ok_or(
            LinearAlgebraPartOneError::ShapeOverflow("eigh eigenvalue end"),
        )?;
        let eigenvector_start =
            batch
                .checked_mul(matrix_elements)
                .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
                    "eigh eigenvector offset",
                ))?;
        let eigenvector_end = eigenvector_start.checked_add(matrix_elements).ok_or(
            LinearAlgebraPartOneError::ShapeOverflow("eigh eigenvector end"),
        )?;
        symmetric_eigen_decomposition_into(
            &symmetric,
            geometry.size,
            &mut work_values,
            &mut work_vectors,
            &mut order,
            eigenvalues
                .get_mut(eigenvalue_start..eigenvalue_end)
                .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
                    "eigh eigenvalue batch",
                ))?,
            eigenvectors
                .get_mut(eigenvector_start..eigenvector_end)
                .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
                    "eigh eigenvector batch",
                ))?,
            context.cancellation,
        )?;
    }
    drop(values);
    let mut values_shape = geometry.batch_shape_u64.clone();
    values_shape.push(
        u64::try_from(geometry.size)
            .map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("eigh values"))?,
    );
    let eigenvalue_tensor = upload_f64_with_context(
        backend,
        &values_shape,
        input.descriptor().stream(),
        &eigenvalues,
        context,
    )?;
    drop(eigenvalues);
    let eigenvector_tensor = upload_f64_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &eigenvectors,
        context,
    )?;
    Ok(EighOutput {
        eigenvalues: eigenvalue_tensor,
        eigenvectors: eigenvector_tensor,
    })
}

pub fn matmul_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    let plan = EinsumPlan::from_matmul(input, other, MATMUL_OPERATION_ID)?;
    let operands = [input.clone(), other.clone()];
    let values = read_operands_with_context(backend, &operands, MATMUL_OPERATION_ID, context)?;
    let slices = values
        .iter()
        .map(|values| values.as_ref())
        .collect::<Vec<&[f64]>>();
    let mut output = workspace_filled(backend, context, plan.output_count()?, 0.0_f64)?;
    plan.evaluate_into(&slices, &mut output, context.cancellation)?;
    drop(slices);
    drop(values);
    upload_f64_with_context(
        backend,
        &plan.output_shape_u64()?,
        plan.stream,
        &output,
        context,
    )
}

pub fn matmul_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<MatmulGradients, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    let plan = EinsumPlan::from_matmul(input, other, MATMUL_OPERATION_ID)?;
    let operands = [input.clone(), other.clone()];
    let mut gradients = contraction_gradients_with_context(
        backend,
        &plan,
        &operands,
        output_gradient,
        MATMUL_OPERATION_ID,
        context,
    )?
    .into_iter();
    let input_values = gradients
        .next()
        .ok_or_else(|| invalid_error(MATMUL_OPERATION_ID, "missing input gradient"))?;
    let other_values = gradients
        .next()
        .ok_or_else(|| invalid_error(MATMUL_OPERATION_ID, "missing other gradient"))?;
    Ok(MatmulGradients {
        input: upload_f64_with_context(
            backend,
            input.descriptor().shape(),
            input.descriptor().stream(),
            &input_values,
            context,
        )?,
        other: upload_f64_with_context(
            backend,
            other.descriptor().shape(),
            other.descriptor().stream(),
            &other_values,
            context,
        )?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    other: &Tensor,
    input_tangent: &Tensor,
    other_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    let plan = EinsumPlan::from_matmul(input, other, MATMUL_OPERATION_ID)?;
    let operands = [input.clone(), other.clone()];
    let tangents = [input_tangent.clone(), other_tangent.clone()];
    let output = contraction_jvp_with_context(
        backend,
        &plan,
        &operands,
        &tangents,
        MATMUL_OPERATION_ID,
        context,
    )?;
    upload_f64_with_context(
        backend,
        &plan.output_shape_u64()?,
        plan.stream,
        &output,
        context,
    )
}

pub fn mm_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    mat2: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    if input.descriptor().rank() != 2 || mat2.descriptor().rank() != 2 {
        return invalid(MM_OPERATION_ID, "mm requires exactly two rank-two tensors");
    }
    matmul_with_context_exact_native(backend, input, mat2, context)
}

pub fn mm_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    mat2: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<MatmulGradients, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    if input.descriptor().rank() != 2 || mat2.descriptor().rank() != 2 {
        return invalid(MM_OPERATION_ID, "mm requires exactly two rank-two tensors");
    }
    matmul_vjp_with_context_exact_native(backend, input, mat2, output_gradient, context)
}

#[allow(clippy::too_many_arguments)]
pub fn mm_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    mat2: &Tensor,
    input_tangent: &Tensor,
    mat2_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    if input.descriptor().rank() != 2 || mat2.descriptor().rank() != 2 {
        return invalid(MM_OPERATION_ID, "mm requires exactly two rank-two tensors");
    }
    matmul_jvp_with_context_exact_native(backend, input, mat2, input_tangent, mat2_tangent, context)
}

pub fn transpose_last_two_with_context_exact_native(
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    if input.descriptor().rank() < 2 {
        return invalid(MATMUL_OPERATION_ID, "transpose requires rank at least two");
    }
    let mut shape = input.descriptor().shape().to_vec();
    let mut strides = input.descriptor().strides().to_vec();
    let rank = shape.len();
    shape.swap(rank - 2, rank - 1);
    strides.swap(rank - 2, rank - 1);
    let descriptor = TensorDescriptor::new_strided(
        shape,
        strides,
        input.descriptor().offset_elements(),
        input.descriptor().dtype(),
        Layout::Strided,
        input.descriptor().device(),
        input.descriptor().stream(),
    )?;
    let output = input.reinterpret_read_only(descriptor)?;
    context.cancellation.check()?;
    Ok(output)
}

pub fn vector_norm_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    order: f64,
    dimensions: &[i64],
    keep_dimension: bool,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    require_f32_cpu(input, VECTOR_NORM_OPERATION_ID)?;
    if order.is_nan() {
        return invalid(VECTOR_NORM_OPERATION_ID, "norm order must not be NaN");
    }
    if dtype.is_some_and(|dtype| dtype != DType::F32) {
        return Err(LinearAlgebraPartOneError::UnsupportedDType {
            operation: VECTOR_NORM_OPERATION_ID,
            dtype: dtype.unwrap_or(DType::F32),
        });
    }
    let axes = normalized_axes(
        input.descriptor().rank(),
        dimensions,
        VECTOR_NORM_OPERATION_ID,
    )?;
    let geometry = ReductionGeometry::new(input.descriptor().shape(), &axes, keep_dimension)?;
    let input_values = tensor_f64_with_context(backend, input, VECTOR_NORM_OPERATION_ID, context)?;
    let initial = if order == f64::NEG_INFINITY {
        f64::INFINITY
    } else {
        0.0
    };
    let mut output = workspace_filled(backend, context, geometry.output_count()?, initial)?;
    for (linear, value) in input_values.iter().copied().enumerate() {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &geometry.input_shape)?;
        let output_index = geometry.output_index(&indices)?;
        update_norm_accumulator(&mut output[output_index], value, order);
    }
    for value in output.iter_mut() {
        *value = finish_norm_accumulator(*value, order);
    }
    drop(input_values);
    upload_f64_with_context(
        backend,
        &geometry.output_shape,
        input.descriptor().stream(),
        &output,
        context,
    )
}

pub(crate) fn optional_vector_norm_dimensions(
    rank: usize,
    dimensions: Option<&[i64]>,
) -> Result<Vec<i64>, LinearAlgebraPartOneError> {
    match dimensions {
        Some(dimensions) => Ok(dimensions.to_vec()),
        None => (0..rank)
            .map(|dimension| {
                i64::try_from(dimension)
                    .map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("vector norm dimension"))
            })
            .collect(),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn vector_norm_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    order: f64,
    dimensions: &[i64],
    keep_dimension: bool,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    let output = vector_norm_with_context_exact_native(
        backend,
        input,
        order,
        dimensions,
        keep_dimension,
        dtype,
        context,
    )?;
    require_matching(&output, output_gradient, VECTOR_NORM_OPERATION_ID)?;
    let axes = normalized_axes(
        input.descriptor().rank(),
        dimensions,
        VECTOR_NORM_OPERATION_ID,
    )?;
    let geometry = ReductionGeometry::new(input.descriptor().shape(), &axes, keep_dimension)?;
    let input_values = tensor_f64_with_context(backend, input, VECTOR_NORM_OPERATION_ID, context)?;
    let norm_values = tensor_f64_with_context(backend, &output, VECTOR_NORM_OPERATION_ID, context)?;
    let upstream =
        tensor_f64_with_context(backend, output_gradient, VECTOR_NORM_OPERATION_ID, context)?;
    let mut gradients = workspace_filled(backend, context, input_values.len(), 0.0_f64)?;
    let tie_counts = extremum_tie_counts_with_context(
        backend,
        context,
        &geometry,
        &input_values,
        &norm_values,
        order,
    )?;
    for (linear, value) in input_values.iter().copied().enumerate() {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &geometry.input_shape)?;
        let output_index = geometry.output_index(&indices)?;
        gradients[linear] = upstream[output_index]
            * norm_derivative(
                value,
                norm_values[output_index],
                order,
                tie_counts[output_index],
            );
    }
    drop(tie_counts);
    drop(upstream);
    drop(norm_values);
    drop(input_values);
    upload_f64_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &gradients,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn vector_norm_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    order: f64,
    dimensions: &[i64],
    keep_dimension: bool,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    require_matching(input, input_tangent, VECTOR_NORM_OPERATION_ID)?;
    let output = vector_norm_with_context_exact_native(
        backend,
        input,
        order,
        dimensions,
        keep_dimension,
        dtype,
        context,
    )?;
    let axes = normalized_axes(
        input.descriptor().rank(),
        dimensions,
        VECTOR_NORM_OPERATION_ID,
    )?;
    let geometry = ReductionGeometry::new(input.descriptor().shape(), &axes, keep_dimension)?;
    let input_values = tensor_f64_with_context(backend, input, VECTOR_NORM_OPERATION_ID, context)?;
    let tangents =
        tensor_f64_with_context(backend, input_tangent, VECTOR_NORM_OPERATION_ID, context)?;
    let norm_values = tensor_f64_with_context(backend, &output, VECTOR_NORM_OPERATION_ID, context)?;
    let tie_counts = extremum_tie_counts_with_context(
        backend,
        context,
        &geometry,
        &input_values,
        &norm_values,
        order,
    )?;
    let mut output_tangent = workspace_filled(backend, context, geometry.output_count()?, 0.0_f64)?;
    for (linear, (value, tangent)) in input_values
        .iter()
        .copied()
        .zip(tangents.iter().copied())
        .enumerate()
    {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &geometry.input_shape)?;
        let output_index = geometry.output_index(&indices)?;
        output_tangent[output_index] += tangent
            * norm_derivative(
                value,
                norm_values[output_index],
                order,
                tie_counts[output_index],
            );
    }
    drop(tie_counts);
    drop(norm_values);
    drop(tangents);
    drop(input_values);
    upload_f64_with_context(
        backend,
        &geometry.output_shape,
        input.descriptor().stream(),
        &output_tangent,
        context,
    )
}

#[derive(Clone, Debug)]
struct ReductionGeometry {
    input_shape: Vec<usize>,
    output_shape_usize: Vec<usize>,
    output_shape: Vec<u64>,
    reduced: BTreeSet<usize>,
    keep_dimension: bool,
}

impl ReductionGeometry {
    fn new(
        shape: &[u64],
        axes: &[usize],
        keep_dimension: bool,
    ) -> Result<Self, LinearAlgebraPartOneError> {
        let input_shape = usize_shape(shape, "norm input")?;
        let reduced = axes.iter().copied().collect::<BTreeSet<_>>();
        let output_shape_usize = input_shape
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(axis, extent)| {
                if reduced.contains(&axis) {
                    keep_dimension.then_some(1)
                } else {
                    Some(extent)
                }
            })
            .collect::<Vec<_>>();
        let output_shape = output_shape_usize
            .iter()
            .copied()
            .map(|value| {
                u64::try_from(value)
                    .map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("norm output"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            input_shape,
            output_shape_usize,
            output_shape,
            reduced,
            keep_dimension,
        })
    }

    fn output_count(&self) -> Result<usize, LinearAlgebraPartOneError> {
        checked_product(&self.output_shape_usize, "norm output")
    }

    fn output_index(&self, input_indices: &[usize]) -> Result<usize, LinearAlgebraPartOneError> {
        let output_indices = input_indices
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(axis, index)| {
                if self.reduced.contains(&axis) {
                    self.keep_dimension.then_some(0)
                } else {
                    Some(index)
                }
            })
            .collect::<Vec<_>>();
        linear_index(
            &output_indices,
            &self.output_shape_usize,
            "norm output index",
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn symmetric_eigen_decomposition_into(
    matrix: &[f64],
    size: usize,
    values: &mut [f64],
    vectors: &mut [f64],
    order: &mut [usize],
    eigenvalues: &mut [f64],
    ordered_vectors: &mut [f64],
    cancellation: &CancellationToken,
) -> Result<(), LinearAlgebraPartOneError> {
    cancellation.check()?;
    let matrix_elements =
        size.checked_mul(size)
            .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
                "symmetric eigendecomposition matrix",
            ))?;
    if matrix_elements != matrix.len()
        || values.len() != matrix_elements
        || vectors.len() != matrix_elements
        || ordered_vectors.len() != matrix_elements
        || order.len() != size
        || eigenvalues.len() != size
    {
        return invalid(
            EIGH_OPERATION_ID,
            "symmetric eigendecomposition matrix size is invalid",
        );
    }
    values.copy_from_slice(matrix);
    for row in 0..size {
        for column in 0..row {
            let left = values[row * size + column];
            let right = values[column * size + row];
            let scale = left.abs().max(right.abs()).max(1.0);
            if (left - right).abs() > 1.0e-10 * scale {
                return invalid(EIGH_OPERATION_ID, "eigh input must be symmetric");
            }
            let mean = 0.5 * (left + right);
            values[row * size + column] = mean;
            values[column * size + row] = mean;
        }
    }
    vectors.fill(0.0);
    for index in 0..size {
        vectors[index * size + index] = 1.0;
    }
    let sweeps = size
        .checked_mul(size)
        .and_then(|value| value.checked_mul(64))
        .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
            "eigendecomposition sweeps",
        ))?;
    let mut converged = size < 2;
    for sweep in 0..sweeps {
        check_periodically(sweep, cancellation)?;
        let mut pivot = (0, 1, 0.0_f64);
        for row in 0..size {
            check_periodically(row, cancellation)?;
            for column in (row + 1)..size {
                let magnitude = values[row * size + column].abs();
                if magnitude > pivot.2 {
                    pivot = (row, column, magnitude);
                }
            }
        }
        let diagonal_scale = (0..size)
            .map(|index| values[index * size + index].abs())
            .fold(0.0_f64, f64::max);
        if pivot.2 <= f64::EPSILON * 64.0 * diagonal_scale {
            converged = true;
            break;
        }
        let (p, q, _) = pivot;
        let app = values[p * size + p];
        let aqq = values[q * size + q];
        let apq = values[p * size + q];
        let angle = 0.5 * (2.0 * apq).atan2(aqq - app);
        let cosine = angle.cos();
        let sine = angle.sin();
        for index in 0..size {
            check_periodically(index, cancellation)?;
            if index != p && index != q {
                let aip = values[index * size + p];
                let aiq = values[index * size + q];
                let next_p = cosine * aip - sine * aiq;
                let next_q = sine * aip + cosine * aiq;
                values[index * size + p] = next_p;
                values[p * size + index] = next_p;
                values[index * size + q] = next_q;
                values[q * size + index] = next_q;
            }
        }
        values[p * size + p] =
            cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        values[q * size + q] =
            sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        values[p * size + q] = 0.0;
        values[q * size + p] = 0.0;
        for row in 0..size {
            check_periodically(row, cancellation)?;
            let vip = vectors[row * size + p];
            let viq = vectors[row * size + q];
            vectors[row * size + p] = cosine * vip - sine * viq;
            vectors[row * size + q] = sine * vip + cosine * viq;
        }
    }
    if !converged {
        return Err(LinearAlgebraPartOneError::DidNotConverge {
            operation: EIGH_OPERATION_ID,
        });
    }
    for (index, value) in order.iter_mut().enumerate() {
        *value = index;
    }
    order.sort_by(|left, right| {
        values[*left * size + *left].total_cmp(&values[*right * size + *right])
    });
    for (output, input) in eigenvalues.iter_mut().zip(order.iter().copied()) {
        *output = values[input * size + input];
    }
    ordered_vectors.fill(0.0);
    for (output_column, input_column) in order.iter().copied().enumerate() {
        let sign_row = (0..size)
            .max_by(|left, right| {
                vectors[*left * size + input_column]
                    .abs()
                    .total_cmp(&vectors[*right * size + input_column].abs())
            })
            .unwrap_or(0);
        let sign = if vectors[sign_row * size + input_column].is_sign_negative() {
            -1.0
        } else {
            1.0
        };
        for row in 0..size {
            ordered_vectors[row * size + output_column] = sign * vectors[row * size + input_column];
        }
    }
    cancellation.check()?;
    Ok(())
}

pub fn symmetric_eigen_decomposition_with_context(
    backend: &CpuBackend,
    matrix: &[f64],
    size: usize,
    context: &ExecutionContext<'_>,
) -> Result<(CpuWorkspaceVec<f64>, CpuWorkspaceVec<f64>), LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    let matrix_elements =
        size.checked_mul(size)
            .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
                "symmetric eigendecomposition matrix",
            ))?;
    let mut values = workspace_filled(backend, context, matrix_elements, 0.0_f64)?;
    let mut vectors = workspace_filled(backend, context, matrix_elements, 0.0_f64)?;
    let mut order = workspace_filled(backend, context, size, 0_usize)?;
    let mut eigenvalues = workspace_filled(backend, context, size, 0.0_f64)?;
    let mut ordered_vectors = workspace_filled(backend, context, matrix_elements, 0.0_f64)?;
    symmetric_eigen_decomposition_into(
        matrix,
        size,
        &mut values,
        &mut vectors,
        &mut order,
        &mut eigenvalues,
        &mut ordered_vectors,
        context.cancellation,
    )?;
    drop(order);
    drop(vectors);
    drop(values);
    Ok((eigenvalues, ordered_vectors))
}

#[derive(Debug)]
struct WorkspaceLuFactor {
    values: CpuWorkspaceVec<f64>,
    size: usize,
    pivots: CpuWorkspaceVec<usize>,
}

impl WorkspaceLuFactor {
    fn solve_into(
        &self,
        right_hand_side: &[f64],
        columns: usize,
        solution: &mut [f64],
        cancellation: &CancellationToken,
    ) -> Result<(), LinearAlgebraPartOneError> {
        lu_solve_into(
            &self.values,
            &self.pivots,
            self.size,
            right_hand_side,
            columns,
            solution,
            cancellation,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn lu_solve_into(
    values: &[f64],
    pivots: &[usize],
    size: usize,
    right_hand_side: &[f64],
    columns: usize,
    solution: &mut [f64],
    cancellation: &CancellationToken,
) -> Result<(), LinearAlgebraPartOneError> {
    let expected = size
        .checked_mul(columns)
        .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
            "LU right-hand side",
        ))?;
    if right_hand_side.len() != expected
        || solution.len() != expected
        || values.len() != size.checked_mul(size).unwrap_or(usize::MAX)
        || pivots.len() != size
    {
        return invalid(
            INVERSE_OPERATION_ID,
            "LU factor or right-hand-side storage is invalid",
        );
    }
    solution.fill(0.0);
    for row in 0..size {
        check_periodically(row, cancellation)?;
        let source_row = pivots[row];
        for column in 0..columns {
            let mut value = right_hand_side[source_row * columns + column];
            for inner in 0..row {
                value -= values[row * size + inner] * solution[inner * columns + column];
            }
            solution[row * columns + column] = value;
        }
    }
    for row in (0..size).rev() {
        check_periodically(row, cancellation)?;
        for column in 0..columns {
            let mut value = solution[row * columns + column];
            for inner in (row + 1)..size {
                value -= values[row * size + inner] * solution[inner * columns + column];
            }
            solution[row * columns + column] = value / values[row * size + row];
        }
    }
    cancellation.check()?;
    Ok(())
}

fn lu_factor_with_context(
    backend: &CpuBackend,
    matrix: &[f64],
    size: usize,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<WorkspaceLuFactor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    if size.checked_mul(size) != Some(matrix.len()) {
        return invalid(operation, "matrix storage does not match its dimensions");
    }
    let mut values = backend.workspace_vec(context, matrix.len())?;
    for value in matrix.iter().copied() {
        values.try_push(value)?;
    }
    let mut pivots = backend.workspace_vec(context, size)?;
    for pivot in 0..size {
        pivots.try_push(pivot)?;
    }
    factor_lu_in_place(
        &mut values,
        &mut pivots,
        size,
        operation,
        context.cancellation,
    )?;
    Ok(WorkspaceLuFactor {
        values,
        size,
        pivots,
    })
}

fn factor_lu_in_place(
    values: &mut [f64],
    pivots: &mut [usize],
    size: usize,
    operation: &'static str,
    cancellation: &CancellationToken,
) -> Result<(), LinearAlgebraPartOneError> {
    if size.checked_mul(size) != Some(values.len()) || pivots.len() != size {
        return invalid(operation, "matrix storage does not match its dimensions");
    }
    for column in 0..size {
        check_periodically(column, cancellation)?;
        let pivot_row = (column..size)
            .max_by(|left, right| {
                values[*left * size + column]
                    .abs()
                    .total_cmp(&values[*right * size + column].abs())
            })
            .ok_or(LinearAlgebraPartOneError::Singular { operation })?;
        let column_scale = (column..size)
            .map(|row| values[row * size + column].abs())
            .fold(0.0_f64, f64::max);
        if values[pivot_row * size + column].abs() <= f64::EPSILON * 64.0 * column_scale {
            return Err(LinearAlgebraPartOneError::Singular { operation });
        }
        if pivot_row != column {
            for inner in 0..size {
                values.swap(column * size + inner, pivot_row * size + inner);
            }
            pivots.swap(column, pivot_row);
        }
        let pivot = values[column * size + column];
        for row in (column + 1)..size {
            check_periodically(row, cancellation)?;
            let multiplier = values[row * size + column] / pivot;
            values[row * size + column] = multiplier;
            for inner in (column + 1)..size {
                check_periodically(inner, cancellation)?;
                values[row * size + inner] -= multiplier * values[column * size + inner];
            }
        }
    }
    cancellation.check()?;
    Ok(())
}

fn determinant_gradient_matrix_with_context(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    matrix: &[f64],
    size: usize,
) -> Result<CpuWorkspaceVec<f64>, LinearAlgebraPartOneError> {
    if size.checked_mul(size) != Some(matrix.len()) {
        return invalid(
            DETERMINANT_OPERATION_ID,
            "matrix storage does not match its dimensions",
        );
    }
    let mut gradient = workspace_filled(backend, context, matrix.len(), 0.0_f64)?;
    if size == 0 {
        return Ok(gradient);
    }
    if size == 1 {
        gradient[0] = 1.0;
        return Ok(gradient);
    }
    let minor_size = size - 1;
    let minor_elements =
        minor_size
            .checked_mul(minor_size)
            .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
                "determinant minor",
            ))?;
    let mut minor = workspace_filled(backend, context, minor_elements, 0.0_f64)?;
    let mut determinant_work = workspace_filled(backend, context, minor_elements, 0.0_f64)?;
    for row in 0..size {
        check_periodically(row, context.cancellation)?;
        for column in 0..size {
            let mut destination = 0;
            for source_row in 0..size {
                if source_row == row {
                    continue;
                }
                for source_column in 0..size {
                    if source_column != column {
                        minor[destination] = matrix[source_row * size + source_column];
                        destination += 1;
                    }
                }
            }
            determinant_work.copy_from_slice(&minor);
            let sign = if (row + column) % 2 == 0 { 1.0 } else { -1.0 };
            gradient[row * size + column] = sign
                * determinant_value_in_place(
                    &mut determinant_work,
                    minor_size,
                    DETERMINANT_OPERATION_ID,
                    context.cancellation,
                )?;
        }
    }
    context.cancellation.check()?;
    Ok(gradient)
}

fn determinant_value_with_context(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    matrix: &[f64],
    size: usize,
    operation: &'static str,
) -> Result<f64, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    if size.checked_mul(size) != Some(matrix.len()) {
        return invalid(operation, "matrix storage does not match its dimensions");
    }
    if size == 0 {
        return Ok(1.0);
    }
    let mut values = backend.workspace_vec(context, matrix.len())?;
    for value in matrix.iter().copied() {
        values.try_push(value)?;
    }
    determinant_value_in_place(&mut values, size, operation, context.cancellation)
}

fn determinant_value_in_place(
    values: &mut [f64],
    size: usize,
    operation: &'static str,
    cancellation: &CancellationToken,
) -> Result<f64, LinearAlgebraPartOneError> {
    if size.checked_mul(size) != Some(values.len()) {
        return invalid(operation, "matrix storage does not match its dimensions");
    }
    let mut sign = 1.0;
    for column in 0..size {
        check_periodically(column, cancellation)?;
        let pivot_row = (column..size)
            .max_by(|left, right| {
                values[*left * size + column]
                    .abs()
                    .total_cmp(&values[*right * size + column].abs())
            })
            .ok_or_else(|| invalid_error(operation, "determinant pivot is unavailable"))?;
        let column_scale = (column..size)
            .map(|row| values[row * size + column].abs())
            .fold(0.0_f64, f64::max);
        if values[pivot_row * size + column].abs() <= f64::EPSILON * 64.0 * column_scale {
            return Ok(0.0);
        }
        if pivot_row != column {
            for inner in 0..size {
                values.swap(column * size + inner, pivot_row * size + inner);
            }
            sign = -sign;
        }
        let pivot = values[column * size + column];
        for row in (column + 1)..size {
            check_periodically(row, cancellation)?;
            let multiplier = values[row * size + column] / pivot;
            for inner in (column + 1)..size {
                check_periodically(inner, cancellation)?;
                values[row * size + inner] -= multiplier * values[column * size + inner];
            }
        }
    }
    cancellation.check()?;
    Ok(sign
        * (0..size)
            .map(|index| values[index * size + index])
            .product::<f64>())
}

#[allow(clippy::too_many_arguments)]
fn matrix_product_into(
    left: &[f64],
    right: &[f64],
    rows: usize,
    inner: usize,
    columns: usize,
    output: &mut [f64],
    cancellation: &CancellationToken,
) -> Result<(), LinearAlgebraPartOneError> {
    cancellation.check()?;
    if rows.checked_mul(inner) != Some(left.len())
        || inner.checked_mul(columns) != Some(right.len())
        || rows.checked_mul(columns) != Some(output.len())
    {
        return invalid(
            MATMUL_OPERATION_ID,
            "matrix-product storage does not match its dimensions",
        );
    }
    output.fill(0.0);
    for row in 0..rows {
        check_periodically(row, cancellation)?;
        for column in 0..columns {
            let mut value = 0.0;
            for index in 0..inner {
                check_periodically(index, cancellation)?;
                value += left[row * inner + index] * right[index * columns + column];
            }
            output[row * columns + column] = value;
        }
    }
    cancellation.check()?;
    Ok(())
}

fn transpose_matrix_into(
    matrix: &[f64],
    rows: usize,
    columns: usize,
    output: &mut [f64],
    cancellation: &CancellationToken,
) -> Result<(), LinearAlgebraPartOneError> {
    cancellation.check()?;
    if rows.checked_mul(columns) != Some(matrix.len()) || output.len() != matrix.len() {
        return invalid(
            MATMUL_OPERATION_ID,
            "transpose storage does not match its dimensions",
        );
    }
    output.fill(0.0);
    for row in 0..rows {
        check_periodically(row, cancellation)?;
        for column in 0..columns {
            check_periodically(column, cancellation)?;
            output[column * rows + row] = matrix[row * columns + column];
        }
    }
    cancellation.check()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn modified_gram_schmidt_into(
    matrix: &[f64],
    rows: usize,
    columns: usize,
    q_columns: usize,
    r_rows: usize,
    q: &mut [f64],
    r: &mut [f64],
    candidate: &mut [f64],
    basis: &mut [f64],
    cancellation: &CancellationToken,
) -> Result<(), LinearAlgebraPartOneError> {
    if rows.checked_mul(columns) != Some(matrix.len())
        || rows.checked_mul(q_columns) != Some(q.len())
        || r_rows.checked_mul(columns) != Some(r.len())
        || candidate.len() != rows
        || basis.len() != rows
        || q_columns > rows
        || r_rows != q_columns
    {
        return invalid(QR_OPERATION_ID, "QR geometry is invalid");
    }
    q.fill(0.0);
    r.fill(0.0);
    let scale = matrix.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    let tolerance = f64::EPSILON * 128.0 * scale;
    for q_column in 0..q_columns {
        check_periodically(q_column, cancellation)?;
        if q_column < columns {
            for row in 0..rows {
                candidate[row] = matrix[row * columns + q_column];
            }
        } else {
            canonical_vector_into(candidate, q_column);
        }
        orthogonalize(candidate, q, rows, q_columns, q_column, cancellation)?;
        orthogonalize(candidate, q, rows, q_columns, q_column, cancellation)?;
        let mut norm = candidate
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if norm <= tolerance {
            let mut replacement_norm = None;
            for axis in 0..rows {
                check_periodically(axis, cancellation)?;
                canonical_vector_into(basis, axis);
                orthogonalize(basis, q, rows, q_columns, q_column, cancellation)?;
                orthogonalize(basis, q, rows, q_columns, q_column, cancellation)?;
                let basis_norm = basis.iter().map(|value| value * value).sum::<f64>().sqrt();
                if basis_norm > f64::EPSILON * 128.0 {
                    candidate.copy_from_slice(basis);
                    replacement_norm = Some(basis_norm);
                    break;
                }
            }
            norm = replacement_norm.ok_or_else(|| {
                invalid_error(
                    QR_OPERATION_ID,
                    "could not construct an orthogonal QR basis",
                )
            })?;
        }
        for row in 0..rows {
            q[row * q_columns + q_column] = candidate[row] / norm;
        }
    }
    for row in 0..r_rows {
        check_periodically(row, cancellation)?;
        for column in 0..columns {
            let mut value = 0.0;
            for source_row in 0..rows {
                check_periodically(source_row, cancellation)?;
                value += q[source_row * q_columns + row] * matrix[source_row * columns + column];
            }
            r[row * columns + column] = value;
        }
    }
    cancellation.check()?;
    Ok(())
}

fn canonical_vector_into(vector: &mut [f64], axis: usize) {
    vector.fill(0.0);
    if let Some(value) = vector.get_mut(axis) {
        *value = 1.0;
    }
}

fn orthogonalize(
    candidate: &mut [f64],
    q: &[f64],
    rows: usize,
    q_columns: usize,
    populated_columns: usize,
    cancellation: &CancellationToken,
) -> Result<(), LinearAlgebraPartOneError> {
    cancellation.check()?;
    for column in 0..populated_columns {
        check_periodically(column, cancellation)?;
        let mut projection = 0.0;
        for row in 0..rows {
            check_periodically(row, cancellation)?;
            projection += q[row * q_columns + column] * candidate[row];
        }
        for row in 0..rows {
            check_periodically(row, cancellation)?;
            candidate[row] -= projection * q[row * q_columns + column];
        }
    }
    cancellation.check()?;
    Ok(())
}

fn triangle_symmetric_into(
    matrix: &[f64],
    size: usize,
    use_upper_triangle: bool,
    output: &mut [f64],
    cancellation: &CancellationToken,
) -> Result<(), LinearAlgebraPartOneError> {
    cancellation.check()?;
    if size.checked_mul(size) != Some(matrix.len()) || output.len() != matrix.len() {
        return invalid(
            EIGH_OPERATION_ID,
            "matrix storage does not match its dimensions",
        );
    }
    output.copy_from_slice(matrix);
    for row in 0..size {
        check_periodically(row, cancellation)?;
        for column in 0..row {
            check_periodically(column, cancellation)?;
            let value = if use_upper_triangle {
                matrix[column * size + row]
            } else {
                matrix[row * size + column]
            };
            output[row * size + column] = value;
            output[column * size + row] = value;
        }
    }
    cancellation.check()?;
    Ok(())
}

fn update_norm_accumulator(accumulator: &mut f64, value: f64, order: f64) {
    let magnitude = value.abs();
    if value.is_nan() || accumulator.is_nan() {
        *accumulator = f64::NAN;
    } else if order == 0.0 {
        *accumulator += f64::from(value != 0.0);
    } else if order == f64::INFINITY {
        *accumulator = accumulator.max(magnitude);
    } else if order == f64::NEG_INFINITY {
        *accumulator = accumulator.min(magnitude);
    } else {
        *accumulator += magnitude.powf(order);
    }
}

fn finish_norm_accumulator(value: f64, order: f64) -> f64 {
    if order == 0.0 || order.is_infinite() {
        value
    } else {
        value.powf(1.0 / order)
    }
}

fn norm_derivative(value: f64, norm: f64, order: f64, tie_count: usize) -> f64 {
    if order == 0.0 || norm == 0.0 || value == 0.0 {
        0.0
    } else if order.is_infinite() {
        if value.abs() == norm {
            value.signum() / tie_count.max(1) as f64
        } else {
            0.0
        }
    } else {
        value.signum() * value.abs().powf(order - 1.0) * norm.powf(1.0 - order)
    }
}

fn extremum_tie_counts_with_context(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    geometry: &ReductionGeometry,
    input: &[f64],
    norms: &[f64],
    order: f64,
) -> Result<CpuWorkspaceVec<usize>, LinearAlgebraPartOneError> {
    let mut counts = workspace_filled(backend, context, geometry.output_count()?, 1_usize)?;
    if !order.is_infinite() {
        return Ok(counts);
    }
    counts.fill(0);
    for (linear, value) in input.iter().copied().enumerate() {
        check_periodically(linear, context.cancellation)?;
        let indices = unravel_index(linear, &geometry.input_shape)?;
        let output_index = geometry.output_index(&indices)?;
        if value.abs() == norms[output_index] {
            counts[output_index] += 1;
        }
    }
    Ok(counts)
}

fn normalized_axes(
    rank: usize,
    dimensions: &[i64],
    operation: &'static str,
) -> Result<Vec<usize>, LinearAlgebraPartOneError> {
    if dimensions.is_empty() {
        return invalid(operation, "at least one reduction dimension is required");
    }
    let rank_i64 =
        i64::try_from(rank).map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("tensor rank"))?;
    let mut axes = BTreeSet::new();
    for dimension in dimensions {
        let normalized = if *dimension < 0 {
            dimension.checked_add(rank_i64)
        } else {
            Some(*dimension)
        }
        .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
            "normalized dimension",
        ))?;
        let axis = usize::try_from(normalized)
            .map_err(|_| invalid_error(operation, "dimension is out of bounds"))?;
        if axis >= rank || !axes.insert(axis) {
            return invalid(operation, "dimensions must be unique and in bounds");
        }
    }
    Ok(axes.into_iter().collect())
}

fn normalized_axes_in_order(
    rank: usize,
    dimensions: &[i64],
    operation: &'static str,
) -> Result<Vec<usize>, LinearAlgebraPartOneError> {
    if dimensions.is_empty() {
        return Ok(Vec::new());
    }
    let rank_i64 =
        i64::try_from(rank).map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("tensor rank"))?;
    let mut seen = BTreeSet::new();
    let mut axes = Vec::with_capacity(dimensions.len());
    for dimension in dimensions {
        let normalized = if *dimension < 0 {
            dimension.checked_add(rank_i64)
        } else {
            Some(*dimension)
        }
        .ok_or(LinearAlgebraPartOneError::ShapeOverflow(
            "normalized dimension",
        ))?;
        let axis = usize::try_from(normalized)
            .map_err(|_| invalid_error(operation, "dimension is out of bounds"))?;
        if axis >= rank || !seen.insert(axis) {
            return invalid(operation, "dimensions must be unique and in bounds");
        }
        axes.push(axis);
    }
    Ok(axes)
}

fn parse_equation_term(
    term: &str,
    operation: &'static str,
) -> Result<Vec<EquationToken>, LinearAlgebraPartOneError> {
    let compact = term
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>();
    let bytes = compact.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index..].starts_with(b"...") {
            tokens.push(EquationToken::Ellipsis);
            index += 3;
            continue;
        }
        let byte = bytes[index];
        let label = match byte {
            b'A'..=b'Z' => usize::from(byte - b'A'),
            b'a'..=b'z' => 26 + usize::from(byte - b'a'),
            _ => {
                return invalid(
                    operation,
                    "equation labels must be ASCII letters or an ellipsis",
                );
            }
        };
        tokens.push(EquationToken::Label(label));
        index += 1;
    }
    Ok(tokens)
}

fn expand_tokens(
    tokens: &[EquationToken],
    ellipsis_rank: usize,
    maximum_ellipsis: usize,
) -> Result<Vec<usize>, LinearAlgebraPartOneError> {
    if ellipsis_rank > maximum_ellipsis {
        return invalid(
            EINSUM_OPERATION_ID,
            "ellipsis rank exceeds the equation maximum",
        );
    }
    let ellipsis_start = maximum_ellipsis - ellipsis_rank;
    let mut labels = Vec::new();
    for token in tokens {
        match token {
            EquationToken::Label(label) => labels.push(*label),
            EquationToken::Ellipsis => {
                labels.extend((ellipsis_start..maximum_ellipsis).map(|index| 52 + index));
            }
        }
    }
    Ok(labels)
}

fn validate_operand_descriptors(
    operands: &[Tensor],
    operation: &'static str,
) -> Result<(), LinearAlgebraPartOneError> {
    let Some(first) = operands.first() else {
        return invalid(operation, "at least one operand is required");
    };
    require_f32_cpu(first, operation)?;
    for operand in &operands[1..] {
        require_pair(first, operand, operation)?;
    }
    Ok(())
}

fn read_operands_with_context(
    backend: &CpuBackend,
    operands: &[Tensor],
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Vec<CpuWorkspaceVec<f64>>, LinearAlgebraPartOneError> {
    validate_operand_descriptors(operands, operation)?;
    operands
        .iter()
        .map(|operand| tensor_f64_with_context(backend, operand, operation, context))
        .collect()
}

fn contraction_gradients_with_context(
    backend: &CpuBackend,
    plan: &EinsumPlan,
    operands: &[Tensor],
    output_gradient: &Tensor,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<Vec<CpuWorkspaceVec<f64>>, LinearAlgebraPartOneError> {
    require_f32_cpu(output_gradient, operation)?;
    if output_gradient.descriptor().stream() != plan.stream
        || output_gradient.descriptor().shape() != plan.output_shape_u64()?
    {
        return invalid(
            operation,
            "contraction output gradient descriptor does not match",
        );
    }
    let values = read_operands_with_context(backend, operands, operation, context)?;
    let upstream = tensor_f64_with_context(backend, output_gradient, operation, context)?;
    let value_slices = values
        .iter()
        .map(|values| values.as_ref())
        .collect::<Vec<&[f64]>>();
    let mut gradients = values
        .iter()
        .map(|values| workspace_filled(backend, context, values.len(), 0.0_f64))
        .collect::<Result<Vec<_>, _>>()?;
    let mut gradient_slices = gradients
        .iter_mut()
        .map(|gradient| gradient.as_mut())
        .collect::<Vec<&mut [f64]>>();
    plan.gradients_into(
        &value_slices,
        &upstream,
        &mut gradient_slices,
        context.cancellation,
    )?;
    drop(gradient_slices);
    drop(value_slices);
    drop(upstream);
    drop(values);
    Ok(gradients)
}

fn contraction_jvp_with_context(
    backend: &CpuBackend,
    plan: &EinsumPlan,
    operands: &[Tensor],
    operand_tangents: &[Tensor],
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f64>, LinearAlgebraPartOneError> {
    if operands.len() != operand_tangents.len() {
        return invalid(operation, "contraction requires one tangent per operand");
    }
    for (operand, tangent) in operands.iter().zip(operand_tangents) {
        require_matching(operand, tangent, operation)?;
    }
    let values = read_operands_with_context(backend, operands, operation, context)?;
    let tangents = read_operands_with_context(backend, operand_tangents, operation, context)?;
    let mut output = workspace_filled(backend, context, plan.output_count()?, 0.0_f64)?;
    let mut contribution = workspace_filled(backend, context, plan.output_count()?, 0.0_f64)?;
    for differentiated in 0..operands.len() {
        check_periodically(differentiated, context.cancellation)?;
        let substituted = values
            .iter()
            .enumerate()
            .map(|(index, values)| {
                if index == differentiated {
                    tangents[index].as_ref()
                } else {
                    values.as_ref()
                }
            })
            .collect::<Vec<&[f64]>>();
        plan.evaluate_into(&substituted, &mut contribution, context.cancellation)?;
        for (target, value) in output.iter_mut().zip(contribution.iter()) {
            *target += *value;
        }
    }
    drop(contribution);
    drop(tangents);
    drop(values);
    Ok(output)
}

fn require_pair(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), LinearAlgebraPartOneError> {
    require_f32_cpu(left, operation)?;
    require_f32_cpu(right, operation)?;
    require_stream(left, right, operation)
}

fn require_f32_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), LinearAlgebraPartOneError> {
    if tensor.descriptor().device() != DeviceId::CPU {
        return Err(LinearAlgebraPartOneError::UnsupportedDevice {
            operation,
            device: tensor.descriptor().device(),
        });
    }
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(LinearAlgebraPartOneError::UnsupportedDType {
            operation,
            dtype: tensor.descriptor().dtype(),
        });
    }
    Ok(())
}

fn require_stream(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), LinearAlgebraPartOneError> {
    if left.descriptor().stream() != right.descriptor().stream() {
        return invalid(operation, "tensor streams do not match");
    }
    Ok(())
}

fn require_matching(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), LinearAlgebraPartOneError> {
    require_pair(left, right, operation)?;
    if left.descriptor().shape() != right.descriptor().shape() {
        return invalid(operation, "tensor shapes do not match");
    }
    Ok(())
}

pub(crate) fn tensor_f64_with_context(
    backend: &CpuBackend,
    tensor: &Tensor,
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f64>, LinearAlgebraPartOneError> {
    require_f32_cpu(tensor, operation)?;
    let count = usize::try_from(tensor.descriptor().element_count()?)
        .map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("tensor read"))?;
    let mut values = backend.workspace_vec(context, count)?;
    for linear in 0..count {
        check_periodically(linear, context.cancellation)?;
        let linear = u64::try_from(linear)
            .map_err(|_| LinearAlgebraPartOneError::ShapeOverflow("tensor index"))?;
        match DType::F32.decode_scalar(tensor.linear_element_bytes(linear)?)? {
            DecodedScalar::Real(value) => values.try_push(value)?,
            _ => {
                return Err(LinearAlgebraPartOneError::UnsupportedDType {
                    operation,
                    dtype: tensor.descriptor().dtype(),
                });
            }
        }
    }
    Ok(values)
}

pub(crate) fn upload_f64_with_context(
    backend: &CpuBackend,
    shape: &[u64],
    stream: StreamId,
    values: &[f64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    let mut converted = backend.workspace_vec(context, values.len())?;
    for (index, value) in values.iter().copied().enumerate() {
        check_periodically(index, context.cancellation)?;
        converted.try_push(value as f32)?;
    }
    let descriptor =
        TensorDescriptor::contiguous(shape.to_vec(), DType::F32, DeviceId::CPU, stream)?;
    Ok(backend.upload_f32(descriptor, &converted, context)?.0)
}

pub(crate) fn workspace_filled<T: Copy>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    capacity: usize,
    value: T,
) -> Result<CpuWorkspaceVec<T>, LinearAlgebraPartOneError> {
    let mut values = backend.workspace_vec(context, capacity)?;
    for _ in 0..capacity {
        values.try_push(value)?;
    }
    Ok(values)
}

fn usize_shape(
    shape: &[u64],
    subject: &'static str,
) -> Result<Vec<usize>, LinearAlgebraPartOneError> {
    shape
        .iter()
        .map(|value| {
            usize::try_from(*value).map_err(|_| LinearAlgebraPartOneError::ShapeOverflow(subject))
        })
        .collect()
}

fn checked_product(
    shape: &[usize],
    subject: &'static str,
) -> Result<usize, LinearAlgebraPartOneError> {
    shape.iter().try_fold(1usize, |product, extent| {
        product
            .checked_mul(*extent)
            .ok_or(LinearAlgebraPartOneError::ShapeOverflow(subject))
    })
}

fn unravel_index(
    mut linear: usize,
    shape: &[usize],
) -> Result<Vec<usize>, LinearAlgebraPartOneError> {
    let mut indices = vec![0; shape.len()];
    for (index, extent) in indices.iter_mut().zip(shape).rev() {
        if *extent == 0 {
            return invalid(
                MATMUL_OPERATION_ID,
                "cannot unravel an index through an empty dimension",
            );
        }
        *index = linear % extent;
        linear /= extent;
    }
    Ok(indices)
}

fn linear_index(
    indices: &[usize],
    shape: &[usize],
    subject: &'static str,
) -> Result<usize, LinearAlgebraPartOneError> {
    if indices.len() != shape.len() {
        return Err(LinearAlgebraPartOneError::ShapeOverflow(subject));
    }
    indices
        .iter()
        .zip(shape)
        .try_fold(0usize, |linear, (index, extent)| {
            if index >= extent {
                return Err(LinearAlgebraPartOneError::ShapeOverflow(subject));
            }
            linear
                .checked_mul(*extent)
                .and_then(|value| value.checked_add(*index))
                .ok_or(LinearAlgebraPartOneError::ShapeOverflow(subject))
        })
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), LinearAlgebraPartOneError> {
    if index.is_multiple_of(256) {
        cancellation.check()?;
    }
    Ok(())
}

fn invalid<T>(
    operation: &'static str,
    reason: impl Into<String>,
) -> Result<T, LinearAlgebraPartOneError> {
    Err(invalid_error(operation, reason))
}

fn invalid_error(operation: &'static str, reason: impl Into<String>) -> LinearAlgebraPartOneError {
    LinearAlgebraPartOneError::Invalid {
        operation,
        reason: reason.into(),
    }
}

// The one-column specialization retains SciPy 1.16.2 MINPACK Householder,
// Givens and trust-parameter arithmetic order; a normal-equation shortcut
// changes source rounding. Copyright and redistribution notices are retained
// in script/licenses/template.md.hbs and the generated application acknowledgments.
#[derive(Clone, Copy, Debug)]
pub struct MinpackScalarFactor {
    upper: f64,
    column_norm: f64,
    transformed_right: f64,
}

fn minpack_norm(values: &[f64]) -> f64 {
    let giant = f64::from_bits(0x43e6_a09e_667f_3bcd);
    let dwarf = 0.5 / giant;
    let threshold = giant / values.len() as f64;
    let mut large_sum = 0.0;
    let mut middle_sum = 0.0;
    let mut small_sum = 0.0;
    let mut large_maximum = 0.0;
    let mut small_maximum = 0.0;
    for value in values {
        let absolute = value.abs();
        if absolute <= dwarf || absolute >= threshold {
            if absolute > dwarf {
                if absolute > large_maximum {
                    let ratio = large_maximum / absolute;
                    large_sum = 1.0 + large_sum * (ratio * ratio);
                    large_maximum = absolute;
                } else {
                    let ratio = absolute / large_maximum;
                    large_sum += ratio * ratio;
                }
            } else if absolute > small_maximum {
                let ratio = small_maximum / absolute;
                small_sum = 1.0 + small_sum * (ratio * ratio);
                small_maximum = absolute;
            } else if absolute != 0.0 {
                let ratio = absolute / small_maximum;
                small_sum += ratio * ratio;
            }
        } else {
            middle_sum += absolute * absolute;
        }
    }
    if large_sum != 0.0 {
        large_maximum * (large_sum + (middle_sum / large_maximum) / large_maximum).sqrt()
    } else if middle_sum != 0.0 {
        if middle_sum >= small_maximum {
            (middle_sum * (1.0 + (small_maximum / middle_sum) * (small_maximum * small_sum))).sqrt()
        } else {
            (small_maximum * ((middle_sum / small_maximum) + (small_maximum * small_sum))).sqrt()
        }
    } else {
        small_maximum * small_sum.sqrt()
    }
}

fn minpack_factor(column: &mut [f64], residual: &mut [f64]) -> Option<MinpackScalarFactor> {
    let first_column = *column.first()?;
    if column.len() != residual.len() {
        return None;
    }
    let column_norm = minpack_norm(column);
    let signed_norm = if first_column < 0.0 {
        -column_norm
    } else {
        column_norm
    };
    if signed_norm != 0.0 {
        for value in column.iter_mut() {
            *value /= signed_norm;
        }
        *column.first_mut()? += 1.0;
    }
    let first_householder = *column.first()?;
    if first_householder != 0.0 {
        let mut sum = 0.0;
        for (left, right) in column.iter().zip(residual.iter()) {
            sum += left * right;
        }
        let temporary = -sum / first_householder;
        for (right, left) in residual.iter_mut().zip(column.iter()) {
            *right += left * temporary;
        }
    }
    Some(MinpackScalarFactor {
        upper: -signed_norm,
        column_norm,
        transformed_right: *residual.first()?,
    })
}

fn minpack_damped_solve(factor: MinpackScalarFactor, damping: f64) -> (f64, f64) {
    let mut coefficient = factor.upper;
    let mut right = factor.transformed_right;
    if damping != 0.0 {
        let (cosine, sine) = if coefficient.abs() < damping.abs() {
            let cotangent = coefficient / damping;
            let sine = 0.5 / (0.25 + 0.25 * (cotangent * cotangent)).sqrt();
            (sine * cotangent, sine)
        } else {
            let tangent = damping / coefficient;
            let cosine = 0.5 / (0.25 + 0.25 * (tangent * tangent)).sqrt();
            (cosine, cosine * tangent)
        };
        coefficient *= cosine;
        coefficient += sine * damping;
        right = cosine * right + sine * 0.0;
    }
    let solution = if coefficient == 0.0 {
        0.0
    } else {
        (right - 0.0) / coefficient
    };
    (solution, coefficient)
}

fn minpack_step(
    factor: MinpackScalarFactor,
    scale: f64,
    radius: f64,
    mut parameter: f64,
) -> (f64, f64) {
    let mut solution = if factor.upper == 0.0 {
        0.0
    } else {
        factor.transformed_right / factor.upper
    };
    let mut scaled_solution = scale * solution;
    let mut direction_norm = minpack_norm(&[scaled_solution]);
    let mut residual = direction_norm - radius;
    if residual <= 0.1 * radius {
        return (0.0, solution);
    }
    let mut lower = 0.0;
    if factor.upper != 0.0 {
        let intermediate = (scale * (scaled_solution / direction_norm) - 0.0) / factor.upper;
        let norm = minpack_norm(&[intermediate]);
        lower = ((residual / radius) / norm) / norm;
    }
    let gradient = minpack_norm(&[(0.0 + factor.upper * factor.transformed_right) / scale]);
    let mut upper = gradient / radius;
    if upper == 0.0 {
        upper = f64::MIN_POSITIVE / radius.min(0.1);
    }
    parameter = parameter.max(lower).min(upper);
    if parameter == 0.0 {
        parameter = gradient / direction_norm;
    }
    for iteration in 1..=10 {
        if parameter == 0.0 {
            parameter = f64::MIN_POSITIVE.max(upper * 0.001);
        }
        let (next_solution, diagonal) = minpack_damped_solve(factor, parameter.sqrt() * scale);
        solution = next_solution;
        scaled_solution = scale * solution;
        direction_norm = minpack_norm(&[scaled_solution]);
        let previous = residual;
        residual = direction_norm - radius;
        if residual.abs() <= 0.1 * radius
            || (lower == 0.0 && residual <= previous && previous < 0.0)
            || iteration == 10
        {
            return (parameter, solution);
        }
        let norm = minpack_norm(&[(scale * (scaled_solution / direction_norm)) / diagonal]);
        let correction = ((residual / radius) / norm) / norm;
        if residual > 0.0 {
            lower = lower.max(parameter);
        }
        if residual < 0.0 {
            upper = upper.min(parameter);
        }
        parameter = lower.max(parameter + correction);
    }
    (parameter, solution)
}

impl MinpackScalarFactor {
    pub fn upper_coefficient(self) -> f64 {
        self.upper
    }
    pub fn column_norm(self) -> f64 {
        self.column_norm
    }
    pub fn transformed_right(self) -> f64 {
        self.transformed_right
    }
}

pub fn minpack_vector_norm_with_context_exact_native(
    values: &[f64],
    context: &ExecutionContext<'_>,
) -> Result<f64, LinearAlgebraPartOneError> {
    validate_minpack_values(values, "MINPACK norm", context)?;
    let result = minpack_norm(values);
    check_minpack_context(context)?;
    if !result.is_finite() {
        return invalid("MINPACK norm", "norm is not finite");
    }
    Ok(result)
}

pub fn minpack_scalar_qr_with_context_exact_native(
    backend: &CpuBackend,
    column: &[f64],
    right: &[f64],
    context: &ExecutionContext<'_>,
) -> Result<MinpackScalarFactor, LinearAlgebraPartOneError> {
    validate_minpack_values(column, "MINPACK scalar QR", context)?;
    validate_minpack_values(right, "MINPACK scalar QR", context)?;
    if column.len() != right.len() {
        return invalid("MINPACK scalar QR", "column and right-hand lengths differ");
    }
    let mut column_workspace = backend.workspace_vec(context, column.len())?;
    let mut right_workspace = backend.workspace_vec(context, right.len())?;
    for (index, (coefficient, value)) in column.iter().zip(right).enumerate() {
        check_periodically(index, context.cancellation)?;
        column_workspace.try_push(*coefficient)?;
        right_workspace.try_push(*value)?;
    }
    let factor = minpack_factor(&mut column_workspace, &mut right_workspace).ok_or_else(|| {
        LinearAlgebraPartOneError::Invalid {
            operation: "MINPACK scalar QR",
            reason: "missing factor state".to_owned(),
        }
    })?;
    check_minpack_context(context)?;
    if !factor.upper.is_finite()
        || !factor.column_norm.is_finite()
        || !factor.transformed_right.is_finite()
    {
        return invalid("MINPACK scalar QR", "factor state is not finite");
    }
    Ok(factor)
}

pub fn minpack_scalar_damped_solve_with_context_exact_native(
    factor: MinpackScalarFactor,
    damping: f64,
    context: &ExecutionContext<'_>,
) -> Result<(f64, f64), LinearAlgebraPartOneError> {
    check_minpack_context(context)?;
    if !damping.is_finite() || damping < 0.0 {
        return invalid(
            "MINPACK scalar solve",
            "damping must be finite and nonnegative",
        );
    }
    let result = minpack_damped_solve(factor, damping);
    check_minpack_context(context)?;
    if !result.0.is_finite() || !result.1.is_finite() {
        return invalid("MINPACK scalar solve", "solution is not finite");
    }
    Ok(result)
}

pub fn minpack_scalar_step_with_context_exact_native(
    factor: MinpackScalarFactor,
    scale: f64,
    radius: f64,
    parameter: f64,
    context: &ExecutionContext<'_>,
) -> Result<(f64, f64), LinearAlgebraPartOneError> {
    check_minpack_context(context)?;
    if !scale.is_finite()
        || scale <= 0.0
        || !radius.is_finite()
        || radius <= 0.0
        || !parameter.is_finite()
        || parameter < 0.0
    {
        return invalid(
            "MINPACK scalar step",
            "scale and radius must be positive finite and parameter nonnegative finite",
        );
    }
    let result = minpack_step(factor, scale, radius, parameter);
    check_minpack_context(context)?;
    if !result.0.is_finite() || !result.1.is_finite() {
        return invalid("MINPACK scalar step", "step is not finite");
    }
    Ok(result)
}

fn validate_minpack_values(
    values: &[f64],
    operation: &'static str,
    context: &ExecutionContext<'_>,
) -> Result<(), LinearAlgebraPartOneError> {
    check_minpack_context(context)?;
    if values.is_empty() || values.len() > 8192 {
        return invalid(operation, "expected one through 8192 elements");
    }
    for (index, value) in values.iter().enumerate() {
        check_periodically(index, context.cancellation)?;
        if !value.is_finite() {
            return invalid(operation, "values must be finite");
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct MinpackScalarResult {
    pub value: f64,
    pub status: u8,
    pub function_evaluations: usize,
    pub jacobian_evaluations: usize,
}

pub fn minpack_scalar_least_squares_with_context_exact_native<E>(
    backend: &CpuBackend,
    residual_count: usize,
    mut evaluate: impl FnMut(f64, &mut [f64]) -> Result<(), E>,
    context: &ExecutionContext<'_>,
) -> Result<MinpackScalarResult, E>
where
    E: From<LinearAlgebraPartOneError>,
{
    check_minpack_context(context).map_err(E::from)?;
    if residual_count == 0 || residual_count > 8192 {
        return Err(E::from(LinearAlgebraPartOneError::Invalid {
            operation: "MINPACK scalar least squares",
            reason: "expected one through 8192 residuals".to_owned(),
        }));
    }
    let allocate = || -> Result<_, LinearAlgebraPartOneError> {
        let mut values = backend.workspace_vec(context, residual_count)?;
        for index in 0..residual_count {
            check_periodically(index, context.cancellation)?;
            values.try_push(0.0_f64)?;
        }
        Ok(values)
    };
    let mut residual = allocate().map_err(E::from)?;
    let mut candidate_residual = allocate().map_err(E::from)?;
    let mut column = allocate().map_err(E::from)?;
    let mut transformed_residual = allocate().map_err(E::from)?;
    let mut checked_evaluate = |value, output: &mut [f64]| -> Result<(), E> {
        check_minpack_context(context).map_err(E::from)?;
        evaluate(value, output)?;
        validate_minpack_values(output, "MINPACK scalar residual", context).map_err(E::from)?;
        Ok(())
    };
    let mut value = 0.0_f64;
    checked_evaluate(value, &mut residual)?;
    let mut norm = minpack_norm(&residual);
    let mut evaluations = 1;
    let mut jacobians = 0;
    let mut iteration = 1;
    let mut parameter = 0.0_f64;
    let mut radius = 0.0_f64;
    let mut scaled_value_norm = 0.0_f64;
    let mut scale = 1.0_f64;
    loop {
        let sign = if value >= 0.0 { 1.0 } else { -1.0 };
        let displacement = f64::EPSILON.sqrt() * sign * value.abs().max(1.0);
        let displaced_value = value + displacement;
        checked_evaluate(displaced_value, &mut candidate_residual)?;
        for (index, ((coefficient, next), current)) in column
            .iter_mut()
            .zip(candidate_residual.iter())
            .zip(residual.iter())
            .enumerate()
        {
            check_periodically(index, context.cancellation).map_err(E::from)?;
            *coefficient = (next - current) / (displaced_value - value);
        }
        validate_minpack_values(&column, "MINPACK scalar Jacobian", context).map_err(E::from)?;
        jacobians += 1;
        transformed_residual.copy_from_slice(&residual);
        let factor = minpack_factor(&mut column, &mut transformed_residual).ok_or_else(|| {
            E::from(LinearAlgebraPartOneError::Invalid {
                operation: "MINPACK scalar least squares",
                reason: "missing admitted factor state".to_owned(),
            })
        })?;
        if iteration == 1 {
            scale = if factor.column_norm == 0.0 {
                1.0
            } else {
                factor.column_norm
            };
            scaled_value_norm = minpack_norm(&[scale * value]);
            radius = 100.0 * scaled_value_norm;
            if radius == 0.0 {
                radius = 100.0;
            }
        }
        let gradient = if norm == 0.0 || factor.column_norm == 0.0 {
            0.0
        } else {
            ((0.0 + factor.upper * (factor.transformed_right / norm)) / factor.column_norm).abs()
        };
        if gradient <= 1.0e-8 {
            return Ok(MinpackScalarResult {
                value,
                status: 4,
                function_evaluations: evaluations,
                jacobian_evaluations: jacobians,
            });
        }
        scale = scale.max(factor.column_norm);
        loop {
            let (next_parameter, solution) = minpack_scalar_step_with_context_exact_native(
                factor, scale, radius, parameter, context,
            )
            .map_err(E::from)?;
            parameter = next_parameter;
            let direction = -solution;
            let candidate = value + direction;
            let direction_norm = minpack_norm(&[scale * direction]);
            if iteration == 1 {
                radius = radius.min(direction_norm);
            }
            checked_evaluate(candidate, &mut candidate_residual)?;
            evaluations += 1;
            let candidate_norm = minpack_norm(&candidate_residual);
            let actual_reduction = if 0.1 * candidate_norm < norm {
                1.0 - (candidate_norm / norm).powi(2)
            } else {
                -1.0
            };
            let first = minpack_norm(&[0.0 + factor.upper * direction]) / norm;
            let second = (parameter.sqrt() * direction_norm) / norm;
            let predicted_reduction = first * first + second * second / 0.5;
            let directional_derivative = -(first * first + second * second);
            let ratio = if predicted_reduction == 0.0 {
                0.0
            } else {
                actual_reduction / predicted_reduction
            };
            if ratio <= 0.25 {
                let mut contraction = if actual_reduction < 0.0 {
                    (0.5 * directional_derivative)
                        / (directional_derivative + 0.5 * actual_reduction)
                } else {
                    0.5
                };
                if 0.1 * candidate_norm >= norm || contraction < 0.1 {
                    contraction = 0.1;
                }
                radius = contraction * radius.min(direction_norm / 0.1);
                parameter /= contraction;
            } else if parameter == 0.0 || ratio >= 0.75 {
                radius = direction_norm / 0.5;
                parameter *= 0.5;
            }
            if ratio >= 0.0001 {
                value = candidate;
                scaled_value_norm = minpack_norm(&[scale * value]);
                residual.copy_from_slice(&candidate_residual);
                norm = candidate_norm;
                iteration += 1;
            }
            let converged_cost = actual_reduction.abs() <= 0.001
                && predicted_reduction <= 0.001
                && 0.5 * ratio <= 1.0;
            let converged_step = radius <= 1.0e-8 * scaled_value_norm;
            let mut status = match (converged_cost, converged_step) {
                (true, true) => 3,
                (true, false) => 1,
                (false, true) => 2,
                (false, false) => 0,
            };
            if status == 0 {
                if evaluations >= 100 {
                    status = 5;
                }
                if actual_reduction.abs() <= f64::EPSILON
                    && predicted_reduction <= f64::EPSILON
                    && 0.5 * ratio <= 1.0
                {
                    status = 6;
                }
                if radius <= f64::EPSILON * scaled_value_norm {
                    status = 7;
                }
                if gradient <= f64::EPSILON {
                    status = 8;
                }
            }
            if status != 0 {
                return Ok(MinpackScalarResult {
                    value,
                    status,
                    function_evaluations: evaluations,
                    jacobian_evaluations: jacobians,
                });
            }
            if ratio >= 0.0001 {
                break;
            }
        }
    }
}

fn check_minpack_context(context: &ExecutionContext<'_>) -> Result<(), LinearAlgebraPartOneError> {
    context.cancellation.check()?;
    Ok(())
}
