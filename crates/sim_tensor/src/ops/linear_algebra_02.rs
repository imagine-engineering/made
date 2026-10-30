use crate::{
    BinaryOperation, CancellationToken, CpuBackend, CpuWorkspaceVec, DType, DeviceId,
    ExecutionContext, LinearAlgebraOperation, Tensor, TensorBackend,
    TensorDescriptor, TensorError,
    generated_linear_algebra_01::{
        LinearAlgebraPartOneError, symmetric_eigen_decomposition_with_context,
        optional_vector_norm_dimensions,
        tensor_f64_with_context, transpose_last_two_with_context_exact_native,
        upload_f64_with_context,
        vector_norm_jvp_with_context_exact_native, vector_norm_vjp_with_context_exact_native,
        vector_norm_with_context_exact_native,
    },
};
use thiserror::Error;

pub const NORM_OPERATION_ID: &str = "COMFY-TENSOR-OP-A5D623C79A18";
pub const SVD_OPERATION_ID: &str = "COMFY-TENSOR-OP-B42F17255D7D";
pub const BMM_OPERATION_ID: &str = "COMFY-TENSOR-OP-C31767F422EE";

#[derive(Debug, Error)]
pub enum LinearAlgebraPartTwoError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Canonical(#[from] LinearAlgebraPartOneError),
    #[error("linear-algebra part-two execution was cancelled")]
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
    #[error("the SVD derivative is undefined for a repeated or zero singular spectrum")]
    UnstableSvdDerivative,
}

impl From<sim_types::CancellationError> for LinearAlgebraPartTwoError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Clone, Debug)]
pub struct BmmGradients {
    pub input: Tensor,
    pub mat2: Tensor,
}

#[derive(Clone, Debug)]
pub struct SvdOutput {
    pub u: Tensor,
    pub s: Tensor,
    pub vh: Tensor,
}

pub fn bmm_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    mat2: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartTwoError> {
    context.cancellation.check()?;
    let [batch, rows, contracted] = input.descriptor().shape() else {
        return Err(invalid(BMM_OPERATION_ID, "input must have rank three"));
    };
    let [other_batch, other_contracted, columns] = mat2.descriptor().shape() else {
        return Err(invalid(BMM_OPERATION_ID, "mat2 must have rank three"));
    };
    if batch != other_batch || contracted != other_contracted {
        return Err(invalid(
            BMM_OPERATION_ID,
            "batch or contraction dimensions do not match",
        ));
    }
    require_pair(input, mat2, BMM_OPERATION_ID)?;
    let output = TensorDescriptor::contiguous(
        vec![*batch, *rows, *columns],
        DType::F32,
        DeviceId::CPU,
        input.descriptor().stream(),
    )?;
    Ok(backend
        .linear_algebra(
            LinearAlgebraOperation::BatchMatrixMultiply,
            &[input.clone(), mat2.clone()],
            output,
            context,
        )?
        .0)
}

pub fn bmm_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    mat2: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<BmmGradients, LinearAlgebraPartTwoError> {
    context.cancellation.check()?;
    let output = bmm_with_context_exact_native(backend, input, mat2, context)?;
    require_matching_tensor(&output, output_gradient, BMM_OPERATION_ID)?;
    let input_transpose = transpose_last_two_with_context_exact_native(input, context)?;
    let mat2_transpose = transpose_last_two_with_context_exact_native(mat2, context)?;
    Ok(BmmGradients {
        input: bmm_with_context_exact_native(backend, output_gradient, &mat2_transpose, context)?,
        mat2: bmm_with_context_exact_native(backend, &input_transpose, output_gradient, context)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn bmm_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    mat2: &Tensor,
    input_tangent: &Tensor,
    mat2_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartTwoError> {
    context.cancellation.check()?;
    require_matching_tensor(input, input_tangent, BMM_OPERATION_ID)?;
    require_matching_tensor(mat2, mat2_tangent, BMM_OPERATION_ID)?;
    let left = bmm_with_context_exact_native(backend, input_tangent, mat2, context)?;
    let right = bmm_with_context_exact_native(backend, input, mat2_tangent, context)?;
    let descriptor = left.descriptor().clone();
    Ok(backend
        .binary(BinaryOperation::Add, &left, &right, descriptor, context)?
        .0)
}

pub fn norm_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    order: f64,
    dimensions: Option<&[i64]>,
    keep_dimension: bool,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartTwoError> {
    context.cancellation.check()?;
    let dimensions = optional_vector_norm_dimensions(input.descriptor().rank(), dimensions)?;
    Ok(vector_norm_with_context_exact_native(
        backend,
        input,
        order,
        &dimensions,
        keep_dimension,
        dtype,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn norm_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    order: f64,
    dimensions: Option<&[i64]>,
    keep_dimension: bool,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartTwoError> {
    context.cancellation.check()?;
    let dimensions = optional_vector_norm_dimensions(input.descriptor().rank(), dimensions)?;
    Ok(vector_norm_vjp_with_context_exact_native(
        backend,
        input,
        output_gradient,
        order,
        &dimensions,
        keep_dimension,
        dtype,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn norm_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    order: f64,
    dimensions: Option<&[i64]>,
    keep_dimension: bool,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartTwoError> {
    context.cancellation.check()?;
    let dimensions = optional_vector_norm_dimensions(input.descriptor().rank(), dimensions)?;
    Ok(vector_norm_jvp_with_context_exact_native(
        backend,
        input,
        input_tangent,
        order,
        &dimensions,
        keep_dimension,
        dtype,
        context,
    )?)
}

pub fn svd_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    full_matrices: bool,
    context: &ExecutionContext<'_>,
) -> Result<SvdOutput, LinearAlgebraPartTwoError> {
    context.cancellation.check()?;
    let geometry = SvdGeometry::new(input, full_matrices)?;
    let input_values = tensor_f64_with_context(backend, input, SVD_OPERATION_ID, context)?;
    let mut u_values = workspace_zeroed(backend, context, geometry.u_element_count()?)?;
    let mut singular_values = workspace_zeroed(backend, context, geometry.s_element_count()?)?;
    let mut vh_values = workspace_zeroed(backend, context, geometry.vh_element_count()?)?;
    let matrix_elements = checked_product(geometry.rows, geometry.columns, "SVD matrix")?;
    for batch in 0..geometry.batch_count {
        check_periodically(batch, context.cancellation)?;
        let input_start = checked_product(batch, matrix_elements, "SVD input offset")?;
        let input_end = input_start
            .checked_add(matrix_elements)
            .ok_or(LinearAlgebraPartTwoError::ShapeOverflow("SVD input end"))?;
        let decomposition = decompose_matrix_with_context(
            backend,
            input_values
                .get(input_start..input_end)
                .ok_or_else(|| invalid(SVD_OPERATION_ID, "SVD input batch is unavailable"))?,
            geometry.rows,
            geometry.columns,
            full_matrices,
            context,
        )?;
        if decomposition.u_columns != geometry.u_columns
            || decomposition.vh_rows != geometry.vh_rows
        {
            return Err(invalid(
                SVD_OPERATION_ID,
                "decomposition output shape does not match the requested profile",
            ));
        }
        copy_batch(
            &mut u_values,
            batch,
            checked_product(geometry.rows, geometry.u_columns, "SVD U batch")?,
            &decomposition.u,
            "SVD U",
        )?;
        copy_batch(
            &mut singular_values,
            batch,
            geometry.reduced,
            &decomposition.s,
            "SVD singular values",
        )?;
        copy_batch(
            &mut vh_values,
            batch,
            checked_product(geometry.vh_rows, geometry.columns, "SVD Vh batch")?,
            &decomposition.vh,
            "SVD Vh",
        )?;
    }
    drop(input_values);
    let u = upload_f64_with_context(
        backend,
        &geometry.u_shape,
        input.descriptor().stream(),
        &u_values,
        context,
    )?;
    drop(u_values);
    let s = upload_f64_with_context(
        backend,
        &geometry.s_shape,
        input.descriptor().stream(),
        &singular_values,
        context,
    )?;
    drop(singular_values);
    let vh = upload_f64_with_context(
        backend,
        &geometry.vh_shape,
        input.descriptor().stream(),
        &vh_values,
        context,
    )?;
    Ok(SvdOutput { u, s, vh })
}

pub fn svd_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    full_matrices: bool,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<SvdOutput, LinearAlgebraPartTwoError> {
    context.cancellation.check()?;
    require_matching_tensor(input, input_tangent, SVD_OPERATION_ID)?;
    let geometry = SvdGeometry::new(input, full_matrices)?;
    let decomposition = svd_with_context_exact_native(backend, input, full_matrices, context)?;
    let tangent_values =
        tensor_f64_with_context(backend, input_tangent, SVD_OPERATION_ID, context)?;
    let decomposition_u =
        tensor_f64_with_context(backend, &decomposition.u, SVD_OPERATION_ID, context)?;
    let decomposition_s =
        tensor_f64_with_context(backend, &decomposition.s, SVD_OPERATION_ID, context)?;
    let decomposition_vh =
        tensor_f64_with_context(backend, &decomposition.vh, SVD_OPERATION_ID, context)?;
    let mut u_values = workspace_zeroed(backend, context, geometry.u_element_count()?)?;
    let mut s_values = workspace_zeroed(backend, context, geometry.s_element_count()?)?;
    let mut vh_values = workspace_zeroed(backend, context, geometry.vh_element_count()?)?;
    let matrix_elements = checked_product(geometry.rows, geometry.columns, "SVD JVP input")?;
    for batch in 0..geometry.batch_count {
        check_periodically(batch, context.cancellation)?;
        let factors = reduced_svd_factors_with_context(
            backend,
            context,
            &decomposition_u,
            &decomposition_s,
            &decomposition_vh,
            &geometry,
            batch,
        )?;
        let start = checked_product(batch, matrix_elements, "SVD JVP input offset")?;
        let end = start
            .checked_add(matrix_elements)
            .ok_or(LinearAlgebraPartTwoError::ShapeOverflow("SVD JVP input end"))?;
        let directional = svd_jvp_batch_with_context(
            backend,
            context,
            &factors,
            tangent_values
                .get(start..end)
                .ok_or_else(|| invalid(SVD_OPERATION_ID, "SVD tangent batch is unavailable"))?,
            geometry.rows,
            geometry.columns,
        )?;
        copy_padded_columns(
            &mut u_values,
            batch,
            &directional.u,
            geometry.rows,
            geometry.reduced,
            geometry.u_columns,
        )?;
        copy_batch(
            &mut s_values,
            batch,
            geometry.reduced,
            &directional.s,
            "SVD JVP singular values",
        )?;
        copy_padded_rows(
            &mut vh_values,
            batch,
            &directional.vh,
            geometry.reduced,
            geometry.columns,
            geometry.vh_rows,
        )?;
    }
    drop(decomposition_vh);
    drop(decomposition_s);
    drop(decomposition_u);
    drop(tangent_values);
    let u = upload_f64_with_context(
        backend,
        &geometry.u_shape,
        input.descriptor().stream(),
        &u_values,
        context,
    )?;
    drop(u_values);
    let s = upload_f64_with_context(
        backend,
        &geometry.s_shape,
        input.descriptor().stream(),
        &s_values,
        context,
    )?;
    drop(s_values);
    let vh = upload_f64_with_context(
        backend,
        &geometry.vh_shape,
        input.descriptor().stream(),
        &vh_values,
        context,
    )?;
    Ok(SvdOutput { u, s, vh })
}

#[allow(clippy::too_many_arguments)]
pub fn svd_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    full_matrices: bool,
    u_gradient: &Tensor,
    s_gradient: &Tensor,
    vh_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, LinearAlgebraPartTwoError> {
    context.cancellation.check()?;
    let geometry = SvdGeometry::new(input, full_matrices)?;
    let decomposition = svd_with_context_exact_native(backend, input, full_matrices, context)?;
    require_matching_tensor(&decomposition.u, u_gradient, SVD_OPERATION_ID)?;
    require_matching_tensor(&decomposition.s, s_gradient, SVD_OPERATION_ID)?;
    require_matching_tensor(&decomposition.vh, vh_gradient, SVD_OPERATION_ID)?;
    let decomposition_u =
        tensor_f64_with_context(backend, &decomposition.u, SVD_OPERATION_ID, context)?;
    let decomposition_s =
        tensor_f64_with_context(backend, &decomposition.s, SVD_OPERATION_ID, context)?;
    let decomposition_vh =
        tensor_f64_with_context(backend, &decomposition.vh, SVD_OPERATION_ID, context)?;
    let u_gradients = tensor_f64_with_context(backend, u_gradient, SVD_OPERATION_ID, context)?;
    let s_gradients = tensor_f64_with_context(backend, s_gradient, SVD_OPERATION_ID, context)?;
    let vh_gradients = tensor_f64_with_context(backend, vh_gradient, SVD_OPERATION_ID, context)?;
    let matrix_elements = checked_product(geometry.rows, geometry.columns, "SVD VJP output")?;
    let mut input_values = workspace_zeroed(
        backend,
        context,
        checked_product(geometry.batch_count, matrix_elements, "SVD VJP batches")?,
    )?;
    let u_batch_elements = checked_product(geometry.rows, geometry.u_columns, "SVD U gradient")?;
    let vh_batch_elements =
        checked_product(geometry.vh_rows, geometry.columns, "SVD Vh gradient")?;
    for batch in 0..geometry.batch_count {
        check_periodically(batch, context.cancellation)?;
        let factors = reduced_svd_factors_with_context(
            backend,
            context,
            &decomposition_u,
            &decomposition_s,
            &decomposition_vh,
            &geometry,
            batch,
        )?;
        let u_start = checked_product(batch, u_batch_elements, "SVD U gradient offset")?;
        let s_start = checked_product(batch, geometry.reduced, "SVD S gradient offset")?;
        let vh_start = checked_product(batch, vh_batch_elements, "SVD Vh gradient offset")?;
        let u_end = checked_sum(u_start, u_batch_elements, "SVD U gradient end")?;
        let s_end = checked_sum(s_start, geometry.reduced, "SVD S gradient end")?;
        let vh_end = checked_sum(vh_start, vh_batch_elements, "SVD Vh gradient end")?;
        let gradient = svd_vjp_batch_with_context(
            backend,
            context,
            &factors,
            u_gradients
                .get(u_start..u_end)
                .ok_or_else(|| invalid(SVD_OPERATION_ID, "SVD U gradient batch is unavailable"))?,
            s_gradients
                .get(s_start..s_end)
                .ok_or_else(|| invalid(SVD_OPERATION_ID, "SVD S gradient batch is unavailable"))?,
            vh_gradients
                .get(vh_start..vh_end)
                .ok_or_else(|| invalid(SVD_OPERATION_ID, "SVD Vh gradient batch is unavailable"))?,
            geometry.rows,
            geometry.columns,
            geometry.u_columns,
            geometry.vh_rows,
        )?;
        copy_batch(
            &mut input_values,
            batch,
            matrix_elements,
            &gradient,
            "SVD VJP output",
        )?;
    }
    drop(vh_gradients);
    drop(s_gradients);
    drop(u_gradients);
    drop(decomposition_vh);
    drop(decomposition_s);
    drop(decomposition_u);
    Ok(upload_f64_with_context(
        backend,
        input.descriptor().shape(),
        input.descriptor().stream(),
        &input_values,
        context,
    )?)
}

#[derive(Debug)]
struct WorkspaceMatrixSvd {
    u: CpuWorkspaceVec<f64>,
    s: CpuWorkspaceVec<f64>,
    vh: CpuWorkspaceVec<f64>,
    u_columns: usize,
    vh_rows: usize,
}

#[derive(Debug)]
struct WorkspaceReducedSvdFactors {
    u: CpuWorkspaceVec<f64>,
    s: CpuWorkspaceVec<f64>,
    v: CpuWorkspaceVec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn reduced_svd_factors_with_context(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    all_u: &[f64],
    all_s: &[f64],
    all_vh: &[f64],
    geometry: &SvdGeometry,
    batch: usize,
) -> Result<WorkspaceReducedSvdFactors, LinearAlgebraPartTwoError> {
    let full_u_elements = checked_product(geometry.rows, geometry.u_columns, "SVD U batch")?;
    let u_start = checked_product(batch, full_u_elements, "SVD U offset")?;
    let u_end = checked_sum(u_start, full_u_elements, "SVD U end")?;
    let full_u = all_u
        .get(u_start..u_end)
        .ok_or_else(|| invalid(SVD_OPERATION_ID, "SVD U batch is unavailable"))?;
    let mut u = workspace_zeroed(
        backend,
        context,
        checked_product(geometry.rows, geometry.reduced, "reduced SVD U")?,
    )?;
    for row in 0..geometry.rows {
        for column in 0..geometry.reduced {
            set_matrix_value(
                &mut u,
                geometry.rows,
                geometry.reduced,
                row,
                column,
                matrix_value(full_u, geometry.rows, geometry.u_columns, row, column)?,
            )?;
        }
    }
    let s_start = checked_product(batch, geometry.reduced, "SVD S offset")?;
    let s_end = checked_sum(s_start, geometry.reduced, "SVD S end")?;
    let mut s = workspace_zeroed(backend, context, geometry.reduced)?;
    s.copy_from_slice(
        all_s
            .get(s_start..s_end)
            .ok_or_else(|| invalid(SVD_OPERATION_ID, "SVD S batch is unavailable"))?,
    );
    let full_vh_elements =
        checked_product(geometry.vh_rows, geometry.columns, "SVD Vh batch")?;
    let vh_start = checked_product(batch, full_vh_elements, "SVD Vh offset")?;
    let vh_end = checked_sum(vh_start, full_vh_elements, "SVD Vh end")?;
    let full_vh = all_vh
        .get(vh_start..vh_end)
        .ok_or_else(|| invalid(SVD_OPERATION_ID, "SVD Vh batch is unavailable"))?;
    let mut v = workspace_zeroed(
        backend,
        context,
        checked_product(geometry.columns, geometry.reduced, "reduced SVD V")?,
    )?;
    for row in 0..geometry.reduced {
        for column in 0..geometry.columns {
            set_matrix_value(
                &mut v,
                geometry.columns,
                geometry.reduced,
                column,
                row,
                matrix_value(full_vh, geometry.vh_rows, geometry.columns, row, column)?,
            )?;
        }
    }
    Ok(WorkspaceReducedSvdFactors { u, s, v })
}

fn svd_jvp_batch_with_context(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    factors: &WorkspaceReducedSvdFactors,
    tangent: &[f64],
    rows: usize,
    columns: usize,
) -> Result<WorkspaceMatrixSvd, LinearAlgebraPartTwoError> {
    let reduced = rows.min(columns);
    require_stable_spectrum(&factors.s, true)?;
    let reduced_square = checked_product(reduced, reduced, "SVD reduced square")?;
    let mut product = workspace_zeroed(backend, context, reduced_square)?;
    triple_product_ut_x_v_into(
        &factors.u,
        tangent,
        &factors.v,
        rows,
        columns,
        reduced,
        &mut product,
        context.cancellation,
    )?;
    let mut omega_u = workspace_zeroed(backend, context, reduced_square)?;
    let mut omega_v = workspace_zeroed(backend, context, reduced_square)?;
    for left in 0..reduced {
        for right in 0..reduced {
            check_periodically(left * reduced + right, context.cancellation)?;
            if left == right {
                continue;
            }
            let left_s = vector_value(&factors.s, left)?;
            let right_s = vector_value(&factors.s, right)?;
            let denominator = right_s * right_s - left_s * left_s;
            let left_right = matrix_value(&product, reduced, reduced, left, right)?;
            let right_left = matrix_value(&product, reduced, reduced, right, left)?;
            set_matrix_value(
                &mut omega_u,
                reduced,
                reduced,
                left,
                right,
                (right_s * left_right + left_s * right_left) / denominator,
            )?;
            set_matrix_value(
                &mut omega_v,
                reduced,
                reduced,
                left,
                right,
                (left_s * left_right + right_s * right_left) / denominator,
            )?;
        }
    }
    let mut u_tangent = workspace_zeroed(
        backend,
        context,
        checked_product(rows, reduced, "SVD U tangent")?,
    )?;
    matrix_product_into(
        &factors.u,
        rows,
        reduced,
        &omega_u,
        reduced,
        reduced,
        &mut u_tangent,
        context.cancellation,
    )?;
    drop(omega_u);
    let mut v_tangent = workspace_zeroed(
        backend,
        context,
        checked_product(columns, reduced, "SVD V tangent")?,
    )?;
    matrix_product_into(
        &factors.v,
        columns,
        reduced,
        &omega_v,
        reduced,
        reduced,
        &mut v_tangent,
        context.cancellation,
    )?;
    drop(omega_v);
    if rows > reduced {
        for column in 0..reduced {
            let inverse = vector_value(&factors.s, column)?.recip();
            for row in 0..rows {
                let mut tangent_v = 0.0;
                for inner in 0..columns {
                    tangent_v += matrix_value(tangent, rows, columns, row, inner)?
                        * matrix_value(&factors.v, columns, reduced, inner, column)?;
                }
                let mut projection = 0.0;
                for inner in 0..reduced {
                    projection += matrix_value(&factors.u, rows, reduced, row, inner)?
                        * matrix_value(&product, reduced, reduced, inner, column)?;
                }
                add_matrix_value(
                    &mut u_tangent,
                    rows,
                    reduced,
                    row,
                    column,
                    (tangent_v - projection) * inverse,
                )?;
            }
        }
    }
    if columns > reduced {
        for column in 0..reduced {
            let inverse = vector_value(&factors.s, column)?.recip();
            for row in 0..columns {
                let mut tangent_transpose_u = 0.0;
                for inner in 0..rows {
                    tangent_transpose_u += matrix_value(tangent, rows, columns, inner, row)?
                        * matrix_value(&factors.u, rows, reduced, inner, column)?;
                }
                let mut projection = 0.0;
                for inner in 0..reduced {
                    projection += matrix_value(&factors.v, columns, reduced, row, inner)?
                        * matrix_value(&product, reduced, reduced, column, inner)?;
                }
                add_matrix_value(
                    &mut v_tangent,
                    columns,
                    reduced,
                    row,
                    column,
                    (tangent_transpose_u - projection) * inverse,
                )?;
            }
        }
    }
    let mut singular_tangent = workspace_zeroed(backend, context, reduced)?;
    for diagonal in 0..reduced {
        singular_tangent[diagonal] = matrix_value(&product, reduced, reduced, diagonal, diagonal)?;
    }
    drop(product);
    let mut vh = workspace_zeroed(
        backend,
        context,
        checked_product(reduced, columns, "SVD Vh tangent")?,
    )?;
    transpose_into(&v_tangent, columns, reduced, &mut vh)?;
    Ok(WorkspaceMatrixSvd {
        u: u_tangent,
        s: singular_tangent,
        vh,
        u_columns: reduced,
        vh_rows: reduced,
    })
}

#[allow(clippy::too_many_arguments)]
fn svd_vjp_batch_with_context(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    factors: &WorkspaceReducedSvdFactors,
    u_gradient: &[f64],
    s_gradient: &[f64],
    vh_gradient: &[f64],
    rows: usize,
    columns: usize,
    u_columns: usize,
    vh_rows: usize,
) -> Result<CpuWorkspaceVec<f64>, LinearAlgebraPartTwoError> {
    let reduced = rows.min(columns);
    require_stable_spectrum(&factors.s, true)?;
    if vh_rows < reduced {
        return Err(invalid(SVD_OPERATION_ID, "Vh gradient row count is invalid"));
    }
    let mut reduced_u_gradient = workspace_zeroed(
        backend,
        context,
        checked_product(rows, reduced, "reduced SVD U gradient")?,
    )?;
    for row in 0..rows {
        for column in 0..reduced {
            set_matrix_value(
                &mut reduced_u_gradient,
                rows,
                reduced,
                row,
                column,
                matrix_value(u_gradient, rows, u_columns, row, column)?,
            )?;
        }
    }
    let mut v_gradient = workspace_zeroed(
        backend,
        context,
        checked_product(columns, reduced, "SVD V gradient")?,
    )?;
    for row in 0..reduced {
        for column in 0..columns {
            set_matrix_value(
                &mut v_gradient,
                columns,
                reduced,
                column,
                row,
                matrix_value(vh_gradient, vh_rows, columns, row, column)?,
            )?;
        }
    }
    let reduced_square = checked_product(reduced, reduced, "SVD reduced square")?;
    let mut h = workspace_zeroed(backend, context, reduced_square)?;
    transpose_left_product_into(
        &factors.u,
        &reduced_u_gradient,
        rows,
        reduced,
        &mut h,
        context.cancellation,
    )?;
    let mut k = workspace_zeroed(backend, context, reduced_square)?;
    transpose_left_product_into(
        &factors.v,
        &v_gradient,
        columns,
        reduced,
        &mut k,
        context.cancellation,
    )?;
    let mut coefficient = workspace_zeroed(backend, context, reduced_square)?;
    for diagonal in 0..reduced {
        set_matrix_value(
            &mut coefficient,
            reduced,
            reduced,
            diagonal,
            diagonal,
            vector_value(s_gradient, diagonal)?,
        )?;
    }
    for left in 0..reduced {
        for right in 0..reduced {
            check_periodically(left * reduced + right, context.cancellation)?;
            if left == right {
                continue;
            }
            let left_s = vector_value(&factors.s, left)?;
            let right_s = vector_value(&factors.s, right)?;
            let denominator = right_s * right_s - left_s * left_s;
            let h_value = matrix_value(&h, reduced, reduced, left, right)?;
            add_matrix_value(
                &mut coefficient,
                reduced,
                reduced,
                left,
                right,
                h_value * right_s / denominator,
            )?;
            add_matrix_value(
                &mut coefficient,
                reduced,
                reduced,
                right,
                left,
                h_value * left_s / denominator,
            )?;
            let k_value = matrix_value(&k, reduced, reduced, left, right)?;
            add_matrix_value(
                &mut coefficient,
                reduced,
                reduced,
                left,
                right,
                k_value * left_s / denominator,
            )?;
            add_matrix_value(
                &mut coefficient,
                reduced,
                reduced,
                right,
                left,
                k_value * right_s / denominator,
            )?;
        }
    }
    let mut gradient = workspace_zeroed(
        backend,
        context,
        checked_product(rows, columns, "SVD VJP gradient")?,
    )?;
    triple_product_u_x_vt_into(
        &factors.u,
        &coefficient,
        &factors.v,
        rows,
        columns,
        reduced,
        &mut gradient,
        context.cancellation,
    )?;
    drop(coefficient);
    if rows > reduced {
        let mut projected = workspace_zeroed(
            backend,
            context,
            checked_product(rows, reduced, "SVD projected U gradient")?,
        )?;
        matrix_product_into(
            &factors.u,
            rows,
            reduced,
            &h,
            reduced,
            reduced,
            &mut projected,
            context.cancellation,
        )?;
        for singular in 0..reduced {
            let inverse = vector_value(&factors.s, singular)?.recip();
            for row in 0..rows {
                let perpendicular =
                    matrix_value(&reduced_u_gradient, rows, reduced, row, singular)?
                        - matrix_value(&projected, rows, reduced, row, singular)?;
                for column in 0..columns {
                    add_matrix_value(
                        &mut gradient,
                        rows,
                        columns,
                        row,
                        column,
                        perpendicular
                            * inverse
                            * matrix_value(&factors.v, columns, reduced, column, singular)?,
                    )?;
                }
            }
        }
    }
    drop(h);
    drop(reduced_u_gradient);
    if columns > reduced {
        let mut projected = workspace_zeroed(
            backend,
            context,
            checked_product(columns, reduced, "SVD projected V gradient")?,
        )?;
        matrix_product_into(
            &factors.v,
            columns,
            reduced,
            &k,
            reduced,
            reduced,
            &mut projected,
            context.cancellation,
        )?;
        for singular in 0..reduced {
            let inverse = vector_value(&factors.s, singular)?.recip();
            for column in 0..columns {
                let perpendicular = matrix_value(&v_gradient, columns, reduced, column, singular)?
                    - matrix_value(&projected, columns, reduced, column, singular)?;
                for row in 0..rows {
                    add_matrix_value(
                        &mut gradient,
                        rows,
                        columns,
                        row,
                        column,
                        matrix_value(&factors.u, rows, reduced, row, singular)?
                            * inverse
                            * perpendicular,
                    )?;
                }
            }
        }
    }
    Ok(gradient)
}

fn decompose_matrix_with_context(
    backend: &CpuBackend,
    matrix: &[f64],
    rows: usize,
    columns: usize,
    full_matrices: bool,
    context: &ExecutionContext<'_>,
) -> Result<WorkspaceMatrixSvd, LinearAlgebraPartTwoError> {
    context.cancellation.check()?;
    let expected = checked_product(rows, columns, "SVD input matrix")?;
    if matrix.len() != expected {
        return Err(invalid(SVD_OPERATION_ID, "matrix workspace length changed"));
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(invalid(SVD_OPERATION_ID, "input contains NaN or infinity"));
    }
    let reduced = rows.min(columns);
    if reduced == 0 {
        let u_columns = if full_matrices { rows } else { 0 };
        let vh_rows = if full_matrices { columns } else { 0 };
        let mut u = workspace_zeroed(
            backend,
            context,
            checked_product(rows, u_columns, "empty SVD U")?,
        )?;
        for diagonal in 0..rows.min(u_columns) {
            set_matrix_value(&mut u, rows, u_columns, diagonal, diagonal, 1.0)?;
        }
        let mut vh = workspace_zeroed(
            backend,
            context,
            checked_product(vh_rows, columns, "empty SVD Vh")?,
        )?;
        for diagonal in 0..vh_rows.min(columns) {
            set_matrix_value(&mut vh, vh_rows, columns, diagonal, diagonal, 1.0)?;
        }
        return Ok(WorkspaceMatrixSvd {
            u,
            s: workspace_zeroed(backend, context, 0)?,
            vh,
            u_columns,
            vh_rows,
        });
    }
    if rows >= columns {
        decompose_tall_with_context(backend, matrix, rows, columns, full_matrices, context)
    } else {
        decompose_wide_with_context(backend, matrix, rows, columns, full_matrices, context)
    }
}

fn decompose_tall_with_context(
    backend: &CpuBackend,
    matrix: &[f64],
    rows: usize,
    columns: usize,
    full_matrices: bool,
    context: &ExecutionContext<'_>,
) -> Result<WorkspaceMatrixSvd, LinearAlgebraPartTwoError> {
    let columns_square = checked_product(columns, columns, "SVD right Gram")?;
    let mut gram = workspace_zeroed(backend, context, columns_square)?;
    gram_right_into(matrix, rows, columns, &mut gram, context.cancellation)?;
    let (eigenvalues, eigenvectors) =
        symmetric_eigen_decomposition_with_context(backend, &gram, columns, context)?;
    drop(gram);
    let order = descending_eigenvalue_order_with_context(backend, context, &eigenvalues)?;
    let singular = ordered_singular_values_with_context(backend, context, &eigenvalues, &order)?;
    drop(eigenvalues);
    let mut v = workspace_zeroed(backend, context, columns_square)?;
    for (destination, source) in order.iter().copied().enumerate() {
        copy_column(
            &eigenvectors,
            columns,
            columns,
            source,
            &mut v,
            columns,
            destination,
        )?;
        canonicalize_column(&mut v, columns, columns, destination)?;
    }
    drop(order);
    drop(eigenvectors);
    let u_columns = if full_matrices { rows } else { columns };
    let mut u = workspace_zeroed(
        backend,
        context,
        checked_product(rows, u_columns, "SVD tall U")?,
    )?;
    for (column, singular_value) in singular.iter().copied().enumerate() {
        check_periodically(column, context.cancellation)?;
        if singular_value > singular_tolerance(&singular) {
            for row in 0..rows {
                let mut value = 0.0;
                for inner in 0..columns {
                    value += matrix_value(matrix, rows, columns, row, inner)?
                        * matrix_value(&v, columns, columns, inner, column)?;
                }
                set_matrix_value(&mut u, rows, u_columns, row, column, value / singular_value)?;
            }
        }
    }
    complete_orthonormal_columns_with_context(
        backend,
        context,
        &mut u,
        rows,
        u_columns,
        columns,
    )?;
    let mut vh = workspace_zeroed(backend, context, columns_square)?;
    transpose_into(&v, columns, columns, &mut vh)?;
    Ok(WorkspaceMatrixSvd {
        u,
        s: singular,
        vh,
        u_columns,
        vh_rows: columns,
    })
}

fn decompose_wide_with_context(
    backend: &CpuBackend,
    matrix: &[f64],
    rows: usize,
    columns: usize,
    full_matrices: bool,
    context: &ExecutionContext<'_>,
) -> Result<WorkspaceMatrixSvd, LinearAlgebraPartTwoError> {
    let rows_square = checked_product(rows, rows, "SVD left Gram")?;
    let mut gram = workspace_zeroed(backend, context, rows_square)?;
    gram_left_into(matrix, rows, columns, &mut gram, context.cancellation)?;
    let (eigenvalues, eigenvectors) =
        symmetric_eigen_decomposition_with_context(backend, &gram, rows, context)?;
    drop(gram);
    let order = descending_eigenvalue_order_with_context(backend, context, &eigenvalues)?;
    let singular = ordered_singular_values_with_context(backend, context, &eigenvalues, &order)?;
    drop(eigenvalues);
    let mut u = workspace_zeroed(backend, context, rows_square)?;
    for (destination, source) in order.iter().copied().enumerate() {
        copy_column(
            &eigenvectors,
            rows,
            rows,
            source,
            &mut u,
            rows,
            destination,
        )?;
        canonicalize_column(&mut u, rows, rows, destination)?;
    }
    drop(order);
    drop(eigenvectors);
    let vh_rows = if full_matrices { columns } else { rows };
    let mut v = workspace_zeroed(
        backend,
        context,
        checked_product(columns, vh_rows, "SVD wide V")?,
    )?;
    for (column, singular_value) in singular.iter().copied().enumerate() {
        check_periodically(column, context.cancellation)?;
        if singular_value > singular_tolerance(&singular) {
            for row in 0..columns {
                let mut value = 0.0;
                for inner in 0..rows {
                    value += matrix_value(matrix, rows, columns, inner, row)?
                        * matrix_value(&u, rows, rows, inner, column)?;
                }
                set_matrix_value(&mut v, columns, vh_rows, row, column, value / singular_value)?;
            }
        }
    }
    complete_orthonormal_columns_with_context(
        backend,
        context,
        &mut v,
        columns,
        vh_rows,
        rows,
    )?;
    let mut vh = workspace_zeroed(
        backend,
        context,
        checked_product(vh_rows, columns, "SVD wide Vh")?,
    )?;
    transpose_into(&v, columns, vh_rows, &mut vh)?;
    Ok(WorkspaceMatrixSvd {
        u,
        s: singular,
        vh,
        u_columns: rows,
        vh_rows,
    })
}

fn descending_eigenvalue_order_with_context(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    eigenvalues: &[f64],
) -> Result<CpuWorkspaceVec<usize>, LinearAlgebraPartTwoError> {
    if eigenvalues.iter().any(|value| !value.is_finite()) {
        return Err(invalid(
            SVD_OPERATION_ID,
            "symmetric eigendecomposition returned a non-finite value",
        ));
    }
    let mut order = backend.workspace_vec(context, eigenvalues.len())?;
    for index in 0..eigenvalues.len() {
        order.try_push(index)?;
    }
    order.sort_by(|left, right| {
        eigenvalues[*right]
            .total_cmp(&eigenvalues[*left])
            .then_with(|| left.cmp(right))
    });
    Ok(order)
}

fn ordered_singular_values_with_context(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    eigenvalues: &[f64],
    order: &[usize],
) -> Result<CpuWorkspaceVec<f64>, LinearAlgebraPartTwoError> {
    let scale = eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    let negative_tolerance = (scale * 1e-10).max(1e-12);
    let mut singular = backend.workspace_vec(context, order.len())?;
    for index in order.iter().copied() {
        let value = *eigenvalues
            .get(index)
            .ok_or_else(|| invalid(SVD_OPERATION_ID, "eigenvalue ordering is out of bounds"))?;
        if value < -negative_tolerance {
            return Err(invalid(
                SVD_OPERATION_ID,
                "symmetric eigendecomposition produced a negative Gram eigenvalue",
            ));
        }
        singular.try_push(value.max(0.0).sqrt())?;
    }
    Ok(singular)
}

fn complete_orthonormal_columns_with_context(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    matrix: &mut [f64],
    rows: usize,
    columns: usize,
    seeded_columns: usize,
) -> Result<(), LinearAlgebraPartTwoError> {
    if columns > rows || seeded_columns > columns {
        return Err(invalid(
            SVD_OPERATION_ID,
            "orthonormal basis dimensions are invalid",
        ));
    }
    let mut candidate = workspace_zeroed(backend, context, rows)?;
    let mut best = workspace_zeroed(backend, context, rows)?;
    for column in 0..columns {
        check_periodically(column, context.cancellation)?;
        let norm = column_norm(matrix, rows, columns, column)?;
        if column < seeded_columns && norm > 1e-12 {
            normalize_column(matrix, rows, columns, column, norm)?;
            orthogonalize_against_previous(matrix, rows, columns, column)?;
            let norm = column_norm(matrix, rows, columns, column)?;
            if norm > 1e-12 {
                normalize_column(matrix, rows, columns, column, norm)?;
                continue;
            }
        }
        let mut best_norm = 0.0_f64;
        for basis in 0..rows {
            candidate.fill(0.0);
            candidate[basis] = 1.0;
            for _ in 0..2 {
                for previous in 0..column {
                    let mut dot = 0.0;
                    for row in 0..rows {
                        dot += candidate[row]
                            * matrix_value(matrix, rows, columns, row, previous)?;
                    }
                    for row in 0..rows {
                        candidate[row] -=
                            dot * matrix_value(matrix, rows, columns, row, previous)?;
                    }
                }
            }
            let norm = candidate.iter().map(|value| value * value).sum::<f64>().sqrt();
            if norm > best_norm {
                best.copy_from_slice(&candidate);
                best_norm = norm;
            }
        }
        if best_norm <= 1e-12 {
            return Err(invalid(
                SVD_OPERATION_ID,
                "could not complete orthonormal basis",
            ));
        }
        for row in 0..rows {
            set_matrix_value(
                matrix,
                rows,
                columns,
                row,
                column,
                best[row] / best_norm,
            )?;
        }
    }
    Ok(())
}

fn gram_right_into(
    matrix: &[f64],
    rows: usize,
    columns: usize,
    gram: &mut [f64],
    cancellation: &CancellationToken,
) -> Result<(), LinearAlgebraPartTwoError> {
    for left in 0..columns {
        for right in left..columns {
            check_periodically(left * columns + right, cancellation)?;
            let mut value = 0.0;
            for row in 0..rows {
                value += matrix_value(matrix, rows, columns, row, left)?
                    * matrix_value(matrix, rows, columns, row, right)?;
            }
            set_matrix_value(gram, columns, columns, left, right, value)?;
            set_matrix_value(gram, columns, columns, right, left, value)?;
        }
    }
    Ok(())
}

fn gram_left_into(
    matrix: &[f64],
    rows: usize,
    columns: usize,
    gram: &mut [f64],
    cancellation: &CancellationToken,
) -> Result<(), LinearAlgebraPartTwoError> {
    for upper in 0..rows {
        for lower in upper..rows {
            check_periodically(upper * rows + lower, cancellation)?;
            let mut value = 0.0;
            for column in 0..columns {
                value += matrix_value(matrix, rows, columns, upper, column)?
                    * matrix_value(matrix, rows, columns, lower, column)?;
            }
            set_matrix_value(gram, rows, rows, upper, lower, value)?;
            set_matrix_value(gram, rows, rows, lower, upper, value)?;
        }
    }
    Ok(())
}

fn transpose_into(
    matrix: &[f64],
    rows: usize,
    columns: usize,
    output: &mut [f64],
) -> Result<(), LinearAlgebraPartTwoError> {
    if output.len() != checked_product(rows, columns, "transpose output")? {
        return Err(invalid(SVD_OPERATION_ID, "transpose output length changed"));
    }
    for row in 0..rows {
        for column in 0..columns {
            set_matrix_value(
                output,
                columns,
                rows,
                column,
                row,
                matrix_value(matrix, rows, columns, row, column)?,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn matrix_product_into(
    left: &[f64],
    left_rows: usize,
    contracted: usize,
    right: &[f64],
    right_rows: usize,
    right_columns: usize,
    output: &mut [f64],
    cancellation: &CancellationToken,
) -> Result<(), LinearAlgebraPartTwoError> {
    if contracted != right_rows
        || output.len() != checked_product(left_rows, right_columns, "matrix product output")?
    {
        return Err(invalid(
            SVD_OPERATION_ID,
            "matrix product dimensions do not match",
        ));
    }
    for row in 0..left_rows {
        for column in 0..right_columns {
            check_periodically(row * right_columns + column, cancellation)?;
            let mut value = 0.0;
            for inner in 0..contracted {
                value += matrix_value(left, left_rows, contracted, row, inner)?
                    * matrix_value(right, right_rows, right_columns, inner, column)?;
            }
            set_matrix_value(output, left_rows, right_columns, row, column, value)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn triple_product_ut_x_v_into(
    u: &[f64],
    matrix: &[f64],
    v: &[f64],
    rows: usize,
    columns: usize,
    reduced: usize,
    output: &mut [f64],
    cancellation: &CancellationToken,
) -> Result<(), LinearAlgebraPartTwoError> {
    if output.len() != checked_product(reduced, reduced, "SVD triple-product output")? {
        return Err(invalid(SVD_OPERATION_ID, "SVD triple-product output changed"));
    }
    for left in 0..reduced {
        for right in 0..reduced {
            check_periodically(left * reduced + right, cancellation)?;
            let mut value = 0.0;
            for row in 0..rows {
                for column in 0..columns {
                    value += matrix_value(u, rows, reduced, row, left)?
                        * matrix_value(matrix, rows, columns, row, column)?
                        * matrix_value(v, columns, reduced, column, right)?;
                }
            }
            set_matrix_value(output, reduced, reduced, left, right, value)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn triple_product_u_x_vt_into(
    u: &[f64],
    middle: &[f64],
    v: &[f64],
    rows: usize,
    columns: usize,
    reduced: usize,
    output: &mut [f64],
    cancellation: &CancellationToken,
) -> Result<(), LinearAlgebraPartTwoError> {
    if output.len() != checked_product(rows, columns, "SVD triple-product output")? {
        return Err(invalid(SVD_OPERATION_ID, "SVD triple-product output changed"));
    }
    for row in 0..rows {
        for column in 0..columns {
            check_periodically(row * columns + column, cancellation)?;
            let mut value = 0.0;
            for left in 0..reduced {
                for right in 0..reduced {
                    value += matrix_value(u, rows, reduced, row, left)?
                        * matrix_value(middle, reduced, reduced, left, right)?
                        * matrix_value(v, columns, reduced, column, right)?;
                }
            }
            set_matrix_value(output, rows, columns, row, column, value)?;
        }
    }
    Ok(())
}

fn transpose_left_product_into(
    left: &[f64],
    right: &[f64],
    rows: usize,
    columns: usize,
    output: &mut [f64],
    cancellation: &CancellationToken,
) -> Result<(), LinearAlgebraPartTwoError> {
    if output.len() != checked_product(columns, columns, "transpose product output")? {
        return Err(invalid(SVD_OPERATION_ID, "transpose product output changed"));
    }
    for output_row in 0..columns {
        for output_column in 0..columns {
            check_periodically(output_row * columns + output_column, cancellation)?;
            let mut value = 0.0;
            for row in 0..rows {
                value += matrix_value(left, rows, columns, row, output_row)?
                    * matrix_value(right, rows, columns, row, output_column)?;
            }
            set_matrix_value(
                output,
                columns,
                columns,
                output_row,
                output_column,
                value,
            )?;
        }
    }
    Ok(())
}

fn workspace_zeroed<T: Copy + Default>(
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
    count: usize,
) -> Result<CpuWorkspaceVec<T>, LinearAlgebraPartTwoError> {
    let mut output = backend.workspace_vec(context, count)?;
    for _ in 0..count {
        output.try_push(T::default())?;
    }
    Ok(output)
}

fn copy_batch<T: Copy>(
    destination: &mut [T],
    batch: usize,
    batch_elements: usize,
    source: &[T],
    label: &'static str,
) -> Result<(), LinearAlgebraPartTwoError> {
    if source.len() != batch_elements {
        return Err(invalid(SVD_OPERATION_ID, format!("{label} batch length changed")));
    }
    let start = checked_product(batch, batch_elements, label)?;
    let end = start
        .checked_add(batch_elements)
        .ok_or(LinearAlgebraPartTwoError::ShapeOverflow(label))?;
    destination
        .get_mut(start..end)
        .ok_or_else(|| invalid(SVD_OPERATION_ID, format!("{label} output is unavailable")))?
        .copy_from_slice(source);
    Ok(())
}

fn copy_padded_columns(
    destination: &mut [f64],
    batch: usize,
    source: &[f64],
    rows: usize,
    source_columns: usize,
    destination_columns: usize,
) -> Result<(), LinearAlgebraPartTwoError> {
    let batch_elements = checked_product(rows, destination_columns, "SVD padded columns")?;
    let start = checked_product(batch, batch_elements, "SVD padded-column offset")?;
    let end = checked_sum(start, batch_elements, "SVD padded-column end")?;
    let output = destination
        .get_mut(start..end)
        .ok_or_else(|| invalid(SVD_OPERATION_ID, "SVD padded-column output is unavailable"))?;
    for row in 0..rows {
        for column in 0..source_columns {
            set_matrix_value(
                output,
                rows,
                destination_columns,
                row,
                column,
                matrix_value(source, rows, source_columns, row, column)?,
            )?;
        }
    }
    Ok(())
}

fn copy_padded_rows(
    destination: &mut [f64],
    batch: usize,
    source: &[f64],
    source_rows: usize,
    columns: usize,
    destination_rows: usize,
) -> Result<(), LinearAlgebraPartTwoError> {
    let batch_elements = checked_product(destination_rows, columns, "SVD padded rows")?;
    let start = checked_product(batch, batch_elements, "SVD padded-row offset")?;
    let end = checked_sum(start, batch_elements, "SVD padded-row end")?;
    let output = destination
        .get_mut(start..end)
        .ok_or_else(|| invalid(SVD_OPERATION_ID, "SVD padded-row output is unavailable"))?;
    let source_elements = checked_product(source_rows, columns, "SVD source rows")?;
    output
        .get_mut(..source_elements)
        .ok_or_else(|| invalid(SVD_OPERATION_ID, "SVD padded-row prefix is unavailable"))?
        .copy_from_slice(source);
    Ok(())
}

fn require_stable_spectrum(
    singular: &[f64],
    require_positive: bool,
) -> Result<(), LinearAlgebraPartTwoError> {
    let scale = singular.first().copied().unwrap_or(0.0).max(1.0);
    let tolerance = scale * 1e-7;
    if require_positive && singular.iter().any(|value| *value <= tolerance) {
        return Err(LinearAlgebraPartTwoError::UnstableSvdDerivative);
    }
    for left in 0..singular.len() {
        for right in left + 1..singular.len() {
            if (vector_value(singular, left)? - vector_value(singular, right)?).abs() <= tolerance {
                return Err(LinearAlgebraPartTwoError::UnstableSvdDerivative);
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct SvdGeometry {
    batch_count: usize,
    rows: usize,
    columns: usize,
    reduced: usize,
    u_columns: usize,
    vh_rows: usize,
    u_shape: Vec<u64>,
    s_shape: Vec<u64>,
    vh_shape: Vec<u64>,
}

impl SvdGeometry {
    fn new(input: &Tensor, full_matrices: bool) -> Result<Self, LinearAlgebraPartTwoError> {
        require_f32_cpu(input, SVD_OPERATION_ID)?;
        let rank = input.descriptor().rank();
        if rank < 2 {
            return Err(invalid(
                SVD_OPERATION_ID,
                "input must have rank at least two",
            ));
        }
        let rows = as_usize(
            *input
                .descriptor()
                .shape()
                .get(rank - 2)
                .ok_or_else(|| invalid(SVD_OPERATION_ID, "matrix row dimension is missing"))?,
        )?;
        let columns =
            as_usize(
                *input.descriptor().shape().get(rank - 1).ok_or_else(|| {
                    invalid(SVD_OPERATION_ID, "matrix column dimension is missing")
                })?,
            )?;
        let reduced = rows.min(columns);
        let u_columns = if full_matrices { rows } else { reduced };
        let vh_rows = if full_matrices { columns } else { reduced };
        let batch_shape = input
            .descriptor()
            .shape()
            .get(..rank - 2)
            .ok_or_else(|| invalid(SVD_OPERATION_ID, "batch dimensions are invalid"))?
            .to_vec();
        let batch_count = element_count(&batch_shape)?;
        let mut u_shape = batch_shape.clone();
        u_shape.extend([as_u64(rows)?, as_u64(u_columns)?]);
        let mut s_shape = batch_shape.clone();
        s_shape.push(as_u64(reduced)?);
        let mut vh_shape = batch_shape;
        vh_shape.extend([as_u64(vh_rows)?, as_u64(columns)?]);
        Ok(Self {
            batch_count,
            rows,
            columns,
            reduced,
            u_columns,
            vh_rows,
            u_shape,
            s_shape,
            vh_shape,
        })
    }

    fn u_element_count(&self) -> Result<usize, LinearAlgebraPartTwoError> {
        checked_product(
            self.batch_count,
            checked_product(self.rows, self.u_columns, "SVD U matrix")?,
            "SVD U output",
        )
    }

    fn s_element_count(&self) -> Result<usize, LinearAlgebraPartTwoError> {
        checked_product(self.batch_count, self.reduced, "SVD singular values")
    }

    fn vh_element_count(&self) -> Result<usize, LinearAlgebraPartTwoError> {
        checked_product(
            self.batch_count,
            checked_product(self.vh_rows, self.columns, "SVD Vh matrix")?,
            "SVD Vh output",
        )
    }
}

fn orthogonalize_against_previous(
    matrix: &mut [f64],
    rows: usize,
    columns: usize,
    column: usize,
) -> Result<(), LinearAlgebraPartTwoError> {
    for previous in 0..column {
        let mut dot = 0.0;
        for row in 0..rows {
            dot += matrix_value(matrix, rows, columns, row, column)?
                * matrix_value(matrix, rows, columns, row, previous)?;
        }
        for row in 0..rows {
            let value = matrix_value(matrix, rows, columns, row, column)?
                - dot * matrix_value(matrix, rows, columns, row, previous)?;
            set_matrix_value(matrix, rows, columns, row, column, value)?;
        }
    }
    Ok(())
}

fn canonicalize_column(
    matrix: &mut [f64],
    rows: usize,
    columns: usize,
    column: usize,
) -> Result<(), LinearAlgebraPartTwoError> {
    let mut pivot = 0.0_f64;
    let mut pivot_absolute = -1.0_f64;
    for row in 0..rows {
        let value = matrix_value(matrix, rows, columns, row, column)?;
        if value.abs() > pivot_absolute {
            pivot = value;
            pivot_absolute = value.abs();
        }
    }
    if pivot < 0.0 {
        for row in 0..rows {
            let value = -matrix_value(matrix, rows, columns, row, column)?;
            set_matrix_value(matrix, rows, columns, row, column, value)?;
        }
    }
    Ok(())
}

fn copy_column(
    source: &[f64],
    source_rows: usize,
    source_columns: usize,
    source_column: usize,
    destination: &mut [f64],
    destination_columns: usize,
    destination_column: usize,
) -> Result<(), LinearAlgebraPartTwoError> {
    for row in 0..source_rows {
        set_matrix_value(
            destination,
            source_rows,
            destination_columns,
            row,
            destination_column,
            matrix_value(source, source_rows, source_columns, row, source_column)?,
        )?;
    }
    Ok(())
}

fn column_norm(
    matrix: &[f64],
    rows: usize,
    columns: usize,
    column: usize,
) -> Result<f64, LinearAlgebraPartTwoError> {
    let mut sum = 0.0;
    for row in 0..rows {
        let value = matrix_value(matrix, rows, columns, row, column)?;
        sum += value * value;
    }
    Ok(sum.sqrt())
}

fn normalize_column(
    matrix: &mut [f64],
    rows: usize,
    columns: usize,
    column: usize,
    norm: f64,
) -> Result<(), LinearAlgebraPartTwoError> {
    for row in 0..rows {
        let value = matrix_value(matrix, rows, columns, row, column)? / norm;
        set_matrix_value(matrix, rows, columns, row, column, value)?;
    }
    Ok(())
}

fn singular_tolerance(singular: &[f64]) -> f64 {
    singular.first().copied().unwrap_or(0.0).max(1.0) * 1e-10
}

fn matrix_value(
    matrix: &[f64],
    rows: usize,
    columns: usize,
    row: usize,
    column: usize,
) -> Result<f64, LinearAlgebraPartTwoError> {
    if row >= rows || column >= columns {
        return Err(invalid(SVD_OPERATION_ID, "matrix index is out of bounds"));
    }
    let index = checked_product(row, columns, "matrix row offset")?
        .checked_add(column)
        .ok_or(LinearAlgebraPartTwoError::ShapeOverflow("matrix index"))?;
    matrix
        .get(index)
        .copied()
        .ok_or_else(|| invalid(SVD_OPERATION_ID, "matrix storage is shorter than its shape"))
}

fn set_matrix_value(
    matrix: &mut [f64],
    rows: usize,
    columns: usize,
    row: usize,
    column: usize,
    value: f64,
) -> Result<(), LinearAlgebraPartTwoError> {
    if row >= rows || column >= columns {
        return Err(invalid(SVD_OPERATION_ID, "matrix index is out of bounds"));
    }
    let index = checked_product(row, columns, "matrix row offset")?
        .checked_add(column)
        .ok_or(LinearAlgebraPartTwoError::ShapeOverflow("matrix index"))?;
    *matrix
        .get_mut(index)
        .ok_or_else(|| invalid(SVD_OPERATION_ID, "matrix storage is shorter than its shape"))? =
        value;
    Ok(())
}

fn add_matrix_value(
    matrix: &mut [f64],
    rows: usize,
    columns: usize,
    row: usize,
    column: usize,
    value: f64,
) -> Result<(), LinearAlgebraPartTwoError> {
    let updated = matrix_value(matrix, rows, columns, row, column)? + value;
    set_matrix_value(matrix, rows, columns, row, column, updated)
}

fn vector_value(values: &[f64], index: usize) -> Result<f64, LinearAlgebraPartTwoError> {
    values
        .get(index)
        .copied()
        .ok_or_else(|| invalid(SVD_OPERATION_ID, "vector index is out of bounds"))
}

fn require_pair(
    left: &Tensor,
    right: &Tensor,
    operation: &'static str,
) -> Result<(), LinearAlgebraPartTwoError> {
    require_f32_cpu(left, operation)?;
    require_f32_cpu(right, operation)?;
    if left.descriptor().stream() != right.descriptor().stream() {
        return Err(invalid(operation, "inputs must use one stream"));
    }
    Ok(())
}

fn require_f32_cpu(
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), LinearAlgebraPartTwoError> {
    if tensor.descriptor().device() != DeviceId::CPU {
        return Err(LinearAlgebraPartTwoError::UnsupportedDevice {
            operation,
            device: tensor.descriptor().device(),
        });
    }
    if tensor.descriptor().dtype() != DType::F32 {
        return Err(LinearAlgebraPartTwoError::UnsupportedDType {
            operation,
            dtype: tensor.descriptor().dtype(),
        });
    }
    Ok(())
}

fn require_matching_tensor(
    input: &Tensor,
    other: &Tensor,
    operation: &'static str,
) -> Result<(), LinearAlgebraPartTwoError> {
    require_pair(input, other, operation)?;
    if input.descriptor().shape() != other.descriptor().shape() {
        return Err(invalid(operation, "tensor shape does not match"));
    }
    Ok(())
}

fn check_periodically(
    index: usize,
    cancellation: &CancellationToken,
) -> Result<(), LinearAlgebraPartTwoError> {
    if index.is_multiple_of(64) {
        cancellation.check()?;
    }
    Ok(())
}

fn element_count(shape: &[u64]) -> Result<usize, LinearAlgebraPartTwoError> {
    shape.iter().try_fold(1_usize, |count, dimension| {
        checked_product(count, as_usize(*dimension)?, "element count")
    })
}

fn checked_product(
    left: usize,
    right: usize,
    label: &'static str,
) -> Result<usize, LinearAlgebraPartTwoError> {
    left.checked_mul(right)
        .ok_or(LinearAlgebraPartTwoError::ShapeOverflow(label))
}

fn checked_sum(
    left: usize,
    right: usize,
    label: &'static str,
) -> Result<usize, LinearAlgebraPartTwoError> {
    left.checked_add(right)
        .ok_or(LinearAlgebraPartTwoError::ShapeOverflow(label))
}

fn as_usize(value: u64) -> Result<usize, LinearAlgebraPartTwoError> {
    usize::try_from(value).map_err(|_| LinearAlgebraPartTwoError::ShapeOverflow("dimension"))
}

fn as_u64(value: usize) -> Result<u64, LinearAlgebraPartTwoError> {
    u64::try_from(value).map_err(|_| LinearAlgebraPartTwoError::ShapeOverflow("index"))
}

fn invalid(operation: &'static str, reason: impl Into<String>) -> LinearAlgebraPartTwoError {
    LinearAlgebraPartTwoError::Invalid {
        operation,
        reason: reason.into(),
    }
}
