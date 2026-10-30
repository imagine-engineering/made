use crate::{
    CpuBackend, DType, ExecutionContext, ReductionOperation, Tensor, TensorError,
    generated_elementwise_or_runtime_operation_03::ElementwiseOperand,
    generated_elementwise_or_runtime_operation_05::{
        ElementwiseRuntimePartFiveError, minimum_with_context_exact_native,
    },
    generated_linear_algebra_01::{
        LinearAlgebraPartOneError, optional_vector_norm_dimensions,
        vector_norm_jvp_with_context_exact_native,
        vector_norm_vjp_with_context_exact_native, vector_norm_with_context_exact_native,
    },
    generated_reduction_01::{
        DifferentiableReduction, ReductionForward, ReductionPartOneError,
        floating_output_dtype, reduce, reduction_jvp_with_context_exact_native,
        reduction_vjp_with_context_exact_native,
    },
};
use thiserror::Error;

pub const TENSOR_AMAX_OPERATION_ID: &str = "COMFY-TENSOR-OP-C6AA020CA8A9";
pub const TENSOR_AMIN_OPERATION_ID: &str = "COMFY-TENSOR-OP-F2192DF5C3E8";
pub const TENSOR_ARGMIN_OPERATION_ID: &str = "COMFY-TENSOR-OP-E0B19A56E204";
pub const TENSOR_STD_OPERATION_ID: &str = "COMFY-TENSOR-OP-FD8CFCCD7A33";
pub const TENSOR_SUM_OPERATION_ID: &str = "COMFY-TENSOR-OP-F5FEAF87A86C";
pub const TORCH_AMAX_OPERATION_ID: &str = "COMFY-TENSOR-OP-A7004339E01F";
pub const TORCH_ANY_OPERATION_ID: &str = "COMFY-TENSOR-OP-9F681B3616F6";
pub const TORCH_ARGMAX_OPERATION_ID: &str = "COMFY-TENSOR-OP-C61A44CC6281";
pub const TORCH_MEAN_OPERATION_ID: &str = "COMFY-TENSOR-OP-FA9A98724D9A";
pub const TORCH_MIN_OPERATION_ID: &str = "COMFY-TENSOR-OP-E430C5E4DFE8";
pub const TORCH_NORM_OPERATION_ID: &str = "COMFY-TENSOR-OP-A9FBD2C51ABA";
pub const TORCH_SUM_OPERATION_ID: &str = "COMFY-TENSOR-OP-ADA87E1A44AA";

#[derive(Debug, Error)]
pub enum ReductionPartTwoError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Reduction(#[from] ReductionPartOneError),
    #[error(transparent)]
    LinearAlgebra(#[from] LinearAlgebraPartOneError),
    #[error(transparent)]
    Elementwise(#[from] ElementwiseRuntimePartFiveError),
}

pub enum TorchMinimumArgument<'a> {
    All,
    Dimension(i64),
    Tensor(&'a Tensor),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DifferentiableReductionPartTwo {
    TensorAmax,
    TensorAmin,
    TensorStandardDeviation { correction: u64 },
    TensorSum,
    TorchMean,
    TorchMinimum,
    TorchSum,
}

pub fn tensor_amax_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartTwoError> {
    Ok(reduce(
        backend,
        input,
        ReductionOperation::Maximum,
        dimensions,
        keep_dimensions,
        input.descriptor().dtype(),
        0,
        TENSOR_AMAX_OPERATION_ID,
        context,
    )?)
}

pub fn tensor_amin_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartTwoError> {
    Ok(reduce(
        backend,
        input,
        ReductionOperation::Minimum,
        dimensions,
        keep_dimensions,
        input.descriptor().dtype(),
        0,
        TENSOR_AMIN_OPERATION_ID,
        context,
    )?)
}

pub fn tensor_argmin_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: Option<i64>,
    keep_dimensions: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartTwoError> {
    Ok(reduce(
        backend,
        input,
        ReductionOperation::ArgMinimum,
        dimension.as_ref().map(std::slice::from_ref),
        keep_dimensions,
        DType::I64,
        0,
        TENSOR_ARGMIN_OPERATION_ID,
        context,
    )?)
}

pub fn tensor_std_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: Option<&[i64]>,
    correction: u64,
    keep_dimensions: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartTwoError> {
    let dtype = floating_output_dtype(input, None, TENSOR_STD_OPERATION_ID)?;
    Ok(reduce(
        backend,
        input,
        ReductionOperation::StandardDeviation,
        dimensions,
        keep_dimensions,
        dtype,
        correction,
        TENSOR_STD_OPERATION_ID,
        context,
    )?)
}

pub fn tensor_sum_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartTwoError> {
    let dtype = sum_output_dtype(input, dtype, TENSOR_SUM_OPERATION_ID)?;
    Ok(reduce(
        backend,
        input,
        ReductionOperation::Sum,
        dimensions,
        keep_dimensions,
        dtype,
        0,
        TENSOR_SUM_OPERATION_ID,
        context,
    )?)
}

pub fn torch_amax_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartTwoError> {
    Ok(reduce(
        backend,
        input,
        ReductionOperation::Maximum,
        dimensions,
        keep_dimensions,
        input.descriptor().dtype(),
        0,
        TORCH_AMAX_OPERATION_ID,
        context,
    )?)
}

pub fn torch_any_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartTwoError> {
    Ok(reduce(
        backend,
        input,
        ReductionOperation::Any,
        dimensions,
        keep_dimensions,
        DType::Bool,
        0,
        TORCH_ANY_OPERATION_ID,
        context,
    )?)
}

pub fn torch_argmax_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimension: Option<i64>,
    keep_dimensions: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartTwoError> {
    Ok(reduce(
        backend,
        input,
        ReductionOperation::ArgMaximum,
        dimension.as_ref().map(std::slice::from_ref),
        keep_dimensions,
        DType::I64,
        0,
        TORCH_ARGMAX_OPERATION_ID,
        context,
    )?)
}

pub fn torch_mean_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartTwoError> {
    let dtype = floating_output_dtype(input, dtype, TORCH_MEAN_OPERATION_ID)?;
    Ok(reduce(
        backend,
        input,
        ReductionOperation::Mean,
        dimensions,
        keep_dimensions,
        dtype,
        0,
        TORCH_MEAN_OPERATION_ID,
        context,
    )?)
}

pub fn torch_min_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    argument: TorchMinimumArgument<'_>,
    keep_dimensions: bool,
    context: &ExecutionContext<'_>,
) -> Result<ReductionForward, ReductionPartTwoError> {
    context.cancellation.check().map_err(TensorError::from)?;
    match argument {
        TorchMinimumArgument::All => Ok(ReductionForward {
            values: reduce(
                backend,
                input,
                ReductionOperation::Minimum,
                None,
                keep_dimensions,
                input.descriptor().dtype(),
                0,
                TORCH_MIN_OPERATION_ID,
                context,
            )?,
            indices: None,
        }),
        TorchMinimumArgument::Dimension(dimension) => {
            let dimensions = [dimension];
            Ok(ReductionForward {
                values: reduce(
                    backend,
                    input,
                    ReductionOperation::Minimum,
                    Some(&dimensions),
                    keep_dimensions,
                    input.descriptor().dtype(),
                    0,
                    TORCH_MIN_OPERATION_ID,
                    context,
                )?,
                indices: Some(reduce(
                    backend,
                    input,
                    ReductionOperation::ArgMinimum,
                    Some(&dimensions),
                    keep_dimensions,
                    DType::I64,
                    0,
                    TORCH_MIN_OPERATION_ID,
                    context,
                )?),
            })
        }
        TorchMinimumArgument::Tensor(other) => Ok(ReductionForward {
            values: minimum_with_context_exact_native(
                backend,
                input,
                ElementwiseOperand::Tensor(other),
                context,
            )?,
            indices: None,
        }),
    }
}

pub fn torch_norm_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    order: f64,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartTwoError> {
    context.cancellation.check().map_err(TensorError::from)?;
    let dimensions = optional_vector_norm_dimensions(input.descriptor().rank(), dimensions)?;
    Ok(vector_norm_with_context_exact_native(
        backend,
        input,
        order,
        &dimensions,
        keep_dimensions,
        dtype,
        context,
    )?)
}

pub fn torch_sum_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartTwoError> {
    let dtype = sum_output_dtype(input, dtype, TORCH_SUM_OPERATION_ID)?;
    Ok(reduce(
        backend,
        input,
        ReductionOperation::Sum,
        dimensions,
        keep_dimensions,
        dtype,
        0,
        TORCH_SUM_OPERATION_ID,
        context,
    )?)
}

fn sum_output_dtype(
    input: &Tensor,
    requested: Option<DType>,
    operation: &'static str,
) -> Result<DType, ReductionPartOneError> {
    let input_dtype = input.descriptor().dtype();
    let input_supported = matches!(
        input_dtype,
        DType::F64
            | DType::F32
            | DType::F16
            | DType::Bf16
            | DType::I64
            | DType::I32
            | DType::I16
            | DType::I8
            | DType::U64
            | DType::U32
            | DType::U16
            | DType::U8
            | DType::Bool
    );
    let default_dtype = if matches!(
        input_dtype,
        DType::I64
            | DType::I32
            | DType::I16
            | DType::I8
            | DType::U64
            | DType::U32
            | DType::U16
            | DType::U8
            | DType::Bool
    ) {
        DType::I64
    } else {
        input_dtype
    };
    let output_dtype = requested.unwrap_or(default_dtype);
    if input_supported
        && matches!(
            output_dtype,
            DType::I64 | DType::F64 | DType::F32 | DType::F16 | DType::Bf16
        )
    {
        Ok(output_dtype)
    } else {
        Err(ReductionPartOneError::UnsupportedDType {
            operation,
            dtype: output_dtype,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn reduction_part_two_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output: &Tensor,
    indices: Option<&Tensor>,
    output_gradient: &Tensor,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    operation: DifferentiableReductionPartTwo,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartTwoError> {
    Ok(reduction_vjp_with_context_exact_native(
        backend,
        input,
        output,
        indices,
        output_gradient,
        dimensions,
        keep_dimensions,
        canonical_derivative(operation),
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn reduction_part_two_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    output: &Tensor,
    indices: Option<&Tensor>,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    operation: DifferentiableReductionPartTwo,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartTwoError> {
    Ok(reduction_jvp_with_context_exact_native(
        backend,
        input,
        input_tangent,
        output,
        indices,
        dimensions,
        keep_dimensions,
        canonical_derivative(operation),
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn torch_norm_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    order: f64,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartTwoError> {
    context.cancellation.check().map_err(TensorError::from)?;
    let dimensions = optional_vector_norm_dimensions(input.descriptor().rank(), dimensions)?;
    Ok(vector_norm_vjp_with_context_exact_native(
        backend,
        input,
        output_gradient,
        order,
        &dimensions,
        keep_dimensions,
        dtype,
        context,
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn torch_norm_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    order: f64,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartTwoError> {
    context.cancellation.check().map_err(TensorError::from)?;
    let dimensions = optional_vector_norm_dimensions(input.descriptor().rank(), dimensions)?;
    Ok(vector_norm_jvp_with_context_exact_native(
        backend,
        input,
        input_tangent,
        order,
        &dimensions,
        keep_dimensions,
        dtype,
        context,
    )?)
}

fn canonical_derivative(operation: DifferentiableReductionPartTwo) -> DifferentiableReduction {
    match operation {
        DifferentiableReductionPartTwo::TensorAmax => DifferentiableReduction::Maximum,
        DifferentiableReductionPartTwo::TensorAmin
        | DifferentiableReductionPartTwo::TorchMinimum => DifferentiableReduction::Minimum,
        DifferentiableReductionPartTwo::TensorStandardDeviation { correction } => {
            DifferentiableReduction::StandardDeviation { correction }
        }
        DifferentiableReductionPartTwo::TensorSum => DifferentiableReduction::Sum {
            operation_id: TENSOR_SUM_OPERATION_ID,
        },
        DifferentiableReductionPartTwo::TorchMean => DifferentiableReduction::Mean,
        DifferentiableReductionPartTwo::TorchSum => DifferentiableReduction::Sum {
            operation_id: TORCH_SUM_OPERATION_ID,
        },
    }
}

#[cfg(test)]
mod validation_tests {
    use std::collections::BTreeMap;

    #[test]
    fn writes_task_validation_artifacts() -> Result<(), Box<dyn std::error::Error>> {
        let fixture_digests = BTreeMap::from([
            (
                "COMFY-TENSOR-OP-C6AA020CA8A9",
                "227e525b856aef4c99f348b7345ea2eb19fa16e6196af09f5c30d999e1991281",
            ),
            (
                "COMFY-TENSOR-OP-F2192DF5C3E8",
                "2e3e91b80e7d7a81ef9769a1dc3419ce2696d3964144e5ef214808686cfa4793",
            ),
            (
                "COMFY-TENSOR-OP-E0B19A56E204",
                "6489945635329d1e4c38ceb042ec76e422843b413b69bf686cb203e39aa73e6b",
            ),
            (
                "COMFY-TENSOR-OP-FD8CFCCD7A33",
                "5e7fc013d971114b2d71ccd181fa2731eb677c76b3e6883edd04a3a7d445e844",
            ),
            (
                "COMFY-TENSOR-OP-F5FEAF87A86C",
                "15f4d1fa3e9bc1bafa8029eefe793f0cb79db43427b75ffc8f84ee0dfe4434fe",
            ),
            (
                "COMFY-TENSOR-OP-A7004339E01F",
                "fb64fb8578f44839461b196c2d925e75aa129bf064c88d911097f9f6df5cd42f",
            ),
            (
                "COMFY-TENSOR-OP-9F681B3616F6",
                "f655fb053fb42ac9bb75a2de654de4777d80449b0dedaa9d235e45f614f56015",
            ),
            (
                "COMFY-TENSOR-OP-C61A44CC6281",
                "82c9d1c799de1d79dd140f093933ee054403011d03b054d7b1eb45dbe7c8c475",
            ),
            (
                "COMFY-TENSOR-OP-FA9A98724D9A",
                "a8af5b4736db88f042b081dd1be0c8a5e0ee83993a8bfd0d369796a1d90a4efa",
            ),
            (
                "COMFY-TENSOR-OP-E430C5E4DFE8",
                "643b97f25099cd6318d20d8d25af4c26423c9f8a5ef88873eeae247e653f8c6d",
            ),
            (
                "COMFY-TENSOR-OP-A9FBD2C51ABA",
                "ab1bfe13158df93fb415f0bf9cf93855a6f2fcc321fbab19adf2f9665fb76b0e",
            ),
            (
                "COMFY-TENSOR-OP-ADA87E1A44AA",
                "6ace6124c0ae094110120d18b76e59abe77ad0ba5aa2f8a07d3ac7f1c2bf9baa",
            ),
        ]);
        let cases = fixture_digests
            .keys()
            .map(|operation_id| (*operation_id, true))
            .collect::<BTreeMap<_, _>>();
        crate::validation_artifacts::write(
            "val-tensor-reduction-02.json",
            "VAL-TENSOR-001",
            "Task 84 reduction part two: 12 exact adapters over canonical reduction and vector norm owners",
            "task",
            &fixture_digests,
            &cases,
            &["full native parity release validation remains pending"],
        )?;
        crate::validation_artifacts::write(
            "val-autograd-reduction-02.json",
            "VAL-AUTOGRAD-001",
            "Task 84 reduction differentiability classifications and delegated analytical maps",
            "task",
            &fixture_digests,
            &cases,
            &["full native parity release validation remains pending"],
        )?;
        Ok(())
    }
}
