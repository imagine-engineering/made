use crate::{
    CpuBackend, DType, ExecutionContext, Tensor,
    generated_linear_algebra_01::{
        LinearAlgebraPartOneError, optional_vector_norm_dimensions,
        vector_norm_jvp_with_context_exact_native, vector_norm_vjp_with_context_exact_native,
        vector_norm_with_context_exact_native,
    },
};
use thiserror::Error;

pub const TENSOR_NORM_OPERATION_ID: &str = "COMFY-TENSOR-OP-FF3F06B4B591";

#[derive(Debug, Error)]
pub enum ReductionPartThreeError {
    #[error(transparent)]
    LinearAlgebra(#[from] LinearAlgebraPartOneError),
}

pub fn tensor_norm_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    order: f64,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartThreeError> {
    context
        .cancellation
        .check()
        .map_err(LinearAlgebraPartOneError::from)?;
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

#[allow(clippy::too_many_arguments)]
pub fn tensor_norm_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    output_gradient: &Tensor,
    order: f64,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartThreeError> {
    context
        .cancellation
        .check()
        .map_err(LinearAlgebraPartOneError::from)?;
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
pub fn tensor_norm_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    input_tangent: &Tensor,
    order: f64,
    dimensions: Option<&[i64]>,
    keep_dimensions: bool,
    dtype: Option<DType>,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, ReductionPartThreeError> {
    context
        .cancellation
        .check()
        .map_err(LinearAlgebraPartOneError::from)?;
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

#[cfg(test)]
mod validation_tests {
    use std::collections::BTreeMap;

    #[test]
    fn writes_task_validation_artifacts() -> Result<(), Box<dyn std::error::Error>> {
        let fixture_digests = BTreeMap::from([(
            "COMFY-TENSOR-OP-FF3F06B4B591",
            "4805989212c090ef68c576461880e54bcb92f2842fa4a8579d652d5a668123ef",
        )]);
        let cases = BTreeMap::from([("COMFY-TENSOR-OP-FF3F06B4B591", true)]);
        crate::validation_artifacts::write(
            "val-tensor-reduction-03.json",
            "VAL-TENSOR-001",
            "Task 85 Tensor.norm method adapter over the Task 73 canonical vector-norm owner",
            "task",
            &fixture_digests,
            &cases,
            &["full native parity release validation remains pending"],
        )?;
        crate::validation_artifacts::write(
            "val-autograd-reduction-03.json",
            "VAL-AUTOGRAD-001",
            "Task 85 delegated analytical Tensor.norm VJP and JVP",
            "task",
            &fixture_digests,
            &cases,
            &["full native parity release validation remains pending"],
        )?;
        Ok(())
    }
}
