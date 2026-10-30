use crate::{
    CpuBackend, ExecutionContext, Scalar, Tensor,
    generated_indexing_masking_01::{
        IndexingMaskingPartOneError, NonzeroOutput, masked_fill_in_place_with_context_exact_native,
        masked_fill_jvp_with_context_exact_native, masked_fill_vjp_with_context_exact_native,
        nonzero_with_context_exact_native,
    },
};
use thiserror::Error;

pub const MASKED_FILL_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-E9A313720D5D";
pub const NONZERO_METHOD_OPERATION_ID: &str = "COMFY-TENSOR-OP-F76D5ACB74F3";

#[derive(Debug, Error)]
pub enum IndexingMaskingPartTwoError {
    #[error(transparent)]
    Canonical(#[from] IndexingMaskingPartOneError),
}

pub fn masked_fill_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    mask: &Tensor,
    value: Scalar,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartTwoError> {
    let mut output = input.clone();
    masked_fill_in_place_with_context_exact_native(backend, &mut output, mask, value, context)?;
    Ok(output)
}

pub fn masked_fill_method_vjp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    mask: &Tensor,
    output_gradient: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartTwoError> {
    Ok(masked_fill_vjp_with_context_exact_native(
        backend,
        input,
        mask,
        output_gradient,
        context,
    )?)
}

pub fn masked_fill_method_jvp_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    mask: &Tensor,
    input_tangent: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, IndexingMaskingPartTwoError> {
    Ok(masked_fill_jvp_with_context_exact_native(
        backend,
        input,
        mask,
        input_tangent,
        context,
    )?)
}

pub fn nonzero_method_with_context_exact_native(
    backend: &CpuBackend,
    input: &Tensor,
    as_tuple: bool,
    context: &ExecutionContext<'_>,
) -> Result<NonzeroOutput, IndexingMaskingPartTwoError> {
    Ok(nonzero_with_context_exact_native(
        backend, input, as_tuple, context,
    )?)
}
