use crate::{
    CpuBackend, ExecutionContext, Tensor,
    generated_sim_operator_indirection_01::{
        OperatorIndirectionError, cast_to_with_context_exact_native,
    },
};

pub const ZERO_INIT_PARAMETER_OPERATION_ID: &str = "COMFY-TENSOR-OP-A0BD98DDA517";
pub const DISABLE_WEIGHT_INIT_CONV1D_OPERATION_ID: &str = "COMFY-TENSOR-OP-A553C4928CA6";
pub const DISABLE_WEIGHT_INIT_LAYER_NORM_OPERATION_ID: &str = "COMFY-TENSOR-OP-A88C934F4A40";
pub const DISABLE_WEIGHT_INIT_GROUP_NORM_OPERATION_ID: &str = "COMFY-TENSOR-OP-C9049FCF1A75";
pub const PICK_OPERATIONS_OPERATION_ID: &str = "COMFY-TENSOR-OP-D63C669FCD27";
pub const MANUAL_CAST_LINEAR_OPERATION_ID: &str = "COMFY-TENSOR-OP-DAC4074BC3B2";
pub const CAST_TO_INPUT_OPERATION_ID: &str = "COMFY-TENSOR-OP-FDDDAF202C6D";

pub fn cast_to_input_with_context_exact_native(
    backend: &CpuBackend,
    weight: &Tensor,
    input: &Tensor,
    non_blocking: bool,
    copy: bool,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, OperatorIndirectionError> {
    context.cancellation.check()?;
    cast_to_with_context_exact_native(
        backend,
        weight,
        input.descriptor().dtype(),
        input.descriptor().device(),
        non_blocking,
        copy,
        context,
    )
}
