use crate::{
    CpuBackend, DType, DeviceId, ExecutionContext, Layout, RngTransaction,
    generated_random_number_generation_01::{
        RandomNumberGenerationPartOneError, RandomTensorForward,
        standard_normal_tensor_with_context,
    },
};
use thiserror::Error;

pub const RANDN_OPERATION_ID: &str = "COMFY-TENSOR-OP-FD729B8A5363";

#[derive(Debug, Error)]
pub enum RandomNumberGenerationPartTwoError {
    #[error("random-number-generation part-two operation was cancelled")]
    Cancelled,
    #[error(transparent)]
    Canonical(RandomNumberGenerationPartOneError),
    #[error("operation {operation} supports only CPU, not {device:?}")]
    UnsupportedDevice {
        operation: &'static str,
        device: DeviceId,
    },
    #[error("operation {operation} supports only torch.strided layout, not {layout:?}")]
    UnsupportedLayout {
        operation: &'static str,
        layout: Layout,
    },
}

impl From<sim_types::CancellationError> for RandomNumberGenerationPartTwoError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<RandomNumberGenerationPartOneError> for RandomNumberGenerationPartTwoError {
    fn from(error: RandomNumberGenerationPartOneError) -> Self {
        match error {
            RandomNumberGenerationPartOneError::Cancelled => Self::Cancelled,
            error => Self::Canonical(error),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn randn_with_context_exact_native(
    backend: &CpuBackend,
    shape: &[u64],
    dtype: DType,
    layout: Layout,
    device: DeviceId,
    transaction: RngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<RandomTensorForward, RandomNumberGenerationPartTwoError> {
    context.cancellation.check()?;
    if device != DeviceId::CPU {
        return Err(RandomNumberGenerationPartTwoError::UnsupportedDevice {
            operation: RANDN_OPERATION_ID,
            device,
        });
    }
    if layout != Layout::Strided {
        return Err(RandomNumberGenerationPartTwoError::UnsupportedLayout {
            operation: RANDN_OPERATION_ID,
            layout,
        });
    }
    transaction
        .require_device(device)
        .map_err(RandomNumberGenerationPartOneError::from)?;
    Ok(standard_normal_tensor_with_context(
        backend,
        shape,
        dtype,
        transaction,
        RANDN_OPERATION_ID,
        context,
    )?)
}
