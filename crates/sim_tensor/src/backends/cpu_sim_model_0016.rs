use crate::{BackendWorkspaceAuthority, CpuBackend, CpuWorkspaceAuthority, TensorError};

pub type CpuTensorBackend = CpuBackend;
pub type CpuTensorWorkspaceAuthority = CpuWorkspaceAuthority;

pub fn initialize_cpu_tensor_backend(
    memory_limit_bytes: u64,
) -> Result<(CpuTensorBackend, CpuTensorWorkspaceAuthority), TensorError> {
    BackendWorkspaceAuthority::create_backend(memory_limit_bytes)
}
