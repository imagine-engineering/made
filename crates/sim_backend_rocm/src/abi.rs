#![allow(dead_code)]

use std::ffi::{c_char, c_int, c_uint, c_void};

pub(crate) const ABI_SCHEMA_VERSION: u16 = 1;
pub(crate) const ABI_FLOOR: RocmVersion = RocmVersion::new(6, 1, 0);
pub(crate) const HIP_RUNTIME_FLOOR: i32 = 60_100_000;
pub(crate) const REQUIRED_TARGET: &str = "x86_64-unknown-linux-gnu";

pub(crate) type HipError = c_int;
pub(crate) type HipDeviceAttribute = c_int;
pub(crate) type HipInitFlags = c_uint;
pub(crate) type HipStreamFlags = c_uint;
pub(crate) type HipEventFlags = c_uint;
pub(crate) type HipMemcpyKind = c_int;
pub(crate) type HipRtcResult = c_int;
pub(crate) type RocblasStatus = c_int;
pub(crate) type RocblasOperation = c_int;
pub(crate) type MiopenStatus = c_int;
pub(crate) type HipStream = *mut c_void;
pub(crate) type HipEvent = *mut c_void;
pub(crate) type HipModule = *mut c_void;
pub(crate) type HipFunction = *mut c_void;
pub(crate) type HipRtcProgram = *mut c_void;
pub(crate) type RocblasHandle = *mut c_void;
pub(crate) type MiopenHandle = *mut c_void;
pub(crate) type MiopenTensorDescriptor = *mut c_void;
pub(crate) type MiopenConvolutionDescriptor = *mut c_void;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RocmVersion {
    pub(crate) major: u32,
    pub(crate) minor: u32,
    pub(crate) patch: u32,
}

impl RocmVersion {
    pub(crate) const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SymbolContract {
    pub(crate) library: &'static str,
    pub(crate) name: &'static str,
    pub(crate) signature: &'static str,
}

#[derive(Clone, Copy)]
pub(crate) struct HeaderEvidence {
    pub(crate) name: &'static str,
    pub(crate) source_url: &'static str,
    pub(crate) byte_length: usize,
    pub(crate) sha256: &'static str,
}

#[derive(Clone, Copy)]
pub(crate) struct BindingEvidence {
    pub(crate) name: &'static str,
    pub(crate) header: &'static str,
    pub(crate) line_start: usize,
    pub(crate) line_end: usize,
    pub(crate) excerpt_sha256: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CompletionEvidence {
    Unavailable,
    Verified {
        artifact_sha256: &'static str,
        run_id: &'static str,
    },
}

include!(concat!(env!("OUT_DIR"), "/rocm_abi_bindings.rs"));

#[cfg(test)]
mod tests {
    use std::mem::{align_of, size_of};

    use super::*;

    #[test]
    fn generated_contract_has_one_reviewed_record_per_abi_item() {
        assert_eq!(HEADER_EVIDENCE.len(), 9);
        assert_eq!(SYMBOLS.len(), 52);
        assert_eq!(BINDING_EVIDENCE.len(), 74);
        assert_eq!(HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 23);
        assert_eq!(HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 61);
        assert_eq!(HIP_INIT_FLAGS_ZERO, 0);
        assert_eq!(HIP_STREAM_NON_BLOCKING, 1);
        assert_eq!(HIP_EVENT_DISABLE_TIMING, 2);
        assert_eq!(HIP_MEMCPY_HOST_TO_DEVICE, 1);
        assert_eq!(HIP_MEMCPY_DEVICE_TO_HOST, 2);
        assert_eq!(HIP_MEMCPY_DEVICE_TO_DEVICE, 3);
        assert_eq!(HIP_SUCCESS, 0);
        assert_eq!(HIP_ERROR_OUT_OF_MEMORY, 2);
        assert_eq!(HIP_ERROR_INVALID_CONTEXT, 201);
        assert_eq!(HIP_ERROR_ILLEGAL_ADDRESS, 700);
        assert_eq!(HIP_ERROR_CONTEXT_IS_DESTROYED, 709);
        assert_eq!(HIP_ERROR_LAUNCH_FAILURE, 719);
        assert_eq!(ROCBLAS_STATUS_SUCCESS, 0);
        assert_eq!(ROCBLAS_STATUS_MEMORY_ERROR, 5);
        assert_eq!(ROCBLAS_OPERATION_NONE, 111);
        assert_eq!(HIPRTC_SUCCESS, 0);
        assert_eq!(MIOPEN_STATUS_SUCCESS, 0);
        assert!(HEADER_EVIDENCE.iter().all(|header| {
            header.source_url.contains("/rocm-6.1.2/")
                && header.sha256.len() == 64
                && header.byte_length > 0
        }));
        assert!(BINDING_EVIDENCE.iter().all(|binding| {
            binding.line_start <= binding.line_end && binding.excerpt_sha256.len() == 64
        }));
    }

    #[test]
    fn generated_layouts_match_the_reviewed_x86_64_contract() {
        assert_eq!((size_of::<HipUuid>(), align_of::<HipUuid>()), (16, 1));
        assert_eq!(
            (size_of::<HipIpcMemHandle>(), align_of::<HipIpcMemHandle>()),
            (64, 1)
        );
        assert_eq!(
            (
                size_of::<MiopenConvAlgoPerf>(),
                align_of::<MiopenConvAlgoPerf>()
            ),
            (16, 8)
        );
    }

    #[test]
    fn completion_claim_is_typed_and_fail_closed() {
        match env!("SIM_ROCM_COMPLETION_EVIDENCE_STATE") {
            "unavailable" => assert_eq!(COMPLETION_EVIDENCE, CompletionEvidence::Unavailable),
            "verified" => match COMPLETION_EVIDENCE {
                CompletionEvidence::Verified {
                    artifact_sha256,
                    run_id,
                } => {
                    assert_eq!(artifact_sha256.len(), 64);
                    assert!(!run_id.is_empty());
                }
                CompletionEvidence::Unavailable => {
                    panic!("build claimed verified completion evidence without a proof artifact")
                }
            },
            state => panic!("unexpected completion-evidence state {state}"),
        }
    }
}
