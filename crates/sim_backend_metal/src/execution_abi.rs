use serde::Deserialize;
use std::collections::BTreeMap;
use thiserror::Error;

pub const EXECUTION_ABI_JSON: &str = include_str!("../abi/execution-v1.json");
pub const EXECUTION_CONTRACT: &str = "zed-sim-metal-execution-v1";
pub const EXECUTION_UNSAFE_OWNER: &str = "sim_backend_metal::execution";
pub const METAL_ADD_F16_FUNCTION: &str = "zed_sim_metal_add_f16_v1";
pub const METAL_ADD_F32_FUNCTION: &str = "zed_sim_metal_add_f32_v1";
pub const MAXIMUM_COMMAND_BUFFERS_PER_STREAM: usize = 64;

const RESOURCE_SELECTORS: [(&str, &str, &str, ReturnNullability); 29] = [
    (
        "MTLBlitCommandEncoder",
        "endEncoding",
        "v16@0:8",
        ReturnNullability::NotApplicable,
    ),
    (
        "MTLBlitCommandEncoder",
        "synchronizeResource:",
        "v24@0:8@16",
        ReturnNullability::NotApplicable,
    ),
    (
        "MTLBuffer",
        "contents",
        "^v16@0:8",
        ReturnNullability::Nonnull,
    ),
    (
        "MTLBuffer",
        "didModifyRange:",
        "v32@0:8{_NSRange=QQ}16",
        ReturnNullability::NotApplicable,
    ),
    (
        "MTLBuffer",
        "length",
        "Q16@0:8",
        ReturnNullability::NotApplicable,
    ),
    (
        "MTLCommandBuffer",
        "blitCommandEncoder",
        "@16@0:8",
        ReturnNullability::Nullable,
    ),
    (
        "MTLCommandBuffer",
        "commit",
        "v16@0:8",
        ReturnNullability::NotApplicable,
    ),
    (
        "MTLCommandBuffer",
        "computeCommandEncoder",
        "@16@0:8",
        ReturnNullability::Nullable,
    ),
    (
        "MTLCommandBuffer",
        "error",
        "@16@0:8",
        ReturnNullability::Nullable,
    ),
    (
        "MTLCommandBuffer",
        "status",
        "Q16@0:8",
        ReturnNullability::NotApplicable,
    ),
    (
        "MTLCommandBuffer",
        "waitUntilCompleted",
        "v16@0:8",
        ReturnNullability::NotApplicable,
    ),
    (
        "MTLCommandQueue",
        "commandBuffer",
        "@16@0:8",
        ReturnNullability::Nullable,
    ),
    (
        "MTLComputeCommandEncoder",
        "dispatchThreads:threadsPerThreadgroup:",
        "v64@0:8{?=QQQ}16{?=QQQ}40",
        ReturnNullability::NotApplicable,
    ),
    (
        "MTLComputeCommandEncoder",
        "endEncoding",
        "v16@0:8",
        ReturnNullability::NotApplicable,
    ),
    (
        "MTLComputeCommandEncoder",
        "setBuffer:offset:atIndex:",
        "v40@0:8@16Q24Q32",
        ReturnNullability::NotApplicable,
    ),
    (
        "MTLComputeCommandEncoder",
        "setBytes:length:atIndex:",
        "v40@0:8r^v16Q24Q32",
        ReturnNullability::NotApplicable,
    ),
    (
        "MTLComputeCommandEncoder",
        "setComputePipelineState:",
        "v24@0:8@16",
        ReturnNullability::NotApplicable,
    ),
    (
        "MTLComputePipelineState",
        "maxTotalThreadsPerThreadgroup",
        "Q16@0:8",
        ReturnNullability::NotApplicable,
    ),
    (
        "MTLComputePipelineState",
        "threadExecutionWidth",
        "Q16@0:8",
        ReturnNullability::NotApplicable,
    ),
    (
        "MTLDevice",
        "hasUnifiedMemory",
        "B16@0:8",
        ReturnNullability::NotApplicable,
    ),
    ("MTLDevice", "name", "@16@0:8", ReturnNullability::Nonnull),
    (
        "MTLDevice",
        "newBufferWithLength:options:",
        "@32@0:8Q16Q24",
        ReturnNullability::Nullable,
    ),
    (
        "MTLDevice",
        "newCommandQueueWithMaxCommandBufferCount:",
        "@24@0:8Q16",
        ReturnNullability::Nullable,
    ),
    (
        "MTLDevice",
        "newComputePipelineStateWithFunction:error:",
        "@32@0:8@16^@24",
        ReturnNullability::Nullable,
    ),
    (
        "MTLDevice",
        "newLibraryWithData:error:",
        "@32@0:8@16^@24",
        ReturnNullability::Nullable,
    ),
    (
        "MTLDevice",
        "recommendedMaxWorkingSetSize",
        "Q16@0:8",
        ReturnNullability::NotApplicable,
    ),
    (
        "MTLDevice",
        "registryID",
        "Q16@0:8",
        ReturnNullability::NotApplicable,
    ),
    (
        "MTLLibrary",
        "newFunctionWithName:",
        "@24@0:8@16",
        ReturnNullability::Nullable,
    ),
    (
        "NSError",
        "code",
        "q16@0:8",
        ReturnNullability::NotApplicable,
    ),
];

const REVIEWED_HEADERS: [(&str, &str); 12] = [
    (
        "System/Library/Frameworks/Foundation.framework/Headers/NSError.h",
        "38eb1743d3fe217c51c438de7d1ba4b8dd87d928e31948d7511147eee80ca9ee",
    ),
    (
        "System/Library/Frameworks/Metal.framework/Headers/MTLBlitCommandEncoder.h",
        "46995fb123b4fa11934565cc5423aa850bab890d2b915dc3c6dffea804470037",
    ),
    (
        "System/Library/Frameworks/Metal.framework/Headers/MTLBuffer.h",
        "733a9cdaf29cf00e106f226afa119ea56efda456cd07bb1b87d354d5ced3f63a",
    ),
    (
        "System/Library/Frameworks/Metal.framework/Headers/MTLCommandBuffer.h",
        "a5a1ead36de20b4c1f4dce8193b945fb981e67456ca4a3a1395d167c9df475e7",
    ),
    (
        "System/Library/Frameworks/Metal.framework/Headers/MTLCommandEncoder.h",
        "3b074115dd01ec1d33155ef8d68ede4373ea471ea0a2be660344845d6c36f35b",
    ),
    (
        "System/Library/Frameworks/Metal.framework/Headers/MTLCommandQueue.h",
        "6e9da7a33496e30ef20cd415a51dfc72cf1d3f6892b1d8d7e8cfdeab451f5761",
    ),
    (
        "System/Library/Frameworks/Metal.framework/Headers/MTLComputeCommandEncoder.h",
        "610bcf8f3e6cb6a7067622f4395d8aa292c56226afde457ac6cb902937872b7b",
    ),
    (
        "System/Library/Frameworks/Metal.framework/Headers/MTLComputePipeline.h",
        "4109810fedcc753e9e92e50e13ba22bc150e389f3cbe89163bb1d055b8939099",
    ),
    (
        "System/Library/Frameworks/Metal.framework/Headers/MTLDevice.h",
        "41c2dbbb0e572e0cd8ea64022a649bfcea51a880f74ff2194d6ee7a68377f36d",
    ),
    (
        "System/Library/Frameworks/Metal.framework/Headers/MTLLibrary.h",
        "9b190787cc3ca1148c96c9914df805ceb926038b2acbbc41f38fb9003fea9893",
    ),
    (
        "System/Library/Frameworks/Metal.framework/Headers/MTLResource.h",
        "6100bf17139327113e3a0e6782f1cd3f08493eee37432f92afb8c37bd39dca78",
    ),
    (
        "System/Library/Frameworks/Metal.framework/Headers/MTLTypes.h",
        "835ba381014a9a666b61dfa9277c6213b6f73cec9d2d64f1a567bb5cd51de1ec",
    ),
];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MetalExecutionAbi {
    pub schema_version: u32,
    pub contract: String,
    pub backend: String,
    pub abi_floor: String,
    pub targets: Vec<String>,
    pub reviewed_sdk_version: String,
    pub reviewed_xcode_build: String,
    pub unsafe_owner: String,
    pub runtime_compilation_forbidden: bool,
    pub resource_selectors: Vec<ResourceSelectorContract>,
    pub reviewed_headers: Vec<ReviewedHeaderContract>,
    pub kernels: Vec<MetalKernelContract>,
    pub storage_modes: StorageModeContract,
    pub maximum_command_buffers_per_stream: usize,
    pub source_sha256: String,
    pub metallib_sha256_by_target: BTreeMap<String, String>,
    pub authorization_state: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ResourceSelectorContract {
    pub class: String,
    pub selector: String,
    pub encoding: String,
    pub return_nullability: ReturnNullability,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ReturnNullability {
    NotApplicable,
    Nonnull,
    Nullable,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReviewedHeaderContract {
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MetalKernelContract {
    pub dtype: String,
    pub function: String,
    pub input_buffers: Vec<u64>,
    pub output_buffer: u64,
    pub element_count_buffer: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StorageModeContract {
    pub unified_memory: String,
    pub discrete_memory: String,
    pub managed_cpu_write_requires_did_modify_range: bool,
    pub managed_gpu_write_requires_synchronize_resource: bool,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum MetalExecutionAbiError {
    #[error("Metal execution ABI is not strict JSON: {0}")]
    Json(String),
    #[error("Metal execution ABI violates the reviewed contract: {0}")]
    Contract(String),
}

impl MetalExecutionAbi {
    pub fn embedded() -> Result<Self, MetalExecutionAbiError> {
        let contract = serde_json::from_str::<Self>(EXECUTION_ABI_JSON)
            .map_err(|error| MetalExecutionAbiError::Json(error.to_string()))?;
        contract.validate()?;
        Ok(contract)
    }

    pub fn validate(&self) -> Result<(), MetalExecutionAbiError> {
        if self.schema_version != 1
            || self.contract != EXECUTION_CONTRACT
            || self.backend != "metal"
            || self.abi_floor != crate::ABI_FLOOR
            || self.reviewed_sdk_version != "26.2"
            || self.reviewed_xcode_build != "17C52"
            || self.unsafe_owner != EXECUTION_UNSAFE_OWNER
            || !self.runtime_compilation_forbidden
            || self.authorization_state != "requires-native-ffi-registry-and-signed-package"
        {
            return Err(MetalExecutionAbiError::Contract(
                "identity, floor, toolchain, unsafe owner, compilation, or authorization differs"
                    .to_owned(),
            ));
        }
        if self.targets != ["aarch64-apple-darwin", "x86_64-apple-darwin"] {
            return Err(MetalExecutionAbiError::Contract(
                "targets differ from the two reviewed Darwin targets".to_owned(),
            ));
        }
        if self.maximum_command_buffers_per_stream != MAXIMUM_COMMAND_BUFFERS_PER_STREAM {
            return Err(MetalExecutionAbiError::Contract(
                "command-buffer bound differs".to_owned(),
            ));
        }
        if self.storage_modes
            != (StorageModeContract {
                unified_memory: "shared".to_owned(),
                discrete_memory: "managed".to_owned(),
                managed_cpu_write_requires_did_modify_range: true,
                managed_gpu_write_requires_synchronize_resource: true,
            })
        {
            return Err(MetalExecutionAbiError::Contract(
                "storage synchronization rules differ".to_owned(),
            ));
        }
        if self.kernels
            != [
                MetalKernelContract {
                    dtype: "f16".to_owned(),
                    function: METAL_ADD_F16_FUNCTION.to_owned(),
                    input_buffers: vec![0, 1],
                    output_buffer: 2,
                    element_count_buffer: 3,
                },
                MetalKernelContract {
                    dtype: "f32".to_owned(),
                    function: METAL_ADD_F32_FUNCTION.to_owned(),
                    input_buffers: vec![0, 1],
                    output_buffer: 2,
                    element_count_buffer: 3,
                },
            ]
        {
            return Err(MetalExecutionAbiError::Contract(
                "kernel bindings differ".to_owned(),
            ));
        }
        if self.resource_selectors.len() != RESOURCE_SELECTORS.len()
            || self
                .resource_selectors
                .iter()
                .zip(RESOURCE_SELECTORS)
                .any(|(actual, expected)| {
                    actual.class != expected.0
                        || actual.selector != expected.1
                        || actual.encoding != expected.2
                        || actual.return_nullability != expected.3
                })
        {
            return Err(MetalExecutionAbiError::Contract(
                "resource selector names, exact encodings, or return nullability differ".to_owned(),
            ));
        }
        if self.reviewed_headers.len() != REVIEWED_HEADERS.len()
            || self
                .reviewed_headers
                .iter()
                .zip(REVIEWED_HEADERS)
                .any(|(actual, expected)| actual.path != expected.0 || actual.sha256 != expected.1)
        {
            return Err(MetalExecutionAbiError::Contract(
                "reviewed SDK header identities differ".to_owned(),
            ));
        }
        if !is_sha256(&self.source_sha256)
            || self.metallib_sha256_by_target.len() != self.targets.len()
            || self.targets.iter().any(|target| {
                self.metallib_sha256_by_target
                    .get(target)
                    .is_none_or(|digest| !is_sha256(digest))
            })
        {
            return Err(MetalExecutionAbiError::Contract(
                "source or target metallib digest coverage is invalid".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn metallib_digest_for_target(&self, target: &str) -> Option<&str> {
        self.metallib_sha256_by_target
            .get(target)
            .map(String::as_str)
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_execution_abi_is_complete_and_separate_from_foundation_contract() {
        let contract = MetalExecutionAbi::embedded().expect("execution ABI must validate");
        assert_eq!(contract.resource_selectors.len(), 29);
        assert_eq!(contract.reviewed_headers.len(), 12);
        assert_eq!(contract.kernels.len(), 2);
        assert!(!crate::ABI_MANIFEST_JSON.contains(EXECUTION_CONTRACT));
    }

    #[test]
    fn execution_abi_rejects_unknown_fields_and_authorization_changes() {
        let unknown = EXECUTION_ABI_JSON.replacen("{", "{\"unknown\":true,", 1);
        assert!(serde_json::from_str::<MetalExecutionAbi>(&unknown).is_err());
        let changed = EXECUTION_ABI_JSON.replace(
            "requires-native-ffi-registry-and-signed-package",
            "self-authorized",
        );
        let changed = serde_json::from_str::<MetalExecutionAbi>(&changed)
            .expect("changed fixture remains structural JSON");
        assert!(changed.validate().is_err());

        let changed = EXECUTION_ABI_JSON.replace(
            "\"status\", \"encoding\": \"Q16@0:8\"",
            "\"status\", \"encoding\": \"I16@0:8\"",
        );
        let changed = serde_json::from_str::<MetalExecutionAbi>(&changed)
            .expect("encoding mutation remains structural JSON");
        assert!(changed.validate().is_err());

        let changed = EXECUTION_ABI_JSON.replace("threadExecutionWidth", "threadExecutionWidth2");
        let changed = serde_json::from_str::<MetalExecutionAbi>(&changed)
            .expect("selector mutation remains structural JSON");
        assert!(changed.validate().is_err());

        let changed = EXECUTION_ABI_JSON.replacen(
            "\"return_nullability\": \"nullable\"",
            "\"return_nullability\": \"nonnull\"",
            1,
        );
        let changed = serde_json::from_str::<MetalExecutionAbi>(&changed)
            .expect("nullability mutation remains structural JSON");
        assert!(changed.validate().is_err());
    }
}
