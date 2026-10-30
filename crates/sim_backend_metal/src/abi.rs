use serde::Deserialize;
use thiserror::Error;

pub const ABI_MANIFEST: &str = include_str!("../abi/symbols-v1.json");
pub const ABI_FLOOR: &str = "macos-13-metal-3";
pub const READINESS_FUNCTION: &str = "zed_sim_metal_readiness_v1";
pub const UNSAFE_OWNER: &str = "sim_backend_metal::loader";
pub const METAL_3_FAMILY_VALUE: u64 = 5_001;
pub const MPS_DATA_TYPE_FLOAT16: u32 = 0x1000_0010;
pub const MPS_DATA_TYPE_FLOAT32: u32 = 0x1000_0020;

const FRAMEWORKS: [(&str, &str, &str, &[&str]); 3] = [
    (
        "Metal",
        "/System/Library/Frameworks/Metal.framework/Metal",
        "/System/Library/Frameworks/Metal.framework/Versions/A/Metal",
        &["MTLCreateSystemDefaultDevice"],
    ),
    (
        "MetalPerformanceShaders",
        "/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders",
        "/System/Library/Frameworks/MetalPerformanceShaders.framework/Versions/A/MetalPerformanceShaders",
        &["MPSSupportsMTLDevice"],
    ),
    (
        "MetalPerformanceShadersGraph",
        "/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/MetalPerformanceShadersGraph",
        "/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/Versions/A/MetalPerformanceShadersGraph",
        &[],
    ),
];

const MPS_GRAPH_SELECTORS: &[(SelectorKind, &str, &str)] = &[
    (SelectorKind::Class, "new", "@16@0:8"),
    (
        SelectorKind::Instance,
        "additionWithPrimaryTensor:secondaryTensor:name:",
        "@40@0:8@16@24@32",
    ),
    (
        SelectorKind::Instance,
        "compileWithDevice:feeds:targetTensors:targetOperations:compilationDescriptor:",
        "@56@0:8@16@24@32@40@48",
    ),
    (
        SelectorKind::Instance,
        "constantWithScalar:dataType:",
        "@28@0:8d16I24",
    ),
    (SelectorKind::Instance, "init", "@16@0:8"),
    (
        SelectorKind::Instance,
        "placeholderWithShape:dataType:name:",
        "@36@0:8@16I24@28",
    ),
    (
        SelectorKind::Instance,
        "runWithFeeds:targetTensors:targetOperations:",
        "@40@0:8@16@24@32",
    ),
    (
        SelectorKind::Instance,
        "runWithMTLCommandQueue:feeds:targetTensors:targetOperations:",
        "@48@0:8@16@24@32@40",
    ),
];
const MPS_GRAPH_DEVICE_SELECTORS: &[(SelectorKind, &str, &str)] = &[
    (SelectorKind::Class, "deviceWithMTLDevice:", "@24@0:8@16"),
    (SelectorKind::Instance, "metalDevice", "@16@0:8"),
];
const MPS_GRAPH_TENSOR_DATA_SELECTORS: &[(SelectorKind, &str, &str)] = &[
    (
        SelectorKind::Instance,
        "initWithMTLBuffer:shape:dataType:",
        "@36@0:8@16@24I32",
    ),
    (
        SelectorKind::Instance,
        "initWithMTLBuffer:shape:dataType:rowBytes:",
        "@44@0:8@16@24I32Q36",
    ),
];
const CLASSES: [(&str, &str, &[(SelectorKind, &str, &str)]); 3] = [
    (
        "MPSGraph",
        "/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/Versions/A/MetalPerformanceShadersGraph",
        MPS_GRAPH_SELECTORS,
    ),
    (
        "MPSGraphDevice",
        "/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/Versions/A/MetalPerformanceShadersGraph",
        MPS_GRAPH_DEVICE_SELECTORS,
    ),
    (
        "MPSGraphTensorData",
        "/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/Versions/A/MetalPerformanceShadersGraph",
        MPS_GRAPH_TENSOR_DATA_SELECTORS,
    ),
];
const LAYOUTS: [(&str, usize, usize); 5] = [
    ("MPSDataType", 4, 4),
    ("MTLOrigin", 24, 8),
    ("MTLRegion", 48, 8),
    ("MTLSize", 24, 8),
    ("Objective-C BOOL", 1, 1),
];
const HEADERS: [(&str, &str); 12] = [
    (
        "Metal/MTLDevice.h",
        "41c2dbbb0e572e0cd8ea64022a649bfcea51a880f74ff2194d6ee7a68377f36d",
    ),
    (
        "Metal/MTLTypes.h",
        "835ba381014a9a666b61dfa9277c6213b6f73cec9d2d64f1a567bb5cd51de1ec",
    ),
    (
        "MetalPerformanceShaders/MetalPerformanceShaders.h",
        "06e3752402c2f317d0477093662e41323121497a1317ca838ed4282afeb380f4",
    ),
    (
        "MPSCore/MPSCoreTypes.h",
        "20b2f712c77dfaacec665b1070470373f873b1c1f3206a6f930c7ace8ab2dcf2",
    ),
    (
        "MPSGraph/MPSGraph.h",
        "45e2e964841078426524210f1458e98a46cafbb1ea10394137bb3a90c1fbe154",
    ),
    (
        "MPSGraph/MPSGraphArithmeticOps.h",
        "5b1c407099d58aba9604084045a84d28fb72438559965d5c99ff29da12595cde",
    ),
    (
        "MPSGraph/MPSGraphCore.h",
        "b8d58b60d15042e63f6960c84e6e4db8098aa8d98af2312a90c8590e86332b05",
    ),
    (
        "MPSGraph/MPSGraphDevice.h",
        "ed4e690f455fb93d96754c9b31967a6fdff2150d5c5a4226dc8649810326d272",
    ),
    (
        "MPSGraph/MPSGraphMemoryOps.h",
        "42b3e91d664c0dec1df508716f1a6b55158f7b2a0162df32be20f6b0fc3d3eb6",
    ),
    (
        "MPSGraph/MPSGraphTensorData.h",
        "29cc86c667009d31a469ca63da2e87705c4246445cae7a530bcf5d298879ff73",
    ),
    (
        "usr/include/dlfcn.h",
        "04294faa7f4d8f08cef5b11c0813595ba0e87aa345677576ff62423aefa77e33",
    ),
    (
        "usr/include/objc/runtime.h",
        "e34d98bb24db821fde2a06653daa66b72cbeaa8baa55a38c34a6fbdba1adef8c",
    ),
];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AbiManifest {
    pub schema_version: u32,
    pub backend: String,
    pub abi_floor: String,
    pub targets: Vec<String>,
    pub frameworks: Vec<FrameworkContract>,
    pub classes: Vec<ClassContract>,
    pub layouts: Vec<LayoutContract>,
    pub headers: Vec<HeaderContract>,
    pub readiness_function: String,
    pub unsafe_owner: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FrameworkContract {
    pub name: String,
    pub install_name: String,
    pub image_name: String,
    pub symbols: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ClassContract {
    pub name: String,
    pub image_name: String,
    pub selectors: Vec<SelectorContract>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelectorContract {
    pub kind: SelectorKind,
    pub name: String,
    pub encoding: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SelectorKind {
    Class,
    Instance,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LayoutContract {
    pub name: String,
    pub size: usize,
    pub align: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HeaderContract {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum AbiManifestError {
    #[error("Metal ABI manifest is not strict JSON: {0}")]
    Json(String),
    #[error("Metal ABI manifest violates the reviewed contract: {0}")]
    Contract(String),
}

impl AbiManifest {
    pub fn embedded() -> Result<Self, AbiManifestError> {
        let manifest: Self = serde_json::from_str(ABI_MANIFEST)
            .map_err(|error| AbiManifestError::Json(error.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), AbiManifestError> {
        if self.schema_version != 1
            || self.backend != "metal"
            || self.abi_floor != ABI_FLOOR
            || self.readiness_function != READINESS_FUNCTION
            || self.unsafe_owner != UNSAFE_OWNER
        {
            return Err(AbiManifestError::Contract(
                "identity, version, readiness function, or unsafe owner differs".to_owned(),
            ));
        }
        require_sorted_unique(&self.targets, "targets")?;
        if self.targets != ["aarch64-apple-darwin", "x86_64-apple-darwin"] {
            return Err(AbiManifestError::Contract(
                "targets must be the two reviewed 64-bit Darwin targets".to_owned(),
            ));
        }
        if self.frameworks.len() != FRAMEWORKS.len()
            || self.classes.len() != CLASSES.len()
            || self.layouts.len() != LAYOUTS.len()
            || self.headers.len() != HEADERS.len()
        {
            return Err(AbiManifestError::Contract(
                "framework, class, layout, or header coverage is incomplete".to_owned(),
            ));
        }
        for (framework, expected) in self.frameworks.iter().zip(FRAMEWORKS) {
            if framework.name != expected.0
                || framework.install_name != expected.1
                || framework.image_name != expected.2
                || framework
                    .symbols
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    != expected.3
            {
                return Err(AbiManifestError::Contract(format!(
                    "framework contract {} differs from the reviewed ABI",
                    framework.name
                )));
            }
        }
        for (class, expected) in self.classes.iter().zip(CLASSES) {
            let selectors = class
                .selectors
                .iter()
                .map(|selector| {
                    (
                        selector.kind,
                        selector.name.as_str(),
                        selector.encoding.as_str(),
                    )
                })
                .collect::<Vec<_>>();
            if class.name != expected.0 || class.image_name != expected.1 || selectors != expected.2
            {
                return Err(AbiManifestError::Contract(format!(
                    "Objective-C class contract {} differs from the reviewed ABI",
                    class.name
                )));
            }
        }
        for (layout, expected) in self.layouts.iter().zip(LAYOUTS) {
            if layout.name != expected.0 || layout.size != expected.1 || layout.align != expected.2
            {
                return Err(AbiManifestError::Contract(format!(
                    "layout contract {} differs from the reviewed ABI",
                    layout.name
                )));
            }
        }
        for (header, expected) in self.headers.iter().zip(HEADERS) {
            if header.path != expected.0 || header.sha256 != expected.1 {
                return Err(AbiManifestError::Contract(format!(
                    "header contract {} differs from the reviewed ABI",
                    header.path
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> AbiManifest {
        serde_json::from_str(ABI_MANIFEST).expect("embedded fixture must parse")
    }

    #[test]
    fn exact_reviewed_contract_rejects_count_preserving_substitutions() {
        let mut cases: Vec<AbiManifest> = Vec::new();

        let mut changed = manifest();
        changed.frameworks[0].install_name.push_str(".other");
        cases.push(changed);
        let mut changed = manifest();
        changed.classes[0].name.push_str("Other");
        cases.push(changed);
        let mut changed = manifest();
        changed.classes[0].selectors[0].name.push_str(":");
        cases.push(changed);
        let mut changed = manifest();
        changed.classes[0].selectors[0].encoding.push('x');
        cases.push(changed);
        let mut changed = manifest();
        changed.layouts[0].size += 1;
        cases.push(changed);
        let mut changed = manifest();
        changed.headers[0].path.push_str(".other");
        cases.push(changed);
        let mut changed = manifest();
        changed.headers[0].sha256.replace_range(..1, "0");
        cases.push(changed);
        let mut changed = manifest();
        changed.targets[0].push_str("-other");
        cases.push(changed);
        let mut changed = manifest();
        changed.unsafe_owner.push_str("::other");
        cases.push(changed);

        assert!(cases.into_iter().all(|case| case.validate().is_err()));
    }
}

fn require_sorted_unique(values: &[String], field: &str) -> Result<(), AbiManifestError> {
    if values.is_empty() || values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(AbiManifestError::Contract(format!(
            "{field} must be nonempty, sorted, and unique"
        )));
    }
    Ok(())
}
