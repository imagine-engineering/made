use serde::{Deserialize, Serialize};
use sim_types::DeviceKind;
use std::{collections::HashSet, fmt};
use thiserror::Error;

pub const MODEL_DESCRIPTOR_SCHEMA_VERSION: u16 = 1;
const MAX_DESCRIPTOR_TEXT_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelComponentDescriptor {
    pub identifier: String,
    pub role: String,
    pub required: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorKeyRule {
    pub source_prefix: String,
    pub target_prefix: String,
    pub required: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryEstimatorDescriptor {
    pub fixed_bytes: u64,
    pub bytes_per_parameter: u32,
    pub activation_bytes_per_element: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDescriptor {
    pub schema_version: u16,
    pub identifier: String,
    pub family: String,
    pub architecture_version: String,
    pub latent_format: String,
    pub component_graph: Vec<ModelComponentDescriptor>,
    pub tensor_key_rules: Vec<TensorKeyRule>,
    pub required_keys: Vec<String>,
    pub optional_keys: Vec<String>,
    pub supported_dtypes: Vec<String>,
    pub supported_devices: Vec<DeviceKind>,
    pub memory_estimator: MemoryEstimatorDescriptor,
}

impl ModelDescriptor {
    pub fn validate(&self) -> Result<(), ModelDescriptorError> {
        if self.schema_version != MODEL_DESCRIPTOR_SCHEMA_VERSION {
            return Err(ModelDescriptorError::UnsupportedSchema(self.schema_version));
        }
        for (field, value) in [
            ("identifier", self.identifier.as_str()),
            ("family", self.family.as_str()),
            ("architecture_version", self.architecture_version.as_str()),
            ("latent_format", self.latent_format.as_str()),
        ] {
            validate_text(field, value)?;
        }

        if self.component_graph.is_empty() {
            return Err(ModelDescriptorError::EmptyCollection("component_graph"));
        }
        let mut component_identifiers = HashSet::new();
        for component in &self.component_graph {
            validate_text("component.identifier", &component.identifier)?;
            validate_text("component.role", &component.role)?;
            if !component_identifiers.insert(component.identifier.as_str()) {
                return Err(ModelDescriptorError::DuplicateValue(
                    "component.identifier",
                    component.identifier.clone(),
                ));
            }
        }

        if self.tensor_key_rules.is_empty() {
            return Err(ModelDescriptorError::EmptyCollection("tensor_key_rules"));
        }
        let mut source_prefixes = HashSet::new();
        for rule in &self.tensor_key_rules {
            validate_text("tensor_key_rule.source_prefix", &rule.source_prefix)?;
            validate_text("tensor_key_rule.target_prefix", &rule.target_prefix)?;
            if !source_prefixes.insert(rule.source_prefix.as_str()) {
                return Err(ModelDescriptorError::DuplicateValue(
                    "tensor_key_rule.source_prefix",
                    rule.source_prefix.clone(),
                ));
            }
        }

        let required_keys = validate_unique_text("required_key", &self.required_keys)?;
        let optional_keys = validate_unique_text("optional_key", &self.optional_keys)?;
        if let Some(key) = required_keys.intersection(&optional_keys).next() {
            return Err(ModelDescriptorError::ConflictingKey((*key).to_owned()));
        }
        validate_unique_text("supported_dtype", &self.supported_dtypes)?;
        if self.supported_devices.is_empty() {
            return Err(ModelDescriptorError::EmptyCollection("supported_devices"));
        }
        let mut devices = HashSet::new();
        for device in &self.supported_devices {
            if !devices.insert(*device) {
                return Err(ModelDescriptorError::DuplicateDevice(*device));
            }
        }
        if self.memory_estimator.bytes_per_parameter == 0
            || self.memory_estimator.activation_bytes_per_element == 0
        {
            return Err(ModelDescriptorError::InvalidMemoryEstimator);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ModelDescriptorError {
    #[error("unsupported model descriptor schema version {0}")]
    UnsupportedSchema(u16),
    #[error("model descriptor field `{0}` is empty or invalid")]
    InvalidText(&'static str),
    #[error("model descriptor collection `{0}` is empty")]
    EmptyCollection(&'static str),
    #[error("model descriptor field `{0}` repeats `{1}`")]
    DuplicateValue(&'static str, String),
    #[error("model descriptor key `{0}` is both required and optional")]
    ConflictingKey(String),
    #[error("model descriptor repeats device `{0:?}`")]
    DuplicateDevice(DeviceKind),
    #[error("model descriptor memory estimator must use non-zero element sizes")]
    InvalidMemoryEstimator,
}

fn validate_text(field: &'static str, value: &str) -> Result<(), ModelDescriptorError> {
    if value.is_empty()
        || value.len() > MAX_DESCRIPTOR_TEXT_BYTES
        || value.chars().any(char::is_control)
    {
        Err(ModelDescriptorError::InvalidText(field))
    } else {
        Ok(())
    }
}

fn validate_unique_text<'a>(
    field: &'static str,
    values: &'a [String],
) -> Result<HashSet<&'a str>, ModelDescriptorError> {
    if values.is_empty() {
        return Err(ModelDescriptorError::EmptyCollection(field));
    }
    let mut unique = HashSet::new();
    for value in values {
        validate_text(field, value)?;
        if !unique.insert(value.as_str()) {
            return Err(ModelDescriptorError::DuplicateValue(field, value.clone()));
        }
    }
    Ok(unique)
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum ModelCatalogKind {
    #[serde(rename = "attention backend")]
    AttentionBackend,
    #[serde(rename = "dtype")]
    Dtype,
    #[serde(rename = "hardware backend")]
    HardwareBackend,
    #[serde(rename = "latent format")]
    LatentFormat,
    #[serde(rename = "memory mode")]
    MemoryMode,
    #[serde(rename = "model family")]
    ModelFamily,
    #[serde(rename = "quantization")]
    Quantization,
    #[serde(rename = "sampler")]
    Sampler,
    #[serde(rename = "scheduler")]
    Scheduler,
}

impl ModelCatalogKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AttentionBackend => "attention backend",
            Self::Dtype => "dtype",
            Self::HardwareBackend => "hardware backend",
            Self::LatentFormat => "latent format",
            Self::MemoryMode => "memory mode",
            Self::ModelFamily => "model family",
            Self::Quantization => "quantization",
            Self::Sampler => "sampler",
            Self::Scheduler => "scheduler",
        }
    }
}

impl fmt::Display for ModelCatalogKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelCatalogAvailability {
    Active,
    Conditional,
    PlatformSpecific,
}

impl ModelCatalogAvailability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Conditional => "conditional",
            Self::PlatformSpecific => "platform-specific",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelEvidenceLevel {
    CodeInferred,
    TestBacked,
}

impl ModelEvidenceLevel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CodeInferred => "code-inferred",
            Self::TestBacked => "test-backed",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelCatalogConfidence {
    High,
}

impl ModelCatalogConfidence {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::High => "high",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelParityStatus {
    Missing,
    Partial,
}

impl ModelParityStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::Partial => "partial",
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ModelCatalogKey {
    pub kind: ModelCatalogKind,
    pub name: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogModelDescriptor {
    pub schema_version: u16,
    pub kind: ModelCatalogKind,
    pub name: String,
    pub classification: String,
    pub availability: ModelCatalogAvailability,
    pub evidence_level: ModelEvidenceLevel,
    pub confidence: ModelCatalogConfidence,
    pub identifier_or_format: String,
    pub inputs_defaults: String,
    pub success_behavior: String,
    pub failure_behavior: String,
    pub dependencies_platform: String,
    pub source_file: String,
    pub source_symbol: String,
    pub source_line: Option<u32>,
    pub test_evidence: String,
    pub zed_status: ModelParityStatus,
    pub parity_gap: String,
    pub feature_id: String,
}

impl CatalogModelDescriptor {
    pub fn key(&self) -> ModelCatalogKey {
        ModelCatalogKey {
            kind: self.kind,
            name: self.name.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_model_descriptor_schema_round_trips() -> Result<(), Box<dyn std::error::Error>> {
        let descriptor = ModelDescriptor {
            schema_version: MODEL_DESCRIPTOR_SCHEMA_VERSION,
            identifier: "sd15".to_owned(),
            family: "SD15".to_owned(),
            architecture_version: "1".to_owned(),
            latent_format: "SD15".to_owned(),
            component_graph: vec![ModelComponentDescriptor {
                identifier: "unet".to_owned(),
                role: "denoiser".to_owned(),
                required: true,
            }],
            tensor_key_rules: vec![TensorKeyRule {
                source_prefix: "model.diffusion_model.".to_owned(),
                target_prefix: "denoiser.".to_owned(),
                required: true,
            }],
            required_keys: vec!["model.diffusion_model.input_blocks.0.0.weight".to_owned()],
            optional_keys: vec!["model_ema.".to_owned()],
            supported_dtypes: vec!["float32".to_owned()],
            supported_devices: vec![DeviceKind::Cpu],
            memory_estimator: MemoryEstimatorDescriptor {
                fixed_bytes: 0,
                bytes_per_parameter: 4,
                activation_bytes_per_element: 4,
            },
        };
        let encoded = serde_json::to_vec(&descriptor)?;
        descriptor.validate()?;
        assert_eq!(
            serde_json::from_slice::<ModelDescriptor>(&encoded)?,
            descriptor
        );
        Ok(())
    }

    #[test]
    fn invalid_native_model_descriptors_fail_closed() {
        let mut descriptor = ModelDescriptor {
            schema_version: MODEL_DESCRIPTOR_SCHEMA_VERSION + 1,
            identifier: "sd15".to_owned(),
            family: "SD15".to_owned(),
            architecture_version: "1".to_owned(),
            latent_format: "SD15".to_owned(),
            component_graph: Vec::new(),
            tensor_key_rules: Vec::new(),
            required_keys: vec!["weight".to_owned()],
            optional_keys: vec!["optional".to_owned()],
            supported_dtypes: vec!["float32".to_owned()],
            supported_devices: vec![DeviceKind::Cpu],
            memory_estimator: MemoryEstimatorDescriptor {
                fixed_bytes: 0,
                bytes_per_parameter: 4,
                activation_bytes_per_element: 4,
            },
        };
        assert!(matches!(
            descriptor.validate(),
            Err(ModelDescriptorError::UnsupportedSchema(_))
        ));
        descriptor.schema_version = MODEL_DESCRIPTOR_SCHEMA_VERSION;
        assert!(matches!(
            descriptor.validate(),
            Err(ModelDescriptorError::EmptyCollection("component_graph"))
        ));
        descriptor.component_graph = vec![ModelComponentDescriptor {
            identifier: "unet".to_owned(),
            role: "denoiser".to_owned(),
            required: true,
        }];
        descriptor.tensor_key_rules = vec![TensorKeyRule {
            source_prefix: "model.".to_owned(),
            target_prefix: "denoiser.".to_owned(),
            required: true,
        }];
        descriptor.optional_keys = vec!["weight".to_owned()];
        assert!(matches!(
            descriptor.validate(),
            Err(ModelDescriptorError::ConflictingKey(_))
        ));
    }
}
