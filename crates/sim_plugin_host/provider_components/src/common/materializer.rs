use std::{collections::BTreeMap, error::Error, fmt};

use sim_plugin_sdk::{CanonicalTypeId, PluginValue, TypeRegistry, ValueFamily};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedOutput {
    pub port_id: String,
    pub type_id: CanonicalTypeId,
    pub family: ValueFamily,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpectedOutput {
    pub port_id: String,
    pub type_id: CanonicalTypeId,
    pub family: ValueFamily,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterializedBatch {
    outputs: Vec<EncodedOutput>,
    resident_bytes: usize,
}

impl MaterializedBatch {
    pub fn outputs(&self) -> &[EncodedOutput] {
        &self.outputs
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
}

pub trait MaterializationSink {
    fn publish(&mut self, batch: MaterializedBatch) -> Result<(), MaterializationError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterializationError {
    InvalidSchema,
    MissingOutput,
    DuplicateOutput,
    TypeMismatch,
    EmptyValue,
    MalformedValue,
    OutputLimit,
    PublishFailed,
}

impl fmt::Display for MaterializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidSchema => "invalid materialization schema",
            Self::MissingOutput => "provider result is missing an output",
            Self::DuplicateOutput => "provider result has a duplicate output",
            Self::TypeMismatch => "provider output type does not match its port",
            Self::EmptyValue => "provider output has no encoded value",
            Self::MalformedValue => "provider output has malformed ABI bytes",
            Self::OutputLimit => "provider output exceeds its materialization bound",
            Self::PublishFailed => "provider output publication failed",
        })
    }
}

impl Error for MaterializationError {}

pub fn validate_outputs(
    expected: &[ExpectedOutput],
    outputs: Vec<EncodedOutput>,
    maximum_output_bytes: usize,
) -> Result<MaterializedBatch, MaterializationError> {
    if expected.len() > 1_024 || outputs.len() != expected.len() || maximum_output_bytes == 0 {
        return Err(MaterializationError::InvalidSchema);
    }
    let mut expected_by_port = BTreeMap::new();
    for output in expected {
        if !valid_identifier(&output.port_id)
            || !valid_type_id(&output.type_id)
            || expected_by_port
                .insert(output.port_id.as_str(), output)
                .is_some()
        {
            return Err(MaterializationError::InvalidSchema);
        }
    }

    let mut previous_port: Option<&str> = None;
    let mut resident_bytes = 0usize;
    for output in &outputs {
        if previous_port.is_some_and(|previous| previous >= output.port_id.as_str()) {
            return Err(MaterializationError::DuplicateOutput);
        }
        let expected = expected_by_port
            .get(output.port_id.as_str())
            .ok_or(MaterializationError::MissingOutput)?;
        if output.type_id != expected.type_id || output.family != expected.family {
            return Err(MaterializationError::TypeMismatch);
        }
        if output.bytes.is_empty() {
            return Err(MaterializationError::EmptyValue);
        }
        let registry = TypeRegistry::built_in().map_err(|_| MaterializationError::InvalidSchema)?;
        registry
            .require_family(&output.type_id, output.family)
            .map_err(|_| MaterializationError::TypeMismatch)?;
        let value = PluginValue::from_abi_bytes(&output.bytes, &registry)
            .map_err(|_| MaterializationError::MalformedValue)?;
        if value.type_id() != &output.type_id || value.family() != output.family {
            return Err(MaterializationError::TypeMismatch);
        }
        resident_bytes = resident_bytes
            .checked_add(output.bytes.len())
            .ok_or(MaterializationError::OutputLimit)?;
        if resident_bytes > maximum_output_bytes {
            return Err(MaterializationError::OutputLimit);
        }
        previous_port = Some(output.port_id.as_str());
    }

    Ok(MaterializedBatch {
        outputs,
        resident_bytes,
    })
}

pub fn publish_validated(
    batch: MaterializedBatch,
    sink: &mut dyn MaterializationSink,
) -> Result<(), MaterializationError> {
    sink.publish(batch)
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

fn valid_type_id(value: &CanonicalTypeId) -> bool {
    value.to_string().len() <= 256
}
