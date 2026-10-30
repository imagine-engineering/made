use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, error::Error, fmt, sync::OnceLock};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "LoadImage",
    "ImageScale",
    "ImageInvert",
    "PreviewImage",
    "SaveImage",
];
pub const NATIVE_IMAGE_DESCRIPTOR_SCHEMA_VERSION: u16 = 1;

const DESCRIPTORS: &str = include_str!("native_image.descriptors.json");
static PARSED_DESCRIPTORS: OnceLock<Result<Vec<NativeImageDescriptor>, NativeImageDescriptorError>> =
    OnceLock::new();

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeImagePort {
    pub name: String,
    pub type_name: String,
    pub required: bool,
    #[serde(default)]
    pub hidden: bool,
    #[serde(default)]
    pub choices: Vec<String>,
    #[serde(default)]
    pub options: BTreeMap<String, Value>,
    #[serde(default)]
    pub choices_from_input_assets: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeImageCatalogContract {
    pub inputs: String,
    pub outputs: String,
    pub input_is_list: String,
    pub output_is_list: String,
    pub lazy_inputs: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeImageEffect {
    Pure,
    ReadsArtifact,
    WritesArtifact,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeImageDescriptor {
    pub schema_version: u16,
    pub class_type: String,
    pub display_name: String,
    pub category: String,
    pub description: String,
    pub python_module: String,
    pub search_aliases: Vec<String>,
    pub essentials_category: Option<String>,
    pub has_intermediate_output: bool,
    pub implementation_version: String,
    pub inputs: Vec<NativeImagePort>,
    pub outputs: Vec<NativeImagePort>,
    pub output_node: bool,
    pub effect: NativeImageEffect,
    pub cache_by_input_identity: bool,
    pub catalog_contract: NativeImageCatalogContract,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeImageDescriptorError(String);

impl fmt::Display for NativeImageDescriptorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for NativeImageDescriptorError {}

pub fn native_image_descriptors()
-> Result<&'static [NativeImageDescriptor], NativeImageDescriptorError> {
    PARSED_DESCRIPTORS
        .get_or_init(parse_descriptors)
        .as_ref()
        .map(Vec::as_slice)
        .map_err(Clone::clone)
}

fn parse_descriptors() -> Result<Vec<NativeImageDescriptor>, NativeImageDescriptorError> {
    parse_and_validate_native_descriptors(
        DESCRIPTORS,
        NODE_DESCRIPTOR_IDS,
        NATIVE_IMAGE_DESCRIPTOR_SCHEMA_VERSION,
    )
}

pub fn parse_and_validate_native_descriptors(
    source: &str,
    expected_identifiers: &[&str],
    schema_version: u16,
) -> Result<Vec<NativeImageDescriptor>, NativeImageDescriptorError> {
    let descriptors: Vec<NativeImageDescriptor> = serde_json::from_str(source)
        .map_err(|error| NativeImageDescriptorError(error.to_string()))?;
    let catalog = crate::NodeRegistry::built_in()
        .map_err(|error| NativeImageDescriptorError(error.to_string()))?;
    if descriptors.len() != expected_identifiers.len() {
        return Err(NativeImageDescriptorError(format!(
            "native descriptor count is {}, expected {}",
            descriptors.len(),
            expected_identifiers.len()
        )));
    }
    for (descriptor, expected) in descriptors.iter().zip(expected_identifiers) {
        let catalog_descriptor = catalog.descriptor(expected).ok_or_else(|| {
            NativeImageDescriptorError(format!(
                "native image descriptor `{expected}` has no canonical catalog row"
            ))
        })?;
        if descriptor.schema_version != schema_version
            || descriptor.class_type != *expected
            || descriptor.display_name != catalog_descriptor.display_name
            || descriptor.category != catalog_descriptor.category
            || descriptor.output_node != catalog_descriptor.output_node
            || descriptor.catalog_contract.inputs != catalog_descriptor.inputs
            || descriptor.catalog_contract.outputs != catalog_descriptor.outputs
            || descriptor.catalog_contract.input_is_list != catalog_descriptor.input_is_list
            || descriptor.catalog_contract.output_is_list != catalog_descriptor.output_is_list
            || descriptor.catalog_contract.lazy_inputs != catalog_descriptor.lazy_inputs
            || descriptor.python_module.trim().is_empty()
            || descriptor.implementation_version.trim().is_empty()
            || descriptor.inputs.iter().any(|port| {
                port.name.trim().is_empty()
                    || port.type_name.trim().is_empty()
                    || (port.choices_from_input_assets && !port.choices.is_empty())
                    || (port.hidden
                        && (port.required
                            || !port.choices.is_empty()
                            || port.choices_from_input_assets
                            || !port.options.is_empty()))
            })
            || descriptor.outputs.iter().any(|port| {
                port.name.trim().is_empty()
                    || port.type_name.trim().is_empty()
                    || port.hidden
                    || !port.choices.is_empty()
                    || !port.options.is_empty()
                    || port.choices_from_input_assets
            })
            || descriptor.inputs.iter().any(|port| {
                !catalog_contract_contains_input(&descriptor.catalog_contract.inputs, &port.name)
            })
        {
            return Err(NativeImageDescriptorError(format!(
                "native image descriptor `{}` is invalid",
                descriptor.class_type
            )));
        }
    }
    Ok(descriptors)
}

fn catalog_contract_contains_input(contract: &str, name: &str) -> bool {
    contract.contains(&format!("'{name}'"))
        || contract.split(" | ").any(|entry| {
            entry.split_once(':').is_some_and(|(_, field)| {
                field
                    .split_once('(')
                    .map_or(field, |(field, _)| field)
                    == name
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_has_exact_executable_image_slice() -> Result<(), NativeImageDescriptorError> {
        let descriptors = native_image_descriptors()?;
        assert_eq!(
            descriptors
                .iter()
                .map(|descriptor| descriptor.class_type.as_str())
                .collect::<Vec<_>>(),
            NODE_DESCRIPTOR_IDS
        );
        assert_eq!(descriptors.iter().filter(|descriptor| descriptor.output_node).count(), 2);
        assert!(descriptors.iter().all(|descriptor| {
            descriptor
                .outputs
                .iter()
                .all(|output| output.type_name == "IMAGE" || output.type_name == "MASK")
        }));
        Ok(())
    }
}
