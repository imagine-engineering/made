use crate::{NodeRegistry, NodeRegistryError};
use serde::Serialize;
use std::{error::Error, fmt};

pub const IMAGE_SLICE_NODE_IDS: &[&str] = &[
    "LoadImage",
    "ImageScale",
    "ImageInvert",
    "PreviewImage",
    "SaveImage",
];

pub const DIFFUSION_SLICE_NODE_IDS: &[&str] = &[
    "CheckpointLoaderSimple",
    "CLIPTextEncode",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
    "SaveImage",
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EarlySliceRegistry {
    image: Vec<String>,
    diffusion: Vec<String>,
}

impl EarlySliceRegistry {
    pub fn from_node_registry(registry: &NodeRegistry) -> Result<Self, SliceRegistryError> {
        let image = validate_slice("image", IMAGE_SLICE_NODE_IDS, registry, true)?;
        let diffusion = validate_slice("diffusion", DIFFUSION_SLICE_NODE_IDS, registry, true)?;
        Ok(Self { image, diffusion })
    }

    pub fn built_in() -> Result<Self, SliceRegistryError> {
        let registry = NodeRegistry::built_in()?;
        Self::from_node_registry(&registry)
    }

    pub fn image(&self) -> &[String] {
        &self.image
    }

    pub fn diffusion(&self) -> &[String] {
        &self.diffusion
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SliceRegistryError {
    NodeRegistry(NodeRegistryError),
    DuplicateNode {
        slice: &'static str,
        identifier: String,
    },
    UnknownNode {
        slice: &'static str,
        identifier: String,
    },
    MissingCompiledDescriptor {
        slice: &'static str,
        identifier: String,
    },
}

impl fmt::Display for SliceRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeRegistry(error) => error.fmt(formatter),
            Self::DuplicateNode { slice, identifier } => {
                write!(formatter, "{slice} slice repeats node `{identifier}`")
            }
            Self::UnknownNode { slice, identifier } => {
                write!(
                    formatter,
                    "{slice} slice references unknown node `{identifier}`"
                )
            }
            Self::MissingCompiledDescriptor { slice, identifier } => {
                write!(
                    formatter,
                    "{slice} slice has no compiled descriptor for node `{identifier}`"
                )
            }
        }
    }
}

impl Error for SliceRegistryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::NodeRegistry(error) => Some(error),
            _ => None,
        }
    }
}

impl From<NodeRegistryError> for SliceRegistryError {
    fn from(error: NodeRegistryError) -> Self {
        Self::NodeRegistry(error)
    }
}

fn validate_slice(
    slice: &'static str,
    identifiers: &[&str],
    registry: &NodeRegistry,
    require_compiled_descriptor: bool,
) -> Result<Vec<String>, SliceRegistryError> {
    let mut values = Vec::with_capacity(identifiers.len());
    for identifier in identifiers {
        if values.iter().any(|value| value == identifier) {
            return Err(SliceRegistryError::DuplicateNode {
                slice,
                identifier: (*identifier).to_owned(),
            });
        }
        registry
            .descriptor(identifier)
            .ok_or_else(|| SliceRegistryError::UnknownNode {
                slice,
                identifier: (*identifier).to_owned(),
            })?;
        if require_compiled_descriptor && !crate::GENERATED_DESCRIPTOR_IDS.contains(identifier) {
            return Err(SliceRegistryError::MissingCompiledDescriptor {
                slice,
                identifier: (*identifier).to_owned(),
            });
        }
        values.push((*identifier).to_owned());
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn early_slices_have_exact_ordered_membership() -> Result<(), Box<dyn Error>> {
        let slices = EarlySliceRegistry::built_in()?;
        assert_eq!(
            slices.image(),
            IMAGE_SLICE_NODE_IDS
                .iter()
                .map(|identifier| (*identifier).to_owned())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            slices.diffusion(),
            DIFFUSION_SLICE_NODE_IDS
                .iter()
                .map(|identifier| (*identifier).to_owned())
                .collect::<Vec<_>>()
        );
        assert_eq!(slices.image().len(), 5);
        assert_eq!(slices.diffusion().len(), 6);
        Ok(())
    }
}
