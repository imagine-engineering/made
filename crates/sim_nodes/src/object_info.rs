use crate::{CatalogNodeDescriptor, CatalogNodeSchemaMetadata, CatalogNodeStatus, NodeRegistry};
use serde::Serialize;
use std::collections::BTreeMap;

pub const OBJECT_INFO_SCHEMA_VERSION: u16 = 2;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ObjectInfoInputSchema {
    pub raw: String,
    pub input_is_list: String,
    pub lazy_inputs: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ObjectInfoOutputSchema {
    pub raw: String,
    pub output_is_list: String,
    pub output_node: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ObjectInfoNode {
    pub schema_version: u16,
    pub node_identifier: String,
    pub display_name: String,
    pub category: String,
    pub source_python_module: String,
    pub schema_source: String,
    pub source_schema: Option<CatalogNodeSchemaMetadata>,
    pub input: ObjectInfoInputSchema,
    pub output: ObjectInfoOutputSchema,
    pub availability: String,
    pub catalog_status: CatalogNodeStatus,
    pub inactive_reason: Option<String>,
    pub feature_id: String,
}

impl ObjectInfoNode {
    fn from_catalog(
        descriptor: &CatalogNodeDescriptor,
        source_python_module: String,
        source_schema: Option<CatalogNodeSchemaMetadata>,
    ) -> Self {
        Self {
            schema_version: OBJECT_INFO_SCHEMA_VERSION,
            node_identifier: descriptor.node_identifier.clone(),
            display_name: descriptor.display_name.clone(),
            category: descriptor.category.clone(),
            source_python_module,
            schema_source: descriptor.schema_source.clone(),
            source_schema,
            input: ObjectInfoInputSchema {
                raw: descriptor.inputs.clone(),
                input_is_list: descriptor.input_is_list.clone(),
                lazy_inputs: descriptor.lazy_inputs.clone(),
            },
            output: ObjectInfoOutputSchema {
                raw: descriptor.outputs.clone(),
                output_is_list: descriptor.output_is_list.clone(),
                output_node: descriptor.output_node,
            },
            availability: descriptor.availability.clone(),
            catalog_status: descriptor.catalog_status,
            inactive_reason: descriptor.inactive_reason.clone(),
            feature_id: descriptor.feature_id.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ObjectInfoRegistry {
    nodes: BTreeMap<String, ObjectInfoNode>,
}

impl ObjectInfoRegistry {
    pub fn from_node_registry(registry: &NodeRegistry) -> Result<Self, crate::NodeRegistryError> {
        let nodes = registry
            .registered()
            .iter()
            .chain(registry.inactive())
            .map(|(identifier, descriptor)| {
                let source_python_module =
                    registry.source_python_module(identifier).ok_or_else(|| {
                        crate::NodeRegistryError::MissingSourceProjection(identifier.clone())
                    })?;
                Ok((
                    identifier.clone(),
                    ObjectInfoNode::from_catalog(
                        descriptor,
                        source_python_module,
                        registry.source_schema(identifier).cloned(),
                    ),
                ))
            })
            .collect::<Result<_, crate::NodeRegistryError>>()?;
        Ok(Self { nodes })
    }

    pub fn nodes(&self) -> &BTreeMap<String, ObjectInfoNode> {
        &self.nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn object_info_is_a_read_only_schema_projection() -> Result<(), Box<dyn Error>> {
        let registry = NodeRegistry::built_in()?;
        let object_info = ObjectInfoRegistry::from_node_registry(&registry)?;
        assert_eq!(object_info.nodes().len(), 801);
        let load_image = &object_info.nodes()["LoadImage"];
        assert_eq!(load_image.display_name, "Load Image");
        assert_eq!(load_image.category, "image");
        assert_eq!(load_image.source_python_module, "nodes");
        assert!(load_image.input.raw.contains("image_upload"));
        assert!(load_image.output.raw.contains("IMAGE"));
        assert_eq!(load_image.catalog_status, CatalogNodeStatus::DescriptorOnly);
        assert!(load_image.source_schema.is_some());
        assert!(load_image.schema_source.contains("LoadImage.INPUT_TYPES"));
        assert_eq!(
            object_info.nodes()["AddNoise"].source_python_module,
            "comfy_extras.nodes_custom_sampler"
        );
        assert_eq!(
            object_info.nodes()["BeebleSwitchXImageEdit"].source_python_module,
            "comfy_api_nodes.nodes_beeble"
        );
        let inactive = &object_info.nodes()["AutogrowNamesTestNode"];
        assert_eq!(inactive.catalog_status, CatalogNodeStatus::Inactive);
        assert_eq!(inactive.source_python_module, "comfy_extras.nodes_logic");
        assert!(inactive.inactive_reason.is_some());
        assert_eq!(
            object_info.nodes()["MinimaxSubjectToVideoNode"].source_python_module,
            "comfy_api_nodes.nodes_minimax"
        );
        Ok(())
    }
}
