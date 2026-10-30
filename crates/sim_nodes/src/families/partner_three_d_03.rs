use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCachePolicy, NativeDynamicInputDescriptor,
    NativeEffectClass, NativeInputDescriptor, NativeInputRequirement, NativeNodeBinding,
    NativeNodeBindingsFactory, NativeNodeContractError, NativeNodeDescriptor,
    NativeNodePresentation, NativeOutputDescriptor, NativePortCardinality, NativeValueType,
    authoritative_provider_namespace, built_in_source_schema, native_value_type_for_output_schema,
    native_value_types_for_input_schema,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "TripoConversionNode",
    "TripoImageToModelNode",
    "TripoImportModelNode",
    "TripoMultiviewToModelNode",
    "TripoP1ImageToModelNode",
    "TripoP1MultiviewToModelNode",
    "TripoP1TextToModelNode",
    "TripoRefineNode",
    "TripoRetargetNode",
    "TripoRigNode",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const CATEGORY: &str = "partner/3d/Tripo";
const PROVIDER_REASON: &str = "cloud provider authorization is required";
const IMPLEMENTATION_VERSION: &str = "source-d380b5bb-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TripoNodeKind {
    Conversion,
    ImageToModel,
    ImportModel,
    MultiviewToModel,
    P1ImageToModel,
    P1MultiviewToModel,
    P1TextToModel,
    Refine,
    Retarget,
    Rig,
}

impl TripoNodeKind {
    const ALL: [Self; 10] = [
        Self::Conversion,
        Self::ImageToModel,
        Self::ImportModel,
        Self::MultiviewToModel,
        Self::P1ImageToModel,
        Self::P1MultiviewToModel,
        Self::P1TextToModel,
        Self::Refine,
        Self::Retarget,
        Self::Rig,
    ];

    const fn feature_id(self) -> &'static str {
        match self {
            Self::Conversion => "COMFY-NODE-0686",
            Self::ImageToModel => "COMFY-NODE-0687",
            Self::ImportModel => "COMFY-NODE-0688",
            Self::MultiviewToModel => "COMFY-NODE-0689",
            Self::P1ImageToModel => "COMFY-NODE-0690",
            Self::P1MultiviewToModel => "COMFY-NODE-0691",
            Self::P1TextToModel => "COMFY-NODE-0692",
            Self::Refine => "COMFY-NODE-0693",
            Self::Retarget => "COMFY-NODE-0694",
            Self::Rig => "COMFY-NODE-0695",
        }
    }

    const fn class_type(self) -> &'static str {
        match self {
            Self::Conversion => "TripoConversionNode",
            Self::ImageToModel => "TripoImageToModelNode",
            Self::ImportModel => "TripoImportModelNode",
            Self::MultiviewToModel => "TripoMultiviewToModelNode",
            Self::P1ImageToModel => "TripoP1ImageToModelNode",
            Self::P1MultiviewToModel => "TripoP1MultiviewToModelNode",
            Self::P1TextToModel => "TripoP1TextToModelNode",
            Self::Refine => "TripoRefineNode",
            Self::Retarget => "TripoRetargetNode",
            Self::Rig => "TripoRigNode",
        }
    }

    const fn output_names(self) -> &'static [&'static str] {
        match self {
            Self::Conversion => &[],
            Self::ImportModel => &["model_task_id"],
            Self::Retarget => &["model_file", "retarget_task_id", "glb"],
            Self::Rig => &["model_file", "rig_task_id", "glb"],
            Self::ImageToModel
            | Self::MultiviewToModel
            | Self::P1ImageToModel
            | Self::P1MultiviewToModel
            | Self::P1TextToModel
            | Self::Refine => &["model_file", "model_task_id", "glb"],
        }
    }

    const fn output_node(self) -> bool {
        matches!(
            self,
            Self::Conversion
                | Self::ImageToModel
                | Self::MultiviewToModel
                | Self::Refine
                | Self::Retarget
                | Self::Rig
        )
    }
}

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    TripoNodeKind::ALL
        .into_iter()
        .map(native_node_binding)
        .collect()
}

fn native_node_binding(kind: TripoNodeKind) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let provider = authoritative_provider_namespace(kind.feature_id(), kind.class_type())
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let catalog_schema = built_in_source_schema(kind.class_type())
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let input_names = catalog_schema
        .inputs
        .iter()
        .map(|input| input.schema.name.clone())
        .collect::<Vec<_>>();
    let dynamic_schema = catalog_schema.dynamic_inputs.clone();
    let output_names = kind
        .output_names()
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    let inputs = catalog_schema
        .inputs
        .iter()
        .map(source_input_descriptor)
        .collect::<Result<Vec<_>, _>>()?;
    let dynamic_inputs = dynamic_schema
        .iter()
        .map(source_dynamic_input_descriptor)
        .collect::<Result<Vec<_>, _>>()?;
    let presentation_metadata = catalog_schema.presentation.clone();
    let source_schema = catalog_schema
        .bind_execution_ports(&input_names, &dynamic_schema, &output_names)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let outputs = source_schema
        .outputs
        .iter()
        .map(|output| {
            Ok(NativeOutputDescriptor {
                name: output.name.clone(),
                produced_type: native_value_type_for_output_schema(output).map_err(|error| {
                    NativeNodeContractError::InvalidSourceSchema(error.to_string())
                })?,
                is_list: false,
            })
        })
        .collect::<Result<Vec<_>, NativeNodeContractError>>()?;

    Ok(NativeNodeBinding::ProviderRequired {
        feature_id: kind.feature_id().to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: kind.class_type().to_owned(),
            implementation_version: IMPLEMENTATION_VERSION.to_owned(),
            source_schema: Some(source_schema),
            inputs,
            dynamic_inputs,
            outputs,
            output_node: kind.output_node(),
            effect: NativeEffectClass::Provider,
            cache: NativeCachePolicy::Never,
        },
        presentation: NativeNodePresentation {
            display_name: presentation_metadata
                .display_name
                .unwrap_or_else(|| kind.class_type().to_owned()),
            category: CATEGORY.to_owned(),
            description: presentation_metadata.description.unwrap_or_default(),
            output_names,
            search_aliases: Vec::new(),
            is_deprecated: presentation_metadata.is_deprecated,
            is_experimental: presentation_metadata.is_experimental,
        },
        provider: provider.to_owned(),
        reason: PROVIDER_REASON.to_owned(),
    })
}

fn source_input_descriptor(
    input: &crate::CatalogNodeInputSchemaMetadata,
) -> Result<NativeInputDescriptor, NativeNodeContractError> {
    let accepted_types = native_value_types_for_input_schema(&input.schema)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let accepts_handle = accepted_types
        .members()
        .iter()
        .any(|member| matches!(member, NativeValueType::Handle(_)));
    Ok(NativeInputDescriptor {
        name: input.schema.name.clone(),
        accepted_types,
        required: input.requirement == NativeInputRequirement::Required,
        hidden: false,
        lazy: false,
        cardinality: NativePortCardinality::Scalar,
        allows_literal: !accepts_handle,
    })
}

fn source_dynamic_input_descriptor(
    dynamic: &crate::NativeDynamicSchemaMetadata,
) -> Result<NativeDynamicInputDescriptor, NativeNodeContractError> {
    let accepted_types = native_value_types_for_input_schema(&dynamic.input)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let accepts_handle = accepted_types
        .members()
        .iter()
        .any(|member| matches!(member, NativeValueType::Handle(_)));
    Ok(NativeDynamicInputDescriptor {
        name_template: dynamic.identity.clone(),
        start_index: dynamic.start_index,
        minimum_count: dynamic.minimum_count,
        maximum_count: dynamic.maximum_count,
        input: NativeInputDescriptor {
            name: dynamic.input.name.clone(),
            accepted_types,
            required: true,
            hidden: false,
            lazy: false,
            cardinality: NativePortCardinality::Scalar,
            allows_literal: !accepts_handle,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NativeNodeBindingDisposition, NodeRegistry};
    use std::error::Error;

    #[test]
    fn assigned_tripo_rows_are_exact_provider_bindings() -> Result<(), Box<dyn Error>> {
        let bindings = native_node_bindings()?;
        assert_eq!(bindings.len(), NODE_DESCRIPTOR_IDS.len());
        let registry = NodeRegistry::built_in()?;

        for ((binding, class_type), kind) in bindings
            .iter()
            .zip(NODE_DESCRIPTOR_IDS)
            .zip(TripoNodeKind::ALL)
        {
            assert_eq!(binding.feature_id(), kind.feature_id());
            assert_eq!(binding.descriptor().class_type, *class_type);
            assert_eq!(
                binding.disposition(),
                NativeNodeBindingDisposition::ProviderRequired
            );
            assert_eq!(binding.descriptor().effect, NativeEffectClass::Provider);
            assert_eq!(binding.descriptor().cache, NativeCachePolicy::Never);
            assert_eq!(binding.descriptor().output_node, kind.output_node());
            assert_eq!(binding.presentation().category, CATEGORY);
            assert_eq!(
                binding.presentation().output_names,
                kind.output_names()
                    .iter()
                    .map(|name| (*name).to_owned())
                    .collect::<Vec<_>>()
            );
            match binding {
                NativeNodeBinding::ProviderRequired {
                    provider, reason, ..
                } => {
                    assert_eq!(
                        provider,
                        authoritative_provider_namespace(
                            binding.feature_id(),
                            &binding.descriptor().class_type,
                        )?
                    );
                    assert_eq!(reason, PROVIDER_REASON);
                }
                _ => unreachable!("Tripo API row must remain provider-required"),
            }
            binding.descriptor().validate_exact_schema_v2()?;
            registry.validate_native_binding(binding)?;
        }
        Ok(())
    }

    #[test]
    fn provider_handles_and_literals_follow_source_ownership() -> Result<(), Box<dyn Error>> {
        let bindings = native_node_bindings()?;
        for binding in &bindings {
            for input in &binding.descriptor().inputs {
                let accepts_handle = input
                    .accepted_types
                    .members()
                    .iter()
                    .any(|member| matches!(member, NativeValueType::Handle(_)));
                assert_eq!(input.allows_literal, !accepts_handle);
                assert!(!input.hidden);
                assert!(!input.lazy);
                assert_eq!(input.cardinality, NativePortCardinality::Scalar);
            }
            assert!(binding.descriptor().dynamic_inputs.is_empty());
            assert!(
                binding
                    .descriptor()
                    .outputs
                    .iter()
                    .all(|output| !output.is_list)
            );
        }
        Ok(())
    }
}
