use crate::{
    NATIVE_NODE_CONTRACT_SCHEMA_VERSION, NativeCachePolicy, NativeDynamicInputDescriptor,
    NativeEffectClass, NativeInputDescriptor, NativeInputRequirement, NativeNodeBinding,
    NativeNodeBindingsFactory, NativeNodeContractError, NativeNodeDescriptor,
    NativeNodePresentation, NativeOutputDescriptor, NativePortCardinality, NativeValueType,
    authoritative_provider_namespace, built_in_source_schema, native_value_type_for_output_schema,
    native_value_types_for_input_schema,
};

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "Rodin3D_Gen25_Text",
    "Rodin3D_Regular",
    "Rodin3D_Sketch",
    "Rodin3D_Smooth",
    "Tencent3DPartNode",
    "Tencent3DTextureEditNode",
    "TencentImageToModelNode",
    "TencentModelTo3DUVNode",
    "TencentSmartTopologyNode",
    "TencentTextToModelNode",
];
pub const NATIVE_NODE_BINDINGS: NativeNodeBindingsFactory = native_node_bindings;

const PROVIDER_REASON: &str = "cloud provider authorization is required";

struct ProviderNodeSpec {
    feature_id: &'static str,
    class_type: &'static str,
    implementation_version: &'static str,
    display_name: &'static str,
    category: &'static str,
    description: &'static str,
    output_names: &'static [&'static str],
    output_node: bool,
}

const PROVIDER_NODES: &[ProviderNodeSpec] = &[
    ProviderNodeSpec {
        feature_id: "COMFY-NODE-0552",
        class_type: "Rodin3D_Gen25_Text",
        implementation_version: "source-bc7ea327-provider-v1",
        display_name: "Rodin 3D Gen-2.5 - Text to 3D",
        category: "partner/3d/Rodin",
        description: "Generate a 3D model from a text prompt via Rodin Gen-2.5. Pick a mode (Fast / Regular / Extreme-High) to tune quality vs. cost.",
        output_names: &["model_file"],
        output_node: false,
    },
    ProviderNodeSpec {
        feature_id: "COMFY-NODE-0553",
        class_type: "Rodin3D_Regular",
        implementation_version: "source-545376e5-provider-v1",
        display_name: "Rodin 3D Generate - Regular Generate",
        category: "partner/3d/Rodin",
        description: "",
        output_names: &["3D Model Path", "GLB"],
        output_node: false,
    },
    ProviderNodeSpec {
        feature_id: "COMFY-NODE-0554",
        class_type: "Rodin3D_Sketch",
        implementation_version: "source-ffaf11d6-provider-v1",
        display_name: "Rodin 3D Generate - Sketch Generate",
        category: "partner/3d/Rodin",
        description: "",
        output_names: &["3D Model Path", "GLB"],
        output_node: false,
    },
    ProviderNodeSpec {
        feature_id: "COMFY-NODE-0555",
        class_type: "Rodin3D_Smooth",
        implementation_version: "source-ac5f6c65-provider-v1",
        display_name: "Rodin 3D Generate - Smooth Generate",
        category: "partner/3d/Rodin",
        description: "",
        output_names: &["3D Model Path", "GLB"],
        output_node: false,
    },
    ProviderNodeSpec {
        feature_id: "COMFY-NODE-0658",
        class_type: "Tencent3DPartNode",
        implementation_version: "source-48c2fdcc-provider-v1",
        display_name: "Hunyuan3D: 3D Part",
        category: "partner/3d/Tencent",
        description: "Automatically perform component identification and generation based on the model structure.",
        output_names: &["FBX"],
        output_node: false,
    },
    ProviderNodeSpec {
        feature_id: "COMFY-NODE-0659",
        class_type: "Tencent3DTextureEditNode",
        implementation_version: "source-3644ef37-provider-v1",
        display_name: "Hunyuan3D: 3D Texture Edit",
        category: "partner/3d/Tencent",
        description: "After inputting the 3D model, perform 3D model texture redrawing.",
        output_names: &["GLB", "OBJ", "texture_image"],
        output_node: false,
    },
    ProviderNodeSpec {
        feature_id: "COMFY-NODE-0660",
        class_type: "TencentImageToModelNode",
        implementation_version: "source-05af6346-provider-v1",
        display_name: "Hunyuan3D: Image(s) to Model",
        category: "partner/3d/Tencent",
        description: "",
        output_names: &[
            "model_file",
            "GLB",
            "OBJ",
            "texture_image",
            "optional_metallic",
            "optional_normal",
            "optional_roughness",
        ],
        output_node: true,
    },
    ProviderNodeSpec {
        feature_id: "COMFY-NODE-0661",
        class_type: "TencentModelTo3DUVNode",
        implementation_version: "source-58884dd3-provider-v1",
        display_name: "Hunyuan3D: Model to UV",
        category: "partner/3d/Tencent",
        description: "Perform UV unfolding on a 3D model to generate UV texture. Input model must have less than 30000 faces.",
        output_names: &["OBJ", "FBX", "uv_image"],
        output_node: false,
    },
    ProviderNodeSpec {
        feature_id: "COMFY-NODE-0662",
        class_type: "TencentSmartTopologyNode",
        implementation_version: "source-4aef0abe-provider-v1",
        display_name: "Hunyuan3D: Smart Topology",
        category: "partner/3d/Tencent",
        description: "Perform smart retopology on a 3D model. Supports GLB/OBJ formats; max 200MB; recommended for high-poly models.",
        output_names: &["OBJ"],
        output_node: false,
    },
    ProviderNodeSpec {
        feature_id: "COMFY-NODE-0663",
        class_type: "TencentTextToModelNode",
        implementation_version: "source-cd2190f0-provider-v1",
        display_name: "Hunyuan3D: Text to Model",
        category: "partner/3d/Tencent",
        description: "",
        output_names: &["model_file", "GLB", "OBJ", "texture_image"],
        output_node: true,
    },
];

fn native_node_bindings() -> Result<Vec<NativeNodeBinding>, NativeNodeContractError> {
    PROVIDER_NODES.iter().map(provider_binding).collect()
}

fn provider_binding(spec: &ProviderNodeSpec) -> Result<NativeNodeBinding, NativeNodeContractError> {
    let provider = authoritative_provider_namespace(spec.feature_id, spec.class_type)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let catalog_schema = built_in_source_schema(spec.class_type)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let input_names = catalog_schema
        .inputs
        .iter()
        .map(|input| input.schema.name.clone())
        .collect::<Vec<_>>();
    let dynamic_schema = catalog_schema.dynamic_inputs.clone();
    let output_names = spec
        .output_names
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    let source_schema = catalog_schema
        .bind_execution_ports(&input_names, &dynamic_schema, &output_names)
        .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
    let inputs = catalog_schema
        .inputs
        .iter()
        .map(|input| {
            let accepted_types = native_value_types_for_input_schema(&input.schema)
                .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
            let allows_literal = accepted_types
                .members()
                .iter()
                .all(|member| !matches!(member, NativeValueType::Any | NativeValueType::Handle(_)));
            Ok(NativeInputDescriptor {
                name: input.schema.name.clone(),
                accepted_types,
                required: input.requirement == NativeInputRequirement::Required,
                hidden: input.requirement == NativeInputRequirement::Hidden,
                lazy: false,
                cardinality: NativePortCardinality::Scalar,
                allows_literal,
            })
        })
        .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
    let dynamic_inputs = dynamic_schema
        .iter()
        .map(|dynamic| {
            let accepted_types = native_value_types_for_input_schema(&dynamic.input)
                .map_err(|error| NativeNodeContractError::InvalidSourceSchema(error.to_string()))?;
            let allows_literal = accepted_types
                .members()
                .iter()
                .all(|member| !matches!(member, NativeValueType::Any | NativeValueType::Handle(_)));
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
                    allows_literal,
                },
            })
        })
        .collect::<Result<Vec<_>, NativeNodeContractError>>()?;
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
        feature_id: spec.feature_id.to_owned(),
        descriptor: NativeNodeDescriptor {
            schema_version: NATIVE_NODE_CONTRACT_SCHEMA_VERSION,
            class_type: spec.class_type.to_owned(),
            implementation_version: spec.implementation_version.to_owned(),
            source_schema: Some(source_schema),
            inputs,
            dynamic_inputs,
            outputs,
            output_node: spec.output_node,
            effect: NativeEffectClass::Provider,
            cache: NativeCachePolicy::Never,
        },
        presentation: NativeNodePresentation {
            display_name: spec.display_name.to_owned(),
            category: spec.category.to_owned(),
            description: spec.description.to_owned(),
            output_names,
            search_aliases: Vec::new(),
            is_deprecated: false,
            is_experimental: false,
        },
        provider: provider.to_owned(),
        reason: PROVIDER_REASON.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NativeNodeBindingDisposition, NodeRegistry, validate_generated_family_bindings};
    use serde::Deserialize;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../sim_test_support/fixtures/nodes/partner-three-d-comfy-node-0552/fixture.json"
    ));

    #[derive(Deserialize)]
    struct Fixture {
        schema_version: u32,
        stable_task_id: String,
        provider: ProviderFixture,
        nodes: Vec<NodeFixture>,
    }

    #[derive(Deserialize)]
    struct ProviderFixture {
        identifier: String,
        disposition: String,
        reason: String,
        effect: String,
        cache: String,
    }

    #[derive(Deserialize)]
    struct NodeFixture {
        feature_id: String,
        class_type: String,
        definition_sha256: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    }

    #[test]
    fn provider_bindings_match_pinned_catalog_and_fixture() -> Result<(), Box<dyn std::error::Error>>
    {
        let fixture: Fixture = serde_json::from_str(FIXTURE)?;
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(
            fixture.stable_task_id,
            "comfy-parity-native-nodes-partner-three-d-comfy-node-0552"
        );
        assert_eq!(fixture.provider.identifier, "sim-api");
        assert_eq!(fixture.provider.disposition, "provider_required");
        assert_eq!(fixture.provider.reason, PROVIDER_REASON);
        assert_eq!(fixture.provider.effect, "provider");
        assert_eq!(fixture.provider.cache, "never");

        let bindings = native_node_bindings()?;
        assert_eq!(bindings.len(), 10);
        validate_generated_family_bindings(&bindings, NODE_DESCRIPTOR_IDS)?;
        let registry = NodeRegistry::built_in()?;
        for binding in &bindings {
            registry.validate_native_binding(binding)?;
            assert_eq!(
                binding.disposition(),
                NativeNodeBindingDisposition::ProviderRequired
            );
            let expected = fixture
                .nodes
                .iter()
                .find(|node| node.class_type == binding.descriptor().class_type)
                .ok_or("fixture node is absent")?;
            assert_eq!(binding.feature_id(), expected.feature_id);
            assert_eq!(binding.descriptor().effect, NativeEffectClass::Provider);
            assert_eq!(binding.descriptor().cache, NativeCachePolicy::Never);
            assert_eq!(
                binding
                    .descriptor()
                    .inputs
                    .iter()
                    .map(|input| input.name.as_str())
                    .collect::<Vec<_>>(),
                expected
                    .inputs
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                binding
                    .descriptor()
                    .outputs
                    .iter()
                    .map(|output| output.name.as_str())
                    .collect::<Vec<_>>(),
                expected
                    .outputs
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
            );
            let source_schema = binding
                .descriptor()
                .source_schema
                .as_ref()
                .ok_or("source schema is absent")?;
            assert_eq!(
                source_schema.node.definition_sha256,
                Some(expected.definition_sha256.clone())
            );
            let encoded = serde_json::to_vec(binding.descriptor())?;
            let restored: NativeNodeDescriptor = serde_json::from_slice(&encoded)?;
            assert_eq!(&restored, binding.descriptor());
            let NativeNodeBinding::ProviderRequired {
                provider, reason, ..
            } = binding
            else {
                return Err("binding is executable instead of provider-required".into());
            };
            assert_eq!(
                provider,
                authoritative_provider_namespace(
                    binding.feature_id(),
                    &binding.descriptor().class_type,
                )?
            );
            assert_eq!(reason, PROVIDER_REASON);
        }
        assert!(fixture.nodes.iter().all(|expected| {
            bindings
                .iter()
                .any(|binding| binding.feature_id() == expected.feature_id)
        }));
        Ok(())
    }

    #[test]
    fn provider_inputs_preserve_literal_and_handle_boundaries()
    -> Result<(), Box<dyn std::error::Error>> {
        let bindings = native_node_bindings()?;
        let image_to_model = bindings
            .iter()
            .find(|binding| binding.feature_id() == "COMFY-NODE-0660")
            .ok_or("TencentImageToModelNode binding is absent")?;
        let image = image_to_model
            .descriptor()
            .inputs
            .iter()
            .find(|input| input.name == "image")
            .ok_or("image input is absent")?;
        assert!(!image.allows_literal);
        let generate_type = image_to_model
            .descriptor()
            .inputs
            .iter()
            .find(|input| input.name == "generate_type")
            .ok_or("generate_type input is absent")?;
        assert!(generate_type.allows_literal);
        assert!(generate_type.required);
        assert!(!generate_type.hidden);

        let texture_edit = bindings
            .iter()
            .find(|binding| binding.feature_id() == "COMFY-NODE-0659")
            .ok_or("Tencent3DTextureEditNode binding is absent")?;
        let model = texture_edit
            .descriptor()
            .inputs
            .iter()
            .find(|input| input.name == "model_3d")
            .ok_or("model_3d input is absent")?;
        assert!(!model.allows_literal);
        assert_eq!(model.accepted_types.members().len(), 2);

        let text = bindings
            .iter()
            .find(|binding| binding.feature_id() == "COMFY-NODE-0552")
            .ok_or("Rodin text binding is absent")?;
        assert_eq!(text.descriptor().inputs.len(), 14);
        let prompt = text
            .descriptor()
            .inputs
            .first()
            .ok_or("Rodin prompt input is absent")?;
        assert!(prompt.allows_literal);
        assert!(prompt.required);
        assert!(
            text.descriptor()
                .inputs
                .iter()
                .skip(4)
                .all(|input| !input.required)
        );
        Ok(())
    }
}
