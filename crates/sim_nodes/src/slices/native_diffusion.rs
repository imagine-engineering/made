use crate::generated_native_image::{
    NativeImageDescriptor, NativeImageDescriptorError, parse_and_validate_native_descriptors,
};
use std::sync::OnceLock;

pub const NODE_DESCRIPTOR_IDS: &[&str] = &[
    "CheckpointLoaderSimple",
    "CLIPTextEncode",
    "EmptyLatentImage",
    "KSampler",
    "TorchCompileModel",
    "ModelSamplingDiscrete",
    "ModelSamplingStableCascade",
    "ModelSamplingSD3",
    "ModelSamplingAuraFlow",
    "ModelSamplingFlux",
    "ModelSamplingContinuousEDM",
    "ModelSamplingContinuousV",
    "APG",
    "BasicGuider",
    "CFGGuider",
    "DualCFGGuider",
    "DualModelGuider",
    "SamplerCustomAdvanced",
    "KSamplerAdvanced",
];
pub const NATIVE_DIFFUSION_DESCRIPTOR_SCHEMA_VERSION: u16 = 1;

const DESCRIPTORS: &str = include_str!("native_diffusion.descriptors.json");
static PARSED_DESCRIPTORS: OnceLock<
    Result<Vec<NativeImageDescriptor>, NativeImageDescriptorError>,
> = OnceLock::new();

pub fn native_diffusion_descriptors()
-> Result<&'static [NativeImageDescriptor], NativeImageDescriptorError> {
    PARSED_DESCRIPTORS
        .get_or_init(|| {
            parse_and_validate_native_descriptors(
                DESCRIPTORS,
                NODE_DESCRIPTOR_IDS,
                NATIVE_DIFFUSION_DESCRIPTOR_SCHEMA_VERSION,
            )
        })
        .as_ref()
        .map(Vec::as_slice)
        .map_err(Clone::clone)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torch_compile_model_leaf_preserves_exact_source_schema_and_single_registration()
    -> Result<(), Box<dyn std::error::Error>> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../sim_test_support/fixtures/nodes/experimental-comfy-node-0680/fixture.json"
        ))?;
        assert_eq!(fixture["feature_id"], "COMFY-NODE-0680");
        assert_eq!(
            fixture["source"]["symbol"]["sha256"],
            "240c6eb84ce2b93fe4d7c04fcf69df571e662f17893a4fd8e721241ea082edc8"
        );
        let descriptors = native_diffusion_descriptors()?;
        let matching = descriptors
            .iter()
            .filter(|descriptor| descriptor.class_type == "TorchCompileModel")
            .collect::<Vec<_>>();
        let [descriptor] = matching.as_slice() else {
            return Err("TorchCompileModel must have exactly one native descriptor".into());
        };
        let expected: NativeImageDescriptor =
            serde_json::from_value(fixture["native_descriptor"].clone())?;
        assert_eq!(*descriptor, &expected);
        assert_eq!(descriptor.category, "experimental");
        assert!(!descriptor.output_node);
        assert!(descriptor.cache_by_input_identity);
        assert_eq!(
            descriptor.effect,
            crate::generated_native_image::NativeImageEffect::Pure
        );
        assert_eq!(descriptor.catalog_contract.input_is_list, "False");
        assert_eq!(descriptor.catalog_contract.output_is_list, "[false]");
        assert_eq!(descriptor.catalog_contract.lazy_inputs, "False");
        let schema = crate::built_in_source_schema("TorchCompileModel")?.bind_execution_ports(
            &["model".to_owned(), "backend".to_owned()],
            &[],
            &["model".to_owned()],
        )?;
        let schema = serde_json::to_value(schema)?;
        assert_eq!(schema["node"], fixture["schema"]["node"]);
        assert_eq!(
            schema["dynamic_inputs"],
            fixture["schema"]["dynamic_inputs"]
        );
        let mut inputs = fixture["schema"]["inputs"].clone();
        for input in inputs.as_array_mut().ok_or("source inputs are missing")? {
            input
                .as_object_mut()
                .ok_or("source input is not an object")?
                .remove("requirement");
        }
        assert_eq!(schema["inputs"], inputs);
        let mut outputs = fixture["schema"]["outputs"].clone();
        for output in outputs.as_array_mut().ok_or("source outputs are missing")? {
            let output = output
                .as_object_mut()
                .ok_or("source output is not an object")?;
            output.remove("ordinal");
            output.remove("source_name");
            output.insert("name".to_owned(), serde_json::json!("model"));
        }
        assert_eq!(schema["outputs"], outputs);
        let catalog = crate::NodeRegistry::built_in()?;
        let catalog = catalog
            .descriptor("TorchCompileModel")
            .ok_or("catalog descriptor missing")?;
        assert_eq!(catalog.availability, "experimental");
        assert_eq!(catalog.inputs, descriptor.catalog_contract.inputs);
        assert_eq!(catalog.outputs, descriptor.catalog_contract.outputs);
        let restored: NativeImageDescriptor =
            serde_json::from_slice(&serde_json::to_vec(descriptor)?)?;
        assert_eq!(&restored, *descriptor);
        let prompt: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec(&fixture["prompt"])?)?;
        assert_eq!(prompt, fixture["prompt"]);
        Ok(())
    }

    #[test]
    fn manifest_has_exact_new_diffusion_adapters() -> Result<(), Box<dyn std::error::Error>> {
        let raw: Vec<NativeImageDescriptor> = serde_json::from_str(DESCRIPTORS)?;
        let catalog = crate::NodeRegistry::built_in()?;
        for descriptor in &raw {
            let catalog_descriptor = catalog
                .descriptor(&descriptor.class_type)
                .ok_or_else(|| format!("{} has no catalog descriptor", descriptor.class_type))?;
            assert_eq!(
                descriptor.display_name, catalog_descriptor.display_name,
                "{} display name",
                descriptor.class_type
            );
            assert_eq!(
                descriptor.catalog_contract.inputs, catalog_descriptor.inputs,
                "{} input contract",
                descriptor.class_type
            );
            assert_eq!(
                descriptor.catalog_contract.outputs, catalog_descriptor.outputs,
                "{} output contract",
                descriptor.class_type
            );
        }
        let descriptors = native_diffusion_descriptors()?;
        assert_eq!(
            descriptors
                .iter()
                .map(|descriptor| descriptor.class_type.as_str())
                .collect::<Vec<_>>(),
            NODE_DESCRIPTOR_IDS
        );
        assert!(descriptors.iter().all(|descriptor| !descriptor.output_node));
        Ok(())
    }
}
