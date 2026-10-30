use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_nodes::{
    CatalogNodeStatus, NativeNodeBinding, NodeRegistry, PROVIDER_COMPONENT_CONTRACT_CATALOG,
    PROVIDER_COMPONENT_CONTRACT_CATALOG_SHA256, PROVIDER_NAMESPACE_COUNT,
    PROVIDER_NODE_CONTRACT_COUNT, ProviderContractError, authoritative_provider_namespace,
    authoritative_provider_namespace_projection, generated_family_node_bindings,
    project_authoritative_provider_bindings, validate_provider_component_catalog,
};
use std::collections::BTreeMap;

fn provider(binding: &NativeNodeBinding) -> Option<&str> {
    match binding {
        NativeNodeBinding::ProviderRequired { provider, .. } => Some(provider),
        NativeNodeBinding::Executable { .. } | NativeNodeBinding::Unavailable { .. } => None,
    }
}

fn with_provider(binding: NativeNodeBinding, next_provider: &str) -> NativeNodeBinding {
    match binding {
        NativeNodeBinding::ProviderRequired {
            feature_id,
            descriptor,
            presentation,
            reason,
            ..
        } => NativeNodeBinding::ProviderRequired {
            feature_id,
            descriptor,
            presentation,
            provider: next_provider.to_owned(),
            reason,
        },
        binding => binding,
    }
}

fn mutated_catalog(
    mutator: impl FnOnce(&mut Value),
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut catalog: Value = serde_json::from_slice(PROVIDER_COMPONENT_CONTRACT_CATALOG)?;
    mutator(&mut catalog);
    Ok(serde_json::to_vec(&catalog)?)
}

#[test]
fn finite_source_dynamic_options_are_consumable_by_native_schema_parser()
-> Result<(), Box<dyn std::error::Error>> {
    let registry = NodeRegistry::built_in()?;
    let mut checked = std::collections::BTreeSet::new();
    for name in registry.registered().keys() {
        let schema = registry
            .source_schema(name)
            .ok_or("missing source schema")?;
        for input in &schema.inputs {
            if input
                .schema
                .source_type_names
                .iter()
                .any(|source_type| source_type == "COMFY_DYNAMICCOMBO_V3")
                && !input.schema.choices.is_empty()
            {
                input
                    .schema
                    .structured_options()
                    .map_err(|error| format!("{name}.{}: {error}", input.schema.name))?;
                checked.insert(name.as_str());
            }
        }
    }
    for name in [
        "ClaudeNode",
        "OpenRouterLLMNode",
        "QuiverImageToSVGNode",
        "OpenAIGPTImageNodeV2",
        "LumaImageNode2",
        "ElevenLabsSpeechToSpeech",
        "MediaPipeFaceMask",
        "MediaPipeFaceMeshVisualize",
    ] {
        assert!(
            checked.contains(name),
            "source options were not checked for {name}"
        );
    }
    Ok(())
}

#[test]
fn provider_catalog_projects_exact_registered_namespace_closure()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(
        format!("{:x}", Sha256::digest(PROVIDER_COMPONENT_CONTRACT_CATALOG)),
        PROVIDER_COMPONENT_CONTRACT_CATALOG_SHA256
    );
    let projection = authoritative_provider_namespace_projection()?;
    assert_eq!(projection.len(), PROVIDER_NODE_CONTRACT_COUNT);
    assert_eq!(projection.namespace_len(), PROVIDER_NAMESPACE_COUNT);

    let expected_counts = BTreeMap::from([
        ("anthropic", 1),
        ("beeble", 2),
        ("bfl", 10),
        ("bria", 6),
        ("bytedance", 14),
        ("elevenlabs", 8),
        ("gemini", 8),
        ("grok", 7),
        ("hitpaw", 2),
        ("hunyuan3d", 6),
        ("ideogram", 2),
        ("kling", 25),
        ("krea", 2),
        ("ltxv", 2),
        ("luma", 15),
        ("magnific", 5),
        ("meshy", 7),
        ("minimax", 3),
        ("openai", 8),
        ("openrouter", 1),
        ("pixverse", 4),
        ("quiver", 2),
        ("recraft", 18),
        ("reve", 3),
        ("rodin", 7),
        ("runway", 7),
        ("sonilo", 2),
        ("topaz", 3),
        ("tripo", 12),
        ("veo2", 3),
        ("vidu", 13),
        ("wan", 14),
        ("wavespeed", 2),
    ]);
    assert_eq!(expected_counts.len(), PROVIDER_NAMESPACE_COUNT);
    for (vendor, expected_count) in expected_counts {
        let namespace = format!("zed.sim.provider.{vendor}");
        assert_eq!(
            projection
                .namespace_members(&namespace)
                .map(<[String]>::len),
            Some(expected_count),
            "{namespace}"
        );
    }

    let registry = NodeRegistry::built_in()?;
    let provider_rows = registry
        .registered()
        .values()
        .filter(|descriptor| descriptor.catalog_status == CatalogNodeStatus::ProviderRequired)
        .collect::<Vec<_>>();
    assert_eq!(provider_rows.len(), PROVIDER_NODE_CONTRACT_COUNT);
    for descriptor in provider_rows {
        let contract = projection
            .contract_for_feature_id(&descriptor.feature_id)
            .ok_or("provider feature was not projected")?;
        assert_eq!(contract.node_identifier(), descriptor.node_identifier);
        assert_eq!(
            projection.contract_for_node(&descriptor.node_identifier),
            Some(contract)
        );
        assert_eq!(
            contract.implementation_namespace(),
            format!("zed.sim.provider.{}", contract.vendor())
        );
        assert!(!contract.implementation_namespace().contains("sim-node-"));
    }
    Ok(())
}

#[test]
fn completed_provider_leaves_use_catalog_projection_atomically()
-> Result<(), Box<dyn std::error::Error>> {
    let mut bindings = generated_family_node_bindings()?;
    let provider_indices = bindings
        .iter()
        .enumerate()
        .filter_map(|(index, binding)| provider(binding).map(|_| index))
        .collect::<Vec<_>>();
    assert_eq!(provider_indices.len(), 20);
    for index in &provider_indices {
        let binding = &bindings[*index];
        assert_eq!(
            provider(binding),
            Some(authoritative_provider_namespace(
                binding.feature_id(),
                &binding.descriptor().class_type,
            )?)
        );
    }
    project_authoritative_provider_bindings(&mut bindings)?;
    for index in &provider_indices {
        let binding = &bindings[*index];
        assert_eq!(
            provider(binding),
            Some(authoritative_provider_namespace(
                binding.feature_id(),
                &binding.descriptor().class_type,
            )?)
        );
    }
    let projected_vendors = provider_indices
        .iter()
        .filter_map(|index| provider(&bindings[*index]))
        .collect::<std::collections::BTreeSet<_>>();
    assert!(projected_vendors.contains("zed.sim.provider.rodin"));
    assert!(projected_vendors.contains("zed.sim.provider.hunyuan3d"));
    assert!(projected_vendors.contains("zed.sim.provider.tripo"));

    let invalid_index = *provider_indices
        .last()
        .ok_or("provider family bindings were absent")?;
    for invalid_provider in ["sim-api", "zed.sim.provider.comfy-node-0454"] {
        let mut invalid = bindings.clone();
        invalid[invalid_index] = with_provider(invalid[invalid_index].clone(), invalid_provider);
        let before = invalid
            .iter()
            .filter_map(provider)
            .map(str::to_owned)
            .collect::<Vec<_>>();
        assert!(matches!(
            project_authoritative_provider_bindings(&mut invalid),
            Err(ProviderContractError::BindingNamespaceMismatch { .. })
        ));
        assert_eq!(
            invalid
                .iter()
                .filter_map(provider)
                .map(str::to_owned)
                .collect::<Vec<_>>(),
            before
        );
    }
    Ok(())
}

#[test]
fn malformed_provider_claims_fail_closed_with_typed_errors()
-> Result<(), Box<dyn std::error::Error>> {
    let missing = mutated_catalog(|catalog| {
        catalog["nodes"].as_array_mut().map(Vec::pop);
    })?;
    assert!(matches!(
        validate_provider_component_catalog(&missing),
        Err(ProviderContractError::SummaryMismatch)
    ));

    let duplicate = mutated_catalog(|catalog| {
        let nodes = catalog["nodes"].as_array_mut().expect("nodes array");
        nodes[1]["feature_id"] = nodes[0]["feature_id"].clone();
    })?;
    assert!(matches!(
        validate_provider_component_catalog(&duplicate),
        Err(ProviderContractError::DuplicateFeature(_))
    ));

    for namespace in [
        "zed.sim.provider.openai",
        "zed.sim.provider.comfy-node-0020",
    ] {
        let mismatched = mutated_catalog(|catalog| {
            catalog["nodes"][0]["namespace"] = json!(namespace);
        })?;
        assert!(matches!(
            validate_provider_component_catalog(&mismatched),
            Err(ProviderContractError::NamespaceMismatch { .. })
        ));
    }

    let stale = mutated_catalog(|catalog| {
        catalog["nodes"][0]["feature_id"] = json!("COMFY-NODE-0001");
    })?;
    assert!(matches!(
        validate_provider_component_catalog(&stale),
        Err(ProviderContractError::VendorClaimMismatch(_))
    ));
    assert!(matches!(
        authoritative_provider_namespace("COMFY-NODE-0020", "stale-node"),
        Err(ProviderContractError::BindingIdentityMismatch { .. })
    ));
    assert!(matches!(
        authoritative_provider_namespace("COMFY-NODE-9999", "missing-node"),
        Err(ProviderContractError::MissingBinding(_))
    ));
    Ok(())
}
