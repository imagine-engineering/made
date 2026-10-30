wit_bindgen::generate!({
    path: "../../../../sim_plugin_sdk/wit",
    world: "sim-provider-plugin",
});

use exports::zed::sim_plugin::{plugin, provider_binding};
use zed::sim_plugin::{host, types};

struct ProviderComponent;

impl plugin::Guest for ProviderComponent {
    fn manifest() -> types::ManifestProjection {
        types::ManifestProjection {
            component_world: "zed:sim-provider-plugin@1.0.0".to_owned(),
            schema_version: 1,
            identifier: "zed.sim.provider.openrouter".to_owned(),
            plugin_version: version(1, 0, 0),
            api: types::ApiRequirement {
                major: 1,
                minimum_minor: 0,
                maximum_minor: 0,
                required_features: vec!["provider.bindings.v1".to_owned()],
            },
            nodes: vec![types::Node {
                id: "OpenRouterLLMNode".to_owned(),
                version: version(1, 0, 0),
                display_name: "OpenRouter LLM".to_owned(),
                category: "partner/text/OpenRouter".to_owned(),
                ports: vec![
                    types::Port {
                        id: "prompt".to_owned(),
                        name: "prompt".to_owned(),
                        direction: types::PortDirection::Input,
                        type_id: "sim:string@1".to_owned(),
                        cardinality: types::PortCardinality::Singular,
                        presence: types::PortPresence::Required,
                        hidden: false,
                        lazy: false,
                        default: None,
                        serialization: types::PortSerialization::Inline,
                        accepted_legacy_names: Vec::new(),
                    },
                    types::Port {
                        id: "model".to_owned(),
                        name: "model".to_owned(),
                        direction: types::PortDirection::Input,
                        type_id: "sim:dynamic-combo@1".to_owned(),
                        cardinality: types::PortCardinality::Singular,
                        presence: types::PortPresence::Required,
                        hidden: false,
                        lazy: false,
                        default: None,
                        serialization: types::PortSerialization::Inline,
                        accepted_legacy_names: Vec::new(),
                    },
                    types::Port {
                        id: "seed".to_owned(),
                        name: "seed".to_owned(),
                        direction: types::PortDirection::Input,
                        type_id: "sim:integer@1".to_owned(),
                        cardinality: types::PortCardinality::Singular,
                        presence: types::PortPresence::Required,
                        hidden: false,
                        lazy: false,
                        default: None,
                        serialization: types::PortSerialization::Inline,
                        accepted_legacy_names: Vec::new(),
                    },
                    types::Port {
                        id: "system_prompt".to_owned(),
                        name: "system_prompt".to_owned(),
                        direction: types::PortDirection::Input,
                        type_id: "sim:string@1".to_owned(),
                        cardinality: types::PortCardinality::Singular,
                        presence: types::PortPresence::Optional,
                        hidden: false,
                        lazy: false,
                        default: None,
                        serialization: types::PortSerialization::Inline,
                        accepted_legacy_names: Vec::new(),
                    },
                    types::Port {
                        id: "output_0".to_owned(),
                        name: "output_0".to_owned(),
                        direction: types::PortDirection::Output,
                        type_id: "sim:string@1".to_owned(),
                        cardinality: types::PortCardinality::Singular,
                        presence: types::PortPresence::Required,
                        hidden: false,
                        lazy: false,
                        default: None,
                        serialization: types::PortSerialization::Inline,
                        accepted_legacy_names: Vec::new(),
                    },
                ],
                determinism: types::DeterminismPolicy::External,
                cache: types::CachePolicy::Never,
                effects: types::EffectPolicy::Provider,
            }],
            capabilities: [
                types::CapabilityKind::NetworkProvider,
                types::CapabilityKind::ProviderUpload,
                types::CapabilityKind::ProviderCost,
            ]
            .into_iter()
            .map(|kind| types::CapabilityRequest {
                kind,
                scope: "fixture|https://fixture.invalid/v1/generate".to_owned(),
                quota: types::CapabilityQuota {
                    maximum_operations: 16,
                    maximum_request_bytes: 16 * 1024 * 1024,
                    maximum_response_bytes: 64 * 1024 * 1024,
                    maximum_total_bytes: 80 * 1024 * 1024,
                    maximum_handles: 8,
                    timeout_milliseconds: 5_000,
                },
            })
            .collect(),
            ui: Vec::new(),
            routes: Vec::new(),
            legacy_mappings: Vec::new(),
        }
    }

    fn create_node(node_id: String) -> Result<u64, types::InvocationError> {
        if node_id == "OpenRouterLLMNode" {
            Ok(1)
        } else {
            Err(types::InvocationError::PluginFailure(
                "unknown provider fixture node".to_owned(),
            ))
        }
    }

    fn invoke(_instance: u64) -> Result<(), types::InvocationError> {
        Err(types::InvocationError::PluginFailure(
            "provider fixture requires invoke-provider".to_owned(),
        ))
    }

    fn cancel(_instance: u64, _reason: types::CancelReason) -> Result<(), types::InvocationError> {
        Ok(())
    }

    fn drop_node(_instance: u64) {}
}

impl provider_binding::Guest for ProviderComponent {
    fn binding_set() -> types::ProviderBindingSet {
        types::ProviderBindingSet {
            schema_version: 1,
            implementation_namespace: "zed.sim.provider.openrouter".to_owned(),
            bindings_sha256: "8b941f79f9560c7ec53819291d84e42d8c68ae2a7a70722143e55e6ea411f741"
                .to_owned(),
            bindings: vec![types::ProviderBindingClaim {
                feature_id: "COMFY-NODE-0466".to_owned(),
                node_id: "OpenRouterLLMNode".to_owned(),
                contract_sha256: "2c6dbff1b7f89da063de276e6737f58e1dc98a85ad736f620bea96cab9e0a3d5"
                    .to_owned(),
                transport_schema: "zed:sim-provider-transport@1".to_owned(),
                materializer_schema: "zed:sim-provider-materializer@1".to_owned(),
            }],
        }
    }

    fn invoke_provider(
        class_type: String,
        request: Vec<u8>,
    ) -> Result<types::ProviderInvocationResponse, types::InvocationError> {
        if class_type != "OpenRouterLLMNode" || request.is_empty() {
            return Err(types::InvocationError::PluginFailure(
                "invalid provider request".to_owned(),
            ));
        }
        let receipt = if request.starts_with(b"zed.sim.provider-transport-request\0") {
            let receipt = host::provider_request(
                "fixture",
                "https://fixture.invalid/v1/generate",
                &request,
                None,
            )?;
            receipt_set(&[receipt.as_slice()])
        } else if request == b"invalid-provider-receipt" {
            b"guest-controlled-receipt".to_vec()
        } else {
            receipt_set(&[b"provider-fixture-receipt"])
        };
        let outputs = if request == b"guest-authored-output" {
            vec![types::ProviderMaterializedOutput {
                port_id: "output_0".to_owned(),
                value: types::EncodedValue {
                    type_id: "sim:string@1".to_owned(),
                    family: types::ValueFamily::Scalar,
                    abi_bytes: request,
                },
            }]
        } else {
            Vec::new()
        };
        Ok(types::ProviderInvocationResponse { outputs, receipt })
    }
}

fn receipt_set(receipts: &[&[u8]]) -> Vec<u8> {
    let mut encoded = b"zed.sim.provider-result-receipt-set\0".to_vec();
    encoded.extend_from_slice(&1_u16.to_le_bytes());
    encoded.extend_from_slice(&(receipts.len() as u32).to_le_bytes());
    for receipt in receipts {
        encoded.extend_from_slice(&(receipt.len() as u32).to_le_bytes());
        encoded.extend_from_slice(receipt);
    }
    encoded
}

fn version(major: u16, minor: u16, patch: u16) -> types::ApiVersion {
    types::ApiVersion {
        major,
        minor,
        patch,
    }
}

export!(ProviderComponent);
