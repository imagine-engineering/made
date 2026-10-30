wit_bindgen::generate!({
    path: "../../../../sim_plugin_sdk/wit/provider-v2",
    world: "sim-provider-plugin",
    with: {
        "zed:sim-plugin/types@1.0.0": generate,
    },
});

use exports::zed::sim_provider_plugin::provider_node;
use zed::sim_plugin::types as plugin_types;
use zed::sim_provider_plugin::{invocation_input_host, provider_streaming_host, types};

struct ProviderStreamingComponent;

impl provider_node::Guest for ProviderStreamingComponent {
    fn manifest() -> types::ManifestV2 {
        types::ManifestV2 {
            schema_version: 2,
            component_world: "zed:sim-provider-plugin@2.0.0".to_owned(),
            manifest: manifest_projection(),
            provider_binding: plugin_types::ProviderBindingSet {
                schema_version: 1,
                implementation_namespace: "zed.sim.provider.openrouter".to_owned(),
                bindings_sha256: "8b941f79f9560c7ec53819291d84e42d8c68ae2a7a70722143e55e6ea411f741"
                    .to_owned(),
                bindings: vec![plugin_types::ProviderBindingClaim {
                    feature_id: "COMFY-NODE-0466".to_owned(),
                    node_id: "OpenRouterLLMNode".to_owned(),
                    contract_sha256:
                        "2c6dbff1b7f89da063de276e6737f58e1dc98a85ad736f620bea96cab9e0a3d5"
                            .to_owned(),
                    transport_schema: "zed:sim-provider-transport@1".to_owned(),
                    materializer_schema: "zed:sim-provider-materializer@1".to_owned(),
                }],
            },
            streaming: types::StreamingContract {
                methods: vec![
                    types::HttpMethod::Delete,
                    types::HttpMethod::Get,
                    types::HttpMethod::Head,
                    types::HttpMethod::Options,
                    types::HttpMethod::Patch,
                    types::HttpMethod::Post,
                    types::HttpMethod::Put,
                ],
                maximum_headers: 8,
                maximum_header_bytes: 4096,
                maximum_request_body_bytes: 16384,
                maximum_response_body_bytes: 65536,
                maximum_chunk_bytes: 4096,
                maximum_ndjson_line_bytes: 4096,
                maximum_wait_milliseconds: 1000,
                maximum_uploads: 1,
                maximum_upload_body_bytes: 4096,
                maximum_cost_requests: 1,
                maximum_progress_total: 100,
                uploads: true,
                cost_requests: true,
            },
        }
    }

    fn invoke(
        context: types::InvocationContext,
        node_id: String,
    ) -> Result<types::InvocationResult, types::StreamError> {
        if node_id != "OpenRouterLLMNode" {
            return Err(types::StreamError::InvalidRequestAuthority);
        }
        if !invocation_input_host::secret_exists("fixture-secret")
            .map_err(|_| types::StreamError::InvalidRequestAuthority)?
        {
            return Err(types::StreamError::InvalidRequestAuthority);
        }
        let handle = provider_streaming_host::start_request(
            context,
            &types::RequestHead {
                endpoint: "https://fixture.invalid/v2/stream".to_owned(),
                secret_id: Some("fixture-secret".to_owned()),
                method: types::HttpMethod::Post,
                headers: vec![
                    types::Header {
                        name: "x-first".to_owned(),
                        value: "1".to_owned(),
                    },
                    types::Header {
                        name: "x-second".to_owned(),
                        value: "2".to_owned(),
                    },
                ],
                declared_body_bytes: None,
            },
        )?;
        let input = invocation_input_host::read_scalar_input("prompt", 0)
            .map_err(|_| types::StreamError::InvalidRequestAuthority)?;
        provider_streaming_host::write_request_chunk(&types::RequestChunk {
            handle,
            sequence: 0,
            bytes: input.abi_bytes.clone(),
            end: true,
        })?;
        provider_streaming_host::check_cancelled(handle)?;
        let upload = provider_streaming_host::start_upload(&types::UploadRequest {
            handle,
            port_id: "reference".to_owned(),
            media_type: "application/octet-stream".to_owned(),
            byte_length: 3,
            content_sha256: "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81"
                .to_owned(),
        })?;
        provider_streaming_host::write_upload_chunk(&types::RequestChunk {
            handle: upload,
            sequence: 0,
            bytes: vec![1, 2, 3],
            end: true,
        })?;
        let completion =
            provider_streaming_host::complete_upload(types::UploadCompletionRequest {
                handle: upload,
            })?;
        if completion.download_url != "https://fixture.invalid/uploads/reference" {
            return Err(types::StreamError::InvalidRequestAuthority);
        }
        let cost = provider_streaming_host::request_cost(&types::CostRequest {
            handle,
            operation: "fixture".to_owned(),
            currency: "USD".to_owned(),
            maximum_microunits: 1000,
        })?;
        if !cost.accepted {
            return Err(types::StreamError::InvalidCostRequest);
        }
        provider_streaming_host::report_progress(&types::Progress {
            handle,
            sequence: 0,
            completed: 1,
            total: 1,
            message: Some("fixture".to_owned()),
        })?;
        let mut after_sequence = None;
        let receipt = loop {
            match provider_streaming_host::wait_response(types::WaitRequest {
                handle,
                after_sequence,
                timeout_milliseconds: 100,
            })? {
                types::WaitOutcome::Frame(types::ResponseFrame {
                    event: types::ResponseFrameEvent::Terminal(types::Terminal::Completed(receipt)),
                    ..
                }) => break receipt,
                types::WaitOutcome::Frame(frame) => after_sequence = Some(frame.sequence),
                types::WaitOutcome::TimedOut => {}
                types::WaitOutcome::Cancelled => return Err(types::StreamError::Cancelled),
            }
        };
        Ok(types::InvocationResult {
            outputs: vec![types::MaterializedOutput {
                port_id: "output_0".to_owned(),
                value: types::EncodedValue {
                    type_id: input.type_id,
                    family: input.family,
                    abi_bytes: input.abi_bytes,
                },
            }],
            receipt,
        })
    }
}

fn manifest_projection() -> plugin_types::ManifestProjection {
    plugin_types::ManifestProjection {
        component_world: "zed:sim-provider-plugin@1.0.0".to_owned(),
        schema_version: 1,
        identifier: "zed.sim.provider.openrouter".to_owned(),
        plugin_version: version(1, 0, 0),
        api: plugin_types::ApiRequirement {
            major: 1,
            minimum_minor: 0,
            maximum_minor: 0,
            required_features: vec![
                "provider.bindings.v1".to_owned(),
                "provider.streaming.v2".to_owned(),
            ],
        },
        nodes: vec![plugin_types::Node {
            id: "OpenRouterLLMNode".to_owned(),
            version: version(1, 0, 0),
            display_name: "OpenRouter LLM".to_owned(),
            category: "partner/text/OpenRouter".to_owned(),
            ports: vec![
                port(
                    "prompt",
                    "sim:string@1",
                    plugin_types::PortDirection::Input,
                    plugin_types::PortPresence::Required,
                ),
                port(
                    "model",
                    "sim:dynamic-combo@1",
                    plugin_types::PortDirection::Input,
                    plugin_types::PortPresence::Required,
                ),
                port(
                    "seed",
                    "sim:integer@1",
                    plugin_types::PortDirection::Input,
                    plugin_types::PortPresence::Required,
                ),
                port(
                    "system_prompt",
                    "sim:string@1",
                    plugin_types::PortDirection::Input,
                    plugin_types::PortPresence::Optional,
                ),
                port(
                    "output_0",
                    "sim:string@1",
                    plugin_types::PortDirection::Output,
                    plugin_types::PortPresence::Required,
                ),
            ],
            determinism: plugin_types::DeterminismPolicy::External,
            cache: plugin_types::CachePolicy::Never,
            effects: plugin_types::EffectPolicy::Provider,
        }],
        capabilities: vec![
            plugin_types::CapabilityRequest {
                kind: plugin_types::CapabilityKind::NetworkProvider,
                scope: "zed.sim.provider.openrouter|https://fixture.invalid/v2/stream".to_owned(),
                quota: plugin_types::CapabilityQuota {
                    maximum_operations: 1,
                    maximum_request_bytes: 16384,
                    maximum_response_bytes: 65536,
                    maximum_total_bytes: 81920,
                    maximum_handles: 1,
                    timeout_milliseconds: 1000,
                },
            },
            plugin_types::CapabilityRequest {
                kind: plugin_types::CapabilityKind::Secret,
                scope: "fixture-secret".to_owned(),
                quota: plugin_types::CapabilityQuota {
                    maximum_operations: 1,
                    maximum_request_bytes: 14,
                    maximum_response_bytes: 1,
                    maximum_total_bytes: 15,
                    maximum_handles: 1,
                    timeout_milliseconds: 1000,
                },
            },
            plugin_types::CapabilityRequest {
                kind: plugin_types::CapabilityKind::ProviderUpload,
                scope: "zed.sim.provider.openrouter|https://fixture.invalid/v2/stream".to_owned(),
                quota: plugin_types::CapabilityQuota {
                    maximum_operations: 1,
                    maximum_request_bytes: 4096,
                    maximum_response_bytes: 1,
                    maximum_total_bytes: 4096,
                    maximum_handles: 1,
                    timeout_milliseconds: 1000,
                },
            },
            plugin_types::CapabilityRequest {
                kind: plugin_types::CapabilityKind::ProviderCost,
                scope: "zed.sim.provider.openrouter|https://fixture.invalid/v2/stream".to_owned(),
                quota: plugin_types::CapabilityQuota {
                    maximum_operations: 1,
                    maximum_request_bytes: 512,
                    maximum_response_bytes: 32781,
                    maximum_total_bytes: 33293,
                    maximum_handles: 1,
                    timeout_milliseconds: 1000,
                },
            },
        ],
        ui: Vec::new(),
        routes: Vec::new(),
        legacy_mappings: Vec::new(),
    }
}

fn port(
    id: &str,
    type_id: &str,
    direction: plugin_types::PortDirection,
    presence: plugin_types::PortPresence,
) -> plugin_types::Port {
    plugin_types::Port {
        id: id.to_owned(),
        name: id.to_owned(),
        direction,
        type_id: type_id.to_owned(),
        cardinality: plugin_types::PortCardinality::Singular,
        presence,
        hidden: false,
        lazy: false,
        default: None,
        serialization: plugin_types::PortSerialization::Inline,
        accepted_legacy_names: Vec::new(),
    }
}

fn version(major: u16, minor: u16, patch: u16) -> plugin_types::ApiVersion {
    plugin_types::ApiVersion {
        major,
        minor,
        patch,
    }
}

export!(ProviderStreamingComponent);
