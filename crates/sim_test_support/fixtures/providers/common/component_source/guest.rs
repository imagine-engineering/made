wit_bindgen::generate!({
    path: "../../../../../sim_plugin_sdk/wit/provider-v2",
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
            provider_binding: provider_binding_projection(),
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
                maximum_uploads: if cfg!(feature = "multi-contract") {
                    3
                } else {
                    1
                },
                maximum_upload_body_bytes: if cfg!(feature = "multi-contract") {
                    8192
                } else {
                    4096
                },
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
        if node_id == "TypedBatchNode" {
            return invoke_typed_batch(context);
        }
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
        let upload_completion =
            provider_streaming_host::complete_upload(types::UploadCompletionRequest {
                handle: upload,
            })?;
        if !upload_completion.download_url.starts_with("https://") {
            return Err(types::StreamError::InvalidUpload);
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
        #[cfg(feature = "malformed-result")]
        let output_bytes = b"not-a-plugin-value".to_vec();
        #[cfg(feature = "oversized-result")]
        let output_bytes = vec![0; 16 * 1024 * 1024 + 1];
        #[cfg(not(any(feature = "malformed-result", feature = "oversized-result")))]
        let output_bytes = input.abi_bytes;
        Ok(types::InvocationResult {
            outputs: vec![types::MaterializedOutput {
                port_id: "output_0".to_owned(),
                value: types::EncodedValue {
                    type_id: input.type_id,
                    family: input.family,
                    abi_bytes: output_bytes,
                },
            }],
            receipt,
        })
    }
}

fn provider_binding_projection() -> plugin_types::ProviderBindingSet {
    let bindings = vec![plugin_types::ProviderBindingClaim {
        feature_id: "COMFY-NODE-0466".to_owned(),
        node_id: "OpenRouterLLMNode".to_owned(),
        contract_sha256: "2c6dbff1b7f89da063de276e6737f58e1dc98a85ad736f620bea96cab9e0a3d5"
            .to_owned(),
        transport_schema: "zed:sim-provider-transport@1".to_owned(),
        materializer_schema: "zed:sim-provider-materializer@1".to_owned(),
    }];
    #[cfg(feature = "multi-contract")]
    let bindings = {
        let mut bindings = bindings;
        bindings.push(plugin_types::ProviderBindingClaim {
            feature_id: "COMFY-NODE-TASK434-MULTI".to_owned(),
            node_id: "TypedBatchNode".to_owned(),
            contract_sha256: "2222222222222222222222222222222222222222222222222222222222222222"
                .to_owned(),
            transport_schema: "zed:sim-provider-transport@1".to_owned(),
            materializer_schema: "zed:sim-provider-materializer@1".to_owned(),
        });
        bindings
    };
    plugin_types::ProviderBindingSet {
        schema_version: 1,
        implementation_namespace: "zed.sim.provider.openrouter".to_owned(),
        bindings_sha256: if cfg!(feature = "multi-contract") {
            "a46ffa690dd864126f123de514a577957711d43259d653eacf9a4673251f0076".to_owned()
        } else {
            "8b941f79f9560c7ec53819291d84e42d8c68ae2a7a70722143e55e6ea411f741".to_owned()
        },
        bindings,
    }
}

fn invoke_typed_batch(
    context: types::InvocationContext,
) -> Result<types::InvocationResult, types::StreamError> {
    #[cfg(not(feature = "multi-contract"))]
    {
        let _ = context;
        return Err(types::StreamError::InvalidRequestAuthority);
    }
    #[cfg(feature = "multi-contract")]
    {
        let first = invocation_input_host::read_scalar_input("items", 0)
            .map_err(|_| types::StreamError::InvalidRequestAuthority)?;
        let image_state = invocation_input_host::get_input_state("reference_image")
            .map_err(|_| types::StreamError::InvalidRequestAuthority)?;
        let mut image = None;
        let mut media = Vec::new();
        for value_index in 0..image_state.length {
            let image_handle = invocation_input_host::take_input("reference_image", value_index)
                .map_err(|_| types::StreamError::InvalidRequestAuthority)?;
            let value = invocation_input_host::read_handle(image_handle)
                .map_err(|_| types::StreamError::InvalidRequestAuthority)?;
            if image.is_none() {
                image = Some(value);
            }
            let media_count = invocation_input_host::media_input_count(image_handle)
                .map_err(|_| types::StreamError::InvalidUpload)?;
            for batch_index in 0..media_count {
                media.push(
                    invocation_input_host::read_media_input_at(image_handle, batch_index)
                        .map_err(|_| types::StreamError::InvalidUpload)?,
                );
            }
        }
        let image = image.ok_or(types::StreamError::InvalidUpload)?;
        let mut handle = provider_streaming_host::start_request(
            context,
            &types::RequestHead {
                endpoint: "https://fixture.invalid/v2/stream".to_owned(),
                secret_id: None,
                method: types::HttpMethod::Post,
                headers: vec![types::Header {
                    name: "x-node".to_owned(),
                    value: "typed".to_owned(),
                }],
                declared_body_bytes: None,
            },
        )?;
        provider_streaming_host::write_request_chunk(&types::RequestChunk {
            handle,
            sequence: 0,
            bytes: first.abi_bytes.clone(),
            end: true,
        })?;
        provider_streaming_host::check_cancelled(handle)?;
        for media in media {
            let upload = provider_streaming_host::start_upload(&types::UploadRequest {
                handle,
                port_id: "reference_image".to_owned(),
                media_type: media.media_type,
                byte_length: u64::try_from(media.bytes.len())
                    .map_err(|_| types::StreamError::BodyLimit)?,
                content_sha256: media.content_sha256,
            })?;
            let mut chunks = media.bytes.chunks(16).peekable();
            let mut sequence = 0_u64;
            while let Some(chunk) = chunks.next() {
                provider_streaming_host::write_upload_chunk(&types::RequestChunk {
                    handle: upload,
                    sequence,
                    bytes: chunk.to_vec(),
                    end: chunks.peek().is_none(),
                })?;
                sequence = sequence
                    .checked_add(1)
                    .ok_or(types::StreamError::InvalidSequence)?;
            }
            let upload_completion =
                provider_streaming_host::complete_upload(types::UploadCompletionRequest {
                    handle: upload,
                })?;
            if !upload_completion.download_url.starts_with("https://") {
                return Err(types::StreamError::InvalidUpload);
            }
        }
        let cost = provider_streaming_host::request_cost(&types::CostRequest {
            handle,
            operation: "typed-batch".to_owned(),
            currency: "USD".to_owned(),
            maximum_microunits: 2000,
        })?;
        if !cost.accepted {
            return Err(types::StreamError::InvalidCostRequest);
        }
        provider_streaming_host::report_progress(&types::Progress {
            handle,
            sequence: 0,
            completed: 2,
            total: 2,
            message: Some("typed".to_owned()),
        })?;
        let mut after_sequence = None;
        let mut retried = false;
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
                types::WaitOutcome::Frame(types::ResponseFrame {
                    event: types::ResponseFrameEvent::Head(head),
                    ..
                }) if !retried && matches!(head.status, 408 | 429 | 500 | 502 | 503 | 504) => {
                    handle = provider_streaming_host::retry_request(types::RetryRequestOptions {
                        handle,
                        delay_milliseconds: 1,
                    })?;
                    provider_streaming_host::write_request_chunk(&types::RequestChunk {
                        handle,
                        sequence: 0,
                        bytes: first.abi_bytes.clone(),
                        end: true,
                    })?;
                    provider_streaming_host::report_progress(&types::Progress {
                        handle,
                        sequence: 1,
                        completed: 2,
                        total: 2,
                        message: Some("typed retry".to_owned()),
                    })?;
                    after_sequence = None;
                    retried = true;
                }
                types::WaitOutcome::Frame(frame) => after_sequence = Some(frame.sequence),
                types::WaitOutcome::TimedOut => {}
                types::WaitOutcome::Cancelled => return Err(types::StreamError::Cancelled),
            }
        };
        #[cfg(feature = "malformed-result")]
        let first_bytes = b"not-a-plugin-value".to_vec();
        #[cfg(feature = "oversized-result")]
        let first_bytes = vec![0; 16 * 1024 * 1024 + 1];
        #[cfg(not(any(feature = "malformed-result", feature = "oversized-result")))]
        let first_bytes = first.abi_bytes;
        Ok(types::InvocationResult {
            outputs: vec![
                types::MaterializedOutput {
                    port_id: "output_0".to_owned(),
                    value: types::EncodedValue {
                        type_id: first.type_id,
                        family: first.family,
                        abi_bytes: first_bytes,
                    },
                },
                types::MaterializedOutput {
                    port_id: "output_1".to_owned(),
                    value: types::EncodedValue {
                        type_id: image.type_id,
                        family: image.family,
                        abi_bytes: image.abi_bytes,
                    },
                },
            ],
            receipt,
        })
    }
}

fn manifest_projection() -> plugin_types::ManifestProjection {
    let nodes = vec![plugin_types::Node {
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
    }];
    #[cfg(feature = "multi-contract")]
    let nodes = {
        let mut nodes = nodes;
        nodes.push(plugin_types::Node {
            id: "TypedBatchNode".to_owned(),
            version: version(1, 0, 0),
            display_name: "Typed Batch".to_owned(),
            category: "fixture/typed".to_owned(),
            ports: vec![
                port_full(
                    "items",
                    "sim:integer@1",
                    plugin_types::PortDirection::Input,
                    plugin_types::PortPresence::Required,
                    plugin_types::PortCardinality::List,
                    plugin_types::PortSerialization::Inline,
                ),
                port_full(
                    "reference_image",
                    "sim:image@1",
                    plugin_types::PortDirection::Input,
                    plugin_types::PortPresence::Required,
                    plugin_types::PortCardinality::List,
                    plugin_types::PortSerialization::Handle,
                ),
                port(
                    "output_0",
                    "sim:integer@1",
                    plugin_types::PortDirection::Output,
                    plugin_types::PortPresence::Required,
                ),
                port_full(
                    "output_1",
                    "sim:image@1",
                    plugin_types::PortDirection::Output,
                    plugin_types::PortPresence::Required,
                    plugin_types::PortCardinality::Singular,
                    plugin_types::PortSerialization::Handle,
                ),
            ],
            determinism: plugin_types::DeterminismPolicy::External,
            cache: plugin_types::CachePolicy::Never,
            effects: plugin_types::EffectPolicy::Provider,
        });
        nodes
    };
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
        nodes,
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
                    maximum_request_bytes: 64,
                    maximum_response_bytes: 1,
                    maximum_total_bytes: 65,
                    maximum_handles: 1,
                    timeout_milliseconds: 1000,
                },
            },
            plugin_types::CapabilityRequest {
                kind: plugin_types::CapabilityKind::ProviderUpload,
                scope: "zed.sim.provider.openrouter|https://fixture.invalid/v2/stream".to_owned(),
                quota: plugin_types::CapabilityQuota {
                    maximum_operations: if cfg!(feature = "multi-contract") {
                        3
                    } else {
                        1
                    },
                    maximum_request_bytes: if cfg!(feature = "multi-contract") {
                        8192
                    } else {
                        4096
                    },
                    maximum_response_bytes: if cfg!(feature = "multi-contract") {
                        3
                    } else {
                        1
                    },
                    maximum_total_bytes: if cfg!(feature = "multi-contract") {
                        8195
                    } else {
                        4096
                    },
                    maximum_handles: if cfg!(feature = "multi-contract") {
                        3
                    } else {
                        1
                    },
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
        routes: if cfg!(feature = "multi-contract") {
            vec![plugin_types::RouteDeclaration {
                id: "task434.typed".to_owned(),
                method: "POST".to_owned(),
                path: "/task434/typed".to_owned(),
                maximum_request_bytes: 1024,
                maximum_response_bytes: 2048,
            }]
        } else {
            Vec::new()
        },
        legacy_mappings: Vec::new(),
    }
}

fn port(
    id: &str,
    type_id: &str,
    direction: plugin_types::PortDirection,
    presence: plugin_types::PortPresence,
) -> plugin_types::Port {
    port_full(
        id,
        type_id,
        direction,
        presence,
        plugin_types::PortCardinality::Singular,
        plugin_types::PortSerialization::Inline,
    )
}

fn port_full(
    id: &str,
    type_id: &str,
    direction: plugin_types::PortDirection,
    presence: plugin_types::PortPresence,
    cardinality: plugin_types::PortCardinality,
    serialization: plugin_types::PortSerialization,
) -> plugin_types::Port {
    plugin_types::Port {
        id: id.to_owned(),
        name: id.to_owned(),
        direction,
        type_id: type_id.to_owned(),
        cardinality,
        presence,
        hidden: false,
        lazy: cardinality == plugin_types::PortCardinality::List
            && direction == plugin_types::PortDirection::Input,
        default: None,
        serialization,
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
