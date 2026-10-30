wit_bindgen::generate!({
    path: "../../../../../sim_plugin_sdk/wit/provider-v2",
    world: "sim-provider-plugin",
    with: { "zed:sim-plugin/types@1.0.0": generate },
});

use exports::zed::sim_provider_plugin::provider_node;
use zed::sim_plugin::types as plugin;
use zed::sim_provider_plugin::{invocation_input_host, provider_streaming_host, types};

#[cfg(target_arch = "wasm32")]
#[allow(
    dead_code,
    reason = "The shared source contract exposes constants used by native conformance"
)]
#[path = "../../../../../sim_plugin_host/provider_components/src/vendors/openrouter.rs"]
mod openrouter;

struct OpenRouterComponent;
// The pinned Comfy client allows three ordinary and sixteen rate-limit retries.
// Every transmitted attempt may be billed, including one that is later retried.
const MAXIMUM_BILLABLE_ATTEMPTS: u64 = 20;

impl provider_node::Guest for OpenRouterComponent {
    fn manifest() -> types::ManifestV2 {
        types::ManifestV2 {
            schema_version: 2,
            component_world: "zed:sim-provider-plugin@2.0.0".into(),
            manifest: manifest(),
            provider_binding: plugin::ProviderBindingSet {
                schema_version: 1,
                implementation_namespace: "zed.sim.provider.openrouter".into(),
                bindings_sha256: "8b941f79f9560c7ec53819291d84e42d8c68ae2a7a70722143e55e6ea411f741"
                    .into(),
                bindings: vec![plugin::ProviderBindingClaim {
                    feature_id: "COMFY-NODE-0466".into(),
                    node_id: "OpenRouterLLMNode".into(),
                    contract_sha256:
                        "2c6dbff1b7f89da063de276e6737f58e1dc98a85ad736f620bea96cab9e0a3d5".into(),
                    transport_schema: "zed:sim-provider-transport@1".into(),
                    materializer_schema: "zed:sim-provider-materializer@1".into(),
                }],
            },
            streaming: types::StreamingContract {
                methods: vec![types::HttpMethod::Post],
                maximum_headers: 8,
                maximum_header_bytes: 16_384,
                maximum_request_body_bytes: 1_048_576,
                maximum_response_body_bytes: 16_777_216,
                maximum_chunk_bytes: 65_536,
                maximum_ndjson_line_bytes: 65_536,
                maximum_wait_milliseconds: 1_000,
                maximum_uploads: 0,
                maximum_upload_body_bytes: 0,
                maximum_cost_requests: 1,
                maximum_progress_total: 100,
                uploads: false,
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
        let prompt = scalar_text(&input("prompt")?.abi_bytes)?;
        if prompt.len() > 65_536 {
            return Err(types::StreamError::BodyLimit);
        }
        let (model, reasoning_effort) = model_selection(&input("model")?.abi_bytes)?;
        let seed = scalar_integer(&input("seed")?.abi_bytes)?;
        let system_prompt = optional_text("system_prompt")?.unwrap_or_default();
        let input = openrouter::ChatInput {
            model: &model,
            prompt: &prompt,
            system_prompt: &system_prompt,
            seed,
            reasoning_effort: reasoning_effort.as_deref(),
            search_context_size: None,
        };
        let body = openrouter::bounded_request_body(
            &input,
            &[],
            &[],
            openrouter::MAXIMUM_COMPLETION_TOKENS,
        )
        .map_err(|_| types::StreamError::InvalidRequestAuthority)?;
        let per_attempt_estimate = openrouter::estimated_cost_microunits(
            &input,
            &body,
            0,
            0,
            openrouter::MAXIMUM_COMPLETION_TOKENS,
        )
        .map_err(|_| types::StreamError::InvalidCostRequest)?;
        let estimate = per_attempt_estimate
            .checked_mul(MAXIMUM_BILLABLE_ATTEMPTS)
            .ok_or(types::StreamError::InvalidCostRequest)?;
        let (secret_id, authorization_header) =
            if invocation_input_host::secret_exists("sim_org.auth_token")
                .map_err(|_| types::StreamError::InvalidRequestAuthority)?
            {
                (
                    "sim_org.auth_token",
                    header("Authorization", "host-capability:auth_token_sim_org"),
                )
            } else if invocation_input_host::secret_exists("sim_org.api_key")
                .map_err(|_| types::StreamError::InvalidRequestAuthority)?
            {
                (
                    "sim_org.api_key",
                    header("X-API-KEY", "host-capability:api_key_sim_org"),
                )
            } else {
                return Err(types::StreamError::InvalidRequestAuthority);
            };
        let mut handle = provider_streaming_host::start_request(
            context,
            &types::RequestHead {
                endpoint: "https://api.comfy.org/proxy/openrouter/api/v1/chat/completions".into(),
                secret_id: Some(secret_id.into()),
                method: types::HttpMethod::Post,
                headers: vec![
                    header("Accept", "application/json"),
                    authorization_header,
                    header("Comfy-Env", "host"),
                    header("Comfy-Usage-Source", "comfyui-api"),
                    header("Content-Type", "application/json"),
                ],
                declared_body_bytes: None,
            },
        )?;
        write_chunks(handle, &body)?;
        let approval = provider_streaming_host::request_cost(&types::CostRequest {
            handle,
            operation: "openrouter.chat.completions.estimated".into(),
            currency: "USD".into(),
            maximum_microunits: estimate,
        })?;
        if !approval.accepted
            || approval.approved_microunits != estimate
            || approval.receipt.is_empty()
        {
            return Err(types::StreamError::InvalidCostRequest);
        }
        provider_streaming_host::report_progress(&types::Progress {
            handle,
            sequence: 0,
            completed: 0,
            total: 100,
            message: Some("Waiting for server".into()),
        })?;
        let mut after_sequence = None;
        let mut response = Vec::new();
        let mut retry_count = 0_u64;
        let mut ordinary_retries = 0_u64;
        let mut rate_limit_retries = 0_u64;
        let mut status = 200_u16;
        let receipt = loop {
            provider_streaming_host::check_cancelled(handle)?;
            match provider_streaming_host::wait_response(types::WaitRequest {
                handle,
                after_sequence,
                timeout_milliseconds: 1_000,
            })? {
                types::WaitOutcome::Frame(types::ResponseFrame {
                    event: types::ResponseFrameEvent::Terminal(types::Terminal::Completed(receipt)),
                    ..
                }) => break receipt,
                types::WaitOutcome::Frame(types::ResponseFrame {
                    sequence,
                    event: types::ResponseFrameEvent::Head(head),
                    ..
                }) if head.status != 200 => {
                    let retry_delay = if head.status == 429 && rate_limit_retries < 16 {
                        rate_limit_retries += 1;
                        Some((1_000_u64 << (rate_limit_retries - 1)).min(30_000))
                    } else if matches!(head.status, 408 | 500 | 502 | 503 | 504)
                        && ordinary_retries < 3
                    {
                        ordinary_retries += 1;
                        Some(1_000_u64 << (ordinary_retries - 1))
                    } else {
                        None
                    };
                    if let Some(delay_milliseconds) = retry_delay {
                        retry_count += 1;
                        provider_streaming_host::report_progress(&types::Progress {
                            handle,
                            sequence: retry_count,
                            completed: 0,
                            total: 100,
                            message: Some("Retrying request".into()),
                        })?;
                        handle =
                            provider_streaming_host::retry_request(types::RetryRequestOptions {
                                handle,
                                // The host honors Retry-After (seconds or HTTP date) and
                                // clamps the server hint to the pinned 150-second limit.
                                delay_milliseconds,
                            })?;
                        write_chunks(handle, &body)?;
                        after_sequence = None;
                        response.clear();
                        status = 200;
                    } else {
                        status = head.status;
                        after_sequence = Some(sequence);
                    }
                }
                types::WaitOutcome::Frame(types::ResponseFrame {
                    sequence,
                    event: types::ResponseFrameEvent::Chunk(chunk),
                    ..
                }) => {
                    let bytes = match chunk {
                        types::ResponseChunkValue::Binary(bytes) => bytes,
                        types::ResponseChunkValue::Text(text)
                        | types::ResponseChunkValue::NdjsonLine(text) => text.into_bytes(),
                    };
                    if response.len().saturating_add(bytes.len())
                        > openrouter::MAXIMUM_RESPONSE_BYTES
                    {
                        return Err(types::StreamError::BodyLimit);
                    }
                    response.extend_from_slice(&bytes);
                    after_sequence = Some(sequence);
                }
                types::WaitOutcome::Frame(frame) => after_sequence = Some(frame.sequence),
                types::WaitOutcome::TimedOut => {}
                types::WaitOutcome::Cancelled => return Err(types::StreamError::Cancelled),
            }
        };
        if status != 200 {
            return Err(types::StreamError::HostFailure);
        }
        let parsed = openrouter::parse_response(&response)
            .map_err(|_| types::StreamError::InvalidTerminal)?;
        // Actual billed cost may exceed the approved estimate. The receipt records
        // the server report; the external key/account budget is the backstop.
        provider_streaming_host::report_progress(&types::Progress {
            handle,
            sequence: retry_count + 1,
            completed: 100,
            total: 100,
            message: Some("Completed".into()),
        })?;
        Ok(types::InvocationResult {
            outputs: vec![types::MaterializedOutput {
                port_id: "output_0".into(),
                value: types::EncodedValue {
                    type_id: "sim:string@1".into(),
                    family: plugin::ValueFamily::Scalar,
                    abi_bytes: result_output(&parsed.text)?,
                },
            }],
            receipt,
        })
    }
}

fn input(port: &str) -> Result<plugin::EncodedValue, types::StreamError> {
    invocation_input_host::read_scalar_input(port, 0)
        .map_err(|_| types::StreamError::InvalidRequestAuthority)
}

fn optional_text(port: &str) -> Result<Option<String>, types::StreamError> {
    let state = invocation_input_host::get_input_state(port)
        .map_err(|_| types::StreamError::InvalidRequestAuthority)?;
    if state.present {
        Ok(Some(scalar_text(&input(port)?.abi_bytes)?))
    } else {
        Ok(None)
    }
}

fn scalar_node(bytes: &[u8]) -> Result<serde_json::Value, types::StreamError> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|_| types::StreamError::InvalidRequestAuthority)?;
    value
        .pointer("/representation/value")
        .cloned()
        .ok_or(types::StreamError::InvalidRequestAuthority)
}

fn scalar_text(bytes: &[u8]) -> Result<String, types::StreamError> {
    let node = scalar_node(bytes)?;
    if node.get("kind").and_then(serde_json::Value::as_str) != Some("string") {
        return Err(types::StreamError::InvalidRequestAuthority);
    }
    node.get("value")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or(types::StreamError::InvalidRequestAuthority)
}

fn model_selection(bytes: &[u8]) -> Result<(String, Option<String>), types::StreamError> {
    let node = scalar_node(bytes)?;
    if node.get("kind").and_then(serde_json::Value::as_str) == Some("string") {
        return Ok((
            node.get("value")
                .and_then(serde_json::Value::as_str)
                .ok_or(types::StreamError::InvalidRequestAuthority)?
                .to_owned(),
            None,
        ));
    }
    if node.get("kind").and_then(serde_json::Value::as_str) != Some("record") {
        return Err(types::StreamError::InvalidRequestAuthority);
    }
    let fields = node
        .get("value")
        .and_then(serde_json::Value::as_array)
        .ok_or(types::StreamError::InvalidRequestAuthority)?;
    let mut model = None;
    let mut reasoning_effort = None;
    for field in fields {
        let pair = field
            .as_array()
            .filter(|pair| pair.len() == 2)
            .ok_or(types::StreamError::InvalidRequestAuthority)?;
        let name = pair[0]
            .as_str()
            .ok_or(types::StreamError::InvalidRequestAuthority)?;
        match name {
            "model" if model.is_none() => {
                model = Some(scalar_field_text(&pair[1])?);
            }
            "reasoning_effort" if reasoning_effort.is_none() => {
                reasoning_effort = Some(scalar_field_text(&pair[1])?);
            }
            "search_context_size" => return Err(types::StreamError::InvalidCostRequest),
            "images" | "videos" if media_field_is_empty(&pair[1]) => {}
            "images" | "videos" => return Err(types::StreamError::InvalidCostRequest),
            _ => return Err(types::StreamError::InvalidRequestAuthority),
        }
    }
    Ok((
        model.ok_or(types::StreamError::InvalidRequestAuthority)?,
        reasoning_effort,
    ))
}

fn scalar_field_text(node: &serde_json::Value) -> Result<String, types::StreamError> {
    if node.get("kind").and_then(serde_json::Value::as_str) != Some("string") {
        return Err(types::StreamError::InvalidRequestAuthority);
    }
    node.get("value")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or(types::StreamError::InvalidRequestAuthority)
}

fn media_field_is_empty(node: &serde_json::Value) -> bool {
    matches!(
        node.get("kind").and_then(serde_json::Value::as_str),
        Some("null")
    ) || matches!(
        node.get("kind").and_then(serde_json::Value::as_str),
        Some("record" | "list")
    ) && node
        .get("value")
        .and_then(serde_json::Value::as_array)
        .is_some_and(Vec::is_empty)
}

fn scalar_integer(bytes: &[u8]) -> Result<i64, types::StreamError> {
    let node = scalar_node(bytes)?;
    if node.get("kind").and_then(serde_json::Value::as_str) != Some("integer") {
        return Err(types::StreamError::InvalidRequestAuthority);
    }
    node.get("value")
        .and_then(serde_json::Value::as_i64)
        .ok_or(types::StreamError::InvalidRequestAuthority)
}

fn header(name: &str, value: &str) -> types::Header {
    types::Header {
        name: name.into(),
        value: value.into(),
    }
}

fn write_chunks(handle: types::StreamHandle, bytes: &[u8]) -> Result<(), types::StreamError> {
    let mut chunks = bytes.chunks(65_536).peekable();
    let mut sequence = 0_u64;
    while let Some(chunk) = chunks.next() {
        provider_streaming_host::write_request_chunk(&types::RequestChunk {
            handle,
            sequence,
            bytes: chunk.to_vec(),
            end: chunks.peek().is_none(),
        })?;
        sequence = sequence
            .checked_add(1)
            .ok_or(types::StreamError::BodyLimit)?;
    }
    Ok(())
}

fn result_output(value: &str) -> Result<Vec<u8>, types::StreamError> {
    #[cfg(feature = "malformed-result")]
    {
        let _ = value;
        return Ok(b"not-a-plugin-value".to_vec());
    }
    #[cfg(all(feature = "oversized-result", not(feature = "malformed-result")))]
    {
        let _ = value;
        return Ok(vec![0; 16 * 1024 * 1024 + 1]);
    }
    #[cfg(not(any(feature = "malformed-result", feature = "oversized-result")))]
    {
        serde_json::to_vec(&serde_json::json!({
            "type_id": "sim:string@1",
            "representation": {"family":"scalar", "value":{"kind":"string", "value":value}}
        }))
        .map_err(|_| types::StreamError::InvalidInvocationResult)
    }
}

fn manifest() -> plugin::ManifestProjection {
    plugin::ManifestProjection {
        component_world: "zed:sim-provider-plugin@1.0.0".into(),
        schema_version: 1,
        identifier: "zed.sim.provider.openrouter".into(),
        plugin_version: version(1, 0, 0),
        api: plugin::ApiRequirement {
            major: 1,
            minimum_minor: 0,
            maximum_minor: 0,
            required_features: vec![
                "provider.bindings.v1".into(),
                "provider.streaming.v2".into(),
            ],
        },
        nodes: vec![plugin::Node {
            id: "OpenRouterLLMNode".into(),
            version: version(1, 0, 0),
            display_name: "OpenRouter LLM".into(),
            category: "partner/text/OpenRouter".into(),
            ports: vec![
                port(
                    "prompt",
                    "sim:string@1",
                    plugin::PortDirection::Input,
                    plugin::PortPresence::Required,
                ),
                port(
                    "model",
                    "sim:dynamic-combo@1",
                    plugin::PortDirection::Input,
                    plugin::PortPresence::Required,
                ),
                port(
                    "seed",
                    "sim:integer@1",
                    plugin::PortDirection::Input,
                    plugin::PortPresence::Required,
                ),
                port(
                    "system_prompt",
                    "sim:string@1",
                    plugin::PortDirection::Input,
                    plugin::PortPresence::Optional,
                ),
                port(
                    "output_0",
                    "sim:string@1",
                    plugin::PortDirection::Output,
                    plugin::PortPresence::Required,
                ),
            ],
            determinism: plugin::DeterminismPolicy::External,
            cache: plugin::CachePolicy::Never,
            effects: plugin::EffectPolicy::Provider,
        }],
        capabilities: vec![
            capability(
                plugin::CapabilityKind::NetworkProvider,
                "zed.sim.provider.openrouter|https://api.comfy.org/proxy/openrouter/api/v1/chat/completions",
                20,
            ),
            capability(plugin::CapabilityKind::Secret, "sim_org.auth_token", 1),
            capability(plugin::CapabilityKind::Secret, "sim_org.api_key", 1),
            capability(
                plugin::CapabilityKind::ProviderCost,
                "zed.sim.provider.openrouter|https://api.comfy.org/proxy/openrouter/api/v1/chat/completions",
                1,
            ),
        ],
        ui: Vec::new(),
        routes: vec![plugin::RouteDeclaration {
            id: "COMFY-API-EXT-0127".into(),
            method: "POST".into(),
            path: "/proxy/openrouter/api/v1/chat/completions".into(),
            maximum_request_bytes: 1_048_576,
            maximum_response_bytes: 16_777_216,
        }],
        legacy_mappings: Vec::new(),
    }
}

fn port(
    id: &str,
    type_id: &str,
    direction: plugin::PortDirection,
    presence: plugin::PortPresence,
) -> plugin::Port {
    plugin::Port {
        id: id.into(),
        name: id.into(),
        direction,
        type_id: type_id.into(),
        cardinality: plugin::PortCardinality::Singular,
        presence,
        hidden: false,
        lazy: false,
        default: None,
        serialization: plugin::PortSerialization::Inline,
        accepted_legacy_names: Vec::new(),
    }
}

fn capability(
    kind: plugin::CapabilityKind,
    scope: &str,
    operations: u64,
) -> plugin::CapabilityRequest {
    plugin::CapabilityRequest {
        kind,
        scope: scope.into(),
        quota: plugin::CapabilityQuota {
            maximum_operations: operations,
            maximum_request_bytes: 1_048_576,
            maximum_response_bytes: 16_777_216,
            maximum_total_bytes: 17_825_792,
            maximum_handles: 8,
            timeout_milliseconds: 60_000,
        },
    }
}

fn version(major: u16, minor: u16, patch: u16) -> plugin::ApiVersion {
    plugin::ApiVersion {
        major,
        minor,
        patch,
    }
}

export!(OpenRouterComponent);
