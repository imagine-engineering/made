wit_bindgen::generate!({
    path: "../../../../../sim_plugin_sdk/wit/provider-v2",
    world: "sim-provider-plugin",
    with: { "zed:sim-plugin/types@1.0.0": generate },
});

use exports::zed::sim_provider_plugin::provider_node;
use zed::sim_plugin::types as plugin;
use zed::sim_provider_plugin::{invocation_input_host, provider_streaming_host, types};

struct AnthropicComponent;

impl provider_node::Guest for AnthropicComponent {
    fn manifest() -> types::ManifestV2 {
        types::ManifestV2 {
            schema_version: 2,
            component_world: "zed:sim-provider-plugin@2.0.0".into(),
            manifest: manifest(),
            provider_binding: plugin::ProviderBindingSet {
                schema_version: 1,
                implementation_namespace: "zed.sim.provider.anthropic".into(),
                bindings_sha256: "cf0228cb4b3832a98ab8984f2fd050f5615b575cdecb5fda1446aa8d25fb193b"
                    .into(),
                bindings: vec![plugin::ProviderBindingClaim {
                    feature_id: "COMFY-NODE-0056".into(),
                    node_id: "ClaudeNode".into(),
                    contract_sha256:
                        "5bf889c4754da87caf8ace62b5495da1afe895a45aad0b8bdc0d497a8f632bbb".into(),
                    transport_schema: "zed:sim-provider-transport@1".into(),
                    materializer_schema: "zed:sim-provider-materializer@1".into(),
                }],
            },
            streaming: types::StreamingContract {
                methods: vec![types::HttpMethod::Post, types::HttpMethod::Put],
                maximum_headers: 8,
                maximum_header_bytes: 16_384,
                maximum_request_body_bytes: 1_048_576,
                maximum_response_body_bytes: 16_777_216,
                maximum_chunk_bytes: 65_536,
                maximum_ndjson_line_bytes: 65_536,
                maximum_wait_milliseconds: 1_000,
                maximum_uploads: 20,
                maximum_upload_body_bytes: 49_172_480,
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
        if node_id != "ClaudeNode" {
            return Err(types::StreamError::InvalidRequestAuthority);
        }
        let prompt = scalar_text(&input("prompt")?.abi_bytes)?;
        if prompt.len() > 65_536 {
            return Err(types::StreamError::BodyLimit);
        }
        let model = scalar_text(&input("model")?.abi_bytes)?;
        let seed = scalar_integer(&input("seed")?.abi_bytes)?;
        let temperature = scalar_float(&input("temperature")?.abi_bytes)?;
        let max_tokens = scalar_integer(&input("max_tokens")?.abi_bytes)?;
        let reasoning_effort_state = invocation_input_host::get_input_state("reasoning_effort")
            .map_err(|_| types::StreamError::InvalidRequestAuthority)?;
        let reasoning_effort = if reasoning_effort_state.present {
            scalar_text(&input("reasoning_effort")?.abi_bytes)?
        } else if model == "Haiku 4.5" {
            "off".to_owned()
        } else {
            return Err(types::StreamError::InvalidRequestAuthority);
        };
        let system_prompt_state = invocation_input_host::get_input_state("system_prompt")
            .map_err(|_| types::StreamError::InvalidRequestAuthority)?;
        let system_prompt = if system_prompt_state.present {
            Some(scalar_text(&input("system_prompt")?.abi_bytes)?)
        } else {
            None
        };
        let image_state = invocation_input_host::get_input_state("images")
            .map_err(|_| types::StreamError::InvalidRequestAuthority)?;
        if image_state.length > 20 {
            return Err(types::StreamError::InvalidUpload);
        }
        let mut image_urls = Vec::new();
        let mut admitted_media = Vec::new();
        let mut total_images = 0_u32;
        let mut total_image_pixels = 0_u64;
        for value_index in 0..image_state.length {
            let image_handle = invocation_input_host::take_input("images", value_index)
                .map_err(|_| types::StreamError::InvalidUpload)?;
            let image = invocation_input_host::read_handle(image_handle)
                .map_err(|_| types::StreamError::InvalidUpload)?;
            let (batch, image_pixels) = image_batch(&image.abi_bytes)?;
            let media_count = invocation_input_host::media_input_count(image_handle)
                .map_err(|_| types::StreamError::InvalidUpload)?;
            if media_count != batch {
                return Err(types::StreamError::InvalidUpload);
            }
            total_images = total_images
                .checked_add(media_count)
                .filter(|count| *count <= 20)
                .ok_or(types::StreamError::InvalidUpload)?;
            total_image_pixels = total_image_pixels
                .checked_add(
                    image_pixels
                        .checked_mul(u64::from(batch))
                        .ok_or(types::StreamError::InvalidUpload)?,
                )
                .ok_or(types::StreamError::InvalidUpload)?;
            for batch_index in 0..media_count {
                let media = invocation_input_host::read_media_input_at(image_handle, batch_index)
                    .map_err(|_| types::StreamError::InvalidUpload)?;
                admitted_media.push((value_index, batch_index, media));
            }
        }
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
        let request_head = types::RequestHead {
            endpoint: "https://api.comfy.org/proxy/anthropic/v1/messages".into(),
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
        };
        let mut handle = provider_streaming_host::start_request(context, &request_head)?;
        for (_, _, media) in admitted_media {
            let upload = provider_streaming_host::start_upload(&types::UploadRequest {
                handle,
                port_id: "images".into(),
                media_type: media.media_type,
                byte_length: u64::try_from(media.bytes.len())
                    .map_err(|_| types::StreamError::BodyLimit)?,
                content_sha256: media.content_sha256,
            })?;
            write_chunks(upload, &media.bytes, true)?;
            let completion =
                provider_streaming_host::complete_upload(types::UploadCompletionRequest {
                    handle: upload,
                })?;
            image_urls.push(completion.download_url);
        }
        let body = request_body(
            &prompt,
            &model,
            seed,
            temperature,
            max_tokens,
            &reasoning_effort,
            system_prompt.as_deref(),
            &image_urls,
        )?;
        write_chunks(handle, &body, false)?;
        let maximum_microunits =
            maximum_cost_microunits(&model, max_tokens, body.len(), total_image_pixels)?;
        let cost = provider_streaming_host::request_cost(&types::CostRequest {
            handle,
            operation: "anthropic.messages".into(),
            currency: "USD".into(),
            maximum_microunits,
        })?;
        if !cost.accepted || cost.receipt.is_empty() {
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
        let mut retry_count = 0_u8;
        let mut normal_retries = 0_u8;
        let mut rate_limit_retries = 0_u8;
        let mut normal_delay_milliseconds = 1_000_u64;
        let mut rate_limit_delay_milliseconds = 1_000_u64;
        let mut response_status = 200u16;
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
                    let fallback_delay = if head.status == 429 && rate_limit_retries < 16 {
                        rate_limit_retries = rate_limit_retries
                            .checked_add(1)
                            .ok_or(types::StreamError::WaitLimit)?;
                        let delay = rate_limit_delay_milliseconds.min(30_000);
                        rate_limit_delay_milliseconds =
                            rate_limit_delay_milliseconds.saturating_mul(2);
                        Some(delay)
                    } else if matches!(head.status, 408 | 500 | 502 | 503 | 504)
                        && normal_retries < 3
                    {
                        normal_retries = normal_retries
                            .checked_add(1)
                            .ok_or(types::StreamError::WaitLimit)?;
                        let delay = normal_delay_milliseconds;
                        normal_delay_milliseconds = normal_delay_milliseconds.saturating_mul(2);
                        Some(delay)
                    } else {
                        None
                    };
                    let Some(fallback_delay) = fallback_delay else {
                        response_status = head.status;
                        after_sequence = Some(sequence);
                        continue;
                    };
                    retry_count = retry_count
                        .checked_add(1)
                        .ok_or(types::StreamError::WaitLimit)?;
                    provider_streaming_host::report_progress(&types::Progress {
                        handle,
                        sequence: u64::from(retry_count),
                        completed: 0,
                        total: 100,
                        message: Some("Retrying request".into()),
                    })?;
                    handle = provider_streaming_host::retry_request(types::RetryRequestOptions {
                        handle,
                        delay_milliseconds: fallback_delay,
                    })?;
                    write_chunks(handle, &body, false)?;
                    after_sequence = None;
                    response.clear();
                    response_status = 200;
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
                    if response.len().saturating_add(bytes.len()) > 16_777_216 {
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
        if response_status != 200 {
            return Err(types::StreamError::HostFailure);
        }
        let parsed = parse_anthropic_response(&response)?;
        if parsed
            .cost_microunits
            .is_some_and(|actual| actual > cost.approved_microunits)
        {
            return Err(types::StreamError::InvalidCostRequest);
        }
        provider_streaming_host::report_progress(&types::Progress {
            handle,
            sequence: u64::from(retry_count).saturating_add(1),
            completed: 100,
            total: 100,
            message: Some("Completed".into()),
        })?;
        let output = result_output(&parsed.text)?;
        Ok(types::InvocationResult {
            outputs: vec![types::MaterializedOutput {
                port_id: "output_0".into(),
                value: types::EncodedValue {
                    type_id: "sim:string@1".into(),
                    family: plugin::ValueFamily::Scalar,
                    abi_bytes: output,
                },
            }],
            receipt,
        })
    }
}

struct ParsedResponse {
    text: String,
    cost_microunits: Option<u64>,
}

fn parse_anthropic_response(response: &[u8]) -> Result<ParsedResponse, types::StreamError> {
    let response: MessagesResponse =
        serde_json::from_slice(response).map_err(|_| types::StreamError::InvalidTerminal)?;
    let text = response
        .content
        .unwrap_or_default()
        .into_iter()
        .filter_map(|block| match block {
            ResponseBlock::Text { text } if !text.is_empty() => Some(text),
            ResponseBlock::Text { .. } | ResponseBlock::Thinking { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let cost_microunits =
        usage_cost_microunits(response.model.as_deref(), response.usage.as_ref())?;
    Ok(ParsedResponse {
        text: if text.is_empty() {
            "Empty response from Claude model.".into()
        } else {
            text
        },
        cost_microunits,
    })
}

#[derive(serde::Deserialize)]
struct MessagesResponse {
    model: Option<String>,
    content: Option<Vec<ResponseBlock>>,
    usage: Option<Usage>,
}

#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ResponseBlock {
    Text {
        text: String,
    },
    Thinking {
        #[serde(rename = "thinking")]
        _thinking: String,
    },
}

#[derive(serde::Deserialize)]
struct Usage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation: Option<CacheCreation>,
}

#[derive(serde::Deserialize)]
struct CacheCreation {
    ephemeral_5m_input_tokens: Option<u64>,
    ephemeral_1h_input_tokens: Option<u64>,
}

fn result_output(value: &str) -> Result<Vec<u8>, types::StreamError> {
    #[cfg(feature = "malformed-result")]
    {
        let _ = value;
        Ok(b"not-a-plugin-value".to_vec())
    }
    #[cfg(all(feature = "oversized-result", not(feature = "malformed-result")))]
    {
        let _ = value;
        Ok(vec![0; 16 * 1024 * 1024 + 1])
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

fn input(port: &str) -> Result<plugin::EncodedValue, types::StreamError> {
    invocation_input_host::read_scalar_input(port, 0)
        .map_err(|_| types::StreamError::InvalidRequestAuthority)
}

fn scalar_node(bytes: &[u8]) -> Result<serde_json::Value, types::StreamError> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|_| types::StreamError::InvalidRequestAuthority)?;
    value
        .pointer("/representation/value")
        .cloned()
        .ok_or(types::StreamError::InvalidRequestAuthority)
}

fn image_batch(bytes: &[u8]) -> Result<(u32, u64), types::StreamError> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|_| types::StreamError::InvalidUpload)?;
    let shape = value
        .pointer("/representation/value/descriptor/shape")
        .and_then(serde_json::Value::as_array)
        .ok_or(types::StreamError::InvalidUpload)?;
    if shape.len() != 4 {
        return Err(types::StreamError::InvalidUpload);
    }
    let batch = shape[0]
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value != 0)
        .ok_or(types::StreamError::InvalidUpload)?;
    let height = shape[1]
        .as_u64()
        .filter(|value| *value != 0)
        .ok_or(types::StreamError::InvalidUpload)?;
    let width = shape[2]
        .as_u64()
        .filter(|value| *value != 0)
        .ok_or(types::StreamError::InvalidUpload)?;
    let pixels = height
        .checked_mul(width)
        .ok_or(types::StreamError::InvalidUpload)?;
    Ok((batch, pixels.min(1_568 * 1_568)))
}

fn maximum_cost_microunits(
    model_label: &str,
    max_tokens: i64,
    request_body_bytes: usize,
    image_pixels: u64,
) -> Result<u64, types::StreamError> {
    let (input_rate_quarters, output_rate_quarters) = match model_label {
        "Opus 4.7" | "Opus 4.6" => (20_u64, 100_u64),
        "Sonnet 4.6" | "Sonnet 4.5" => (12, 60),
        "Haiku 4.5" => (4, 20),
        _ => return Err(types::StreamError::InvalidCostRequest),
    };
    let input_tokens = u64::try_from(request_body_bytes)
        .ok()
        .and_then(|bytes| bytes.checked_add(image_pixels))
        .ok_or(types::StreamError::InvalidCostRequest)?;
    let output_tokens =
        u64::try_from(max_tokens).map_err(|_| types::StreamError::InvalidCostRequest)?;
    // One token per retained request byte or downscaled source pixel is a conservative upper
    // bound; applying every source cache multiplier to that same bound cannot under-authorize.
    let input_quarters = input_tokens
        .checked_mul(input_rate_quarters)
        .ok_or(types::StreamError::InvalidCostRequest)?;
    let quarters = input_quarters
        .checked_add(
            output_tokens
                .checked_mul(output_rate_quarters)
                .ok_or(types::StreamError::InvalidCostRequest)?,
        )
        .and_then(|value| value.checked_add(input_quarters / 10))
        .and_then(|value| value.checked_add(input_quarters.checked_mul(5)?.checked_div(4)?))
        .and_then(|value| value.checked_add(input_quarters.checked_mul(2)?))
        .ok_or(types::StreamError::InvalidCostRequest)?;
    Ok(quarters / 4)
}

fn write_chunks(
    handle: types::StreamHandle,
    bytes: &[u8],
    upload: bool,
) -> Result<(), types::StreamError> {
    let maximum_chunk_bytes = if upload { 16 } else { 65_536 };
    let mut chunks = bytes.chunks(maximum_chunk_bytes).peekable();
    let mut sequence = 0_u64;
    while let Some(chunk) = chunks.next() {
        let request = types::RequestChunk {
            handle,
            sequence,
            bytes: chunk.to_vec(),
            end: chunks.peek().is_none(),
        };
        if upload {
            provider_streaming_host::write_upload_chunk(&request)?;
        } else {
            provider_streaming_host::write_request_chunk(&request)?;
        }
        sequence = sequence
            .checked_add(1)
            .ok_or(types::StreamError::BodyLimit)?;
    }
    Ok(())
}

fn usage_cost_microunits(
    model: Option<&str>,
    usage: Option<&Usage>,
) -> Result<Option<u64>, types::StreamError> {
    let (Some(model), Some(usage)) = (model, usage) else {
        return Ok(None);
    };
    let (input_rate_quarters, output_rate_quarters) = if model.contains("opus-4-") {
        (20_u64, 100_u64)
    } else if model.contains("sonnet-4") {
        (12, 60)
    } else if model.contains("haiku-4-5") {
        (4, 20)
    } else {
        return Ok(None);
    };
    let cache_creation = usage.cache_creation.as_ref();
    let quarters = usage
        .input_tokens
        .unwrap_or(0)
        .checked_mul(input_rate_quarters)
        .and_then(|value| {
            value.checked_add(
                usage
                    .output_tokens
                    .unwrap_or(0)
                    .checked_mul(output_rate_quarters)?,
            )
        })
        .and_then(|value| {
            value.checked_add(
                usage
                    .cache_read_input_tokens
                    .unwrap_or(0)
                    .checked_mul(input_rate_quarters)?
                    / 10,
            )
        })
        .and_then(|value| {
            value.checked_add(
                cache_creation
                    .and_then(|cache| cache.ephemeral_5m_input_tokens)
                    .unwrap_or(0)
                    .checked_mul(input_rate_quarters)?
                    .checked_mul(5)?
                    / 4,
            )
        })
        .and_then(|value| {
            value.checked_add(
                cache_creation
                    .and_then(|cache| cache.ephemeral_1h_input_tokens)
                    .unwrap_or(0)
                    .checked_mul(input_rate_quarters)?
                    .checked_mul(2)?,
            )
        })
        .ok_or(types::StreamError::InvalidTerminal)?;
    Ok(Some(quarters / 4))
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

fn scalar_integer(bytes: &[u8]) -> Result<i64, types::StreamError> {
    let node = scalar_node(bytes)?;
    if node.get("kind").and_then(serde_json::Value::as_str) != Some("integer") {
        return Err(types::StreamError::InvalidRequestAuthority);
    }
    node.get("value")
        .and_then(serde_json::Value::as_i64)
        .ok_or(types::StreamError::InvalidRequestAuthority)
}

fn scalar_float(bytes: &[u8]) -> Result<f64, types::StreamError> {
    let node = scalar_node(bytes)?;
    if node.get("kind").and_then(serde_json::Value::as_str) != Some("float") {
        return Err(types::StreamError::InvalidRequestAuthority);
    }
    node.get("value")
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or(types::StreamError::InvalidRequestAuthority)
}

fn header(name: &str, value: &str) -> types::Header {
    types::Header {
        name: name.into(),
        value: value.into(),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "The request builder mirrors the source node's explicit input ports"
)]
fn request_body(
    prompt: &str,
    model_label: &str,
    seed: i64,
    temperature: f64,
    max_tokens: i64,
    reasoning_effort: &str,
    system_prompt: Option<&str>,
    image_urls: &[String],
) -> Result<Vec<u8>, types::StreamError> {
    if prompt.trim().is_empty()
        || !(0..=2_147_483_647).contains(&seed)
        || !(0.0..=1.0).contains(&temperature)
        || !(4_096..=64_000).contains(&max_tokens)
        || image_urls.len() > 20
    {
        return Err(types::StreamError::InvalidRequestAuthority);
    }
    let model = match model_label {
        "Opus 4.7" => "claude-opus-4-7",
        "Opus 4.6" => "claude-opus-4-6",
        "Sonnet 4.6" => "claude-sonnet-4-6",
        "Sonnet 4.5" => "claude-sonnet-4-5-20250929",
        "Haiku 4.5" => "claude-haiku-4-5-20251001",
        _ => return Err(types::StreamError::InvalidRequestAuthority),
    };
    let thinking = reasoning_effort != "off" && model_label != "Haiku 4.5";
    let mut content = image_urls
        .iter()
        .map(|url| serde_json::json!({"type":"image","source":{"type":"url","url":url}}))
        .collect::<Vec<_>>();
    content.push(serde_json::json!({"type":"text","text":prompt}));
    let mut request = serde_json::json!({
        "model": model,
        "messages": [{"role":"user","content":content}],
        "max_tokens": max_tokens,
    });
    if let Some(system_prompt) = system_prompt.filter(|value| !value.is_empty()) {
        request["system"] = serde_json::Value::String(system_prompt.to_owned());
    }
    if thinking {
        if matches!(model_label, "Opus 4.7" | "Opus 4.6" | "Sonnet 4.6") {
            request["thinking"] = serde_json::json!({"type":"adaptive"});
            request["output_config"] = serde_json::json!({"effort":reasoning_effort});
        } else {
            let requested = match reasoning_effort {
                "low" => 2_048,
                "medium" => 8_192,
                "high" => 16_384,
                _ => return Err(types::StreamError::InvalidRequestAuthority),
            };
            request["thinking"] = serde_json::json!({
                "type":"enabled",
                "budget_tokens": requested.min((max_tokens - 1_024).max(1_024)),
            });
        }
    } else if model_label != "Opus 4.7" {
        request["temperature"] = serde_json::json!(temperature);
    }
    let body =
        serde_json::to_vec(&request).map_err(|_| types::StreamError::InvalidRequestAuthority)?;
    if body.len() > 1_048_576 {
        return Err(types::StreamError::BodyLimit);
    }
    Ok(body)
}

fn manifest() -> plugin::ManifestProjection {
    plugin::ManifestProjection {
        component_world: "zed:sim-provider-plugin@1.0.0".into(),
        schema_version: 1,
        identifier: "zed.sim.provider.anthropic".into(),
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
            id: "ClaudeNode".into(),
            version: version(1, 0, 0),
            display_name: "Anthropic Claude".into(),
            category: "partner/text/Anthropic".into(),
            ports: vec![
                port(
                    "prompt",
                    "sim:string@1",
                    plugin::PortCardinality::Singular,
                    plugin::PortPresence::Required,
                ),
                port(
                    "model",
                    "sim:dynamic-combo@1",
                    plugin::PortCardinality::Singular,
                    plugin::PortPresence::Required,
                ),
                port(
                    "seed",
                    "sim:integer@1",
                    plugin::PortCardinality::Singular,
                    plugin::PortPresence::Required,
                ),
                port(
                    "temperature",
                    "sim:float@1",
                    plugin::PortCardinality::Singular,
                    plugin::PortPresence::Required,
                ),
                port(
                    "max_tokens",
                    "sim:integer@1",
                    plugin::PortCardinality::Singular,
                    plugin::PortPresence::Required,
                ),
                port(
                    "reasoning_effort",
                    "sim:string@1",
                    plugin::PortCardinality::Singular,
                    plugin::PortPresence::Optional,
                ),
                port(
                    "system_prompt",
                    "sim:string@1",
                    plugin::PortCardinality::Singular,
                    plugin::PortPresence::Optional,
                ),
                port(
                    "images",
                    "sim:image@1",
                    plugin::PortCardinality::List,
                    plugin::PortPresence::Optional,
                ),
                output_port(),
            ],
            determinism: plugin::DeterminismPolicy::External,
            cache: plugin::CachePolicy::Never,
            effects: plugin::EffectPolicy::Provider,
        }],
        capabilities: vec![
            capability(
                plugin::CapabilityKind::NetworkProvider,
                "zed.sim.provider.anthropic|https://api.comfy.org/proxy/anthropic/v1/messages",
                32,
            ),
            capability(plugin::CapabilityKind::Secret, "sim_org.auth_token", 1),
            capability(plugin::CapabilityKind::Secret, "sim_org.api_key", 1),
            capability(
                plugin::CapabilityKind::ProviderUpload,
                "zed.sim.provider.anthropic|https://api.comfy.org/proxy/anthropic/v1/messages",
                40,
            ),
            capability(
                plugin::CapabilityKind::ProviderCost,
                "zed.sim.provider.anthropic|https://api.comfy.org/proxy/anthropic/v1/messages",
                1,
            ),
        ],
        ui: Vec::new(),
        routes: vec![plugin::RouteDeclaration {
            id: "COMFY-API-EXT-0001".into(),
            method: "POST".into(),
            path: "/proxy/anthropic/v1/messages".into(),
            maximum_request_bytes: 1_048_576,
            maximum_response_bytes: 16_777_216,
        }],
        legacy_mappings: Vec::new(),
    }
}

fn port(
    id: &str,
    type_id: &str,
    cardinality: plugin::PortCardinality,
    presence: plugin::PortPresence,
) -> plugin::Port {
    plugin::Port {
        id: id.into(),
        name: id.into(),
        direction: plugin::PortDirection::Input,
        type_id: type_id.into(),
        cardinality,
        presence,
        hidden: false,
        lazy: cardinality == plugin::PortCardinality::List,
        default: None,
        serialization: if type_id == "sim:image@1" {
            plugin::PortSerialization::Handle
        } else {
            plugin::PortSerialization::Inline
        },
        accepted_legacy_names: Vec::new(),
    }
}

fn output_port() -> plugin::Port {
    let mut output = port(
        "output_0",
        "sim:string@1",
        plugin::PortCardinality::Singular,
        plugin::PortPresence::Required,
    );
    output.direction = plugin::PortDirection::Output;
    output
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
            maximum_request_bytes: 49_172_480,
            maximum_response_bytes: 16_777_216,
            maximum_total_bytes: 65_949_696,
            maximum_handles: 32,
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

export!(AnthropicComponent);
