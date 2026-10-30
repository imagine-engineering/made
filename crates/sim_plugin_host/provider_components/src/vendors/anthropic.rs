use std::{collections::BTreeMap, error::Error, fmt};

use crate::common::{
    CanonicalTypeId, ConformanceError, CostRequest, CostResponse, EncodedOutput, ExpectedOutput,
    Header, HttpMethod, MaterializationError, MaterializationSink, MaterializedBatch, PluginValue,
    Progress, ProviderTransport, RequestHead, ResponseEvent, ScalarValue, ScriptStep,
    ScriptedActuator, SignedPutRequest, StorageCreateRequest, StorageCreateResponse, StreamHandle,
    TypeRegistry, ValueFamily, publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};

pub const NAMESPACE: &str = "zed.sim.provider.anthropic";
pub const SOURCE_SHA256: &str = "36ca216053c7a84cc847d5a88d971b76a945240e58aef88148efcfb969d62edb";
pub const NODE_FEATURE_ID: &str = "COMFY-NODE-0056";
pub const NODE_IDENTIFIER: &str = "ClaudeNode";
pub const ROUTE_FEATURE_ID: &str = "COMFY-API-EXT-0001";
pub const MESSAGES_PATH: &str = "/proxy/anthropic/v1/messages";

const MAXIMUM_FIXTURE_BYTES: usize = 1024 * 1024;
const MAXIMUM_REQUEST_BYTES: usize = 1024 * 1024;
const MAXIMUM_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAXIMUM_CHUNK_BYTES: usize = 64 * 1024;
const MAXIMUM_RECEIPT_BYTES: usize = 32 * 1024;
const MAXIMUM_IMAGES: usize = 20;
const MAXIMUM_POLLS: u32 = 64;
const WAIT_MILLISECONDS: u64 = 1_000;

const REQUIRED_FIXTURE_KEYS: &[&str] = &[
    "aliases",
    "auth_header",
    "auth_value",
    "environment",
    "expected_cost_microunits",
    "expected_output",
    "image_hex",
    "image_url",
    "max_tokens",
    "method",
    "model_label",
    "namespace",
    "node_count",
    "node_feature_id",
    "node_identifier",
    "path",
    "prompt",
    "reasoning_effort",
    "request_body",
    "response_body",
    "route_count",
    "route_feature_id",
    "schema",
    "source_sha256",
    "system_prompt",
    "usage_source",
];

#[derive(Debug)]
enum AnthropicError {
    Contract(&'static str),
    Transport(ConformanceError),
    Materialization(MaterializationError),
    Json(serde_json::Error),
}

impl fmt::Display for AnthropicError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(message) => formatter.write_str(message),
            Self::Transport(error) => write!(formatter, "{error}"),
            Self::Materialization(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
        }
    }
}

impl Error for AnthropicError {}

impl From<ConformanceError> for AnthropicError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}

impl From<MaterializationError> for AnthropicError {
    fn from(error: MaterializationError) -> Self {
        Self::Materialization(error)
    }
}

impl From<serde_json::Error> for AnthropicError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Clone, Debug)]
struct AnthropicFixture {
    values: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
struct ImageInput {
    file_name: String,
    content_type: String,
    width: u32,
    height: u32,
    bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
struct ClaudeInput {
    prompt: String,
    model: Model,
    max_tokens: u64,
    temperature: f64,
    seed: u64,
    reasoning_effort: ReasoningEffort,
    system_prompt: Option<String>,
    images: Vec<ImageInput>,
}

impl AnthropicFixture {
    fn parse(bytes: &str) -> Result<Self, AnthropicError> {
        if bytes.is_empty() || bytes.len() > MAXIMUM_FIXTURE_BYTES {
            return Err(AnthropicError::Contract("invalid anthropic fixture size"));
        }
        let mut values = BTreeMap::new();
        for line in bytes.lines() {
            if line.is_empty() {
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .ok_or(AnthropicError::Contract("invalid anthropic fixture line"))?;
            if key.is_empty()
                || value.len() > MAXIMUM_RESPONSE_BYTES
                || values.insert(key.to_owned(), value.to_owned()).is_some()
            {
                return Err(AnthropicError::Contract("invalid anthropic fixture field"));
            }
        }
        let keys = values.keys().map(String::as_str).collect::<Vec<_>>();
        if keys != REQUIRED_FIXTURE_KEYS {
            return Err(AnthropicError::Contract(
                "anthropic fixture fields do not match the sealed schema",
            ));
        }
        Ok(Self { values })
    }

    fn value(&self, key: &str) -> Result<&str, AnthropicError> {
        self.values
            .get(key)
            .map(String::as_str)
            .ok_or(AnthropicError::Contract("missing anthropic fixture field"))
    }

    fn unsigned(&self, key: &str) -> Result<u64, AnthropicError> {
        self.value(key)?
            .parse()
            .map_err(|_| AnthropicError::Contract("invalid anthropic fixture integer"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReasoningEffort {
    Off,
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    fn parse(value: &str) -> Result<Self, AnthropicError> {
        match value {
            "off" => Ok(Self::Off),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            _ => Err(AnthropicError::Contract(
                "unsupported anthropic reasoning effort",
            )),
        }
    }

    fn as_str(self) -> Option<&'static str> {
        match self {
            Self::Off => None,
            Self::Low => Some("low"),
            Self::Medium => Some("medium"),
            Self::High => Some("high"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Model {
    Opus47,
    Opus46,
    Sonnet46,
    Sonnet45,
    Haiku45,
}

impl Model {
    fn parse(label: &str) -> Result<Self, AnthropicError> {
        match label {
            "Opus 4.7" => Ok(Self::Opus47),
            "Opus 4.6" => Ok(Self::Opus46),
            "Sonnet 4.6" => Ok(Self::Sonnet46),
            "Sonnet 4.5" => Ok(Self::Sonnet45),
            "Haiku 4.5" => Ok(Self::Haiku45),
            _ => Err(AnthropicError::Contract("unsupported anthropic model")),
        }
    }

    fn identifier(self) -> &'static str {
        match self {
            Self::Opus47 => "claude-opus-4-7",
            Self::Opus46 => "claude-opus-4-6",
            Self::Sonnet46 => "claude-sonnet-4-6",
            Self::Sonnet45 => "claude-sonnet-4-5-20250929",
            Self::Haiku45 => "claude-haiku-4-5-20251001",
        }
    }

    fn uses_adaptive_thinking(self) -> bool {
        matches!(self, Self::Opus47 | Self::Opus46 | Self::Sonnet46)
    }

    fn supports_thinking(self) -> bool {
        !matches!(self, Self::Haiku45)
    }
}

#[derive(Serialize)]
struct MessagesRequest<'a> {
    model: &'a str,
    messages: Vec<Message<'a>>,
    max_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<&'a str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<Thinking<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<OutputConfig<'a>>,
}

#[derive(Serialize)]
struct Message<'a> {
    role: &'static str,
    content: Vec<Content<'a>>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Content<'a> {
    Image { source: ImageSource<'a> },
    Text { text: &'a str },
}

#[derive(Serialize)]
struct ImageSource<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    url: &'a str,
}

#[derive(Serialize)]
struct Thinking<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    budget_tokens: Option<u64>,
    #[serde(skip)]
    _lifetime: std::marker::PhantomData<&'a ()>,
}

#[derive(Serialize)]
struct OutputConfig<'a> {
    effort: &'a str,
}

#[derive(Deserialize)]
struct MessagesResponse {
    model: Option<String>,
    content: Option<Vec<ResponseBlock>>,
    usage: Option<Usage>,
}

#[derive(Deserialize)]
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

#[derive(Deserialize)]
struct Usage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation: Option<CacheCreation>,
}

#[derive(Deserialize)]
struct CacheCreation {
    ephemeral_5m_input_tokens: Option<u64>,
    ephemeral_1h_input_tokens: Option<u64>,
}

#[derive(Default)]
struct RecordingSink {
    batches: Vec<MaterializedBatch>,
}

impl MaterializationSink for RecordingSink {
    fn publish(&mut self, batch: MaterializedBatch) -> Result<(), MaterializationError> {
        self.batches.push(batch);
        Ok(())
    }
}

pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let fixture = AnthropicFixture::parse(fixture).map_err(map_error)?;
    validate_claims(&fixture).map_err(map_error)?;
    run_success(&fixture, 2).map_err(map_error)?;
    run_success(&fixture, 3).map_err(map_error)?;
    run_error_cases(&fixture).map_err(map_error)?;
    validate_unsigned_artifacts().map_err(map_error)?;
    Ok(())
}

fn validate_claims(fixture: &AnthropicFixture) -> Result<(), AnthropicError> {
    let exact = fixture.value("schema")? == "zed.sim.provider.anthropic.v1"
        && fixture.value("namespace")? == NAMESPACE
        && fixture.value("aliases")?.is_empty()
        && fixture.unsigned("node_count")? == 1
        && fixture.value("node_feature_id")? == NODE_FEATURE_ID
        && fixture.value("node_identifier")? == NODE_IDENTIFIER
        && fixture.unsigned("route_count")? == 1
        && fixture.value("route_feature_id")? == ROUTE_FEATURE_ID
        && fixture.value("method")? == "POST"
        && fixture.value("path")? == MESSAGES_PATH
        && fixture.value("source_sha256")? == SOURCE_SHA256;
    if !exact {
        return Err(AnthropicError::Contract(
            "anthropic claims do not match the reviewed catalog",
        ));
    }
    Ok(())
}

fn request_body(fixture: &AnthropicFixture) -> Result<Vec<u8>, AnthropicError> {
    let body = build_request_body(fixture)?;
    if body != fixture.value("request_body")?.as_bytes() {
        return Err(AnthropicError::Contract(
            "anthropic request does not match the source fixture",
        ));
    }
    Ok(body)
}

fn build_request_body(fixture: &AnthropicFixture) -> Result<Vec<u8>, AnthropicError> {
    let input = fixture_input(fixture)?;
    let image_urls = input
        .images
        .iter()
        .map(|_| fixture.value("image_url").map(str::to_owned))
        .collect::<Result<Vec<_>, _>>()?;
    build_request_for_input(&input, &image_urls)
}

fn fixture_input(fixture: &AnthropicFixture) -> Result<ClaudeInput, AnthropicError> {
    Ok(ClaudeInput {
        prompt: fixture.value("prompt")?.to_owned(),
        model: Model::parse(fixture.value("model_label")?)?,
        max_tokens: fixture.unsigned("max_tokens")?,
        temperature: 1.0,
        seed: 7,
        reasoning_effort: ReasoningEffort::parse(fixture.value("reasoning_effort")?)?,
        system_prompt: (!fixture.value("system_prompt")?.is_empty())
            .then(|| fixture.value("system_prompt").map(str::to_owned))
            .transpose()?,
        images: vec![ImageInput {
            file_name: "reference.png".into(),
            content_type: "image/png".into(),
            width: 1,
            height: 1,
            bytes: decode_hex(fixture.value("image_hex")?)?,
        }],
    })
}

fn build_request_for_input(
    input: &ClaudeInput,
    image_urls: &[String],
) -> Result<Vec<u8>, AnthropicError> {
    validate_input(input)?;
    if image_urls.len() != input.images.len()
        || image_urls
            .iter()
            .any(|url| !url.starts_with("https://") || url.len() > 2_048)
    {
        return Err(AnthropicError::Contract("invalid anthropic image URL"));
    }
    let model = input.model;
    let max_tokens = input.max_tokens;
    let effort = input.reasoning_effort;
    let thinking_enabled = effort != ReasoningEffort::Off && model.supports_thinking();
    let thinking = if thinking_enabled && model.uses_adaptive_thinking() {
        Some(Thinking {
            kind: "adaptive",
            budget_tokens: None,
            _lifetime: std::marker::PhantomData,
        })
    } else if thinking_enabled {
        let requested_budget = match effort {
            ReasoningEffort::Low => 2_048,
            ReasoningEffort::Medium => 8_192,
            ReasoningEffort::High => 16_384,
            ReasoningEffort::Off => 0,
        };
        Some(Thinking {
            kind: "enabled",
            budget_tokens: Some(requested_budget.min(max_tokens.saturating_sub(1_024).max(1_024))),
            _lifetime: std::marker::PhantomData,
        })
    } else {
        None
    };
    let output_config = if thinking_enabled && model.uses_adaptive_thinking() {
        Some(OutputConfig {
            effort: effort
                .as_str()
                .ok_or(AnthropicError::Contract("missing reasoning effort"))?,
        })
    } else {
        None
    };
    let mut content = Vec::new();
    content
        .try_reserve(image_urls.len().saturating_add(1))
        .map_err(|_| AnthropicError::Contract("anthropic request allocation failed"))?;
    for image_url in image_urls {
        content.push(Content::Image {
            source: ImageSource {
                kind: "url",
                url: image_url,
            },
        });
    }
    content.push(Content::Text {
        text: &input.prompt,
    });
    let request = MessagesRequest {
        model: model.identifier(),
        messages: vec![Message {
            role: "user",
            content,
        }],
        max_tokens,
        system: input.system_prompt.as_deref(),
        temperature: (!thinking_enabled && model != Model::Opus47).then_some(input.temperature),
        top_p: None,
        top_k: None,
        stop_sequences: None,
        thinking,
        output_config,
    };
    let body = serde_json::to_vec(&request)?;
    if body.is_empty() || body.len() > MAXIMUM_REQUEST_BYTES {
        return Err(AnthropicError::Contract(
            "anthropic request exceeds its bound",
        ));
    }
    Ok(body)
}

fn python_strip(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
    })
}

fn validate_input(input: &ClaudeInput) -> Result<(), AnthropicError> {
    if python_strip(&input.prompt).is_empty()
        || input.prompt.len() > MAXIMUM_REQUEST_BYTES
        || !(4_096..=64_000).contains(&input.max_tokens)
        || !input.temperature.is_finite()
        || !(0.0..=1.0).contains(&input.temperature)
        || input.seed > 2_147_483_647
        || input.images.len() > MAXIMUM_IMAGES
    {
        return Err(AnthropicError::Contract("invalid anthropic node input"));
    }
    for image in &input.images {
        let pixels = u64::from(image.width)
            .checked_mul(u64::from(image.height))
            .ok_or(AnthropicError::Contract(
                "anthropic image dimensions overflow",
            ))?;
        if pixels == 0
            || image.bytes.is_empty()
            || image.bytes.len() > MAXIMUM_REQUEST_BYTES
            || image.file_name.is_empty()
            || image.file_name.len() > 256
            || !image.content_type.starts_with("image/")
        {
            return Err(AnthropicError::Contract("invalid anthropic image input"));
        }
    }
    Ok(())
}

fn request_head(fixture: &AnthropicFixture) -> Result<RequestHead, AnthropicError> {
    let auth_header = fixture.value("auth_header")?;
    if !matches!(auth_header, "Authorization" | "X-API-KEY") {
        return Err(AnthropicError::Contract("invalid anthropic auth selection"));
    }
    let auth_value = fixture.value("auth_value")?;
    if auth_value != "host-capability:auth_token_sim_org"
        && auth_value != "host-capability:api_key_sim_org"
    {
        return Err(AnthropicError::Contract(
            "anthropic auth must remain an opaque host capability",
        ));
    }
    let auth = if auth_header == "Authorization" {
        resolve_auth_header(Some(auth_value), None)?
    } else {
        resolve_auth_header(None, Some(auth_value))?
    };
    let secret_id = if auth_header == "Authorization" {
        "sim_org.auth_token"
    } else {
        "sim_org.api_key"
    };
    Ok(RequestHead {
        method: HttpMethod::Post,
        path: MESSAGES_PATH.to_owned(),
        secret_id: Some(secret_id.into()),
        headers: vec![
            Header {
                name: "Accept".into(),
                value: "application/json".into(),
            },
            auth,
            Header {
                name: "Comfy-Env".into(),
                value: fixture.value("environment")?.into(),
            },
            Header {
                name: "Comfy-Usage-Source".into(),
                value: fixture.value("usage_source")?.into(),
            },
            Header {
                name: "Content-Type".into(),
                value: "application/json".into(),
            },
        ],
    })
}

fn resolve_auth_header(
    auth_token: Option<&str>,
    api_key: Option<&str>,
) -> Result<Header, AnthropicError> {
    if let Some(auth_token) = auth_token {
        if auth_token != "host-capability:auth_token_sim_org" {
            return Err(AnthropicError::Contract(
                "invalid anthropic auth capability",
            ));
        }
        return Ok(Header {
            name: "Authorization".into(),
            value: auth_token.into(),
        });
    }
    if let Some(api_key) = api_key {
        if api_key != "host-capability:api_key_sim_org" {
            return Err(AnthropicError::Contract(
                "invalid anthropic auth capability",
            ));
        }
        return Ok(Header {
            name: "X-API-KEY".into(),
            value: api_key.into(),
        });
    }
    Err(AnthropicError::Contract(
        "missing anthropic auth capability",
    ))
}

fn run_success(fixture: &AnthropicFixture, generation: u32) -> Result<(), AnthropicError> {
    let body = request_body(fixture)?;
    let input = fixture_input(fixture)?;
    let head = request_head(fixture)?;
    let handle = StreamHandle {
        invocation: 41,
        slot: 1,
        generation,
    };
    let response = fixture.value("response_body")?.as_bytes();
    let midpoint = response.len() / 2;
    let chunks = [&response[..midpoint], &response[midpoint..]];
    if chunks.iter().any(|chunk| chunk.len() > MAXIMUM_CHUNK_BYTES) {
        return Err(AnthropicError::Contract(
            "anthropic response chunk exceeds its bound",
        ));
    }
    let cost = CostRequest {
        operation: "anthropic.messages".into(),
        currency: "USD".into(),
        maximum_microunits: fixture.unsigned("expected_cost_microunits")?,
    };
    let progress = Progress {
        sequence: 2,
        completed: 100,
        total: 100,
        message: Some("Completed".into()),
    };
    let mut script = vec![
        ScriptStep::CheckCancelled(None),
        ScriptStep::Start(head.clone(), Err(ConformanceError::WorkerLost)),
        ScriptStep::CheckCancelled(None),
        ScriptStep::Start(head, Ok(handle)),
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::CreateStorage(
            handle,
            storage_request(&input.images[0])?,
            StorageCreateResponse {
                upload_url: "https://fixture.invalid/signed/reference.png".into(),
                download_url: fixture.value("image_url")?.into(),
            },
        ),
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::PutSigned(
            handle,
            SignedPutRequest {
                upload_url: "https://fixture.invalid/signed/reference.png".into(),
                content_type: None,
                bytes: input.images[0].bytes.clone(),
            },
        ),
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::RequestChunk {
            handle,
            sequence: 0,
            bytes: body,
            end: true,
        },
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Cost(
            handle,
            cost.clone(),
            CostResponse {
                accepted: true,
                approved_microunits: cost.maximum_microunits,
                receipt: b"anthropic-cost".to_vec(),
            },
        ),
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Progress(
            handle,
            Progress {
                sequence: 0,
                completed: 0,
                total: 100,
                message: Some("Waiting for server".into()),
            },
        ),
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Wait {
            handle,
            after_sequence: None,
            timeout_milliseconds: WAIT_MILLISECONDS,
            outcome: ResponseEvent::TimedOut,
        },
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Progress(
            handle,
            Progress {
                sequence: 1,
                completed: 1,
                total: 100,
                message: Some("Waiting for server".into()),
            },
        ),
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Wait {
            handle,
            after_sequence: None,
            timeout_milliseconds: WAIT_MILLISECONDS,
            outcome: ResponseEvent::Head {
                status: 200,
                headers: vec![Header {
                    name: "Content-Type".into(),
                    value: "application/json".into(),
                }],
            },
        },
    ];
    for (index, chunk) in chunks.iter().enumerate() {
        script.push(ScriptStep::CheckCancelled(Some(handle)));
        script.push(ScriptStep::Wait {
            handle,
            after_sequence: Some(
                u64::try_from(index).map_err(|_| {
                    AnthropicError::Contract("anthropic response sequence overflow")
                })?,
            ),
            timeout_milliseconds: WAIT_MILLISECONDS,
            outcome: ResponseEvent::Chunk(chunk.to_vec()),
        });
    }
    script.extend([
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Wait {
            handle,
            after_sequence: Some(2),
            timeout_milliseconds: WAIT_MILLISECONDS,
            outcome: ResponseEvent::Completed(b"anthropic-terminal".to_vec()),
        },
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Progress(handle, progress.clone()),
    ]);
    let mut actuator = ScriptedActuator::new(script);
    let response_bytes = invoke_transport(
        fixture,
        &body_for_retry(fixture)?,
        &input,
        &cost,
        &mut actuator,
        handle,
    )?;
    let response: MessagesResponse = serde_json::from_slice(&response_bytes)?;
    let output = response_text(&response);
    if output != fixture.value("expected_output")?.replace("\\n", "\n") {
        return Err(AnthropicError::Contract(
            "anthropic text materialization differs from source",
        ));
    }
    let cost_microunits = response_cost_microunits(&response)
        .ok_or(AnthropicError::Contract("missing anthropic response cost"))?;
    if cost_microunits != cost.maximum_microunits {
        return Err(AnthropicError::Contract(
            "anthropic cost differs from the source equation",
        ));
    }
    let mut sink = RecordingSink::default();
    publish_text(&output, &mut sink)?;
    if sink.batches.len() != 1 || sink.batches[0].outputs().len() != 1 {
        return Err(AnthropicError::Contract(
            "anthropic output was not published atomically",
        ));
    }
    actuator.finish()?;
    Ok(())
}

fn body_for_retry(fixture: &AnthropicFixture) -> Result<Vec<u8>, AnthropicError> {
    request_body(fixture)
}

fn invoke_transport(
    fixture: &AnthropicFixture,
    body: &[u8],
    input: &ClaudeInput,
    cost_request: &CostRequest,
    transport: &mut dyn ProviderTransport,
    expected_handle: StreamHandle,
) -> Result<Vec<u8>, AnthropicError> {
    let head = request_head(fixture)?;
    let mut worker_attempts = 0u32;
    let mut rate_limit_attempts = 0u32;
    let handle = loop {
        worker_attempts = worker_attempts
            .checked_add(1)
            .ok_or(AnthropicError::Contract("anthropic retry overflow"))?;
        if worker_attempts > 3 {
            return Err(AnthropicError::Transport(ConformanceError::RetryLimit));
        }
        transport.check_cancelled(None)?;
        match transport.start_request(&head) {
            Ok(handle) => break handle,
            Err(ConformanceError::WorkerLost) if worker_attempts < 3 => continue,
            Err(ConformanceError::RateLimited {
                retry_after_milliseconds,
            }) if retry_after_milliseconds > 0
                && retry_after_milliseconds <= 3_600_000
                && rate_limit_attempts < 16 =>
            {
                rate_limit_attempts = rate_limit_attempts
                    .checked_add(1)
                    .ok_or(AnthropicError::Contract("anthropic retry overflow"))?;
                worker_attempts = worker_attempts.saturating_sub(1);
                continue;
            }
            Err(error) => return Err(error.into()),
        }
    };
    if handle != expected_handle {
        return Err(AnthropicError::Transport(ConformanceError::InvalidHandle));
    }
    let mut download_urls = Vec::new();
    download_urls
        .try_reserve(input.images.len())
        .map_err(|_| AnthropicError::Transport(ConformanceError::BodyLimit))?;
    for image in &input.images {
        transport.check_cancelled(Some(handle))?;
        let response = transport.create_storage(handle, &storage_request(image)?)?;
        if !response.upload_url.starts_with("https://")
            || !response.download_url.starts_with("https://")
            || response.upload_url.len() > 2_048
            || response.download_url.len() > 2_048
        {
            return Err(AnthropicError::Transport(ConformanceError::InvalidResponse));
        }
        transport.check_cancelled(Some(handle))?;
        transport.put_signed(
            handle,
            &SignedPutRequest {
                upload_url: response.upload_url,
                content_type: None,
                bytes: image.bytes.clone(),
            },
        )?;
        download_urls.push(response.download_url);
    }
    let rebuilt_body = build_request_for_input(input, &download_urls)?;
    if rebuilt_body != body {
        return Err(AnthropicError::Contract(
            "anthropic upload URLs do not match the request body",
        ));
    }
    transport.check_cancelled(Some(handle))?;
    transport.write_request_chunk(handle, 0, body, true)?;
    transport.check_cancelled(Some(handle))?;
    let cost = transport.request_cost(handle, cost_request)?;
    validate_cost_response(cost_request, &cost)?;
    transport.check_cancelled(Some(handle))?;
    transport.report_progress(
        handle,
        &Progress {
            sequence: 0,
            completed: 0,
            total: 100,
            message: Some("Waiting for server".into()),
        },
    )?;
    let mut response = Vec::new();
    response
        .try_reserve(4096)
        .map_err(|_| AnthropicError::Transport(ConformanceError::BodyLimit))?;
    let mut after_sequence = None;
    let mut saw_head = false;
    let mut progress_sequence = 1u64;
    for poll in 0..MAXIMUM_POLLS {
        transport.check_cancelled(Some(handle))?;
        match transport.wait_response(handle, after_sequence, WAIT_MILLISECONDS)? {
            ResponseEvent::TimedOut => {
                transport.check_cancelled(Some(handle))?;
                transport.report_progress(
                    handle,
                    &Progress {
                        sequence: progress_sequence,
                        completed: u64::from(poll).saturating_add(1).min(99),
                        total: 100,
                        message: Some("Waiting for server".into()),
                    },
                )?;
                progress_sequence = progress_sequence.saturating_add(1);
            }
            ResponseEvent::Head {
                status: 200,
                headers,
            } if !saw_head => {
                validate_response_headers(&headers)?;
                saw_head = true;
                after_sequence = Some(0);
            }
            ResponseEvent::Cancelled => {
                return Err(AnthropicError::Transport(ConformanceError::Cancelled));
            }
            ResponseEvent::Head { .. } => {
                return Err(AnthropicError::Transport(ConformanceError::InvalidResponse));
            }
            ResponseEvent::Chunk(bytes) if saw_head => {
                if bytes.is_empty() || bytes.len() > MAXIMUM_CHUNK_BYTES {
                    return Err(AnthropicError::Transport(ConformanceError::ChunkLimit));
                }
                let next = response
                    .len()
                    .checked_add(bytes.len())
                    .ok_or(AnthropicError::Transport(ConformanceError::BodyLimit))?;
                if next > MAXIMUM_RESPONSE_BYTES {
                    return Err(AnthropicError::Transport(ConformanceError::BodyLimit));
                }
                response
                    .try_reserve(bytes.len())
                    .map_err(|_| AnthropicError::Transport(ConformanceError::BodyLimit))?;
                response.extend_from_slice(&bytes);
                after_sequence = Some(
                    after_sequence
                        .and_then(|sequence| sequence.checked_add(1))
                        .ok_or(AnthropicError::Transport(ConformanceError::InvalidResponse))?,
                );
            }
            ResponseEvent::Completed(receipt) if saw_head => {
                if receipt.is_empty() || receipt.len() > MAXIMUM_RECEIPT_BYTES {
                    return Err(AnthropicError::Transport(ConformanceError::InvalidResponse));
                }
                transport.check_cancelled(Some(handle))?;
                transport.report_progress(
                    handle,
                    &Progress {
                        sequence: progress_sequence,
                        completed: 100,
                        total: 100,
                        message: Some("Completed".into()),
                    },
                )?;
                return Ok(response);
            }
            _ => {
                return Err(AnthropicError::Transport(ConformanceError::InvalidResponse));
            }
        }
    }
    Err(AnthropicError::Transport(ConformanceError::PollLimit))
}

fn storage_request(image: &ImageInput) -> Result<StorageCreateRequest, AnthropicError> {
    Ok(StorageCreateRequest {
        file_name: image.file_name.clone(),
        content_type: None,
        byte_length: u64::try_from(image.bytes.len())
            .map_err(|_| AnthropicError::Contract("anthropic image size overflow"))?,
    })
}

fn validate_response_headers(headers: &[Header]) -> Result<(), AnthropicError> {
    if headers.len() != 1
        || headers[0].name != "Content-Type"
        || headers[0].value != "application/json"
    {
        return Err(AnthropicError::Transport(ConformanceError::InvalidResponse));
    }
    Ok(())
}

fn response_text(response: &MessagesResponse) -> String {
    let text = response
        .content
        .as_deref()
        .unwrap_or_default()
        .iter()
        .filter_map(|block| match block {
            ResponseBlock::Text { text } if !text.is_empty() => Some(text.as_str()),
            ResponseBlock::Thinking { .. } => None,
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() {
        "Empty response from Claude model.".to_owned()
    } else {
        text
    }
}

fn response_cost_microunits(response: &MessagesResponse) -> Option<u64> {
    let model = response.model.as_deref()?;
    let (input_rate_quarters, output_rate_quarters) =
        if model.contains("opus-4-7") || model.contains("opus-4-6") || model.contains("opus-4-5") {
            (20u64, 100u64)
        } else if model.contains("sonnet-4") {
            (12, 60)
        } else if model.contains("haiku-4-5") {
            (4, 20)
        } else {
            return None;
        };
    let usage = response.usage.as_ref()?;
    let input = usage
        .input_tokens
        .unwrap_or(0)
        .checked_mul(input_rate_quarters)?;
    let output = usage
        .output_tokens
        .unwrap_or(0)
        .checked_mul(output_rate_quarters)?;
    let cache_read = usage
        .cache_read_input_tokens
        .unwrap_or(0)
        .checked_mul(input_rate_quarters)?
        / 10;
    let cache_five_minutes = usage
        .cache_creation
        .as_ref()
        .and_then(|cache| cache.ephemeral_5m_input_tokens)
        .unwrap_or(0)
        .checked_mul(input_rate_quarters)?
        .checked_mul(5)?
        / 4;
    let cache_one_hour = usage
        .cache_creation
        .as_ref()
        .and_then(|cache| cache.ephemeral_1h_input_tokens)
        .unwrap_or(0)
        .checked_mul(input_rate_quarters)?
        .checked_mul(2)?;
    input
        .checked_add(output)?
        .checked_add(cache_read)?
        .checked_add(cache_five_minutes)?
        .checked_add(cache_one_hour)?
        .checked_div(4)
}

fn validate_cost_response(
    request: &CostRequest,
    response: &CostResponse,
) -> Result<(), AnthropicError> {
    let accepted = response.accepted
        && response.approved_microunits != 0
        && response.approved_microunits <= request.maximum_microunits
        && !response.receipt.is_empty()
        && response.receipt.len() <= MAXIMUM_RECEIPT_BYTES;
    let denied =
        !response.accepted && response.approved_microunits == 0 && response.receipt.is_empty();
    if !accepted || denied {
        return Err(AnthropicError::Contract(
            "invalid anthropic cost authorization",
        ));
    }
    Ok(())
}

fn publish_text(text: &str, sink: &mut dyn MaterializationSink) -> Result<(), AnthropicError> {
    let registry = TypeRegistry::built_in()
        .map_err(|_| AnthropicError::Contract("built-in type registry is unavailable"))?;
    let type_id = CanonicalTypeId::new("sim", "string", 1)
        .map_err(|_| AnthropicError::Contract("built-in string type is unavailable"))?;
    let value = PluginValue::scalar(
        type_id.clone(),
        ScalarValue::String(text.to_owned()),
        &registry,
    )
    .map_err(|_| AnthropicError::Contract("invalid anthropic string output"))?;
    let batch = validate_outputs(
        &[ExpectedOutput {
            port_id: "output_0".into(),
            type_id: type_id.clone(),
            family: ValueFamily::Scalar,
        }],
        vec![EncodedOutput {
            port_id: "output_0".into(),
            type_id,
            family: ValueFamily::Scalar,
            bytes: value
                .abi_bytes()
                .map_err(|_| AnthropicError::Contract("failed to encode anthropic output"))?,
        }],
        MAXIMUM_RESPONSE_BYTES,
    )?;
    publish_validated(batch, sink)?;
    Ok(())
}

fn run_error_cases(fixture: &AnthropicFixture) -> Result<(), AnthropicError> {
    let body = request_body(fixture)?;
    let input = fixture_input(fixture)?;
    let head = request_head(fixture)?;
    let handle = StreamHandle {
        invocation: 44,
        slot: 1,
        generation: 4,
    };
    let cost_request = CostRequest {
        operation: "anthropic.messages".into(),
        currency: "USD".into(),
        maximum_microunits: fixture.unsigned("expected_cost_microunits")?,
    };

    let mut api_key_fixture = fixture.clone();
    api_key_fixture
        .values
        .insert("auth_header".into(), "X-API-KEY".into());
    api_key_fixture.values.insert(
        "auth_value".into(),
        "host-capability:api_key_sim_org".into(),
    );
    let api_key_head = request_head(&api_key_fixture)?;
    if api_key_head.headers[1].name != "X-API-KEY"
        || api_key_head.headers[1].value != "host-capability:api_key_sim_org"
        || api_key_head.secret_id.as_deref() != Some("sim_org.api_key")
    {
        return Err(AnthropicError::Contract(
            "anthropic API key fallback changed",
        ));
    }
    let mut invalid_auth = fixture.clone();
    invalid_auth
        .values
        .insert("auth_value".into(), "sk-ant-forbidden".into());
    if request_head(&invalid_auth).is_ok() {
        return Err(AnthropicError::Contract(
            "raw anthropic credential was accepted",
        ));
    }

    let mut cancelled = ScriptedActuator::new(vec![
        ScriptStep::CheckCancelled(None),
        ScriptStep::Start(head.clone(), Ok(handle)),
        ScriptStep::Cancel(Some(handle)),
    ]);
    let error = match invoke_transport(
        fixture,
        &body,
        &input,
        &cost_request,
        &mut cancelled,
        handle,
    ) {
        Err(error) => error,
        Ok(_) => {
            return Err(AnthropicError::Contract(
                "cancelled anthropic invocation succeeded",
            ));
        }
    };
    if !matches!(
        error,
        AnthropicError::Transport(ConformanceError::Cancelled)
    ) {
        return Err(AnthropicError::Contract(
            "anthropic cancellation was not typed",
        ));
    }

    let mut rate_limited = ScriptedActuator::new(vec![
        ScriptStep::CheckCancelled(None),
        ScriptStep::Start(
            head.clone(),
            Err(ConformanceError::RateLimited {
                retry_after_milliseconds: 2_000,
            }),
        ),
        ScriptStep::CheckCancelled(None),
        ScriptStep::Start(head.clone(), Ok(handle)),
        ScriptStep::Cancel(Some(handle)),
    ]);
    if !matches!(
        invoke_transport(
            fixture,
            &body,
            &input,
            &cost_request,
            &mut rate_limited,
            handle,
        ),
        Err(AnthropicError::Transport(ConformanceError::Cancelled))
    ) {
        return Err(AnthropicError::Contract(
            "anthropic Retry-After did not create a bounded fresh attempt",
        ));
    }

    let mut worker_loss = ScriptedActuator::new(vec![
        ScriptStep::CheckCancelled(None),
        ScriptStep::Start(head.clone(), Err(ConformanceError::WorkerLost)),
        ScriptStep::CheckCancelled(None),
        ScriptStep::Start(head.clone(), Err(ConformanceError::WorkerLost)),
        ScriptStep::CheckCancelled(None),
        ScriptStep::Start(head.clone(), Err(ConformanceError::WorkerLost)),
    ]);
    if !matches!(
        invoke_transport(
            fixture,
            &body,
            &input,
            &cost_request,
            &mut worker_loss,
            handle
        ),
        Err(AnthropicError::Transport(ConformanceError::WorkerLost))
    ) {
        return Err(AnthropicError::Contract(
            "anthropic worker-loss retry bound changed",
        ));
    }

    let stale_handle = StreamHandle {
        generation: handle.generation - 1,
        ..handle
    };
    let mut stale = ScriptedActuator::new(vec![
        ScriptStep::CheckCancelled(None),
        ScriptStep::Start(head.clone(), Ok(stale_handle)),
    ]);
    if !matches!(
        invoke_transport(fixture, &body, &input, &cost_request, &mut stale, handle),
        Err(AnthropicError::Transport(ConformanceError::InvalidHandle))
    ) {
        return Err(AnthropicError::Contract(
            "stale anthropic restart handle was accepted",
        ));
    }

    let mut http_error = success_prefix(fixture, &head, handle, &input, &body, &cost_request)?;
    http_error.extend([
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Wait {
            handle,
            after_sequence: None,
            timeout_milliseconds: WAIT_MILLISECONDS,
            outcome: ResponseEvent::Head {
                status: 401,
                headers: vec![Header {
                    name: "Content-Type".into(),
                    value: "application/json".into(),
                }],
            },
        },
    ]);
    let mut http_error = ScriptedActuator::new(http_error);
    if !matches!(
        invoke_transport(
            fixture,
            &body,
            &input,
            &cost_request,
            &mut http_error,
            handle
        ),
        Err(AnthropicError::Transport(ConformanceError::InvalidResponse))
    ) {
        return Err(AnthropicError::Contract(
            "anthropic HTTP error was accepted",
        ));
    }

    let mut poll_limit = success_prefix(fixture, &head, handle, &input, &body, &cost_request)?;
    for poll in 0..MAXIMUM_POLLS {
        poll_limit.push(ScriptStep::CheckCancelled(Some(handle)));
        poll_limit.push(ScriptStep::Wait {
            handle,
            after_sequence: None,
            timeout_milliseconds: WAIT_MILLISECONDS,
            outcome: ResponseEvent::TimedOut,
        });
        poll_limit.push(ScriptStep::CheckCancelled(Some(handle)));
        poll_limit.push(ScriptStep::Progress(
            handle,
            Progress {
                sequence: u64::from(poll).saturating_add(1),
                completed: u64::from(poll).saturating_add(1).min(99),
                total: 100,
                message: Some("Waiting for server".into()),
            },
        ));
    }
    let mut poll_limit = ScriptedActuator::new(poll_limit);
    if !matches!(
        invoke_transport(
            fixture,
            &body,
            &input,
            &cost_request,
            &mut poll_limit,
            handle
        ),
        Err(AnthropicError::Transport(ConformanceError::PollLimit))
    ) {
        return Err(AnthropicError::Contract("anthropic polling bound changed"));
    }

    let mut malformed = success_prefix(fixture, &head, handle, &input, &body, &cost_request)?;
    malformed.extend([
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Wait {
            handle,
            after_sequence: None,
            timeout_milliseconds: WAIT_MILLISECONDS,
            outcome: ResponseEvent::Head {
                status: 200,
                headers: vec![Header {
                    name: "Content-Type".into(),
                    value: "application/json".into(),
                }],
            },
        },
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Wait {
            handle,
            after_sequence: Some(0),
            timeout_milliseconds: WAIT_MILLISECONDS,
            outcome: ResponseEvent::Chunk(b"{malformed".to_vec()),
        },
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Wait {
            handle,
            after_sequence: Some(1),
            timeout_milliseconds: WAIT_MILLISECONDS,
            outcome: ResponseEvent::Completed(vec![1]),
        },
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Progress(
            handle,
            Progress {
                sequence: 1,
                completed: 100,
                total: 100,
                message: Some("Completed".into()),
            },
        ),
    ]);
    let mut malformed = ScriptedActuator::new(malformed);
    let response = invoke_transport(
        fixture,
        &body,
        &input,
        &cost_request,
        &mut malformed,
        handle,
    )?;
    if serde_json::from_slice::<MessagesResponse>(&response).is_ok() {
        return Err(AnthropicError::Contract(
            "malformed anthropic response was accepted",
        ));
    }

    let oversized_chunk = vec![0; MAXIMUM_CHUNK_BYTES + 1];
    let mut oversized = success_prefix(fixture, &head, handle, &input, &body, &cost_request)?;
    oversized.extend([
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Wait {
            handle,
            after_sequence: None,
            timeout_milliseconds: WAIT_MILLISECONDS,
            outcome: ResponseEvent::Head {
                status: 200,
                headers: vec![Header {
                    name: "Content-Type".into(),
                    value: "application/json".into(),
                }],
            },
        },
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Wait {
            handle,
            after_sequence: Some(0),
            timeout_milliseconds: WAIT_MILLISECONDS,
            outcome: ResponseEvent::Chunk(oversized_chunk),
        },
    ]);
    let mut oversized = ScriptedActuator::new(oversized);
    if !matches!(
        invoke_transport(
            fixture,
            &body,
            &input,
            &cost_request,
            &mut oversized,
            handle
        ),
        Err(AnthropicError::Transport(ConformanceError::ChunkLimit))
    ) {
        return Err(AnthropicError::Contract(
            "oversized anthropic response was not rejected",
        ));
    }

    for invalid_cost in [
        CostResponse {
            accepted: false,
            approved_microunits: 0,
            receipt: Vec::new(),
        },
        CostResponse {
            accepted: true,
            approved_microunits: cost_request.maximum_microunits + 1,
            receipt: vec![1],
        },
        CostResponse {
            accepted: true,
            approved_microunits: 1,
            receipt: vec![1; MAXIMUM_RECEIPT_BYTES + 1],
        },
    ] {
        if validate_cost_response(&cost_request, &invalid_cost).is_ok() {
            return Err(AnthropicError::Contract(
                "invalid anthropic cost response was accepted",
            ));
        }
    }

    let sink = RecordingSink::default();
    let string_type = CanonicalTypeId::new("sim", "string", 1)
        .map_err(|_| AnthropicError::Contract("built-in string type is unavailable"))?;
    let invalid = validate_outputs(
        &[ExpectedOutput {
            port_id: "output_0".into(),
            type_id: string_type.clone(),
            family: ValueFamily::Scalar,
        }],
        vec![EncodedOutput {
            port_id: "output_0".into(),
            type_id: string_type,
            family: ValueFamily::Scalar,
            bytes: b"fake result".to_vec(),
        }],
        MAXIMUM_RESPONSE_BYTES,
    );
    if !matches!(invalid, Err(MaterializationError::MalformedValue)) || !sink.batches.is_empty() {
        return Err(AnthropicError::Contract(
            "anthropic materialization failure was not atomic",
        ));
    }
    Ok(())
}

fn success_prefix(
    fixture: &AnthropicFixture,
    head: &RequestHead,
    handle: StreamHandle,
    input: &ClaudeInput,
    body: &[u8],
    cost: &CostRequest,
) -> Result<Vec<ScriptStep>, AnthropicError> {
    let image = input
        .images
        .first()
        .ok_or(AnthropicError::Contract("missing anthropic fixture image"))?;
    Ok(vec![
        ScriptStep::CheckCancelled(None),
        ScriptStep::Start(head.clone(), Ok(handle)),
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::CreateStorage(
            handle,
            storage_request(image)?,
            StorageCreateResponse {
                upload_url: "https://fixture.invalid/signed/reference.png".into(),
                download_url: fixture.value("image_url")?.into(),
            },
        ),
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::PutSigned(
            handle,
            SignedPutRequest {
                upload_url: "https://fixture.invalid/signed/reference.png".into(),
                content_type: None,
                bytes: image.bytes.clone(),
            },
        ),
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::RequestChunk {
            handle,
            sequence: 0,
            bytes: body.to_vec(),
            end: true,
        },
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Cost(
            handle,
            cost.clone(),
            CostResponse {
                accepted: true,
                approved_microunits: cost.maximum_microunits,
                receipt: b"anthropic-cost".to_vec(),
            },
        ),
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Progress(
            handle,
            Progress {
                sequence: 0,
                completed: 0,
                total: 100,
                message: Some("Waiting for server".into()),
            },
        ),
    ])
}

fn validate_unsigned_artifacts() -> Result<(), AnthropicError> {
    let component = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/anthropic/provider_v2_component"
    ));
    let signing = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/anthropic/signing_payload.fixture"
    ));
    if component.len() != 161_751
        || !component.starts_with(b"\0asm")
        || component.get(4..8) != Some(&[0x0d, 0x00, 0x01, 0x00])
        || !signing.contains(NAMESPACE)
        || !signing.contains(NODE_FEATURE_ID)
        || !signing.contains(ROUTE_FEATURE_ID)
        || !signing.contains(SOURCE_SHA256)
        || !signing.contains(
            "component_sha256=f4b54f14702753f93b9a09495afbd4d1ceb5e1ec216a066207e668908d9aad50",
        )
    {
        return Err(AnthropicError::Contract(
            "anthropic unsigned component or signing payload changed",
        ));
    }
    let lowercase = signing.to_ascii_lowercase();
    for forbidden in [
        "sk-ant",
        "private_key",
        "begin private",
        "bearer ",
        "fake result",
    ] {
        if lowercase.contains(forbidden) {
            return Err(AnthropicError::Contract(
                "anthropic unsigned artifacts contain forbidden credential or result data",
            ));
        }
    }
    Ok(())
}

fn decode_hex(value: &str) -> Result<Vec<u8>, AnthropicError> {
    if value.is_empty() || !value.len().is_multiple_of(2) || value.len() > MAXIMUM_REQUEST_BYTES * 2
    {
        return Err(AnthropicError::Contract("invalid anthropic upload hex"));
    }
    let mut output = Vec::new();
    output
        .try_reserve(value.len() / 2)
        .map_err(|_| AnthropicError::Contract("anthropic upload allocation failed"))?;
    for pair in value.as_bytes().chunks_exact(2) {
        let high = hex_nibble(pair[0])?;
        let low = hex_nibble(pair[1])?;
        output.push((high << 4) | low);
    }
    Ok(output)
}

fn hex_nibble(byte: u8) -> Result<u8, AnthropicError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(AnthropicError::Contract("invalid anthropic upload hex")),
    }
}

fn map_error(error: AnthropicError) -> ConformanceError {
    match error {
        AnthropicError::Transport(error) => error,
        AnthropicError::Materialization(error) => ConformanceError::Materialization(error),
        AnthropicError::Contract(_) | AnthropicError::Json(_) => ConformanceError::InvalidResponse,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/anthropic/conformance.fixture"
    ));

    #[test]
    fn anthropic_upload_preserves_png_identity_and_source_mime_omission()
    -> Result<(), Box<dyn Error>> {
        let source: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../sim_test_support/fixtures/providers/common/source_storage_upload.json"
        ))?;
        assert_eq!(source["anthropic_upload"]["source_sha256"], SOURCE_SHA256);
        assert!(
            source["anthropic_upload"]["options"]
                .get("mime_type")
                .is_none()
        );
        let fixture = AnthropicFixture::parse(FIXTURE)?;
        let input = fixture_input(&fixture)?;
        let image = input.images.first().ok_or("fixture image missing")?;
        assert_eq!(image.content_type, "image/png");
        let request = storage_request(image)?;
        assert_eq!(request.content_type, None);
        let body: serde_json::Value = serde_json::from_slice(
            &request.checked_body(crate::common::TransportLimits::conformance())?,
        )?;
        assert!(body.get("content_type").is_none());
        Ok(())
    }

    #[test]
    fn anthropic_claims_and_source_fingerprint_are_exact() -> Result<(), Box<dyn Error>> {
        validate_claims(&AnthropicFixture::parse(FIXTURE)?)?;
        assert_eq!(
            resolve_auth_header(
                Some("host-capability:auth_token_sim_org"),
                Some("host-capability:api_key_sim_org"),
            )?
            .name,
            "Authorization"
        );
        Ok(())
    }

    #[test]
    fn anthropic_success_retry_stream_download_cost_and_restart_are_exact()
    -> Result<(), Box<dyn Error>> {
        let fixture = AnthropicFixture::parse(FIXTURE)?;
        run_success(&fixture, 2)?;
        run_success(&fixture, 3)?;
        Ok(())
    }

    #[test]
    fn anthropic_failures_are_typed_and_materialization_is_atomic() -> Result<(), Box<dyn Error>> {
        run_error_cases(&AnthropicFixture::parse(FIXTURE)?)?;
        Ok(())
    }

    #[test]
    fn anthropic_response_cancellation_preserves_terminal_reason() -> Result<(), Box<dyn Error>> {
        let fixture = AnthropicFixture::parse(FIXTURE)?;
        let body = request_body(&fixture)?;
        let input = fixture_input(&fixture)?;
        let head = request_head(&fixture)?;
        let handle = StreamHandle {
            invocation: 45,
            slot: 1,
            generation: 1,
        };
        let cost = CostRequest {
            operation: "anthropic.messages".into(),
            currency: "USD".into(),
            maximum_microunits: fixture.unsigned("expected_cost_microunits")?,
        };

        for received_head in [false, true] {
            let mut script = success_prefix(&fixture, &head, handle, &input, &body, &cost)?;
            if received_head {
                script.extend([
                    ScriptStep::CheckCancelled(Some(handle)),
                    ScriptStep::Wait {
                        handle,
                        after_sequence: None,
                        timeout_milliseconds: WAIT_MILLISECONDS,
                        outcome: ResponseEvent::Head {
                            status: 200,
                            headers: vec![Header {
                                name: "Content-Type".into(),
                                value: "application/json".into(),
                            }],
                        },
                    },
                ]);
            }
            script.extend([
                ScriptStep::CheckCancelled(Some(handle)),
                ScriptStep::Wait {
                    handle,
                    after_sequence: received_head.then_some(0),
                    timeout_milliseconds: WAIT_MILLISECONDS,
                    outcome: ResponseEvent::Cancelled,
                },
            ]);
            let mut transport = ScriptedActuator::new(script);
            assert!(matches!(
                invoke_transport(&fixture, &body, &input, &cost, &mut transport, handle),
                Err(AnthropicError::Transport(ConformanceError::Cancelled))
            ));
            transport.finish()?;
        }
        Ok(())
    }

    #[test]
    fn anthropic_unsigned_artifacts_are_credential_free() -> Result<(), Box<dyn Error>> {
        validate_unsigned_artifacts()?;
        Ok(())
    }

    #[test]
    fn anthropic_model_reasoning_and_temperature_matrix_matches_source()
    -> Result<(), Box<dyn Error>> {
        let base = AnthropicFixture::parse(FIXTURE)?;
        for (model, effort, max_tokens, expected_fields, absent_fields) in [
            (
                "Opus 4.7",
                "high",
                "32768",
                &[
                    "\"model\":\"claude-opus-4-7\"",
                    "\"thinking\":{\"type\":\"adaptive\"}",
                    "\"output_config\":{\"effort\":\"high\"}",
                ][..],
                &["\"temperature\"", "\"budget_tokens\""][..],
            ),
            (
                "Opus 4.6",
                "off",
                "32768",
                &["\"model\":\"claude-opus-4-6\"", "\"temperature\":1.0"][..],
                &["\"thinking\"", "\"output_config\""][..],
            ),
            (
                "Sonnet 4.5",
                "high",
                "4096",
                &[
                    "\"model\":\"claude-sonnet-4-5-20250929\"",
                    "\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":3072}",
                ][..],
                &["\"temperature\"", "\"output_config\""][..],
            ),
            (
                "Haiku 4.5",
                "high",
                "32768",
                &[
                    "\"model\":\"claude-haiku-4-5-20251001\"",
                    "\"temperature\":1.0",
                ][..],
                &["\"thinking\"", "\"output_config\""][..],
            ),
        ] {
            let mut fixture = base.clone();
            fixture.values.insert("model_label".into(), model.into());
            fixture
                .values
                .insert("reasoning_effort".into(), effort.into());
            fixture
                .values
                .insert("max_tokens".into(), max_tokens.into());
            let body = String::from_utf8(build_request_body(&fixture)?)?;
            for expected in expected_fields {
                assert!(body.contains(expected), "missing {expected} in {body}");
            }
            for absent in absent_fields {
                assert!(!body.contains(absent), "unexpected {absent} in {body}");
            }
        }
        Ok(())
    }

    #[test]
    fn anthropic_node_inputs_cover_seed_temperature_and_zero_to_twenty_images()
    -> Result<(), Box<dyn Error>> {
        let fixture = AnthropicFixture::parse(FIXTURE)?;
        let mut input = fixture_input(&fixture)?;
        input.reasoning_effort = ReasoningEffort::Off;
        input.temperature = 0.25;
        input.seed = 2_147_483_647;
        input.images.clear();
        let without_images = String::from_utf8(build_request_for_input(&input, &[])?)?;
        assert!(without_images.contains("\"temperature\":0.25"));
        assert!(!without_images.contains("image"));
        assert!(!without_images.contains("seed"));

        let image = ImageInput {
            file_name: "pixel.png".into(),
            content_type: "image/png".into(),
            width: 1,
            height: 1,
            bytes: vec![1],
        };
        input.images = vec![image.clone(); MAXIMUM_IMAGES];
        let urls = (0..MAXIMUM_IMAGES)
            .map(|index| format!("https://fixture.invalid/download/{index}.png"))
            .collect::<Vec<_>>();
        let with_twenty = String::from_utf8(build_request_for_input(&input, &urls)?)?;
        assert_eq!(with_twenty.matches("\"type\":\"image\"").count(), 20);
        assert!(
            with_twenty
                .find("download/0.png")
                .is_some_and(|image_index| {
                    with_twenty
                        .find("\"type\":\"text\"")
                        .is_some_and(|text_index| image_index < text_index)
                })
        );

        input.images.push(image.clone());
        assert!(build_request_for_input(&input, &vec![urls[0].clone(); 21]).is_err());
        input.images = vec![ImageInput {
            width: 1_569,
            height: 1_569,
            ..image
        }];
        assert!(build_request_for_input(&input, &[urls[0].clone()]).is_ok());
        Ok(())
    }

    #[test]
    fn anthropic_zero_and_twenty_image_transport_runs_exact_storage_and_signed_put_flow()
    -> Result<(), Box<dyn Error>> {
        let fixture = AnthropicFixture::parse(FIXTURE)?;
        for image_count in [0usize, MAXIMUM_IMAGES] {
            let mut input = fixture_input(&fixture)?;
            input.images = (0..image_count)
                .map(|index| ImageInput {
                    file_name: format!("image-{index}.png"),
                    content_type: "image/png".into(),
                    width: 1,
                    height: 1,
                    bytes: vec![u8::try_from(index).unwrap_or(0).saturating_add(1)],
                })
                .collect();
            let urls = (0..image_count)
                .map(|index| format!("https://fixture.invalid/download/image-{index}.png"))
                .collect::<Vec<_>>();
            let body = build_request_for_input(&input, &urls)?;
            let handle = StreamHandle {
                invocation: 71,
                slot: 1,
                generation: u32::try_from(image_count).unwrap_or(0).saturating_add(1),
            };
            let head = request_head(&fixture)?;
            let cost = CostRequest {
                operation: "anthropic.messages".into(),
                currency: "USD".into(),
                maximum_microunits: fixture.unsigned("expected_cost_microunits")?,
            };
            let mut script = vec![
                ScriptStep::CheckCancelled(None),
                ScriptStep::Start(head, Ok(handle)),
            ];
            for (index, image) in input.images.iter().enumerate() {
                let upload_url = format!("https://fixture.invalid/signed/image-{index}.png");
                script.extend([
                    ScriptStep::CheckCancelled(Some(handle)),
                    ScriptStep::CreateStorage(
                        handle,
                        storage_request(image)?,
                        StorageCreateResponse {
                            upload_url: upload_url.clone(),
                            download_url: urls[index].clone(),
                        },
                    ),
                    ScriptStep::CheckCancelled(Some(handle)),
                    ScriptStep::PutSigned(
                        handle,
                        SignedPutRequest {
                            upload_url,
                            content_type: None,
                            bytes: image.bytes.clone(),
                        },
                    ),
                ]);
            }
            script.extend([
                ScriptStep::CheckCancelled(Some(handle)),
                ScriptStep::RequestChunk {
                    handle,
                    sequence: 0,
                    bytes: body.clone(),
                    end: true,
                },
                ScriptStep::CheckCancelled(Some(handle)),
                ScriptStep::Cost(
                    handle,
                    cost.clone(),
                    CostResponse {
                        accepted: true,
                        approved_microunits: cost.maximum_microunits,
                        receipt: vec![1],
                    },
                ),
                ScriptStep::CheckCancelled(Some(handle)),
                ScriptStep::Progress(
                    handle,
                    Progress {
                        sequence: 0,
                        completed: 0,
                        total: 100,
                        message: Some("Waiting for server".into()),
                    },
                ),
                ScriptStep::CheckCancelled(Some(handle)),
                ScriptStep::Wait {
                    handle,
                    after_sequence: None,
                    timeout_milliseconds: WAIT_MILLISECONDS,
                    outcome: ResponseEvent::Head {
                        status: 200,
                        headers: vec![Header {
                            name: "Content-Type".into(),
                            value: "application/json".into(),
                        }],
                    },
                },
                ScriptStep::CheckCancelled(Some(handle)),
                ScriptStep::Wait {
                    handle,
                    after_sequence: Some(0),
                    timeout_milliseconds: WAIT_MILLISECONDS,
                    outcome: ResponseEvent::Chunk(
                        fixture.value("response_body")?.as_bytes().to_vec(),
                    ),
                },
                ScriptStep::CheckCancelled(Some(handle)),
                ScriptStep::Wait {
                    handle,
                    after_sequence: Some(1),
                    timeout_milliseconds: WAIT_MILLISECONDS,
                    outcome: ResponseEvent::Completed(vec![1]),
                },
                ScriptStep::CheckCancelled(Some(handle)),
                ScriptStep::Progress(
                    handle,
                    Progress {
                        sequence: 1,
                        completed: 100,
                        total: 100,
                        message: Some("Completed".into()),
                    },
                ),
            ]);
            let mut actuator = ScriptedActuator::new(script);
            let response = invoke_transport(&fixture, &body, &input, &cost, &mut actuator, handle)?;
            assert_eq!(response, fixture.value("response_body")?.as_bytes());
            let observed = actuator.finish()?;
            assert_eq!(
                observed
                    .iter()
                    .filter(|step| matches!(step, ScriptStep::CreateStorage(..)))
                    .count(),
                image_count
            );
            assert_eq!(
                observed
                    .iter()
                    .filter(|step| matches!(step, ScriptStep::PutSigned(..)))
                    .count(),
                image_count
            );
        }
        Ok(())
    }

    #[test]
    fn anthropic_source_control_only_prompt_fails_before_request_construction()
    -> Result<(), Box<dyn Error>> {
        let fixture = AnthropicFixture::parse(FIXTURE)?;
        for prompt in [
            "\u{1c}",
            "\u{1d}",
            "\u{1e}",
            "\u{1f}",
            " \t\n\u{1c}\u{1d}\u{1e}\u{1f}\u{a0}",
        ] {
            let mut input = fixture_input(&fixture)?;
            input.prompt = prompt.to_owned();
            // A malformed URL must not hide the source's earlier prompt rejection.
            assert!(matches!(
                build_request_for_input(&input, &["http://invalid".into()]),
                Err(AnthropicError::Contract("invalid anthropic node input"))
            ));
        }
        Ok(())
    }

    #[test]
    fn anthropic_source_admitted_prompt_retains_raw_bytes_and_nonstrip_controls()
    -> Result<(), Box<dyn Error>> {
        let fixture = AnthropicFixture::parse(FIXTURE)?;
        for prompt in [
            "\u{1c} text \u{1f}",
            "\u{200b}",
            "\u{feff}",
            "\u{1c}\u{200b}\u{1f}",
        ] {
            let mut input = fixture_input(&fixture)?;
            input.images.clear();
            input.prompt = prompt.to_owned();
            let body = build_request_for_input(&input, &[])?;
            let parsed: serde_json::Value = serde_json::from_slice(&body)?;
            assert_eq!(parsed["messages"][0]["content"][0]["text"], prompt);
        }
        Ok(())
    }

    #[test]
    fn anthropic_source_padded_prompt_reaches_scripted_transport_and_cancellation()
    -> Result<(), Box<dyn Error>> {
        let mut fixture = AnthropicFixture::parse(FIXTURE)?;
        let original_prompt = serde_json::to_string(fixture.value("prompt")?)?;
        let prompt = "\u{1c}bounded source prompt\u{1f}";
        let replacement = serde_json::to_string(prompt)?;
        let original_body = fixture.value("request_body")?.to_owned();
        assert_eq!(original_body.matches(&original_prompt).count(), 1);
        fixture.values.insert("prompt".into(), prompt.into());
        fixture.values.insert(
            "request_body".into(),
            original_body.replacen(&original_prompt, &replacement, 1),
        );
        let input = fixture_input(&fixture)?;
        let body = request_body(&fixture)?;
        let handle = StreamHandle {
            invocation: 41,
            slot: 1,
            generation: 1,
        };
        let cost = CostRequest {
            operation: "anthropic.messages".into(),
            currency: "USD".into(),
            maximum_microunits: fixture.unsigned("expected_cost_microunits")?,
        };
        let mut cancelled = ScriptedActuator::new(vec![ScriptStep::Cancel(None)]);
        assert!(matches!(
            invoke_transport(&fixture, &body, &input, &cost, &mut cancelled, handle),
            Err(AnthropicError::Transport(ConformanceError::Cancelled))
        ));
        cancelled.finish()?;
        run_success(&fixture, 2)?;
        run_success(&fixture, 3)?;
        Ok(())
    }
}
