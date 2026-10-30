use crate::common::{
    ConformanceError, CostRequest, EncodedOutput, ExpectedOutput, Header, HttpMethod,
    MaterializationSink, Progress, ProviderTransport, RequestHead, ResponseEvent, StreamHandle,
    TypeRegistry, publish_validated, validate_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    error::Error,
    fmt,
};

pub const NAMESPACE: &str = "zed.sim.provider.sonilo";
pub const ALIASES: &[&str] = &[];
pub const SOURCE_SHA256: &str = "0fedcb4659918bc385f511e1b04b202d1f50f578b95f7e5df93d2700836be727";
pub const MAXIMUM_BODY_BYTES: usize = 16 * 1024 * 1024;
pub const MAXIMUM_CHUNK_BYTES: usize = 64 * 1024;
const MAXIMUM_RECEIPT_BYTES: usize = 32 * 1024;
const MAXIMUM_STREAMS: usize = 128;
const MAXIMUM_EVENTS: u64 = 100_000;
#[derive(Clone, Copy, Debug, Serialize)]
pub struct NodeClaim {
    pub feature_id: &'static str,
    pub node_identifier: &'static str,
}
pub const NODES: &[NodeClaim] = &[
    NodeClaim {
        feature_id: "COMFY-NODE-0626",
        node_identifier: "SoniloTextToMusic",
    },
    NodeClaim {
        feature_id: "COMFY-NODE-0627",
        node_identifier: "SoniloVideoToMusic",
    },
];
#[derive(Clone, Copy, Debug, Serialize)]
pub struct RouteClaim {
    pub feature_id: &'static str,
    pub method: &'static str,
    pub path: &'static str,
    pub disposition: &'static str,
}
pub const ROUTES: &[RouteClaim] = &[
    RouteClaim {
        feature_id: "COMFY-API-EXT-0162",
        method: "POST",
        path: "/proxy/sonilo/t2m/generate",
        disposition: "executable",
    },
    RouteClaim {
        feature_id: "COMFY-API-EXT-0163",
        method: "POST",
        path: "/proxy/sonilo/v2m/generate",
        disposition: "executable",
    },
];
#[derive(Debug)]
pub enum SoniloError {
    Input(&'static str),
    Response(&'static str),
    Provider(String),
    Transport(ConformanceError),
    Json(serde_json::Error),
}
impl fmt::Display for SoniloError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(message) | Self::Response(message) => formatter.write_str(message),
            Self::Provider(message) => formatter.write_str(message),
            Self::Transport(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
        }
    }
}
impl Error for SoniloError {}
impl From<ConformanceError> for SoniloError {
    fn from(error: ConformanceError) -> Self {
        Self::Transport(error)
    }
}
impl From<serde_json::Error> for SoniloError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}
fn oracle() -> Result<Value, SoniloError> {
    Ok(serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../sim_test_support/fixtures/providers/sonilo/source-contract.json"
    )))?)
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub node: String,
    pub arguments: BTreeMap<String, Value>,
}
fn normalized(input: &Input) -> Result<Input, SoniloError> {
    if !NODES.iter().any(|node| node.node_identifier == input.node) {
        return Err(SoniloError::Input("Unknown Sonilo node"));
    }
    let source = oracle()?;
    let contract = &source["contracts"][&input.node];
    let names = contract["arguments"]
        .as_array()
        .ok_or(SoniloError::Input("Missing source arguments"))?;
    if input
        .arguments
        .keys()
        .any(|name| !names.iter().any(|value| value.as_str() == Some(name)))
    {
        return Err(SoniloError::Input("Unknown Sonilo argument"));
    }
    let mut input = input.clone();
    for name in names {
        let name = name
            .as_str()
            .ok_or(SoniloError::Input("Invalid source argument"))?;
        if !input.arguments.contains_key(name) {
            input.arguments.insert(
                name.into(),
                contract["defaults"]
                    .get(name)
                    .ok_or(SoniloError::Input("Missing Sonilo argument"))?
                    .clone(),
            );
        }
    }
    Ok(input)
}
fn argument<'a>(input: &'a Input, name: &str) -> Result<&'a Value, SoniloError> {
    input
        .arguments
        .get(name)
        .ok_or(SoniloError::Input("Missing Sonilo argument"))
}
fn text(value: &Value) -> Result<&str, SoniloError> {
    value
        .as_str()
        .ok_or(SoniloError::Input("Expected source string"))
}
fn source_strip(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || matches!(character, '\u{001c}'..='\u{001f}')
    })
}
fn text_preflight(input: &Input) -> Result<(), SoniloError> {
    if input.node == "SoniloTextToMusic" {
        let length = source_strip(text(argument(input, "prompt")?)?)
            .chars()
            .count();
        if !(1..=1000).contains(&length) {
            return Err(SoniloError::Input(
                "Source prompt outside 1–1000 characters",
            ));
        }
    }
    Ok(())
}
fn receipt(bytes: &[u8]) -> Result<(), ConformanceError> {
    if bytes.is_empty() || bytes.len() > MAXIMUM_RECEIPT_BYTES {
        Err(ConformanceError::InvalidResponse)
    } else {
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub struct CredentialAvailability {
    pub auth_token: bool,
    pub api_key: bool,
}
fn head(
    path: &str,
    credentials: CredentialAvailability,
    mime: &str,
) -> Result<RequestHead, SoniloError> {
    let (secret, name, value) = if credentials.auth_token {
        (
            "sim_org.auth_token",
            "Authorization",
            "host-capability:auth_token_sim_org",
        )
    } else if credentials.api_key {
        (
            "sim_org.api_key",
            "X-API-KEY",
            "host-capability:api_key_sim_org",
        )
    } else {
        return Err(SoniloError::Input("Missing Comfy API credentials"));
    };
    Ok(RequestHead {
        method: HttpMethod::Post,
        path: path.into(),
        secret_id: Some(secret.into()),
        headers: vec![
            Header {
                name: name.into(),
                value: value.into(),
            },
            Header {
                name: "Comfy-Env".into(),
                value: "host".into(),
            },
            Header {
                name: "Comfy-Usage-Source".into(),
                value: "comfyui-api".into(),
            },
            Header {
                name: "Content-Type".into(),
                value: mime.into(),
            },
        ],
    })
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FormField {
    pub name: String,
    pub value: String,
}
#[derive(Clone, Debug)]
pub struct PreparedForm {
    pub bytes: Vec<u8>,
    pub content_type: String,
    pub receipt: Vec<u8>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarStyle {
    PythonString,
    FixedOneDecimal,
    TruncatedInteger,
}
impl ScalarStyle {
    fn name(self) -> &'static str {
        match self {
            Self::PythonString => "PythonString",
            Self::FixedOneDecimal => "FixedOneDecimal",
            Self::TruncatedInteger => "TruncatedInteger",
        }
    }
}
pub trait SoniloPreparation {
    fn encode_form(&mut self, _fields: &[FormField]) -> Result<PreparedForm, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn scalar(&mut self, _value: &Value, _style: ScalarStyle) -> Result<String, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn decode_base64(&mut self, _value: &Value) -> Result<Vec<u8>, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn http_error(
        &mut self,
        _bytes: &[u8],
        _headers: &[Header],
    ) -> Result<String, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VideoPolicy {
    pub maximum_duration_seconds: u64,
    pub container: &'static str,
    pub codec: &'static str,
    pub wait_label: &'static str,
}
#[derive(Clone, Debug)]
pub struct UploadedVideo {
    pub url: String,
    pub receipt: Vec<u8>,
}
pub trait SoniloVideoOwner {
    fn upload_video(
        &mut self,
        _video: &Value,
        _policy: &VideoPolicy,
    ) -> Result<UploadedVideo, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
#[derive(Clone, Debug)]
pub struct Request {
    pub path: &'static str,
    pub fields: Vec<FormField>,
    pub form: PreparedForm,
    pub upload_receipt: Option<Vec<u8>>,
}
fn path(input: &Input) -> &'static str {
    if input.node == "SoniloTextToMusic" {
        "/proxy/sonilo/t2m/generate"
    } else {
        "/proxy/sonilo/v2m/generate"
    }
}
pub fn request(
    input: &Input,
    preparation: &mut dyn SoniloPreparation,
    video: &mut dyn SoniloVideoOwner,
) -> Result<Request, SoniloError> {
    let input = normalized(input)?;
    text_preflight(&input)?;
    let mut upload_receipt = None;
    let mut fields = vec![];
    if input.node == "SoniloTextToMusic" {
        fields.push(FormField {
            name: "prompt".into(),
            value: text(argument(&input, "prompt")?)?.into(),
        });
        fields.push(FormField {
            name: "duration".into(),
            value: preparation.scalar(argument(&input, "duration")?, ScalarStyle::PythonString)?,
        });
    } else {
        let uploaded = video.upload_video(
            argument(&input, "video")?,
            &VideoPolicy {
                maximum_duration_seconds: 360,
                container: "MP4",
                codec: "H264",
                wait_label: "Uploading",
            },
        )?;
        receipt(&uploaded.receipt)?;
        if uploaded.url.len() > 8192
            || !uploaded.url.starts_with("https://")
            || uploaded.url.contains(['\r', '\n', '#', '\\'])
            || uploaded.url.chars().any(char::is_whitespace)
        {
            return Err(SoniloError::Response("Invalid canonical video upload URL"));
        }
        upload_receipt = Some(uploaded.receipt);
        fields.push(FormField {
            name: "video_url".into(),
            value: uploaded.url,
        });
        let prompt = source_strip(text(argument(&input, "prompt")?)?);
        if !prompt.is_empty() {
            fields.push(FormField {
                name: "prompt".into(),
                value: prompt.into(),
            });
        }
    }
    let form = preparation.encode_form(&fields)?;
    receipt(&form.receipt)?;
    if form.content_type != "application/x-www-form-urlencoded" {
        return Err(SoniloError::Response(
            "Canonical Sonilo form must preserve URLencoded MIME",
        ));
    }
    if form.bytes.len() > MAXIMUM_BODY_BYTES {
        return Err(ConformanceError::BodyLimit.into());
    }
    Ok(Request {
        path: path(&input),
        fields,
        form,
        upload_receipt,
    })
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResponsePolicy {
    pub total_timeout_milliseconds: u64,
    pub socket_read_timeout_milliseconds: u64,
}
pub trait SoniloTransport: ProviderTransport {
    fn report_malformed_ndjson(
        &mut self,
        _handle: StreamHandle,
        _warning: &str,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn configure_response(
        &mut self,
        handle: StreamHandle,
        policy: &ResponsePolicy,
    ) -> Result<(), ConformanceError>;
    fn monotonic_seconds(&mut self) -> Result<f64, ConformanceError>;
    fn close_completed_stream(
        &mut self,
        _handle: StreamHandle,
    ) -> Result<Vec<u8>, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
}
#[derive(Clone, Debug)]
pub struct ResultValue {
    pub plan: Request,
    pub audio_bytes: Vec<u8>,
    pub selected_stream: i64,
    pub chunks: u64,
    pub response_receipt: Vec<u8>,
    pub approval_receipt: Vec<u8>,
    pub approved_maximum_microunits: u64,
    pub source_complete: bool,
}
fn report(
    transport: &mut dyn SoniloTransport,
    handle: StreamHandle,
    sequence: &mut u64,
    message: String,
    completed: bool,
) -> Result<(), SoniloError> {
    if message.len() > MAXIMUM_RECEIPT_BYTES {
        return Err(ConformanceError::InvalidProgress.into());
    }
    transport.report_progress(
        handle,
        &Progress {
            sequence: *sequence,
            completed: u64::from(completed),
            total: 1,
            message: Some(message),
        },
    )?;
    *sequence = sequence
        .checked_add(1)
        .ok_or(ConformanceError::InvalidProgress)?;
    Ok(())
}
struct StreamState {
    streams: BTreeMap<i64, Vec<u8>>,
    title: Option<Value>,
    chunks: u64,
    decoded_bytes: usize,
    last_chunk_status: f64,
    started: f64,
    last_clock: f64,
    progress: u64,
    events: u64,
}
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_none_or(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}
fn consume_line(
    raw: &[u8],
    state: &mut StreamState,
    handle: StreamHandle,
    transport: &mut dyn SoniloTransport,
    preparation: &mut dyn SoniloPreparation,
) -> Result<bool, SoniloError> {
    state.events = state
        .events
        .checked_add(1)
        .ok_or(ConformanceError::BodyLimit)?;
    if state.events > MAXIMUM_EVENTS {
        return Err(ConformanceError::BodyLimit.into());
    }
    let line = source_strip(
        std::str::from_utf8(raw).map_err(|_| SoniloError::Response("Invalid source UTF-8 line"))?,
    );
    if line.is_empty() {
        return Ok(false);
    }
    let event: Value = match serde_json::from_str(line) {
        Ok(value) => value,
        Err(_) => {
            transport.report_malformed_ndjson(handle, "Sonilo: skipping malformed NDJSON line")?;
            return Ok(false);
        }
    };
    let event = event
        .as_object()
        .ok_or(SoniloError::Response("Expected source event mapping"))?;
    match event.get("type").and_then(Value::as_str) {
        Some("error") => {
            let code = preparation.scalar(
                event.get("code").unwrap_or(&json!("UNKNOWN")),
                ScalarStyle::PythonString,
            )?;
            let message = preparation.scalar(
                event.get("message").unwrap_or(&json!("Unknown error")),
                ScalarStyle::PythonString,
            )?;
            return Err(SoniloError::Provider(format!(
                "Sonilo generation error ({code}): {message}"
            )));
        }
        Some("duration") => {
            if let Some(duration) = event.get("duration_sec").filter(|value| !value.is_null()) {
                let duration = preparation.scalar(duration, ScalarStyle::FixedOneDecimal)?;
                report(
                    transport,
                    handle,
                    &mut state.progress,
                    format!("Status: Generating\nVideo duration: {duration}s"),
                    false,
                )?;
            }
        }
        Some("titles") | Some("title") => {
            if event.get("type").and_then(Value::as_str) == Some("titles") {
                if let Some(titles) = event.get("titles").filter(|value| truthy(value)) {
                    let title = titles
                        .as_array()
                        .and_then(|values| values.first())
                        .ok_or(SoniloError::Response("Expected source titles list"))?;
                    state.title = Some(title.clone());
                }
            } else if let Some(title) = event.get("title").filter(|value| truthy(value)) {
                state.title = Some(title.clone());
            }
            if let Some(title) = state.title.as_ref().filter(|value| truthy(value)) {
                let title = preparation.scalar(title, ScalarStyle::PythonString)?;
                report(
                    transport,
                    handle,
                    &mut state.progress,
                    format!("Status: Generating\nTitle: {title}"),
                    false,
                )?;
            }
        }
        Some("audio_chunk") => {
            let stream = event
                .get("stream_index")
                .unwrap_or(&json!(0))
                .as_i64()
                .ok_or(SoniloError::Response(
                    "Canonical source stream index must be an integer",
                ))?;
            let decoded = preparation.decode_base64(
                event
                    .get("data")
                    .ok_or(SoniloError::Response("Missing source base64 data"))?,
            )?;
            state.decoded_bytes = state
                .decoded_bytes
                .checked_add(decoded.len())
                .ok_or(ConformanceError::BodyLimit)?;
            if state.decoded_bytes > MAXIMUM_BODY_BYTES
                || !state.streams.contains_key(&stream) && state.streams.len() >= MAXIMUM_STREAMS
            {
                return Err(ConformanceError::BodyLimit.into());
            }
            state
                .streams
                .entry(stream)
                .or_default()
                .extend_from_slice(&decoded);
            state.chunks = state
                .chunks
                .checked_add(1)
                .ok_or(ConformanceError::BodyLimit)?;
            let now = transport.monotonic_seconds()?;
            if !now.is_finite() || now < state.last_clock {
                return Err(ConformanceError::InvalidProgress.into());
            }
            state.last_clock = now;
            if now - state.last_chunk_status >= 1.0 {
                let mut message = "Status: Receiving audio".to_owned();
                if let Some(title) = state.title.as_ref().filter(|value| truthy(value)) {
                    message.push_str(&format!(
                        "\nTitle: {}",
                        preparation.scalar(title, ScalarStyle::PythonString)?
                    ));
                }
                let chunks = preparation.scalar(&json!(state.chunks), ScalarStyle::PythonString)?;
                let elapsed = preparation
                    .scalar(&json!(now - state.started), ScalarStyle::TruncatedInteger)?;
                message.push_str(&format!(
                    "\nChunks received: {chunks}\nTime elapsed: {elapsed}s"
                ));
                report(transport, handle, &mut state.progress, message, false)?;
                state.last_chunk_status = now;
            }
        }
        Some("complete") => return Ok(true),
        _ => {}
    }
    Ok(false)
}
pub fn invoke(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_microunits: u64,
    transport: &mut dyn SoniloTransport,
    preparation: &mut dyn SoniloPreparation,
    video: &mut dyn SoniloVideoOwner,
) -> Result<ResultValue, SoniloError> {
    let input = normalized(input)?;
    text_preflight(&input)?;
    transport.check_cancelled(None)?;
    let request_head = head(
        path(&input),
        credentials,
        "application/x-www-form-urlencoded",
    )?;
    let handle = transport.start_request(&request_head)?;
    if handle.invocation == 0 || handle.slot == 0 || handle.generation == 0 {
        return Err(ConformanceError::InvalidHandle.into());
    }
    let approval = transport.request_cost(
        handle,
        &CostRequest {
            operation: "sonilo.generation.maximum".into(),
            currency: "USD".into(),
            maximum_microunits,
        },
    )?;
    if !approval.accepted || approval.approved_microunits < maximum_microunits {
        return Err(ConformanceError::InvalidCost.into());
    }
    receipt(&approval.receipt)?;
    transport.check_cancelled(Some(handle))?;
    let plan = request(&input, preparation, video)?;
    let started = transport.monotonic_seconds()?;
    if !started.is_finite() || started < 0.0 {
        return Err(ConformanceError::InvalidProgress.into());
    }
    let mut state = StreamState {
        streams: BTreeMap::new(),
        title: None,
        chunks: 0,
        decoded_bytes: 0,
        last_chunk_status: 0.0,
        started,
        last_clock: started,
        progress: 0,
        events: 0,
    };
    transport.configure_response(
        handle,
        &ResponsePolicy {
            total_timeout_milliseconds: 1_200_000,
            socket_read_timeout_milliseconds: 300_000,
        },
    )?;
    report(
        transport,
        handle,
        &mut state.progress,
        "Status: Queued".into(),
        false,
    )?;
    for (sequence, chunk) in plan.form.bytes.chunks(MAXIMUM_CHUNK_BYTES).enumerate() {
        transport.check_cancelled(Some(handle))?;
        let end = (sequence + 1) * MAXIMUM_CHUNK_BYTES >= plan.form.bytes.len();
        transport.write_request_chunk(handle, sequence as u64, chunk, end)?;
    }
    let mut status = None;
    let mut headers = vec![];
    let mut pending = vec![];
    let mut error_body = vec![];
    let mut received_bytes = 0usize;
    let mut sequence = None;
    let mut source_complete = false;
    let response_receipt = loop {
        transport.check_cancelled(Some(handle))?;
        match transport.wait_response(handle, sequence, 300_000)? {
            ResponseEvent::Head {
                status: code,
                headers: response_headers,
            } => {
                if status.is_some()
                    || !(100..=599).contains(&code)
                    || response_headers.len() > 16
                    || response_headers
                        .iter()
                        .map(|header| header.name.len() + header.value.len())
                        .sum::<usize>()
                        > 16384
                    || response_headers.iter().any(|header| {
                        header.name.contains(['\r', '\n']) || header.value.contains(['\r', '\n'])
                    })
                {
                    return Err(ConformanceError::InvalidResponse.into());
                }
                status = Some(code);
                headers = response_headers;
            }
            ResponseEvent::Chunk(bytes) => {
                let code = status.ok_or(ConformanceError::InvalidResponse)?;
                if bytes.len() > MAXIMUM_CHUNK_BYTES {
                    return Err(ConformanceError::ChunkLimit.into());
                }
                received_bytes = received_bytes
                    .checked_add(bytes.len())
                    .ok_or(ConformanceError::BodyLimit)?;
                if received_bytes > MAXIMUM_BODY_BYTES {
                    return Err(ConformanceError::BodyLimit.into());
                }
                if code >= 400 {
                    error_body.extend_from_slice(&bytes);
                } else {
                    pending.extend_from_slice(&bytes);
                    while let Some(end) = pending.iter().position(|byte| *byte == b'\n') {
                        transport.check_cancelled(Some(handle))?;
                        let line: Vec<_> = pending.drain(..=end).collect();
                        if consume_line(&line, &mut state, handle, transport, preparation)? {
                            source_complete = true;
                            break;
                        }
                    }
                    if source_complete {
                        break transport.close_completed_stream(handle)?;
                    }
                }
            }
            ResponseEvent::Completed(response_receipt) => {
                let code = status.ok_or(ConformanceError::InvalidResponse)?;
                receipt(&response_receipt)?;
                if code >= 400 {
                    return Err(SoniloError::Provider(format!(
                        "Sonilo API error ({code}): {}",
                        preparation.http_error(&error_body, &headers)?
                    )));
                }
                if !pending.is_empty() {
                    source_complete =
                        consume_line(&pending, &mut state, handle, transport, preparation)?;
                }
                break response_receipt;
            }
            ResponseEvent::Cancelled => return Err(ConformanceError::Cancelled.into()),
            ResponseEvent::TimedOut => return Err(ConformanceError::HostFailure.into()),
        }
        sequence = Some(sequence.map_or(0u64, |sequence| sequence + 1));
    };
    receipt(&response_receipt)?;
    let selected_stream = if state.streams.contains_key(&0) {
        0
    } else {
        *state
            .streams
            .keys()
            .next()
            .ok_or(SoniloError::Response("Sonilo API returned no audio data."))?
    };
    let audio_bytes = state
        .streams
        .remove(&selected_stream)
        .ok_or(SoniloError::Response("Missing selected stream"))?;
    report(
        transport,
        handle,
        &mut state.progress,
        "Status: Completed".into(),
        true,
    )?;
    Ok(ResultValue {
        plan,
        audio_bytes,
        selected_stream,
        chunks: state.chunks,
        response_receipt,
        approval_receipt: approval.receipt,
        approved_maximum_microunits: maximum_microunits,
        source_complete,
    })
}
pub trait SoniloAudioMaterializer {
    fn materialize_audio(
        &mut self,
        result: &ResultValue,
    ) -> Result<Vec<EncodedOutput>, ConformanceError>;
}
pub struct MaterializationOwners<'a> {
    pub preparation: &'a mut dyn SoniloPreparation,
    pub video: &'a mut dyn SoniloVideoOwner,
    pub audio: &'a mut dyn SoniloAudioMaterializer,
    pub sink: &'a mut dyn MaterializationSink,
}
pub fn invoke_and_materialize(
    input: &Input,
    credentials: CredentialAvailability,
    maximum_microunits: u64,
    transport: &mut dyn SoniloTransport,
    owners: &mut MaterializationOwners<'_>,
) -> Result<(), SoniloError> {
    let result = invoke(
        input,
        credentials,
        maximum_microunits,
        transport,
        owners.preparation,
        owners.video,
    )?;
    transport.check_cancelled(None)?;
    let outputs = owners.audio.materialize_audio(&result)?;
    let registry = TypeRegistry::built_in().map_err(|_| ConformanceError::HostFailure)?;
    let type_id = registry
        .resolve("AUDIO")
        .map_err(|_| ConformanceError::HostFailure)?
        .clone();
    let family = registry
        .family(&type_id)
        .map_err(|_| ConformanceError::HostFailure)?;
    let expected = [ExpectedOutput {
        port_id: "output_0".into(),
        type_id,
        family,
    }];
    let batch =
        validate_outputs(&expected, outputs, MAXIMUM_BODY_BYTES).map_err(ConformanceError::from)?;
    transport.check_cancelled(None)?;
    publish_validated(batch, owners.sink).map_err(ConformanceError::from)?;
    Ok(())
}
fn decode_hex(value: &Value) -> Result<Vec<u8>, ConformanceError> {
    let value = value.as_str().ok_or(ConformanceError::InvalidRequest)?;
    if value.len() % 2 != 0 || value.len() > 2 * MAXIMUM_BODY_BYTES {
        return Err(ConformanceError::InvalidRequest);
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            std::str::from_utf8(pair)
                .map_err(|_| ConformanceError::InvalidRequest)
                .and_then(|pair| {
                    u8::from_str_radix(pair, 16).map_err(|_| ConformanceError::InvalidRequest)
                })
        })
        .collect()
}
struct SourceProbePreparation {
    source: Value,
    case: Value,
    fields: Vec<FormField>,
    unavailable: bool,
    bad_receipt: bool,
}
impl SourceProbePreparation {
    fn new(case: &Value) -> Result<Self, SoniloError> {
        Ok(Self {
            source: oracle()?,
            case: case.clone(),
            fields: vec![],
            unavailable: false,
            bad_receipt: false,
        })
    }
}
impl SoniloPreparation for SourceProbePreparation {
    fn encode_form(&mut self, fields: &[FormField]) -> Result<PreparedForm, ConformanceError> {
        if self.unavailable {
            return Err(ConformanceError::HostFailure);
        }
        self.fields = fields.to_vec();
        let request = self.case["requests"]
            .as_array()
            .and_then(|values| values.first())
            .ok_or(ConformanceError::InvalidRequest)?;
        let actual: Vec<Value> = fields
            .iter()
            .map(|field| json!([field.name, field.value]))
            .collect();
        if json!(actual) != request["fields"] {
            return Err(ConformanceError::InvalidRequest);
        }
        let fields = fields
            .iter()
            .map(|field| crate::common::UrlEncodedField {
                name: field.name.clone(),
                value: field.value.clone(),
            })
            .collect::<Vec<_>>();
        let bytes = crate::common::UrlEncodedField::encode_checked(
            &fields,
            crate::common::TransportLimits {
                maximum_body_bytes: MAXIMUM_BODY_BYTES,
                ..crate::common::TransportLimits::conformance()
            },
        )?;
        if bytes != decode_hex(&request["body_hex"])? {
            return Err(ConformanceError::InvalidRequest);
        }
        Ok(PreparedForm {
            bytes,
            content_type: request["content_type"]
                .as_str()
                .ok_or(ConformanceError::InvalidRequest)?
                .into(),
            receipt: if self.bad_receipt {
                vec![]
            } else {
                b"source-form-preparation-boundary".to_vec()
            },
        })
    }
    fn scalar(&mut self, value: &Value, style: ScalarStyle) -> Result<String, ConformanceError> {
        if self.unavailable {
            return Err(ConformanceError::HostFailure);
        }
        self.source["scalar_probes"]
            .as_array()
            .and_then(|probes| {
                probes
                    .iter()
                    .find(|probe| probe["value"] == *value && probe["style"] == style.name())
            })
            .and_then(|probe| probe["output"].as_str())
            .map(str::to_owned)
            .ok_or(ConformanceError::InvalidRequest)
    }
    fn decode_base64(&mut self, value: &Value) -> Result<Vec<u8>, ConformanceError> {
        let probe = self.source["base64_probes"]
            .as_array()
            .and_then(|probes| probes.iter().find(|probe| probe["value"] == *value))
            .ok_or(ConformanceError::InvalidRequest)?;
        decode_hex(&probe["bytes_hex"])
    }
    fn http_error(
        &mut self,
        _bytes: &[u8],
        _headers: &[Header],
    ) -> Result<String, ConformanceError> {
        self.case["message"]
            .as_str()
            .and_then(|message| message.split_once(": "))
            .map(|(_, message)| message.to_owned())
            .ok_or(ConformanceError::InvalidRequest)
    }
}
#[derive(Default)]
struct SourceProbeVideo {
    calls: Vec<(Value, VideoPolicy)>,
    reject: bool,
    bad_receipt: bool,
}
impl SoniloVideoOwner for SourceProbeVideo {
    fn upload_video(
        &mut self,
        video: &Value,
        policy: &VideoPolicy,
    ) -> Result<UploadedVideo, ConformanceError> {
        self.calls.push((video.clone(), policy.clone()));
        if self.reject
            || video["fail_duration"] == true
            || video["duration"]
                .as_f64()
                .is_some_and(|duration| duration > policy.maximum_duration_seconds as f64)
        {
            return Err(ConformanceError::HostFailure);
        }
        Ok(UploadedVideo {
            url: "https://fixture.invalid/uploaded.mp4".into(),
            receipt: if self.bad_receipt {
                vec![]
            } else {
                b"injected-source-video-upload-boundary".to_vec()
            },
        })
    }
}
struct SourceProbeTransport {
    events: VecDeque<Result<ResponseEvent, ConformanceError>>,
    clocks: VecDeque<f64>,
    body: Vec<u8>,
    heads: Vec<RequestHead>,
    costs: Vec<CostRequest>,
    progresses: Vec<Progress>,
    policy: Option<ResponsePolicy>,
    close_count: usize,
    diagnostics: Vec<String>,
    cancel: bool,
    deny_cost: bool,
    bad_receipt: bool,
}
impl SourceProbeTransport {
    fn new(case: &Value, fragment: usize) -> Result<Self, ConformanceError> {
        let status = case["status"]
            .as_u64()
            .ok_or(ConformanceError::InvalidRequest)? as u16;
        let mut body = vec![];
        if status >= 400 {
            body = if case["fail_json"] == true {
                case["error_text"]
                    .as_str()
                    .ok_or(ConformanceError::InvalidRequest)?
                    .as_bytes()
                    .to_vec()
            } else {
                serde_json::to_vec(&case["error_json"])
                    .map_err(|_| ConformanceError::InvalidRequest)?
            };
        } else {
            for line in case["lines_hex"]
                .as_array()
                .ok_or(ConformanceError::InvalidRequest)?
            {
                body.extend_from_slice(&decode_hex(line)?);
            }
        }
        let mut events = VecDeque::from([Ok(ResponseEvent::Head {
            status,
            headers: vec![],
        })]);
        for chunk in body.chunks(fragment.max(1)) {
            events.push_back(Ok(ResponseEvent::Chunk(chunk.to_vec())));
        }
        events.push_back(Ok(ResponseEvent::Completed(
            b"injected-response-receipt".to_vec(),
        )));
        let clocks = case["clock"]
            .as_array()
            .ok_or(ConformanceError::InvalidRequest)?
            .iter()
            .map(|value| value.as_f64().ok_or(ConformanceError::InvalidRequest))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            events,
            clocks,
            body: vec![],
            heads: vec![],
            costs: vec![],
            progresses: vec![],
            policy: None,
            close_count: 0,
            diagnostics: vec![],
            cancel: case["cancel"] == true,
            deny_cost: false,
            bad_receipt: false,
        })
    }
}
impl ProviderTransport for SourceProbeTransport {
    fn check_cancelled(&mut self, _handle: Option<StreamHandle>) -> Result<(), ConformanceError> {
        if self.cancel && !self.body.is_empty() {
            Err(ConformanceError::Cancelled)
        } else {
            Ok(())
        }
    }
    fn start_request(&mut self, head: &RequestHead) -> Result<StreamHandle, ConformanceError> {
        self.heads.push(head.clone());
        Ok(StreamHandle {
            invocation: 41,
            slot: self.heads.len() as u32,
            generation: 1,
        })
    }
    fn write_request_chunk(
        &mut self,
        _handle: StreamHandle,
        sequence: u64,
        bytes: &[u8],
        _end: bool,
    ) -> Result<(), ConformanceError> {
        if sequence != self.body.len().div_ceil(MAXIMUM_CHUNK_BYTES) as u64 {
            return Err(ConformanceError::InvalidRequest);
        }
        self.body.extend_from_slice(bytes);
        Ok(())
    }
    fn upload(&mut self, _handle: StreamHandle, _bytes: &[u8]) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn create_storage(
        &mut self,
        _handle: StreamHandle,
        _request: &crate::common::StorageCreateRequest,
    ) -> Result<crate::common::StorageCreateResponse, ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn put_signed(
        &mut self,
        _handle: StreamHandle,
        _request: &crate::common::SignedPutRequest,
    ) -> Result<(), ConformanceError> {
        Err(ConformanceError::HostFailure)
    }
    fn request_cost(
        &mut self,
        _handle: StreamHandle,
        request: &CostRequest,
    ) -> Result<crate::common::CostResponse, ConformanceError> {
        self.costs.push(request.clone());
        Ok(crate::common::CostResponse {
            accepted: !self.deny_cost,
            approved_microunits: request.maximum_microunits,
            receipt: b"source-cost-boundary".to_vec(),
        })
    }
    fn report_progress(
        &mut self,
        _handle: StreamHandle,
        progress: &Progress,
    ) -> Result<(), ConformanceError> {
        self.progresses.push(progress.clone());
        Ok(())
    }
    fn wait_response(
        &mut self,
        _handle: StreamHandle,
        _after_sequence: Option<u64>,
        _timeout_milliseconds: u64,
    ) -> Result<ResponseEvent, ConformanceError> {
        self.events
            .pop_front()
            .ok_or(ConformanceError::InvalidResponse)?
    }
}
impl SoniloTransport for SourceProbeTransport {
    fn report_malformed_ndjson(
        &mut self,
        _handle: StreamHandle,
        warning: &str,
    ) -> Result<(), ConformanceError> {
        self.diagnostics.push(warning.into());
        Ok(())
    }
    fn configure_response(
        &mut self,
        _handle: StreamHandle,
        policy: &ResponsePolicy,
    ) -> Result<(), ConformanceError> {
        self.policy = Some(policy.clone());
        Ok(())
    }
    fn monotonic_seconds(&mut self) -> Result<f64, ConformanceError> {
        self.clocks.pop_front().ok_or(ConformanceError::HostFailure)
    }
    fn close_completed_stream(
        &mut self,
        _handle: StreamHandle,
    ) -> Result<Vec<u8>, ConformanceError> {
        self.close_count += 1;
        Ok(if self.bad_receipt {
            vec![]
        } else {
            b"injected-source-close-receipt".to_vec()
        })
    }
}
fn source_input(case: &Value) -> Result<Input, SoniloError> {
    Ok(serde_json::from_value(case["input"].clone())?)
}
fn credentials() -> CredentialAvailability {
    CredentialAvailability {
        auth_token: true,
        api_key: true,
    }
}
fn compare_case(case: &Value, failure: bool, fragment: usize) -> Result<(), Box<dyn Error>> {
    let input = source_input(case)?;
    let mut preparation = SourceProbePreparation::new(case)?;
    let mut video = SourceProbeVideo::default();
    let mut transport = SourceProbeTransport::new(case, fragment)?;
    let result = invoke(
        &input,
        credentials(),
        1,
        &mut transport,
        &mut preparation,
        &mut video,
    );
    if failure {
        if result.is_ok() {
            return Err("Expected source Sonilo failure".into());
        }
    } else {
        let result = result?;
        if result.audio_bytes != decode_hex(&case["outputs"][0]["encoded_source_probe_only"])?
            || result.plan.path
                != case["requests"][0]["url"]
                    .as_str()
                    .ok_or("source URL")?
                    .strip_prefix("https://fixture.invalid/api")
                    .ok_or("source URL prefix")?
        {
            return Err("Sonilo source audio/path differs".into());
        }
    }
    let messages: Vec<_> = transport
        .progresses
        .iter()
        .map(|progress| progress.message.clone())
        .collect();
    if json!(messages) != case["progress"] {
        return Err(format!(
            "Sonilo source progress differs for {:?}: {:?} vs {}",
            input.arguments, messages, case["progress"]
        )
        .into());
    }
    if let Some(expected) = case["requests"]
        .as_array()
        .and_then(|requests| requests.first())
    {
        if transport.body != decode_hex(&expected["body_hex"])? {
            return Err("Sonilo URLencoded source body differs".into());
        }
        let head = transport.heads.first().ok_or("source request head")?;
        if head.headers.iter().any(|header| header.name == "Accept")
            || head.method != HttpMethod::Post
            || head.headers.last().map(|header| header.value.as_str())
                != expected["content_type"].as_str()
            || transport.policy
                != Some(ResponsePolicy {
                    total_timeout_milliseconds: 1_200_000,
                    socket_read_timeout_milliseconds: 300_000,
                })
        {
            return Err("Sonilo source header/policy differs".into());
        }
    } else if !transport.body.is_empty() {
        return Err("Sonilo source preflight sent a body".into());
    }
    if input.node == "SoniloVideoToMusic"
        && (video.calls.len() != 1
            || video
                .calls
                .first()
                .map(|(_, policy)| policy.maximum_duration_seconds)
                != Some(360))
    {
        return Err("Sonilo video ordering/owner policy differs".into());
    }
    if transport
        .costs
        .iter()
        .any(|cost| cost.operation != "sonilo.generation.maximum" || cost.maximum_microunits != 1)
    {
        return Err("Sonilo source cost identity differs".into());
    }
    Ok(())
}
pub fn run_conformance(fixture: &str) -> Result<(), ConformanceError> {
    let checked = || -> Result<(), Box<dyn Error>> {
        let claims: Value = serde_json::from_str(fixture)?;
        if claims["namespace"] != NAMESPACE
            || claims["source_sha256"] != SOURCE_SHA256
            || claims["nodes"] != serde_json::to_value(NODES)?
            || claims["routes"] != serde_json::to_value(ROUTES)?
            || claims["aliases"] != json!(ALIASES)
        {
            return Err("Sonilo claim drift".into());
        }
        let source = oracle()?;
        if source["source_sha256"]["comfy_api_nodes/nodes_sonilo.py"] != SOURCE_SHA256
            || source["aiohttp_version"] != "3.11.8"
        {
            return Err("Sonilo source fingerprint drift".into());
        }
        let mut covered = BTreeSet::new();
        let mut paths = BTreeSet::new();
        for case in source["cases"].as_array().ok_or("Sonilo source cases")? {
            compare_case(case, false, MAXIMUM_CHUNK_BYTES)?;
            covered.insert(source_input(case)?.node);
            paths.insert(
                case["requests"][0]["url"]
                    .as_str()
                    .ok_or("URL")?
                    .strip_prefix("https://fixture.invalid/api")
                    .ok_or("URL prefix")?
                    .to_owned(),
            );
        }
        for case in source["errors"].as_array().ok_or("Sonilo source errors")? {
            compare_case(case, true, MAXIMUM_CHUNK_BYTES)?;
        }
        if covered
            != NODES
                .iter()
                .map(|node| node.node_identifier.to_owned())
                .collect()
            || paths != ROUTES.iter().map(|route| route.path.to_owned()).collect()
        {
            return Err("Sonilo source coverage differs".into());
        }
        Ok(())
    };
    checked().map_err(|_| ConformanceError::InvalidRequest)
}
#[cfg(test)]
mod tests {
    use super::*;
    type TestResult = Result<(), Box<dyn Error>>;
    fn first_case() -> Result<Value, SoniloError> {
        oracle()?["cases"]
            .as_array()
            .and_then(|cases| cases.first())
            .cloned()
            .ok_or(SoniloError::Response("Missing source case"))
    }
    #[test]
    fn exact_claims_and_actual_source_success_failures() -> TestResult {
        run_conformance(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../sim_test_support/fixtures/providers/sonilo/conformance.fixture"
        )))?;
        Ok(())
    }
    #[test]
    fn ndjson_split_frames_preserve_unicode_audio_and_progress() -> TestResult {
        for case in oracle()?["cases"].as_array().ok_or("cases")? {
            for fragment in [1, 7, 31] {
                compare_case(case, false, fragment)?;
            }
        }
        Ok(())
    }
    #[test]
    fn malformed_line_reports_only_constant_warning_and_keeps_real_audio() -> TestResult {
        let mut case = first_case()?;
        let lines = case["lines_hex"].as_array_mut().ok_or("source lines")?;
        lines.insert(
            0,
            json!("707269766174652d726573706f6e73652d6e6f742d6a736f6e0a"),
        );
        let mut transport = SourceProbeTransport::new(&case, 7)?;
        let result = invoke(
            &source_input(&case)?,
            credentials(),
            1,
            &mut transport,
            &mut SourceProbePreparation::new(&case)?,
            &mut SourceProbeVideo::default(),
        )?;
        assert_eq!(
            transport.diagnostics,
            vec!["Sonilo: skipping malformed NDJSON line"]
        );
        assert_eq!(
            result.audio_bytes,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../sim_test_support/fixtures/providers/sonilo/source-response.wav"
            ))
        );
        Ok(())
    }
    #[test]
    fn source_defaults_and_ignored_seed() -> TestResult {
        let input = Input {
            node: "SoniloTextToMusic".into(),
            arguments: BTreeMap::from([("prompt".into(), json!("music"))]),
        };
        let input = normalized(&input)?;
        assert_eq!(input.arguments["duration"], json!(1));
        assert_eq!(input.arguments["seed"], json!(0));
        let mut changed = input.clone();
        changed.arguments.insert("seed".into(), json!(-1));
        assert_eq!(normalized(&changed)?.arguments["duration"], json!(1));
        let mut unknown = input;
        unknown.arguments.insert("unknown".into(), Value::Null);
        assert!(normalized(&unknown).is_err());
        Ok(())
    }
    #[test]
    fn default_unbound_preparation_rejects_without_publishing() -> TestResult {
        struct Unbound;
        impl SoniloPreparation for Unbound {}
        impl SoniloVideoOwner for Unbound {}
        let input = source_input(&first_case()?)?;
        assert!(request(&input, &mut Unbound, &mut Unbound).is_err());
        Ok(())
    }
    #[test]
    fn auth_and_cost_denial_precede_unapproved_body() -> TestResult {
        let case = first_case()?;
        let input = source_input(&case)?;
        for missing_auth in [true, false] {
            let mut transport = SourceProbeTransport::new(&case, 1024)?;
            transport.deny_cost = true;
            let mut preparation = SourceProbePreparation::new(&case)?;
            let mut video = SourceProbeVideo::default();
            let auth = if missing_auth {
                CredentialAvailability {
                    auth_token: false,
                    api_key: false,
                }
            } else {
                credentials()
            };
            assert!(
                invoke(
                    &input,
                    auth,
                    1,
                    &mut transport,
                    &mut preparation,
                    &mut video
                )
                .is_err()
            );
            assert!(transport.body.is_empty());
            assert!(preparation.fields.is_empty());
            if !missing_auth {
                assert_eq!(
                    transport.costs.first().map(|cost| cost.operation.as_str()),
                    Some("sonilo.generation.maximum")
                );
            }
        }
        Ok(())
    }
    #[test]
    fn api_key_header_and_exact_stream_timeouts() -> TestResult {
        let case = first_case()?;
        let mut transport = SourceProbeTransport::new(&case, 1024)?;
        let mut preparation = SourceProbePreparation::new(&case)?;
        invoke(
            &source_input(&case)?,
            CredentialAvailability {
                auth_token: false,
                api_key: true,
            },
            1,
            &mut transport,
            &mut preparation,
            &mut SourceProbeVideo::default(),
        )?;
        assert_eq!(
            transport
                .heads
                .first()
                .and_then(|head| head.secret_id.as_deref()),
            Some("sim_org.api_key")
        );
        assert_eq!(
            transport.policy,
            Some(ResponsePolicy {
                total_timeout_milliseconds: 1_200_000,
                socket_read_timeout_milliseconds: 300_000
            })
        );
        Ok(())
    }
    #[test]
    fn video_upload_precedes_prompt_type_failure() -> TestResult {
        let input = Input {
            node: "SoniloVideoToMusic".into(),
            arguments: BTreeMap::from([
                ("video".into(), json!({"duration":30})),
                ("prompt".into(), json!(42)),
            ]),
        };
        let mut video = SourceProbeVideo::default();
        assert!(
            request(
                &input,
                &mut SourceProbePreparation::new(&first_case()?)?,
                &mut video
            )
            .is_err()
        );
        assert_eq!(video.calls.len(), 1);
        Ok(())
    }
    #[test]
    fn empty_or_oversized_owner_receipt_is_rejected() -> TestResult {
        let case = first_case()?;
        let mut preparation = SourceProbePreparation::new(&case)?;
        preparation.bad_receipt = true;
        assert!(
            request(
                &source_input(&case)?,
                &mut preparation,
                &mut SourceProbeVideo::default()
            )
            .is_err()
        );
        assert!(receipt(&vec![1; MAXIMUM_RECEIPT_BYTES + 1]).is_err());
        Ok(())
    }
    #[test]
    fn invalid_response_order_chunk_bound_and_worker_loss() -> TestResult {
        let case = first_case()?;
        for events in [
            vec![Ok(ResponseEvent::Chunk(vec![1]))],
            vec![
                Ok(ResponseEvent::Head {
                    status: 200,
                    headers: vec![],
                }),
                Ok(ResponseEvent::Chunk(vec![0; MAXIMUM_CHUNK_BYTES + 1])),
            ],
            vec![Err(ConformanceError::WorkerLost)],
            vec![Ok(ResponseEvent::TimedOut)],
        ] {
            let mut transport = SourceProbeTransport::new(&case, 1024)?;
            transport.events = events.into();
            assert!(
                invoke(
                    &source_input(&case)?,
                    credentials(),
                    1,
                    &mut transport,
                    &mut SourceProbePreparation::new(&case)?,
                    &mut SourceProbeVideo::default()
                )
                .is_err()
            );
            assert_eq!(transport.heads.len(), 1);
        }
        compare_case(&case, false, 1024)?;
        Ok(())
    }
    #[test]
    fn backward_clock_and_missing_close_receipt_fail_closed() -> TestResult {
        let mut case = first_case()?;
        case["clock"] = json!([10, 9, 11]);
        let mut transport = SourceProbeTransport::new(&case, 1024)?;
        assert!(
            invoke(
                &source_input(&case)?,
                credentials(),
                1,
                &mut transport,
                &mut SourceProbePreparation::new(&case)?,
                &mut SourceProbeVideo::default()
            )
            .is_err()
        );
        let case = first_case()?;
        let mut transport = SourceProbeTransport::new(&case, 1024)?;
        transport.bad_receipt = true;
        assert!(
            invoke(
                &source_input(&case)?,
                credentials(),
                1,
                &mut transport,
                &mut SourceProbePreparation::new(&case)?,
                &mut SourceProbeVideo::default()
            )
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn complete_ignores_late_error_eof_and_empty_chunks_remain_source_behaviors() -> TestResult {
        let source = oracle()?;
        assert!(
            source["cases"]
                .as_array()
                .ok_or("cases")?
                .iter()
                .any(|case| case["outputs"][0]["encoded_source_probe_only"] == "")
        );
        for case in source["cases"]
            .as_array()
            .ok_or("cases")?
            .iter()
            .rev()
            .take(7)
        {
            compare_case(case, false, 1)?;
        }
        Ok(())
    }
    #[test]
    fn real_pcm_audio_decoder_rejection_has_no_native_publication() -> TestResult {
        use crate::common::{MaterializationError, MaterializedBatch};
        struct Audio {
            calls: usize,
        }
        impl SoniloAudioMaterializer for Audio {
            fn materialize_audio(
                &mut self,
                result: &ResultValue,
            ) -> Result<Vec<EncodedOutput>, ConformanceError> {
                if result.audio_bytes
                    != include_bytes!(concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/../../sim_test_support/fixtures/providers/sonilo/source-response.wav"
                    ))
                {
                    return Err(ConformanceError::InvalidResponse);
                }
                self.calls += 1;
                Err(ConformanceError::HostFailure)
            }
        }
        #[derive(Default)]
        struct Sink {
            calls: usize,
        }
        impl MaterializationSink for Sink {
            fn publish(&mut self, _batch: MaterializedBatch) -> Result<(), MaterializationError> {
                self.calls += 1;
                Ok(())
            }
        }
        let case = first_case()?;
        let mut transport = SourceProbeTransport::new(&case, 1024)?;
        let mut preparation = SourceProbePreparation::new(&case)?;
        let mut video = SourceProbeVideo::default();
        let mut audio = Audio { calls: 0 };
        let mut sink = Sink::default();
        assert!(
            invoke_and_materialize(
                &source_input(&case)?,
                credentials(),
                1,
                &mut transport,
                &mut MaterializationOwners {
                    preparation: &mut preparation,
                    video: &mut video,
                    audio: &mut audio,
                    sink: &mut sink
                }
            )
            .is_err()
        );
        assert_eq!(audio.calls, 1);
        assert_eq!(sink.calls, 0);
        Ok(())
    }
}
