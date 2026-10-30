use std::{collections::VecDeque, error::Error, fmt};

use super::{
    EncodedOutput, ExpectedOutput, MaterializationError, MaterializationSink, publish_validated,
    validate_outputs,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HttpMethod {
    Delete,
    Get,
    Head,
    Options,
    Patch,
    Post,
    Put,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Header {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestHead {
    pub method: HttpMethod,
    pub path: String,
    pub secret_id: Option<String>,
    pub headers: Vec<Header>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StorageCreateRequest {
    pub file_name: String,
    pub content_type: Option<String>,
    pub byte_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StorageCreateResponse {
    pub upload_url: String,
    pub download_url: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignedPutRequest {
    pub upload_url: String,
    pub content_type: Option<String>,
    pub bytes: Vec<u8>,
}

impl StorageCreateRequest {
    pub fn checked_body(&self, limits: TransportLimits) -> Result<Vec<u8>, ConformanceError> {
        validate_limits(limits)?;
        if self.file_name.is_empty()
            || self.file_name.len() > 256
            || self
                .file_name
                .bytes()
                .any(|byte| byte < 0x20 || byte == 0x7f)
            || self.byte_length > limits.maximum_body_bytes as u64
        {
            return Err(ConformanceError::InvalidRequest);
        }
        validate_upload_content_type(self.content_type.as_deref())?;
        let mut body = serde_json::Map::new();
        body.insert(
            "file_name".into(),
            serde_json::Value::String(self.file_name.clone()),
        );
        if let Some(content_type) = &self.content_type {
            body.insert(
                "content_type".into(),
                serde_json::Value::String(content_type.clone()),
            );
        }
        let encoded = serde_json::to_vec(&body).map_err(|_| ConformanceError::InvalidRequest)?;
        if encoded.len() > limits.maximum_body_bytes {
            return Err(ConformanceError::BodyLimit);
        }
        Ok(encoded)
    }
}

impl SignedPutRequest {
    pub fn checked_headers(
        &self,
        limits: TransportLimits,
    ) -> Result<Vec<Header>, ConformanceError> {
        validate_limits(limits)?;
        if self.bytes.len() > limits.maximum_body_bytes {
            return Err(ConformanceError::BodyLimit);
        }
        validate_upload_content_type(self.content_type.as_deref())?;
        let headers = self
            .content_type
            .as_ref()
            .map(|content_type| Header {
                name: "Content-Type".into(),
                value: content_type.clone(),
            })
            .into_iter()
            .collect::<Vec<_>>();
        validate_headers(&headers, limits)?;
        Ok(headers)
    }

    pub fn suppress_automatic_content_type(&self) -> bool {
        self.content_type.is_none()
    }
}

fn validate_upload_content_type(content_type: Option<&str>) -> Result<(), ConformanceError> {
    if content_type.is_some_and(|value| {
        value.is_empty()
            || value.len() > 256
            || !value.is_ascii()
            || !value.contains('/')
            || value.bytes().any(|byte| byte < 0x20 || byte == 0x7f)
    }) {
        return Err(ConformanceError::InvalidRequest);
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UrlEncodedField {
    pub name: String,
    pub value: String,
}

impl UrlEncodedField {
    pub fn encode_checked(
        fields: &[Self],
        limits: TransportLimits,
    ) -> Result<Vec<u8>, ConformanceError> {
        validate_limits(limits)?;
        let mut body = Vec::new();
        body.try_reserve(limits.maximum_body_bytes.min(4096))
            .map_err(|_| ConformanceError::BodyLimit)?;
        for (index, field) in fields.iter().enumerate() {
            if index != 0 {
                append_bounded(&mut body, b"&", limits.maximum_body_bytes)?;
            }
            append_urlencoded_string(&mut body, &field.name, limits.maximum_body_bytes)?;
            append_bounded(&mut body, b"=", limits.maximum_body_bytes)?;
            append_urlencoded_string(&mut body, &field.value, limits.maximum_body_bytes)?;
        }
        Ok(body)
    }
}

fn append_urlencoded_string(
    body: &mut Vec<u8>,
    value: &str,
    maximum_body_bytes: usize,
) -> Result<(), ConformanceError> {
    let hexadecimal = |nibble: u8| {
        if nibble < 10 {
            b'0' + nibble
        } else {
            b'A' + (nibble - 10)
        }
    };
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            append_bounded(body, &[byte], maximum_body_bytes)?;
        } else if byte == b' ' {
            append_bounded(body, b"+", maximum_body_bytes)?;
        } else {
            // aiohttp's string form delegates to Python quote_plus, which
            // preserves tilde and percent-encodes literal plus and asterisk.
            let escaped = [b'%', hexadecimal(byte >> 4), hexadecimal(byte & 0x0f)];
            append_bounded(body, &escaped, maximum_body_bytes)?;
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultipartField {
    pub name: String,
    pub file_name: Option<String>,
    pub media_type: Option<String>,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamHandle {
    pub invocation: u64,
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CostRequest {
    pub operation: String,
    pub currency: String,
    pub maximum_microunits: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CostResponse {
    pub accepted: bool,
    pub approved_microunits: u64,
    pub receipt: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Progress {
    pub sequence: u64,
    pub completed: u64,
    pub total: u64,
    pub message: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResponseEvent {
    TimedOut,
    Head { status: u16, headers: Vec<Header> },
    Chunk(Vec<u8>),
    Completed(Vec<u8>),
    Cancelled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransportLimits {
    pub maximum_headers: usize,
    pub maximum_header_bytes: usize,
    pub maximum_body_bytes: usize,
    pub maximum_chunk_bytes: usize,
    pub maximum_attempts: u32,
    pub maximum_polls: u32,
    pub maximum_wait_milliseconds: u64,
    pub maximum_progress_total: u64,
    pub maximum_output_bytes: usize,
}

impl TransportLimits {
    pub const fn conformance() -> Self {
        Self {
            maximum_headers: 16,
            maximum_header_bytes: 16 * 1024,
            maximum_body_bytes: 1024 * 1024,
            maximum_chunk_bytes: 64 * 1024,
            maximum_attempts: 3,
            maximum_polls: 16,
            maximum_wait_milliseconds: 1_000,
            maximum_progress_total: 1_000_000,
            maximum_output_bytes: 16 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvocationPlan {
    pub request: RequestHead,
    pub multipart_boundary: String,
    pub multipart_fields: Vec<MultipartField>,
    pub upload: Vec<u8>,
    pub cost: CostRequest,
    pub progress: Progress,
    pub wait_milliseconds: u64,
    pub expected_outputs: Vec<ExpectedOutput>,
    pub outputs: Vec<EncodedOutput>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConformanceError {
    InvalidRequest,
    HeaderLimit,
    BodyLimit,
    ChunkLimit,
    RetryLimit,
    PollLimit,
    InvalidCost,
    InvalidProgress,
    InvalidResponse,
    InvalidHandle,
    Cancelled,
    WorkerLost,
    RateLimited { retry_after_milliseconds: u64 },
    HostFailure,
    Materialization(MaterializationError),
}

impl fmt::Display for ConformanceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Materialization(error) => write!(formatter, "{error}"),
            other => formatter.write_str(match other {
                Self::InvalidRequest => "invalid provider request",
                Self::HeaderLimit => "provider headers exceed their bound",
                Self::BodyLimit => "provider body exceeds its bound",
                Self::ChunkLimit => "provider chunk exceeds its bound",
                Self::RetryLimit => "provider retries exceed their bound",
                Self::PollLimit => "provider polls exceed their bound",
                Self::InvalidCost => "invalid provider cost response",
                Self::InvalidProgress => "invalid provider progress",
                Self::InvalidResponse => "invalid provider response order",
                Self::InvalidHandle => "invalid provider stream handle",
                Self::Cancelled => "provider invocation was cancelled",
                Self::WorkerLost => "provider worker was lost",
                Self::RateLimited { .. } => "provider request was rate limited",
                Self::HostFailure => "provider host operation failed",
                Self::Materialization(_) => unreachable!(),
            }),
        }
    }
}

impl Error for ConformanceError {}

impl From<MaterializationError> for ConformanceError {
    fn from(error: MaterializationError) -> Self {
        Self::Materialization(error)
    }
}

pub trait ProviderTransport {
    fn check_cancelled(&mut self, handle: Option<StreamHandle>) -> Result<(), ConformanceError>;
    fn start_request(&mut self, head: &RequestHead) -> Result<StreamHandle, ConformanceError>;
    fn write_request_chunk(
        &mut self,
        handle: StreamHandle,
        sequence: u64,
        bytes: &[u8],
        end: bool,
    ) -> Result<(), ConformanceError>;
    fn upload(&mut self, handle: StreamHandle, bytes: &[u8]) -> Result<(), ConformanceError>;
    fn create_storage(
        &mut self,
        handle: StreamHandle,
        request: &StorageCreateRequest,
    ) -> Result<StorageCreateResponse, ConformanceError>;
    fn put_signed(
        &mut self,
        handle: StreamHandle,
        request: &SignedPutRequest,
    ) -> Result<(), ConformanceError>;
    fn request_cost(
        &mut self,
        handle: StreamHandle,
        request: &CostRequest,
    ) -> Result<CostResponse, ConformanceError>;
    fn report_progress(
        &mut self,
        handle: StreamHandle,
        progress: &Progress,
    ) -> Result<(), ConformanceError>;
    fn wait_response(
        &mut self,
        handle: StreamHandle,
        after_sequence: Option<u64>,
        timeout_milliseconds: u64,
    ) -> Result<ResponseEvent, ConformanceError>;
}

pub fn execute_plan(
    plan: &InvocationPlan,
    limits: TransportLimits,
    transport: &mut dyn ProviderTransport,
    sink: &mut dyn MaterializationSink,
) -> Result<(), ConformanceError> {
    validate_limits(limits)?;
    let body = preflight(plan, limits)?;
    let outputs = validate_outputs(
        &plan.expected_outputs,
        plan.outputs.clone(),
        limits.maximum_output_bytes,
    )?;
    let mut attempt = 0u32;
    let handle = loop {
        attempt = attempt.checked_add(1).ok_or(ConformanceError::RetryLimit)?;
        if attempt > limits.maximum_attempts {
            return Err(ConformanceError::RetryLimit);
        }
        transport.check_cancelled(None)?;
        match transport.start_request(&plan.request) {
            Ok(handle) if handle.invocation != 0 && handle.slot != 0 && handle.generation != 0 => {
                break handle;
            }
            Ok(_) => return Err(ConformanceError::InvalidHandle),
            Err(ConformanceError::WorkerLost) if attempt < limits.maximum_attempts => continue,
            Err(error) => return Err(error),
        }
    };

    for (sequence, chunk) in body.chunks(limits.maximum_chunk_bytes).enumerate() {
        transport.check_cancelled(Some(handle))?;
        transport.write_request_chunk(
            handle,
            u64::try_from(sequence).map_err(|_| ConformanceError::ChunkLimit)?,
            chunk,
            (sequence + 1) * limits.maximum_chunk_bytes >= body.len(),
        )?;
    }
    transport.check_cancelled(Some(handle))?;
    transport.upload(handle, &plan.upload)?;
    transport.check_cancelled(Some(handle))?;
    let cost = transport.request_cost(handle, &plan.cost)?;
    if !cost.accepted
        || cost.approved_microunits == 0
        || cost.approved_microunits > plan.cost.maximum_microunits
        || cost.receipt.is_empty()
        || cost.receipt.len() > 32 * 1024
    {
        return Err(ConformanceError::InvalidCost);
    }
    transport.check_cancelled(Some(handle))?;
    transport.report_progress(handle, &plan.progress)?;

    let mut next_event = 0u64;
    let mut after_sequence = None;
    let mut saw_head = false;
    let mut saw_terminal = false;
    let mut response_bytes = 0usize;
    for _ in 0..limits.maximum_polls {
        transport.check_cancelled(Some(handle))?;
        match transport.wait_response(handle, after_sequence, plan.wait_milliseconds)? {
            ResponseEvent::TimedOut => continue,
            ResponseEvent::Cancelled => return Err(ConformanceError::Cancelled),
            ResponseEvent::Head { status, headers }
                if !saw_head
                    && (200..=599).contains(&status)
                    && validate_headers(&headers, limits).is_ok() =>
            {
                saw_head = true;
                after_sequence = Some(next_event);
                next_event += 1;
            }
            ResponseEvent::Chunk(bytes) if saw_head && !saw_terminal && !bytes.is_empty() => {
                response_bytes = response_bytes
                    .checked_add(bytes.len())
                    .ok_or(ConformanceError::BodyLimit)?;
                if bytes.len() > limits.maximum_chunk_bytes
                    || response_bytes > limits.maximum_body_bytes
                {
                    return Err(ConformanceError::BodyLimit);
                }
                after_sequence = Some(next_event);
                next_event += 1;
            }
            ResponseEvent::Completed(receipt)
                if saw_head && !receipt.is_empty() && receipt.len() <= 32 * 1024 =>
            {
                saw_terminal = true;
                break;
            }
            _ => return Err(ConformanceError::InvalidResponse),
        }
    }
    if !saw_terminal {
        return Err(ConformanceError::PollLimit);
    }

    publish_validated(outputs, sink)?;
    Ok(())
}

fn validate_limits(limits: TransportLimits) -> Result<(), ConformanceError> {
    if limits.maximum_headers == 0
        || limits.maximum_headers > 256
        || limits.maximum_header_bytes == 0
        || limits.maximum_header_bytes > 256 * 1024
        || limits.maximum_body_bytes == 0
        || limits.maximum_body_bytes > 64 * 1024 * 1024
        || limits.maximum_chunk_bytes == 0
        || limits.maximum_chunk_bytes > 1024 * 1024
        || limits.maximum_attempts == 0
        || limits.maximum_attempts > 16
        || limits.maximum_polls == 0
        || limits.maximum_polls > 1024
        || limits.maximum_wait_milliseconds == 0
        || limits.maximum_wait_milliseconds > 60_000
        || limits.maximum_progress_total == 0
        || limits.maximum_progress_total > 1_000_000_000
        || limits.maximum_output_bytes == 0
        || limits.maximum_output_bytes > 64 * 1024 * 1024
    {
        return Err(ConformanceError::InvalidRequest);
    }
    Ok(())
}

fn preflight(plan: &InvocationPlan, limits: TransportLimits) -> Result<Vec<u8>, ConformanceError> {
    if !valid_path(&plan.request.path)
        || plan.request.secret_id.as_deref().is_some_and(|secret_id| {
            secret_id.is_empty()
                || secret_id.len() > 256
                || !secret_id.bytes().all(|byte| {
                    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b':' | b'>')
                })
        })
        || plan.multipart_boundary.is_empty()
        || plan.multipart_boundary.len() > 70
        || !plan
            .multipart_boundary
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        || plan.wait_milliseconds == 0
        || plan.wait_milliseconds > limits.maximum_wait_milliseconds
    {
        return Err(ConformanceError::InvalidRequest);
    }
    validate_headers(&plan.request.headers, limits)?;
    if plan.upload.len() > limits.maximum_body_bytes
        || plan.progress.total == 0
        || plan.progress.completed > plan.progress.total
        || plan.progress.total > limits.maximum_progress_total
        || plan.progress.sequence != 0
        || plan
            .progress
            .message
            .as_ref()
            .is_some_and(|message| message.is_empty() || message.len() > 1_024)
        || plan.cost.maximum_microunits == 0
        || plan.cost.currency.len() != 3
        || !plan
            .cost
            .currency
            .bytes()
            .all(|byte| byte.is_ascii_uppercase())
    {
        return Err(ConformanceError::InvalidRequest);
    }

    MultipartField::encode_checked(&plan.multipart_fields, &plan.multipart_boundary, limits)
}

impl MultipartField {
    pub(crate) fn encode_checked(
        fields: &[Self],
        boundary: &str,
        limits: TransportLimits,
    ) -> Result<Vec<u8>, ConformanceError> {
        validate_limits(limits)?;
        if boundary.is_empty()
            || boundary.len() > 70
            || !boundary
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(ConformanceError::InvalidRequest);
        }
        let mut body = Vec::new();
        body.try_reserve(limits.maximum_body_bytes.min(4096))
            .map_err(|_| ConformanceError::BodyLimit)?;
        for field in fields {
            validate_multipart_field(field)?;
            append_bounded(&mut body, b"--", limits.maximum_body_bytes)?;
            append_bounded(&mut body, boundary.as_bytes(), limits.maximum_body_bytes)?;
            append_bounded(&mut body, b"\r\n", limits.maximum_body_bytes)?;
            if let Some(media_type) = &field.media_type {
                append_bounded(&mut body, b"Content-Type: ", limits.maximum_body_bytes)?;
                append_bounded(&mut body, media_type.as_bytes(), limits.maximum_body_bytes)?;
                append_bounded(&mut body, b"\r\n", limits.maximum_body_bytes)?;
            }
            append_bounded(
                &mut body,
                b"Content-Disposition: form-data; name=\"",
                limits.maximum_body_bytes,
            )?;
            append_bounded(&mut body, field.name.as_bytes(), limits.maximum_body_bytes)?;
            append_bounded(&mut body, b"\"", limits.maximum_body_bytes)?;
            if let Some(file_name) = &field.file_name {
                append_bounded(&mut body, b"; filename=\"", limits.maximum_body_bytes)?;
                append_bounded(&mut body, file_name.as_bytes(), limits.maximum_body_bytes)?;
                append_bounded(&mut body, b"\"", limits.maximum_body_bytes)?;
            }
            append_bounded(&mut body, b"\r\n\r\n", limits.maximum_body_bytes)?;
            append_bounded(&mut body, &field.bytes, limits.maximum_body_bytes)?;
            append_bounded(&mut body, b"\r\n", limits.maximum_body_bytes)?;
        }
        append_bounded(&mut body, b"--", limits.maximum_body_bytes)?;
        append_bounded(&mut body, boundary.as_bytes(), limits.maximum_body_bytes)?;
        append_bounded(&mut body, b"--\r\n", limits.maximum_body_bytes)?;
        if body.is_empty() || body.len() > limits.maximum_body_bytes {
            return Err(ConformanceError::BodyLimit);
        }
        Ok(body)
    }
}

fn validate_headers(headers: &[Header], limits: TransportLimits) -> Result<(), ConformanceError> {
    if headers.len() > limits.maximum_headers {
        return Err(ConformanceError::HeaderLimit);
    }
    let mut header_bytes = 0usize;
    for header in headers {
        if header.name.is_empty()
            || header.name.len() > 256
            || !header.name.bytes().all(valid_header_name_byte)
            || header.value.len() > 8 * 1024
            || header.value.bytes().any(|byte| {
                byte == b'\r' || byte == b'\n' || byte == 0x7f || (byte < 0x20 && byte != b'\t')
            })
        {
            return Err(ConformanceError::InvalidRequest);
        }
        header_bytes = header_bytes
            .checked_add(header.name.len())
            .and_then(|bytes| bytes.checked_add(header.value.len()))
            .ok_or(ConformanceError::HeaderLimit)?;
    }
    if header_bytes > limits.maximum_header_bytes {
        return Err(ConformanceError::HeaderLimit);
    }
    Ok(())
}

fn append_bounded(
    output: &mut Vec<u8>,
    bytes: &[u8],
    maximum: usize,
) -> Result<(), ConformanceError> {
    let next = output
        .len()
        .checked_add(bytes.len())
        .ok_or(ConformanceError::BodyLimit)?;
    if next > maximum {
        return Err(ConformanceError::BodyLimit);
    }
    output
        .try_reserve(bytes.len())
        .map_err(|_| ConformanceError::BodyLimit)?;
    output.extend_from_slice(bytes);
    Ok(())
}

fn validate_multipart_field(field: &MultipartField) -> Result<(), ConformanceError> {
    let valid = |value: &str, array_field: bool| {
        !value.is_empty()
            && value.len() <= 256
            && value.bytes().all(|byte| {
                byte.is_ascii_alphanumeric()
                    || matches!(byte, b'_' | b'-' | b'.')
                    || (array_field && matches!(byte, b'[' | b']'))
            })
    };
    if !valid(&field.name, true)
        || field
            .file_name
            .as_deref()
            .is_some_and(|value| !valid(value, false))
        || field.media_type.as_deref().is_some_and(|value| {
            value.is_empty()
                || value.len() > 256
                || !value.is_ascii()
                || !value.contains('/')
                || value.bytes().any(|byte| byte < 0x20 || byte == 0x7f)
        })
    {
        return Err(ConformanceError::InvalidRequest);
    }
    Ok(())
}

fn valid_path(path: &str) -> bool {
    path.starts_with('/')
        && path.len() <= 2_048
        && !path.contains("//")
        && !path.split('/').any(|segment| matches!(segment, "." | ".."))
        && path.bytes().all(|byte| byte.is_ascii_graphic())
}

fn valid_header_name_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'|'
                | b'~'
        )
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScriptStep {
    CheckCancelled(Option<StreamHandle>),
    Cancel(Option<StreamHandle>),
    Start(RequestHead, Result<StreamHandle, ConformanceError>),
    RequestChunk {
        handle: StreamHandle,
        sequence: u64,
        bytes: Vec<u8>,
        end: bool,
    },
    Upload(StreamHandle, Vec<u8>),
    CreateStorage(StreamHandle, StorageCreateRequest, StorageCreateResponse),
    PutSigned(StreamHandle, SignedPutRequest),
    Cost(StreamHandle, CostRequest, CostResponse),
    Progress(StreamHandle, Progress),
    Wait {
        handle: StreamHandle,
        after_sequence: Option<u64>,
        timeout_milliseconds: u64,
        outcome: ResponseEvent,
    },
}

pub struct ScriptedActuator {
    script: VecDeque<ScriptStep>,
    observed: Vec<ScriptStep>,
}

impl ScriptedActuator {
    pub fn new(script: Vec<ScriptStep>) -> Self {
        Self {
            script: script.into(),
            observed: Vec::new(),
        }
    }

    pub fn observed(&self) -> &[ScriptStep] {
        &self.observed
    }

    pub fn finish(self) -> Result<Vec<ScriptStep>, ConformanceError> {
        if self.script.is_empty() {
            Ok(self.observed)
        } else {
            Err(ConformanceError::HostFailure)
        }
    }

    fn next(&mut self) -> Result<ScriptStep, ConformanceError> {
        self.script.pop_front().ok_or(ConformanceError::HostFailure)
    }
}

impl ProviderTransport for ScriptedActuator {
    fn check_cancelled(&mut self, handle: Option<StreamHandle>) -> Result<(), ConformanceError> {
        match self.next()? {
            ScriptStep::CheckCancelled(expected) if expected == handle => {
                self.observed.push(ScriptStep::CheckCancelled(handle));
                Ok(())
            }
            ScriptStep::Cancel(expected) if expected == handle => Err(ConformanceError::Cancelled),
            other => {
                self.script.push_front(other);
                Err(ConformanceError::InvalidHandle)
            }
        }
    }

    fn start_request(&mut self, head: &RequestHead) -> Result<StreamHandle, ConformanceError> {
        match self.next()? {
            ScriptStep::Start(expected, result) if expected == *head => {
                self.observed
                    .push(ScriptStep::Start(expected, result.clone()));
                result
            }
            _ => Err(ConformanceError::HostFailure),
        }
    }

    fn write_request_chunk(
        &mut self,
        handle: StreamHandle,
        sequence: u64,
        bytes: &[u8],
        end: bool,
    ) -> Result<(), ConformanceError> {
        match self.next()? {
            ScriptStep::RequestChunk {
                handle: expected_handle,
                sequence: expected_sequence,
                bytes: expected_bytes,
                end: expected_end,
            } if expected_handle == handle
                && expected_sequence == sequence
                && expected_bytes == bytes
                && expected_end == end =>
            {
                self.observed.push(ScriptStep::RequestChunk {
                    handle,
                    sequence,
                    bytes: bytes.to_vec(),
                    end,
                });
                Ok(())
            }
            _ => Err(ConformanceError::InvalidHandle),
        }
    }

    fn upload(&mut self, handle: StreamHandle, bytes: &[u8]) -> Result<(), ConformanceError> {
        match self.next()? {
            ScriptStep::Upload(expected_handle, expected)
                if expected_handle == handle && expected == bytes =>
            {
                self.observed.push(ScriptStep::Upload(handle, expected));
                Ok(())
            }
            ScriptStep::Upload(expected_handle, _) if expected_handle != handle => {
                Err(ConformanceError::InvalidHandle)
            }
            _ => Err(ConformanceError::HostFailure),
        }
    }

    fn create_storage(
        &mut self,
        handle: StreamHandle,
        request: &StorageCreateRequest,
    ) -> Result<StorageCreateResponse, ConformanceError> {
        match self.next()? {
            ScriptStep::CreateStorage(expected_handle, expected, response)
                if expected_handle == handle && expected == *request =>
            {
                self.observed.push(ScriptStep::CreateStorage(
                    handle,
                    expected,
                    response.clone(),
                ));
                Ok(response)
            }
            ScriptStep::CreateStorage(expected_handle, _, _) if expected_handle != handle => {
                Err(ConformanceError::InvalidHandle)
            }
            _ => Err(ConformanceError::HostFailure),
        }
    }

    fn put_signed(
        &mut self,
        handle: StreamHandle,
        request: &SignedPutRequest,
    ) -> Result<(), ConformanceError> {
        match self.next()? {
            ScriptStep::PutSigned(expected_handle, expected)
                if expected_handle == handle && expected == *request =>
            {
                self.observed.push(ScriptStep::PutSigned(handle, expected));
                Ok(())
            }
            ScriptStep::PutSigned(expected_handle, _) if expected_handle != handle => {
                Err(ConformanceError::InvalidHandle)
            }
            _ => Err(ConformanceError::HostFailure),
        }
    }

    fn request_cost(
        &mut self,
        handle: StreamHandle,
        request: &CostRequest,
    ) -> Result<CostResponse, ConformanceError> {
        match self.next()? {
            ScriptStep::Cost(expected_handle, expected, response)
                if expected_handle == handle && expected == *request =>
            {
                self.observed
                    .push(ScriptStep::Cost(handle, expected, response.clone()));
                Ok(response)
            }
            ScriptStep::Cost(expected_handle, _, _) if expected_handle != handle => {
                Err(ConformanceError::InvalidHandle)
            }
            _ => Err(ConformanceError::HostFailure),
        }
    }

    fn report_progress(
        &mut self,
        handle: StreamHandle,
        progress: &Progress,
    ) -> Result<(), ConformanceError> {
        match self.next()? {
            ScriptStep::Progress(expected_handle, expected)
                if expected_handle == handle && expected == *progress =>
            {
                self.observed.push(ScriptStep::Progress(handle, expected));
                Ok(())
            }
            ScriptStep::Progress(expected_handle, _) if expected_handle != handle => {
                Err(ConformanceError::InvalidHandle)
            }
            _ => Err(ConformanceError::HostFailure),
        }
    }

    fn wait_response(
        &mut self,
        handle: StreamHandle,
        after_sequence: Option<u64>,
        timeout_milliseconds: u64,
    ) -> Result<ResponseEvent, ConformanceError> {
        match self.next()? {
            ScriptStep::Wait {
                handle: expected_handle,
                after_sequence: expected_after,
                timeout_milliseconds: expected_timeout,
                outcome,
            } if expected_handle == handle
                && expected_after == after_sequence
                && expected_timeout == timeout_milliseconds =>
            {
                self.observed.push(ScriptStep::Wait {
                    handle,
                    after_sequence,
                    timeout_milliseconds,
                    outcome: outcome.clone(),
                });
                Ok(outcome)
            }
            ScriptStep::Wait {
                handle: expected_handle,
                ..
            } if expected_handle != handle => Err(ConformanceError::InvalidHandle),
            _ => Err(ConformanceError::HostFailure),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urlencoded_matches_actual_aiohttp_string_forms() -> Result<(), Box<dyn Error>> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../sim_test_support/fixtures/providers/common/source_multipart.json"
        ))?;
        let cases = fixture["urlencoded"]["cases"]
            .as_array()
            .ok_or("source cases missing")?;
        assert_eq!(cases.len(), 8);
        for case in cases {
            let fields = case["fields"]
                .as_array()
                .ok_or("source fields missing")?
                .iter()
                .map(|field| {
                    Ok(UrlEncodedField {
                        name: field["name"].as_str().ok_or("source name missing")?.into(),
                        value: field["value"]
                            .as_str()
                            .ok_or("source value missing")?
                            .into(),
                    })
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
            let expected = case["body_utf8"]
                .as_str()
                .ok_or("source body missing")?
                .as_bytes();
            assert_eq!(
                UrlEncodedField::encode_checked(&fields, TransportLimits::conformance())?,
                expected
            );
            if !expected.is_empty() {
                let mut limits = TransportLimits::conformance();
                limits.maximum_body_bytes = expected.len();
                assert_eq!(UrlEncodedField::encode_checked(&fields, limits)?, expected);
                limits.maximum_body_bytes =
                    expected.len().checked_sub(1).ok_or("empty source body")?;
                assert_eq!(
                    UrlEncodedField::encode_checked(&fields, limits),
                    Err(ConformanceError::BodyLimit)
                );
            }
        }
        let mut invalid = TransportLimits::conformance();
        invalid.maximum_body_bytes = 0;
        assert_eq!(
            UrlEncodedField::encode_checked(&[], invalid),
            Err(ConformanceError::InvalidRequest)
        );
        Ok(())
    }

    #[test]
    fn multipart_matches_actual_aiohttp_fields_and_empty_values() -> Result<(), Box<dyn Error>> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../sim_test_support/fixtures/providers/common/source_multipart.json"
        ))?;
        let boundary = fixture["boundary"]
            .as_str()
            .ok_or("source boundary missing")?;
        for case in fixture["cases"].as_array().ok_or("source cases missing")? {
            let fields = case["fields"]
                .as_array()
                .ok_or("source fields missing")?
                .iter()
                .map(|field| {
                    Ok(MultipartField {
                        name: field["name"].as_str().ok_or("source name missing")?.into(),
                        file_name: field["file_name"].as_str().map(str::to_owned),
                        media_type: field["media_type"].as_str().map(str::to_owned),
                        bytes: decode_hex(
                            field["bytes_hex"]
                                .as_str()
                                .ok_or("source field bytes missing")?,
                        )?,
                    })
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
            let expected = decode_hex(case["body_hex"].as_str().ok_or("source body missing")?)?;
            assert_eq!(
                MultipartField::encode_checked(&fields, boundary, TransportLimits::conformance())?,
                expected
            );
            let mut limits = TransportLimits::conformance();
            limits.maximum_body_bytes = expected.len();
            assert_eq!(
                MultipartField::encode_checked(&fields, boundary, limits)?,
                expected
            );
            limits.maximum_body_bytes = expected.len().checked_sub(1).ok_or("empty source body")?;
            assert_eq!(
                MultipartField::encode_checked(&fields, boundary, limits),
                Err(ConformanceError::BodyLimit)
            );
        }
        Ok(())
    }

    #[test]
    fn storage_upload_matches_actual_source_mime_omission() -> Result<(), Box<dyn Error>> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../sim_test_support/fixtures/providers/common/source_storage_upload.json"
        ))?;
        for case in fixture["cases"]
            .as_array()
            .ok_or("source upload cases missing")?
        {
            let bytes = decode_hex(
                case["bytes_hex"]
                    .as_str()
                    .ok_or("source upload bytes missing")?,
            )?;
            let content_type = case["declared_content_type"].as_str().map(str::to_owned);
            let storage = StorageCreateRequest {
                file_name: case["file_name"]
                    .as_str()
                    .ok_or("source filename missing")?
                    .into(),
                content_type: content_type.clone(),
                byte_length: u64::try_from(bytes.len())?,
            };
            let actual: serde_json::Value =
                serde_json::from_slice(&storage.checked_body(TransportLimits::conformance())?)?;
            assert_eq!(actual, case["storage"]["body"]);
            let signed = SignedPutRequest {
                upload_url: "https://fixture.invalid/signed".into(),
                content_type,
                bytes,
            };
            let headers = signed.checked_headers(TransportLimits::conformance())?;
            let actual_headers = headers
                .into_iter()
                .map(|header| (header.name, serde_json::Value::String(header.value)))
                .collect::<serde_json::Map<_, _>>();
            assert_eq!(
                serde_json::Value::Object(actual_headers),
                case["signed_put_headers"]
            );
            assert_eq!(
                signed.suppress_automatic_content_type(),
                case["skip_auto_headers"]
                    .as_array()
                    .ok_or("source header suppression missing")?
                    .iter()
                    .any(|name| name == "Content-Type")
            );
        }
        Ok(())
    }

    #[test]
    fn storage_upload_rejects_invalid_mime_and_payload_bounds() {
        for content_type in ["", "image/png\r\nInjected: value", "image/\0png"] {
            let request = SignedPutRequest {
                upload_url: "https://fixture.invalid/signed".into(),
                content_type: Some(content_type.into()),
                bytes: vec![1],
            };
            assert_eq!(
                request.checked_headers(TransportLimits::conformance()),
                Err(ConformanceError::InvalidRequest)
            );
        }
        let mut limits = TransportLimits::conformance();
        limits.maximum_body_bytes = 1;
        let request = SignedPutRequest {
            upload_url: "https://fixture.invalid/signed".into(),
            content_type: None,
            bytes: vec![1, 2],
        };
        assert_eq!(
            request.checked_headers(limits),
            Err(ConformanceError::BodyLimit)
        );
    }

    #[test]
    fn multipart_retains_repeated_source_image_array_names() -> Result<(), ConformanceError> {
        let fields = ['a', 'b'].map(|value| MultipartField {
            name: "image[]".into(),
            file_name: Some("image.png".into()),
            media_type: Some("image/png".into()),
            bytes: vec![value as u8],
        });
        let encoded =
            MultipartField::encode_checked(&fields, "boundary", TransportLimits::conformance())?;
        let expected = b"--boundary\r\nContent-Type: image/png\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"image.png\"\r\n\r\na\r\n--boundary\r\nContent-Type: image/png\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"image.png\"\r\n\r\nb\r\n--boundary--\r\n";
        assert_eq!(encoded, expected);
        for (name, file_name) in [
            ("image\r\nInjected: value", "image.png"),
            ("image\"", "image.png"),
            ("image[]", "image[].png"),
            ("image[]", "image\".png"),
        ] {
            let field = MultipartField {
                name: name.into(),
                file_name: Some(file_name.into()),
                media_type: Some("image/png".into()),
                bytes: vec![1],
            };
            assert_eq!(
                MultipartField::encode_checked(
                    &[field],
                    "boundary",
                    TransportLimits::conformance()
                ),
                Err(ConformanceError::InvalidRequest)
            );
        }
        Ok(())
    }

    #[test]
    fn multipart_rejects_header_injection_before_encoding() {
        for media_type in [
            "image/png\r\nInjected: value",
            "image/\0png",
            "image/\u{7f}png",
        ] {
            let field = MultipartField {
                name: "file".into(),
                file_name: Some("image.png".into()),
                media_type: Some(media_type.into()),
                bytes: vec![1],
            };
            assert_eq!(
                MultipartField::encode_checked(
                    &[field],
                    "boundary",
                    TransportLimits::conformance()
                ),
                Err(ConformanceError::InvalidRequest)
            );
        }
    }

    fn decode_hex(value: &str) -> Result<Vec<u8>, Box<dyn Error>> {
        if !value.len().is_multiple_of(2) || !value.is_ascii() {
            return Err("invalid source hex".into());
        }
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let encoded = std::str::from_utf8(pair)?;
                Ok(u8::from_str_radix(encoded, 16)?)
            })
            .collect()
    }
}
