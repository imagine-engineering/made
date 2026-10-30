use std::{collections::BTreeSet, fmt};

pub const MAXIMUM_FIXTURE_BYTES: usize = 256 * 1024;
pub const MAXIMUM_LINE_BYTES: usize = 64 * 1024;
pub const MAXIMUM_HEADERS: usize = 64;
pub const MAXIMUM_BODY_BYTES: usize = 2 * 1024 * 1024;
pub const MAXIMUM_CHUNKS: usize = 256;
pub const MAXIMUM_MULTIPART_PARTS: usize = 64;
pub const MAXIMUM_PROGRESS_UPDATES: usize = 256;
pub const MAXIMUM_MATERIALIZATIONS: usize = 64;
pub const MAXIMUM_RETRIES: u32 = 16;
pub const MAXIMUM_POLLS: u32 = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HttpMethod {
    Delete,
    Get,
    Patch,
    Post,
    Put,
}

impl HttpMethod {
    fn parse(value: &str) -> Result<Self, FixtureError> {
        match value {
            "DELETE" => Ok(Self::Delete),
            "GET" => Ok(Self::Get),
            "PATCH" => Ok(Self::Patch),
            "POST" => Ok(Self::Post),
            "PUT" => Ok(Self::Put),
            _ => Err(FixtureError::InvalidValue {
                field: "method",
                value: value.to_string(),
            }),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Header {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultipartPart {
    pub name: String,
    pub file_name: Option<String>,
    pub content_type: Option<String>,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgressUpdate {
    pub completed: u64,
    pub total: u64,
    pub message: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterializationKind {
    Binary,
    Image,
    Json,
    Text,
}

impl MaterializationKind {
    fn parse(value: &str) -> Result<Self, FixtureError> {
        match value {
            "binary" => Ok(Self::Binary),
            "image" => Ok(Self::Image),
            "json" => Ok(Self::Json),
            "text" => Ok(Self::Text),
            _ => Err(FixtureError::InvalidValue {
                field: "materialization kind",
                value: value.to_string(),
            }),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Materialization {
    pub port: String,
    pub kind: MaterializationKind,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TerminalDisposition {
    Success,
    Cancelled,
    WorkerLost,
    ProviderError(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderCase {
    pub provider: String,
    pub name: String,
    pub method: HttpMethod,
    pub path: String,
    pub headers: Vec<Header>,
    pub body: Vec<u8>,
    pub request_chunks: Vec<Vec<u8>>,
    pub response_chunks: Vec<Vec<u8>>,
    pub multipart: Vec<MultipartPart>,
    pub retry_statuses: Vec<u16>,
    pub retry_limit: u32,
    pub poll_states: Vec<String>,
    pub poll_limit: u32,
    pub cost_microunits: u64,
    pub progress: Vec<ProgressUpdate>,
    pub materializations: Vec<Materialization>,
    pub terminal: TerminalDisposition,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FixtureError {
    OversizedFixture { actual: usize, maximum: usize },
    InvalidUtf8,
    OversizedLine { line: usize, actual: usize },
    MalformedLine { line: usize },
    UnknownField { line: usize, field: String },
    DuplicateField(&'static str),
    MissingField(&'static str),
    InvalidValue { field: &'static str, value: String },
    InvalidHex { field: &'static str },
    LimitExceeded { field: &'static str, maximum: usize },
}

impl fmt::Display for FixtureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for FixtureError {}

pub fn parse_fixture(bytes: &[u8]) -> Result<ProviderCase, FixtureError> {
    if bytes.len() > MAXIMUM_FIXTURE_BYTES {
        return Err(FixtureError::OversizedFixture {
            actual: bytes.len(),
            maximum: MAXIMUM_FIXTURE_BYTES,
        });
    }
    let text = std::str::from_utf8(bytes).map_err(|_| FixtureError::InvalidUtf8)?;
    let mut builder = ProviderCaseBuilder::default();
    for (line_index, raw_line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        if raw_line.len() > MAXIMUM_LINE_BYTES {
            return Err(FixtureError::OversizedLine {
                line: line_number,
                actual: raw_line.len(),
            });
        }
        if raw_line.is_empty() || raw_line.starts_with('#') {
            continue;
        }
        let (field, value) = raw_line
            .split_once('=')
            .ok_or(FixtureError::MalformedLine { line: line_number })?;
        builder.push(field, value, line_number)?;
    }
    builder.finish()
}

#[derive(Default)]
struct ProviderCaseBuilder {
    singular_fields: BTreeSet<&'static str>,
    schema: Option<u32>,
    provider: Option<String>,
    name: Option<String>,
    method: Option<HttpMethod>,
    path: Option<String>,
    headers: Vec<Header>,
    body: Option<Vec<u8>>,
    request_chunks: Vec<Vec<u8>>,
    response_chunks: Vec<Vec<u8>>,
    multipart: Vec<MultipartPart>,
    retry_statuses: Vec<u16>,
    retry_limit: Option<u32>,
    poll_states: Vec<String>,
    poll_limit: Option<u32>,
    cost_microunits: Option<u64>,
    progress: Vec<ProgressUpdate>,
    materializations: Vec<Materialization>,
    terminal: Option<TerminalDisposition>,
}

impl ProviderCaseBuilder {
    fn push(&mut self, field: &str, value: &str, line: usize) -> Result<(), FixtureError> {
        match field {
            "schema" => {
                self.set_once("schema")?;
                self.schema = Some(parse_number("schema", value)?);
            }
            "provider" => {
                self.set_once("provider")?;
                validate_identifier("provider", value)?;
                self.provider = Some(value.to_string());
            }
            "case" => {
                self.set_once("case")?;
                validate_identifier("case", value)?;
                self.name = Some(value.to_string());
            }
            "method" => {
                self.set_once("method")?;
                self.method = Some(HttpMethod::parse(value)?);
            }
            "path" => {
                self.set_once("path")?;
                if !value.starts_with('/') || contains_control(value) {
                    return Err(FixtureError::InvalidValue {
                        field: "path",
                        value: value.to_string(),
                    });
                }
                self.path = Some(value.to_string());
            }
            "header" => {
                push_bounded(
                    &mut self.headers,
                    MAXIMUM_HEADERS,
                    "headers",
                    parse_header(value)?,
                )?;
            }
            "body_hex" => {
                self.set_once("body_hex")?;
                self.body = Some(parse_hex_bounded("body_hex", value, MAXIMUM_BODY_BYTES)?);
            }
            "request_chunk_hex" => push_bounded(
                &mut self.request_chunks,
                MAXIMUM_CHUNKS,
                "request chunks",
                parse_hex_bounded("request_chunk_hex", value, MAXIMUM_BODY_BYTES)?,
            )?,
            "response_chunk_hex" => push_bounded(
                &mut self.response_chunks,
                MAXIMUM_CHUNKS,
                "response chunks",
                parse_hex_bounded("response_chunk_hex", value, MAXIMUM_BODY_BYTES)?,
            )?,
            "multipart" => push_bounded(
                &mut self.multipart,
                MAXIMUM_MULTIPART_PARTS,
                "multipart parts",
                parse_multipart(value)?,
            )?,
            "retry_status" => self
                .retry_statuses
                .push(parse_number("retry_status", value)?),
            "retry_limit" => {
                self.set_once("retry_limit")?;
                self.retry_limit = Some(parse_number("retry_limit", value)?);
            }
            "poll_state" => {
                if value.is_empty() || contains_control(value) {
                    return Err(FixtureError::InvalidValue {
                        field: "poll_state",
                        value: value.to_string(),
                    });
                }
                self.poll_states.push(value.to_string());
            }
            "poll_limit" => {
                self.set_once("poll_limit")?;
                self.poll_limit = Some(parse_number("poll_limit", value)?);
            }
            "cost_microunits" => {
                self.set_once("cost_microunits")?;
                self.cost_microunits = Some(parse_number("cost_microunits", value)?);
            }
            "progress" => push_bounded(
                &mut self.progress,
                MAXIMUM_PROGRESS_UPDATES,
                "progress updates",
                parse_progress(value)?,
            )?,
            "materialization" => push_bounded(
                &mut self.materializations,
                MAXIMUM_MATERIALIZATIONS,
                "materializations",
                parse_materialization(value)?,
            )?,
            "terminal" => {
                self.set_once("terminal")?;
                self.terminal = Some(parse_terminal(value)?);
            }
            _ => {
                return Err(FixtureError::UnknownField {
                    line,
                    field: field.to_string(),
                });
            }
        }
        Ok(())
    }

    fn set_once(&mut self, field: &'static str) -> Result<(), FixtureError> {
        if !self.singular_fields.insert(field) {
            return Err(FixtureError::DuplicateField(field));
        }
        Ok(())
    }

    fn finish(self) -> Result<ProviderCase, FixtureError> {
        if self.schema != Some(1) {
            return Err(FixtureError::InvalidValue {
                field: "schema",
                value: self
                    .schema
                    .map_or_else(|| "missing".to_string(), |value| value.to_string()),
            });
        }
        let retry_limit = self
            .retry_limit
            .ok_or(FixtureError::MissingField("retry_limit"))?;
        let poll_limit = self
            .poll_limit
            .ok_or(FixtureError::MissingField("poll_limit"))?;
        if retry_limit > MAXIMUM_RETRIES || self.retry_statuses.len() > retry_limit as usize {
            return Err(FixtureError::LimitExceeded {
                field: "retries",
                maximum: MAXIMUM_RETRIES as usize,
            });
        }
        if poll_limit > MAXIMUM_POLLS || self.poll_states.len() > poll_limit as usize {
            return Err(FixtureError::LimitExceeded {
                field: "polls",
                maximum: MAXIMUM_POLLS as usize,
            });
        }
        validate_progress(&self.progress)?;
        validate_materializations(&self.materializations)?;
        let terminal = self
            .terminal
            .ok_or(FixtureError::MissingField("terminal"))?;
        if !matches!(terminal, TerminalDisposition::Success) && !self.materializations.is_empty() {
            return Err(FixtureError::InvalidValue {
                field: "materialization",
                value: "non-success case may not publish materializations".to_string(),
            });
        }
        Ok(ProviderCase {
            provider: self
                .provider
                .ok_or(FixtureError::MissingField("provider"))?,
            name: self.name.ok_or(FixtureError::MissingField("case"))?,
            method: self.method.ok_or(FixtureError::MissingField("method"))?,
            path: self.path.ok_or(FixtureError::MissingField("path"))?,
            headers: self.headers,
            body: self.body.ok_or(FixtureError::MissingField("body_hex"))?,
            request_chunks: self.request_chunks,
            response_chunks: self.response_chunks,
            multipart: self.multipart,
            retry_statuses: self.retry_statuses,
            retry_limit,
            poll_states: self.poll_states,
            poll_limit,
            cost_microunits: self
                .cost_microunits
                .ok_or(FixtureError::MissingField("cost_microunits"))?,
            progress: self.progress,
            materializations: self.materializations,
            terminal,
        })
    }
}

fn parse_header(value: &str) -> Result<Header, FixtureError> {
    let (name, value) = value
        .split_once(':')
        .ok_or_else(|| FixtureError::InvalidValue {
            field: "header",
            value: value.to_string(),
        })?;
    if name.is_empty()
        || !name.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_')
        })
        || contains_control(value)
    {
        return Err(FixtureError::InvalidValue {
            field: "header",
            value: format!("{name}:{value}"),
        });
    }
    Ok(Header {
        name: name.to_string(),
        value: value.to_string(),
    })
}

fn parse_multipart(value: &str) -> Result<MultipartPart, FixtureError> {
    let fields = value.splitn(4, '|').collect::<Vec<_>>();
    let [name, file_name, content_type, bytes] = fields.as_slice() else {
        return Err(FixtureError::InvalidValue {
            field: "multipart",
            value: value.to_string(),
        });
    };
    validate_identifier("multipart name", name)?;
    Ok(MultipartPart {
        name: (*name).to_string(),
        file_name: nonempty_clean(file_name, "multipart file name")?,
        content_type: nonempty_clean(content_type, "multipart content type")?,
        bytes: parse_hex_bounded("multipart", bytes, MAXIMUM_BODY_BYTES)?,
    })
}

fn parse_progress(value: &str) -> Result<ProgressUpdate, FixtureError> {
    let (counts, message) = value
        .split_once('|')
        .ok_or_else(|| FixtureError::InvalidValue {
            field: "progress",
            value: value.to_string(),
        })?;
    let (completed, total) = counts
        .split_once('/')
        .ok_or_else(|| FixtureError::InvalidValue {
            field: "progress",
            value: value.to_string(),
        })?;
    if message.is_empty() || contains_control(message) {
        return Err(FixtureError::InvalidValue {
            field: "progress",
            value: value.to_string(),
        });
    }
    Ok(ProgressUpdate {
        completed: parse_number("progress completed", completed)?,
        total: parse_number("progress total", total)?,
        message: message.to_string(),
    })
}

fn parse_materialization(value: &str) -> Result<Materialization, FixtureError> {
    let fields = value.splitn(3, '|').collect::<Vec<_>>();
    let [port, kind, bytes] = fields.as_slice() else {
        return Err(FixtureError::InvalidValue {
            field: "materialization",
            value: value.to_string(),
        });
    };
    validate_identifier("materialization port", port)?;
    Ok(Materialization {
        port: (*port).to_string(),
        kind: MaterializationKind::parse(kind)?,
        bytes: parse_hex_bounded("materialization", bytes, MAXIMUM_BODY_BYTES)?,
    })
}

fn parse_terminal(value: &str) -> Result<TerminalDisposition, FixtureError> {
    match value {
        "success" => Ok(TerminalDisposition::Success),
        "cancelled" => Ok(TerminalDisposition::Cancelled),
        "worker_lost" => Ok(TerminalDisposition::WorkerLost),
        _ => value
            .strip_prefix("provider_error:")
            .filter(|code| !code.is_empty() && !contains_control(code))
            .map(|code| TerminalDisposition::ProviderError(code.to_string()))
            .ok_or_else(|| FixtureError::InvalidValue {
                field: "terminal",
                value: value.to_string(),
            }),
    }
}

fn validate_progress(progress: &[ProgressUpdate]) -> Result<(), FixtureError> {
    let mut previous = 0_u64;
    for update in progress {
        if update.total == 0 || update.completed > update.total || update.completed < previous {
            return Err(FixtureError::InvalidValue {
                field: "progress",
                value: format!("{}/{}", update.completed, update.total),
            });
        }
        previous = update.completed;
    }
    Ok(())
}

fn validate_materializations(materializations: &[Materialization]) -> Result<(), FixtureError> {
    let mut ports = BTreeSet::new();
    for materialization in materializations {
        if !ports.insert(materialization.port.as_str()) {
            return Err(FixtureError::InvalidValue {
                field: "materialization port",
                value: materialization.port.clone(),
            });
        }
        match materialization.kind {
            MaterializationKind::Json => {
                let text = std::str::from_utf8(&materialization.bytes).map_err(|_| {
                    FixtureError::InvalidValue {
                        field: "json materialization",
                        value: "non-UTF-8".to_string(),
                    }
                })?;
                if !(text.starts_with('{') && text.ends_with('}')) {
                    return Err(FixtureError::InvalidValue {
                        field: "json materialization",
                        value: text.to_string(),
                    });
                }
            }
            MaterializationKind::Text => {
                std::str::from_utf8(&materialization.bytes).map_err(|_| {
                    FixtureError::InvalidValue {
                        field: "text materialization",
                        value: "non-UTF-8".to_string(),
                    }
                })?;
            }
            MaterializationKind::Binary | MaterializationKind::Image => {}
        }
    }
    Ok(())
}

fn parse_hex_bounded(
    field: &'static str,
    value: &str,
    maximum: usize,
) -> Result<Vec<u8>, FixtureError> {
    if value.len() % 2 != 0 || value.len() / 2 > maximum {
        return Err(FixtureError::InvalidHex { field });
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|digits| {
            let high = hex_digit(digits[0]).ok_or(FixtureError::InvalidHex { field })?;
            let low = hex_digit(digits[1]).ok_or(FixtureError::InvalidHex { field })?;
            Ok(high << 4 | low)
        })
        .collect()
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn parse_number<T>(field: &'static str, value: &str) -> Result<T, FixtureError>
where
    T: std::str::FromStr,
{
    value.parse().map_err(|_| FixtureError::InvalidValue {
        field,
        value: value.to_string(),
    })
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), FixtureError> {
    if value.is_empty()
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-' | b'.')
        })
    {
        return Err(FixtureError::InvalidValue {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn nonempty_clean(value: &str, field: &'static str) -> Result<Option<String>, FixtureError> {
    if value.is_empty() {
        return Ok(None);
    }
    if contains_control(value) || value.contains('/') || value.contains('\\') {
        return Err(FixtureError::InvalidValue {
            field,
            value: value.to_string(),
        });
    }
    Ok(Some(value.to_string()))
}

fn contains_control(value: &str) -> bool {
    value.chars().any(char::is_control)
}

fn push_bounded<T>(
    values: &mut Vec<T>,
    maximum: usize,
    field: &'static str,
    value: T,
) -> Result<(), FixtureError> {
    if values.len() == maximum {
        return Err(FixtureError::LimitExceeded { field, maximum });
    }
    values.push(value);
    Ok(())
}
