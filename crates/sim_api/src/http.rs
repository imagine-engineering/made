use async_channel::{Receiver, Sender};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sim_runtime::{AssetIdentity, AssetNamespace};
use sim_types::{HttpMethod, MAX_COMPATIBILITY_JSON_BYTES, RouteContract, RouteIdentity};
#[cfg(test)]
use std::sync::Mutex;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{
        Arc, OnceLock, RwLock,
        atomic::{AtomicUsize, Ordering},
    },
};
use thiserror::Error;

const ROUTE_CATALOG_CSV: &str =
    include_str!("../../../crates/sim_test_support/parity/catalogs/backend-http-routes.csv");

pub const HTTP_ROUTE_COUNT: usize = 141;
pub const HTTP_FORWARDING_SUPPORTED: bool = false;
pub const DEFAULT_MAX_UPLOAD_BYTES: usize = 100 * 1024 * 1024;
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 128 * 1024 * 1024;
pub const DEFAULT_MAX_RANGE_BYTES: usize = 16 * 1024 * 1024;
pub const DEFAULT_STREAM_CHUNK_BYTES: usize = 256 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RouteAvailability {
    Active,
    Conditional,
    Experimental,
    DeveloperOnly,
    Uncertain,
}

impl RouteAvailability {
    fn parse(value: &str) -> Result<Self, CatalogError> {
        match value {
            "active" => Ok(Self::Active),
            "conditional" => Ok(Self::Conditional),
            "experimental" => Ok(Self::Experimental),
            "developer-only" => Ok(Self::DeveloperOnly),
            "uncertain" => Ok(Self::Uncertain),
            other => Err(CatalogError::InvalidAvailability(other.to_owned())),
        }
    }

    fn is_active(self) -> bool {
        self == Self::Active
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HttpRouteDescriptor {
    pub contract: RouteContract,
    pub availability: RouteAvailability,
    pub classification: String,
    pub evidence_level: String,
    pub summary: String,
    pub error_behavior: String,
    pub schema_confidence: String,
    pub unresolved_schema: String,
}

impl HttpRouteDescriptor {
    pub fn feature_id(&self) -> &str {
        &self.contract.feature_id
    }

    pub fn is_mutation(&self) -> bool {
        !matches!(
            self.contract.identity.method,
            HttpMethod::Get | HttpMethod::Head
        )
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CatalogError {
    #[error("HTTP route catalog contains an unterminated quoted field")]
    UnterminatedQuotedField,
    #[error("HTTP route catalog has no header row")]
    MissingHeader,
    #[error("HTTP route catalog row {row} has {actual} columns; expected {expected}")]
    ColumnCount {
        row: usize,
        expected: usize,
        actual: usize,
    },
    #[error("HTTP route catalog is missing the {0} column")]
    MissingColumn(String),
    #[error("HTTP route catalog contains unsupported method {0}")]
    InvalidMethod(String),
    #[error("HTTP route catalog contains unsupported availability {0}")]
    InvalidAvailability(String),
    #[error("HTTP route catalog contains {actual} routes; expected {expected}")]
    RouteCount { expected: usize, actual: usize },
    #[error("HTTP route catalog contains duplicate feature ID {0}")]
    DuplicateFeatureId(String),
}

static ROUTE_CATALOG: OnceLock<Result<Vec<HttpRouteDescriptor>, CatalogError>> = OnceLock::new();

pub fn http_route_catalog() -> Result<&'static [HttpRouteDescriptor], CatalogError> {
    match ROUTE_CATALOG.get_or_init(build_route_catalog) {
        Ok(routes) => Ok(routes),
        Err(error) => Err(error.clone()),
    }
}

fn build_route_catalog() -> Result<Vec<HttpRouteDescriptor>, CatalogError> {
    let rows = parse_csv(ROUTE_CATALOG_CSV)?;
    let (header, data) = rows.split_first().ok_or(CatalogError::MissingHeader)?;
    let columns = header
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let required = [
        "method",
        "path",
        "canonical_path",
        "classification",
        "availability",
        "evidence_level",
        "summary",
        "request_body",
        "path_parameters",
        "query_parameters",
        "success_behavior",
        "error_behavior",
        "permissions_flags",
        "side_effects",
        "alias_of",
        "feature_id",
        "request_schema_detail",
        "response_schema_detail",
        "status_content_types",
        "schema_confidence",
        "unresolved_schema",
    ];
    for name in required {
        if !columns.contains_key(name) {
            return Err(CatalogError::MissingColumn(name.to_owned()));
        }
    }

    let mut routes = Vec::with_capacity(data.len());
    let mut feature_ids = BTreeMap::new();
    for (offset, row) in data.iter().enumerate() {
        if row.len() != header.len() {
            return Err(CatalogError::ColumnCount {
                row: offset + 2,
                expected: header.len(),
                actual: row.len(),
            });
        }
        let field = |name: &str| -> Result<&str, CatalogError> {
            let index = columns
                .get(name)
                .copied()
                .ok_or_else(|| CatalogError::MissingColumn(name.to_owned()))?;
            row.get(index)
                .map(String::as_str)
                .ok_or_else(|| CatalogError::MissingColumn(name.to_owned()))
        };
        let feature_id = field("feature_id")?.to_owned();
        if feature_ids.insert(feature_id.clone(), ()).is_some() {
            return Err(CatalogError::DuplicateFeatureId(feature_id));
        }
        let method = parse_method(field("method")?)?;
        let path = field("path")?.to_owned();
        let canonical_path = field("canonical_path")?.to_owned();
        let request_body = field("request_body")?;
        let request_detail = field("request_schema_detail")?;
        let response_detail = field("response_schema_detail")?;
        let status_content_types = field("status_content_types")?;
        let permissions = split_catalog_list(field("permissions_flags")?);
        let side_effects = split_catalog_list(field("side_effects")?);
        let availability = RouteAvailability::parse(field("availability")?)?;
        let mut unknown = BTreeMap::new();
        unknown.insert("availability".to_owned(), json!(field("availability")?));
        unknown.insert("request_body".to_owned(), json!(request_body));
        unknown.insert(
            "success_behavior".to_owned(),
            json!(field("success_behavior")?),
        );
        unknown.insert("error_behavior".to_owned(), json!(field("error_behavior")?));
        unknown.insert("request_schema_detail".to_owned(), json!(request_detail));
        unknown.insert("response_schema_detail".to_owned(), json!(response_detail));
        unknown.insert(
            "status_content_types".to_owned(),
            json!(status_content_types),
        );
        unknown.insert(
            "schema_confidence".to_owned(),
            json!(field("schema_confidence")?),
        );
        unknown.insert(
            "unresolved_schema".to_owned(),
            json!(field("unresolved_schema")?),
        );

        let path_parameters = merged_path_parameters(&path, field("path_parameters")?);
        let query_parameters = split_pipe_list(field("query_parameters")?);
        let request_headers =
            modeled_headers(method, &path, request_body, field("permissions_flags")?);
        let feature_gates = modeled_feature_gates(
            availability,
            field("permissions_flags")?,
            field("classification")?,
        );
        let request_schema = (request_body != "none" && !request_body.is_empty()).then(|| {
            json!({
                "body": request_body,
                "detail": request_detail,
                "unknown_fields": "preserved"
            })
        });
        let response_schema = Some(json!({
            "detail": response_detail,
            "status_and_content_types": status_content_types,
            "unknown_fields": "preserved"
        }));
        let content_types =
            modeled_content_types(request_body, response_detail, status_content_types);
        let status_codes = modeled_status_codes(status_content_types, response_detail);
        let success_behavior = field("success_behavior")?.to_ascii_lowercase();
        let streaming = success_behavior.contains("stream")
            || response_detail
                .to_ascii_lowercase()
                .contains("streamresponse");

        routes.push(HttpRouteDescriptor {
            contract: RouteContract {
                feature_id,
                identity: RouteIdentity {
                    method,
                    path,
                    canonical_path,
                    alias_of: nonempty(field("alias_of")?),
                },
                path_parameters,
                query_parameters,
                request_headers,
                request_schema,
                response_schema,
                content_types,
                status_codes,
                streaming,
                permissions,
                feature_gates,
                side_effects,
                unknown,
            },
            availability,
            classification: field("classification")?.to_owned(),
            evidence_level: field("evidence_level")?.to_owned(),
            summary: field("summary")?.to_owned(),
            error_behavior: field("error_behavior")?.to_owned(),
            schema_confidence: field("schema_confidence")?.to_owned(),
            unresolved_schema: field("unresolved_schema")?.to_owned(),
        });
    }
    if routes.len() != HTTP_ROUTE_COUNT {
        return Err(CatalogError::RouteCount {
            expected: HTTP_ROUTE_COUNT,
            actual: routes.len(),
        });
    }
    Ok(routes)
}

pub(crate) fn parse_csv(input: &str) -> Result<Vec<Vec<String>>, CatalogError> {
    let bytes = input.as_bytes();
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = Vec::new();
    let mut quoted = false;
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if quoted {
            if byte == b'"' {
                if bytes.get(index + 1) == Some(&b'"') {
                    field.push(b'"');
                    index += 2;
                    continue;
                }
                quoted = false;
            } else {
                field.push(byte);
            }
        } else {
            match byte {
                b'"' if field.is_empty() => quoted = true,
                b',' => {
                    row.push(String::from_utf8_lossy(&field).into_owned());
                    field.clear();
                }
                b'\n' => {
                    row.push(String::from_utf8_lossy(&field).into_owned());
                    field.clear();
                    rows.push(std::mem::take(&mut row));
                }
                b'\r' if bytes.get(index + 1) == Some(&b'\n') => {}
                _ => field.push(byte),
            }
        }
        index += 1;
    }
    if quoted {
        return Err(CatalogError::UnterminatedQuotedField);
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(String::from_utf8_lossy(&field).into_owned());
        rows.push(row);
    }
    Ok(rows)
}

fn parse_method(method: &str) -> Result<HttpMethod, CatalogError> {
    match method {
        "GET" => Ok(HttpMethod::Get),
        "POST" => Ok(HttpMethod::Post),
        "PUT" => Ok(HttpMethod::Put),
        "PATCH" => Ok(HttpMethod::Patch),
        "DELETE" => Ok(HttpMethod::Delete),
        "HEAD" => Ok(HttpMethod::Head),
        other => Err(CatalogError::InvalidMethod(other.to_owned())),
    }
}

fn nonempty(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_owned())
}

fn split_pipe_list(value: &str) -> Vec<String> {
    value
        .split('|')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect()
}

fn split_catalog_list(value: &str) -> Vec<String> {
    if value.is_empty() {
        Vec::new()
    } else {
        vec![value.to_owned()]
    }
}

fn merged_path_parameters(path: &str, catalog_parameters: &str) -> Vec<String> {
    let mut parameters = split_pipe_list(catalog_parameters);
    for segment in path.split('/') {
        if let Some(parameter) = template_parameter(segment)
            && !parameters.iter().any(|existing| existing == parameter.name)
        {
            parameters.push(parameter.name.to_owned());
        }
    }
    parameters.sort();
    parameters
}

fn modeled_headers(
    method: HttpMethod,
    path: &str,
    request_body: &str,
    permissions: &str,
) -> Vec<String> {
    let mut headers = vec!["accept".to_owned()];
    if permissions.contains("sim-user") {
        headers.push("sim-user".to_owned());
    }
    if request_body != "none" && !request_body.is_empty() {
        headers.push("content-type".to_owned());
    }
    if !matches!(method, HttpMethod::Get | HttpMethod::Head) {
        headers.push("idempotency-key".to_owned());
    }
    if matches!(method, HttpMethod::Get | HttpMethod::Head)
        && (path.contains("content")
            || path.contains("view")
            || path.contains("{path:.*}")
            || path.contains("{filename:.*}"))
    {
        headers.push("range".to_owned());
        headers.push("if-none-match".to_owned());
    }
    headers
}

fn modeled_feature_gates(
    availability: RouteAvailability,
    permissions: &str,
    classification: &str,
) -> Vec<String> {
    let mut gates = permissions
        .split_whitespace()
        .filter(|part| part.starts_with("--"))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if !availability.is_active() {
        gates.push(format!("availability:{availability:?}").to_ascii_lowercase());
    }
    if classification.contains("static") {
        gates.push("native-static-root".to_owned());
    }
    gates
}

fn modeled_content_types(
    request_body: &str,
    response_detail: &str,
    status_content_types: &str,
) -> Vec<String> {
    const KNOWN: [&str; 13] = [
        "application/json",
        "application/octet-stream",
        "multipart/form-data",
        "text/plain",
        "text/html",
        "text/css",
        "text/javascript",
        "image/png",
        "image/jpeg",
        "image/webp",
        "audio/wav",
        "video/mp4",
        "model/gltf-binary",
    ];
    let haystack =
        format!("{request_body} {response_detail} {status_content_types}").to_ascii_lowercase();
    let mut content_types = KNOWN
        .iter()
        .filter(|content_type| haystack.contains(**content_type))
        .map(|content_type| (*content_type).to_owned())
        .collect::<Vec<_>>();
    if content_types.is_empty() && response_detail.contains("json_response") {
        content_types.push("application/json".to_owned());
    }
    content_types
}

fn modeled_status_codes(status_content_types: &str, response_detail: &str) -> Vec<u16> {
    let mut codes = status_content_types
        .split(|character: char| !character.is_ascii_digit())
        .filter(|token| token.len() == 3)
        .filter_map(|token| token.parse::<u16>().ok())
        .filter(|code| (100..=599).contains(code))
        .collect::<Vec<_>>();
    if status_content_types.contains("exact OpenAPI excerpt") {
        codes.extend(response_detail.split('|').filter_map(|line| {
            let yaml = line.trim().split_once(": ")?.1.trim();
            let status = yaml.strip_prefix('"')?.strip_suffix("\":")?;
            (status.len() == 3)
                .then(|| status.parse::<u16>().ok())
                .flatten()
                .filter(|code| (100..=599).contains(code))
        }));
    }
    codes.sort_unstable();
    codes.dedup();
    codes
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TemplateParameter<'a> {
    name: &'a str,
    constraint: Option<&'a str>,
}

fn template_parameter(segment: &str) -> Option<TemplateParameter<'_>> {
    let inner = segment.strip_prefix('{')?.strip_suffix('}')?;
    let (name, constraint) = match inner.split_once(':') {
        Some((name, constraint)) => (name, Some(constraint)),
        None => (inner, None),
    };
    (!name.is_empty()).then_some(TemplateParameter { name, constraint })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpLimits {
    pub maximum_request_bytes: usize,
    pub maximum_upload_bytes: usize,
    pub maximum_response_bytes: usize,
    pub maximum_range_bytes: usize,
    pub maximum_stream_chunk_bytes: usize,
    pub stream_channel_capacity: usize,
    pub idempotency_capacity: usize,
    pub maximum_query_values: usize,
    pub maximum_query_value_bytes: usize,
}

impl Default for HttpLimits {
    fn default() -> Self {
        Self {
            maximum_request_bytes: MAX_COMPATIBILITY_JSON_BYTES,
            maximum_upload_bytes: DEFAULT_MAX_UPLOAD_BYTES,
            maximum_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
            maximum_range_bytes: DEFAULT_MAX_RANGE_BYTES,
            maximum_stream_chunk_bytes: DEFAULT_STREAM_CHUNK_BYTES,
            stream_channel_capacity: 8,
            idempotency_capacity: 4_096,
            maximum_query_values: 256,
            maximum_query_value_bytes: 16 * 1024,
        }
    }
}

impl HttpLimits {
    pub fn validate(&self) -> Result<(), HttpRouteError> {
        if self.maximum_request_bytes == 0
            || self.maximum_upload_bytes == 0
            || self.maximum_response_bytes == 0
            || self.maximum_range_bytes == 0
            || self.maximum_stream_chunk_bytes == 0
            || self.stream_channel_capacity == 0
            || self.idempotency_capacity == 0
            || self.maximum_query_values == 0
            || self.maximum_query_value_bytes == 0
        {
            return Err(HttpRouteError::unaddressed(
                500,
                "invalid_http_limits",
                "native HTTP limits must all be greater than zero",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpRequest {
    pub method: HttpMethod,
    pub path: String,
    pub query: BTreeMap<String, Vec<String>>,
    pub headers: BTreeMap<String, Vec<String>>,
    pub body: Bytes,
}

impl HttpRequest {
    pub fn new(method: HttpMethod, path: impl Into<String>) -> Self {
        Self {
            method,
            path: path.into(),
            query: BTreeMap::new(),
            headers: BTreeMap::new(),
            body: Bytes::new(),
        }
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers
            .entry(name.into().to_ascii_lowercase())
            .or_default()
            .push(value.into());
        self
    }

    pub fn with_query(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.query
            .entry(name.into())
            .or_default()
            .push(value.into());
        self
    }

    pub fn with_body(mut self, body: impl Into<Bytes>) -> Self {
        self.body = body.into();
        self
    }

    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(candidate, _)| candidate.eq_ignore_ascii_case(name))
            .and_then(|(_, values)| values.first())
            .map(String::as_str)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RouteAddress {
    pub feature_id: String,
    pub method: HttpMethod,
    pub requested_path: String,
    pub canonical_path: String,
}

#[derive(Clone, Debug, Error, Eq, PartialEq, Serialize, Deserialize)]
#[error("{code}: {message}")]
pub struct HttpRouteError {
    pub status: u16,
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route: Option<Box<RouteAddress>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub details: BTreeMap<String, String>,
}

impl HttpRouteError {
    fn unaddressed(status: u16, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status,
            code: code.into(),
            message: message.into(),
            route: None,
            details: BTreeMap::new(),
        }
    }

    fn addressed(
        status: u16,
        code: impl Into<String>,
        message: impl Into<String>,
        matched: &MatchedRoute,
    ) -> Self {
        Self {
            status,
            code: code.into(),
            message: message.into(),
            route: Some(Box::new(matched.address())),
            details: BTreeMap::new(),
        }
    }

    pub fn into_response(self) -> HttpResponse {
        let status = self.status;
        HttpResponse::json(status, json!({ "error": self }))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum CapabilityState {
    Available,
    Disabled { reason: String },
    Unavailable { dependency: String, reason: String },
}

impl CapabilityState {
    fn error(&self, matched: &MatchedRoute) -> Option<HttpRouteError> {
        match self {
            Self::Available => None,
            Self::Disabled { reason } => Some(HttpRouteError::addressed(
                503,
                "route_disabled",
                reason,
                matched,
            )),
            Self::Unavailable { dependency, reason } => {
                let mut error =
                    HttpRouteError::addressed(501, "route_capability_unavailable", reason, matched);
                error
                    .details
                    .insert("dependency".to_owned(), dependency.clone());
                Some(error)
            }
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct HttpCapabilities {
    #[serde(default)]
    states: BTreeMap<String, CapabilityState>,
}

impl HttpCapabilities {
    pub fn set(&mut self, feature_id: impl Into<String>, state: CapabilityState) {
        self.states.insert(feature_id.into(), state);
    }

    pub fn state_for(&self, route: &HttpRouteDescriptor) -> CapabilityState {
        self.states
            .get(route.feature_id())
            .cloned()
            .unwrap_or_else(|| {
                if route.availability.is_active() {
                    CapabilityState::Available
                } else {
                    CapabilityState::Unavailable {
                        dependency: route.contract.feature_gates.join(", "),
                        reason: format!(
                            "{} is a {:?} compatibility route and was not enabled",
                            route.feature_id(),
                            route.availability
                        ),
                    }
                }
            })
    }

    pub fn negotiated_routes(
        &self,
        catalog: &[HttpRouteDescriptor],
    ) -> BTreeMap<String, CapabilityState> {
        catalog
            .iter()
            .map(|route| (route.feature_id().to_owned(), self.state_for(route)))
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatchedRoute {
    pub requested_feature_id: String,
    pub canonical_feature_id: String,
    pub method: HttpMethod,
    pub requested_path: String,
    pub canonical_path: String,
    pub path_parameters: BTreeMap<String, String>,
    descriptor_index: usize,
    canonical_descriptor_index: usize,
}

impl MatchedRoute {
    fn address(&self) -> RouteAddress {
        RouteAddress {
            feature_id: self.requested_feature_id.clone(),
            method: self.method,
            requested_path: self.requested_path.clone(),
            canonical_path: self.canonical_path.clone(),
        }
    }
}

pub fn match_http_route(
    method: HttpMethod,
    path: &str,
) -> Result<Option<MatchedRoute>, CatalogError> {
    let catalog = http_route_catalog()?;
    let mut candidates = catalog
        .iter()
        .enumerate()
        .filter(|(_, route)| route.contract.identity.method == method)
        .filter_map(|(index, route)| {
            match_template(&route.contract.identity.path, path)
                .map(|(parameters, score)| (index, route, parameters, score))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.3.cmp(&left.3).then(left.0.cmp(&right.0)));
    let Some((descriptor_index, route, parameters, _)) = candidates.first() else {
        return Ok(None);
    };
    let canonical_descriptor_index = catalog
        .iter()
        .position(|candidate| {
            candidate.contract.identity.method == method
                && candidate.contract.identity.path == route.contract.identity.canonical_path
                && candidate.contract.identity.alias_of.is_none()
        })
        .unwrap_or(*descriptor_index);
    let canonical = catalog.get(canonical_descriptor_index).unwrap_or(route);
    Ok(Some(MatchedRoute {
        requested_feature_id: route.feature_id().to_owned(),
        canonical_feature_id: canonical.feature_id().to_owned(),
        method,
        requested_path: path.to_owned(),
        canonical_path: route.contract.identity.canonical_path.clone(),
        path_parameters: parameters.clone(),
        descriptor_index: *descriptor_index,
        canonical_descriptor_index,
    }))
}

fn match_template(template: &str, requested: &str) -> Option<(BTreeMap<String, String>, usize)> {
    let template_segments = template
        .trim_start_matches('/')
        .split('/')
        .collect::<Vec<_>>();
    let requested_segments = requested
        .trim_start_matches('/')
        .split('/')
        .collect::<Vec<_>>();
    let mut parameters = BTreeMap::new();
    let mut requested_index = 0;
    let mut score = 0;
    for (template_index, segment) in template_segments.iter().enumerate() {
        if let Some(parameter) = template_parameter(segment) {
            if parameter.constraint == Some(".*") {
                let raw = requested_segments.get(requested_index..)?.join("/");
                let decoded = decode_uri_component(&raw, false).ok()?;
                parameters.insert(parameter.name.to_owned(), decoded);
                requested_index = requested_segments.len();
                score += 1;
                if template_index + 1 != template_segments.len() {
                    return None;
                }
                continue;
            }
            let raw = *requested_segments.get(requested_index)?;
            if raw.is_empty() {
                return None;
            }
            let decoded = decode_uri_component(raw, false).ok()?;
            if parameter.constraint == Some("{UUID_RE}") && !looks_like_uuid(&decoded) {
                return None;
            }
            parameters.insert(parameter.name.to_owned(), decoded);
            requested_index += 1;
            score += 10;
        } else {
            if requested_segments.get(requested_index).copied() != Some(*segment) {
                return None;
            }
            requested_index += 1;
            score += 100;
        }
    }
    (requested_index == requested_segments.len()).then_some((parameters, score))
}

fn looks_like_uuid(value: &str) -> bool {
    value.len() == 36
        && value
            .chars()
            .enumerate()
            .all(|(index, character)| match index {
                8 | 13 | 18 | 23 => character == '-',
                _ => character.is_ascii_hexdigit(),
            })
}

pub(crate) fn decode_uri_component(value: &str, plus_as_space: bool) -> Result<String, ()> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let high = bytes
                .get(index + 1)
                .copied()
                .and_then(hex_digit)
                .ok_or(())?;
            let low = bytes
                .get(index + 2)
                .copied()
                .and_then(hex_digit)
                .ok_or(())?;
            decoded.push((high << 4) | low);
            index += 3;
        } else if bytes[index] == b'+' && plus_as_space {
            decoded.push(b' ');
            index += 1;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded).map_err(|_| ())
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SafeVirtualPath(String);

impl SafeVirtualPath {
    pub fn parse(value: &str) -> Result<Self, SafePathError> {
        AssetIdentity::new("http-wire", AssetNamespace::Temporary, value)
            .map_err(|error| SafePathError::Canonical(error.to_string()))?;
        Ok(Self(value.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_asset_identity(
        self,
        profile_id: impl Into<String>,
        namespace: AssetNamespace,
    ) -> Result<AssetIdentity, SafePathError> {
        AssetIdentity::new(profile_id, namespace, self.0)
            .map_err(|error| SafePathError::Canonical(error.to_string()))
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum SafePathError {
    #[error("virtual path is not a canonical asset identity: {0}")]
    Canonical(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidatedUpload {
    pub filename: SafeVirtualPath,
    pub content_type: String,
    pub bytes: Bytes,
}

impl ValidatedUpload {
    pub fn validate(
        filename: &str,
        content_type: &str,
        bytes: Bytes,
        limits: &HttpLimits,
    ) -> Result<Self, HttpRouteError> {
        if bytes.len() > limits.maximum_upload_bytes {
            return Err(HttpRouteError::unaddressed(
                413,
                "upload_too_large",
                format!(
                    "upload contains {} bytes; maximum is {}",
                    bytes.len(),
                    limits.maximum_upload_bytes
                ),
            ));
        }
        let filename = SafeVirtualPath::parse(filename).map_err(|error| {
            HttpRouteError::unaddressed(400, "unsafe_upload_filename", error.to_string())
        })?;
        if content_type.trim().is_empty() {
            return Err(HttpRouteError::unaddressed(
                400,
                "missing_upload_content_type",
                "upload content type is required",
            ));
        }
        Ok(Self {
            filename,
            content_type: content_type.to_owned(),
            bytes,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RangeRequest {
    pub start: usize,
    pub end_inclusive: Option<usize>,
}

fn parse_range(value: &str, maximum: usize) -> Result<RangeRequest, HttpRouteError> {
    let range = value.strip_prefix("bytes=").ok_or_else(|| {
        HttpRouteError::unaddressed(416, "invalid_range", "only byte ranges are supported")
    })?;
    if range.contains(',') {
        return Err(HttpRouteError::unaddressed(
            416,
            "multiple_ranges_unsupported",
            "only one byte range may be requested",
        ));
    }
    let (start, end) = range.split_once('-').ok_or_else(|| {
        HttpRouteError::unaddressed(416, "invalid_range", "range must contain a hyphen")
    })?;
    if start.is_empty() {
        return Err(HttpRouteError::unaddressed(
            416,
            "suffix_range_unsupported",
            "suffix ranges require an already resolved asset length",
        ));
    }
    let start = start.parse::<usize>().map_err(|_| {
        HttpRouteError::unaddressed(416, "invalid_range", "range start is not an integer")
    })?;
    let end_inclusive = if end.is_empty() {
        None
    } else {
        Some(end.parse::<usize>().map_err(|_| {
            HttpRouteError::unaddressed(416, "invalid_range", "range end is not an integer")
        })?)
    };
    if let Some(end) = end_inclusive {
        if end < start {
            return Err(HttpRouteError::unaddressed(
                416,
                "invalid_range",
                "range end precedes range start",
            ));
        }
        let length = end
            .checked_sub(start)
            .and_then(|difference| difference.checked_add(1))
            .ok_or_else(|| {
                HttpRouteError::unaddressed(416, "invalid_range", "range length overflowed")
            })?;
        if length > maximum {
            return Err(HttpRouteError::unaddressed(
                416,
                "range_too_large",
                format!("range contains {length} bytes; maximum is {maximum}"),
            ));
        }
    }
    Ok(RangeRequest {
        start,
        end_inclusive,
    })
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum MutationIdentity {
    IdempotencyKey(String),
    DurableAttempt(String),
    Untracked,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeServiceOperation {
    Root,
    Features,
    PromptStatus,
    SubmitPrompt,
    QueueSnapshot,
    QueueMutation,
    HistoryRead,
    HistoryMutation,
    Interrupt,
    Jobs,
    Assets,
    Upload,
    UserData,
    Settings,
    Models,
    NodeCatalog,
    Extensions,
    StaticAsset,
    WebSocketUpgrade,
    CatalogRoute { feature_id: String },
}

fn operation_for(
    method: HttpMethod,
    canonical_path: &str,
    feature_id: &str,
) -> NativeServiceOperation {
    match (method, canonical_path) {
        (HttpMethod::Get, "/") => NativeServiceOperation::Root,
        (HttpMethod::Get, "/features") => NativeServiceOperation::Features,
        (HttpMethod::Get, "/prompt") => NativeServiceOperation::PromptStatus,
        (HttpMethod::Post, "/prompt") => NativeServiceOperation::SubmitPrompt,
        (HttpMethod::Get, "/queue") => NativeServiceOperation::QueueSnapshot,
        (HttpMethod::Post, "/queue") => NativeServiceOperation::QueueMutation,
        (HttpMethod::Post, "/interrupt") => NativeServiceOperation::Interrupt,
        (HttpMethod::Get, path) if path == "/history" || path.starts_with("/history/") => {
            NativeServiceOperation::HistoryRead
        }
        (_, path) if path == "/history" || path.starts_with("/history/") => {
            NativeServiceOperation::HistoryMutation
        }
        (_, path) if path == "/api/jobs" || path.starts_with("/api/jobs/") => {
            NativeServiceOperation::Jobs
        }
        (_, path) if path.starts_with("/api/assets") || path == "/api/tags" => {
            NativeServiceOperation::Assets
        }
        (HttpMethod::Post, "/upload/image" | "/upload/mask") => NativeServiceOperation::Upload,
        (_, path) if path.starts_with("/userdata") || path.starts_with("/v2/userdata") => {
            NativeServiceOperation::UserData
        }
        (_, path) if path.starts_with("/settings") => NativeServiceOperation::Settings,
        (_, path) if path.starts_with("/models") || path.starts_with("/experiment/models") => {
            NativeServiceOperation::Models
        }
        (_, path) if path.starts_with("/object_info") => NativeServiceOperation::NodeCatalog,
        (_, path) if path == "/extensions" || path.starts_with("/extensions/") => {
            NativeServiceOperation::Extensions
        }
        (HttpMethod::Get, "/ws") => NativeServiceOperation::WebSocketUpgrade,
        (HttpMethod::Get, path) if path.contains("{path:.*}") || path.contains("{filename:.*}") => {
            NativeServiceOperation::StaticAsset
        }
        _ => NativeServiceOperation::CatalogRoute {
            feature_id: feature_id.to_owned(),
        },
    }
}

#[derive(Clone, Debug)]
pub struct NativeServiceRequest {
    pub route: MatchedRoute,
    pub operation: NativeServiceOperation,
    pub query: BTreeMap<String, Vec<String>>,
    pub headers: BTreeMap<String, Vec<String>>,
    pub body: Bytes,
    pub json_body: Option<Value>,
    pub range: Option<RangeRequest>,
    pub mutation_identity: MutationIdentity,
    pub authority: Option<NativeRequestAuthority>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeRequestAuthority {
    pub profile_id: String,
    pub principal: String,
    pub scopes: std::collections::BTreeSet<String>,
    pub plugin_id: Option<String>,
    pub plugin_digest: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeServiceErrorKind {
    Invalid,
    Unauthorized,
    Forbidden,
    NotFound,
    Conflict,
    Oversized,
    Cancelled,
    Timeout,
    Unavailable,
    Internal,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("{code}: {message}")]
pub struct NativeServiceError {
    pub kind: NativeServiceErrorKind,
    pub code: String,
    pub message: String,
}

impl NativeServiceError {
    pub fn new(
        kind: NativeServiceErrorKind,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            code: code.into(),
            message: message.into(),
        }
    }

    fn status(&self) -> u16 {
        match self.kind {
            NativeServiceErrorKind::Invalid => 400,
            NativeServiceErrorKind::Unauthorized => 401,
            NativeServiceErrorKind::Forbidden => 403,
            NativeServiceErrorKind::NotFound => 404,
            NativeServiceErrorKind::Conflict => 409,
            NativeServiceErrorKind::Oversized => 413,
            NativeServiceErrorKind::Cancelled => 409,
            NativeServiceErrorKind::Timeout => 504,
            NativeServiceErrorKind::Unavailable => 503,
            NativeServiceErrorKind::Internal => 500,
        }
    }
}

pub trait NativeHttpServices: Send + Sync + 'static {
    fn dispatch(
        &self,
        request: NativeServiceRequest,
    ) -> Result<NativeServiceResponse, NativeServiceError>;

    fn reconcile_mutation(
        &self,
        _request: &NativeServiceRequest,
    ) -> Result<NativeMutationReconciliation, NativeServiceError> {
        Ok(NativeMutationReconciliation::Unresolved {
            reason: "the native service does not expose a mutation reconciler".to_owned(),
        })
    }

    fn status_projection(&self) -> Result<Option<Value>, NativeServiceError> {
        Ok(None)
    }
}

#[derive(Debug)]
pub enum NativeMutationReconciliation {
    Committed(NativeServiceResponse),
    NotApplied,
    Unresolved { reason: String },
}

pub enum HttpBody {
    Empty,
    Bytes(Bytes),
    Json(Value),
    Stream(BoundedResponseStream),
}

impl std::fmt::Debug for HttpBody {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => formatter.write_str("Empty"),
            Self::Bytes(bytes) => formatter.debug_tuple("Bytes").field(&bytes.len()).finish(),
            Self::Json(value) => formatter.debug_tuple("Json").field(value).finish(),
            Self::Stream(stream) => formatter.debug_tuple("Stream").field(stream).finish(),
        }
    }
}

#[derive(Debug)]
pub struct NativeServiceResponse {
    pub status: u16,
    pub content_type: String,
    pub headers: BTreeMap<String, String>,
    pub body: HttpBody,
}

impl NativeServiceResponse {
    pub fn json(status: u16, value: Value) -> Self {
        Self {
            status,
            content_type: "application/json".to_owned(),
            headers: BTreeMap::new(),
            body: HttpBody::Json(value),
        }
    }

    pub fn bytes(status: u16, content_type: impl Into<String>, bytes: impl Into<Bytes>) -> Self {
        Self {
            status,
            content_type: content_type.into(),
            headers: BTreeMap::new(),
            body: HttpBody::Bytes(bytes.into()),
        }
    }
}

#[derive(Debug)]
pub struct HttpResponse {
    pub status: u16,
    pub content_type: String,
    pub headers: BTreeMap<String, String>,
    pub body: HttpBody,
}

impl HttpResponse {
    pub fn json(status: u16, value: Value) -> Self {
        Self {
            status,
            content_type: "application/json".to_owned(),
            headers: BTreeMap::new(),
            body: HttpBody::Json(value),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BoundedResponseSender {
    sender: Sender<Result<Bytes, StreamError>>,
    maximum_chunk_bytes: usize,
    maximum_total_bytes: usize,
    sent_bytes: Arc<AtomicUsize>,
}

#[derive(Clone, Debug)]
pub struct BoundedResponseStream {
    receiver: Receiver<Result<Bytes, StreamError>>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum StreamError {
    #[error("stream chunk contains {actual} bytes; maximum is {maximum}")]
    ChunkTooLarge { actual: usize, maximum: usize },
    #[error("stream contains more than {maximum} bytes")]
    TotalTooLarge { maximum: usize },
    #[error("stream receiver was closed")]
    Closed,
    #[error("native stream failed: {0}")]
    Service(String),
}

pub fn bounded_response_stream(
    limits: &HttpLimits,
) -> Result<(BoundedResponseSender, BoundedResponseStream), HttpRouteError> {
    limits.validate()?;
    let (sender, receiver) = async_channel::bounded(limits.stream_channel_capacity);
    Ok((
        BoundedResponseSender {
            sender,
            maximum_chunk_bytes: limits.maximum_stream_chunk_bytes,
            maximum_total_bytes: limits.maximum_response_bytes,
            sent_bytes: Arc::new(AtomicUsize::new(0)),
        },
        BoundedResponseStream { receiver },
    ))
}

impl BoundedResponseSender {
    pub async fn send(&self, bytes: Bytes) -> Result<(), StreamError> {
        if bytes.len() > self.maximum_chunk_bytes {
            return Err(StreamError::ChunkTooLarge {
                actual: bytes.len(),
                maximum: self.maximum_chunk_bytes,
            });
        }
        let previous = self.sent_bytes.fetch_add(bytes.len(), Ordering::AcqRel);
        let total = previous
            .checked_add(bytes.len())
            .ok_or(StreamError::TotalTooLarge {
                maximum: self.maximum_total_bytes,
            })?;
        if total > self.maximum_total_bytes {
            self.sent_bytes.fetch_sub(bytes.len(), Ordering::AcqRel);
            return Err(StreamError::TotalTooLarge {
                maximum: self.maximum_total_bytes,
            });
        }
        if self.sender.send(Ok(bytes)).await.is_err() {
            self.sent_bytes
                .fetch_sub(total - previous, Ordering::AcqRel);
            return Err(StreamError::Closed);
        }
        Ok(())
    }

    pub async fn fail(&self, error: StreamError) -> Result<(), StreamError> {
        self.sender
            .send(Err(error))
            .await
            .map_err(|_| StreamError::Closed)
    }
}

impl BoundedResponseStream {
    pub async fn next(&self) -> Option<Result<Bytes, StreamError>> {
        self.receiver.recv().await.ok()
    }
}

pub(crate) struct NativeHttpRouter<S> {
    services: Arc<S>,
    limits: HttpLimits,
    capabilities: RwLock<HttpCapabilities>,
}

struct PreparedNativeRequest {
    matched: MatchedRoute,
    range: Option<RangeRequest>,
    service_request: NativeServiceRequest,
}

pub(crate) enum RoutedMutationReconciliation {
    Committed(HttpResponse),
    NotApplied,
    Unresolved { reason: String },
}

impl<S> NativeHttpRouter<S>
where
    S: NativeHttpServices,
{
    pub fn new(
        services: Arc<S>,
        limits: HttpLimits,
        capabilities: HttpCapabilities,
    ) -> Result<Self, HttpRouteError> {
        limits.validate()?;
        Ok(Self {
            services,
            limits,
            capabilities: RwLock::new(capabilities),
        })
    }

    #[cfg(test)]
    pub fn set_capability(
        &self,
        feature_id: impl Into<String>,
        state: CapabilityState,
    ) -> Result<(), HttpRouteError> {
        let mut capabilities = self.capabilities.write().map_err(|_| {
            HttpRouteError::unaddressed(
                500,
                "capability_state_poisoned",
                "native HTTP capability state could not be updated",
            )
        })?;
        capabilities.set(feature_id, state);
        Ok(())
    }

    pub fn status_projection(&self) -> Result<Option<Value>, HttpRouteError> {
        self.services
            .status_projection()
            .map_err(|error| HttpRouteError::unaddressed(error.status(), error.code, error.message))
    }

    #[cfg(test)]
    pub fn route(&self, request: HttpRequest) -> Result<HttpResponse, HttpRouteError> {
        self.route_authorized(request, None)
    }

    pub fn route_authorized(
        &self,
        request: HttpRequest,
        authority: Option<NativeRequestAuthority>,
    ) -> Result<HttpResponse, HttpRouteError> {
        let prepared = self.prepare_authorized(request, authority)?;
        let native = self
            .services
            .dispatch(prepared.service_request)
            .map_err(|error| {
                HttpRouteError::addressed(
                    error.status(),
                    error.code,
                    error.message,
                    &prepared.matched,
                )
            })?;
        project_response(native, prepared.range, &self.limits, &prepared.matched)
    }

    pub(crate) fn reconcile_authorized(
        &self,
        request: HttpRequest,
        authority: NativeRequestAuthority,
    ) -> Result<RoutedMutationReconciliation, HttpRouteError> {
        let prepared = self.prepare_authorized(request, Some(authority))?;
        match self
            .services
            .reconcile_mutation(&prepared.service_request)
            .map_err(|error| {
                HttpRouteError::addressed(
                    error.status(),
                    error.code,
                    error.message,
                    &prepared.matched,
                )
            })? {
            NativeMutationReconciliation::Committed(native) => {
                project_response(native, prepared.range, &self.limits, &prepared.matched)
                    .map(RoutedMutationReconciliation::Committed)
            }
            NativeMutationReconciliation::NotApplied => {
                Ok(RoutedMutationReconciliation::NotApplied)
            }
            NativeMutationReconciliation::Unresolved { reason } => {
                Ok(RoutedMutationReconciliation::Unresolved { reason })
            }
        }
    }

    fn prepare_authorized(
        &self,
        request: HttpRequest,
        authority: Option<NativeRequestAuthority>,
    ) -> Result<PreparedNativeRequest, HttpRouteError> {
        let catalog = http_route_catalog().map_err(catalog_route_error)?;
        let matched = match_http_route(request.method, &request.path)
            .map_err(catalog_route_error)?
            .ok_or_else(|| {
                HttpRouteError::unaddressed(
                    404,
                    "route_not_found",
                    format!(
                        "no native {:?} route matches {}",
                        request.method, request.path
                    ),
                )
            })?;
        let descriptor = catalog.get(matched.descriptor_index).ok_or_else(|| {
            HttpRouteError::unaddressed(
                500,
                "catalog_index_invalid",
                "matched route is absent from the native catalog",
            )
        })?;
        let canonical_descriptor =
            catalog
                .get(matched.canonical_descriptor_index)
                .ok_or_else(|| {
                    HttpRouteError::unaddressed(
                        500,
                        "canonical_catalog_index_invalid",
                        "canonical route is absent from the native catalog",
                    )
                })?;

        validate_request_headers(&request, &matched)?;
        validate_query(&request, &self.limits, &matched)?;
        validate_virtual_paths(&matched)?;
        let capability_state = self
            .capabilities
            .read()
            .map_err(|_| {
                HttpRouteError::addressed(
                    500,
                    "capability_state_poisoned",
                    "native HTTP capability state could not be read",
                    &matched,
                )
            })?
            .state_for(descriptor);
        if let Some(error) = capability_state.error(&matched) {
            return Err(error);
        }

        let maximum_body = if descriptor
            .contract
            .unknown
            .get("request_body")
            .and_then(Value::as_str)
            == Some("multipart/form-data")
        {
            self.limits.maximum_upload_bytes
        } else {
            self.limits.maximum_request_bytes
        };
        if request.body.len() > maximum_body {
            return Err(HttpRouteError::addressed(
                413,
                "request_too_large",
                format!(
                    "request contains {} bytes; maximum is {maximum_body}",
                    request.body.len()
                ),
                &matched,
            ));
        }
        let json_body = decode_json_body(descriptor, &request, &matched)?;
        let range = request
            .header("range")
            .map(|value| parse_range(value, self.limits.maximum_range_bytes))
            .transpose()
            .map_err(|mut error| {
                error.route = Some(Box::new(matched.address()));
                error
            })?;
        let mutation_identity = mutation_identity(
            descriptor,
            &request,
            json_body.as_ref(),
            &matched,
            authority.as_ref(),
        );
        let operation = operation_for(
            request.method,
            &canonical_descriptor.contract.identity.path,
            canonical_descriptor.feature_id(),
        );
        Ok(PreparedNativeRequest {
            matched: matched.clone(),
            range: range.clone(),
            service_request: NativeServiceRequest {
                route: matched,
                operation,
                query: request.query,
                headers: request.headers,
                body: request.body,
                json_body,
                range,
                mutation_identity,
                authority,
            },
        })
    }

    #[cfg(test)]
    pub fn handle(&self, request: HttpRequest) -> HttpResponse {
        self.route(request)
            .unwrap_or_else(HttpRouteError::into_response)
    }
}

fn catalog_route_error(error: CatalogError) -> HttpRouteError {
    HttpRouteError::unaddressed(500, "http_catalog_invalid", error.to_string())
}

pub(crate) fn validate_request_headers(
    request: &HttpRequest,
    matched: &MatchedRoute,
) -> Result<(), HttpRouteError> {
    const SINGLETON_HEADERS: [&str; 28] = [
        "authorization",
        "sim-user",
        "connection",
        "content-encoding",
        "content-length",
        "content-type",
        "expect",
        "forwarded",
        "host",
        "idempotency-key",
        "if-none-match",
        "origin",
        "proxy-authorization",
        "range",
        "sec-websocket-key",
        "sec-websocket-protocol",
        "sec-websocket-version",
        "transfer-encoding",
        "upgrade",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
        "x-real-ip",
        "x-zed-plugin-capabilities",
        "x-zed-plugin-digest",
        "x-zed-plugin-id",
        "x-zed-plugin-profile",
        "x-zed-plugin-version",
    ];
    let mut counts = BTreeMap::<String, usize>::new();
    for (name, values) in &request.headers {
        if name.is_empty()
            || values.is_empty()
            || name
                .bytes()
                .any(|byte| byte.is_ascii_control() || matches!(byte, b' ' | b':'))
            || values.iter().any(|value| {
                value
                    .bytes()
                    .any(|byte| matches!(byte, b'\0' | b'\r' | b'\n'))
            })
        {
            let mut error = HttpRouteError::addressed(
                400,
                "invalid_request_header",
                "request headers must not contain empty names, controls, whitespace, or line breaks",
                matched,
            );
            error
                .details
                .insert("header".to_owned(), name.to_ascii_lowercase());
            return Err(error);
        }
        *counts.entry(name.to_ascii_lowercase()).or_default() += values.len();
    }
    if let Some((name, count)) = counts
        .iter()
        .find(|(name, count)| SINGLETON_HEADERS.contains(&name.as_str()) && **count != 1)
    {
        let mut error = HttpRouteError::addressed(
            400,
            "duplicate_request_header",
            format!("security or framing header {name} appeared {count} times"),
            matched,
        );
        error.details.insert("header".to_owned(), name.clone());
        return Err(error);
    }
    if counts.contains_key("content-length") && counts.contains_key("transfer-encoding") {
        return Err(HttpRouteError::addressed(
            400,
            "ambiguous_request_framing",
            "content-length and transfer-encoding cannot be combined",
            matched,
        ));
    }
    if let Some(content_length) = request.header("content-length") {
        let content_length = content_length.parse::<usize>().map_err(|_| {
            HttpRouteError::addressed(
                400,
                "invalid_content_length",
                "content-length must be a non-negative decimal integer",
                matched,
            )
        })?;
        if content_length != request.body.len() {
            let mut error = HttpRouteError::addressed(
                400,
                "content_length_mismatch",
                "content-length does not match the decoded request body",
                matched,
            );
            error
                .details
                .insert("declared".to_owned(), content_length.to_string());
            error
                .details
                .insert("observed".to_owned(), request.body.len().to_string());
            return Err(error);
        }
    }
    if let Some(transfer_encoding) = request.header("transfer-encoding")
        && !transfer_encoding.eq_ignore_ascii_case("chunked")
    {
        return Err(HttpRouteError::addressed(
            400,
            "unsupported_transfer_encoding",
            "native HTTP requests support only chunked transfer encoding",
            matched,
        ));
    }
    Ok(())
}

fn validate_query(
    request: &HttpRequest,
    limits: &HttpLimits,
    matched: &MatchedRoute,
) -> Result<(), HttpRouteError> {
    let value_count = request.query.values().map(Vec::len).sum::<usize>();
    if value_count > limits.maximum_query_values {
        return Err(HttpRouteError::addressed(
            413,
            "too_many_query_values",
            format!(
                "request has {value_count} query values; maximum is {}",
                limits.maximum_query_values
            ),
            matched,
        ));
    }
    if let Some((name, value)) = request
        .query
        .iter()
        .flat_map(|(name, values)| values.iter().map(move |value| (name, value)))
        .find(|(_, value)| value.len() > limits.maximum_query_value_bytes)
    {
        let mut error = HttpRouteError::addressed(
            413,
            "query_value_too_large",
            format!(
                "query value contains {} bytes; maximum is {}",
                value.len(),
                limits.maximum_query_value_bytes
            ),
            matched,
        );
        error.details.insert("parameter".to_owned(), name.clone());
        return Err(error);
    }
    Ok(())
}

fn validate_virtual_paths(matched: &MatchedRoute) -> Result<(), HttpRouteError> {
    const PATH_NAMES: [&str; 8] = [
        "file",
        "dest",
        "path",
        "filename",
        "folder",
        "folder_name",
        "module_name",
        "extension_name",
    ];
    for (name, value) in &matched.path_parameters {
        if PATH_NAMES.contains(&name.as_str()) {
            SafeVirtualPath::parse(value).map_err(|error| {
                let mut route_error = HttpRouteError::addressed(
                    400,
                    "unsafe_virtual_path",
                    error.to_string(),
                    matched,
                );
                route_error
                    .details
                    .insert("parameter".to_owned(), name.clone());
                route_error
            })?;
        }
    }
    Ok(())
}

fn decode_json_body(
    descriptor: &HttpRouteDescriptor,
    request: &HttpRequest,
    matched: &MatchedRoute,
) -> Result<Option<Value>, HttpRouteError> {
    let body_kind = descriptor
        .contract
        .unknown
        .get("request_body")
        .and_then(Value::as_str)
        .unwrap_or("none");
    let content_type_is_json = request
        .header("content-type")
        .is_some_and(|value| value.split(';').next() == Some("application/json"));
    let requires_json = body_kind == "application/json";
    if request.body.is_empty() {
        if requires_json {
            return Err(HttpRouteError::addressed(
                400,
                "missing_request_body",
                "this route requires an application/json body",
                matched,
            ));
        }
        return Ok(None);
    }
    if requires_json || content_type_is_json {
        return serde_json::from_slice(&request.body)
            .map(Some)
            .map_err(|error| {
                let mut route_error = HttpRouteError::addressed(
                    400,
                    "malformed_json",
                    "request body is not valid JSON",
                    matched,
                );
                route_error
                    .details
                    .insert("source".to_owned(), error.to_string());
                route_error
            });
    }
    Ok(None)
}

pub(crate) fn mutation_identity(
    descriptor: &HttpRouteDescriptor,
    request: &HttpRequest,
    json_body: Option<&Value>,
    matched: &MatchedRoute,
    authority: Option<&NativeRequestAuthority>,
) -> MutationIdentity {
    if !descriptor.is_mutation() {
        return MutationIdentity::Untracked;
    }
    if let Some(key) = request
        .header("idempotency-key")
        .filter(|key| !key.is_empty())
    {
        return MutationIdentity::IdempotencyKey(scoped_mutation_value(key, authority));
    }
    if let Some(operation_id) = request
        .header("x-operation-id")
        .filter(|operation_id| !operation_id.is_empty())
    {
        return MutationIdentity::DurableAttempt(scoped_mutation_value(
            &format!("operation_id:{operation_id}"),
            authority,
        ));
    }
    const DURABLE_KEYS: [&str; 5] = ["attempt_id", "request_id", "prompt_id", "job_id", "task_id"];
    if let Some((name, value)) = DURABLE_KEYS.iter().find_map(|name| {
        matched
            .path_parameters
            .get(*name)
            .map(|value| (*name, value.clone()))
    }) {
        return MutationIdentity::DurableAttempt(scoped_mutation_value(
            &format!("{name}:{value}"),
            authority,
        ));
    }
    if let Some((name, value)) = json_body.and_then(Value::as_object).and_then(|object| {
        DURABLE_KEYS.iter().find_map(|name| {
            object.get(*name).and_then(|value| match value {
                Value::String(value) => Some((*name, value.clone())),
                Value::Number(value) => Some((*name, value.to_string())),
                _ => None,
            })
        })
    }) {
        return MutationIdentity::DurableAttempt(scoped_mutation_value(
            &format!("{name}:{value}"),
            authority,
        ));
    }
    MutationIdentity::Untracked
}

fn scoped_mutation_value(value: &str, authority: Option<&NativeRequestAuthority>) -> String {
    let Some(authority) = authority else {
        return value.to_owned();
    };
    let mut digest = Sha256::new();
    for component in [
        authority.profile_id.as_str(),
        authority.principal.as_str(),
        authority.plugin_id.as_deref().unwrap_or(""),
        authority.plugin_digest.as_deref().unwrap_or(""),
    ] {
        digest.update(component.len().to_be_bytes());
        digest.update(component.as_bytes());
    }
    format!("authority:{:x}:{value}", digest.finalize())
}

fn project_response(
    native: NativeServiceResponse,
    range: Option<RangeRequest>,
    limits: &HttpLimits,
    matched: &MatchedRoute,
) -> Result<HttpResponse, HttpRouteError> {
    let mut response = HttpResponse {
        status: native.status,
        content_type: native.content_type,
        headers: native.headers,
        body: native.body,
    };
    match &response.body {
        HttpBody::Bytes(bytes) if bytes.len() > limits.maximum_response_bytes => {
            return Err(HttpRouteError::addressed(
                500,
                "native_response_too_large",
                format!(
                    "native response contains {} bytes; maximum is {}",
                    bytes.len(),
                    limits.maximum_response_bytes
                ),
                matched,
            ));
        }
        HttpBody::Json(value) => {
            let length = serde_json::to_vec(value)
                .map_err(|error| {
                    HttpRouteError::addressed(
                        500,
                        "response_encoding_failed",
                        error.to_string(),
                        matched,
                    )
                })?
                .len();
            if length > limits.maximum_response_bytes {
                return Err(HttpRouteError::addressed(
                    500,
                    "native_response_too_large",
                    format!(
                        "encoded native response contains {length} bytes; maximum is {}",
                        limits.maximum_response_bytes
                    ),
                    matched,
                ));
            }
        }
        HttpBody::Empty | HttpBody::Stream(_) | HttpBody::Bytes(_) => {}
    }
    if let Some(range) = range {
        let HttpBody::Bytes(bytes) = &response.body else {
            return Err(HttpRouteError::addressed(
                416,
                "range_not_supported",
                "this native response is not byte-addressable",
                matched,
            ));
        };
        if range.start >= bytes.len() {
            let mut error = HttpRouteError::addressed(
                416,
                "range_not_satisfiable",
                "range starts beyond the native response length",
                matched,
            );
            error
                .details
                .insert("content_length".to_owned(), bytes.len().to_string());
            return Err(error);
        }
        let maximum_end = range
            .start
            .checked_add(limits.maximum_range_bytes.saturating_sub(1))
            .unwrap_or(usize::MAX);
        let end = range
            .end_inclusive
            .unwrap_or(maximum_end)
            .min(maximum_end)
            .min(bytes.len() - 1);
        let selected = bytes.slice(range.start..=end);
        response.status = 206;
        response.headers.insert(
            "content-range".to_owned(),
            format!("bytes {}-{end}/{}", range.start, bytes.len()),
        );
        response
            .headers
            .insert("content-length".to_owned(), selected.len().to_string());
        response
            .headers
            .insert("accept-ranges".to_owned(), "bytes".to_owned());
        response.body = HttpBody::Bytes(selected);
    }
    Ok(response)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct ProbeServices {
        calls: AtomicUsize,
        operations: Mutex<Vec<NativeServiceOperation>>,
        mutation_identities: Mutex<Vec<MutationIdentity>>,
    }

    impl NativeHttpServices for ProbeServices {
        fn dispatch(
            &self,
            request: NativeServiceRequest,
        ) -> Result<NativeServiceResponse, NativeServiceError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.operations
                .lock()
                .map_err(|_| {
                    NativeServiceError::new(
                        NativeServiceErrorKind::Internal,
                        "probe_poisoned",
                        "probe operation state was poisoned",
                    )
                })?
                .push(request.operation.clone());
            self.mutation_identities
                .lock()
                .map_err(|_| {
                    NativeServiceError::new(
                        NativeServiceErrorKind::Internal,
                        "probe_poisoned",
                        "probe mutation identity state was poisoned",
                    )
                })?
                .push(request.mutation_identity.clone());
            match request.operation {
                NativeServiceOperation::StaticAsset => Err(NativeServiceError::new(
                    NativeServiceErrorKind::NotFound,
                    "asset_not_found",
                    "the native static asset does not exist",
                )),
                NativeServiceOperation::Upload => {
                    Ok(NativeServiceResponse::json(201, json!({ "stored": true })))
                }
                NativeServiceOperation::PromptStatus => Ok(NativeServiceResponse::bytes(
                    200,
                    "application/octet-stream",
                    Bytes::from_static(b"0123456789"),
                )),
                _ => Ok(NativeServiceResponse::json(
                    200,
                    json!({
                        "native": true,
                        "feature_id": request.route.canonical_feature_id,
                    }),
                )),
            }
        }
    }

    struct FailingServices(NativeServiceErrorKind);

    impl NativeHttpServices for FailingServices {
        fn dispatch(
            &self,
            _request: NativeServiceRequest,
        ) -> Result<NativeServiceResponse, NativeServiceError> {
            Err(NativeServiceError::new(
                self.0,
                "fixture_service_error",
                "typed native service failure",
            ))
        }
    }

    struct CountingRuntimeServices {
        inner: crate::services::NativeRuntimeHttpServices,
        calls: AtomicUsize,
    }

    impl NativeHttpServices for CountingRuntimeServices {
        fn dispatch(
            &self,
            request: NativeServiceRequest,
        ) -> Result<NativeServiceResponse, NativeServiceError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.dispatch(request)
        }
    }

    fn router() -> Result<(Arc<ProbeServices>, NativeHttpRouter<ProbeServices>), HttpRouteError> {
        let services = Arc::new(ProbeServices::default());
        let router = NativeHttpRouter::new(
            services.clone(),
            HttpLimits::default(),
            HttpCapabilities::default(),
        )?;
        Ok((services, router))
    }

    fn error_code(response: &HttpResponse) -> Option<&str> {
        match &response.body {
            HttpBody::Json(Value::Object(body)) => body
                .get("error")
                .and_then(Value::as_object)
                .and_then(|error| error.get("code"))
                .and_then(Value::as_str),
            _ => None,
        }
    }

    fn test_authority(principal: &str) -> NativeRequestAuthority {
        NativeRequestAuthority {
            profile_id: "00000000-0000-0000-0000-000000000020".to_owned(),
            principal: principal.to_owned(),
            scopes: std::collections::BTreeSet::from([
                "api:read".to_owned(),
                "api:write".to_owned(),
            ]),
            plugin_id: None,
            plugin_digest: None,
        }
    }

    fn materialize_template(template: &str) -> String {
        materialize_catalog_template(template, "00000000-0000-0000-0000-000000000001")
    }

    fn materialize_catalog_template(template: &str, prompt_id: &str) -> String {
        if template == "/" {
            return "/".to_owned();
        }
        let parts = template
            .trim_start_matches('/')
            .split('/')
            .map(|segment| match template_parameter(segment) {
                Some(parameter) if matches!(parameter.name, "job_id" | "prompt_id" | "task_id") => {
                    prompt_id.to_owned()
                }
                Some(parameter) if parameter.constraint == Some("{UUID_RE}") => {
                    prompt_id.to_owned()
                }
                Some(parameter) if parameter.constraint == Some(".*") => match parameter.name {
                    "filename" => "fixture.webp".to_owned(),
                    _ => "fixture/file.json".to_owned(),
                },
                Some(parameter) => match parameter.name {
                    "node_class" => "LoadImage".to_owned(),
                    "directory_type" => "input".to_owned(),
                    "path_index" => "0".to_owned(),
                    "file" => "fixture.json".to_owned(),
                    "dest" => "moved.json".to_owned(),
                    _ => "fixture".to_owned(),
                },
                None => segment.to_owned(),
            })
            .collect::<Vec<_>>();
        format!("/{}", parts.join("/"))
    }

    fn unsafe_virtual_path(template: &str, prompt_id: &str) -> Option<String> {
        const PATH_NAMES: [&str; 8] = [
            "file",
            "dest",
            "path",
            "filename",
            "folder",
            "folder_name",
            "module_name",
            "extension_name",
        ];
        if !template.split('/').any(|segment| {
            template_parameter(segment)
                .is_some_and(|parameter| PATH_NAMES.contains(&parameter.name))
        }) {
            return None;
        }
        let parts = template
            .trim_start_matches('/')
            .split('/')
            .map(|segment| match template_parameter(segment) {
                Some(parameter) if PATH_NAMES.contains(&parameter.name) => {
                    if parameter.constraint == Some(".*") {
                        "%2E%2E/escape".to_owned()
                    } else {
                        "%2E%2E".to_owned()
                    }
                }
                Some(parameter) if matches!(parameter.name, "job_id" | "prompt_id" | "task_id") => {
                    prompt_id.to_owned()
                }
                Some(parameter) if parameter.constraint == Some("{UUID_RE}") => {
                    prompt_id.to_owned()
                }
                Some(parameter) if parameter.constraint == Some(".*") => {
                    "fixture/file.json".to_owned()
                }
                Some(parameter) => match parameter.name {
                    "node_class" => "LoadImage".to_owned(),
                    "directory_type" => "input".to_owned(),
                    "path_index" => "0".to_owned(),
                    _ => "fixture".to_owned(),
                },
                None => segment.to_owned(),
            })
            .collect::<Vec<_>>();
        Some(format!("/{}", parts.join("/")))
    }

    fn valid_prompt_body(prompt_id: &str) -> Value {
        json!({
            "prompt_id": prompt_id,
            "number": 7,
            "prompt": {
                "1": {"class_type": "LoadImage", "inputs": {"image": "fixture.png"}},
                "2": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}}
            }
        })
    }

    fn valid_json_fixture(route: &HttpRouteDescriptor, prompt_id: &str) -> Value {
        match (
            route.contract.identity.method,
            route.contract.identity.canonical_path.as_str(),
        ) {
            (HttpMethod::Post, "/prompt") => valid_prompt_body(prompt_id),
            (HttpMethod::Post, "/queue" | "/history") => {
                json!({"clear": false, "delete": []})
            }
            (HttpMethod::Post, "/api/jobs/cancel") => json!({"job_ids": []}),
            (HttpMethod::Post, "/interrupt") => json!({}),
            _ => json!({}),
        }
    }

    fn catalog_request(
        route: &HttpRouteDescriptor,
        path: String,
        prompt_id: &str,
    ) -> Result<HttpRequest, serde_json::Error> {
        let mut request = HttpRequest::new(route.contract.identity.method, path);
        let body_kind = route
            .contract
            .unknown
            .get("request_body")
            .and_then(Value::as_str)
            .unwrap_or("none");
        match body_kind {
            "application/json" => {
                request = request
                    .with_header("content-type", "application/json")
                    .with_body(serde_json::to_vec(&valid_json_fixture(route, prompt_id))?);
            }
            "multipart/form-data" | "raw bytes" => {
                request = request
                    .with_header("content-type", body_kind)
                    .with_body(Bytes::from_static(b"fixture"));
            }
            _ => {}
        }
        if route.is_mutation() {
            request = request.with_header(
                "idempotency-key",
                format!("catalog-contract-{}", route.feature_id()),
            );
        }
        Ok(request)
    }

    fn response_body_kind(response: &HttpResponse) -> &'static str {
        match &response.body {
            HttpBody::Empty => "empty",
            HttpBody::Bytes(_) => "bytes",
            HttpBody::Json(_) => "json",
            HttpBody::Stream(_) => "stream",
        }
    }

    fn catalog_success_status_basis(
        route: &HttpRouteDescriptor,
        status: u16,
    ) -> Option<&'static str> {
        if route.contract.status_codes.contains(&status) {
            return Some("catalog-status");
        }
        let status_detail = route
            .contract
            .unknown
            .get("status_content_types")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if status == 200
            && status_detail.contains("response constructors=")
            && (status_detail.contains("json_response")
                || status_detail.contains("Response")
                || status_detail.contains("dynamic/plain"))
        {
            return Some("catalog-constructor-default");
        }
        None
    }

    fn content_type_matches_catalog(route: &HttpRouteDescriptor, response: &HttpResponse) -> bool {
        if matches!(&response.body, HttpBody::Empty) || route.contract.content_types.is_empty() {
            return true;
        }
        let observed = response
            .content_type
            .split(';')
            .next()
            .unwrap_or(response.content_type.as_str());
        route
            .contract
            .content_types
            .iter()
            .any(|expected| expected == observed)
    }

    fn observed_scenario(
        name: &str,
        response: &HttpResponse,
        dispatched: bool,
        expected_status: Option<u16>,
        expected_error: Option<&str>,
    ) -> Value {
        let observed_error = error_code(response);
        let passed = expected_status.is_none_or(|status| response.status == status)
            && expected_error.is_none_or(|code| observed_error == Some(code));
        json!({
            "name": name,
            "passed": passed,
            "status": response.status,
            "content_type": response.content_type,
            "body_kind": response_body_kind(response),
            "error_code": observed_error,
            "native_dispatch": dispatched,
        })
    }

    fn catalog_contract_results() -> Result<BTreeMap<String, Value>, Box<dyn std::error::Error>> {
        let profile_id: sim_runtime::ProfileId =
            serde_json::from_value(Value::String("00000000-0000-0000-0000-000000000020".into()))?;
        let inner = crate::services::NativeRuntimeHttpServices::native_image_for_test(
            profile_id,
            Arc::new(crate::services::AcceptingExecutionController),
        )?;
        let capabilities = inner.http_capabilities()?;
        let services = Arc::new(CountingRuntimeServices {
            inner,
            calls: AtomicUsize::new(0),
        });
        let router = NativeHttpRouter::new(
            services.clone(),
            HttpLimits::default(),
            capabilities.clone(),
        )?;
        let authority = NativeRequestAuthority {
            profile_id: profile_id.0.to_string(),
            principal: "native-contract-validation".into(),
            scopes: std::collections::BTreeSet::from(["api:read".into(), "api:write".into()]),
            plugin_id: None,
            plugin_digest: None,
        };
        let seed_prompt_id = "00000000-0000-0000-0000-000000002020";
        let seed = HttpRequest::new(HttpMethod::Post, "/prompt")
            .with_header("content-type", "application/json")
            .with_header("idempotency-key", "catalog-state-seed")
            .with_body(serde_json::to_vec(&valid_prompt_body(seed_prompt_id))?);
        let seed_response = router.route_authorized(seed, Some(authority.clone()))?;
        if seed_response.status != 200 {
            return Err(format!(
                "catalog native state seed returned {}",
                seed_response.status
            )
            .into());
        }
        let mut results = BTreeMap::new();
        for route in http_route_catalog()? {
            let sequence = route
                .feature_id()
                .strip_prefix("COMFY-API-")
                .ok_or("catalog feature ID prefix changed")?;
            let fixture_prompt_id = format!("00000000-0000-0000-0000-00000000{sequence}");
            let path = materialize_catalog_template(&route.contract.identity.path, seed_prompt_id);
            let request = catalog_request(route, path.clone(), &fixture_prompt_id)?;
            let calls_before = services.calls.load(Ordering::SeqCst);
            let response = router
                .route_authorized(request, Some(authority.clone()))
                .unwrap_or_else(HttpRouteError::into_response);
            let calls_after = services.calls.load(Ordering::SeqCst);
            let dispatched = calls_after == calls_before.saturating_add(1);
            let available = matches!(capabilities.state_for(route), CapabilityState::Available);
            let explicit_capability = !available
                && !dispatched
                && response.status == 501
                && error_code(&response) == Some("route_capability_unavailable");
            let mut scenarios = Vec::new();
            if available && !dispatched {
                return Err(format!(
                    "{} enabled route returned {} without native dispatch",
                    route.feature_id(),
                    response.status
                )
                .into());
            }
            if !available && !explicit_capability {
                return Err(format!(
                    "{} unavailable route returned {} without an explicit capability response",
                    route.feature_id(),
                    response.status
                )
                .into());
            }
            let (primary_passed, status_basis) = if available {
                let status_basis = catalog_success_status_basis(route, response.status);
                (
                    (200..300).contains(&response.status)
                        && status_basis.is_some()
                        && content_type_matches_catalog(route, &response),
                    status_basis,
                )
            } else {
                (explicit_capability, Some("native-capability-response"))
            };
            scenarios.push(json!({
                "name": if available { "valid" } else { "unavailable" },
                "passed": primary_passed,
                "status": response.status,
                "status_basis": status_basis,
                "content_type": response.content_type,
                "content_type_matches_catalog": content_type_matches_catalog(route, &response),
                "body_kind": response_body_kind(&response),
                "error_code": error_code(&response),
                "native_dispatch": dispatched,
            }));

            let body_kind = route
                .contract
                .unknown
                .get("request_body")
                .and_then(Value::as_str)
                .unwrap_or("none");
            if available && body_kind == "application/json" {
                for (name, body, expected_error) in [
                    ("empty-required-body", Bytes::new(), "missing_request_body"),
                    ("malformed-json", Bytes::from_static(b"{"), "malformed_json"),
                ] {
                    let invalid = HttpRequest::new(route.contract.identity.method, path.clone())
                        .with_header("content-type", "application/json")
                        .with_header(
                            "idempotency-key",
                            format!("catalog-{name}-{}", route.feature_id()),
                        )
                        .with_body(body);
                    let calls_before = services.calls.load(Ordering::SeqCst);
                    let invalid_response = router
                        .route_authorized(invalid, Some(authority.clone()))
                        .unwrap_or_else(HttpRouteError::into_response);
                    let invalid_dispatched = services.calls.load(Ordering::SeqCst) != calls_before;
                    scenarios.push(observed_scenario(
                        name,
                        &invalid_response,
                        invalid_dispatched,
                        Some(400),
                        Some(expected_error),
                    ));
                }
            }
            if available
                && route
                    .contract
                    .request_headers
                    .iter()
                    .any(|header| header == "range")
            {
                let range_request = HttpRequest::new(route.contract.identity.method, path.clone())
                    .with_header("range", "items=0-1");
                let calls_before = services.calls.load(Ordering::SeqCst);
                let range_response = router
                    .route_authorized(range_request, Some(authority.clone()))
                    .unwrap_or_else(HttpRouteError::into_response);
                scenarios.push(observed_scenario(
                    "invalid-range-boundary",
                    &range_response,
                    services.calls.load(Ordering::SeqCst) != calls_before,
                    Some(416),
                    Some("invalid_range"),
                ));
            }
            if let Some(unsafe_path) =
                unsafe_virtual_path(&route.contract.identity.path, seed_prompt_id)
            {
                let unsafe_request = HttpRequest::new(route.contract.identity.method, unsafe_path);
                let calls_before = services.calls.load(Ordering::SeqCst);
                let unsafe_response = router
                    .route_authorized(unsafe_request, Some(authority.clone()))
                    .unwrap_or_else(HttpRouteError::into_response);
                scenarios.push(observed_scenario(
                    "unsafe-path-boundary",
                    &unsafe_response,
                    services.calls.load(Ordering::SeqCst) != calls_before,
                    Some(400),
                    Some("unsafe_virtual_path"),
                ));
            }
            if let Some(parameter) = route.contract.query_parameters.first() {
                let query_request = HttpRequest::new(route.contract.identity.method, path)
                    .with_query(
                        parameter,
                        "x".repeat(HttpLimits::default().maximum_query_value_bytes + 1),
                    );
                let calls_before = services.calls.load(Ordering::SeqCst);
                let query_response = router
                    .route_authorized(query_request, Some(authority.clone()))
                    .unwrap_or_else(HttpRouteError::into_response);
                scenarios.push(observed_scenario(
                    "oversized-query-boundary",
                    &query_response,
                    services.calls.load(Ordering::SeqCst) != calls_before,
                    Some(413),
                    Some("query_value_too_large"),
                ));
            }
            let passed = scenarios.iter().all(|scenario| scenario["passed"] == true);
            if !passed {
                return Err(format!(
                    "{} failed executable catalog evidence: {}",
                    route.feature_id(),
                    serde_json::to_string(&scenarios)?
                )
                .into());
            }
            results.insert(
                route.feature_id().to_owned(),
                json!({
                    "passed": passed,
                    "native_dispatch": dispatched,
                    "explicit_capability": explicit_capability,
                    "catalog_availability": format!("{:?}", route.availability),
                    "catalog_status_codes": route.contract.status_codes,
                    "catalog_content_types": route.contract.content_types,
                    "catalog_error_behavior": route.error_behavior,
                    "scenarios": scenarios,
                }),
            );
        }
        Ok(results)
    }

    #[test]
    fn http_001_catalog_has_exact_rows_ids_and_contract_dimensions()
    -> Result<(), Box<dyn std::error::Error>> {
        let catalog = http_route_catalog()?;
        assert_eq!(catalog.len(), HTTP_ROUTE_COUNT);
        for (index, route) in catalog.iter().enumerate() {
            assert_eq!(route.feature_id(), format!("COMFY-API-{:04}", index + 1));
            assert!(route.contract.identity.path.starts_with('/'));
            assert!(route.contract.identity.canonical_path.starts_with('/'));
            if route.contract.status_codes.is_empty() {
                assert_eq!(route.schema_confidence, "documented-only");
                assert!(!route.unresolved_schema.is_empty());
            }
            assert!(route.contract.response_schema.is_some());
            assert!(route.contract.unknown.contains_key("request_schema_detail"));
            assert!(
                route
                    .contract
                    .unknown
                    .contains_key("response_schema_detail")
            );
            assert!(route.contract.unknown.contains_key("status_content_types"));
            let concrete_path = materialize_template(&route.contract.identity.path);
            let matched = match_http_route(route.contract.identity.method, &concrete_path)?
                .ok_or_else(|| format!("{} did not match {concrete_path}", route.feature_id()))?;
            assert_eq!(
                matched.requested_feature_id,
                route.feature_id(),
                "catalog row {} did not win routing for {concrete_path}",
                route.feature_id()
            );
        }
        Ok(())
    }

    #[test]
    fn http_001_models_only_source_backed_status_sets() -> Result<(), CatalogError> {
        let catalog = http_route_catalog()?;
        let status_codes = |feature_id: &str| {
            catalog
                .iter()
                .find(|route| route.feature_id() == feature_id)
                .map(|route| route.contract.status_codes.as_slice())
        };
        assert_eq!(status_codes("COMFY-API-0009"), Some([201].as_slice()));
        assert_eq!(status_codes("COMFY-API-0016"), Some([204].as_slice()));
        assert_eq!(status_codes("COMFY-API-0066"), Some([204].as_slice()));
        assert_eq!(status_codes("COMFY-API-0036"), Some([200].as_slice()));
        assert_eq!(status_codes("COMFY-API-0101"), Some([200, 503].as_slice()));
        Ok(())
    }

    #[test]
    fn http_001_get_and_head_are_read_only_without_idempotency()
    -> Result<(), Box<dyn std::error::Error>> {
        for route in http_route_catalog()?.iter().filter(|route| {
            matches!(
                route.contract.identity.method,
                HttpMethod::Get | HttpMethod::Head
            )
        }) {
            assert!(
                !route.is_mutation(),
                "{} was classified as a mutation",
                route.feature_id()
            );
            assert!(
                !route
                    .contract
                    .request_headers
                    .iter()
                    .any(|header| header == "idempotency-key"),
                "{} requires an idempotency key for a read",
                route.feature_id()
            );
        }

        let (services, router) = router()?;
        let response = router.route_authorized(
            HttpRequest::new(HttpMethod::Get, "/queue"),
            Some(test_authority("reader")),
        )?;
        assert_eq!(response.status, 200);
        assert_eq!(
            services
                .mutation_identities
                .lock()
                .map_err(|_| "probe mutation identity state was poisoned")?
                .as_slice(),
            &[MutationIdentity::Untracked]
        );
        Ok(())
    }

    #[test]
    fn http_001_scopes_service_mutation_identities_by_authority()
    -> Result<(), Box<dyn std::error::Error>> {
        let (services, router) = router()?;
        for principal in ["writer-one", "writer-two"] {
            let response = router.route_authorized(
                HttpRequest::new(HttpMethod::Post, "/queue")
                    .with_header("content-type", "application/json")
                    .with_header("idempotency-key", "shared-client-key")
                    .with_body(Bytes::from_static(br#"{"clear":false}"#)),
                Some(test_authority(principal)),
            )?;
            assert_eq!(response.status, 200);
        }
        let identities = services
            .mutation_identities
            .lock()
            .map_err(|_| "probe mutation identity state was poisoned")?;
        assert_eq!(identities.len(), 2);
        assert_ne!(identities[0], identities[1]);
        for identity in identities.iter() {
            let MutationIdentity::IdempotencyKey(value) = identity else {
                return Err("authorized mutation identity was not idempotency scoped".into());
            };
            assert!(value.starts_with("authority:"));
            assert!(value.ends_with(":shared-client-key"));
            assert!(!value.contains("writer-one"));
            assert!(!value.contains("writer-two"));
        }
        Ok(())
    }

    #[test]
    fn http_001_rejects_duplicate_security_and_framing_headers() -> Result<(), HttpRouteError> {
        let (_, router) = router()?;
        let duplicate_authorization = router.handle(
            HttpRequest::new(HttpMethod::Get, "/queue")
                .with_header("authorization", "Bearer first")
                .with_header("Authorization", "Bearer second"),
        );
        assert_eq!(duplicate_authorization.status, 400);
        assert_eq!(
            error_code(&duplicate_authorization),
            Some("duplicate_request_header")
        );

        let ambiguous_framing = router.handle(
            HttpRequest::new(HttpMethod::Post, "/queue")
                .with_header("content-type", "application/json")
                .with_header("content-length", "2")
                .with_header("transfer-encoding", "chunked")
                .with_body(Bytes::from_static(b"{}")),
        );
        assert_eq!(ambiguous_framing.status, 400);
        assert_eq!(
            error_code(&ambiguous_framing),
            Some("ambiguous_request_framing")
        );

        let mismatched_length = router.handle(
            HttpRequest::new(HttpMethod::Post, "/queue")
                .with_header("content-type", "application/json")
                .with_header("content-length", "3")
                .with_body(Bytes::from_static(b"{}")),
        );
        assert_eq!(mismatched_length.status, 400);
        assert_eq!(
            error_code(&mismatched_length),
            Some("content_length_mismatch")
        );
        Ok(())
    }

    #[test]
    fn http_001_every_catalog_route_dispatches_or_returns_explicit_capability()
    -> Result<(), Box<dyn std::error::Error>> {
        let results = catalog_contract_results()?;
        assert_eq!(results.len(), HTTP_ROUTE_COUNT);
        assert_eq!(
            results
                .values()
                .filter(|result| result["native_dispatch"] == true)
                .count(),
            36
        );
        assert_eq!(
            results
                .values()
                .filter(|result| result["explicit_capability"] == true)
                .count(),
            105
        );
        Ok(())
    }

    #[test]
    fn http_001_aliases_resolve_to_canonical_descriptors() -> Result<(), Box<dyn std::error::Error>>
    {
        let alias = match_http_route(HttpMethod::Post, "/api/prompt")?
            .ok_or("POST /api/prompt did not match")?;
        assert_eq!(alias.requested_feature_id, "COMFY-API-0051");
        assert_eq!(alias.canonical_feature_id, "COMFY-API-0118");
        assert_eq!(alias.canonical_path, "/prompt");

        let double_api = match_http_route(HttpMethod::Post, "/api/api/jobs/cancel")?
            .ok_or("double-api jobs alias did not match")?;
        assert_eq!(double_api.requested_feature_id, "COMFY-API-0004");
        assert_eq!(double_api.canonical_feature_id, "COMFY-API-0042");

        let static_match = match_http_route(HttpMethod::Get, "/not-an-api-file")?
            .ok_or("static catch-all did not match")?;
        assert_eq!(static_match.requested_feature_id, "COMFY-API-0141");
        Ok(())
    }

    #[test]
    fn http_001_dispatches_only_typed_native_service_operations()
    -> Result<(), Box<dyn std::error::Error>> {
        let (services, router) = router()?;
        let response = router.route(HttpRequest::new(HttpMethod::Get, "/queue"))?;
        assert_eq!(response.status, 200);
        assert_eq!(services.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            services
                .operations
                .lock()
                .map_err(|_| "probe poisoned")?
                .as_slice(),
            &[NativeServiceOperation::QueueSnapshot]
        );
        assert!(!HTTP_FORWARDING_SUPPORTED);
        Ok(())
    }

    #[test]
    fn http_001_rejects_malformed_oversized_and_unknown_requests()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_, router) = router()?;
        let malformed = router.handle(
            HttpRequest::new(HttpMethod::Post, "/prompt")
                .with_header("content-type", "application/json")
                .with_body(Bytes::from_static(b"{")),
        );
        assert_eq!(malformed.status, 400);
        assert_eq!(error_code(&malformed), Some("malformed_json"));

        let empty = router.handle(HttpRequest::new(HttpMethod::Post, "/prompt"));
        assert_eq!(empty.status, 400);
        assert_eq!(error_code(&empty), Some("missing_request_body"));

        let oversized = router.handle(
            HttpRequest::new(HttpMethod::Post, "/userdata/file")
                .with_body(Bytes::from(vec![0; MAX_COMPATIBILITY_JSON_BYTES + 1])),
        );
        assert_eq!(oversized.status, 413);
        assert_eq!(error_code(&oversized), Some("request_too_large"));

        let unknown = router.handle(HttpRequest::new(HttpMethod::Post, "/not-a-route"));
        assert_eq!(unknown.status, 404);
        assert_eq!(error_code(&unknown), Some("route_not_found"));

        let static_not_found = router.handle(HttpRequest::new(HttpMethod::Get, "/not-a-route"));
        assert_eq!(static_not_found.status, 404);
        assert_eq!(error_code(&static_not_found), Some("asset_not_found"));
        Ok(())
    }

    #[test]
    fn http_001_exposes_conditional_capability_instead_of_false_not_found()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_, router) = router()?;
        let unavailable = router.handle(HttpRequest::new(HttpMethod::Get, "/api/assets"));
        assert_eq!(unavailable.status, 501);
        assert_eq!(
            error_code(&unavailable),
            Some("route_capability_unavailable")
        );

        router.set_capability("COMFY-API-0007", CapabilityState::Available)?;
        let available = router.handle(HttpRequest::new(HttpMethod::Get, "/api/assets"));
        assert_eq!(available.status, 200);
        Ok(())
    }

    #[test]
    fn http_001_router_projects_mutation_identity_without_owning_replay()
    -> Result<(), Box<dyn std::error::Error>> {
        let (services, router) = router()?;
        let first = HttpRequest::new(HttpMethod::Post, "/prompt")
            .with_header("content-type", "application/json")
            .with_header("idempotency-key", "request-one")
            .with_body(Bytes::from_static(br#"{"prompt":{}}"#));
        assert_eq!(router.route(first.clone())?.status, 200);
        assert_eq!(router.route(first)?.status, 200);
        assert_eq!(services.calls.load(Ordering::SeqCst), 2);

        let changed = router.route(
            HttpRequest::new(HttpMethod::Post, "/prompt")
                .with_header("content-type", "application/json")
                .with_header("idempotency-key", "request-one")
                .with_body(Bytes::from_static(br#"{"prompt":{"changed":true}}"#)),
        )?;
        assert_eq!(changed.status, 200);
        assert_eq!(services.calls.load(Ordering::SeqCst), 3);

        let durable = HttpRequest::new(HttpMethod::Post, "/interrupt")
            .with_header("content-type", "application/json")
            .with_body(Bytes::from_static(br#"{"attempt_id":"attempt-7"}"#));
        assert_eq!(router.route(durable.clone())?.status, 200);
        assert_eq!(router.route(durable)?.status, 200);
        assert_eq!(services.calls.load(Ordering::SeqCst), 5);

        let identities = services
            .mutation_identities
            .lock()
            .map_err(|_| "probe mutation identity state was poisoned")?;
        assert!(matches!(
            identities.first(),
            Some(MutationIdentity::IdempotencyKey(key)) if key == "request-one"
        ));
        assert!(matches!(
            identities.last(),
            Some(MutationIdentity::DurableAttempt(operation_id))
                if operation_id == "attempt_id:attempt-7"
        ));
        Ok(())
    }

    #[test]
    fn http_001_enforces_filename_and_range_boundaries() -> Result<(), Box<dyn std::error::Error>> {
        let (_, router) = router()?;
        let unsafe_path = router.handle(HttpRequest::new(
            HttpMethod::Get,
            "/templates/%2E%2E/secret",
        ));
        assert_eq!(unsafe_path.status, 400);
        assert_eq!(error_code(&unsafe_path), Some("unsafe_virtual_path"));

        let ranged = router.route(
            HttpRequest::new(HttpMethod::Get, "/prompt").with_header("range", "bytes=2-5"),
        )?;
        assert_eq!(ranged.status, 206);
        assert_eq!(
            ranged.headers.get("content-range").map(String::as_str),
            Some("bytes 2-5/10")
        );
        match ranged.body {
            HttpBody::Bytes(bytes) => assert_eq!(bytes, Bytes::from_static(b"2345")),
            other => return Err(format!("unexpected ranged body {other:?}").into()),
        }

        let invalid = router
            .handle(HttpRequest::new(HttpMethod::Get, "/prompt").with_header("range", "items=1-2"));
        assert_eq!(invalid.status, 416);
        assert_eq!(error_code(&invalid), Some("invalid_range"));
        Ok(())
    }

    #[test]
    fn http_001_stream_channel_applies_backpressure_and_size_limits()
    -> Result<(), Box<dyn std::error::Error>> {
        let limits = HttpLimits {
            maximum_response_bytes: 8,
            maximum_stream_chunk_bytes: 4,
            stream_channel_capacity: 1,
            ..HttpLimits::default()
        };
        let (sender, stream) = bounded_response_stream(&limits)?;
        smol::block_on(async {
            sender.send(Bytes::from_static(b"1234")).await?;
            let first = stream.next().await.ok_or(StreamError::Closed)??;
            assert_eq!(first, Bytes::from_static(b"1234"));
            assert_eq!(
                sender.send(Bytes::from_static(b"12345")).await,
                Err(StreamError::ChunkTooLarge {
                    actual: 5,
                    maximum: 4
                })
            );
            sender.send(Bytes::from_static(b"5678")).await?;
            let second = stream.next().await.ok_or(StreamError::Closed)??;
            assert_eq!(second, Bytes::from_static(b"5678"));
            assert_eq!(
                sender.send(Bytes::from_static(b"9")).await,
                Err(StreamError::TotalTooLarge { maximum: 8 })
            );
            Ok::<_, StreamError>(())
        })?;
        Ok(())
    }

    #[test]
    fn http_001_upload_interface_rejects_unsafe_and_oversized_files()
    -> Result<(), Box<dyn std::error::Error>> {
        let limits = HttpLimits {
            maximum_upload_bytes: 4,
            ..HttpLimits::default()
        };
        let unsafe_upload = ValidatedUpload::validate(
            "../escape.png",
            "image/png",
            Bytes::from_static(b"png"),
            &limits,
        )
        .err()
        .ok_or("canonical asset identity unexpectedly accepted parent traversal")?;
        let canonical_error =
            AssetIdentity::new("http-wire", AssetNamespace::Temporary, "../escape.png")
                .err()
                .ok_or("canonical asset identity unexpectedly accepted parent traversal")?;
        assert_eq!(unsafe_upload.status, 400);
        assert_eq!(unsafe_upload.code, "unsafe_upload_filename");
        assert_eq!(
            unsafe_upload.message,
            format!("virtual path is not a canonical asset identity: {canonical_error}")
        );
        let oversized = ValidatedUpload::validate(
            "safe.png",
            "image/png",
            Bytes::from_static(b"12345"),
            &limits,
        )
        .err()
        .ok_or("oversized upload unexpectedly passed")?;
        assert_eq!(oversized.status, 413);
        Ok(())
    }

    #[test]
    fn http_001_maps_typed_native_failures_to_route_addressable_statuses()
    -> Result<(), Box<dyn std::error::Error>> {
        let cases = [
            (NativeServiceErrorKind::Invalid, 400),
            (NativeServiceErrorKind::Unauthorized, 401),
            (NativeServiceErrorKind::Forbidden, 403),
            (NativeServiceErrorKind::NotFound, 404),
            (NativeServiceErrorKind::Conflict, 409),
            (NativeServiceErrorKind::Oversized, 413),
            (NativeServiceErrorKind::Cancelled, 409),
            (NativeServiceErrorKind::Timeout, 504),
            (NativeServiceErrorKind::Unavailable, 503),
            (NativeServiceErrorKind::Internal, 500),
        ];
        for (kind, expected_status) in cases {
            let router = NativeHttpRouter::new(
                Arc::new(FailingServices(kind)),
                HttpLimits::default(),
                HttpCapabilities::default(),
            )?;
            let error = router
                .route(HttpRequest::new(HttpMethod::Get, "/queue"))
                .err()
                .ok_or("fixture service unexpectedly succeeded")?;
            assert_eq!(error.status, expected_status);
            assert_eq!(error.code, "fixture_service_error");
            assert_eq!(
                error.route.as_ref().map(|route| route.feature_id.as_str()),
                Some("COMFY-API-0119")
            );
        }
        Ok(())
    }

    #[test]
    pub(crate) fn val_http_001() -> Result<(), Box<dyn std::error::Error>> {
        let catalog = http_route_catalog()?;
        let cases = catalog_contract_results()?;
        let mut protocol_cases = BTreeMap::new();
        macro_rules! record_protocol_case {
            ($identifier:literal, $coverage:literal, $check:block) => {{
                let result: Result<(), Box<dyn std::error::Error>> = (|| $check)();
                let passed = result.is_ok();
                let diagnostic = result.as_ref().err().map(ToString::to_string);
                protocol_cases.insert(
                    $identifier.to_owned(),
                    json!({
                        "passed": passed,
                        "coverage": $coverage,
                        "diagnostic": diagnostic,
                    }),
                );
                result?;
            }};
        }
        record_protocol_case!(
            "catalog-schema-routing-and-status-model",
            "exact row identities, all contract dimensions, source-backed statuses, aliases, and canonical routing",
            {
                http_001_catalog_has_exact_rows_ids_and_contract_dimensions()?;
                http_001_models_only_source_backed_status_sets()?;
                http_001_aliases_resolve_to_canonical_descriptors()?;
                Ok(())
            }
        );
        record_protocol_case!(
            "read-only-method-and-authority-scoping",
            "GET/HEAD mutation exclusion and principal/profile-scoped service mutation identities",
            {
                http_001_get_and_head_are_read_only_without_idempotency()?;
                http_001_scopes_service_mutation_identities_by_authority()?;
                Ok(())
            }
        );
        record_protocol_case!(
            "duplicate-security-and-framing-headers",
            "duplicate authorization, Content-Length/Transfer-Encoding ambiguity, and body-length mismatch fail before dispatch",
            {
                http_001_rejects_duplicate_security_and_framing_headers()?;
                crate::transport::tests::rejects_ambiguous_http_framing_and_security_headers();
                Ok(())
            }
        );
        record_protocol_case!(
            "typed-native-dispatch-and-no-forwarding",
            "typed Rust service dispatch with the external Comfy forwarding capability disabled",
            {
                http_001_dispatches_only_typed_native_service_operations()?;
                Ok(())
            }
        );
        record_protocol_case!(
            "canonical-bound-node-object-info",
            "the exact early native input options, output names, descriptions, aliases, categories, module compatibility field, and list flags project only from compiled registry bindings",
            {
                crate::services::tests::validate_native_object_info_fixture()?;
                Ok(())
            }
        );
        record_protocol_case!(
            "malformed-empty-oversized-and-not-found",
            "malformed JSON, empty required body, oversized body, unknown route, and missing static asset",
            {
                http_001_rejects_malformed_oversized_and_unknown_requests()?;
                Ok(())
            }
        );
        record_protocol_case!(
            "conditional-capability-negotiation",
            "unavailable conditional route is explicit and becomes dispatchable only when enabled",
            {
                http_001_exposes_conditional_capability_instead_of_false_not_found()?;
                Ok(())
            }
        );
        record_protocol_case!(
            "router-mutation-identity-and-host-idempotency",
            "the router projects canonical mutation identities without owning transitions; the host owns replay, conflict, durable identity, and restart persistence",
            {
                http_001_router_projects_mutation_identity_without_owning_replay()?;
                crate::tests::native_host_idempotency_contracts()?;
                Ok(())
            }
        );
        record_protocol_case!(
            "host-concurrent-and-ambiguous-mutation-reconciliation",
            "the host blocks concurrent duplicates and delegates ambiguous timeout reconciliation to the canonical service receipt owner",
            {
                crate::tests::native_host_concurrent_and_ambiguous_reconciliation()?;
                Ok(())
            }
        );
        record_protocol_case!(
            "path-range-upload-and-stream-bounds",
            "path traversal rejection, successful and invalid ranges, bounded streaming/backpressure, and upload size/name validation",
            {
                http_001_enforces_filename_and_range_boundaries()?;
                http_001_stream_channel_applies_backpressure_and_size_limits()?;
                http_001_upload_interface_rejects_unsafe_and_oversized_files()?;
                Ok(())
            }
        );
        record_protocol_case!(
            "route-addressable-native-service-errors",
            "invalid, unauthorized, forbidden, not-found, conflict, oversized, cancelled, timeout, unavailable, and internal failures",
            {
                http_001_maps_typed_native_failures_to_route_addressable_statuses()?;
                Ok(())
            }
        );
        record_protocol_case!(
            "host-auth-cors-plugin-and-principal-isolation",
            "exact-origin CORS, bearer scopes, principal-bound plugin grants, SHA-256 authority idempotency, and WebSocket client isolation",
            {
                crate::tests::cors_preflight_and_error_responses_use_exact_origin_policy()?;
                crate::tests::principals_plugins_and_websocket_clients_are_strictly_isolated()?;
                Ok(())
            }
        );
        record_protocol_case!(
            "real-loopback-http-websocket-and-shutdown",
            "native loopback HTTP/HEAD/malformed-target handling plus WebSocket upgrade, status, ping, and shutdown close",
            {
                crate::transport::tests::serves_http_and_websocket_over_real_loopback_sockets()?;
                Ok(())
            }
        );
        record_protocol_case!(
            "real-rustls-https-and-remote-header-defense",
            "TLS policy refuses a missing acceptor; trusted rustls HTTPS serves native data and rejects duplicate authorization and ambiguous framing",
            {
                crate::transport::tests::refuses_to_start_when_tls_policy_has_no_real_acceptor()?;
                crate::transport::tests::serves_https_with_a_trusted_rustls_handshake_and_rejects_duplicate_headers()?;
                Ok(())
            }
        );
        let native_dispatches = cases
            .values()
            .filter(|case| case["native_dispatch"] == true)
            .count();
        let explicit_capabilities = cases
            .values()
            .filter(|case| case["explicit_capability"] == true)
            .count();
        let executed_scenarios = cases
            .values()
            .filter_map(|case| case["scenarios"].as_array().map(Vec::len))
            .sum::<usize>();
        if cases.len() != HTTP_ROUTE_COUNT
            || native_dispatches + explicit_capabilities != HTTP_ROUTE_COUNT
        {
            return Err("HTTP artifact does not account for every catalog row".into());
        }
        let fixture_digest = format!("{:x}", Sha256::digest(ROUTE_CATALOG_CSV.as_bytes()));
        let services_source_digest = format!(
            "{:x}",
            Sha256::digest(include_str!("services.rs").as_bytes())
        );
        let native_image_descriptor_digest = format!(
            "{:x}",
            Sha256::digest(
                include_str!("../../sim_nodes/src/slices/native_image.descriptors.json").as_bytes()
            )
        );
        let artifact = json!({
            "validation_id": "VAL-HTTP-001",
            "validation": "VAL-HTTP-001",
            "scope": "native-http-route-contracts",
            "environment": {
                "operating_system": std::env::consts::OS,
                "architecture": std::env::consts::ARCH,
                "backend": "native-rust-services",
                "protocol_version": sim_types::NATIVE_PROTOCOL_VERSION,
                "proxy_or_forwarding": HTTP_FORWARDING_SUPPORTED,
            },
            "fixture_digests": {
                "backend_http_routes_sha256": fixture_digest,
                "native_services_source_sha256": services_source_digest,
                "native_image_descriptor_sha256": native_image_descriptor_digest,
            },
            "catalog": {
                "expected_rows": HTTP_ROUTE_COUNT,
                "observed_rows": catalog.len(),
                "first_feature_id": catalog.first().map(HttpRouteDescriptor::feature_id),
                "last_feature_id": catalog.last().map(HttpRouteDescriptor::feature_id),
            },
            "summary": {
                "passed": cases.len() + protocol_cases.len(),
                "failed": 0,
                "skipped": 0,
                "catalog_passed": cases.len(),
                "protocol_passed": protocol_cases.len(),
                "native_dispatch": native_dispatches,
                "explicit_capability": explicit_capabilities,
                "executed_scenarios": executed_scenarios,
            },
            "cases": cases,
            "protocol_cases": protocol_cases,
            "skipped": [],
        });
        let mut bytes = serde_json::to_vec_pretty(&artifact)?;
        bytes.push(b'\n');
        let artifact_directory = std::env::var_os("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target")
            })
            .join("sim-parity");
        std::fs::create_dir_all(&artifact_directory)?;
        std::fs::write(artifact_directory.join("val-http-001.json"), bytes)?;
        Ok(())
    }
}
