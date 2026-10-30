use crate::{
    CanonicalEvent, EventCodecError, EventId, EventSignature, PublicKey, SignedEvent,
    TimestampPolicy, verify_signed_event,
};
use base64::{Engine as _, engine::general_purpose};
use collaboration_domain::{
    MediaContentHash, MediaDescriptor, MediaMetadata, MediaMetadataError, MediaObjectSelection,
    MediaTenantPath, MediaVariantKind, OperationId, TenantContext,
};
use serde::{Deserialize, Serialize};
use url::Url;
use uuid::Uuid;

pub const BLOSSOM_AUTH_KIND: u16 = 24_242;
pub const MAX_BLOSSOM_AUTH_HEADER_BYTES: usize = 64 * 1024;
pub const MAX_BLOSSOM_AUTH_CONTENT_BYTES: usize = 4 * 1024;
pub const MAX_BLOSSOM_AUTH_TAGS: usize = 64;
pub const MAX_BLOSSOM_AUTH_TAG_VALUES: usize = 8;
pub const MAX_BLOSSOM_AUTH_TAG_VALUE_BYTES: usize = 2 * 1024;
pub const MAX_BLOSSOM_TOKEN_AGE_SECONDS: u64 = 60 * 60;
pub const BLOSSOM_FUTURE_SKEW_SECONDS: u64 = 5;
pub const MAX_BLOSSOM_RANGE_HEADER_BYTES: usize = 128;

const BLOSSOM_AUTH_OPERATION_NAMESPACE: Uuid =
    Uuid::from_u128(0x9499_b430_d946_5a94_987a_32d9_9848_2962);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlossomUploadRoute {
    Bud02,
    LegacyMediaAlias,
}

impl BlossomUploadRoute {
    pub fn parse(path: &str) -> Result<Self, BlossomAdapterError> {
        match path {
            "/upload" => Ok(Self::Bud02),
            "/media/upload" => Ok(Self::LegacyMediaAlias),
            _ => Err(BlossomAdapterError::NotFound),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlossomVerb {
    Upload,
    Get,
}

impl BlossomVerb {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Upload => "upload",
            Self::Get => "get",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlossomAuthorizationScope {
    Blob(MediaContentHash),
    Server,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlossomAuthorization {
    author: PublicKey,
    event_id: EventId,
    operation_id: OperationId,
    verb: BlossomVerb,
    scope: BlossomAuthorizationScope,
    expires_at: u64,
}

impl BlossomAuthorization {
    pub const fn author(self) -> PublicKey {
        self.author
    }

    pub const fn event_id(self) -> EventId {
        self.event_id
    }

    pub const fn operation_id(self) -> OperationId {
        self.operation_id
    }

    pub const fn verb(self) -> BlossomVerb {
        self.verb
    }

    pub const fn scope(self) -> BlossomAuthorizationScope {
        self.scope
    }

    pub const fn expires_at(self) -> u64 {
        self.expires_at
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlossomUploadIntent {
    route: BlossomUploadRoute,
    authorization: BlossomAuthorization,
    content_hash: MediaContentHash,
}

impl BlossomUploadIntent {
    pub const fn route(&self) -> BlossomUploadRoute {
        self.route
    }

    pub const fn authorization(&self) -> BlossomAuthorization {
        self.authorization
    }

    pub const fn content_hash(&self) -> MediaContentHash {
        self.content_hash
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlossomMediaPath {
    original_hash: MediaContentHash,
    selection: MediaObjectSelection,
    extension: Option<String>,
}

impl BlossomMediaPath {
    pub fn parse(path_segment: &str) -> Result<Self, BlossomAdapterError> {
        if path_segment.is_empty()
            || path_segment.len() > 80
            || path_segment.bytes().any(|byte| byte.is_ascii_control())
        {
            return Err(BlossomAdapterError::NotFound);
        }
        let segments = path_segment.split('.').collect::<Vec<_>>();
        let hash = segments
            .first()
            .copied()
            .ok_or(BlossomAdapterError::NotFound)?;
        let original_hash =
            MediaContentHash::from_lower_hex(hash).map_err(|_| BlossomAdapterError::NotFound)?;
        match segments.as_slice() {
            [_] => Ok(Self {
                original_hash,
                selection: MediaObjectSelection::Original,
                extension: None,
            }),
            [_, extension] if is_safe_extension(extension) => Ok(Self {
                original_hash,
                selection: MediaObjectSelection::Original,
                extension: Some((*extension).to_owned()),
            }),
            [_, "thumb", "jpg"] => Ok(Self {
                original_hash,
                selection: MediaObjectSelection::Variant(MediaVariantKind::Thumbnail),
                extension: Some("jpg".to_owned()),
            }),
            _ => Err(BlossomAdapterError::NotFound),
        }
    }

    pub const fn original_hash(&self) -> MediaContentHash {
        self.original_hash
    }

    pub const fn selection(&self) -> MediaObjectSelection {
        self.selection
    }

    pub fn extension(&self) -> Option<&str> {
        self.extension.as_deref()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlossomContentDisposition {
    Inline,
    Attachment,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlossomResolvedObject {
    tenant_path: MediaTenantPath,
    descriptor: MediaDescriptor,
    disposition: BlossomContentDisposition,
}

impl BlossomResolvedObject {
    pub const fn tenant_path(&self) -> MediaTenantPath {
        self.tenant_path
    }

    pub const fn descriptor(&self) -> &MediaDescriptor {
        &self.descriptor
    }

    pub const fn disposition(&self) -> BlossomContentDisposition {
        self.disposition
    }

    pub const fn cache_control(&self) -> &'static str {
        "private, max-age=31536000, immutable"
    }

    pub const fn content_security_policy(&self) -> &'static str {
        "default-src 'none'"
    }

    pub const fn nosniff(&self) -> bool {
        true
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlossomResolvedRange {
    start: u64,
    end_inclusive: u64,
}

impl BlossomResolvedRange {
    pub const fn start(self) -> u64 {
        self.start
    }

    pub const fn end_inclusive(self) -> u64 {
        self.end_inclusive
    }

    pub fn byte_length(self) -> Result<u64, BlossomAdapterError> {
        self.end_inclusive
            .checked_sub(self.start)
            .and_then(|length| length.checked_add(1))
            .ok_or(BlossomAdapterError::InvalidRange)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlossomRangeSelection {
    Full,
    Partial(BlossomResolvedRange),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlossomPublicOrigin(Url);

impl BlossomPublicOrigin {
    pub fn new(value: &str) -> Result<Self, BlossomAdapterError> {
        if value.is_empty()
            || value.len() > 2_048
            || value.bytes().any(|byte| byte.is_ascii_control())
        {
            return Err(BlossomAdapterError::InvalidConfiguration);
        }
        let url = Url::parse(value).map_err(|_| BlossomAdapterError::InvalidConfiguration)?;
        if !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || url.path() != "/"
            || url.host_str().is_none()
            || !origin_scheme_allowed(&url)
        {
            return Err(BlossomAdapterError::InvalidConfiguration);
        }
        Ok(Self(url))
    }

    pub fn as_url(&self) -> &Url {
        &self.0
    }

    fn media_base(&self) -> String {
        format!("{}media", self.0.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct BlossomBlobDescriptor {
    pub url: String,
    pub sha256: String,
    pub size: u64,
    #[serde(rename = "type")]
    pub mime_type: String,
    pub uploaded: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dim: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blurhash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thumb: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum BlossomAdapterError {
    #[error("authentication failed")]
    InvalidAuthentication,
    #[error("insufficient scope")]
    InsufficientScope,
    #[error("not found")]
    NotFound,
    #[error("range not satisfiable")]
    InvalidRange,
    #[error("unsupported media type")]
    UnsupportedMedia,
    #[error("invalid canonical media metadata")]
    InvalidCanonicalMetadata,
    #[error("media service unavailable")]
    ServiceUnavailable,
    #[error("Blossom adapter configuration is invalid")]
    InvalidConfiguration,
}

impl BlossomAdapterError {
    pub const fn http_status(self) -> u16 {
        match self {
            Self::InvalidAuthentication => 401,
            Self::InsufficientScope => 403,
            Self::NotFound => 404,
            Self::InvalidRange => 416,
            Self::UnsupportedMedia => 415,
            Self::InvalidCanonicalMetadata => 500,
            Self::ServiceUnavailable => 503,
            Self::InvalidConfiguration => 500,
        }
    }

    pub const fn public_message(self) -> &'static str {
        match self {
            Self::InvalidAuthentication => "authentication failed",
            Self::InsufficientScope => "insufficient scope",
            Self::NotFound => "not found",
            Self::InvalidRange => "range not satisfiable",
            Self::UnsupportedMedia => "unsupported media type",
            Self::InvalidCanonicalMetadata
            | Self::ServiceUnavailable
            | Self::InvalidConfiguration => "media service unavailable",
        }
    }
}

#[derive(Deserialize)]
struct WireSignedEvent {
    id: String,
    pubkey: String,
    created_at: u64,
    kind: u16,
    tags: Vec<Vec<String>>,
    content: String,
    sig: String,
}

pub fn authorize_blossom_upload(
    route_path: &str,
    authorization_header: &str,
    claimed_hash: Option<&str>,
    trusted_server_host: &str,
    now: u64,
) -> Result<BlossomUploadIntent, BlossomAdapterError> {
    let route = BlossomUploadRoute::parse(route_path)?;
    let signed_event = decode_authorization_header(authorization_header)?;
    let common =
        verify_blossom_authorization(&signed_event, BlossomVerb::Upload, trusted_server_host, now)?;
    let content_hash = claimed_hash
        .ok_or(BlossomAdapterError::InvalidAuthentication)
        .and_then(|hash| {
            MediaContentHash::from_lower_hex(hash)
                .map_err(|_| BlossomAdapterError::InvalidAuthentication)
        })?;
    if !has_hash_scope(&signed_event.event, content_hash) {
        return Err(BlossomAdapterError::InvalidAuthentication);
    }
    Ok(BlossomUploadIntent {
        route,
        authorization: BlossomAuthorization {
            scope: BlossomAuthorizationScope::Blob(content_hash),
            ..common
        },
        content_hash,
    })
}

pub fn authorize_blossom_download(
    authorization_header: &str,
    media_path: &BlossomMediaPath,
    trusted_server_host: &str,
    now: u64,
) -> Result<BlossomAuthorization, BlossomAdapterError> {
    let signed_event = decode_authorization_header(authorization_header)?;
    let mut authorization =
        verify_blossom_authorization(&signed_event, BlossomVerb::Get, trusted_server_host, now)?;
    if has_hash_scope(&signed_event.event, media_path.original_hash) {
        authorization.scope = BlossomAuthorizationScope::Blob(media_path.original_hash);
    } else if has_server_scope(&signed_event.event, trusted_server_host)? {
        authorization.scope = BlossomAuthorizationScope::Server;
    } else {
        return Err(BlossomAdapterError::InsufficientScope);
    }
    Ok(authorization)
}

pub fn resolve_blossom_object(
    metadata: &MediaMetadata,
    tenant: &TenantContext,
    media_path: &BlossomMediaPath,
) -> Result<BlossomResolvedObject, BlossomAdapterError> {
    if metadata.fields().identity.content_hash() != media_path.original_hash {
        return Err(BlossomAdapterError::NotFound);
    }
    let (descriptor, content_type) = match media_path.selection {
        MediaObjectSelection::Original => {
            let descriptor = MediaDescriptor::new(
                metadata.fields().identity.content_hash(),
                metadata.fields().content_type.clone(),
                metadata.fields().byte_size,
            );
            (descriptor, metadata.fields().content_type.as_str())
        }
        MediaObjectSelection::Variant(kind) => {
            let variant = metadata
                .fields()
                .variants
                .iter()
                .find(|variant| variant.kind() == kind)
                .ok_or(BlossomAdapterError::NotFound)?;
            (
                variant.descriptor().clone(),
                variant.descriptor().content_type().as_str(),
            )
        }
    };
    let expected_extension = media_extension(content_type);
    if media_path
        .extension
        .as_deref()
        .is_some_and(|extension| extension != expected_extension)
    {
        return Err(BlossomAdapterError::NotFound);
    }
    let tenant_path = metadata
        .tenant_path(tenant, media_path.selection)
        .map_err(map_metadata_error)?;
    Ok(BlossomResolvedObject {
        tenant_path,
        descriptor,
        disposition: content_disposition(content_type),
    })
}

pub fn resolve_blossom_range(
    range_header: Option<&str>,
    total_size: u64,
    maximum_range_bytes: u64,
) -> Result<BlossomRangeSelection, BlossomAdapterError> {
    if total_size == 0 || maximum_range_bytes == 0 {
        return Err(BlossomAdapterError::InvalidConfiguration);
    }
    let Some(range_header) = range_header else {
        return Ok(BlossomRangeSelection::Full);
    };
    if range_header.len() > MAX_BLOSSOM_RANGE_HEADER_BYTES
        || range_header.bytes().any(|byte| byte.is_ascii_control())
    {
        return Err(BlossomAdapterError::InvalidRange);
    }
    if range_header.contains(',') {
        return Ok(BlossomRangeSelection::Full);
    }
    let range = range_header
        .strip_prefix("bytes=")
        .ok_or(BlossomAdapterError::InvalidRange)?;
    let (start, requested_end) = if let Some(suffix) = range.strip_prefix('-') {
        let length = parse_canonical_u64(suffix).ok_or(BlossomAdapterError::InvalidRange)?;
        if length == 0 {
            return Err(BlossomAdapterError::InvalidRange);
        }
        (total_size.saturating_sub(length), total_size - 1)
    } else {
        let (start, end) = range
            .split_once('-')
            .ok_or(BlossomAdapterError::InvalidRange)?;
        let start = parse_canonical_u64(start).ok_or(BlossomAdapterError::InvalidRange)?;
        let end = if end.is_empty() {
            u64::MAX
        } else {
            parse_canonical_u64(end).ok_or(BlossomAdapterError::InvalidRange)?
        };
        if start > end {
            return Err(BlossomAdapterError::InvalidRange);
        }
        (start, end)
    };
    if start >= total_size {
        return Err(BlossomAdapterError::InvalidRange);
    }
    let end_inclusive = requested_end
        .min(total_size - 1)
        .min(start.saturating_add(maximum_range_bytes - 1));
    Ok(BlossomRangeSelection::Partial(BlossomResolvedRange {
        start,
        end_inclusive,
    }))
}

pub fn project_blossom_descriptor(
    metadata: &MediaMetadata,
    tenant: &TenantContext,
    origin: &BlossomPublicOrigin,
) -> Result<BlossomBlobDescriptor, BlossomAdapterError> {
    metadata
        .tenant_path(tenant, MediaObjectSelection::Original)
        .map_err(map_metadata_error)?;
    let fields = metadata.fields();
    let hash = fields.identity.content_hash().to_lower_hex();
    let extension = media_extension(fields.content_type.as_str());
    let media_base = origin.media_base();
    let uploaded_seconds = fields.uploaded_at_millis / 1_000;
    let uploaded = i64::try_from(uploaded_seconds)
        .map_err(|_| BlossomAdapterError::InvalidCanonicalMetadata)?;
    let thumb = fields
        .variants
        .iter()
        .find(|variant| {
            variant.kind() == MediaVariantKind::Thumbnail
                && variant.descriptor().content_type().as_str() == "image/jpeg"
        })
        .map(|_| format!("{media_base}/{hash}.thumb.jpg"));
    Ok(BlossomBlobDescriptor {
        url: format!("{media_base}/{hash}.{extension}"),
        sha256: hash,
        size: fields.byte_size.get(),
        mime_type: fields.content_type.as_str().to_owned(),
        uploaded,
        dim: None,
        blurhash: None,
        thumb,
        duration: None,
    })
}

fn decode_authorization_header(header: &str) -> Result<SignedEvent, BlossomAdapterError> {
    if header.len() > MAX_BLOSSOM_AUTH_HEADER_BYTES
        || header.bytes().any(|byte| byte.is_ascii_control())
    {
        return Err(BlossomAdapterError::InvalidAuthentication);
    }
    let encoded = header
        .strip_prefix("Nostr ")
        .ok_or(BlossomAdapterError::InvalidAuthentication)?;
    let bytes = general_purpose::URL_SAFE_NO_PAD
        .decode(encoded)
        .or_else(|_| general_purpose::URL_SAFE.decode(encoded))
        .or_else(|_| general_purpose::STANDARD.decode(encoded))
        .map_err(|_| BlossomAdapterError::InvalidAuthentication)?;
    if bytes.len() > MAX_BLOSSOM_AUTH_HEADER_BYTES {
        return Err(BlossomAdapterError::InvalidAuthentication);
    }
    let event: WireSignedEvent =
        serde_json::from_slice(&bytes).map_err(|_| BlossomAdapterError::InvalidAuthentication)?;
    validate_auth_shape(&event)?;
    Ok(SignedEvent {
        claimed_id: EventId::from_hex(&event.id).map_err(map_event_codec_error)?,
        event: CanonicalEvent::new(
            PublicKey::from_hex(&event.pubkey).map_err(map_event_codec_error)?,
            event.created_at,
            event.kind,
            event.tags,
            event.content,
        ),
        signature: EventSignature::from_hex(&event.sig)
            .map_err(|_| BlossomAdapterError::InvalidAuthentication)?,
    })
}

fn validate_auth_shape(event: &WireSignedEvent) -> Result<(), BlossomAdapterError> {
    if event.content.trim().is_empty()
        || event.content.len() > MAX_BLOSSOM_AUTH_CONTENT_BYTES
        || event.content.bytes().any(|byte| byte.is_ascii_control())
        || event.tags.len() > MAX_BLOSSOM_AUTH_TAGS
        || event.tags.iter().any(|tag| {
            tag.is_empty()
                || tag.len() > MAX_BLOSSOM_AUTH_TAG_VALUES
                || tag.iter().any(|value| {
                    value.len() > MAX_BLOSSOM_AUTH_TAG_VALUE_BYTES
                        || value.bytes().any(|byte| byte.is_ascii_control())
                })
        })
    {
        return Err(BlossomAdapterError::InvalidAuthentication);
    }
    Ok(())
}

fn verify_blossom_authorization(
    signed_event: &SignedEvent,
    verb: BlossomVerb,
    trusted_server_host: &str,
    now: u64,
) -> Result<BlossomAuthorization, BlossomAdapterError> {
    if now == 0 || signed_event.event.kind != BLOSSOM_AUTH_KIND {
        return Err(BlossomAdapterError::InvalidAuthentication);
    }
    verify_signed_event(
        signed_event,
        TimestampPolicy::Bounded {
            now,
            max_past_seconds: MAX_BLOSSOM_TOKEN_AGE_SECONDS,
            max_future_seconds: BLOSSOM_FUTURE_SKEW_SECONDS,
        },
    )
    .map_err(|_| BlossomAdapterError::InvalidAuthentication)?;

    let mut verb_seen = false;
    let mut expiration = None;
    let mut server_tags = Vec::new();
    for tag in &signed_event.event.tags {
        let Some(name) = tag.first().map(String::as_str) else {
            return Err(BlossomAdapterError::InvalidAuthentication);
        };
        match name {
            "t" => {
                let value = single_tag_value(tag)?;
                if value != verb.as_str() {
                    return Err(BlossomAdapterError::InvalidAuthentication);
                }
                verb_seen = true;
            }
            "expiration" => {
                let value = parse_canonical_u64(single_tag_value(tag)?)
                    .ok_or(BlossomAdapterError::InvalidAuthentication)?;
                if expiration.replace(value).is_some() {
                    return Err(BlossomAdapterError::InvalidAuthentication);
                }
            }
            "server" => server_tags.push(single_tag_value(tag)?),
            _ => {}
        }
    }
    if !verb_seen || expiration.is_none_or(|expiration| expiration <= now) {
        return Err(BlossomAdapterError::InvalidAuthentication);
    }
    if !server_tags.is_empty()
        && !server_tags
            .iter()
            .any(|server| server_hosts_match(server, trusted_server_host))
    {
        return Err(BlossomAdapterError::InvalidAuthentication);
    }
    let event_id = signed_event.claimed_id;
    Ok(BlossomAuthorization {
        author: signed_event.event.public_key,
        event_id,
        operation_id: OperationId::from_uuid(Uuid::new_v5(
            &BLOSSOM_AUTH_OPERATION_NAMESPACE,
            event_id.as_bytes(),
        )),
        verb,
        scope: BlossomAuthorizationScope::Server,
        expires_at: expiration.ok_or(BlossomAdapterError::InvalidAuthentication)?,
    })
}

fn has_hash_scope(event: &CanonicalEvent, hash: MediaContentHash) -> bool {
    let hash = hash.to_lower_hex();
    event.tags.iter().any(|tag| {
        tag.first().map(String::as_str) == Some("x")
            && tag.get(1).map(String::as_str) == Some(hash.as_str())
    })
}

fn has_server_scope(
    event: &CanonicalEvent,
    trusted_server_host: &str,
) -> Result<bool, BlossomAdapterError> {
    let servers = event
        .tags
        .iter()
        .filter(|tag| tag.first().map(String::as_str) == Some("server"))
        .map(|tag| single_tag_value(tag))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(servers
        .iter()
        .any(|server| server_hosts_match(server, trusted_server_host)))
}

fn single_tag_value(tag: &[String]) -> Result<&str, BlossomAdapterError> {
    if tag.len() != 2 || tag[1].is_empty() {
        return Err(BlossomAdapterError::InvalidAuthentication);
    }
    Ok(&tag[1])
}

fn server_hosts_match(server: &str, trusted_server_host: &str) -> bool {
    let Some(server) = normalize_server_host(server) else {
        return false;
    };
    let Some(trusted) = normalize_server_host(trusted_server_host) else {
        return false;
    };
    server == trusted
}

fn normalize_server_host(value: &str) -> Option<String> {
    if value.is_empty() || value.len() > 2_048 || value.bytes().any(|byte| byte.is_ascii_control())
    {
        return None;
    }
    let authority = match value.split_once("://") {
        Some(("http" | "https", remainder)) => remainder.split('/').next()?,
        Some(_) => return None,
        None => value.split('/').next()?,
    };
    if authority.is_empty()
        || authority.contains('@')
        || authority.contains('?')
        || authority.contains('#')
    {
        return None;
    }
    let mut authority = authority.to_ascii_lowercase();
    if let Some(without_port) = authority
        .strip_suffix(":443")
        .or_else(|| authority.strip_suffix(":80"))
    {
        authority = without_port.to_owned();
    }
    if let Some(without_root) = authority.strip_suffix('.') {
        authority = without_root.to_owned();
    }
    (!authority.is_empty()).then_some(authority)
}

fn parse_canonical_u64(value: &str) -> Option<u64> {
    if value.is_empty()
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    value.parse().ok()
}

fn content_disposition(content_type: &str) -> BlossomContentDisposition {
    if matches!(
        content_type,
        "image/gif" | "image/jpeg" | "image/png" | "image/webp" | "video/mp4"
    ) {
        BlossomContentDisposition::Inline
    } else {
        BlossomContentDisposition::Attachment
    }
}

fn media_extension(content_type: &str) -> &'static str {
    match content_type {
        "image/jpeg" => "jpg",
        "image/png" => "png",
        "image/gif" => "gif",
        "image/webp" => "webp",
        "video/mp4" => "mp4",
        "audio/aac" => "aac",
        "audio/flac" => "flac",
        "audio/mp4" => "m4a",
        "audio/mpeg" => "mp3",
        "audio/ogg" => "ogg",
        "audio/wav" => "wav",
        "application/pdf" => "pdf",
        _ => "bin",
    }
}

fn is_safe_extension(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 8
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
}

fn origin_scheme_allowed(url: &Url) -> bool {
    if url.scheme() == "https" {
        return true;
    }
    url.scheme() == "http" && matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "::1"))
}

fn map_event_codec_error(_error: EventCodecError) -> BlossomAdapterError {
    BlossomAdapterError::InvalidAuthentication
}

fn map_metadata_error(error: MediaMetadataError) -> BlossomAdapterError {
    match error {
        MediaMetadataError::TenantMismatch | MediaMetadataError::MissingVariant => {
            BlossomAdapterError::NotFound
        }
        _ => BlossomAdapterError::InvalidCanonicalMetadata,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use collaboration_domain::{
        CommunityId, MediaByteSize, MediaContentType, MediaIdentity, MediaVariant, PrincipalId,
        TrustedTenantRoute,
    };
    use secp256k1::{Keypair, Message, Secp256k1, SecretKey};
    use sha2::{Digest, Sha256};

    fn tenant(community_id: CommunityId) -> TenantContext {
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_direct_host(community_id, "relay.example").expect("route"),
            ),
            &[],
        )
        .expect("tenant")
    }

    fn signed_authorization_header(now: u64, verb: &str, hash: &str, server: &str) -> String {
        let secret_key = SecretKey::from_slice(&[7; 32]).expect("secret key");
        let keypair = Keypair::from_secret_key(&Secp256k1::new(), &secret_key);
        let (public_key, _) = keypair.x_only_public_key();
        let event = CanonicalEvent::new(
            PublicKey::from_bytes(public_key.serialize()),
            now,
            BLOSSOM_AUTH_KIND,
            vec![
                vec!["t".into(), verb.into()],
                vec!["expiration".into(), (now + 300).to_string()],
                vec!["x".into(), hash.into()],
                vec!["server".into(), server.into()],
            ],
            format!("Authorize Blossom {verb}"),
        );
        let event_id = event.event_id().expect("event id");
        let signature = Secp256k1::new()
            .sign_schnorr_no_aux_rand(&Message::from_digest(*event_id.as_bytes()), &keypair);
        let wire = serde_json::json!({
            "id": event_id.to_hex(),
            "pubkey": event.public_key.to_hex(),
            "created_at": event.created_at,
            "kind": event.kind,
            "tags": event.tags,
            "content": event.content,
            "sig": signature.to_string(),
        });
        format!(
            "Nostr {}",
            general_purpose::URL_SAFE_NO_PAD.encode(serde_json::to_vec(&wire).expect("wire json"))
        )
    }

    fn media_metadata(content_type: &str, bytes: &[u8]) -> MediaMetadata {
        let identity = MediaIdentity::new(
            CommunityId::new(),
            MediaContentHash::from_digest(Sha256::digest(bytes).into()),
        )
        .expect("identity");
        MediaMetadata::new(
            identity,
            PrincipalId::new(),
            MediaContentType::new(content_type).expect("content type"),
            MediaByteSize::new(u64::try_from(bytes.len()).expect("size")).expect("byte size"),
            1_700_000_000_123,
        )
        .expect("metadata")
    }

    #[test]
    fn signed_upload_supports_bud02_and_legacy_alias_with_stable_operation() {
        let now = 1_700_000_000;
        let hash = MediaContentHash::from_digest(Sha256::digest(b"image").into()).to_lower_hex();
        let header = signed_authorization_header(now, "upload", &hash, "relay.example");

        let canonical =
            authorize_blossom_upload("/upload", &header, Some(&hash), "Relay.Example.:443", now)
                .expect("BUD-02 upload");
        let legacy =
            authorize_blossom_upload("/media/upload", &header, Some(&hash), "relay.example", now)
                .expect("legacy upload");

        assert_eq!(canonical.route(), BlossomUploadRoute::Bud02);
        assert_eq!(legacy.route(), BlossomUploadRoute::LegacyMediaAlias);
        assert_eq!(
            canonical.authorization().operation_id(),
            legacy.authorization().operation_id()
        );
        assert_eq!(canonical.content_hash().to_lower_hex(), hash);
    }

    #[test]
    fn upload_and_download_authorization_fail_closed() {
        let now = 1_700_000_000;
        let hash = MediaContentHash::from_digest(Sha256::digest(b"image").into()).to_lower_hex();
        let other_hash =
            MediaContentHash::from_digest(Sha256::digest(b"other").into()).to_lower_hex();
        let upload = signed_authorization_header(now, "upload", &hash, "relay.example");
        assert_eq!(
            authorize_blossom_upload("/upload", &upload, None, "relay.example", now),
            Err(BlossomAdapterError::InvalidAuthentication)
        );
        assert_eq!(
            authorize_blossom_upload("/upload", &upload, Some(&other_hash), "relay.example", now,),
            Err(BlossomAdapterError::InvalidAuthentication)
        );
        assert_eq!(
            authorize_blossom_upload("/upload", &upload, Some(&hash), "other.example", now,),
            Err(BlossomAdapterError::InvalidAuthentication)
        );
        assert_eq!(
            authorize_blossom_upload("/upload", &upload, Some(&hash), "relay.example", now + 301,),
            Err(BlossomAdapterError::InvalidAuthentication)
        );
        let encoded = upload.strip_prefix("Nostr ").expect("authorization scheme");
        let decoded = general_purpose::URL_SAFE_NO_PAD
            .decode(encoded)
            .expect("authorization body");
        let mut tampered: serde_json::Value =
            serde_json::from_slice(&decoded).expect("authorization JSON");
        tampered["content"] = serde_json::Value::String("Tampered authorization".into());
        let tampered = format!(
            "Nostr {}",
            general_purpose::STANDARD.encode(serde_json::to_vec(&tampered).expect("tampered JSON"))
        );
        assert_eq!(
            authorize_blossom_upload("/upload", &tampered, Some(&hash), "relay.example", now,),
            Err(BlossomAdapterError::InvalidAuthentication)
        );

        let get = signed_authorization_header(now, "get", &hash, "relay.example");
        let other_path = BlossomMediaPath::parse(&other_hash).expect("path");
        let server_scoped = authorize_blossom_download(&get, &other_path, "relay.example", now)
            .expect("matching server grants BUD-01 server scope");
        assert_eq!(server_scoped.scope(), BlossomAuthorizationScope::Server);
        assert_eq!(
            authorize_blossom_download(&get, &other_path, "other.example", now),
            Err(BlossomAdapterError::InvalidAuthentication)
        );
    }

    #[test]
    fn download_paths_resolve_original_thumbnail_and_safe_headers() {
        let mut metadata = media_metadata("image/png", b"image");
        let image_tenant = tenant(metadata.fields().identity.community_id());
        let thumb_bytes = b"thumb";
        metadata
            .add_variant(MediaVariant::new(
                MediaVariantKind::Thumbnail,
                MediaDescriptor::new(
                    MediaContentHash::from_digest(Sha256::digest(thumb_bytes).into()),
                    MediaContentType::new("image/jpeg").expect("thumbnail MIME"),
                    MediaByteSize::new(u64::try_from(thumb_bytes.len()).expect("size"))
                        .expect("thumb size"),
                ),
            ))
            .expect("thumbnail");
        let hash = metadata.fields().identity.content_hash().to_lower_hex();

        let original = BlossomMediaPath::parse(&format!("{hash}.png")).expect("original path");
        let original =
            resolve_blossom_object(&metadata, &image_tenant, &original).expect("original object");
        assert_eq!(original.disposition(), BlossomContentDisposition::Inline);
        assert_eq!(original.descriptor().content_type().as_str(), "image/png");
        assert_eq!(original.content_security_policy(), "default-src 'none'");
        assert!(original.nosniff());

        let bare = BlossomMediaPath::parse(&hash).expect("bare alias");
        assert_eq!(
            resolve_blossom_object(&metadata, &image_tenant, &bare)
                .expect("bare original alias")
                .descriptor(),
            original.descriptor()
        );

        let thumbnail =
            BlossomMediaPath::parse(&format!("{hash}.thumb.jpg")).expect("thumbnail path");
        let thumbnail =
            resolve_blossom_object(&metadata, &image_tenant, &thumbnail).expect("thumbnail object");
        assert_eq!(
            thumbnail.tenant_path().selection(),
            MediaObjectSelection::Variant(MediaVariantKind::Thumbnail)
        );
        assert_eq!(thumbnail.descriptor().content_type().as_str(), "image/jpeg");

        let wrong_extension =
            BlossomMediaPath::parse(&format!("{hash}.jpg")).expect("structured path");
        assert_eq!(
            resolve_blossom_object(&metadata, &image_tenant, &wrong_extension),
            Err(BlossomAdapterError::NotFound)
        );

        let document = media_metadata("application/pdf", b"document");
        let document_tenant = tenant(document.fields().identity.community_id());
        let document_path = BlossomMediaPath::parse(&format!(
            "{}.pdf",
            document.fields().identity.content_hash().to_lower_hex()
        ))
        .expect("document path");
        assert_eq!(
            resolve_blossom_object(&document, &document_tenant, &document_path)
                .expect("document object")
                .disposition(),
            BlossomContentDisposition::Attachment
        );
    }

    #[test]
    fn range_adapter_preserves_bud01_single_range_behavior() {
        assert_eq!(
            resolve_blossom_range(None, 100, 16),
            Ok(BlossomRangeSelection::Full)
        );
        assert_eq!(
            resolve_blossom_range(Some("bytes=10-90"), 100, 16),
            Ok(BlossomRangeSelection::Partial(BlossomResolvedRange {
                start: 10,
                end_inclusive: 25,
            }))
        );
        assert_eq!(
            resolve_blossom_range(Some("bytes=-10"), 100, 16),
            Ok(BlossomRangeSelection::Partial(BlossomResolvedRange {
                start: 90,
                end_inclusive: 99,
            }))
        );
        assert_eq!(
            resolve_blossom_range(Some("bytes=0-1,4-5"), 100, 16),
            Ok(BlossomRangeSelection::Full)
        );
        assert_eq!(
            resolve_blossom_range(Some("bytes=100-"), 100, 16),
            Err(BlossomAdapterError::InvalidRange)
        );
    }

    #[test]
    fn descriptor_projection_uses_tenant_origin_and_canonical_metadata() {
        let mut metadata = media_metadata("image/png", b"image");
        let tenant = tenant(metadata.fields().identity.community_id());
        metadata
            .add_variant(MediaVariant::new(
                MediaVariantKind::Thumbnail,
                MediaDescriptor::new(
                    MediaContentHash::from_digest(Sha256::digest(b"thumb").into()),
                    MediaContentType::new("image/jpeg").expect("MIME"),
                    MediaByteSize::new(5).expect("size"),
                ),
            ))
            .expect("variant");
        let origin = BlossomPublicOrigin::new("https://relay.example").expect("origin");
        let descriptor =
            project_blossom_descriptor(&metadata, &tenant, &origin).expect("descriptor");

        assert!(descriptor.url.starts_with("https://relay.example/media/"));
        assert!(descriptor.url.ends_with(".png"));
        assert_eq!(
            descriptor.sha256,
            metadata.fields().identity.content_hash().to_lower_hex()
        );
        assert_eq!(descriptor.mime_type, "image/png");
        assert_eq!(descriptor.uploaded, 1_700_000_000);
        assert!(descriptor.thumb.as_ref().is_some_and(|thumb| {
            thumb.starts_with("https://relay.example/media/") && thumb.ends_with(".thumb.jpg")
        }));
        let json = serde_json::to_value(&descriptor).expect("descriptor JSON");
        assert_eq!(json["type"], "image/png");
        assert!(json.get("duration").is_none());
        assert_eq!(
            BlossomPublicOrigin::new("http://relay.example"),
            Err(BlossomAdapterError::InvalidConfiguration)
        );
    }

    #[test]
    fn protocol_errors_have_closed_compatible_statuses() {
        let cases = [
            (
                BlossomAdapterError::InvalidAuthentication,
                401,
                "authentication failed",
            ),
            (
                BlossomAdapterError::InsufficientScope,
                403,
                "insufficient scope",
            ),
            (BlossomAdapterError::NotFound, 404, "not found"),
            (
                BlossomAdapterError::UnsupportedMedia,
                415,
                "unsupported media type",
            ),
            (
                BlossomAdapterError::InvalidRange,
                416,
                "range not satisfiable",
            ),
            (
                BlossomAdapterError::ServiceUnavailable,
                503,
                "media service unavailable",
            ),
        ];
        for (error, status, message) in cases {
            assert_eq!(error.http_status(), status);
            assert_eq!(error.public_message(), message);
        }
        assert_eq!(
            BlossomAdapterError::InvalidCanonicalMetadata.public_message(),
            BlossomAdapterError::ServiceUnavailable.public_message()
        );
        assert_eq!(
            BlossomMediaPath::parse("../../etc/passwd"),
            Err(BlossomAdapterError::NotFound)
        );
    }
}
