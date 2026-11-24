use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use chrono::DateTime;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

const MAX_SNAPSHOT_BYTES: usize = 128 * 1024 * 1024;
const MAX_LOCAL_STORAGE_ENTRIES: usize = 20_000;
const MAX_DRAFTS_PER_STORE: usize = 100;
const MAX_READ_CONTEXTS: usize = 10_000;
const MAX_FORCED_UNREAD_ENTRIES: usize = 500;
const MAX_ARCHIVED_EVENTS: usize = 1_000_000;
const MAX_ARCHIVE_SCOPES: usize = 4_000_000;
const MAX_ARCHIVE_SUBSCRIPTIONS: usize = 100_000;

const ARCHIVE_MIGRATION_CACHE_READ: &str = "add_cache_read_tokens";
const ARCHIVE_MIGRATION_CACHE_WRITE: &str = "add_cache_write_and_pricing";
const ARCHIVE_MIGRATION_HARNESS: &str = "add_harness_to_metric_index";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BuzzDesktopSourceSnapshot {
    pub snapshot_version: u32,
    pub captured_at_millis: u64,
    pub source_application_id: String,
    #[serde(default)]
    pub general_configuration: BTreeMap<String, Value>,
    #[serde(default)]
    pub local_storage: BTreeMap<String, String>,
    pub archive: BuzzArchiveSnapshot,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BuzzArchiveSnapshot {
    pub schema_version: u32,
    #[serde(default)]
    pub migration_markers: BTreeSet<String>,
    #[serde(default)]
    pub events: Vec<BuzzArchivedEvent>,
    #[serde(default)]
    pub scopes: Vec<BuzzArchivedEventScope>,
    #[serde(default)]
    pub subscriptions: Vec<BuzzArchiveSubscription>,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct BuzzArchivedEvent {
    pub identity_public_key: String,
    pub relay_url: String,
    pub event_id: String,
    pub kind: u32,
    pub author_public_key: String,
    pub created_at: u64,
    pub raw_json: String,
    pub archived_at: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct BuzzArchivedEventScope {
    pub identity_public_key: String,
    pub relay_url: String,
    pub event_id: String,
    pub scope_type: String,
    pub scope_value: String,
    pub archived_at: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct BuzzArchiveSubscription {
    pub identity_public_key: String,
    pub relay_url: String,
    pub scope_type: String,
    pub scope_value: String,
    pub kinds: Vec<u32>,
    pub created_at: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BuzzDesktopSettingImport {
    pub source_key: String,
    pub source_version: u32,
    pub value: Value,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BuzzDraftImport {
    pub source_storage_key: String,
    pub source_version: u32,
    pub identity_public_key: String,
    pub relay_scope: Option<String>,
    pub draft_key: String,
    pub content: String,
    pub selection_start: u32,
    pub selection_end: u32,
    pub channel_id: String,
    pub created_at: String,
    pub updated_at: String,
    pub pending_attachments: Vec<Value>,
    pub mention_references: Vec<BuzzDraftMentionReference>,
    pub spoilered_attachment_urls: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BuzzDraftMentionReference {
    pub display_name: String,
    pub public_key: String,
    pub is_agent: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BuzzReadStateImport {
    pub identity_public_key: String,
    pub contexts: BTreeMap<String, u64>,
    pub publishable_context_ids: BTreeSet<String>,
    pub context_source_created_at: BTreeMap<String, u64>,
    pub forced_unread: BTreeMap<String, BuzzForcedUnreadImport>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BuzzForcedUnreadImport {
    pub marker_at_when_forced: Option<u64>,
    pub sources: BTreeSet<BuzzForcedUnreadSource>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuzzForcedUnreadSource {
    Inbox,
    Manual,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct BuzzDesktopStateImport {
    pub snapshot_version: u32,
    pub captured_at_millis: u64,
    pub source_application_id: String,
    pub settings: Vec<BuzzDesktopSettingImport>,
    pub drafts: Vec<BuzzDraftImport>,
    pub read_state: Vec<BuzzReadStateImport>,
    pub archived_events: Vec<BuzzArchivedEvent>,
    pub archive_scopes: Vec<BuzzArchivedEventScope>,
    pub archive_subscriptions: Vec<BuzzArchiveSubscription>,
    pub skipped_cache_entries: u64,
    pub skipped_legacy_sent_drafts: u64,
    pub source_hash: [u8; 32],
    pub target_hash: [u8; 32],
}

#[derive(Debug, thiserror::Error)]
pub enum BuzzDesktopStateImportError {
    #[error("Buzz desktop snapshot is empty or exceeds its bounded import size")]
    InvalidSnapshotSize,
    #[error("Buzz desktop snapshot JSON is malformed")]
    InvalidSnapshotJson(#[source] serde_json::Error),
    #[error("Buzz desktop snapshot version {0} is unsupported")]
    UnsupportedSnapshotVersion(u32),
    #[error("Buzz desktop archive schema version {0} is unsupported")]
    UnsupportedArchiveVersion(u32),
    #[error("Buzz desktop archive migration markers do not match schema version {0}")]
    ArchiveMarkerMismatch(u32),
    #[error("Buzz desktop snapshot contains too many records")]
    ResourceLimit,
    #[error("Buzz desktop state record {key} is malformed: {reason}")]
    InvalidRecord { key: String, reason: &'static str },
    #[error("Buzz desktop archive record conflicts with another record")]
    IntegrityConflict,
    #[error("Buzz desktop snapshot contains secret material in general setting {0}")]
    SecretMaterial(String),
    #[error("Buzz desktop source file is unavailable")]
    SourceUnavailable(#[source] std::io::Error),
    #[error("Buzz desktop source changed while it was being imported")]
    SourceChanged,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredDraft {
    content: String,
    selection_start: u32,
    selection_end: u32,
    channel_id: String,
    created_at: String,
    updated_at: String,
    pending_imeta: Vec<Value>,
    #[serde(default)]
    mention_refs: Vec<StoredMentionReference>,
    spoilered_attachment_urls: Vec<String>,
    #[serde(default)]
    status: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredMentionReference {
    display_name: String,
    pubkey: String,
    is_agent: bool,
}

struct ImportedArchive {
    events: Vec<BuzzArchivedEvent>,
    scopes: Vec<BuzzArchivedEventScope>,
    subscriptions: Vec<BuzzArchiveSubscription>,
}

pub fn import_desktop_state_file(
    path: &Path,
) -> Result<BuzzDesktopStateImport, BuzzDesktopStateImportError> {
    let before = fs::read(path).map_err(BuzzDesktopStateImportError::SourceUnavailable)?;
    let result = import_desktop_state_bytes(&before)?;
    let after = fs::read(path).map_err(BuzzDesktopStateImportError::SourceUnavailable)?;
    if before != after {
        return Err(BuzzDesktopStateImportError::SourceChanged);
    }
    Ok(result)
}

pub fn import_desktop_state_bytes(
    source_bytes: &[u8],
) -> Result<BuzzDesktopStateImport, BuzzDesktopStateImportError> {
    if source_bytes.is_empty() || source_bytes.len() > MAX_SNAPSHOT_BYTES {
        return Err(BuzzDesktopStateImportError::InvalidSnapshotSize);
    }
    let source_hash = sha256(source_bytes);
    let snapshot: BuzzDesktopSourceSnapshot = serde_json::from_slice(source_bytes)
        .map_err(BuzzDesktopStateImportError::InvalidSnapshotJson)?;
    validate_snapshot_header(&snapshot)?;

    let settings = import_settings(&snapshot)?;
    let (drafts, skipped_legacy_sent_drafts) = import_drafts(&snapshot.local_storage)?;
    let read_state = import_read_state(&snapshot.local_storage)?;
    let archive = import_archive(&snapshot.archive)?;
    let consumed_keys = consumed_local_storage_keys(&snapshot.local_storage);
    let skipped_cache_entries = u64::try_from(
        snapshot
            .local_storage
            .len()
            .saturating_sub(consumed_keys.len()),
    )
    .map_err(|_| BuzzDesktopStateImportError::ResourceLimit)?;

    let mut result = BuzzDesktopStateImport {
        snapshot_version: snapshot.snapshot_version,
        captured_at_millis: snapshot.captured_at_millis,
        source_application_id: snapshot.source_application_id,
        settings,
        drafts,
        read_state,
        archived_events: archive.events,
        archive_scopes: archive.scopes,
        archive_subscriptions: archive.subscriptions,
        skipped_cache_entries,
        skipped_legacy_sent_drafts,
        source_hash,
        target_hash: [0; 32],
    };
    result.target_hash = hash_target(&result)?;
    Ok(result)
}

fn validate_snapshot_header(
    snapshot: &BuzzDesktopSourceSnapshot,
) -> Result<(), BuzzDesktopStateImportError> {
    if !matches!(snapshot.snapshot_version, 1 | 2) {
        return Err(BuzzDesktopStateImportError::UnsupportedSnapshotVersion(
            snapshot.snapshot_version,
        ));
    }
    if snapshot.captured_at_millis == 0
        || !is_source_application_id(&snapshot.source_application_id)
        || snapshot.local_storage.len() > MAX_LOCAL_STORAGE_ENTRIES
    {
        return Err(invalid_record("snapshot", "header fields are invalid"));
    }
    validate_archive_version(&snapshot.archive)
}

fn validate_archive_version(
    archive: &BuzzArchiveSnapshot,
) -> Result<(), BuzzDesktopStateImportError> {
    let expected: &[&str] = match archive.schema_version {
        1 => &[],
        2 => &[ARCHIVE_MIGRATION_CACHE_READ],
        3 => &[ARCHIVE_MIGRATION_CACHE_READ, ARCHIVE_MIGRATION_CACHE_WRITE],
        4 => &[
            ARCHIVE_MIGRATION_CACHE_READ,
            ARCHIVE_MIGRATION_CACHE_WRITE,
            ARCHIVE_MIGRATION_HARNESS,
        ],
        version => {
            return Err(BuzzDesktopStateImportError::UnsupportedArchiveVersion(
                version,
            ));
        }
    };
    let expected = expected
        .iter()
        .map(|marker| (*marker).to_owned())
        .collect::<BTreeSet<_>>();
    if archive.migration_markers != expected {
        return Err(BuzzDesktopStateImportError::ArchiveMarkerMismatch(
            archive.schema_version,
        ));
    }
    if archive.events.len() > MAX_ARCHIVED_EVENTS
        || archive.scopes.len() > MAX_ARCHIVE_SCOPES
        || archive.subscriptions.len() > MAX_ARCHIVE_SUBSCRIPTIONS
    {
        return Err(BuzzDesktopStateImportError::ResourceLimit);
    }
    Ok(())
}

fn import_settings(
    snapshot: &BuzzDesktopSourceSnapshot,
) -> Result<Vec<BuzzDesktopSettingImport>, BuzzDesktopStateImportError> {
    let mut settings = Vec::new();
    for (key, value) in &snapshot.general_configuration {
        if is_secret_key(key) || contains_secret_value(value) {
            return Err(BuzzDesktopStateImportError::SecretMaterial(key.clone()));
        }
        settings.push(BuzzDesktopSettingImport {
            source_key: key.clone(),
            source_version: snapshot.snapshot_version,
            value: value.clone(),
        });
    }
    for (key, raw_value) in &snapshot.local_storage {
        if !is_importable_setting_key(key) {
            continue;
        }
        settings.push(BuzzDesktopSettingImport {
            source_key: key.clone(),
            source_version: setting_version(key),
            value: parse_setting_value(raw_value),
        });
    }
    settings.sort_by(|left, right| left.source_key.cmp(&right.source_key));
    Ok(settings)
}

fn import_drafts(
    local_storage: &BTreeMap<String, String>,
) -> Result<(Vec<BuzzDraftImport>, u64), BuzzDesktopStateImportError> {
    let mut drafts = Vec::new();
    let mut skipped_sent = 0_u64;
    for (storage_key, raw) in local_storage {
        let Some((version, relay_scope, identity_public_key)) =
            parse_draft_storage_key(storage_key)
        else {
            continue;
        };
        validate_public_key(&identity_public_key, storage_key)?;
        let stored: BTreeMap<String, StoredDraft> = serde_json::from_str(raw)
            .map_err(|_| invalid_record(storage_key, "draft store JSON is malformed"))?;
        if stored.len() > MAX_DRAFTS_PER_STORE {
            return Err(BuzzDesktopStateImportError::ResourceLimit);
        }
        for (draft_key, draft) in stored {
            if draft_key.starts_with("sent:") || draft.status.as_deref() == Some("sent") {
                skipped_sent = skipped_sent.saturating_add(1);
                continue;
            }
            validate_draft(storage_key, &draft_key, &draft)?;
            drafts.push(BuzzDraftImport {
                source_storage_key: storage_key.clone(),
                source_version: version,
                identity_public_key: identity_public_key.clone(),
                relay_scope: relay_scope.clone(),
                draft_key,
                content: draft.content,
                selection_start: draft.selection_start,
                selection_end: draft.selection_end,
                channel_id: draft.channel_id,
                created_at: draft.created_at,
                updated_at: draft.updated_at,
                pending_attachments: draft.pending_imeta,
                mention_references: draft
                    .mention_refs
                    .into_iter()
                    .map(|reference| BuzzDraftMentionReference {
                        display_name: reference.display_name,
                        public_key: reference.pubkey,
                        is_agent: reference.is_agent,
                    })
                    .collect(),
                spoilered_attachment_urls: draft.spoilered_attachment_urls,
            });
        }
    }
    drafts.sort_by(|left, right| {
        (
            &left.identity_public_key,
            &left.relay_scope,
            &left.draft_key,
        )
            .cmp(&(
                &right.identity_public_key,
                &right.relay_scope,
                &right.draft_key,
            ))
    });
    Ok((drafts, skipped_sent))
}

fn parse_draft_storage_key(key: &str) -> Option<(u32, Option<String>, String)> {
    if let Some(identity) = key.strip_prefix("buzz-drafts.v1:") {
        return Some((1, None, identity.to_owned()));
    }
    let suffix = key.strip_prefix("buzz-drafts.v2:")?;
    let (relay_scope, identity) = suffix.rsplit_once(':')?;
    if relay_scope.is_empty() || identity.is_empty() {
        return None;
    }
    Some((2, Some(relay_scope.to_owned()), identity.to_owned()))
}

fn validate_draft(
    storage_key: &str,
    draft_key: &str,
    draft: &StoredDraft,
) -> Result<(), BuzzDesktopStateImportError> {
    let content_utf16_length = u32::try_from(draft.content.encode_utf16().count()).ok();
    let created_at = parse_timestamp(&draft.created_at);
    let updated_at = parse_timestamp(&draft.updated_at);
    if draft_key.is_empty()
        || draft_key.len() > 512
        || draft.channel_id.is_empty()
        || draft.channel_id.len() > 512
        || draft.selection_start > draft.selection_end
        || content_utf16_length.is_none()
        || Some(draft.selection_end) > content_utf16_length
        || draft.content.len() > 1_000_000
        || draft.pending_imeta.len() > 64
        || draft
            .pending_imeta
            .iter()
            .any(|attachment| !attachment.is_object())
        || draft.spoilered_attachment_urls.len() > 64
        || !matches!(draft.status.as_deref(), None | Some("active"))
        || created_at.is_none()
        || updated_at.is_none()
        || updated_at < created_at
        || draft.mention_refs.iter().any(|reference| {
            reference.display_name.trim().is_empty()
                || reference.display_name.len() > 256
                || !is_public_key(&reference.pubkey)
        })
    {
        return Err(invalid_record(storage_key, "draft entry is malformed"));
    }
    Ok(())
}

fn import_read_state(
    local_storage: &BTreeMap<String, String>,
) -> Result<Vec<BuzzReadStateImport>, BuzzDesktopStateImportError> {
    let mut identities = BTreeSet::new();
    for key in local_storage.keys() {
        for prefix in [
            "buzz.channel-read-state.v2:",
            "buzz.channel-read-state.publishable.v1:",
            "buzz.channel-read-state.source-created-at.v1:",
            "buzz-forced-unread.v1:",
        ] {
            if let Some(identity) = key.strip_prefix(prefix) {
                validate_public_key(identity, key)?;
                identities.insert(identity.to_owned());
            }
        }
    }

    let mut result = Vec::with_capacity(identities.len());
    for identity in identities {
        let contexts = parse_contexts(local_storage, &identity)?;
        let mut publishable_context_ids = parse_string_set(
            local_storage,
            &format!("buzz.channel-read-state.publishable.v1:{identity}"),
        )?;
        publishable_context_ids.retain(|context| contexts.contains_key(context));
        let mut context_source_created_at = parse_u64_map(
            local_storage,
            &format!("buzz.channel-read-state.source-created-at.v1:{identity}"),
        )?;
        context_source_created_at.retain(|context, _| contexts.contains_key(context));
        let forced_unread = parse_forced_unread(local_storage, &identity)?;
        result.push(BuzzReadStateImport {
            identity_public_key: identity,
            contexts,
            publishable_context_ids,
            context_source_created_at,
            forced_unread,
        });
    }
    Ok(result)
}

fn parse_contexts(
    local_storage: &BTreeMap<String, String>,
    identity: &str,
) -> Result<BTreeMap<String, u64>, BuzzDesktopStateImportError> {
    let key = format!("buzz.channel-read-state.v2:{identity}");
    let Some(raw) = local_storage.get(&key) else {
        return Ok(BTreeMap::new());
    };
    let stored: BTreeMap<String, String> =
        serde_json::from_str(raw).map_err(|_| invalid_record(&key, "read state is malformed"))?;
    if stored.len() > MAX_READ_CONTEXTS {
        return Err(BuzzDesktopStateImportError::ResourceLimit);
    }
    stored
        .into_iter()
        .map(|(context, timestamp)| {
            validate_context_key(&context, &key)?;
            let timestamp = parse_timestamp(&timestamp)
                .ok_or_else(|| invalid_record(&key, "read timestamp is malformed"))?;
            Ok((context, timestamp))
        })
        .collect()
}

fn parse_string_set(
    local_storage: &BTreeMap<String, String>,
    key: &str,
) -> Result<BTreeSet<String>, BuzzDesktopStateImportError> {
    let Some(raw) = local_storage.get(key) else {
        return Ok(BTreeSet::new());
    };
    let values: Vec<String> =
        serde_json::from_str(raw).map_err(|_| invalid_record(key, "string set is malformed"))?;
    if values.len() > MAX_READ_CONTEXTS {
        return Err(BuzzDesktopStateImportError::ResourceLimit);
    }
    values
        .into_iter()
        .map(|value| {
            validate_context_key(&value, key)?;
            Ok(value)
        })
        .collect()
}

fn parse_u64_map(
    local_storage: &BTreeMap<String, String>,
    key: &str,
) -> Result<BTreeMap<String, u64>, BuzzDesktopStateImportError> {
    let Some(raw) = local_storage.get(key) else {
        return Ok(BTreeMap::new());
    };
    let values: BTreeMap<String, u64> =
        serde_json::from_str(raw).map_err(|_| invalid_record(key, "integer map is malformed"))?;
    if values.len() > MAX_READ_CONTEXTS {
        return Err(BuzzDesktopStateImportError::ResourceLimit);
    }
    for context in values.keys() {
        validate_context_key(context, key)?;
    }
    Ok(values)
}

fn parse_forced_unread(
    local_storage: &BTreeMap<String, String>,
    identity: &str,
) -> Result<BTreeMap<String, BuzzForcedUnreadImport>, BuzzDesktopStateImportError> {
    let key = format!("buzz-forced-unread.v1:{identity}");
    let Some(raw) = local_storage.get(&key) else {
        return Ok(BTreeMap::new());
    };
    let values: BTreeMap<String, Value> = serde_json::from_str(raw)
        .map_err(|_| invalid_record(&key, "forced-unread map is malformed"))?;
    if values.len() > MAX_FORCED_UNREAD_ENTRIES {
        return Err(BuzzDesktopStateImportError::ResourceLimit);
    }
    values
        .into_iter()
        .map(|(channel_id, value)| {
            validate_context_key(&channel_id, &key)?;
            let entry = match value {
                Value::Null => BuzzForcedUnreadImport {
                    marker_at_when_forced: None,
                    sources: BTreeSet::from([BuzzForcedUnreadSource::Manual]),
                },
                Value::Number(number) => {
                    BuzzForcedUnreadImport {
                        marker_at_when_forced: Some(number.as_u64().ok_or_else(|| {
                            invalid_record(&key, "forced-unread marker is invalid")
                        })?),
                        sources: BTreeSet::from([BuzzForcedUnreadSource::Manual]),
                    }
                }
                Value::Object(mut object) => {
                    let marker = object
                        .remove("markerAtWhenForced")
                        .ok_or_else(|| invalid_record(&key, "forced-unread marker is absent"))?;
                    let marker_at_when_forced = match marker {
                        Value::Null => None,
                        Value::Number(number) => Some(number.as_u64().ok_or_else(|| {
                            invalid_record(&key, "forced-unread marker is invalid")
                        })?),
                        _ => {
                            return Err(invalid_record(&key, "forced-unread marker is invalid"));
                        }
                    };
                    let sources = object
                        .remove("sources")
                        .and_then(|sources| sources.as_array().cloned())
                        .ok_or_else(|| invalid_record(&key, "forced-unread sources are absent"))?
                        .into_iter()
                        .map(|source| match source.as_str() {
                            Some("inbox") => Ok(BuzzForcedUnreadSource::Inbox),
                            Some("manual") => Ok(BuzzForcedUnreadSource::Manual),
                            _ => Err(invalid_record(&key, "forced-unread source is invalid")),
                        })
                        .collect::<Result<BTreeSet<_>, _>>()?;
                    if sources.is_empty() {
                        return Err(invalid_record(&key, "forced-unread sources are empty"));
                    }
                    BuzzForcedUnreadImport {
                        marker_at_when_forced,
                        sources,
                    }
                }
                _ => return Err(invalid_record(&key, "forced-unread entry is invalid")),
            };
            Ok((channel_id, entry))
        })
        .collect()
}

fn import_archive(
    archive: &BuzzArchiveSnapshot,
) -> Result<ImportedArchive, BuzzDesktopStateImportError> {
    let mut events = archive.events.clone();
    events.sort();
    for event in &events {
        validate_archive_event(event)?;
    }
    if events
        .windows(2)
        .any(|pair| archive_event_key(&pair[0]) == archive_event_key(&pair[1]))
    {
        return Err(BuzzDesktopStateImportError::IntegrityConflict);
    }
    let event_keys = events
        .iter()
        .map(archive_event_key)
        .collect::<BTreeSet<_>>();

    let mut scopes = archive.scopes.clone();
    scopes.sort();
    for scope in &scopes {
        validate_archive_scope(scope)?;
        if !event_keys.contains(&(
            scope.identity_public_key.as_str(),
            scope.relay_url.as_str(),
            scope.event_id.as_str(),
        )) {
            return Err(BuzzDesktopStateImportError::IntegrityConflict);
        }
    }
    if scopes
        .windows(2)
        .any(|pair| archive_scope_key(&pair[0]) == archive_scope_key(&pair[1]))
    {
        return Err(BuzzDesktopStateImportError::IntegrityConflict);
    }

    let mut subscriptions = archive.subscriptions.clone();
    subscriptions.sort();
    for subscription in &subscriptions {
        validate_archive_subscription(subscription)?;
    }
    if subscriptions.windows(2).any(|pair| {
        (
            &pair[0].identity_public_key,
            &pair[0].relay_url,
            &pair[0].scope_type,
            &pair[0].scope_value,
        ) == (
            &pair[1].identity_public_key,
            &pair[1].relay_url,
            &pair[1].scope_type,
            &pair[1].scope_value,
        )
    }) {
        return Err(BuzzDesktopStateImportError::IntegrityConflict);
    }
    Ok(ImportedArchive {
        events,
        scopes,
        subscriptions,
    })
}

fn validate_archive_event(event: &BuzzArchivedEvent) -> Result<(), BuzzDesktopStateImportError> {
    validate_public_key(&event.identity_public_key, "archive-event")?;
    validate_public_key(&event.author_public_key, "archive-event")?;
    if !is_event_id(&event.event_id)
        || !is_relay_url(&event.relay_url)
        || event.raw_json.len() > 1_048_576
    {
        return Err(invalid_record(
            "archive-event",
            "event fields are malformed",
        ));
    }
    let raw: Value = serde_json::from_str(&event.raw_json)
        .map_err(|_| invalid_record("archive-event", "raw event JSON is malformed"))?;
    if raw.get("id").and_then(Value::as_str) != Some(event.event_id.as_str())
        || raw.get("pubkey").and_then(Value::as_str) != Some(event.author_public_key.as_str())
        || raw.get("kind").and_then(Value::as_u64) != Some(u64::from(event.kind))
        || raw.get("created_at").and_then(Value::as_u64) != Some(event.created_at)
    {
        return Err(BuzzDesktopStateImportError::IntegrityConflict);
    }
    Ok(())
}

fn validate_archive_scope(
    scope: &BuzzArchivedEventScope,
) -> Result<(), BuzzDesktopStateImportError> {
    validate_public_key(&scope.identity_public_key, "archive-scope")?;
    if !is_event_id(&scope.event_id)
        || !is_relay_url(&scope.relay_url)
        || !matches!(
            scope.scope_type.as_str(),
            "channel_h" | "owner_p" | "referenced_e"
        )
        || scope.scope_value.is_empty()
        || scope.scope_value.len() > 512
    {
        return Err(invalid_record(
            "archive-scope",
            "scope fields are malformed",
        ));
    }
    Ok(())
}

fn validate_archive_subscription(
    subscription: &BuzzArchiveSubscription,
) -> Result<(), BuzzDesktopStateImportError> {
    validate_public_key(&subscription.identity_public_key, "archive-subscription")?;
    if !is_relay_url(&subscription.relay_url)
        || !matches!(
            subscription.scope_type.as_str(),
            "channel_h" | "owner_p" | "referenced_e"
        )
        || subscription.scope_value.is_empty()
        || subscription.scope_value.len() > 512
        || subscription.kinds.is_empty()
        || subscription.kinds.len() > 1_000
    {
        return Err(invalid_record(
            "archive-subscription",
            "subscription fields are malformed",
        ));
    }
    Ok(())
}

fn archive_event_key(event: &BuzzArchivedEvent) -> (&str, &str, &str) {
    (
        &event.identity_public_key,
        &event.relay_url,
        &event.event_id,
    )
}

fn archive_scope_key(scope: &BuzzArchivedEventScope) -> (&str, &str, &str, &str, &str) {
    (
        &scope.identity_public_key,
        &scope.relay_url,
        &scope.event_id,
        &scope.scope_type,
        &scope.scope_value,
    )
}

fn consumed_local_storage_keys(local_storage: &BTreeMap<String, String>) -> BTreeSet<&str> {
    local_storage
        .keys()
        .filter(|key| {
            is_importable_setting_key(key)
                || parse_draft_storage_key(key).is_some()
                || [
                    "buzz.channel-read-state.v2:",
                    "buzz.channel-read-state.publishable.v1:",
                    "buzz.channel-read-state.source-created-at.v1:",
                    "buzz-forced-unread.v1:",
                ]
                .iter()
                .any(|prefix| key.starts_with(prefix))
        })
        .map(String::as_str)
        .collect()
}

fn is_importable_setting_key(key: &str) -> bool {
    [
        "buzz-theme",
        "buzz-accent-color",
        "buzz-glass-background",
        "buzz-glass-opacity",
        "buzz-prominent-active-tab",
        "buzz:text-scale",
        "buzz-sidebar-width",
        "buzz.channels.threadViewMode",
        "buzz-prevent-sleep",
        "buzz-notification-settings.v2:",
        "buzz-presence-preference:",
        "buzz:observer-archive-default-seeded:",
        "buzz:agent-metric-archive-default-seeded:",
        "buzz-link-preview-style",
    ]
    .iter()
    .any(|candidate| {
        if candidate.ends_with(':') {
            key.starts_with(candidate)
        } else {
            key == *candidate
        }
    })
}

fn setting_version(key: &str) -> u32 {
    key.rsplit_once(".v")
        .and_then(|(_, suffix)| suffix.split(':').next())
        .and_then(|version| version.parse().ok())
        .unwrap_or(1)
}

fn parse_setting_value(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned()))
}

fn is_secret_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    [
        "nsec",
        "private_key",
        "private-key",
        "secret",
        "access_token",
        "auth_token",
        "api_key",
        "api-key",
        "password",
    ]
    .iter()
    .any(|needle| key.contains(needle))
}

fn contains_secret_value(value: &Value) -> bool {
    match value {
        Value::String(value) => value.to_ascii_lowercase().contains("nsec1"),
        Value::Array(values) => values.iter().any(contains_secret_value),
        Value::Object(values) => values
            .iter()
            .any(|(key, value)| is_secret_key(key) || contains_secret_value(value)),
        Value::Null | Value::Bool(_) | Value::Number(_) => false,
    }
}

fn is_source_application_id(value: &str) -> bool {
    [
        "xyz.block.sprout.app",
        "xyz.block.sprout.app.dev",
        "xyz.block.buzz.app",
        "xyz.block.buzz.app.dev",
    ]
    .iter()
    .any(|identifier| {
        value == *identifier
            || value
                .strip_prefix(identifier)
                .is_some_and(|suffix| suffix.starts_with('.'))
    })
}

fn validate_public_key(value: &str, key: &str) -> Result<(), BuzzDesktopStateImportError> {
    if !is_public_key(value) {
        return Err(invalid_record(key, "public key is malformed"));
    }
    Ok(())
}

fn is_public_key(value: &str) -> bool {
    is_lower_hex(value, 64)
}

fn is_event_id(value: &str) -> bool {
    is_lower_hex(value, 64)
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_relay_url(value: &str) -> bool {
    value.len() <= 2_048 && (value.starts_with("ws://") || value.starts_with("wss://"))
}

fn validate_context_key(value: &str, source_key: &str) -> Result<(), BuzzDesktopStateImportError> {
    if value.is_empty() || value.len() > 256 || !value.is_ascii() {
        return Err(invalid_record(source_key, "context key is malformed"));
    }
    Ok(())
}

fn parse_timestamp(value: &str) -> Option<u64> {
    DateTime::parse_from_rfc3339(value)
        .ok()?
        .timestamp()
        .try_into()
        .ok()
}

fn invalid_record(key: &str, reason: &'static str) -> BuzzDesktopStateImportError {
    BuzzDesktopStateImportError::InvalidRecord {
        key: key.to_owned(),
        reason,
    }
}

fn hash_target(result: &BuzzDesktopStateImport) -> Result<[u8; 32], BuzzDesktopStateImportError> {
    #[derive(Serialize)]
    struct Target<'a> {
        snapshot_version: u32,
        settings: &'a [BuzzDesktopSettingImport],
        drafts: &'a [BuzzDraftImport],
        read_state: &'a [BuzzReadStateImport],
        archived_events: &'a [BuzzArchivedEvent],
        archive_scopes: &'a [BuzzArchivedEventScope],
        archive_subscriptions: &'a [BuzzArchiveSubscription],
    }
    let bytes = serde_json::to_vec(&Target {
        snapshot_version: result.snapshot_version,
        settings: &result.settings,
        drafts: &result.drafts,
        read_state: &result.read_state,
        archived_events: &result.archived_events,
        archive_scopes: &result.archive_scopes,
        archive_subscriptions: &result.archive_subscriptions,
    })
    .map_err(|_| invalid_record("target", "target state cannot be hashed"))?;
    Ok(sha256(&bytes))
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(version: u32) -> Vec<u8> {
        let identity = "11".repeat(32);
        let author = "22".repeat(32);
        let event_id = "33".repeat(32);
        let mut local_storage =
            BTreeMap::from([("buzz-theme".to_owned(), "\"One Dark\"".to_owned())]);
        let draft = if version == 1 {
            serde_json::json!({
                "content": "review this",
                "selectionStart": 0,
                "selectionEnd": 11,
                "channelId": "channel-a",
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:01:00Z",
                "pendingImeta": [],
                "spoileredAttachmentUrls": []
            })
        } else {
            serde_json::json!({
                "content": "review this",
                "selectionStart": 0,
                "selectionEnd": 11,
                "channelId": "channel-a",
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:01:00Z",
                "pendingImeta": [],
                "mentionRefs": [{
                    "displayName": "Agent",
                    "pubkey": author,
                    "isAgent": true
                }],
                "spoileredAttachmentUrls": [],
                "status": "active"
            })
        };
        let draft_key = if version == 1 {
            format!("buzz-drafts.v1:{identity}")
        } else {
            format!("buzz-drafts.v2:wss://relay.example:{identity}")
        };
        local_storage.insert(
            draft_key,
            serde_json::json!({"channel-a": draft}).to_string(),
        );
        local_storage.insert(
            format!("buzz.channel-read-state.v2:{identity}"),
            serde_json::json!({"channel-a": "2024-01-01T00:00:00Z"}).to_string(),
        );
        local_storage.insert(
            format!("buzz.channel-read-state.publishable.v1:{identity}"),
            serde_json::json!(["channel-a", "missing-context"]).to_string(),
        );
        local_storage.insert(
            format!("buzz.channel-read-state.source-created-at.v1:{identity}"),
            serde_json::json!({"channel-a": 1_704_067_100_u64}).to_string(),
        );
        local_storage.insert(
            format!("buzz-forced-unread.v1:{identity}"),
            serde_json::json!({
                "channel-a": {
                    "markerAtWhenForced": 1_704_067_200_u64,
                    "sources": ["manual", "inbox"]
                }
            })
            .to_string(),
        );
        local_storage.insert(
            "buzz-channel-messages.v1:wss://relay.example:channel-a".to_owned(),
            "cached timeline".to_owned(),
        );

        let raw_event = serde_json::json!({
            "id": event_id,
            "pubkey": author,
            "created_at": 1_704_067_200_u64,
            "kind": 1,
            "tags": [],
            "content": "hello",
            "sig": "44".repeat(64)
        })
        .to_string();
        let (schema_version, migration_markers, events, scopes, subscriptions) = if version == 1 {
            (1, BTreeSet::new(), Vec::new(), Vec::new(), Vec::new())
        } else {
            (
                4,
                BTreeSet::from([
                    ARCHIVE_MIGRATION_CACHE_READ.to_owned(),
                    ARCHIVE_MIGRATION_CACHE_WRITE.to_owned(),
                    ARCHIVE_MIGRATION_HARNESS.to_owned(),
                ]),
                vec![BuzzArchivedEvent {
                    identity_public_key: identity.clone(),
                    relay_url: "wss://relay.example".to_owned(),
                    event_id: event_id.clone(),
                    kind: 1,
                    author_public_key: author,
                    created_at: 1_704_067_200,
                    raw_json: raw_event,
                    archived_at: 1_704_067_300,
                }],
                vec![BuzzArchivedEventScope {
                    identity_public_key: identity.clone(),
                    relay_url: "wss://relay.example".to_owned(),
                    event_id,
                    scope_type: "channel_h".to_owned(),
                    scope_value: "channel-a".to_owned(),
                    archived_at: 1_704_067_300,
                }],
                vec![BuzzArchiveSubscription {
                    identity_public_key: identity,
                    relay_url: "wss://relay.example".to_owned(),
                    scope_type: "channel_h".to_owned(),
                    scope_value: "channel-a".to_owned(),
                    kinds: vec![1, 6],
                    created_at: 1_704_067_100,
                }],
            )
        };
        serde_json::to_vec(&BuzzDesktopSourceSnapshot {
            snapshot_version: version,
            captured_at_millis: 1_704_067_400_000,
            source_application_id: if version == 1 {
                "xyz.block.sprout.app".to_owned()
            } else {
                "xyz.block.buzz.app.dev.feature".to_owned()
            },
            general_configuration: BTreeMap::from([(
                "prevent_sleep".to_owned(),
                Value::Bool(true),
            )]),
            local_storage,
            archive: BuzzArchiveSnapshot {
                schema_version,
                migration_markers,
                events,
                scopes,
                subscriptions,
            },
        })
        .expect("fixture JSON")
    }

    #[test]
    fn every_snapshot_fixture_version_imports_twice_identically() {
        for version in [1, 2] {
            let bytes = fixture(version);
            let before = bytes.clone();
            let first = import_desktop_state_bytes(&bytes).expect("first import");
            let second = import_desktop_state_bytes(&bytes).expect("second import");

            assert_eq!(first, second);
            assert_eq!(bytes, before);
            assert_eq!(first.snapshot_version, version);
            assert_eq!(first.drafts.len(), 1);
            assert_eq!(first.read_state.len(), 1);
            assert_eq!(first.read_state[0].contexts["channel-a"], 1_704_067_200);
            assert_eq!(
                first.read_state[0].publishable_context_ids,
                BTreeSet::from(["channel-a".to_owned()])
            );
            assert_eq!(first.skipped_cache_entries, 1);
            if version == 1 {
                assert!(first.drafts[0].mention_references.is_empty());
                assert!(first.archived_events.is_empty());
            } else {
                assert_eq!(first.drafts[0].mention_references.len(), 1);
                assert_eq!(first.archived_events.len(), 1);
                assert_eq!(first.archive_scopes.len(), 1);
                assert_eq!(first.archive_subscriptions.len(), 1);
            }
        }
    }

    #[test]
    fn file_import_leaves_source_bytes_unchanged() {
        let path = std::env::temp_dir().join(format!(
            "zed-buzz-desktop-state-{}.json",
            uuid::Uuid::new_v4()
        ));
        let bytes = fixture(2);
        fs::write(&path, &bytes).expect("write fixture");

        let imported = import_desktop_state_file(&path).expect("file import");

        assert_eq!(fs::read(&path).expect("read fixture"), bytes);
        assert_eq!(imported.source_hash, sha256(&bytes));
        fs::remove_file(path).expect("remove fixture");
    }

    #[test]
    fn secret_general_configuration_fails_closed() {
        let mut snapshot: BuzzDesktopSourceSnapshot =
            serde_json::from_slice(&fixture(1)).expect("fixture snapshot");
        snapshot.general_configuration.insert(
            "provider".to_owned(),
            serde_json::json!({"private_key": "nsec1must-not-migrate"}),
        );
        let bytes = serde_json::to_vec(&snapshot).expect("snapshot JSON");

        let error = import_desktop_state_bytes(&bytes).expect_err("secret must fail closed");
        assert!(matches!(
            error,
            BuzzDesktopStateImportError::SecretMaterial(_)
        ));
    }
}
