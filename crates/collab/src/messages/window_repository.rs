use std::{cmp::Reverse, collections::BTreeMap};

use collaboration_domain::{
    AggregateId, AuthorizationAction, AuthorizationDecision, AuthorizationDenial,
    AuthorizationRequest, AuthorizationResourceKind, CommunityId, MAX_THREAD_DEPTH,
    MAX_THREAD_PAGE_ROWS, NostrEventId, PrincipalId, TenantContext, ThreadCursor, authorize,
};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbBackend, DbErr,
    QueryResult, Statement, TransactionTrait, Value,
};

const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";
const CAPTURE_SNAPSHOT_SQL: &str = r#"
SELECT floor(extract(epoch FROM clock_timestamp()) * 1000000)::bigint
    AS snapshot_micros
"#;
const MESSAGE_STATE_CTE: &str = r#"
message_state AS (
    SELECT
        message.community_id,
        message.message_id,
        message.channel_id,
        message.source_event_id,
        COALESCE(latest_edit.auxiliary_event_id, message.source_event_id) AS current_event_id,
        message.author_principal_id,
        message.message_created_at,
        CASE
            WHEN deletion.auxiliary_event_id IS NOT NULL THEN 'deleted'
            WHEN latest_edit.auxiliary_event_id IS NOT NULL THEN 'edited'
            ELSE 'active'
        END AS lifecycle_state,
        (1 + COALESCE(mutations.mutation_count, 0))::text AS message_version_text,
        floor(extract(epoch FROM message.projected_at) * 1000000)::bigint
            AS projected_at_micros,
        source_event.tags
    FROM public.collaboration_messages AS message
    JOIN public.collaboration_events AS source_event
      ON source_event.community_id = message.community_id
     AND source_event.event_id = message.source_event_id
    LEFT JOIN LATERAL (
        SELECT auxiliary.auxiliary_event_id
        FROM public.collaboration_message_auxiliary_events AS auxiliary
        WHERE auxiliary.community_id = message.community_id
          AND auxiliary.channel_id = message.channel_id
          AND auxiliary.target_message_event_id = message.source_event_id
          AND auxiliary.auxiliary_kind = 'edit'
          AND auxiliary.projected_at <= to_timestamp(CAST($3 AS double precision) / 1000000)
        ORDER BY auxiliary.event_created_at DESC, auxiliary.auxiliary_event_id ASC
        LIMIT 1
    ) AS latest_edit ON true
    LEFT JOIN LATERAL (
        SELECT auxiliary.auxiliary_event_id
        FROM public.collaboration_message_auxiliary_events AS auxiliary
        WHERE auxiliary.community_id = message.community_id
          AND auxiliary.channel_id = message.channel_id
          AND auxiliary.target_message_event_id = message.source_event_id
          AND auxiliary.auxiliary_kind = 'delete'
          AND auxiliary.projected_at <= to_timestamp(CAST($3 AS double precision) / 1000000)
        ORDER BY auxiliary.event_created_at ASC, auxiliary.auxiliary_event_id ASC
        LIMIT 1
    ) AS deletion ON true
    LEFT JOIN LATERAL (
        SELECT count(*)::bigint AS mutation_count
        FROM public.collaboration_message_auxiliary_events AS auxiliary
        WHERE auxiliary.community_id = message.community_id
          AND auxiliary.channel_id = message.channel_id
          AND auxiliary.target_message_event_id = message.source_event_id
          AND auxiliary.auxiliary_kind IN ('edit', 'delete')
          AND auxiliary.projected_at <= to_timestamp(CAST($3 AS double precision) / 1000000)
    ) AS mutations ON true
    WHERE message.community_id = $1
      AND message.channel_id = $2
      AND message.projected_at <= to_timestamp(CAST($3 AS double precision) / 1000000)
), classified AS (
    SELECT
        message_state.*,
        reply.parent_event_id,
        EXISTS (
            SELECT 1
            FROM jsonb_array_elements(message_state.tags) AS tag(value)
            WHERE tag.value = '["broadcast", "1"]'::jsonb
        ) AS broadcast
    FROM message_state
    LEFT JOIN LATERAL (
        SELECT decode(tag.value ->> 1, 'hex') AS parent_event_id
        FROM jsonb_array_elements(message_state.tags) WITH ORDINALITY AS tag(value, ordinal)
        WHERE jsonb_typeof(tag.value) = 'array'
          AND jsonb_array_length(tag.value) >= 4
          AND tag.value ->> 0 = 'e'
          AND tag.value ->> 3 = 'reply'
          AND tag.value ->> 1 ~ '^[0-9a-fA-F]{64}$'
        ORDER BY tag.ordinal
        LIMIT 1
    ) AS reply ON true
)
"#;
const MESSAGE_COLUMNS: &str = r#"
    row.community_id,
    row.message_id,
    row.channel_id,
    row.source_event_id,
    row.current_event_id,
    row.author_principal_id,
    row.message_created_at::text AS message_created_at_text,
    row.lifecycle_state,
    row.message_version_text,
    row.projected_at_micros
"#;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ChannelWindowCursor {
    pub message_created_at: u64,
    pub source_event_id: NostrEventId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WindowSnapshot(u64);

impl WindowSnapshot {
    pub const fn from_micros(value: u64) -> Self {
        Self(value)
    }

    pub const fn as_micros(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MessageProjectionLifecycle {
    Active,
    Edited,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageWindowRow {
    pub community_id: CommunityId,
    pub message_id: AggregateId,
    pub channel_id: AggregateId,
    pub source_event_id: NostrEventId,
    pub current_event_id: NostrEventId,
    pub author_principal_id: PrincipalId,
    pub message_created_at: u64,
    pub lifecycle: MessageProjectionLifecycle,
    pub message_version: u64,
    pub projected_at_micros: u64,
}

impl MessageWindowRow {
    pub const fn cursor(&self) -> ChannelWindowCursor {
        ChannelWindowCursor {
            message_created_at: self.message_created_at,
            source_event_id: self.source_event_id,
        }
    }

    pub const fn thread_cursor(&self) -> ThreadCursor {
        ThreadCursor {
            created_at: self.message_created_at,
            event_id: self.source_event_id,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThreadWindowRow {
    pub message: MessageWindowRow,
    pub parent_event_id: NostrEventId,
    pub root_event_id: NostrEventId,
    pub depth: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChannelWindowQuery {
    channel_id: AggregateId,
    limit: usize,
    cursor: Option<ChannelWindowCursor>,
    snapshot: Option<WindowSnapshot>,
}

impl ChannelWindowQuery {
    pub fn head(channel_id: AggregateId, requested_limit: usize) -> Result<Self, WindowError> {
        validate_channel(channel_id)?;
        Ok(Self {
            channel_id,
            limit: bounded_limit(requested_limit),
            cursor: None,
            snapshot: None,
        })
    }

    pub fn continuation(
        channel_id: AggregateId,
        requested_limit: usize,
        cursor: ChannelWindowCursor,
        snapshot: WindowSnapshot,
    ) -> Result<Self, WindowError> {
        validate_channel(channel_id)?;
        validate_event_id(cursor.source_event_id)?;
        Ok(Self {
            channel_id,
            limit: bounded_limit(requested_limit),
            cursor: Some(cursor),
            snapshot: Some(snapshot),
        })
    }

    pub const fn channel_id(self) -> AggregateId {
        self.channel_id
    }

    pub const fn limit(self) -> usize {
        self.limit
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThreadWindowQuery {
    channel_id: AggregateId,
    root_event_id: NostrEventId,
    limit: usize,
    depth_limit: u16,
    cursor: Option<ThreadCursor>,
    snapshot: Option<WindowSnapshot>,
}

impl ThreadWindowQuery {
    pub fn head(
        channel_id: AggregateId,
        root_event_id: NostrEventId,
        requested_limit: usize,
        depth_limit: Option<u16>,
    ) -> Result<Self, WindowError> {
        validate_channel(channel_id)?;
        validate_event_id(root_event_id)?;
        Ok(Self {
            channel_id,
            root_event_id,
            limit: bounded_limit(requested_limit),
            depth_limit: depth_limit
                .unwrap_or(MAX_THREAD_DEPTH)
                .min(MAX_THREAD_DEPTH),
            cursor: None,
            snapshot: None,
        })
    }

    pub fn continuation(
        channel_id: AggregateId,
        root_event_id: NostrEventId,
        requested_limit: usize,
        depth_limit: Option<u16>,
        cursor: ThreadCursor,
        snapshot: WindowSnapshot,
    ) -> Result<Self, WindowError> {
        validate_channel(channel_id)?;
        validate_event_id(root_event_id)?;
        validate_event_id(cursor.event_id)?;
        Ok(Self {
            channel_id,
            root_event_id,
            limit: bounded_limit(requested_limit),
            depth_limit: depth_limit
                .unwrap_or(MAX_THREAD_DEPTH)
                .min(MAX_THREAD_DEPTH),
            cursor: Some(cursor),
            snapshot: Some(snapshot),
        })
    }
}

pub struct WindowAccess<'a> {
    pub authorization: &'a AuthorizationRequest<'a>,
}

impl WindowAccess<'_> {
    const fn tenant(&self) -> &TenantContext {
        self.authorization.tenant
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChannelWindowPage {
    pub rows: Vec<MessageWindowRow>,
    pub has_more: bool,
    pub next_cursor: Option<ChannelWindowCursor>,
    pub request_cursor: Option<ChannelWindowCursor>,
    pub snapshot: WindowSnapshot,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThreadWindowPage {
    pub replies: Vec<ThreadWindowRow>,
    pub has_more: bool,
    pub next_cursor: Option<ThreadCursor>,
    pub request_cursor: Option<ThreadCursor>,
    pub snapshot: WindowSnapshot,
}

#[derive(Debug, thiserror::Error)]
pub enum WindowError {
    #[error("message windows require PostgreSQL")]
    UnsupportedDatabase,
    #[error("message window query is invalid")]
    InvalidQuery,
    #[error("message window authorization shape is invalid")]
    InvalidAuthorization,
    #[error("message window is not authorized: {0:?}")]
    Unauthorized(AuthorizationDenial),
    #[error("message window returned an invalid record")]
    InvalidRecord,
    #[error("message window page conflicts with an immutable row")]
    ConflictingRow,
    #[error("message window is unavailable")]
    Unavailable(#[source] DbErr),
}

pub struct MessageWindowRepository {
    connection: DatabaseConnection,
}

impl MessageWindowRepository {
    pub fn new(connection: DatabaseConnection) -> Result<Self, WindowError> {
        if connection.get_database_backend() != DbBackend::Postgres {
            return Err(WindowError::UnsupportedDatabase);
        }
        Ok(Self { connection })
    }

    pub async fn channel_page(
        &self,
        access: WindowAccess<'_>,
        query: &ChannelWindowQuery,
    ) -> Result<ChannelWindowPage, WindowError> {
        authorize_window(&access, query.channel_id)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, access.tenant()).await?;
            let snapshot = resolve_snapshot(&transaction, query.snapshot).await?;
            let rows = transaction
                .query_all(channel_statement(access.tenant(), query, snapshot)?)
                .await
                .map_err(WindowError::Unavailable)?;
            let mut rows = rows
                .into_iter()
                .map(|row| message_from_row(row, access.tenant().community_id(), query.channel_id))
                .collect::<Result<Vec<_>, _>>()?;
            let has_more = rows.len() > query.limit;
            rows.truncate(query.limit);
            let next_cursor = has_more
                .then(|| rows.last().map(MessageWindowRow::cursor))
                .flatten();
            Ok(ChannelWindowPage {
                rows,
                has_more,
                next_cursor,
                request_cursor: query.cursor,
                snapshot,
            })
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub async fn thread_page(
        &self,
        access: WindowAccess<'_>,
        query: &ThreadWindowQuery,
    ) -> Result<ThreadWindowPage, WindowError> {
        authorize_window(&access, query.channel_id)?;
        let transaction = self.begin().await?;
        let result = async {
            set_tenant(&transaction, access.tenant()).await?;
            let snapshot = resolve_snapshot(&transaction, query.snapshot).await?;
            let rows = transaction
                .query_all(thread_statement(access.tenant(), query, snapshot)?)
                .await
                .map_err(WindowError::Unavailable)?;
            let mut replies = rows
                .into_iter()
                .map(|row| {
                    thread_from_row(
                        row,
                        access.tenant().community_id(),
                        query.channel_id,
                        query.root_event_id,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let has_more = replies.len() > query.limit;
            replies.truncate(query.limit);
            let next_cursor = has_more
                .then(|| replies.last().map(|row| row.message.thread_cursor()))
                .flatten();
            Ok(ThreadWindowPage {
                replies,
                has_more,
                next_cursor,
                request_cursor: query.cursor,
                snapshot,
            })
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub fn into_connection(self) -> DatabaseConnection {
        self.connection
    }

    async fn begin(&self) -> Result<DatabaseTransaction, WindowError> {
        self.connection
            .begin()
            .await
            .map_err(WindowError::Unavailable)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StableChannelWindow {
    channel_id: AggregateId,
    snapshot: Option<WindowSnapshot>,
    next_cursor: Option<ChannelWindowCursor>,
    history: BTreeMap<NostrEventId, MessageWindowRow>,
    live: BTreeMap<NostrEventId, MessageWindowRow>,
}

impl StableChannelWindow {
    pub fn new(channel_id: AggregateId) -> Result<Self, WindowError> {
        validate_channel(channel_id)?;
        Ok(Self {
            channel_id,
            snapshot: None,
            next_cursor: None,
            history: BTreeMap::new(),
            live: BTreeMap::new(),
        })
    }

    pub fn replace_head(&mut self, page: ChannelWindowPage) -> Result<(), WindowError> {
        if page.request_cursor.is_some() {
            return Err(WindowError::InvalidQuery);
        }
        self.require_rows(&page.rows)?;
        validate_page_cursor(&page)?;
        self.snapshot = Some(page.snapshot);
        self.next_cursor = page.next_cursor;
        self.history.clear();
        for row in page.rows {
            self.live.remove(&row.source_event_id);
            self.history.insert(row.source_event_id, row);
        }
        Ok(())
    }

    pub fn append_page(&mut self, page: ChannelWindowPage) -> Result<(), WindowError> {
        if page.request_cursor.is_none()
            || page.request_cursor != self.next_cursor
            || self.snapshot != Some(page.snapshot)
        {
            return Err(WindowError::InvalidQuery);
        }
        self.require_rows(&page.rows)?;
        validate_page_cursor(&page)?;
        for row in page.rows {
            if self
                .history
                .get(&row.source_event_id)
                .is_some_and(|existing| existing != &row)
            {
                return Err(WindowError::ConflictingRow);
            }
            self.live.remove(&row.source_event_id);
            self.history.insert(row.source_event_id, row);
        }
        self.next_cursor = page.next_cursor;
        Ok(())
    }

    pub fn push_live(&mut self, row: MessageWindowRow) -> Result<bool, WindowError> {
        self.require_rows(std::slice::from_ref(&row))?;
        if let Some(existing) = self.history.get(&row.source_event_id) {
            return if existing == &row {
                Ok(false)
            } else {
                Err(WindowError::ConflictingRow)
            };
        }
        match self.live.get(&row.source_event_id) {
            Some(existing) if existing == &row => Ok(false),
            Some(_) => Err(WindowError::ConflictingRow),
            None => {
                self.live.insert(row.source_event_id, row);
                Ok(true)
            }
        }
    }

    pub fn ordered_rows(&self) -> Vec<&MessageWindowRow> {
        let mut rows = self
            .live
            .values()
            .chain(self.history.values())
            .collect::<Vec<_>>();
        rows.sort_by_key(|row| (Reverse(row.message_created_at), row.source_event_id));
        rows
    }

    fn require_rows(&self, rows: &[MessageWindowRow]) -> Result<(), WindowError> {
        if rows.iter().any(|row| row.channel_id != self.channel_id) {
            return Err(WindowError::InvalidRecord);
        }
        Ok(())
    }
}

fn authorize_window(access: &WindowAccess<'_>, channel_id: AggregateId) -> Result<(), WindowError> {
    let request = access.authorization;
    if request.action != AuthorizationAction::Read
        || request.resource.kind != AuthorizationResourceKind::Channel
        || request.resource.resource_id != channel_id
        || request.resource.channel_id != Some(channel_id)
    {
        return Err(WindowError::InvalidAuthorization);
    }
    match authorize(request) {
        AuthorizationDecision::Allowed => Ok(()),
        AuthorizationDecision::Denied(denial) => Err(WindowError::Unauthorized(denial)),
    }
}

async fn resolve_snapshot(
    transaction: &DatabaseTransaction,
    snapshot: Option<WindowSnapshot>,
) -> Result<WindowSnapshot, WindowError> {
    if let Some(snapshot) = snapshot {
        return Ok(snapshot);
    }
    let row = transaction
        .query_one(Statement::from_string(
            DatabaseBackend::Postgres,
            CAPTURE_SNAPSHOT_SQL,
        ))
        .await
        .map_err(WindowError::Unavailable)?
        .ok_or(WindowError::InvalidRecord)?;
    let micros = row_value::<i64>(&row, "snapshot_micros")?;
    let micros = u64::try_from(micros).map_err(|_| WindowError::InvalidRecord)?;
    Ok(WindowSnapshot::from_micros(micros))
}

fn channel_statement(
    tenant: &TenantContext,
    query: &ChannelWindowQuery,
    snapshot: WindowSnapshot,
) -> Result<Statement, WindowError> {
    let mut sql = format!(
        "WITH {MESSAGE_STATE_CTE} SELECT {MESSAGE_COLUMNS} FROM classified AS row LEFT JOIN classified AS parent ON parent.source_event_id = row.parent_event_id WHERE row.lifecycle_state <> 'deleted' AND (row.parent_event_id IS NULL OR (row.broadcast AND parent.source_event_id IS NOT NULL AND parent.parent_event_id IS NULL))"
    );
    let mut values = base_values(tenant, query.channel_id, snapshot)?;
    if let Some(cursor) = query.cursor {
        let created_at = bind_value(&mut values, cursor.message_created_at.to_string().into());
        let event_id = bind_value(
            &mut values,
            cursor.source_event_id.as_bytes().to_vec().into(),
        );
        sql.push_str(&format!(
            " AND (row.message_created_at < CAST({created_at} AS numeric) OR (row.message_created_at = CAST({created_at} AS numeric) AND row.source_event_id > {event_id}))"
        ));
    }
    sql.push_str(" ORDER BY row.message_created_at DESC, row.source_event_id ASC LIMIT ");
    sql.push_str(&bind_value(&mut values, probe_limit(query.limit)?.into()));
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        sql,
        values,
    ))
}

fn thread_statement(
    tenant: &TenantContext,
    query: &ThreadWindowQuery,
    snapshot: WindowSnapshot,
) -> Result<Statement, WindowError> {
    let mut values = base_values(tenant, query.channel_id, snapshot)?;
    let root = bind_value(&mut values, query.root_event_id.as_bytes().to_vec().into());
    let depth_limit = bind_value(&mut values, i32::from(query.depth_limit).into());
    let mut sql = format!(
        "WITH RECURSIVE {MESSAGE_STATE_CTE}, thread AS (SELECT row.*, 0::integer AS depth, ARRAY[row.source_event_id]::bytea[] AS path FROM classified AS row WHERE row.source_event_id = {root} UNION ALL SELECT child.*, thread.depth + 1, thread.path || child.source_event_id FROM classified AS child JOIN thread ON child.parent_event_id = thread.source_event_id WHERE thread.depth < {depth_limit} AND NOT child.source_event_id = ANY(thread.path)) SELECT {MESSAGE_COLUMNS}, row.parent_event_id, row.depth FROM thread AS row WHERE row.depth > 0 AND row.lifecycle_state <> 'deleted'"
    );
    if let Some(cursor) = query.cursor {
        let created_at = bind_value(&mut values, cursor.created_at.to_string().into());
        let event_id = bind_value(&mut values, cursor.event_id.as_bytes().to_vec().into());
        sql.push_str(&format!(
            " AND (row.message_created_at > CAST({created_at} AS numeric) OR (row.message_created_at = CAST({created_at} AS numeric) AND row.source_event_id > {event_id}))"
        ));
    }
    sql.push_str(" ORDER BY row.message_created_at ASC, row.source_event_id ASC LIMIT ");
    sql.push_str(&bind_value(&mut values, probe_limit(query.limit)?.into()));
    Ok(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        sql,
        values,
    ))
}

fn base_values(
    tenant: &TenantContext,
    channel_id: AggregateId,
    snapshot: WindowSnapshot,
) -> Result<Vec<Value>, WindowError> {
    let snapshot = i64::try_from(snapshot.as_micros()).map_err(|_| WindowError::InvalidQuery)?;
    Ok(vec![
        tenant.community_id().as_uuid().into(),
        channel_id.as_uuid().into(),
        snapshot.into(),
    ])
}

fn probe_limit(limit: usize) -> Result<i64, WindowError> {
    i64::try_from(limit.saturating_add(1)).map_err(|_| WindowError::InvalidQuery)
}

fn bind_value(values: &mut Vec<Value>, value: Value) -> String {
    values.push(value);
    format!("${}", values.len())
}

fn message_from_row(
    row: QueryResult,
    expected_community_id: CommunityId,
    expected_channel_id: AggregateId,
) -> Result<MessageWindowRow, WindowError> {
    let community_id = CommunityId::from_uuid(row_value(&row, "community_id")?);
    if community_id != expected_community_id {
        return Err(WindowError::InvalidRecord);
    }
    let lifecycle = match row_value::<String>(&row, "lifecycle_state")?.as_str() {
        "active" => MessageProjectionLifecycle::Active,
        "edited" => MessageProjectionLifecycle::Edited,
        _ => return Err(WindowError::InvalidRecord),
    };
    let source_event_id = event_id_from_row(&row, "source_event_id")?;
    let current_event_id = event_id_from_row(&row, "current_event_id")?;
    let message_created_at = row_value::<String>(&row, "message_created_at_text")?
        .parse::<u64>()
        .map_err(|_| WindowError::InvalidRecord)?;
    let message_version = row_value::<String>(&row, "message_version_text")?
        .parse::<u64>()
        .map_err(|_| WindowError::InvalidRecord)?;
    let projected_at_micros = u64::try_from(row_value::<i64>(&row, "projected_at_micros")?)
        .map_err(|_| WindowError::InvalidRecord)?;
    if message_version == 0 {
        return Err(WindowError::InvalidRecord);
    }
    let channel_id = AggregateId::from_uuid(row_value(&row, "channel_id")?);
    if channel_id != expected_channel_id {
        return Err(WindowError::InvalidRecord);
    }
    Ok(MessageWindowRow {
        community_id,
        message_id: AggregateId::from_uuid(row_value(&row, "message_id")?),
        channel_id,
        source_event_id,
        current_event_id,
        author_principal_id: PrincipalId::from_uuid(row_value(&row, "author_principal_id")?),
        message_created_at,
        lifecycle,
        message_version,
        projected_at_micros,
    })
}

fn thread_from_row(
    row: QueryResult,
    expected_community_id: CommunityId,
    expected_channel_id: AggregateId,
    root_event_id: NostrEventId,
) -> Result<ThreadWindowRow, WindowError> {
    let parent_event_id = event_id_from_row(&row, "parent_event_id")?;
    let depth =
        u16::try_from(row_value::<i32>(&row, "depth")?).map_err(|_| WindowError::InvalidRecord)?;
    if depth == 0 || depth > MAX_THREAD_DEPTH {
        return Err(WindowError::InvalidRecord);
    }
    Ok(ThreadWindowRow {
        message: message_from_row(row, expected_community_id, expected_channel_id)?,
        parent_event_id,
        root_event_id,
        depth,
    })
}

fn validate_page_cursor(page: &ChannelWindowPage) -> Result<(), WindowError> {
    match (page.has_more, page.next_cursor, page.rows.last()) {
        (true, Some(cursor), Some(last)) if cursor == last.cursor() => Ok(()),
        (false, None, _) => Ok(()),
        _ => Err(WindowError::InvalidRecord),
    }
}

fn event_id_from_row(row: &QueryResult, column: &str) -> Result<NostrEventId, WindowError> {
    let bytes: Vec<u8> = row_value(row, column)?;
    let bytes = bytes.try_into().map_err(|_| WindowError::InvalidRecord)?;
    let event_id = NostrEventId::from_bytes(bytes);
    validate_event_id(event_id)?;
    Ok(event_id)
}

fn row_value<T>(row: &QueryResult, column: &str) -> Result<T, WindowError>
where
    T: sea_orm::TryGetable,
{
    row.try_get("", column)
        .map_err(|_| WindowError::InvalidRecord)
}

fn validate_channel(channel_id: AggregateId) -> Result<(), WindowError> {
    if channel_id.as_uuid().is_nil() {
        return Err(WindowError::InvalidQuery);
    }
    Ok(())
}

fn validate_event_id(event_id: NostrEventId) -> Result<(), WindowError> {
    if event_id.as_bytes().iter().all(|byte| *byte == 0) {
        return Err(WindowError::InvalidQuery);
    }
    Ok(())
}

fn bounded_limit(requested_limit: usize) -> usize {
    requested_limit.clamp(1, MAX_THREAD_PAGE_ROWS)
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
) -> Result<(), WindowError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            [tenant.community_id().to_string().into()],
        ))
        .await
        .map_err(WindowError::Unavailable)?;
    Ok(())
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, WindowError>,
) -> Result<T, WindowError> {
    match result {
        Ok(value) => {
            transaction
                .commit()
                .await
                .map_err(WindowError::Unavailable)?;
            Ok(value)
        }
        Err(error) => {
            transaction
                .rollback()
                .await
                .map_err(WindowError::Unavailable)?;
            Err(error)
        }
    }
}
