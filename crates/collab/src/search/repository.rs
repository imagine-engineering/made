use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationDecision, AuthorizationDenial, AuthorizationRequest, AuthorizationResource,
    AuthorizationResourceKind, AuthorizationScope, CommunityMembership, Provenance, SourceRecordId,
    SourceSystem, TenantContext, authorize,
};
use nostr_compat::{EventId, PublicKey};
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, DbBackend, DbErr,
    QueryResult, Statement, TransactionTrait,
};

const SET_TENANT_SQL: &str = "SELECT set_config('app.community_id', $1, true) AS app_community_id";
const SEARCH_SCOPE: &str = "collaboration:search";
const SEARCH_TEXT_MAX_CHARS: usize = 4_096;
const RESULTS_MAX: u32 = 500;
const RESULTS_DEFAULT: u32 = 100;
const PAGE_MAX: u32 = 1_000;

const SEARCH_SQL: &str = r#"
WITH search_query AS (
    SELECT CASE
        WHEN $3 = 'full_text' THEN websearch_to_tsquery('simple', $2)
        ELSE (
            SELECT COALESCE(
                string_agg(
                    quote_literal(lexeme)
                    || CASE WHEN token_ord = max_token_ord THEN ':*' ELSE '' END,
                    ' & ' ORDER BY token_ord, lex_ord
                ),
                ''
            )::tsquery
            FROM (
                SELECT raw_token.token_ord,
                       normalized.lexeme,
                       normalized.lex_ord,
                       raw_token.max_token_ord
                FROM (
                    SELECT token,
                           token_ord,
                           max(token_ord) OVER () AS max_token_ord
                    FROM regexp_split_to_table($2, '\s+')
                         WITH ORDINALITY AS split(token, token_ord)
                ) AS raw_token
                CROSS JOIN LATERAL unnest(
                    tsvector_to_array(to_tsvector('simple', raw_token.token))
                ) WITH ORDINALITY AS normalized(lexeme, lex_ord)
            ) AS prefix_terms
        )
    END AS query
), authorized_candidates AS (
    SELECT 'signed_event'::text AS record_type,
           'nostr'::text AS source_system,
           encode(event.event_id, 'hex') AS source_record_id,
           encode(event.event_id, 'hex') AS source_version,
           event.event_id,
           event.author_public_key,
           event.kind AS event_kind,
           NULL::text AS document_type,
           event.event_created_at::text AS observed_at_millis,
           event.event_created_at AS sort_time,
           ts_rank_cd(event.search_tsv, search_query.query) AS rank
    FROM public.collaboration_events AS event
    CROSS JOIN search_query
    WHERE event.community_id = $1
      AND event.search_tsv IS NOT NULL
      AND event.search_tsv @@ search_query.query

    UNION ALL

    SELECT 'canonical_document'::text AS record_type,
           document.source_system,
           document.source_record_id,
           document.source_version,
           NULL::bytea AS event_id,
           NULL::bytea AS author_public_key,
           NULL::integer AS event_kind,
           document.document_type,
           floor(extract(epoch FROM document.source_observed_at) * 1000)::bigint::text
               AS observed_at_millis,
           extract(epoch FROM document.source_observed_at) AS sort_time,
           ts_rank_cd(document.search_tsv, search_query.query) AS rank
    FROM public.collaboration_search_documents AS document
    CROSS JOIN search_query
    WHERE document.community_id = $1
      AND document.visibility_scope = 'community'
      AND document.search_tsv IS NOT NULL
      AND document.search_tsv @@ search_query.query
)
SELECT record_type,
       source_system,
       source_record_id,
       source_version,
       event_id,
       author_public_key,
       event_kind,
       document_type,
       observed_at_millis,
       rank
FROM authorized_candidates
ORDER BY rank DESC, sort_time DESC, source_system, source_record_id
LIMIT $4 OFFSET $5
"#;

const FRESHNESS_SQL: &str = r#"
SELECT count(*)::bigint AS checkpoint_count,
       COALESCE(bool_and(drift_state = 'clean'), false) AS all_clean,
       COALESCE(
           floor(extract(epoch FROM min(projected_at)) * 1000)::bigint,
           0
       ) AS oldest_projected_at_millis,
       count(*) FILTER (WHERE drift_state <> 'clean')::bigint AS affected_count
FROM public.collaboration_projection_checkpoints
WHERE community_id = $1
  AND projection_name = 'collaboration_search'
"#;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchMode {
    FullText,
    Prefix,
}

impl SearchMode {
    const fn database_name(self) -> &'static str {
        match self {
            Self::FullText => "full_text",
            Self::Prefix => "prefix",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationSearchQuery {
    text: String,
    mode: SearchMode,
    page: u32,
    results_per_page: u32,
}

impl CollaborationSearchQuery {
    pub fn new(
        text: impl Into<String>,
        mode: SearchMode,
        page: u32,
        results_per_page: u32,
    ) -> Result<Self, SearchRepositoryError> {
        let text = normalize_search_text(text.into()).ok_or(SearchRepositoryError::InvalidQuery)?;
        let page = page.clamp(1, PAGE_MAX);
        let results_per_page = if results_per_page == 0 {
            RESULTS_DEFAULT
        } else {
            results_per_page.clamp(1, RESULTS_MAX)
        };
        Ok(Self {
            text,
            mode,
            page,
            results_per_page,
        })
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub const fn mode(&self) -> SearchMode {
        self.mode
    }

    pub const fn page(&self) -> u32 {
        self.page
    }

    pub const fn results_per_page(&self) -> u32 {
        self.results_per_page
    }
}

pub struct SearchAccess<'a> {
    pub tenant: &'a TenantContext,
    pub principal: &'a AuthenticatedPrincipal,
    pub current_membership_version: AggregateVersion,
    pub community_membership: Option<CommunityMembership>,
    pub now_millis: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CollaborationSearchHit {
    pub record: SearchRecordReference,
    pub rank: f32,
    pub observed_at_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SearchRecordReference {
    SignedEvent {
        event_id: EventId,
        author_public_key: PublicKey,
        kind: u16,
    },
    CanonicalDocument {
        document_type: String,
        provenance: Provenance,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchProjectionFreshness {
    Current {
        oldest_projected_at_millis: u64,
    },
    Lagging {
        oldest_projected_at_millis: u64,
        affected_checkpoints: u64,
    },
    Unavailable,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CollaborationSearchResult {
    pub hits: Vec<CollaborationSearchHit>,
    pub page: u32,
    pub projection_freshness: SearchProjectionFreshness,
}

#[derive(Debug, thiserror::Error)]
pub enum SearchRepositoryError {
    #[error("collaboration search requires PostgreSQL")]
    UnsupportedDatabase,
    #[error("collaboration search query is invalid")]
    InvalidQuery,
    #[error("collaboration search is not authorized: {0:?}")]
    Unauthorized(AuthorizationDenial),
    #[error("collaboration search returned an invalid record")]
    InvalidRecord,
    #[error("collaboration search is unavailable")]
    Unavailable(#[source] DbErr),
}

pub struct CollaborationSearchRepository {
    connection: DatabaseConnection,
}

impl CollaborationSearchRepository {
    pub fn new(connection: DatabaseConnection) -> Result<Self, SearchRepositoryError> {
        if connection.get_database_backend() != DbBackend::Postgres {
            return Err(SearchRepositoryError::UnsupportedDatabase);
        }
        Ok(Self { connection })
    }

    pub async fn search(
        &self,
        access: SearchAccess<'_>,
        query: &CollaborationSearchQuery,
    ) -> Result<CollaborationSearchResult, SearchRepositoryError> {
        authorize_search(&access)?;
        let offset = u64::from(query.page.saturating_sub(1))
            .checked_mul(u64::from(query.results_per_page))
            .and_then(|offset| i64::try_from(offset).ok())
            .ok_or(SearchRepositoryError::InvalidQuery)?;
        let transaction = self
            .connection
            .begin()
            .await
            .map_err(SearchRepositoryError::Unavailable)?;
        let result = async {
            set_tenant(&transaction, access.tenant).await?;
            let rows = transaction
                .query_all(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    SEARCH_SQL,
                    [
                        access.tenant.community_id().as_uuid().into(),
                        query.text.clone().into(),
                        query.mode.database_name().into(),
                        i64::from(query.results_per_page).into(),
                        offset.into(),
                    ],
                ))
                .await
                .map_err(SearchRepositoryError::Unavailable)?;
            let hits = rows
                .into_iter()
                .map(search_hit_from_row)
                .collect::<Result<Vec<_>, _>>()?;
            let freshness = transaction
                .query_one(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    FRESHNESS_SQL,
                    [access.tenant.community_id().as_uuid().into()],
                ))
                .await
                .map_err(SearchRepositoryError::Unavailable)?
                .ok_or(SearchRepositoryError::InvalidRecord)
                .and_then(freshness_from_row)?;
            Ok(CollaborationSearchResult {
                hits,
                page: query.page,
                projection_freshness: freshness,
            })
        }
        .await;
        finish_transaction(transaction, result).await
    }

    pub fn into_connection(self) -> DatabaseConnection {
        self.connection
    }
}

fn authorize_search(access: &SearchAccess<'_>) -> Result<(), SearchRepositoryError> {
    let required_scope =
        AuthorizationScope::new(SEARCH_SCOPE).map_err(|_| SearchRepositoryError::InvalidQuery)?;
    let resource = AuthorizationResource {
        community_id: access.tenant.community_id(),
        kind: AuthorizationResourceKind::Community,
        resource_id: AggregateId::from_uuid(access.tenant.community_id().as_uuid()),
        owner_principal_id: None,
        channel_id: None,
    };
    match authorize(&AuthorizationRequest {
        tenant: access.tenant,
        principal: access.principal,
        required_scope: &required_scope,
        action: AuthorizationAction::Read,
        resource,
        current_membership_version: access.current_membership_version,
        community_membership: access.community_membership,
        current_channel_membership_version: None,
        channel_membership: None,
        delegation: None,
        now_millis: access.now_millis,
    }) {
        AuthorizationDecision::Allowed => Ok(()),
        AuthorizationDecision::Denied(denial) => Err(SearchRepositoryError::Unauthorized(denial)),
    }
}

fn normalize_search_text(text: String) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut normalized = String::with_capacity(trimmed.len().min(SEARCH_TEXT_MAX_CHARS));
    for character in trimmed.chars().take(SEARCH_TEXT_MAX_CHARS) {
        normalized.push(if character == '\0' { ' ' } else { character });
    }
    let normalized = normalized.trim();
    (!normalized.is_empty()).then(|| normalized.to_owned())
}

fn search_hit_from_row(row: QueryResult) -> Result<CollaborationSearchHit, SearchRepositoryError> {
    let record_type: String = row
        .try_get("", "record_type")
        .map_err(|_| SearchRepositoryError::InvalidRecord)?;
    let observed_at_millis = row
        .try_get::<String>("", "observed_at_millis")
        .map_err(|_| SearchRepositoryError::InvalidRecord)?
        .parse::<u64>()
        .map_err(|_| SearchRepositoryError::InvalidRecord)?;
    let rank: f32 = row
        .try_get("", "rank")
        .map_err(|_| SearchRepositoryError::InvalidRecord)?;
    if !rank.is_finite() || rank < 0.0 {
        return Err(SearchRepositoryError::InvalidRecord);
    }
    let record = match record_type.as_str() {
        "signed_event" => {
            let event_id = fixed_event_id(
                row.try_get("", "event_id")
                    .map_err(|_| SearchRepositoryError::InvalidRecord)?,
            )?;
            let author_public_key = fixed_public_key(
                row.try_get("", "author_public_key")
                    .map_err(|_| SearchRepositoryError::InvalidRecord)?,
            )?;
            let kind = row
                .try_get::<i32>("", "event_kind")
                .ok()
                .and_then(|kind| u16::try_from(kind).ok())
                .ok_or(SearchRepositoryError::InvalidRecord)?;
            SearchRecordReference::SignedEvent {
                event_id,
                author_public_key,
                kind,
            }
        }
        "canonical_document" => {
            let source_system = source_system_from_database(
                &row.try_get::<String>("", "source_system")
                    .map_err(|_| SearchRepositoryError::InvalidRecord)?,
            )?;
            let source_record_id = SourceRecordId::new(
                row.try_get::<String>("", "source_record_id")
                    .map_err(|_| SearchRepositoryError::InvalidRecord)?,
            )
            .ok_or(SearchRepositoryError::InvalidRecord)?;
            let source_version = row
                .try_get::<String>("", "source_version")
                .map_err(|_| SearchRepositoryError::InvalidRecord)?;
            if source_version.is_empty() || source_version.len() > 1024 {
                return Err(SearchRepositoryError::InvalidRecord);
            }
            let document_type = row
                .try_get::<String>("", "document_type")
                .map_err(|_| SearchRepositoryError::InvalidRecord)?;
            if !matches!(
                document_type.as_str(),
                "profile"
                    | "community"
                    | "channel"
                    | "project"
                    | "repository"
                    | "task"
                    | "agent"
                    | "workflow"
                    | "media"
            ) {
                return Err(SearchRepositoryError::InvalidRecord);
            }
            let provenance = Provenance::new(source_system, source_record_id, observed_at_millis)
                .with_source_version(source_version);
            SearchRecordReference::CanonicalDocument {
                document_type,
                provenance,
            }
        }
        _ => return Err(SearchRepositoryError::InvalidRecord),
    };
    Ok(CollaborationSearchHit {
        record,
        rank,
        observed_at_millis,
    })
}

fn freshness_from_row(
    row: QueryResult,
) -> Result<SearchProjectionFreshness, SearchRepositoryError> {
    let checkpoint_count = database_count(&row, "checkpoint_count")?;
    if checkpoint_count == 0 {
        return Ok(SearchProjectionFreshness::Unavailable);
    }
    let oldest_projected_at_millis = database_count(&row, "oldest_projected_at_millis")?;
    let all_clean = row
        .try_get::<bool>("", "all_clean")
        .map_err(|_| SearchRepositoryError::InvalidRecord)?;
    if all_clean {
        Ok(SearchProjectionFreshness::Current {
            oldest_projected_at_millis,
        })
    } else {
        Ok(SearchProjectionFreshness::Lagging {
            oldest_projected_at_millis,
            affected_checkpoints: database_count(&row, "affected_count")?,
        })
    }
}

fn database_count(row: &QueryResult, column: &str) -> Result<u64, SearchRepositoryError> {
    row.try_get::<i64>("", column)
        .ok()
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(SearchRepositoryError::InvalidRecord)
}

fn fixed_event_id(bytes: Vec<u8>) -> Result<EventId, SearchRepositoryError> {
    let bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| SearchRepositoryError::InvalidRecord)?;
    Ok(EventId::from_bytes(bytes))
}

fn fixed_public_key(bytes: Vec<u8>) -> Result<PublicKey, SearchRepositoryError> {
    let bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| SearchRepositoryError::InvalidRecord)?;
    Ok(PublicKey::from_bytes(bytes))
}

fn source_system_from_database(value: &str) -> Result<SourceSystem, SearchRepositoryError> {
    match value {
        "zed" => Ok(SourceSystem::Zed),
        "buzz" => Ok(SourceSystem::Buzz),
        "nostr" => Ok(SourceSystem::Nostr),
        "acp" => Ok(SourceSystem::Acp),
        "external_git" => Ok(SourceSystem::ExternalGit),
        _ => Err(SearchRepositoryError::InvalidRecord),
    }
}

async fn set_tenant(
    transaction: &DatabaseTransaction,
    tenant: &TenantContext,
) -> Result<(), SearchRepositoryError> {
    transaction
        .execute(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            SET_TENANT_SQL,
            [tenant.community_id().to_string().into()],
        ))
        .await
        .map_err(SearchRepositoryError::Unavailable)?;
    Ok(())
}

async fn finish_transaction<T>(
    transaction: DatabaseTransaction,
    result: Result<T, SearchRepositoryError>,
) -> Result<T, SearchRepositoryError> {
    match result {
        Ok(value) => {
            transaction
                .commit()
                .await
                .map_err(SearchRepositoryError::Unavailable)?;
            Ok(value)
        }
        Err(error) => {
            transaction
                .rollback()
                .await
                .map_err(SearchRepositoryError::Unavailable)?;
            Err(error)
        }
    }
}
