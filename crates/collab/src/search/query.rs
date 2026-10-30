use collaboration_domain::{Provenance, SourceRecordId, SourceSystem};
use nostr_compat::{EventId, PublicKey};

use super::repository::{
    CollaborationSearchQuery, CollaborationSearchRepository, SearchAccess,
    SearchProjectionFreshness, SearchRecordReference, SearchRepositoryError,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CollaborationSearchResultClass {
    Community,
    Channel,
    Member,
    Project,
    Message,
    Repository,
    Task,
    Agent,
    Workflow,
    Media,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct CanonicalSearchIdentity {
    pub class: CollaborationSearchResultClass,
    pub source_system: SourceSystem,
    pub source_record_id: SourceRecordId,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum CollaborationSearchResultIdentity {
    Canonical(CanonicalSearchIdentity),
    MemberPublicKey(PublicKey),
    MessageEvent(EventId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollaborationSearchResultReference {
    Canonical(Provenance),
    MemberProfile {
        event_id: EventId,
        public_key: PublicKey,
    },
    Message {
        event_id: EventId,
        kind: u16,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypedCollaborationSearchHit {
    pub identity: CollaborationSearchResultIdentity,
    pub class: CollaborationSearchResultClass,
    pub reference: CollaborationSearchResultReference,
    pub rank: f32,
    pub observed_at_millis: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypedCollaborationSearchResult {
    pub hits: Vec<TypedCollaborationSearchHit>,
    pub page: u32,
    pub projection_freshness: SearchProjectionFreshness,
}

#[derive(Debug, thiserror::Error)]
pub enum CollaborationSearchQueryError {
    #[error(transparent)]
    Repository(#[from] SearchRepositoryError),
    #[error("collaboration search returned an unsupported signed-event kind")]
    UnsupportedEventKind,
    #[error("collaboration search returned an unsupported canonical document type")]
    UnsupportedDocumentType,
}

pub struct CollaborationSearchQueries {
    repository: CollaborationSearchRepository,
}

impl CollaborationSearchQueries {
    pub fn new(repository: CollaborationSearchRepository) -> Self {
        Self { repository }
    }

    pub async fn query(
        &self,
        access: SearchAccess<'_>,
        query: &CollaborationSearchQuery,
    ) -> Result<TypedCollaborationSearchResult, CollaborationSearchQueryError> {
        let result = self.repository.search(access, query).await?;
        let hits = result
            .hits
            .into_iter()
            .map(|hit| {
                let (identity, class, reference) = classify_record(hit.record)?;
                Ok(TypedCollaborationSearchHit {
                    identity,
                    class,
                    reference,
                    rank: hit.rank,
                    observed_at_millis: hit.observed_at_millis,
                })
            })
            .collect::<Result<Vec<_>, CollaborationSearchQueryError>>()?;
        Ok(TypedCollaborationSearchResult {
            hits,
            page: result.page,
            projection_freshness: result.projection_freshness,
        })
    }

    pub fn into_repository(self) -> CollaborationSearchRepository {
        self.repository
    }
}

fn classify_record(
    record: SearchRecordReference,
) -> Result<
    (
        CollaborationSearchResultIdentity,
        CollaborationSearchResultClass,
        CollaborationSearchResultReference,
    ),
    CollaborationSearchQueryError,
> {
    match record {
        SearchRecordReference::SignedEvent {
            event_id,
            author_public_key,
            kind: 0,
        } => Ok((
            CollaborationSearchResultIdentity::MemberPublicKey(author_public_key),
            CollaborationSearchResultClass::Member,
            CollaborationSearchResultReference::MemberProfile {
                event_id,
                public_key: author_public_key,
            },
        )),
        SearchRecordReference::SignedEvent {
            event_id,
            author_public_key: _,
            kind,
        } if matches!(kind, 9 | 40002 | 45001 | 45003) => Ok((
            CollaborationSearchResultIdentity::MessageEvent(event_id),
            CollaborationSearchResultClass::Message,
            CollaborationSearchResultReference::Message { event_id, kind },
        )),
        SearchRecordReference::SignedEvent { .. } => {
            Err(CollaborationSearchQueryError::UnsupportedEventKind)
        }
        SearchRecordReference::CanonicalDocument {
            document_type,
            provenance,
        } => {
            let class = document_class(&document_type)?;
            let identity = CollaborationSearchResultIdentity::Canonical(CanonicalSearchIdentity {
                class,
                source_system: provenance.source_system,
                source_record_id: provenance.source_record_id.clone(),
            });
            Ok((
                identity,
                class,
                CollaborationSearchResultReference::Canonical(provenance),
            ))
        }
    }
}

fn document_class(
    document_type: &str,
) -> Result<CollaborationSearchResultClass, CollaborationSearchQueryError> {
    match document_type {
        "profile" => Ok(CollaborationSearchResultClass::Member),
        "community" => Ok(CollaborationSearchResultClass::Community),
        "channel" => Ok(CollaborationSearchResultClass::Channel),
        "project" => Ok(CollaborationSearchResultClass::Project),
        "repository" => Ok(CollaborationSearchResultClass::Repository),
        "task" => Ok(CollaborationSearchResultClass::Task),
        "agent" => Ok(CollaborationSearchResultClass::Agent),
        "workflow" => Ok(CollaborationSearchResultClass::Workflow),
        "media" => Ok(CollaborationSearchResultClass::Media),
        _ => Err(CollaborationSearchQueryError::UnsupportedDocumentType),
    }
}
