use std::{collections::BTreeSet, error::Error, fmt};

use async_trait::async_trait;
use collaboration_domain::{
    AggregateVersion, MAX_THREAD_PAGE_ROWS, NostrEventId, ScopedAggregateId, TenantContext,
};
use sha2::{Digest, Sha256};

use crate::{
    messages::window_repository::{
        ChannelWindowCursor, ChannelWindowPage, MessageProjectionLifecycle, MessageWindowRow,
    },
    migration::cutover_checkpoint::{CutoverAuthority, CutoverCheckpoint, CutoverPhase},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadowQueryKind {
    ChannelWindow,
    ThreadWindow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadowWindowCursor {
    created_at: u64,
    event_id: NostrEventId,
}

impl ShadowWindowCursor {
    pub fn new(created_at: u64, event_id: NostrEventId) -> Result<Self, ShadowReadError> {
        if created_at == 0 || event_id.as_bytes().iter().all(|byte| *byte == 0) {
            return Err(ShadowReadError::InvalidInput);
        }
        Ok(Self {
            created_at,
            event_id,
        })
    }

    pub const fn created_at(self) -> u64 {
        self.created_at
    }

    pub const fn event_id(self) -> NostrEventId {
        self.event_id
    }
}

impl TryFrom<ChannelWindowCursor> for ShadowWindowCursor {
    type Error = ShadowReadError;

    fn try_from(cursor: ChannelWindowCursor) -> Result<Self, Self::Error> {
        Self::new(cursor.message_created_at, cursor.source_event_id)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShadowReadQuery {
    aggregate: ScopedAggregateId,
    kind: ShadowQueryKind,
    query_hash: [u8; 32],
    request_cursor: Option<ShadowWindowCursor>,
    overlay_hash: Option<[u8; 32]>,
}

impl ShadowReadQuery {
    pub fn new(
        aggregate: ScopedAggregateId,
        kind: ShadowQueryKind,
        query_hash: [u8; 32],
        request_cursor: Option<ShadowWindowCursor>,
        overlay_hash: Option<[u8; 32]>,
    ) -> Result<Self, ShadowReadError> {
        if aggregate.community_id().as_uuid().is_nil()
            || aggregate.aggregate_id().as_uuid().is_nil()
            || query_hash == [0; 32]
            || overlay_hash.is_some_and(|hash| hash == [0; 32])
        {
            return Err(ShadowReadError::InvalidInput);
        }
        Ok(Self {
            aggregate,
            kind,
            query_hash,
            request_cursor,
            overlay_hash,
        })
    }

    pub const fn aggregate(&self) -> ScopedAggregateId {
        self.aggregate
    }

    pub const fn kind(&self) -> ShadowQueryKind {
        self.kind
    }

    pub const fn query_hash(&self) -> [u8; 32] {
        self.query_hash
    }

    pub const fn request_cursor(&self) -> Option<ShadowWindowCursor> {
        self.request_cursor
    }

    pub const fn overlay_hash(&self) -> Option<[u8; 32]> {
        self.overlay_hash
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadowAuthorizationDenial {
    Unauthenticated,
    NotMember,
    InsufficientRole,
    ResourceDenied,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadowAuthorization {
    Allowed,
    Denied(ShadowAuthorizationDenial),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ShadowReadRow {
    coordinate: [u8; 32],
    version: u64,
    content_hash: [u8; 32],
}

impl ShadowReadRow {
    pub fn new(
        coordinate: [u8; 32],
        version: u64,
        content_hash: [u8; 32],
    ) -> Result<Self, ShadowReadError> {
        if coordinate == [0; 32] || version == 0 || content_hash == [0; 32] {
            return Err(ShadowReadError::InvalidInput);
        }
        Ok(Self {
            coordinate,
            version,
            content_hash,
        })
    }

    pub const fn coordinate(self) -> [u8; 32] {
        self.coordinate
    }

    pub const fn version(self) -> u64 {
        self.version
    }

    pub const fn content_hash(self) -> [u8; 32] {
        self.content_hash
    }

    pub fn from_message(row: &MessageWindowRow) -> Result<Self, ShadowReadError> {
        let mut hasher = Sha256::new();
        hasher.update(row.community_id.as_uuid().as_bytes());
        hasher.update(row.message_id.as_uuid().as_bytes());
        hasher.update(row.channel_id.as_uuid().as_bytes());
        hasher.update(row.source_event_id.as_bytes());
        hasher.update(row.current_event_id.as_bytes());
        hasher.update(row.author_principal_id.as_uuid().as_bytes());
        hasher.update(row.message_created_at.to_be_bytes());
        hasher.update([match row.lifecycle {
            MessageProjectionLifecycle::Active => 0,
            MessageProjectionLifecycle::Edited => 1,
        }]);
        hasher.update(row.message_version.to_be_bytes());
        hasher.update(row.projected_at_micros.to_be_bytes());
        Self::new(
            *row.source_event_id.as_bytes(),
            row.message_version,
            hasher.finalize().into(),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShadowReadResult {
    aggregate: ScopedAggregateId,
    authorization: ShadowAuthorization,
    rows: Vec<ShadowReadRow>,
    next_cursor: Option<ShadowWindowCursor>,
    overlay: Vec<ShadowReadRow>,
}

impl ShadowReadResult {
    pub fn new(
        aggregate: ScopedAggregateId,
        authorization: ShadowAuthorization,
        rows: Vec<ShadowReadRow>,
        next_cursor: Option<ShadowWindowCursor>,
        overlay: Vec<ShadowReadRow>,
    ) -> Result<Self, ShadowReadError> {
        if aggregate.community_id().as_uuid().is_nil()
            || aggregate.aggregate_id().as_uuid().is_nil()
            || rows.len() > MAX_THREAD_PAGE_ROWS
            || overlay.len() > MAX_THREAD_PAGE_ROWS
            || has_duplicate_coordinates(&rows)
            || has_duplicate_coordinates(&overlay)
            || (matches!(authorization, ShadowAuthorization::Denied(_))
                && (!rows.is_empty() || next_cursor.is_some() || !overlay.is_empty()))
        {
            return Err(ShadowReadError::InvalidInput);
        }
        Ok(Self {
            aggregate,
            authorization,
            rows,
            next_cursor,
            overlay,
        })
    }

    pub fn from_channel_page(
        aggregate: ScopedAggregateId,
        page: &ChannelWindowPage,
        overlay: &[MessageWindowRow],
    ) -> Result<Self, ShadowReadError> {
        if page.rows.iter().chain(overlay).any(|row| {
            row.community_id != aggregate.community_id()
                || row.channel_id != aggregate.aggregate_id()
        }) {
            return Err(ShadowReadError::TenantBoundaryViolation);
        }
        let rows = page
            .rows
            .iter()
            .map(ShadowReadRow::from_message)
            .collect::<Result<Vec<_>, _>>()?;
        let overlay = overlay
            .iter()
            .map(ShadowReadRow::from_message)
            .collect::<Result<Vec<_>, _>>()?;
        let next_cursor = page
            .next_cursor
            .map(ShadowWindowCursor::try_from)
            .transpose()?;
        Self::new(
            aggregate,
            ShadowAuthorization::Allowed,
            rows,
            next_cursor,
            overlay,
        )
    }

    pub const fn aggregate(&self) -> ScopedAggregateId {
        self.aggregate
    }

    pub const fn authorization(&self) -> ShadowAuthorization {
        self.authorization
    }

    pub fn rows(&self) -> &[ShadowReadRow] {
        &self.rows
    }

    pub const fn next_cursor(&self) -> Option<ShadowWindowCursor> {
        self.next_cursor
    }

    pub fn overlay(&self) -> &[ShadowReadRow] {
        &self.overlay
    }

    fn fingerprints(&self) -> ShadowResultFingerprints {
        ShadowResultFingerprints {
            authorization_hash: hash_authorization(self.authorization),
            content_hash: hash_rows(&self.rows, true),
            order_hash: hash_rows(&self.rows, false),
            cursor_hash: self.next_cursor.map(hash_cursor),
            overlay_hash: hash_rows(&self.overlay, false),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ShadowDivergence {
    Authorization,
    Content,
    Order,
    Cursor,
    Overlay,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShadowComparison {
    divergences: BTreeSet<ShadowDivergence>,
    legacy: ShadowResultFingerprints,
    canonical: ShadowResultFingerprints,
}

impl ShadowComparison {
    pub fn compare(legacy: &ShadowReadResult, canonical: &ShadowReadResult) -> Self {
        let mut divergences = BTreeSet::new();
        if legacy.authorization != canonical.authorization {
            divergences.insert(ShadowDivergence::Authorization);
        }
        if canonicalized_rows(&legacy.rows) != canonicalized_rows(&canonical.rows) {
            divergences.insert(ShadowDivergence::Content);
        }
        if ordered_coordinates(&legacy.rows) != ordered_coordinates(&canonical.rows) {
            divergences.insert(ShadowDivergence::Order);
        }
        if legacy.next_cursor != canonical.next_cursor {
            divergences.insert(ShadowDivergence::Cursor);
        }
        if legacy.overlay != canonical.overlay {
            divergences.insert(ShadowDivergence::Overlay);
        }
        Self {
            divergences,
            legacy: legacy.fingerprints(),
            canonical: canonical.fingerprints(),
        }
    }

    pub fn is_match(&self) -> bool {
        self.divergences.is_empty()
    }

    pub fn divergences(&self) -> &BTreeSet<ShadowDivergence> {
        &self.divergences
    }

    pub const fn legacy(&self) -> ShadowResultFingerprints {
        self.legacy
    }

    pub const fn canonical(&self) -> ShadowResultFingerprints {
        self.canonical
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShadowResultFingerprints {
    authorization_hash: [u8; 32],
    content_hash: [u8; 32],
    order_hash: [u8; 32],
    cursor_hash: Option<[u8; 32]>,
    overlay_hash: [u8; 32],
}

impl ShadowResultFingerprints {
    pub const fn authorization_hash(self) -> [u8; 32] {
        self.authorization_hash
    }

    pub const fn content_hash(self) -> [u8; 32] {
        self.content_hash
    }

    pub const fn order_hash(self) -> [u8; 32] {
        self.order_hash
    }

    pub const fn cursor_hash(self) -> Option<[u8; 32]> {
        self.cursor_hash
    }

    pub const fn overlay_hash(self) -> [u8; 32] {
        self.overlay_hash
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadowReadSourceError {
    Unavailable,
    InvalidResponse,
}

impl fmt::Display for ShadowReadSourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Unavailable => "shadow read source is unavailable",
            Self::InvalidResponse => "shadow read source returned an invalid response",
        })
    }
}

impl Error for ShadowReadSourceError {}

#[async_trait]
pub trait ShadowReadSource: Send + Sync {
    async fn read(
        &self,
        tenant: &TenantContext,
        query: &ShadowReadQuery,
    ) -> Result<ShadowReadResult, ShadowReadSourceError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShadowDiagnosticOutcome {
    Compared(ShadowComparison),
    CanonicalUnavailable(ShadowReadSourceError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShadowReadObservation {
    aggregate: ScopedAggregateId,
    checkpoint_version: AggregateVersion,
    query_kind: ShadowQueryKind,
    query_hash: [u8; 32],
    request_cursor: Option<ShadowWindowCursor>,
    request_overlay_hash: Option<[u8; 32]>,
    outcome: ShadowDiagnosticOutcome,
}

impl ShadowReadObservation {
    pub const fn aggregate(&self) -> ScopedAggregateId {
        self.aggregate
    }

    pub const fn checkpoint_version(&self) -> AggregateVersion {
        self.checkpoint_version
    }

    pub const fn query_kind(&self) -> ShadowQueryKind {
        self.query_kind
    }

    pub const fn query_hash(&self) -> [u8; 32] {
        self.query_hash
    }

    pub const fn request_cursor(&self) -> Option<ShadowWindowCursor> {
        self.request_cursor
    }

    pub const fn request_overlay_hash(&self) -> Option<[u8; 32]> {
        self.request_overlay_hash
    }

    pub fn outcome(&self) -> &ShadowDiagnosticOutcome {
        &self.outcome
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadowObservationError {
    Unavailable,
}

impl fmt::Display for ShadowObservationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("shadow read observation storage is unavailable")
    }
}

impl Error for ShadowObservationError {}

#[async_trait]
pub trait ShadowObservationStore: Send + Sync {
    async fn record(
        &self,
        tenant: &TenantContext,
        observation: ShadowReadObservation,
    ) -> Result<(), ShadowObservationError>;
}

#[derive(Debug)]
pub struct LegacyShadowResponse {
    served: ShadowReadResult,
    ticket: ShadowComparisonTicket,
}

impl LegacyShadowResponse {
    pub fn served(&self) -> &ShadowReadResult {
        &self.served
    }

    pub fn into_parts(self) -> (ShadowReadResult, ShadowComparisonTicket) {
        (self.served, self.ticket)
    }
}

#[derive(Debug)]
pub struct ShadowComparisonTicket {
    checkpoint_version: AggregateVersion,
    query: ShadowReadQuery,
    legacy: ShadowReadResult,
}

pub struct ShadowReadCoordinator<Legacy, Canonical, Observations> {
    legacy: Legacy,
    canonical: Canonical,
    observations: Observations,
}

impl<Legacy, Canonical, Observations> ShadowReadCoordinator<Legacy, Canonical, Observations>
where
    Legacy: ShadowReadSource,
    Canonical: ShadowReadSource,
    Observations: ShadowObservationStore,
{
    pub const fn new(legacy: Legacy, canonical: Canonical, observations: Observations) -> Self {
        Self {
            legacy,
            canonical,
            observations,
        }
    }

    pub async fn serve_legacy(
        &self,
        tenant: &TenantContext,
        checkpoint: &CutoverCheckpoint,
        query: ShadowReadQuery,
    ) -> Result<LegacyShadowResponse, ShadowReadError> {
        validate_request(tenant, checkpoint, &query)?;
        let legacy = self
            .legacy
            .read(tenant, &query)
            .await
            .map_err(ShadowReadError::LegacyUnavailable)?;
        if legacy.aggregate != query.aggregate {
            return Err(ShadowReadError::LegacyUnavailable(
                ShadowReadSourceError::InvalidResponse,
            ));
        }
        Ok(LegacyShadowResponse {
            served: legacy.clone(),
            ticket: ShadowComparisonTicket {
                checkpoint_version: checkpoint.version(),
                query,
                legacy,
            },
        })
    }

    pub async fn compare(
        &self,
        tenant: &TenantContext,
        ticket: ShadowComparisonTicket,
    ) -> Result<ShadowDiagnosticOutcome, ShadowReadError> {
        if tenant.community_id() != ticket.query.aggregate.community_id() {
            return Err(ShadowReadError::TenantBoundaryViolation);
        }
        let outcome = match self.canonical.read(tenant, &ticket.query).await {
            Ok(canonical) if canonical.aggregate == ticket.query.aggregate => {
                ShadowDiagnosticOutcome::Compared(ShadowComparison::compare(
                    &ticket.legacy,
                    &canonical,
                ))
            }
            Ok(_) => ShadowDiagnosticOutcome::CanonicalUnavailable(
                ShadowReadSourceError::InvalidResponse,
            ),
            Err(error) => ShadowDiagnosticOutcome::CanonicalUnavailable(error),
        };
        let observation = ShadowReadObservation {
            aggregate: ticket.query.aggregate,
            checkpoint_version: ticket.checkpoint_version,
            query_kind: ticket.query.kind,
            query_hash: ticket.query.query_hash,
            request_cursor: ticket.query.request_cursor,
            request_overlay_hash: ticket.query.overlay_hash,
            outcome: outcome.clone(),
        };
        self.observations
            .record(tenant, observation)
            .await
            .map_err(ShadowReadError::ObservationUnavailable)?;
        Ok(outcome)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadowReadError {
    InvalidInput,
    TenantBoundaryViolation,
    CheckpointMismatch,
    ShadowReadNotPermitted,
    LegacyUnavailable(ShadowReadSourceError),
    ObservationUnavailable(ShadowObservationError),
}

impl fmt::Display for ShadowReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidInput => "shadow read input is invalid",
            Self::TenantBoundaryViolation => "shadow read crossed its tenant boundary",
            Self::CheckpointMismatch => "shadow read checkpoint does not match its query",
            Self::ShadowReadNotPermitted => "shadow reads are not permitted for this checkpoint",
            Self::LegacyUnavailable(_) => "legacy serving read is unavailable",
            Self::ObservationUnavailable(_) => "shadow read observation could not be recorded",
        })
    }
}

impl Error for ShadowReadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::LegacyUnavailable(error) => Some(error),
            Self::ObservationUnavailable(error) => Some(error),
            _ => None,
        }
    }
}

fn validate_request(
    tenant: &TenantContext,
    checkpoint: &CutoverCheckpoint,
    query: &ShadowReadQuery,
) -> Result<(), ShadowReadError> {
    if tenant.community_id() != query.aggregate.community_id() {
        return Err(ShadowReadError::TenantBoundaryViolation);
    }
    if checkpoint.aggregate() != query.aggregate {
        return Err(ShadowReadError::CheckpointMismatch);
    }
    if checkpoint.authority() != CutoverAuthority::Legacy
        || checkpoint.phase() < CutoverPhase::CommunicationReadShadow
        || checkpoint.phase() > CutoverPhase::CommunicationWriteCutover
    {
        return Err(ShadowReadError::ShadowReadNotPermitted);
    }
    Ok(())
}

fn has_duplicate_coordinates(rows: &[ShadowReadRow]) -> bool {
    let coordinates = rows
        .iter()
        .map(|row| row.coordinate)
        .collect::<BTreeSet<_>>();
    coordinates.len() != rows.len()
}

fn canonicalized_rows(rows: &[ShadowReadRow]) -> Vec<ShadowReadRow> {
    let mut rows = rows.to_vec();
    rows.sort_unstable();
    rows
}

fn ordered_coordinates(rows: &[ShadowReadRow]) -> Vec<[u8; 32]> {
    rows.iter().map(|row| row.coordinate).collect()
}

fn hash_authorization(authorization: ShadowAuthorization) -> [u8; 32] {
    Sha256::digest([match authorization {
        ShadowAuthorization::Allowed => 0,
        ShadowAuthorization::Denied(ShadowAuthorizationDenial::Unauthenticated) => 1,
        ShadowAuthorization::Denied(ShadowAuthorizationDenial::NotMember) => 2,
        ShadowAuthorization::Denied(ShadowAuthorizationDenial::InsufficientRole) => 3,
        ShadowAuthorization::Denied(ShadowAuthorizationDenial::ResourceDenied) => 4,
    }])
    .into()
}

fn hash_rows(rows: &[ShadowReadRow], canonical_order: bool) -> [u8; 32] {
    let rows = if canonical_order {
        canonicalized_rows(rows)
    } else {
        rows.to_vec()
    };
    let mut hasher = Sha256::new();
    for row in rows {
        hasher.update(row.coordinate);
        hasher.update(row.version.to_be_bytes());
        hasher.update(row.content_hash);
    }
    hasher.finalize().into()
}

fn hash_cursor(cursor: ShadowWindowCursor) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(cursor.created_at.to_be_bytes());
    hasher.update(cursor.event_id.as_bytes());
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use collaboration_domain::{
        AggregateId, AggregateType, CommunityId, OperationId, TrustedTenantRoute,
    };
    use uuid::Uuid;

    use super::*;
    use crate::migration::cutover_checkpoint::{
        CutoverCursor, CutoverGateEvidence, CutoverIntegrity, CutoverTransition,
        CutoverTransitionOutcome,
    };

    fn hash(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn aggregate(community_id: CommunityId) -> ScopedAggregateId {
        ScopedAggregateId::new(
            community_id,
            AggregateType::Conversation,
            AggregateId::from_uuid(Uuid::from_u128(100)),
        )
    }

    fn tenant(community_id: CommunityId) -> TenantContext {
        TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community_id, "shadow-read")
                    .expect("trusted tenant route"),
            ),
            &[],
        )
        .expect("tenant context")
    }

    fn query(community_id: CommunityId) -> ShadowReadQuery {
        ShadowReadQuery::new(
            aggregate(community_id),
            ShadowQueryKind::ChannelWindow,
            hash(90),
            Some(
                ShadowWindowCursor::new(100, NostrEventId::from_bytes(hash(91)))
                    .expect("request cursor"),
            ),
            Some(hash(92)),
        )
        .expect("query")
    }

    fn row(coordinate: u8, content: u8) -> ShadowReadRow {
        ShadowReadRow::new(hash(coordinate), 1, hash(content)).expect("row")
    }

    fn result(
        community_id: CommunityId,
        authorization: ShadowAuthorization,
        rows: Vec<ShadowReadRow>,
        overlay: Vec<ShadowReadRow>,
    ) -> ShadowReadResult {
        let cursor = matches!(authorization, ShadowAuthorization::Allowed).then(|| {
            ShadowWindowCursor::new(200, NostrEventId::from_bytes(hash(93)))
                .expect("response cursor")
        });
        ShadowReadResult::new(
            aggregate(community_id),
            authorization,
            rows,
            cursor,
            overlay,
        )
        .expect("result")
    }

    #[derive(Clone)]
    struct TestSource {
        result: Result<ShadowReadResult, ShadowReadSourceError>,
        calls: Arc<AtomicUsize>,
    }

    impl TestSource {
        fn new(result: Result<ShadowReadResult, ShadowReadSourceError>) -> Self {
            Self {
                result,
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl ShadowReadSource for TestSource {
        async fn read(
            &self,
            _tenant: &TenantContext,
            _query: &ShadowReadQuery,
        ) -> Result<ShadowReadResult, ShadowReadSourceError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.result.clone()
        }
    }

    #[derive(Clone, Default)]
    struct TestObservations {
        records: Arc<Mutex<Vec<ShadowReadObservation>>>,
    }

    impl TestObservations {
        fn records(&self) -> Vec<ShadowReadObservation> {
            self.records.lock().expect("observation lock").clone()
        }
    }

    #[async_trait]
    impl ShadowObservationStore for TestObservations {
        async fn record(
            &self,
            _tenant: &TenantContext,
            observation: ShadowReadObservation,
        ) -> Result<(), ShadowObservationError> {
            self.records
                .lock()
                .map_err(|_| ShadowObservationError::Unavailable)?
                .push(observation);
            Ok(())
        }
    }

    #[tokio::test]
    async fn legacy_is_served_before_content_and_order_comparison() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let checkpoint = CutoverCheckpoint::new(
            aggregate(community_id),
            CutoverPhase::CommunicationReadShadow,
        )
        .expect("checkpoint");
        let legacy_result = result(
            community_id,
            ShadowAuthorization::Allowed,
            vec![row(1, 11), row(2, 12)],
            vec![row(3, 13)],
        );
        let canonical_result = result(
            community_id,
            ShadowAuthorization::Allowed,
            vec![row(2, 12), row(1, 14)],
            vec![row(3, 13)],
        );
        let legacy = TestSource::new(Ok(legacy_result.clone()));
        let canonical = TestSource::new(Ok(canonical_result));
        let observations = TestObservations::default();
        let coordinator =
            ShadowReadCoordinator::new(legacy.clone(), canonical.clone(), observations.clone());

        let response = coordinator
            .serve_legacy(&tenant, &checkpoint, query(community_id))
            .await
            .expect("legacy response");
        assert_eq!(response.served(), &legacy_result);
        assert_eq!(legacy.calls(), 1);
        assert_eq!(canonical.calls(), 0);
        assert!(observations.records().is_empty());

        let (_, ticket) = response.into_parts();
        let ShadowDiagnosticOutcome::Compared(comparison) = coordinator
            .compare(&tenant, ticket)
            .await
            .expect("comparison")
        else {
            panic!("canonical response must be compared");
        };
        assert_eq!(
            comparison.divergences(),
            &BTreeSet::from([ShadowDivergence::Content, ShadowDivergence::Order])
        );
        assert_eq!(canonical.calls(), 1);
        assert_eq!(observations.records().len(), 1);
    }

    #[tokio::test]
    async fn authorization_cursor_and_overlay_divergence_are_attributed() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let checkpoint = CutoverCheckpoint::new(
            aggregate(community_id),
            CutoverPhase::CommunicationReadShadow,
        )
        .expect("checkpoint");
        let legacy_result = result(
            community_id,
            ShadowAuthorization::Allowed,
            vec![row(1, 11)],
            vec![row(2, 12)],
        );
        let canonical_result = ShadowReadResult::new(
            aggregate(community_id),
            ShadowAuthorization::Denied(ShadowAuthorizationDenial::NotMember),
            Vec::new(),
            None,
            Vec::new(),
        )
        .expect("denied result");
        let observations = TestObservations::default();
        let coordinator = ShadowReadCoordinator::new(
            TestSource::new(Ok(legacy_result)),
            TestSource::new(Ok(canonical_result)),
            observations.clone(),
        );
        let response = coordinator
            .serve_legacy(&tenant, &checkpoint, query(community_id))
            .await
            .expect("legacy response");
        let (_, ticket) = response.into_parts();
        let ShadowDiagnosticOutcome::Compared(comparison) = coordinator
            .compare(&tenant, ticket)
            .await
            .expect("comparison")
        else {
            panic!("canonical response must be compared");
        };
        assert_eq!(
            comparison.divergences(),
            &BTreeSet::from([
                ShadowDivergence::Authorization,
                ShadowDivergence::Content,
                ShadowDivergence::Order,
                ShadowDivergence::Cursor,
                ShadowDivergence::Overlay,
            ])
        );
        let records = observations.records();
        assert_eq!(records[0].aggregate(), aggregate(community_id));
        assert_eq!(records[0].query_hash(), hash(90));
        assert_eq!(records[0].request_overlay_hash(), Some(hash(92)));
    }

    #[tokio::test]
    async fn canonical_failure_is_diagnostic_after_legacy_service() {
        let community_id = community(1);
        let tenant = tenant(community_id);
        let checkpoint = CutoverCheckpoint::new(
            aggregate(community_id),
            CutoverPhase::CommunicationReadShadow,
        )
        .expect("checkpoint");
        let served = result(
            community_id,
            ShadowAuthorization::Allowed,
            vec![row(1, 11)],
            Vec::new(),
        );
        let observations = TestObservations::default();
        let coordinator = ShadowReadCoordinator::new(
            TestSource::new(Ok(served.clone())),
            TestSource::new(Err(ShadowReadSourceError::Unavailable)),
            observations.clone(),
        );
        let response = coordinator
            .serve_legacy(&tenant, &checkpoint, query(community_id))
            .await
            .expect("legacy response");
        assert_eq!(response.served(), &served);
        let (_, ticket) = response.into_parts();
        assert_eq!(
            coordinator.compare(&tenant, ticket).await,
            Ok(ShadowDiagnosticOutcome::CanonicalUnavailable(
                ShadowReadSourceError::Unavailable
            ))
        );
        assert!(matches!(
            observations.records()[0].outcome(),
            ShadowDiagnosticOutcome::CanonicalUnavailable(ShadowReadSourceError::Unavailable)
        ));
    }

    #[tokio::test]
    async fn wrong_tenant_and_nonlegacy_authority_reject_before_reads() {
        let community_id = community(1);
        let checkpoint = CutoverCheckpoint::new(
            aggregate(community_id),
            CutoverPhase::CommunicationReadShadow,
        )
        .expect("checkpoint");
        let source = TestSource::new(Ok(result(
            community_id,
            ShadowAuthorization::Allowed,
            Vec::new(),
            Vec::new(),
        )));
        let coordinator =
            ShadowReadCoordinator::new(source.clone(), source.clone(), TestObservations::default());
        assert!(matches!(
            coordinator
                .serve_legacy(&tenant(community(2)), &checkpoint, query(community_id))
                .await,
            Err(ShadowReadError::TenantBoundaryViolation)
        ));
        assert_eq!(source.calls(), 0);

        let cursor = CutoverCursor::new(1, hash(1)).expect("cutover cursor");
        let integrity =
            CutoverIntegrity::new(Some(hash(2)), Some(hash(2))).expect("cutover integrity");
        let gates =
            CutoverGateEvidence::new(Some(hash(3)), Some(hash(4)), Some(hash(5)), Some(hash(6)))
                .expect("cutover gates");
        let transition = CutoverTransition {
            operation_id: OperationId::from_uuid(Uuid::from_u128(200)),
            expected_version: checkpoint.version(),
            phase: CutoverPhase::CommunicationWriteCutover,
            authority: CutoverAuthority::Canonical,
            source_cursor: cursor,
            target_cursor: cursor,
            integrity,
            gates,
            reversible_boundary_label: Some("before-shadow-test".to_string()),
        };
        let CutoverTransitionOutcome::Advanced(canonical_checkpoint) = checkpoint
            .transition(&transition)
            .expect("canonical checkpoint")
        else {
            panic!("authority must advance");
        };
        assert!(matches!(
            coordinator
                .serve_legacy(
                    &tenant(community_id),
                    &canonical_checkpoint,
                    query(community_id),
                )
                .await,
            Err(ShadowReadError::ShadowReadNotPermitted)
        ));
        assert_eq!(source.calls(), 0);
    }
}
