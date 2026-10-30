use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex, MutexGuard},
};

use async_trait::async_trait;
use collab::{
    push::outbox::PushWakeTerminalOutcome,
    search::{
        indexer::{
            CollaborationSearchIndexer, SearchDocumentType, SearchExclusionReason,
            SearchIndexerError, SearchIndexingOutcome, SearchProjectionOperation,
        },
        query::CollaborationSearchQueries,
        repository::{
            CollaborationSearchQuery, CollaborationSearchRepository, SearchAccess, SearchMode,
            SearchRepositoryError,
        },
    },
    tenant_admission::bind_rpc_tenant,
};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationScope, ChannelMembership,
    CommunityId, CommunityMembership, MembershipRole, MembershipStatus, NotificationCandidate,
    NotificationDevicePermissions, NotificationMembership, NotificationPermission,
    NotificationPrivacy, NotificationReadState, NotificationReason, NotificationSourceId,
    NotificationSuppression, PrincipalId, PrincipalScopes, PushCapabilityReference,
    PushEndpointGeneration, PushInstallationId, PushLeaseAddress, PushLeaseGeneration,
    ServiceAccountId, SourceRecordId, SourceSystem, TenantContext, TrustedTenantRoute,
};
use gpui::{AppContext as _, TestAppContext};
use notifications::collaboration::{
    CollaborationNotificationDispatch, CollaborationNotificationDispatcher,
    CollaborationNotificationRecord,
};
use push_gateway::executor::{
    PushAuthorizationDecision, PushAuthorizationError, PushDeliveryRequest, PushExecutionSummary,
    PushGatewayClaim, PushGatewayClock, PushGatewayExecutor, PushGatewayExecutorError,
    PushGatewayWake, PushProvider, PushProviderError, PushProviderOutcome, PushWakeAuthorization,
    PushWakeStore, PushWakeStoreError,
};
use sea_orm::{DatabaseBackend, MockDatabase, MockExecResult, Value as SeaValue};
use search::collaboration_search::{
    CollaborationSearchPresentation, CollaborationSearchView, NativeResultGroup,
    SearchPresentationItem,
};
use uuid::Uuid;
use workspace::collaborative_navigation::CollaborativeNavigationTarget;

const PRIVATE_CONTENT: &str = "cross-community private launch code";
const PRIVATE_TITLE: &str = "New private activity";
const PRIVATE_BODY: &str = "Open Zed to view it.";

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ClientPath {
    BuzzCompatibility,
    ZedConsolidated,
}

impl ClientPath {
    const ALL: [Self; 2] = [Self::BuzzCompatibility, Self::ZedConsolidated];

    const fn source_system(self) -> SourceSystem {
        match self {
            Self::BuzzCompatibility => SourceSystem::Buzz,
            Self::ZedConsolidated => SourceSystem::Zed,
        }
    }

    const fn source_system_database_name(self) -> &'static str {
        match self {
            Self::BuzzCompatibility => "buzz",
            Self::ZedConsolidated => "zed",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::BuzzCompatibility => "buzz-compatibility",
            Self::ZedConsolidated => "zed-consolidated",
        }
    }

    fn private_record_id(self) -> String {
        format!("{}-private-event", self.label())
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum PrivacySurface {
    ForeignSearch,
    OwnSearch,
    PrivateIndex,
    ForeignSearchPresentation,
    PrivateNotification,
    ForeignNotification,
    WakeDelivery,
    ForeignWake,
}

#[derive(Debug)]
enum PublicOutcome {
    Unavailable {
        operation_count: usize,
    },
    References(String),
    Excluded(String),
    Sanitized(String),
    PrivatePreview {
        title: String,
        body: String,
        rendered: String,
    },
    Wake {
        payload: String,
        rendered: String,
    },
}

#[derive(Debug)]
struct PrivacyObservation {
    client: ClientPath,
    surface: PrivacySurface,
    outcome: PublicOutcome,
}

fn audit_privacy_trace(observations: &[PrivacyObservation], surfaces: &[PrivacySurface]) {
    let expected = ClientPath::ALL
        .into_iter()
        .flat_map(|client| {
            surfaces
                .iter()
                .copied()
                .map(move |surface| (client, surface))
        })
        .collect::<BTreeSet<_>>();
    let actual = observations
        .iter()
        .map(|observation| (observation.client, observation.surface))
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected, "privacy trace coverage breach");

    for observation in observations {
        let rendered = match &observation.outcome {
            PublicOutcome::Unavailable { operation_count } => {
                assert_eq!(
                    *operation_count, 0,
                    "foreign request reached {:?} storage/provider work",
                    observation.surface
                );
                "unavailable"
            }
            PublicOutcome::References(rendered)
            | PublicOutcome::Excluded(rendered)
            | PublicOutcome::Sanitized(rendered) => rendered,
            PublicOutcome::PrivatePreview {
                title,
                body,
                rendered,
            } => {
                assert_eq!(title, PRIVATE_TITLE);
                assert_eq!(body, PRIVATE_BODY);
                rendered
            }
            PublicOutcome::Wake { payload, rendered } => {
                assert_eq!(payload, "\"reconnect\"");
                rendered
            }
        };
        assert!(
            !rendered.contains(PRIVATE_CONTENT),
            "private content leaked through {:?} for {:?}",
            observation.surface,
            observation.client
        );
        for client in ClientPath::ALL {
            assert!(
                !rendered.contains(&client.private_record_id()),
                "private source identifier leaked through {:?} for {:?}",
                observation.surface,
                observation.client
            );
        }
        match observation.surface {
            PrivacySurface::ForeignSearch
            | PrivacySurface::ForeignNotification
            | PrivacySurface::ForeignWake => {
                assert!(matches!(
                    &observation.outcome,
                    PublicOutcome::Unavailable { .. }
                ));
            }
            PrivacySurface::PrivateIndex => {
                assert!(matches!(&observation.outcome, PublicOutcome::Excluded(_)));
            }
            PrivacySurface::PrivateNotification => {
                assert!(matches!(
                    &observation.outcome,
                    PublicOutcome::PrivatePreview { .. }
                ));
            }
            PrivacySurface::WakeDelivery => {
                assert!(matches!(&observation.outcome, PublicOutcome::Wake { .. }));
            }
            PrivacySurface::OwnSearch => {
                assert!(matches!(&observation.outcome, PublicOutcome::References(_)));
            }
            PrivacySurface::ForeignSearchPresentation => {
                assert!(matches!(&observation.outcome, PublicOutcome::Sanitized(_)));
            }
        }
    }
}

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn principal_id(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn tenant(community_id: CommunityId) -> TenantContext {
    bind_rpc_tenant(
        Some(
            TrustedTenantRoute::from_listener(community_id, "search-push-privacy")
                .expect("trusted tenant route"),
        ),
        &[],
    )
    .expect("tenant")
}

fn search_principal(community_id: CommunityId, value: u128) -> AuthenticatedPrincipal {
    let scope = AuthorizationScope::new("collaboration:search").expect("search scope");
    AuthenticatedPrincipal::zed_account(
        principal_id(value),
        community_id,
        ServiceAccountId::new(7),
        PrincipalScopes::new([scope]).expect("principal scopes"),
    )
}

fn membership(
    community_id: CommunityId,
    principal: &AuthenticatedPrincipal,
) -> CommunityMembership {
    CommunityMembership {
        community_id,
        principal_id: principal.principal_id(),
        role: MembershipRole::Member,
        status: MembershipStatus::Active,
        version: AggregateVersion::FIRST,
    }
}

fn success() -> MockExecResult {
    MockExecResult {
        last_insert_id: 0,
        rows_affected: 1,
    }
}

fn search_document_row(path: ClientPath) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("record_type".into(), "canonical_document".to_owned().into()),
        (
            "source_system".into(),
            path.source_system_database_name().to_owned().into(),
        ),
        (
            "source_record_id".into(),
            format!("{}:visible", path.label()).into(),
        ),
        ("source_version".into(), "1".to_owned().into()),
        ("document_type".into(), "project".to_owned().into()),
        (
            "observed_at_millis".into(),
            "1900000000000".to_owned().into(),
        ),
        ("rank".into(), 0.5_f32.into()),
    ])
}

fn current_freshness_row() -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("checkpoint_count".into(), 1_i64.into()),
        ("all_clean".into(), true.into()),
        (
            "oldest_projected_at_millis".into(),
            1_900_000_000_000_i64.into(),
        ),
        ("affected_count".into(), 0_i64.into()),
    ])
}

fn outbox_row(path: ClientPath, sequence: i64, payload: Vec<u8>) -> BTreeMap<String, SeaValue> {
    BTreeMap::from([
        ("outbox_sequence".into(), sequence.into()),
        (
            "topic".into(),
            "collaboration.search.document.v1".to_owned().into(),
        ),
        (
            "source_system".into(),
            path.source_system_database_name().to_owned().into(),
        ),
        (
            "source_record_id".into(),
            format!("{}:private", path.label()).into(),
        ),
        ("source_version".into(), sequence.to_string().into()),
        (
            "source_observed_at_millis".into(),
            1_900_000_000_000_i64.into(),
        ),
        (
            "source_integrity_algorithm".into(),
            Option::<String>::None.into(),
        ),
        (
            "source_integrity_value".into(),
            Option::<String>::None.into(),
        ),
        ("payload".into(), payload.into()),
    ])
}

#[tokio::test]
async fn mixed_version_search_authorizes_and_excludes_private_content_before_limit() {
    let own_community = community(1);
    let foreign_community = community(2);
    let own_tenant = tenant(own_community);
    let own_principal = search_principal(own_community, 11);
    let foreign_principal = search_principal(foreign_community, 12);
    let query =
        CollaborationSearchQuery::new("launch", SearchMode::FullText, 1, 1).expect("bounded query");
    let mut observations = Vec::new();

    for path in ClientPath::ALL {
        let denied_repository = CollaborationSearchRepository::new(
            MockDatabase::new(DatabaseBackend::Postgres).into_connection(),
        )
        .expect("search repository");
        let denied = denied_repository
            .search(
                SearchAccess {
                    tenant: &own_tenant,
                    principal: &foreign_principal,
                    current_membership_version: AggregateVersion::FIRST,
                    community_membership: Some(membership(foreign_community, &foreign_principal)),
                    now_millis: 1_900_000_000_000,
                },
                &query,
            )
            .await;
        assert!(matches!(
            denied,
            Err(SearchRepositoryError::Unauthorized(_))
        ));
        let denied_log = denied_repository.into_connection().into_transaction_log();
        observations.push(PrivacyObservation {
            client: path,
            surface: PrivacySurface::ForeignSearch,
            outcome: PublicOutcome::Unavailable {
                operation_count: denied_log.len(),
            },
        });

        let database = MockDatabase::new(DatabaseBackend::Postgres)
            .append_exec_results([success()])
            .append_query_results([
                vec![search_document_row(path)],
                vec![current_freshness_row()],
            ])
            .into_connection();
        let queries = CollaborationSearchQueries::new(
            CollaborationSearchRepository::new(database).expect("search repository"),
        );
        let result = queries
            .query(
                SearchAccess {
                    tenant: &own_tenant,
                    principal: &own_principal,
                    current_membership_version: AggregateVersion::FIRST,
                    community_membership: Some(membership(own_community, &own_principal)),
                    now_millis: 1_900_000_000_000,
                },
                &query,
            )
            .await
            .expect("authorized search");
        assert_eq!(result.hits.len(), 1);
        let rendered_result = format!("{result:?}");
        let log = format!(
            "{:#?}",
            queries
                .into_repository()
                .into_connection()
                .into_transaction_log()
        );
        let visibility = log
            .find("document.visibility_scope = 'community'")
            .expect("privacy predicate");
        let ranking = log.find("ORDER BY rank DESC").expect("ranking");
        let limit = log.find("LIMIT $4").expect("limit");
        assert!(visibility < ranking && ranking < limit);
        assert!(!log.contains("event.content"));
        observations.push(PrivacyObservation {
            client: path,
            surface: PrivacySurface::OwnSearch,
            outcome: PublicOutcome::References(rendered_result),
        });

        let exclusion = SearchProjectionOperation::exclude(
            SearchDocumentType::Project,
            SearchExclusionReason::DirectMessage,
        )
        .encode()
        .expect("private exclusion");
        let content_bearing_exclusion = serde_json::to_vec(&serde_json::json!({
            "contract_version": 1,
            "document_type": "project",
            "mutation": {
                "action": "exclude",
                "reason": "direct_message",
                "body": PRIVATE_CONTENT,
            },
        }))
        .expect("malformed private operation");
        let database = MockDatabase::new(DatabaseBackend::Postgres)
            .append_query_results([
                vec![outbox_row(path, 1, exclusion)],
                vec![outbox_row(path, 2, content_bearing_exclusion)],
            ])
            .append_exec_results([success(), success(), success(), success()])
            .into_connection();
        let indexer = CollaborationSearchIndexer::new(database).expect("search indexer");
        assert_eq!(
            indexer
                .index_outbox_sequence(&own_tenant, 1)
                .await
                .expect("private exclusion"),
            SearchIndexingOutcome::Excluded(SearchExclusionReason::DirectMessage)
        );
        assert!(matches!(
            indexer.index_outbox_sequence(&own_tenant, 2).await,
            Err(SearchIndexerError::InvalidInput)
        ));
        let log = format!("{:#?}", indexer.into_connection().into_transaction_log());
        observations.push(PrivacyObservation {
            client: path,
            surface: PrivacySurface::PrivateIndex,
            outcome: PublicOutcome::Excluded(log),
        });
    }

    audit_privacy_trace(
        &observations,
        &[
            PrivacySurface::ForeignSearch,
            PrivacySurface::OwnSearch,
            PrivacySurface::PrivateIndex,
        ],
    );
}

fn notification_candidate(
    path: ClientPath,
    source_community: CommunityId,
    membership_community: CommunityId,
) -> NotificationCandidate {
    let recipient = principal_id(20);
    let channel_id = AggregateId::from_uuid(Uuid::from_u128(30));
    NotificationCandidate {
        source: NotificationSourceId::new(
            source_community,
            path.source_system(),
            SourceRecordId::new(path.private_record_id()).expect("source record ID"),
        ),
        recipient_principal_id: recipient,
        author_principal_id: principal_id(21),
        channel_id: Some(channel_id),
        reason: NotificationReason::Mention,
        membership: NotificationMembership::channel(
            CommunityMembership {
                community_id: membership_community,
                principal_id: recipient,
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            },
            ChannelMembership {
                community_id: membership_community,
                channel_id,
                principal_id: recipient,
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            },
        ),
        privacy: NotificationPrivacy::Private {
            recipient_is_participant: true,
        },
        read_state: NotificationReadState::Unread,
        muted: false,
    }
}

#[gpui::test]
fn mixed_version_search_presentation_and_native_notification_are_sanitized(
    cx: &mut TestAppContext,
) {
    cx.update(|cx| cx.set_app_identity("dev.zed.search-push-privacy", "Zed Tests"));
    let dispatcher = cx.update(|cx| CollaborationNotificationDispatcher::new(cx, |_, _| true));
    let mut observations = Vec::new();

    for path in ClientPath::ALL {
        let view = cx.new(|cx| {
            CollaborationSearchView::new(
                vec![SearchPresentationItem::native(
                    format!("native-{}", path.label()),
                    NativeResultGroup::File,
                    "local.rs",
                    None,
                )],
                CollaborationSearchPresentation::unauthorized(),
                cx,
            )
        });
        let (item_count, status) = cx.read(|cx| {
            let presentation = view.read(cx);
            (
                presentation.ordered_items().len(),
                presentation.collaboration_status_label().to_string(),
            )
        });
        assert_eq!(item_count, 1);
        assert_eq!(status, "Collaboration results unavailable");
        observations.push(PrivacyObservation {
            client: path,
            surface: PrivacySurface::ForeignSearchPresentation,
            outcome: PublicOutcome::Sanitized(status),
        });

        let own_community = community(40);
        let record = CollaborationNotificationRecord::private(
            notification_candidate(path, own_community, own_community),
            CollaborativeNavigationTarget::channel("private-channel"),
        )
        .expect("private notification record");
        let before = cx.shown_system_notifications().len();
        let dispatch = cx.update(|cx| {
            dispatcher.dispatch(
                record,
                NotificationDevicePermissions {
                    native: NotificationPermission::Granted,
                    push: NotificationPermission::Disabled,
                },
                |_| false,
                cx,
            )
        });
        assert!(matches!(
            dispatch,
            CollaborationNotificationDispatch::Posted(_)
        ));
        let notifications = cx.shown_system_notifications();
        assert_eq!(notifications.len(), before + 1);
        let notification = notifications.last().expect("private notification");
        observations.push(PrivacyObservation {
            client: path,
            surface: PrivacySurface::PrivateNotification,
            outcome: PublicOutcome::PrivatePreview {
                title: notification.title.to_string(),
                body: notification.body.to_string(),
                rendered: format!("{notification:?}"),
            },
        });

        let foreign_record = CollaborationNotificationRecord::private(
            notification_candidate(path, own_community, community(41)),
            CollaborativeNavigationTarget::channel("foreign-channel"),
        )
        .expect("foreign notification record");
        let before = cx.shown_system_notifications().len();
        let dispatch = cx.update(|cx| {
            dispatcher.dispatch(
                foreign_record,
                NotificationDevicePermissions {
                    native: NotificationPermission::Granted,
                    push: NotificationPermission::Granted,
                },
                |_| false,
                cx,
            )
        });
        assert_eq!(
            dispatch,
            CollaborationNotificationDispatch::Suppressed(
                NotificationSuppression::InactiveMembership
            )
        );
        assert_eq!(cx.shown_system_notifications().len(), before);
        observations.push(PrivacyObservation {
            client: path,
            surface: PrivacySurface::ForeignNotification,
            outcome: PublicOutcome::Unavailable { operation_count: 0 },
        });
    }

    audit_privacy_trace(
        &observations,
        &[
            PrivacySurface::ForeignSearchPresentation,
            PrivacySurface::PrivateNotification,
            PrivacySurface::ForeignNotification,
        ],
    );
}

fn generation(value: u64) -> PushLeaseGeneration {
    PushLeaseGeneration::new(value).expect("push generation")
}

fn endpoint_generation(value: u64) -> PushEndpointGeneration {
    PushEndpointGeneration::new(value).expect("endpoint generation")
}

fn capability(value: u8) -> PushCapabilityReference {
    PushCapabilityReference::from_digest([value; 32]).expect("push capability")
}

fn wake(community_id: CommunityId, claim_id: Uuid, path: ClientPath) -> PushGatewayWake {
    PushGatewayWake::new(
        PushLeaseAddress {
            community_id,
            owner_principal_id: principal_id(50),
            installation_id: PushInstallationId::new(format!("{}-installation", path.label()))
                .expect("installation"),
        },
        Uuid::from_u128(60 + path as u128),
        Uuid::from_u128(70 + path as u128),
        generation(1),
        endpoint_generation(1),
        capability(1),
        100_000,
        1,
        claim_id,
        31_000,
    )
    .expect("gateway wake")
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[derive(Clone)]
struct FakePushStore {
    wake: PushGatewayWake,
}

#[async_trait]
impl PushWakeStore for FakePushStore {
    async fn claim(
        &self,
        _tenant: &TenantContext,
        _claim: PushGatewayClaim,
    ) -> Result<Vec<PushGatewayWake>, PushWakeStoreError> {
        Ok(vec![self.wake.clone()])
    }

    async fn revalidate(
        &self,
        _tenant: &TenantContext,
        _wake: &PushGatewayWake,
        _now_millis: u64,
    ) -> Result<bool, PushWakeStoreError> {
        Ok(true)
    }

    async fn retry(
        &self,
        _tenant: &TenantContext,
        _wake: &PushGatewayWake,
        _available_at_millis: u64,
        _now_millis: u64,
    ) -> Result<(), PushWakeStoreError> {
        Err(PushWakeStoreError::Unavailable)
    }

    async fn disable_endpoint(
        &self,
        _tenant: &TenantContext,
        _wake: &PushGatewayWake,
        _disabled_at_millis: u64,
        _now_millis: u64,
    ) -> Result<bool, PushWakeStoreError> {
        Ok(false)
    }

    async fn complete(
        &self,
        _tenant: &TenantContext,
        _wake: &PushGatewayWake,
        _outcome: PushWakeTerminalOutcome,
        _completed_at_millis: u64,
    ) -> Result<(), PushWakeStoreError> {
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct AuthorizedPush;

#[async_trait]
impl PushWakeAuthorization for AuthorizedPush {
    async fn authorize(
        &self,
        _tenant: &TenantContext,
        _wake: &PushGatewayWake,
    ) -> Result<PushAuthorizationDecision, PushAuthorizationError> {
        Ok(PushAuthorizationDecision::Authorized)
    }
}

#[derive(Clone)]
struct CapturingProvider {
    requests: Arc<Mutex<Vec<(String, String)>>>,
}

#[async_trait]
impl PushProvider for CapturingProvider {
    async fn deliver(
        &self,
        request: PushDeliveryRequest,
    ) -> Result<PushProviderOutcome, PushProviderError> {
        let payload = serde_json::to_string(&request.payload()).map_err(|_| PushProviderError)?;
        lock(&self.requests).push((payload, format!("{request:?}")));
        Ok(PushProviderOutcome::Accepted)
    }
}

#[derive(Clone, Copy)]
struct FixedClock;

impl PushGatewayClock for FixedClock {
    fn now_millis(&self) -> Result<u64, PushGatewayExecutorError> {
        Ok(1_000)
    }
}

#[tokio::test]
async fn mixed_version_push_is_wake_only_and_rejects_foreign_claims_before_provider() {
    let own_community = community(80);
    let own_tenant = tenant(own_community);
    let mut observations = Vec::new();

    for path in ClientPath::ALL {
        let claim_id = Uuid::from_u128(90 + path as u128);
        let requests = Arc::new(Mutex::new(Vec::new()));
        let executor = PushGatewayExecutor::with_clock(
            FakePushStore {
                wake: wake(own_community, claim_id, path),
            },
            AuthorizedPush,
            CapturingProvider {
                requests: requests.clone(),
            },
            FixedClock,
        );
        assert_eq!(
            executor
                .run_once(&own_tenant, claim_id)
                .await
                .expect("own push execution"),
            PushExecutionSummary {
                claimed: 1,
                delivered: 1,
                ..PushExecutionSummary::default()
            }
        );
        let request = lock(&requests)
            .first()
            .cloned()
            .expect("captured provider request");
        observations.push(PrivacyObservation {
            client: path,
            surface: PrivacySurface::WakeDelivery,
            outcome: PublicOutcome::Wake {
                payload: request.0,
                rendered: request.1,
            },
        });

        let foreign_requests = Arc::new(Mutex::new(Vec::new()));
        let foreign_executor = PushGatewayExecutor::with_clock(
            FakePushStore {
                wake: wake(community(81), claim_id, path),
            },
            AuthorizedPush,
            CapturingProvider {
                requests: foreign_requests.clone(),
            },
            FixedClock,
        );
        assert!(matches!(
            foreign_executor.run_once(&own_tenant, claim_id).await,
            Err(PushGatewayExecutorError::InvalidWork)
        ));
        observations.push(PrivacyObservation {
            client: path,
            surface: PrivacySurface::ForeignWake,
            outcome: PublicOutcome::Unavailable {
                operation_count: lock(&foreign_requests).len(),
            },
        });
    }

    audit_privacy_trace(
        &observations,
        &[PrivacySurface::WakeDelivery, PrivacySurface::ForeignWake],
    );
}
