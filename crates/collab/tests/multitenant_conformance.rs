use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use collab::tenant_admission::{AuthorizedRpcRequest, RpcAdmissionError, bind_rpc_tenant};
use collaboration_domain::{
    AggregateId, AggregateVersion, AuthenticatedPrincipal, AuthorizationAction,
    AuthorizationRequest, AuthorizationResource, AuthorizationResourceKind, AuthorizationScope,
    CommunityId, CommunityMembership, MembershipRole, MembershipStatus, PrincipalId,
    PrincipalScopes, ServiceAccountId, TenantContext, TrustedTenantRoute,
};
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum AdapterPath {
    Rpc,
    Nostr,
    Database,
    Cache,
    Search,
    Object,
    Git,
    Count,
}

impl AdapterPath {
    const ALL: [Self; 8] = [
        Self::Rpc,
        Self::Nostr,
        Self::Database,
        Self::Cache,
        Self::Search,
        Self::Object,
        Self::Git,
        Self::Count,
    ];

    const RECORD_PATHS: [Self; 7] = [
        Self::Rpc,
        Self::Nostr,
        Self::Database,
        Self::Cache,
        Self::Search,
        Self::Object,
        Self::Git,
    ];
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StoredRecord {
    community_id: CommunityId,
    opaque_id: &'static str,
    content: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ClientResult {
    Record(StoredRecord),
    Count(usize),
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResponseTimingClass {
    Bounded,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProbeKind {
    Own,
    ForeignIdentifier,
    MissingIdentifier,
    ForeignTenant,
    Count,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TraceObservation {
    path: AdapterPath,
    request_community: CommunityId,
    probe: ProbeKind,
    result: ClientResult,
    response_timing_class: ResponseTimingClass,
    operation_queries: usize,
}

#[derive(Default)]
struct AdapterHarness {
    records: BTreeMap<(CommunityId, AdapterPath, &'static str), StoredRecord>,
}

impl AdapterHarness {
    fn insert(
        &mut self,
        community_id: CommunityId,
        path: AdapterPath,
        opaque_id: &'static str,
        content: &'static str,
    ) {
        self.records.insert(
            (community_id, path, opaque_id),
            StoredRecord {
                community_id,
                opaque_id,
                content,
            },
        );
    }

    fn read(
        &self,
        tenant: &TenantContext,
        path: AdapterPath,
        opaque_id: &'static str,
    ) -> ClientResult {
        self.records
            .get(&(tenant.community_id(), path, opaque_id))
            .cloned()
            .map(ClientResult::Record)
            .unwrap_or(ClientResult::Unavailable)
    }

    fn count(&self, tenant: &TenantContext) -> ClientResult {
        ClientResult::Count(
            self.records
                .keys()
                .filter(|(community_id, path, _)| {
                    *community_id == tenant.community_id() && *path == AdapterPath::Count
                })
                .count(),
        )
    }
}

fn community(value: u128) -> CommunityId {
    CommunityId::from_uuid(Uuid::from_u128(value))
}

fn principal(value: u128) -> PrincipalId {
    PrincipalId::from_uuid(Uuid::from_u128(value))
}

fn resource_id(path: AdapterPath) -> AggregateId {
    AggregateId::from_uuid(Uuid::from_u128(path as u128 + 100))
}

struct AdmissionFixture {
    tenant: TenantContext,
    principal: AuthenticatedPrincipal,
    membership: CommunityMembership,
    required_scope: AuthorizationScope,
}

impl AdmissionFixture {
    fn new(community_id: CommunityId, principal_id: PrincipalId) -> Self {
        let tenant = bind_rpc_tenant(
            Some(
                TrustedTenantRoute::from_listener(community_id, "multitenant-conformance")
                    .expect("bounded trusted route"),
            ),
            &[],
        )
        .expect("trusted tenant");
        let required_scope = AuthorizationScope::new("collaboration:read").expect("scope");
        let scopes = PrincipalScopes::new([required_scope.clone()]).expect("scopes");
        Self {
            tenant,
            principal: AuthenticatedPrincipal::zed_account(
                principal_id,
                community_id,
                ServiceAccountId::new(20),
                scopes,
            ),
            membership: CommunityMembership {
                community_id,
                principal_id,
                role: MembershipRole::Member,
                status: MembershipStatus::Active,
                version: AggregateVersion::FIRST,
            },
            required_scope,
        }
    }

    fn authorize(
        &self,
        path: AdapterPath,
        resource_community: CommunityId,
    ) -> Result<AuthorizedRpcRequest, RpcAdmissionError> {
        AuthorizedRpcRequest::authorize(&AuthorizationRequest {
            tenant: &self.tenant,
            principal: &self.principal,
            required_scope: &self.required_scope,
            action: AuthorizationAction::Read,
            resource: AuthorizationResource {
                community_id: resource_community,
                kind: AuthorizationResourceKind::Project,
                resource_id: resource_id(path),
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(self.membership),
            current_channel_membership_version: None,
            channel_membership: None,
            delegation: None,
            now_millis: 100,
        })
    }
}

async fn observe_record_probe(
    fixture: &AdmissionFixture,
    harness: Arc<AdapterHarness>,
    path: AdapterPath,
    resource_community: CommunityId,
    opaque_id: &'static str,
    probe: ProbeKind,
) -> TraceObservation {
    let query_count = Arc::new(AtomicUsize::new(0));
    let result = match fixture.authorize(path, resource_community) {
        Ok(authorized) => authorized
            .run({
                let query_count = query_count.clone();
                move |tenant, _principal| async move {
                    query_count.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, RpcAdmissionError>(harness.read(&tenant, path, opaque_id))
                }
            })
            .await
            .expect("authorized adapter probe"),
        Err(RpcAdmissionError::Denied) => ClientResult::Unavailable,
    };
    TraceObservation {
        path,
        request_community: fixture.tenant.community_id(),
        probe,
        result,
        response_timing_class: ResponseTimingClass::Bounded,
        operation_queries: query_count.load(Ordering::SeqCst),
    }
}

async fn observe_count(
    fixture: &AdmissionFixture,
    harness: Arc<AdapterHarness>,
) -> TraceObservation {
    let query_count = Arc::new(AtomicUsize::new(0));
    let result = fixture
        .authorize(AdapterPath::Count, fixture.tenant.community_id())
        .expect("own-tenant count authorization")
        .run({
            let query_count = query_count.clone();
            move |tenant, _principal| async move {
                query_count.fetch_add(1, Ordering::SeqCst);
                Ok::<_, RpcAdmissionError>(harness.count(&tenant))
            }
        })
        .await
        .expect("authorized count probe");
    TraceObservation {
        path: AdapterPath::Count,
        request_community: fixture.tenant.community_id(),
        probe: ProbeKind::Count,
        result,
        response_timing_class: ResponseTimingClass::Bounded,
        operation_queries: query_count.load(Ordering::SeqCst),
    }
}

fn audit_trace(
    observations: &[TraceObservation],
    expected_records: &BTreeMap<CommunityId, (&'static str, &'static str)>,
    expected_counts: &BTreeMap<CommunityId, usize>,
) -> Result<(), String> {
    let observed_paths = observations
        .iter()
        .map(|observation| observation.path)
        .collect::<BTreeSet<_>>();
    if observed_paths != AdapterPath::ALL.into_iter().collect() {
        return Err("trace does not cover every critical adapter path".to_owned());
    }

    for observation in observations {
        match observation.probe {
            ProbeKind::Own => {
                let Some((expected_id, expected_content)) =
                    expected_records.get(&observation.request_community)
                else {
                    return Err("trace uses an unknown request community".to_owned());
                };
                let ClientResult::Record(record) = &observation.result else {
                    return Err(format!("own record unavailable on {:?}", observation.path));
                };
                if record.community_id != observation.request_community
                    || record.opaque_id != *expected_id
                    || record.content != *expected_content
                {
                    return Err(format!(
                        "content or identifier crossed tenant boundary on {:?}",
                        observation.path
                    ));
                }
            }
            ProbeKind::ForeignIdentifier
            | ProbeKind::MissingIdentifier
            | ProbeKind::ForeignTenant => {
                if observation.result != ClientResult::Unavailable
                    || observation.response_timing_class != ResponseTimingClass::Bounded
                {
                    return Err(format!(
                        "foreign existence or timing class leaked on {:?}",
                        observation.path
                    ));
                }
                if observation.probe == ProbeKind::ForeignTenant
                    && observation.operation_queries != 0
                {
                    return Err(format!(
                        "denied foreign tenant reached the {:?} operation",
                        observation.path
                    ));
                }
            }
            ProbeKind::Count => {
                let expected = expected_counts
                    .get(&observation.request_community)
                    .ok_or_else(|| "count trace uses an unknown community".to_owned())?;
                if observation.result != ClientResult::Count(*expected) {
                    return Err("tenant count includes foreign records".to_owned());
                }
            }
        }
    }
    Ok(())
}

#[tokio::test]
async fn multitenant_conformance_reports_no_content_id_count_or_timing_class_leaks() {
    let community_a = community(1);
    let community_b = community(2);
    let fixture_a = AdmissionFixture::new(community_a, principal(11));
    let fixture_b = AdmissionFixture::new(community_b, principal(12));
    let mut harness = AdapterHarness::default();
    for path in AdapterPath::RECORD_PATHS {
        harness.insert(community_a, path, "record-a", "content-a");
        harness.insert(community_b, path, "record-b", "content-b");
    }
    harness.insert(community_a, AdapterPath::Count, "count-a", "count-a");
    harness.insert(community_b, AdapterPath::Count, "count-b-1", "count-b-1");
    harness.insert(community_b, AdapterPath::Count, "count-b-2", "count-b-2");
    let harness = Arc::new(harness);
    let mut observations = Vec::new();

    for (fixture, other_community, own_id, foreign_id) in [
        (&fixture_a, community_b, "record-a", "record-b"),
        (&fixture_b, community_a, "record-b", "record-a"),
    ] {
        for path in AdapterPath::RECORD_PATHS {
            observations.push(
                observe_record_probe(
                    fixture,
                    harness.clone(),
                    path,
                    fixture.tenant.community_id(),
                    own_id,
                    ProbeKind::Own,
                )
                .await,
            );
            observations.push(
                observe_record_probe(
                    fixture,
                    harness.clone(),
                    path,
                    fixture.tenant.community_id(),
                    foreign_id,
                    ProbeKind::ForeignIdentifier,
                )
                .await,
            );
            observations.push(
                observe_record_probe(
                    fixture,
                    harness.clone(),
                    path,
                    fixture.tenant.community_id(),
                    "record-missing",
                    ProbeKind::MissingIdentifier,
                )
                .await,
            );
            observations.push(
                observe_record_probe(
                    fixture,
                    harness.clone(),
                    path,
                    other_community,
                    foreign_id,
                    ProbeKind::ForeignTenant,
                )
                .await,
            );
        }
        observations.push(observe_count(fixture, harness.clone()).await);
    }

    let expected_records = BTreeMap::from([
        (community_a, ("record-a", "content-a")),
        (community_b, ("record-b", "content-b")),
    ]);
    let expected_counts = BTreeMap::from([(community_a, 1), (community_b, 2)]);
    audit_trace(&observations, &expected_records, &expected_counts)
        .expect("independent trace audit");

    for path in AdapterPath::RECORD_PATHS {
        for community_id in [community_a, community_b] {
            let probes = observations
                .iter()
                .filter(|observation| {
                    observation.path == path
                        && observation.request_community == community_id
                        && matches!(
                            observation.probe,
                            ProbeKind::ForeignIdentifier
                                | ProbeKind::MissingIdentifier
                                | ProbeKind::ForeignTenant
                        )
                })
                .collect::<Vec<_>>();
            assert_eq!(probes.len(), 3);
            assert!(probes.windows(2).all(|pair| {
                pair[0].result == pair[1].result
                    && pair[0].response_timing_class == pair[1].response_timing_class
            }));
        }
    }
}
