use std::{error::Error, fmt};

use crate::CommunityId;

const MAX_ROUTE_REFERENCE_BYTES: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TrustedTenantRouteSource {
    DirectHost,
    TrustedForwardedHost,
    Listener,
    Deployment,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct TrustedTenantRoute {
    community_id: CommunityId,
    source: TrustedTenantRouteSource,
    reference: String,
}

impl TrustedTenantRoute {
    pub fn from_direct_host(
        community_id: CommunityId,
        canonical_host: impl Into<String>,
    ) -> Result<Self, TenantRouteError> {
        Self::new(
            community_id,
            TrustedTenantRouteSource::DirectHost,
            canonical_host,
        )
    }

    pub fn from_trusted_forwarded_host(
        community_id: CommunityId,
        canonical_host: impl Into<String>,
    ) -> Result<Self, TenantRouteError> {
        Self::new(
            community_id,
            TrustedTenantRouteSource::TrustedForwardedHost,
            canonical_host,
        )
    }

    pub fn from_listener(
        community_id: CommunityId,
        listener_identifier: impl Into<String>,
    ) -> Result<Self, TenantRouteError> {
        Self::new(
            community_id,
            TrustedTenantRouteSource::Listener,
            listener_identifier,
        )
    }

    pub fn from_deployment(
        community_id: CommunityId,
        deployment_route: impl Into<String>,
    ) -> Result<Self, TenantRouteError> {
        Self::new(
            community_id,
            TrustedTenantRouteSource::Deployment,
            deployment_route,
        )
    }

    fn new(
        community_id: CommunityId,
        source: TrustedTenantRouteSource,
        reference: impl Into<String>,
    ) -> Result<Self, TenantRouteError> {
        let reference = reference.into();
        if reference.is_empty()
            || reference.len() > MAX_ROUTE_REFERENCE_BYTES
            || reference.trim() != reference
            || reference.chars().any(char::is_control)
        {
            return Err(TenantRouteError::InvalidReference);
        }
        Ok(Self {
            community_id,
            source,
            reference,
        })
    }

    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn source(&self) -> TrustedTenantRouteSource {
        self.source
    }

    pub fn reference(&self) -> &str {
        &self.reference
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum UntrustedTenantClaimSource {
    ChannelMapping,
    TokenStamp,
    SignedUrl,
    EventTag,
    BodyField,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct UntrustedTenantClaim {
    community_id: CommunityId,
    source: UntrustedTenantClaimSource,
}

impl UntrustedTenantClaim {
    pub const fn new(community_id: CommunityId, source: UntrustedTenantClaimSource) -> Self {
        Self {
            community_id,
            source,
        }
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn source(self) -> UntrustedTenantClaimSource {
        self.source
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TenantContext {
    community_id: CommunityId,
    route_source: TrustedTenantRouteSource,
    route_reference: String,
}

impl TenantContext {
    pub fn establish(
        trusted_route: Option<TrustedTenantRoute>,
        untrusted_claims: &[UntrustedTenantClaim],
    ) -> Result<Self, TenantContextError> {
        let trusted_route = trusted_route.ok_or(TenantContextError::MissingTrustedRoute)?;
        if untrusted_claims
            .iter()
            .any(|claim| claim.community_id != trusted_route.community_id)
        {
            return Err(TenantContextError::ConflictingTenantClaim);
        }
        Ok(Self {
            community_id: trusted_route.community_id,
            route_source: trusted_route.source,
            route_reference: trusted_route.reference,
        })
    }

    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn route_source(&self) -> TrustedTenantRouteSource {
        self.route_source
    }

    pub fn route_reference(&self) -> &str {
        &self.route_reference
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TenantRouteError {
    InvalidReference,
}

impl fmt::Display for TenantRouteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("trusted tenant route is invalid")
    }
}

impl Error for TenantRouteError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TenantContextError {
    MissingTrustedRoute,
    ConflictingTenantClaim,
}

impl fmt::Display for TenantContextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("tenant context could not be established")
    }
}

impl Error for TenantContextError {}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    #[test]
    fn tenant_context_accepts_each_trusted_route_class() {
        let community_id = community(1);
        let routes = [
            TrustedTenantRoute::from_direct_host(community_id, "relay.example")
                .expect("direct host"),
            TrustedTenantRoute::from_trusted_forwarded_host(community_id, "relay.example")
                .expect("trusted forwarded host"),
            TrustedTenantRoute::from_listener(community_id, "nostr-primary").expect("listener"),
            TrustedTenantRoute::from_deployment(community_id, "eu-west/collab")
                .expect("deployment"),
        ];

        for route in routes {
            let expected_source = route.source();
            let expected_reference = route.reference().to_owned();
            let context = TenantContext::establish(Some(route), &[]).expect("trusted route");

            assert_eq!(context.community_id(), community_id);
            assert_eq!(context.route_source(), expected_source);
            assert_eq!(context.route_reference(), expected_reference);
        }
    }

    #[test]
    fn tenant_context_rejects_an_absent_trusted_route() {
        assert_eq!(
            TenantContext::establish(None, &[]),
            Err(TenantContextError::MissingTrustedRoute)
        );
    }

    #[test]
    fn tenant_context_rejects_an_event_tag_as_the_only_tenant_source() {
        let claim = UntrustedTenantClaim::new(community(1), UntrustedTenantClaimSource::EventTag);

        assert_eq!(
            TenantContext::establish(None, &[claim]),
            Err(TenantContextError::MissingTrustedRoute)
        );
    }

    #[test]
    fn tenant_context_allows_matching_claims_without_changing_the_route() {
        let community_id = community(1);
        let route =
            TrustedTenantRoute::from_listener(community_id, "rpc-primary").expect("listener route");
        let claims = [
            UntrustedTenantClaim::new(community_id, UntrustedTenantClaimSource::ChannelMapping),
            UntrustedTenantClaim::new(community_id, UntrustedTenantClaimSource::EventTag),
        ];

        let context = TenantContext::establish(Some(route), &claims).expect("matching claims");

        assert_eq!(context.community_id(), community_id);
        assert_eq!(context.route_source(), TrustedTenantRouteSource::Listener);
        assert_eq!(context.route_reference(), "rpc-primary");
    }

    #[test]
    fn tenant_context_rejects_a_conflicting_payload_claim_generically() {
        let route = TrustedTenantRoute::from_direct_host(community(1), "relay.example")
            .expect("direct host");
        let event_claim =
            UntrustedTenantClaim::new(community(2), UntrustedTenantClaimSource::EventTag);
        let body_claim =
            UntrustedTenantClaim::new(community(3), UntrustedTenantClaimSource::BodyField);

        assert_eq!(
            TenantContext::establish(Some(route.clone()), &[event_claim]),
            Err(TenantContextError::ConflictingTenantClaim)
        );
        assert_eq!(
            TenantContext::establish(Some(route), &[body_claim]),
            Err(TenantContextError::ConflictingTenantClaim)
        );
        assert_eq!(
            TenantContextError::MissingTrustedRoute.to_string(),
            TenantContextError::ConflictingTenantClaim.to_string()
        );
    }

    #[test]
    fn tenant_context_rejects_noncanonical_route_references() {
        let community_id = community(1);

        assert_eq!(
            TrustedTenantRoute::from_direct_host(community_id, ""),
            Err(TenantRouteError::InvalidReference)
        );
        assert_eq!(
            TrustedTenantRoute::from_listener(community_id, " listener "),
            Err(TenantRouteError::InvalidReference)
        );
        assert_eq!(
            TrustedTenantRoute::from_deployment(community_id, "x\nroute"),
            Err(TenantRouteError::InvalidReference)
        );
        assert_eq!(
            TrustedTenantRoute::from_deployment(
                community_id,
                "x".repeat(MAX_ROUTE_REFERENCE_BYTES + 1),
            ),
            Err(TenantRouteError::InvalidReference)
        );
    }
}
