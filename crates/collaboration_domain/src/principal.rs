use std::{collections::BTreeSet, error::Error, fmt};

use serde::{Deserialize, Deserializer, Serialize, de};
use uuid::Uuid;

use crate::{
    AccountBinding, BindingId, CommunityId, IdentityProfile, NostrEventId, NostrPublicKey,
    PrincipalId, ProfileId, ProfileKind, ServiceAccountId,
};

const MAX_SCOPE_BYTES: usize = 128;
const MAX_PRINCIPAL_SCOPES: usize = 128;
const MAX_SERVICE_NAME_BYTES: usize = 256;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct AuthorizationScope(String);

impl AuthorizationScope {
    pub fn new(value: impl Into<String>) -> Result<Self, PrincipalError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_SCOPE_BYTES
            || value.trim() != value
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b":._-/*".contains(&byte))
        {
            return Err(PrincipalError::InvalidScope);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for AuthorizationScope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct PrincipalScopes(BTreeSet<AuthorizationScope>);

impl PrincipalScopes {
    pub fn for_nip42_authenticated_key() -> Result<Self, PrincipalError> {
        Self::new(
            [
                "messages:read",
                "messages:write",
                "channels:read",
                "channels:write",
                "admin:channels",
                "users:read",
                "users:write",
                "admin:users",
                "jobs:read",
                "jobs:write",
                "subscriptions:read",
                "subscriptions:write",
                "files:read",
                "files:write",
                "repos:read",
                "repos:write",
            ]
            .into_iter()
            .map(AuthorizationScope::new)
            .collect::<Result<Vec<_>, _>>()?,
        )
    }

    pub fn new(
        scopes: impl IntoIterator<Item = AuthorizationScope>,
    ) -> Result<Self, PrincipalError> {
        let mut normalized = BTreeSet::new();
        for (index, scope) in scopes.into_iter().enumerate() {
            if index >= MAX_PRINCIPAL_SCOPES {
                return Err(PrincipalError::TooManyScopes);
            }
            normalized.insert(scope);
        }
        Ok(Self(normalized))
    }

    pub fn contains(&self, scope: &AuthorizationScope) -> bool {
        self.0.contains(scope)
    }

    pub fn iter(&self) -> impl Iterator<Item = &AuthorizationScope> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<'de> Deserialize<'de> for PrincipalScopes {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let scopes = Vec::<AuthorizationScope>::deserialize(deserializer)?;
        Self::new(scopes).map_err(de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TokenId(Uuid);

impl TokenId {
    pub const fn from_uuid(value: Uuid) -> Self {
        Self(value)
    }

    pub const fn as_uuid(self) -> Uuid {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NostrAuthenticationMethod {
    Nip42,
    Nip98,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActiveBindingIdentity {
    binding_id: BindingId,
    service_account_id: ServiceAccountId,
    profile_id: ProfileId,
    public_key: NostrPublicKey,
}

impl ActiveBindingIdentity {
    fn from_binding(
        community_id: CommunityId,
        binding: &AccountBinding,
    ) -> Result<Self, PrincipalError> {
        if binding.community_id() != community_id {
            return Err(PrincipalError::TenantMismatch);
        }
        if !binding.can_sign() {
            return Err(PrincipalError::BindingNotActive);
        }
        Ok(Self {
            binding_id: binding.binding_id(),
            service_account_id: binding.fields().service_account_id,
            profile_id: binding.fields().profile_id,
            public_key: binding.public_key(),
        })
    }

    pub const fn binding_id(self) -> BindingId {
        self.binding_id
    }

    pub const fn service_account_id(self) -> ServiceAccountId {
        self.service_account_id
    }

    pub const fn profile_id(self) -> ProfileId {
        self.profile_id
    }

    pub const fn public_key(self) -> NostrPublicKey {
        self.public_key
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AuthenticatedPrincipalKind {
    SimAccount {
        service_account_id: ServiceAccountId,
    },
    NostrIdentity {
        public_key: NostrPublicKey,
        authentication_method: NostrAuthenticationMethod,
        active_binding: Option<ActiveBindingIdentity>,
    },
    OwnerAttestedAgent {
        profile_id: ProfileId,
        agent_public_key: NostrPublicKey,
        owner_public_key: NostrPublicKey,
        proof_event_id: NostrEventId,
        authentication_method: NostrAuthenticationMethod,
    },
    ScopedToken {
        token_id: TokenId,
        subject_principal_id: PrincipalId,
    },
    Service {
        service_name: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthenticatedPrincipal {
    principal_id: PrincipalId,
    community_id: CommunityId,
    kind: AuthenticatedPrincipalKind,
    scopes: PrincipalScopes,
}

impl AuthenticatedPrincipal {
    pub fn zed_account(
        principal_id: PrincipalId,
        community_id: CommunityId,
        service_account_id: ServiceAccountId,
        scopes: PrincipalScopes,
    ) -> Self {
        Self {
            principal_id,
            community_id,
            kind: AuthenticatedPrincipalKind::SimAccount { service_account_id },
            scopes,
        }
    }

    pub fn nostr_identity(
        principal_id: PrincipalId,
        community_id: CommunityId,
        public_key: NostrPublicKey,
        authentication_method: NostrAuthenticationMethod,
        scopes: PrincipalScopes,
    ) -> Self {
        Self {
            principal_id,
            community_id,
            kind: AuthenticatedPrincipalKind::NostrIdentity {
                public_key,
                authentication_method,
                active_binding: None,
            },
            scopes,
        }
    }

    pub fn bound_nostr_identity(
        principal_id: PrincipalId,
        community_id: CommunityId,
        binding: &AccountBinding,
        authentication_method: NostrAuthenticationMethod,
        scopes: PrincipalScopes,
    ) -> Result<Self, PrincipalError> {
        let active_binding = ActiveBindingIdentity::from_binding(community_id, binding)?;
        Ok(Self {
            principal_id,
            community_id,
            kind: AuthenticatedPrincipalKind::NostrIdentity {
                public_key: active_binding.public_key,
                authentication_method,
                active_binding: Some(active_binding),
            },
            scopes,
        })
    }

    pub fn owner_attested_agent(
        principal_id: PrincipalId,
        community_id: CommunityId,
        profile: &IdentityProfile,
        authentication_method: NostrAuthenticationMethod,
        scopes: PrincipalScopes,
    ) -> Result<Self, PrincipalError> {
        if profile.community_id() != community_id {
            return Err(PrincipalError::TenantMismatch);
        }
        let ProfileKind::Agent(agent) = profile.kind() else {
            return Err(PrincipalError::OwnerAttestationRequired);
        };
        let (Some(owner_public_key), Some(attestation)) =
            (agent.claimed_owner, agent.owner_attestation.as_ref())
        else {
            return Err(PrincipalError::OwnerAttestationRequired);
        };
        if owner_public_key == profile.author_public_key()
            || attestation.owner_public_key != owner_public_key
            || attestation.agent_public_key != profile.author_public_key()
        {
            return Err(PrincipalError::InvalidOwnerAttestation);
        }
        Ok(Self {
            principal_id,
            community_id,
            kind: AuthenticatedPrincipalKind::OwnerAttestedAgent {
                profile_id: profile.profile_id(),
                agent_public_key: profile.author_public_key(),
                owner_public_key,
                proof_event_id: attestation.proof_event_id,
                authentication_method,
            },
            scopes,
        })
    }

    pub fn scoped_token(
        principal_id: PrincipalId,
        community_id: CommunityId,
        token_id: TokenId,
        subject_principal_id: PrincipalId,
        scopes: PrincipalScopes,
    ) -> Self {
        Self {
            principal_id,
            community_id,
            kind: AuthenticatedPrincipalKind::ScopedToken {
                token_id,
                subject_principal_id,
            },
            scopes,
        }
    }

    pub fn service(
        principal_id: PrincipalId,
        community_id: CommunityId,
        service_name: impl Into<String>,
        scopes: PrincipalScopes,
    ) -> Result<Self, PrincipalError> {
        let service_name = service_name.into();
        if service_name.is_empty()
            || service_name.len() > MAX_SERVICE_NAME_BYTES
            || service_name.trim() != service_name
            || service_name.chars().any(char::is_control)
        {
            return Err(PrincipalError::InvalidServiceName);
        }
        Ok(Self {
            principal_id,
            community_id,
            kind: AuthenticatedPrincipalKind::Service { service_name },
            scopes,
        })
    }

    pub const fn principal_id(&self) -> PrincipalId {
        self.principal_id
    }

    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn kind(&self) -> &AuthenticatedPrincipalKind {
        &self.kind
    }

    pub const fn scopes(&self) -> &PrincipalScopes {
        &self.scopes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrincipalError {
    InvalidScope,
    TooManyScopes,
    TenantMismatch,
    BindingNotActive,
    OwnerAttestationRequired,
    InvalidOwnerAttestation,
    InvalidServiceName,
}

impl fmt::Display for PrincipalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScope => formatter.write_str("principal scope is invalid"),
            Self::TooManyScopes => formatter.write_str("principal has too many scopes"),
            Self::TenantMismatch => formatter.write_str("principal tenant does not match"),
            Self::BindingNotActive => {
                formatter.write_str("principal binding is not active and verified")
            }
            Self::OwnerAttestationRequired => {
                formatter.write_str("agent principal requires owner attestation")
            }
            Self::InvalidOwnerAttestation => {
                formatter.write_str("agent owner attestation is invalid")
            }
            Self::InvalidServiceName => formatter.write_str("service principal name is invalid"),
        }
    }
}

impl Error for PrincipalError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AccountBindingFields, AgentProfile, AggregateVersion, BindingStatus, BindingVerification,
        BindingVerificationMethod, EvidenceReference, OperationId, OrganizationPolicyVersion,
        OwnerAttestationEvidence, ProfileRecordFields,
    };

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn principal(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn key(value: u8) -> NostrPublicKey {
        NostrPublicKey::from_bytes([value; 32])
    }

    fn scopes(values: &[&str]) -> PrincipalScopes {
        PrincipalScopes::new(
            values
                .iter()
                .map(|value| AuthorizationScope::new(*value).expect("valid scope")),
        )
        .expect("valid scope set")
    }

    fn binding(community_id: CommunityId, status: BindingStatus) -> AccountBinding {
        AccountBinding::new(AccountBindingFields {
            binding_id: BindingId::from_uuid(Uuid::from_u128(10)),
            community_id,
            service_account_id: ServiceAccountId::new(20),
            profile_id: ProfileId::from_uuid(Uuid::from_u128(30)),
            public_key: key(4),
            status,
            verification: Some(BindingVerification {
                method: BindingVerificationMethod::ExistingKeyChallenge,
                evidence_reference: EvidenceReference::new("evidence:principal-test")
                    .expect("evidence"),
                verified_at_millis: 20,
            }),
            predecessor: None,
            successor: None,
            created_at_millis: 10,
            activated_at_millis: (status == BindingStatus::Active).then_some(30),
            terminal_at_millis: (status == BindingStatus::Revoked).then_some(40),
            organization_policy_version: OrganizationPolicyVersion::FIRST,
            actor_principal_id: principal(40),
            version: AggregateVersion::FIRST,
            audit_reference: OperationId::from_uuid(Uuid::from_u128(50)),
        })
        .expect("valid binding")
    }

    fn agent_profile(community_id: CommunityId, attested: bool) -> IdentityProfile {
        let agent_public_key = key(6);
        let owner_public_key = key(7);
        IdentityProfile::new(ProfileRecordFields {
            profile_id: ProfileId::from_uuid(Uuid::from_u128(60)),
            community_id,
            author_public_key: agent_public_key,
            kind: ProfileKind::Agent(AgentProfile {
                claimed_owner: attested.then_some(owner_public_key),
                owner_attestation: attested.then_some(OwnerAttestationEvidence {
                    owner_public_key,
                    agent_public_key,
                    proof_event_id: NostrEventId::from_bytes([8; 32]),
                    exact_conditions: "kind=1".to_owned(),
                    verified_at: 100,
                }),
            }),
            metadata: None,
            statuses: Vec::new(),
            social_lists: Vec::new(),
            relay_archive_states: Vec::new(),
            version: AggregateVersion::FIRST,
        })
        .expect("valid agent profile")
    }

    #[test]
    fn authenticated_principal_rejects_unverified_and_cross_tenant_bindings() {
        let community_id = community(1);
        let pending = binding(community_id, BindingStatus::Verified);
        let active_elsewhere = binding(community(2), BindingStatus::Active);

        assert_eq!(
            AuthenticatedPrincipal::bound_nostr_identity(
                principal(1),
                community_id,
                &pending,
                NostrAuthenticationMethod::Nip42,
                PrincipalScopes::default(),
            ),
            Err(PrincipalError::BindingNotActive)
        );
        assert_eq!(
            AuthenticatedPrincipal::bound_nostr_identity(
                principal(1),
                community_id,
                &active_elsewhere,
                NostrAuthenticationMethod::Nip42,
                PrincipalScopes::default(),
            ),
            Err(PrincipalError::TenantMismatch)
        );
    }

    #[test]
    fn authenticated_principal_keeps_zed_accounts_and_direct_nostr_keys_distinct() {
        let community_id = community(1);
        let zed_account = AuthenticatedPrincipal::zed_account(
            principal(1),
            community_id,
            ServiceAccountId::new(20),
            scopes(&["users:read"]),
        );
        let nostr_identity = AuthenticatedPrincipal::nostr_identity(
            principal(2),
            community_id,
            key(4),
            NostrAuthenticationMethod::Nip42,
            scopes(&["messages:read"]),
        );

        assert!(matches!(
            zed_account.kind(),
            AuthenticatedPrincipalKind::SimAccount { service_account_id }
                if *service_account_id == ServiceAccountId::new(20)
        ));
        assert!(matches!(
            nostr_identity.kind(),
            AuthenticatedPrincipalKind::NostrIdentity {
                public_key,
                authentication_method: NostrAuthenticationMethod::Nip42,
                active_binding: None,
            } if *public_key == key(4)
        ));
        assert_ne!(zed_account.principal_id(), nostr_identity.principal_id());
    }

    #[test]
    fn authenticated_principal_retains_active_binding_identity() {
        let community_id = community(1);
        let binding = binding(community_id, BindingStatus::Active);

        let authenticated = AuthenticatedPrincipal::bound_nostr_identity(
            principal(1),
            community_id,
            &binding,
            NostrAuthenticationMethod::Nip98,
            scopes(&["messages:read"]),
        )
        .expect("active binding");

        let AuthenticatedPrincipalKind::NostrIdentity {
            public_key,
            authentication_method,
            active_binding: Some(active_binding),
        } = authenticated.kind()
        else {
            panic!("expected bound Nostr identity");
        };
        assert_eq!(*public_key, binding.public_key());
        assert_eq!(*authentication_method, NostrAuthenticationMethod::Nip98);
        assert_eq!(active_binding.binding_id(), binding.binding_id());
        assert_eq!(
            active_binding.service_account_id(),
            binding.fields().service_account_id
        );
    }

    #[test]
    fn authenticated_principal_requires_attested_agent_provenance() {
        let community_id = community(1);
        let unattested = agent_profile(community_id, false);
        let attested = agent_profile(community_id, true);

        assert_eq!(
            AuthenticatedPrincipal::owner_attested_agent(
                principal(1),
                community_id,
                &unattested,
                NostrAuthenticationMethod::Nip42,
                PrincipalScopes::default(),
            ),
            Err(PrincipalError::OwnerAttestationRequired)
        );

        let authenticated = AuthenticatedPrincipal::owner_attested_agent(
            principal(1),
            community_id,
            &attested,
            NostrAuthenticationMethod::Nip98,
            scopes(&["jobs:write"]),
        )
        .expect("attested agent");
        assert!(matches!(
            authenticated.kind(),
            AuthenticatedPrincipalKind::OwnerAttestedAgent {
                agent_public_key,
                owner_public_key,
                authentication_method: NostrAuthenticationMethod::Nip98,
                ..
            } if *agent_public_key == key(6) && *owner_public_key == key(7)
        ));
    }

    #[test]
    fn authenticated_principal_preserves_token_and_service_scopes() {
        let community_id = community(1);
        let granted = scopes(&["messages:read", "future:capability", "messages:read"]);
        let token = AuthenticatedPrincipal::scoped_token(
            principal(1),
            community_id,
            TokenId::from_uuid(Uuid::from_u128(70)),
            principal(2),
            granted.clone(),
        );
        let service = AuthenticatedPrincipal::service(
            principal(3),
            community_id,
            "workflow-runner",
            granted.clone(),
        )
        .expect("service principal");

        assert_eq!(token.scopes(), &granted);
        assert_eq!(service.scopes(), &granted);
        assert_eq!(granted.len(), 2);
        assert!(
            granted
                .contains(&AuthorizationScope::new("future:capability").expect("extension scope"))
        );
    }

    #[test]
    fn authenticated_principal_scope_deserialization_remains_bounded() {
        assert!(serde_json::from_str::<AuthorizationScope>("\"scope with spaces\"").is_err());
        assert!(
            serde_json::from_str::<AuthorizationScope>(&format!(
                "\"{}\"",
                "x".repeat(MAX_SCOPE_BYTES + 1)
            ))
            .is_err()
        );
        assert!(
            serde_json::from_value::<PrincipalScopes>(serde_json::Value::Array(
                (0..=MAX_PRINCIPAL_SCOPES)
                    .map(|index| serde_json::Value::String(format!("scope:{index}")))
                    .collect(),
            ))
            .is_err()
        );
    }
    #[test]
    fn nip42_scopes_preserve_explicit_buzz_capabilities_without_role_authority() {
        use crate::{
            AggregateId, AggregateVersion, AuthorizationAction, AuthorizationDecision,
            AuthorizationDenial, AuthorizationRequest, AuthorizationResource,
            AuthorizationResourceKind, CommunityMembership, MembershipRole, MembershipStatus,
            TenantContext, TrustedTenantRoute, authorize,
        };
        let granted =
            PrincipalScopes::for_nip42_authenticated_key().expect("canonical NIP42 policy");
        let expected = scopes(&[
            "messages:read",
            "messages:write",
            "channels:read",
            "channels:write",
            "admin:channels",
            "users:read",
            "users:write",
            "admin:users",
            "jobs:read",
            "jobs:write",
            "subscriptions:read",
            "subscriptions:write",
            "files:read",
            "files:write",
            "repos:read",
            "repos:write",
        ]);
        assert_eq!(granted, expected);
        assert_eq!(granted.len(), 16);
        assert!(granted.iter().all(|scope| !scope.as_str().contains('*')));
        assert!(!granted.contains(&AuthorizationScope::new("events:read").expect("unknown scope")));
        let community = community(1);
        let principal = AuthenticatedPrincipal::nostr_identity(
            principal(2),
            community,
            key(3),
            NostrAuthenticationMethod::Nip42,
            granted,
        );
        let tenant = TenantContext::establish(
            Some(
                TrustedTenantRoute::from_listener(community, "nip42-scope-policy").expect("route"),
            ),
            &[],
        )
        .expect("tenant");
        let membership = CommunityMembership {
            community_id: community,
            principal_id: principal.principal_id(),
            role: MembershipRole::Member,
            status: MembershipStatus::Active,
            version: AggregateVersion::FIRST,
        };
        let required_scope = AuthorizationScope::new("admin:channels").expect("scope");
        let request = AuthorizationRequest {
            tenant: &tenant,
            principal: &principal,
            required_scope: &required_scope,
            action: AuthorizationAction::Manage,
            resource: AuthorizationResource {
                community_id: community,
                kind: AuthorizationResourceKind::Community,
                resource_id: AggregateId::from_uuid(Uuid::from_u128(4)),
                owner_principal_id: None,
                channel_id: None,
            },
            current_membership_version: AggregateVersion::FIRST,
            community_membership: Some(membership),
            channel_membership: None,
            current_channel_membership_version: None,
            delegation: None,
            now_millis: 10,
        };
        assert_eq!(
            authorize(&request),
            AuthorizationDecision::Denied(AuthorizationDenial::InsufficientRole)
        );
        let no_membership = AuthorizationRequest {
            community_membership: None,
            ..request
        };
        assert_eq!(
            authorize(&no_membership),
            AuthorizationDecision::Denied(AuthorizationDenial::MissingMembership)
        );
    }
}
