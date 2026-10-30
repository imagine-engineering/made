use crate::{AggregateVersion, CommunityId, OperationId, PrincipalId};
use serde::{Deserialize, Deserializer, Serialize, de};
use std::{collections::BTreeMap, fmt, num::NonZeroU64};
use uuid::Uuid;

const MAX_EVIDENCE_REFERENCE_BYTES: usize = 1_024;

macro_rules! uuid_identifier {
    ($name:ident) => {
        #[derive(
            Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
        )]
        #[serde(transparent)]
        pub struct $name(Uuid);

        impl $name {
            pub const fn from_uuid(value: Uuid) -> Self {
                Self(value)
            }

            pub const fn as_uuid(self) -> Uuid {
                self.0
            }

            pub fn new() -> Self {
                Self(Uuid::now_v7())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }
    };
}

uuid_identifier!(BindingId);
uuid_identifier!(ProfileId);

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ServiceAccountId(u64);

impl ServiceAccountId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ServiceAccountId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct NostrPublicKey([u8; 32]);

impl NostrPublicKey {
    pub const fn from_bytes(value: [u8; 32]) -> Self {
        Self(value)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct OrganizationPolicyVersion(NonZeroU64);

impl OrganizationPolicyVersion {
    pub const FIRST: Self = Self(NonZeroU64::MIN);

    pub const fn new(value: u64) -> Option<Self> {
        match NonZeroU64::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct EvidenceReference(String);

impl EvidenceReference {
    pub fn new(value: impl Into<String>) -> Option<Self> {
        let value = value.into();
        if value.is_empty() || value.len() > MAX_EVIDENCE_REFERENCE_BYTES {
            return None;
        }
        Some(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for EvidenceReference {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).ok_or_else(|| {
            de::Error::custom(format_args!(
                "evidence reference must be 1..={MAX_EVIDENCE_REFERENCE_BYTES} bytes"
            ))
        })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingStatus {
    Pending,
    Verified,
    Active,
    Rotated,
    Revoked,
    Archived,
}

impl BindingStatus {
    pub const fn is_historical(self) -> bool {
        matches!(self, Self::Rotated | Self::Revoked | Self::Archived)
    }

    pub const fn can_sign(self) -> bool {
        matches!(self, Self::Active)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingVerificationMethod {
    GeneratedKeyChallenge,
    ExistingKeyChallenge,
    ImportedKeyChallenge,
    PairedKeyChallenge,
    RestoredKeyChallenge,
    MigratedEvidence,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BindingVerification {
    pub method: BindingVerificationMethod,
    pub evidence_reference: EvidenceReference,
    pub verified_at_millis: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BindingVersionReference {
    pub binding_id: BindingId,
    pub version: AggregateVersion,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AccountBindingFields {
    pub binding_id: BindingId,
    pub community_id: CommunityId,
    pub service_account_id: ServiceAccountId,
    pub profile_id: ProfileId,
    pub public_key: NostrPublicKey,
    pub status: BindingStatus,
    pub verification: Option<BindingVerification>,
    pub predecessor: Option<BindingVersionReference>,
    pub successor: Option<BindingVersionReference>,
    pub created_at_millis: u64,
    pub activated_at_millis: Option<u64>,
    pub terminal_at_millis: Option<u64>,
    pub organization_policy_version: OrganizationPolicyVersion,
    pub actor_principal_id: PrincipalId,
    pub version: AggregateVersion,
    pub audit_reference: OperationId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct AccountBinding(AccountBindingFields);

impl AccountBinding {
    pub fn new(fields: AccountBindingFields) -> Result<Self, AccountBindingError> {
        validate_fields(&fields)?;
        Ok(Self(fields))
    }

    pub const fn fields(&self) -> &AccountBindingFields {
        &self.0
    }

    pub const fn binding_id(&self) -> BindingId {
        self.0.binding_id
    }

    pub const fn community_id(&self) -> CommunityId {
        self.0.community_id
    }

    pub const fn service_account_id(&self) -> ServiceAccountId {
        self.0.service_account_id
    }

    pub const fn profile_id(&self) -> ProfileId {
        self.0.profile_id
    }

    pub const fn public_key(&self) -> NostrPublicKey {
        self.0.public_key
    }

    pub const fn status(&self) -> BindingStatus {
        self.0.status
    }

    pub const fn version(&self) -> AggregateVersion {
        self.0.version
    }

    pub const fn can_sign(&self) -> bool {
        self.status().can_sign()
    }

    pub const fn is_historical(&self) -> bool {
        self.status().is_historical()
    }
}

impl<'de> Deserialize<'de> for AccountBinding {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let fields = AccountBindingFields::deserialize(deserializer)?;
        Self::new(fields).map_err(de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AccountBindingError {
    InvalidPendingState,
    MissingVerification(BindingStatus),
    MigrationEvidenceCannotAuthorize,
    VerificationBeforeCreation,
    MissingActivation,
    ActivationBeforeVerification,
    UnexpectedTerminalTimestamp,
    MissingTerminalTimestamp,
    TerminalBeforeActivation,
    MissingRotationSuccessor,
    SelfReference,
    ActiveProfileConflict { first: BindingId, second: BindingId },
    ActiveOwnerConflict { first: BindingId, second: BindingId },
}

impl fmt::Display for AccountBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPendingState => formatter
                .write_str("pending binding must not contain verification or lifecycle timestamps"),
            Self::MissingVerification(status) => {
                write!(
                    formatter,
                    "{status:?} binding requires verified possession evidence"
                )
            }
            Self::MigrationEvidenceCannotAuthorize => formatter.write_str(
                "migrated evidence can preserve history but cannot authorize a live binding",
            ),
            Self::VerificationBeforeCreation => {
                formatter.write_str("verified timestamp precedes binding creation")
            }
            Self::MissingActivation => {
                formatter.write_str("active or historical binding requires an activation timestamp")
            }
            Self::ActivationBeforeVerification => {
                formatter.write_str("activation timestamp precedes verification")
            }
            Self::UnexpectedTerminalTimestamp => {
                formatter.write_str("live binding must not contain a terminal timestamp")
            }
            Self::MissingTerminalTimestamp => {
                formatter.write_str("historical binding requires a terminal timestamp")
            }
            Self::TerminalBeforeActivation => {
                formatter.write_str("terminal timestamp precedes activation")
            }
            Self::MissingRotationSuccessor => {
                formatter.write_str("rotated binding requires a distinct successor")
            }
            Self::SelfReference => {
                formatter.write_str("binding cannot refer to itself as predecessor or successor")
            }
            Self::ActiveProfileConflict { first, second } => write!(
                formatter,
                "community/account/profile tuple has active bindings {first} and {second}"
            ),
            Self::ActiveOwnerConflict { first, second } => write!(
                formatter,
                "community public key has conflicting active accounts in {first} and {second}"
            ),
        }
    }
}

impl std::error::Error for AccountBindingError {}

fn validate_fields(fields: &AccountBindingFields) -> Result<(), AccountBindingError> {
    if fields
        .predecessor
        .is_some_and(|reference| reference.binding_id == fields.binding_id)
        || fields
            .successor
            .is_some_and(|reference| reference.binding_id == fields.binding_id)
    {
        return Err(AccountBindingError::SelfReference);
    }

    if fields.status == BindingStatus::Pending {
        if fields.verification.is_some()
            || fields.activated_at_millis.is_some()
            || fields.terminal_at_millis.is_some()
            || fields.predecessor.is_some()
            || fields.successor.is_some()
        {
            return Err(AccountBindingError::InvalidPendingState);
        }
        return Ok(());
    }

    let verification = fields
        .verification
        .as_ref()
        .ok_or(AccountBindingError::MissingVerification(fields.status))?;
    if verification.verified_at_millis < fields.created_at_millis {
        return Err(AccountBindingError::VerificationBeforeCreation);
    }
    if verification.method == BindingVerificationMethod::MigratedEvidence
        && !fields.status.is_historical()
    {
        return Err(AccountBindingError::MigrationEvidenceCannotAuthorize);
    }

    match fields.status {
        BindingStatus::Verified => {
            if fields.activated_at_millis.is_some() {
                return Err(AccountBindingError::ActivationBeforeVerification);
            }
            if fields.terminal_at_millis.is_some() {
                return Err(AccountBindingError::UnexpectedTerminalTimestamp);
            }
        }
        BindingStatus::Active | BindingStatus::Rotated | BindingStatus::Archived => {
            let activated_at = fields
                .activated_at_millis
                .ok_or(AccountBindingError::MissingActivation)?;
            if activated_at < verification.verified_at_millis {
                return Err(AccountBindingError::ActivationBeforeVerification);
            }
        }
        BindingStatus::Revoked => {
            if fields
                .activated_at_millis
                .is_some_and(|activated_at| activated_at < verification.verified_at_millis)
            {
                return Err(AccountBindingError::ActivationBeforeVerification);
            }
        }
        BindingStatus::Pending => return Err(AccountBindingError::InvalidPendingState),
    }

    if fields.status.is_historical() {
        let terminal_at = fields
            .terminal_at_millis
            .ok_or(AccountBindingError::MissingTerminalTimestamp)?;
        let lower_bound = fields
            .activated_at_millis
            .unwrap_or(verification.verified_at_millis);
        if terminal_at < lower_bound {
            return Err(AccountBindingError::TerminalBeforeActivation);
        }
    } else if fields.terminal_at_millis.is_some() {
        return Err(AccountBindingError::UnexpectedTerminalTimestamp);
    }

    if fields.status == BindingStatus::Rotated
        && fields
            .successor
            .is_none_or(|successor| successor.binding_id == fields.binding_id)
    {
        return Err(AccountBindingError::MissingRotationSuccessor);
    }
    Ok(())
}

pub fn validate_active_bindings<'a>(
    bindings: impl IntoIterator<Item = &'a AccountBinding>,
) -> Result<(), AccountBindingError> {
    let mut active_profiles = BTreeMap::new();
    let mut active_key_owners = BTreeMap::new();
    for binding in bindings {
        if binding.status() != BindingStatus::Active {
            continue;
        }
        let profile_key = (
            binding.community_id(),
            binding.service_account_id(),
            binding.profile_id(),
        );
        if let Some(first) = active_profiles.insert(profile_key, binding.binding_id()) {
            return Err(AccountBindingError::ActiveProfileConflict {
                first,
                second: binding.binding_id(),
            });
        }

        let owner_key = (binding.community_id(), binding.public_key());
        if let Some((first_account, first_binding)) = active_key_owners.get(&owner_key) {
            if *first_account != binding.service_account_id() {
                return Err(AccountBindingError::ActiveOwnerConflict {
                    first: *first_binding,
                    second: binding.binding_id(),
                });
            }
        }
        active_key_owners.insert(
            owner_key,
            (binding.service_account_id(), binding.binding_id()),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields(status: BindingStatus) -> AccountBindingFields {
        let binding_id = BindingId::from_uuid(Uuid::from_u128(10));
        let verification = (status != BindingStatus::Pending).then(|| BindingVerification {
            method: BindingVerificationMethod::ExistingKeyChallenge,
            evidence_reference: EvidenceReference::new("evidence:challenge-1")
                .expect("bounded evidence reference"),
            verified_at_millis: 20,
        });
        AccountBindingFields {
            binding_id,
            community_id: CommunityId::from_uuid(Uuid::from_u128(1)),
            service_account_id: ServiceAccountId::new(7),
            profile_id: ProfileId::from_uuid(Uuid::from_u128(2)),
            public_key: NostrPublicKey::from_bytes([3; 32]),
            status,
            verification,
            predecessor: None,
            successor: None,
            created_at_millis: 10,
            activated_at_millis: matches!(
                status,
                BindingStatus::Active | BindingStatus::Rotated | BindingStatus::Archived
            )
            .then_some(30),
            terminal_at_millis: status.is_historical().then_some(40),
            organization_policy_version: OrganizationPolicyVersion::FIRST,
            actor_principal_id: PrincipalId::from_uuid(Uuid::from_u128(4)),
            version: AggregateVersion::FIRST,
            audit_reference: OperationId::from_uuid(Uuid::from_u128(5)),
        }
    }

    fn binding(status: BindingStatus) -> AccountBinding {
        let mut fields = fields(status);
        if status == BindingStatus::Rotated {
            fields.successor = Some(BindingVersionReference {
                binding_id: BindingId::from_uuid(Uuid::from_u128(11)),
                version: AggregateVersion::FIRST,
            });
        }
        AccountBinding::new(fields).expect("valid binding")
    }

    #[test]
    fn account_binding_verified_state_requires_bounded_possession_evidence() {
        let verified = binding(BindingStatus::Verified);
        assert_eq!(
            verified
                .fields()
                .verification
                .as_ref()
                .map(|verification| verification.method),
            Some(BindingVerificationMethod::ExistingKeyChallenge)
        );
        assert!(!verified.can_sign());

        let mut missing = fields(BindingStatus::Verified);
        missing.verification = None;
        assert_eq!(
            AccountBinding::new(missing),
            Err(AccountBindingError::MissingVerification(
                BindingStatus::Verified
            ))
        );
        assert!(EvidenceReference::new("").is_none());
        assert!(EvidenceReference::new("x".repeat(1_025)).is_none());

        let mut migrated = fields(BindingStatus::Active);
        migrated
            .verification
            .as_mut()
            .expect("active binding verification")
            .method = BindingVerificationMethod::MigratedEvidence;
        assert_eq!(
            AccountBinding::new(migrated),
            Err(AccountBindingError::MigrationEvidenceCannotAuthorize)
        );
    }

    #[test]
    fn account_binding_active_conflicts_fail_within_the_community() {
        let first = binding(BindingStatus::Active);
        let mut same_profile_fields = fields(BindingStatus::Active);
        same_profile_fields.binding_id = BindingId::from_uuid(Uuid::from_u128(12));
        same_profile_fields.public_key = NostrPublicKey::from_bytes([4; 32]);
        let same_profile = AccountBinding::new(same_profile_fields).expect("valid binding");
        assert!(matches!(
            validate_active_bindings([&first, &same_profile]),
            Err(AccountBindingError::ActiveProfileConflict { .. })
        ));

        let mut other_owner_fields = fields(BindingStatus::Active);
        other_owner_fields.binding_id = BindingId::from_uuid(Uuid::from_u128(13));
        other_owner_fields.profile_id = ProfileId::from_uuid(Uuid::from_u128(14));
        other_owner_fields.service_account_id = ServiceAccountId::new(8);
        let other_owner = AccountBinding::new(other_owner_fields).expect("valid binding");
        assert!(matches!(
            validate_active_bindings([&first, &other_owner]),
            Err(AccountBindingError::ActiveOwnerConflict { .. })
        ));
    }

    #[test]
    fn account_binding_reuse_across_communities_does_not_share_authority() {
        let first = binding(BindingStatus::Active);
        let mut second_fields = fields(BindingStatus::Active);
        second_fields.binding_id = BindingId::from_uuid(Uuid::from_u128(15));
        second_fields.community_id = CommunityId::from_uuid(Uuid::from_u128(16));
        second_fields.service_account_id = ServiceAccountId::new(99);
        let second = AccountBinding::new(second_fields).expect("valid binding");

        validate_active_bindings([&first, &second]).expect("community fence isolates authority");
        assert_ne!(first.community_id(), second.community_id());
        assert_eq!(first.public_key(), second.public_key());
    }

    #[test]
    fn account_binding_revoked_and_historical_states_never_sign() {
        let revoked = binding(BindingStatus::Revoked);
        let rotated = binding(BindingStatus::Rotated);
        let archived = binding(BindingStatus::Archived);
        for historical in [&revoked, &rotated, &archived] {
            assert!(historical.is_historical());
            assert!(!historical.can_sign());
        }

        let encoded = serde_json::to_string(&revoked).expect("serialize revoked binding");
        let decoded: AccountBinding =
            serde_json::from_str(&encoded).expect("deserialize revoked binding");
        assert_eq!(decoded, revoked);

        let mut malformed = fields(BindingStatus::Revoked);
        malformed.terminal_at_millis = None;
        assert_eq!(
            AccountBinding::new(malformed),
            Err(AccountBindingError::MissingTerminalTimestamp)
        );
    }
}
