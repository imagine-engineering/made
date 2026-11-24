use std::{error::Error, fmt, num::NonZeroU64};

use serde::{Deserialize, Deserializer, Serialize, de};

use crate::{CommunityId, PrincipalId};

const MAX_INSTALLATION_ID_BYTES: usize = 64;

#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct PushInstallationId(String);

impl PushInstallationId {
    pub fn new(value: impl Into<String>) -> Option<Self> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_INSTALLATION_ID_BYTES
            || value.chars().any(char::is_control)
        {
            return None;
        }
        Some(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for PushInstallationId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PushInstallationId([redacted])")
    }
}

impl<'de> Deserialize<'de> for PushInstallationId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value)
            .ok_or_else(|| de::Error::custom("push installation id must contain 1..=64 bytes"))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct PushLeaseAddress {
    pub community_id: CommunityId,
    pub owner_principal_id: PrincipalId,
    pub installation_id: PushInstallationId,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct PushCapabilityReference([u8; 32]);

impl PushCapabilityReference {
    pub fn from_digest(digest: [u8; 32]) -> Option<Self> {
        (!digest.iter().all(|byte| *byte == 0)).then_some(Self(digest))
    }

    pub const fn as_digest(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for PushCapabilityReference {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PushCapabilityReference([redacted])")
    }
}

impl<'de> Deserialize<'de> for PushCapabilityReference {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let digest = <[u8; 32]>::deserialize(deserializer)?;
        Self::from_digest(digest)
            .ok_or_else(|| de::Error::custom("push capability digest must be nonzero"))
    }
}

macro_rules! positive_generation {
    ($name:ident) => {
        #[derive(
            Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
        )]
        #[serde(transparent)]
        pub struct $name(NonZeroU64);

        impl $name {
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
    };
}

positive_generation!(PushLeaseGeneration);
positive_generation!(PushEndpointGeneration);

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum PushLeaseState {
    Active {
        capability_reference: PushCapabilityReference,
        endpoint_generation: PushEndpointGeneration,
    },
    Revoked {
        revoked_at_millis: u64,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PushLeaseRecordFields {
    pub address: PushLeaseAddress,
    pub generation: PushLeaseGeneration,
    pub expires_at_millis: u64,
    pub last_active_expires_at_millis: u64,
    pub state: PushLeaseState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PushLeaseActivation {
    pub generation: PushLeaseGeneration,
    pub expires_at_millis: u64,
    pub capability_reference: PushCapabilityReference,
    pub endpoint_generation: PushEndpointGeneration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PushLease {
    fields: PushLeaseRecordFields,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PushWakeRequest {
    pub lease_generation: PushLeaseGeneration,
    pub endpoint_generation: PushEndpointGeneration,
    pub capability_reference: PushCapabilityReference,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PushWakePayload {
    Reconnect,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PushWake {
    address: PushLeaseAddress,
    lease_generation: PushLeaseGeneration,
    endpoint_generation: PushEndpointGeneration,
    capability_reference: PushCapabilityReference,
    expires_at_millis: u64,
}

impl PushLease {
    pub fn activate(
        address: PushLeaseAddress,
        activation: PushLeaseActivation,
        now_millis: u64,
    ) -> Result<Self, PushLeaseError> {
        validate_address(&address)?;
        validate_expiration(activation.expires_at_millis, now_millis)?;
        Ok(Self {
            fields: PushLeaseRecordFields {
                address,
                generation: activation.generation,
                expires_at_millis: activation.expires_at_millis,
                last_active_expires_at_millis: activation.expires_at_millis,
                state: PushLeaseState::Active {
                    capability_reference: activation.capability_reference,
                    endpoint_generation: activation.endpoint_generation,
                },
            },
        })
    }

    pub fn from_record(fields: PushLeaseRecordFields) -> Result<Self, PushLeaseError> {
        validate_address(&fields.address)?;
        match fields.state {
            PushLeaseState::Active { .. }
                if fields.last_active_expires_at_millis != fields.expires_at_millis =>
            {
                return Err(PushLeaseError::InvalidRecord);
            }
            PushLeaseState::Revoked { .. } if fields.last_active_expires_at_millis == 0 => {
                return Err(PushLeaseError::InvalidRecord);
            }
            _ => {}
        }
        Ok(Self { fields })
    }

    pub const fn fields(&self) -> &PushLeaseRecordFields {
        &self.fields
    }

    pub fn replace(
        &mut self,
        activation: PushLeaseActivation,
        now_millis: u64,
    ) -> Result<(), PushLeaseError> {
        self.require_newer_generation(activation.generation)?;
        validate_expiration(activation.expires_at_millis, now_millis)?;
        self.fields.generation = activation.generation;
        self.fields.expires_at_millis = activation.expires_at_millis;
        self.fields.last_active_expires_at_millis = activation.expires_at_millis;
        self.fields.state = PushLeaseState::Active {
            capability_reference: activation.capability_reference,
            endpoint_generation: activation.endpoint_generation,
        };
        Ok(())
    }

    pub fn revoke(
        &mut self,
        generation: PushLeaseGeneration,
        tombstone_expires_at_millis: u64,
        now_millis: u64,
    ) -> Result<(), PushLeaseError> {
        self.require_newer_generation(generation)?;
        validate_expiration(tombstone_expires_at_millis, now_millis)?;
        self.fields.generation = generation;
        self.fields.expires_at_millis = tombstone_expires_at_millis;
        self.fields.state = PushLeaseState::Revoked {
            revoked_at_millis: now_millis,
        };
        Ok(())
    }

    pub fn authorize_wake(
        &self,
        request: PushWakeRequest,
        now_millis: u64,
    ) -> Result<PushWake, PushLeaseError> {
        let PushLeaseState::Active {
            capability_reference,
            endpoint_generation,
        } = self.fields.state
        else {
            return Err(PushLeaseError::Revoked);
        };
        if now_millis > self.fields.expires_at_millis {
            return Err(PushLeaseError::Expired);
        }
        if request.lease_generation != self.fields.generation {
            return Err(PushLeaseError::WrongLeaseGeneration);
        }
        if request.endpoint_generation != endpoint_generation {
            return Err(PushLeaseError::WrongEndpointGeneration);
        }
        if request.capability_reference != capability_reference {
            return Err(PushLeaseError::WrongCapability);
        }
        Ok(PushWake {
            address: self.fields.address.clone(),
            lease_generation: self.fields.generation,
            endpoint_generation,
            capability_reference,
            expires_at_millis: self.fields.expires_at_millis,
        })
    }

    pub fn watermark_retention_until(
        &self,
        max_lease_ttl_millis: u64,
        allowed_skew_millis: u64,
    ) -> Result<u64, PushLeaseError> {
        let retention_base = match self.fields.state {
            PushLeaseState::Active { .. } => self.fields.last_active_expires_at_millis,
            PushLeaseState::Revoked { revoked_at_millis } => {
                self.fields.last_active_expires_at_millis.max(
                    revoked_at_millis
                        .checked_add(max_lease_ttl_millis)
                        .ok_or(PushLeaseError::TimestampOverflow)?,
                )
            }
        };
        retention_base
            .checked_add(allowed_skew_millis)
            .ok_or(PushLeaseError::TimestampOverflow)
    }

    fn require_newer_generation(
        &self,
        generation: PushLeaseGeneration,
    ) -> Result<(), PushLeaseError> {
        if generation <= self.fields.generation {
            return Err(PushLeaseError::StaleGeneration);
        }
        Ok(())
    }
}

impl PushWake {
    pub const fn payload(&self) -> PushWakePayload {
        PushWakePayload::Reconnect
    }

    pub const fn address(&self) -> &PushLeaseAddress {
        &self.address
    }

    pub const fn lease_generation(&self) -> PushLeaseGeneration {
        self.lease_generation
    }

    pub const fn endpoint_generation(&self) -> PushEndpointGeneration {
        self.endpoint_generation
    }

    pub const fn capability_reference(&self) -> PushCapabilityReference {
        self.capability_reference
    }

    pub const fn expires_at_millis(&self) -> u64 {
        self.expires_at_millis
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PushLeaseError {
    InvalidAddress,
    InvalidRecord,
    InvalidExpiration,
    StaleGeneration,
    Revoked,
    Expired,
    WrongLeaseGeneration,
    WrongEndpointGeneration,
    WrongCapability,
    TimestampOverflow,
}

impl fmt::Display for PushLeaseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidAddress => "invalid push lease address",
            Self::InvalidRecord => "invalid push lease record",
            Self::InvalidExpiration => "invalid push lease expiration",
            Self::StaleGeneration => "push lease generation is stale",
            Self::Revoked => "push lease is revoked",
            Self::Expired => "push lease is expired",
            Self::WrongLeaseGeneration => "push lease generation does not match",
            Self::WrongEndpointGeneration => "push endpoint generation does not match",
            Self::WrongCapability => "push capability does not match",
            Self::TimestampOverflow => "push lease timestamp overflow",
        })
    }
}

impl Error for PushLeaseError {}

fn validate_address(address: &PushLeaseAddress) -> Result<(), PushLeaseError> {
    if address.community_id.as_uuid().is_nil() || address.owner_principal_id.as_uuid().is_nil() {
        return Err(PushLeaseError::InvalidAddress);
    }
    Ok(())
}

fn validate_expiration(expires_at_millis: u64, now_millis: u64) -> Result<(), PushLeaseError> {
    if expires_at_millis <= now_millis {
        return Err(PushLeaseError::InvalidExpiration);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;

    fn generation(value: u64) -> PushLeaseGeneration {
        PushLeaseGeneration::new(value).expect("positive lease generation")
    }

    fn endpoint_generation(value: u64) -> PushEndpointGeneration {
        PushEndpointGeneration::new(value).expect("positive endpoint generation")
    }

    fn capability(value: u8) -> PushCapabilityReference {
        PushCapabilityReference::from_digest([value; 32]).expect("nonzero capability digest")
    }

    fn address() -> PushLeaseAddress {
        PushLeaseAddress {
            community_id: CommunityId::from_uuid(Uuid::from_u128(1)),
            owner_principal_id: PrincipalId::from_uuid(Uuid::from_u128(2)),
            installation_id: PushInstallationId::new("random-per-origin-installation")
                .expect("valid installation id"),
        }
    }

    fn activation(lease_generation: u64) -> PushLeaseActivation {
        PushLeaseActivation {
            generation: generation(lease_generation),
            expires_at_millis: 2_000,
            capability_reference: capability(3),
            endpoint_generation: endpoint_generation(1),
        }
    }

    fn wake_request(lease: &PushLease) -> PushWakeRequest {
        let PushLeaseState::Active {
            capability_reference,
            endpoint_generation,
        } = lease.fields().state
        else {
            panic!("active lease")
        };
        PushWakeRequest {
            lease_generation: lease.fields().generation,
            endpoint_generation,
            capability_reference,
        }
    }

    #[test]
    fn push_lease_generation_must_strictly_increase() {
        let mut lease = PushLease::activate(address(), activation(3), 1_000).expect("active lease");
        let stale_request = wake_request(&lease);

        assert_eq!(
            lease.replace(activation(3), 1_000),
            Err(PushLeaseError::StaleGeneration)
        );
        assert_eq!(
            lease.replace(activation(2), 1_000),
            Err(PushLeaseError::StaleGeneration)
        );
        lease
            .replace(
                PushLeaseActivation {
                    generation: generation(7),
                    endpoint_generation: endpoint_generation(2),
                    ..activation(7)
                },
                1_000,
            )
            .expect("newer replacement");
        assert_eq!(lease.fields().generation, generation(7));
        assert_eq!(
            lease.authorize_wake(stale_request, 1_500),
            Err(PushLeaseError::WrongLeaseGeneration)
        );
    }

    #[test]
    fn push_lease_expiry_blocks_new_and_existing_wakes() {
        assert_eq!(
            PushLease::activate(address(), activation(1), 2_000),
            Err(PushLeaseError::InvalidExpiration)
        );
        let lease = PushLease::activate(address(), activation(1), 1_000).expect("active lease");
        let request = wake_request(&lease);

        lease
            .authorize_wake(request, 2_000)
            .expect("lease remains valid at its inclusive expiration boundary");
        assert_eq!(
            lease.authorize_wake(request, 2_001),
            Err(PushLeaseError::Expired)
        );
    }

    #[test]
    fn push_lease_revocation_is_generation_fenced_and_reactivatable() {
        let mut lease = PushLease::activate(address(), activation(1), 1_000).expect("active lease");
        let request = wake_request(&lease);

        lease
            .revoke(generation(2), 2_500, 1_500)
            .expect("newer tombstone");
        assert_eq!(
            lease.authorize_wake(request, 1_500),
            Err(PushLeaseError::Revoked)
        );
        assert_eq!(lease.watermark_retention_until(1_000, 50), Ok(2_550));
        assert_eq!(
            lease.replace(activation(2), 1_500),
            Err(PushLeaseError::StaleGeneration)
        );
        lease
            .replace(activation(3), 1_500)
            .expect("higher-generation reactivation");
        lease
            .authorize_wake(wake_request(&lease), 1_500)
            .expect("reactivated wake");
    }

    #[test]
    fn push_lease_rejects_the_wrong_capability() {
        let lease = PushLease::activate(address(), activation(1), 1_000).expect("active lease");
        let mut request = wake_request(&lease);
        request.capability_reference = capability(4);

        assert_eq!(
            lease.authorize_wake(request, 1_500),
            Err(PushLeaseError::WrongCapability)
        );
    }

    #[test]
    fn push_lease_wake_payload_is_only_the_fixed_reconnect_signal() {
        let lease = PushLease::activate(address(), activation(1), 1_000).expect("active lease");
        let wake = lease
            .authorize_wake(wake_request(&lease), 1_500)
            .expect("authorized wake");

        assert_eq!(wake.payload(), PushWakePayload::Reconnect);
        assert_eq!(
            serde_json::to_string(&wake.payload()).expect("serialize wake payload"),
            r#""reconnect""#
        );
    }

    #[test]
    fn push_lease_address_is_scoped_to_one_community() {
        let first = address();
        let mut second = first.clone();
        second.community_id = CommunityId::from_uuid(Uuid::from_u128(9));

        assert_ne!(first, second);
        assert_eq!(first.installation_id, second.installation_id);
    }
}
