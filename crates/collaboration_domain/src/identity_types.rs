use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

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

uuid_identifier!(CommunityId);
uuid_identifier!(AggregateId);
uuid_identifier!(OperationId);
uuid_identifier!(PrincipalId);

impl AggregateId {
    pub fn from_source(namespace: Uuid, source_identifier: &[u8]) -> Self {
        Self(Uuid::new_v5(&namespace, source_identifier))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregateType {
    Community,
    Project,
    Conversation,
    AgentSession,
    Activity,
    GitChange,
    Workflow,
    Identity,
    Presence,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct ScopedAggregateId {
    community_id: CommunityId,
    aggregate_type: AggregateType,
    aggregate_id: AggregateId,
}

impl ScopedAggregateId {
    pub const fn new(
        community_id: CommunityId,
        aggregate_type: AggregateType,
        aggregate_id: AggregateId,
    ) -> Self {
        Self {
            community_id,
            aggregate_type,
            aggregate_id,
        }
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn aggregate_type(self) -> AggregateType {
        self.aggregate_type
    }

    pub const fn aggregate_id(self) -> AggregateId {
        self.aggregate_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_identity_is_tenant_scoped() {
        let aggregate_id = AggregateId::from_uuid(Uuid::from_u128(11));
        let first = ScopedAggregateId::new(
            CommunityId::from_uuid(Uuid::from_u128(1)),
            AggregateType::Conversation,
            aggregate_id,
        );
        let second = ScopedAggregateId::new(
            CommunityId::from_uuid(Uuid::from_u128(2)),
            AggregateType::Conversation,
            aggregate_id,
        );

        assert_ne!(first, second);
        assert_eq!(first.aggregate_id(), second.aggregate_id());
    }

    #[test]
    fn provenance_source_identity_is_stable() {
        let namespace = Uuid::from_u128(42);

        assert_eq!(
            AggregateId::from_source(namespace, b"buzz-event-id"),
            AggregateId::from_source(namespace, b"buzz-event-id")
        );
        assert_ne!(
            AggregateId::from_source(namespace, b"buzz-event-id"),
            AggregateId::from_source(namespace, b"other-event-id")
        );
    }
}
