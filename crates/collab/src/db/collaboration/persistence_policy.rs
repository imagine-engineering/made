use nostr_compat::{
    generated_kinds::{KindStatus, PrivacyGates, metadata_for_kind},
    head::PersistenceClass,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrivacyAdmission {
    author: bool,
    recipient: bool,
    result_reader: bool,
    explicitly_shared: bool,
}

impl PrivacyAdmission {
    pub const fn community() -> Self {
        Self {
            author: false,
            recipient: false,
            result_reader: false,
            explicitly_shared: false,
        }
    }

    pub const fn author() -> Self {
        Self {
            author: true,
            ..Self::community()
        }
    }

    pub const fn recipient() -> Self {
        Self {
            recipient: true,
            ..Self::community()
        }
    }

    pub const fn with_result_reader(mut self) -> Self {
        self.result_reader = true;
        self
    }

    pub const fn with_explicit_share(mut self) -> Self {
        self.explicitly_shared = true;
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventDurability {
    Durable,
    TransientOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventSearchScope {
    Excluded,
    Community,
    AuthorizedRestricted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchAudience {
    Community,
    AuthorizedRestricted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventPersistenceDecision {
    kind: u16,
    durability: EventDurability,
    search_scope: EventSearchScope,
}

impl EventPersistenceDecision {
    pub const fn kind(self) -> u16 {
        self.kind
    }

    pub const fn durability(self) -> EventDurability {
        self.durability
    }

    pub const fn search_scope(self) -> EventSearchScope {
        self.search_scope
    }

    pub const fn allows_search_for(self, audience: SearchAudience) -> bool {
        matches!(
            (self.search_scope, audience),
            (EventSearchScope::Community, SearchAudience::Community)
                | (
                    EventSearchScope::AuthorizedRestricted,
                    SearchAudience::AuthorizedRestricted
                )
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PersistencePolicyError {
    #[error("event kind has no approved persistence classification")]
    UnclassifiedKind,
    #[error("event kind is not accepted by the relay persistence boundary")]
    NonRelayKind,
    #[error("event privacy admission is insufficient")]
    PrivacyDenied,
    #[error("event persistence decision does not match the event")]
    DecisionMismatch,
}

pub struct EventPersistencePolicy;

impl EventPersistencePolicy {
    pub fn evaluate(
        kind: u16,
        admission: PrivacyAdmission,
    ) -> Result<EventPersistenceDecision, PersistencePolicyError> {
        let metadata =
            metadata_for_kind(u32::from(kind)).ok_or(PersistencePolicyError::UnclassifiedKind)?;
        match metadata.status {
            KindStatus::Registered | KindStatus::DefinedUnused => {}
            KindStatus::InternalNotRelayEvent => {
                return Err(PersistencePolicyError::NonRelayKind);
            }
        }
        if metadata.persistence == PersistenceClass::Ephemeral {
            return Ok(EventPersistenceDecision {
                kind,
                durability: EventDurability::TransientOnly,
                search_scope: EventSearchScope::Excluded,
            });
        }
        if !privacy_gates_satisfied(metadata.privacy, admission) {
            return Err(PersistencePolicyError::PrivacyDenied);
        }
        let search_scope = if metadata.privacy.is_community_visible() {
            EventSearchScope::Community
        } else {
            EventSearchScope::AuthorizedRestricted
        };
        Ok(EventPersistenceDecision {
            kind,
            durability: EventDurability::Durable,
            search_scope,
        })
    }

    pub fn validate_for_event(
        kind: u16,
        decision: EventPersistenceDecision,
    ) -> Result<EventPersistenceDecision, PersistencePolicyError> {
        if decision.kind != kind {
            return Err(PersistencePolicyError::DecisionMismatch);
        }
        Ok(decision)
    }
}

const fn privacy_gates_satisfied(gates: PrivacyGates, admission: PrivacyAdmission) -> bool {
    (!gates.contains(PrivacyGates::AUTHOR_ONLY) || admission.author)
        && (!gates.contains(PrivacyGates::RESULT_GATED) || admission.result_reader)
        && (!gates.contains(PrivacyGates::RECIPIENT_GATED) || admission.recipient)
        && (!gates.contains(PrivacyGates::AUTHOR_OR_SHARED)
            || admission.author
            || admission.explicitly_shared)
}
