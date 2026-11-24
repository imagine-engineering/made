use std::{collections::BTreeMap, error::Error, fmt};

use crate::{CommunityId, PrincipalId};

const MAX_CONTEXT_BYTES: usize = 256;
const MAX_OVERRIDE_CONTEXT_BYTES: usize = MAX_CONTEXT_BYTES - "ov_s:".len();
const MAX_CONTEXTS: usize = 10_000;

#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ReadContextId(String);

impl ReadContextId {
    pub fn new(value: impl Into<String>) -> Result<Self, ReadStateError> {
        let value = value.into();
        if value.is_empty() || value.len() > MAX_CONTEXT_BYTES {
            return Err(ReadStateError::InvalidContext);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn supports_override(&self) -> bool {
        self.0.len() <= MAX_OVERRIDE_CONTEXT_BYTES
    }
}

impl fmt::Debug for ReadContextId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReadContextId")
            .field("redacted", &true)
            .finish()
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct ReadStateScope {
    community_id: CommunityId,
    owner_principal_id: PrincipalId,
}

impl ReadStateScope {
    pub const fn new(community_id: CommunityId, owner_principal_id: PrincipalId) -> Self {
        Self {
            community_id,
            owner_principal_id,
        }
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn owner_principal_id(self) -> PrincipalId {
        self.owner_principal_id
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadStateCompleteness {
    Complete,
    PotentiallyIncomplete,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ManualUnreadRegister {
    set: u32,
    clear: u32,
    baseline: u32,
}

impl ManualUnreadRegister {
    pub const fn new(set: u32, clear: u32, baseline: u32) -> Self {
        Self {
            set,
            clear,
            baseline,
        }
    }

    pub const fn tombstone(counter: u32) -> Self {
        Self {
            set: 0,
            clear: counter,
            baseline: 0,
        }
    }

    pub const fn set(self) -> u32 {
        self.set
    }

    pub const fn clear(self) -> u32 {
        self.clear
    }

    pub const fn baseline(self) -> u32 {
        self.baseline
    }

    pub fn merge(self, other: Self) -> Self {
        Self {
            set: self.set.max(other.set),
            clear: self.clear.max(other.clear),
            baseline: self.baseline.max(other.baseline),
        }
    }

    pub fn is_active(self, effective_frontier: u32) -> bool {
        self.set > 0 && self.set > self.clear && effective_frontier <= self.baseline
    }

    fn mark_unread(&mut self, effective_frontier: u32) -> Result<(), ReadStateError> {
        self.set = self
            .set
            .max(self.clear)
            .checked_add(1)
            .ok_or(ReadStateError::CounterExhausted)?;
        self.baseline = effective_frontier;
        Ok(())
    }

    fn mark_read(&mut self, effective_frontier: u32) -> Result<(), ReadStateError> {
        let Some(next) = self.set.max(self.clear).checked_add(1) else {
            if self.is_active(effective_frontier) {
                return Err(ReadStateError::CounterExhausted);
            }
            return Ok(());
        };
        self.clear = next;
        Ok(())
    }

    fn canonical(self, effective_frontier: u32) -> ManualUnreadState {
        if self.is_active(effective_frontier) {
            ManualUnreadState::Live(self)
        } else if self.set > 0 || self.clear > 0 {
            ManualUnreadState::Tombstone {
                counter: self.set.max(self.clear),
            }
        } else {
            ManualUnreadState::Virgin
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManualUnreadState {
    Virgin,
    Live(ManualUnreadRegister),
    Tombstone { counter: u32 },
}

#[derive(Clone, Eq, PartialEq)]
pub struct OwnerReadStateReplica {
    scope: ReadStateScope,
    frontiers: BTreeMap<ReadContextId, u32>,
    overrides: BTreeMap<ReadContextId, ManualUnreadRegister>,
}

impl OwnerReadStateReplica {
    pub fn new(
        scope: ReadStateScope,
        decrypted_for: PrincipalId,
        frontiers: impl IntoIterator<Item = (ReadContextId, u32)>,
        overrides: impl IntoIterator<Item = (ReadContextId, ManualUnreadRegister)>,
    ) -> Result<Self, ReadStateError> {
        if scope.owner_principal_id != decrypted_for {
            return Err(ReadStateError::OwnerMismatch);
        }

        let mut normalized_frontiers = BTreeMap::<ReadContextId, u32>::new();
        for (context, frontier) in frontiers {
            normalized_frontiers
                .entry(context)
                .and_modify(|current| *current = (*current).max(frontier))
                .or_insert(frontier);
        }
        let mut normalized_overrides = BTreeMap::<ReadContextId, ManualUnreadRegister>::new();
        for (context, register) in overrides {
            if !context.supports_override() {
                return Err(ReadStateError::OverrideContextTooLong);
            }
            normalized_overrides
                .entry(context)
                .and_modify(|current| *current = current.merge(register))
                .or_insert(register);
        }
        if normalized_frontiers.len() + normalized_overrides.len() > MAX_CONTEXTS {
            return Err(ReadStateError::TooManyContexts);
        }
        if normalized_overrides
            .keys()
            .any(|context| !normalized_frontiers.contains_key(context))
        {
            return Err(ReadStateError::OverrideWithoutFrontier);
        }

        Ok(Self {
            scope,
            frontiers: normalized_frontiers,
            overrides: normalized_overrides,
        })
    }

    pub const fn scope(&self) -> ReadStateScope {
        self.scope
    }

    pub fn frontiers(&self) -> impl Iterator<Item = (&ReadContextId, u32)> {
        self.frontiers
            .iter()
            .map(|(context, frontier)| (context, *frontier))
    }

    pub fn overrides(&self) -> impl Iterator<Item = (&ReadContextId, ManualUnreadRegister)> {
        self.overrides
            .iter()
            .map(|(context, register)| (context, *register))
    }
}

impl fmt::Debug for OwnerReadStateReplica {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OwnerReadStateReplica")
            .field("frontier_count", &self.frontiers.len())
            .field("override_count", &self.overrides.len())
            .field("content", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct ReadState {
    scope: ReadStateScope,
    completeness: ReadStateCompleteness,
    frontiers: BTreeMap<ReadContextId, u32>,
    overrides: BTreeMap<ReadContextId, ManualUnreadRegister>,
}

impl ReadState {
    pub fn from_replicas(
        scope: ReadStateScope,
        completeness: ReadStateCompleteness,
        replicas: impl IntoIterator<Item = OwnerReadStateReplica>,
    ) -> Result<Self, ReadStateError> {
        let mut state = Self {
            scope,
            completeness,
            frontiers: BTreeMap::new(),
            overrides: BTreeMap::new(),
        };
        for replica in replicas {
            state.merge(replica)?;
        }
        Ok(state)
    }

    pub const fn scope(&self) -> ReadStateScope {
        self.scope
    }

    pub const fn completeness(&self) -> ReadStateCompleteness {
        self.completeness
    }

    pub fn mark_potentially_incomplete(&mut self) {
        self.completeness = ReadStateCompleteness::PotentiallyIncomplete;
    }

    pub fn merge(&mut self, replica: OwnerReadStateReplica) -> Result<(), ReadStateError> {
        if replica.scope != self.scope {
            return Err(ReadStateError::ScopeMismatch);
        }
        for (context, frontier) in replica.frontiers {
            self.frontiers
                .entry(context)
                .and_modify(|current| *current = (*current).max(frontier))
                .or_insert(frontier);
        }
        for (context, register) in replica.overrides {
            self.overrides
                .entry(context)
                .and_modify(|current| *current = current.merge(register))
                .or_insert(register);
        }
        Ok(())
    }

    pub fn effective_frontier(
        &self,
        viewer: PrincipalId,
        context: &ReadContextId,
        parent: Option<&ReadContextId>,
    ) -> Result<u32, ReadStateError> {
        self.require_owner(viewer)?;
        Ok(self.effective_frontier_unchecked(context, parent))
    }

    pub fn manual_unread_state(
        &self,
        viewer: PrincipalId,
        context: &ReadContextId,
        parent: Option<&ReadContextId>,
    ) -> Result<ManualUnreadState, ReadStateError> {
        self.require_owner(viewer)?;
        let effective_frontier = self.effective_frontier_unchecked(context, parent);
        Ok(self
            .overrides
            .get(context)
            .copied()
            .unwrap_or_default()
            .canonical(effective_frontier))
    }

    pub fn is_unread(
        &self,
        viewer: PrincipalId,
        context: &ReadContextId,
        parent: Option<&ReadContextId>,
        latest_message_at: u32,
    ) -> Result<bool, ReadStateError> {
        self.require_owner(viewer)?;
        let effective_frontier = self.effective_frontier_unchecked(context, parent);
        let override_active = self
            .overrides
            .get(context)
            .is_some_and(|register| register.is_active(effective_frontier));
        Ok(latest_message_at > effective_frontier || override_active)
    }

    pub fn advance_frontier(
        &mut self,
        owner: PrincipalId,
        context: ReadContextId,
        read_through: u32,
    ) -> Result<(), ReadStateError> {
        self.require_complete_owner(owner)?;
        self.frontiers
            .entry(context)
            .and_modify(|current| *current = (*current).max(read_through))
            .or_insert(read_through);
        Ok(())
    }

    pub fn mark_unread(
        &mut self,
        owner: PrincipalId,
        context: ReadContextId,
        parent: Option<&ReadContextId>,
    ) -> Result<(), ReadStateError> {
        self.require_override_action(owner, &context)?;
        let effective_frontier = self.effective_frontier_unchecked(&context, parent);
        self.frontiers.entry(context.clone()).or_default();
        self.overrides
            .entry(context)
            .or_default()
            .mark_unread(effective_frontier)
    }

    pub fn mark_read(
        &mut self,
        owner: PrincipalId,
        context: ReadContextId,
        parent: Option<&ReadContextId>,
        read_through: u32,
    ) -> Result<(), ReadStateError> {
        self.require_override_action(owner, &context)?;
        self.frontiers
            .entry(context.clone())
            .and_modify(|current| *current = (*current).max(read_through))
            .or_insert(read_through);
        let effective_frontier = self.effective_frontier_unchecked(&context, parent);
        self.overrides
            .entry(context)
            .or_default()
            .mark_read(effective_frontier)
    }

    pub fn canonical_replica(
        &self,
        owner: PrincipalId,
        parents: &BTreeMap<ReadContextId, ReadContextId>,
    ) -> Result<OwnerReadStateReplica, ReadStateError> {
        self.require_complete_owner(owner)?;
        let mut overrides = BTreeMap::new();
        for (context, register) in &self.overrides {
            let effective_frontier =
                self.effective_frontier_unchecked(context, parents.get(context));
            match register.canonical(effective_frontier) {
                ManualUnreadState::Virgin => {}
                ManualUnreadState::Live(register) => {
                    overrides.insert(context.clone(), register);
                }
                ManualUnreadState::Tombstone { counter } => {
                    overrides.insert(context.clone(), ManualUnreadRegister::tombstone(counter));
                }
            }
        }
        OwnerReadStateReplica::new(self.scope, owner, self.frontiers.clone(), overrides)
    }

    fn effective_frontier_unchecked(
        &self,
        context: &ReadContextId,
        parent: Option<&ReadContextId>,
    ) -> u32 {
        let own = self.frontiers.get(context).copied().unwrap_or_default();
        parent.map_or(own, |parent| {
            own.max(self.frontiers.get(parent).copied().unwrap_or_default())
        })
    }

    fn require_owner(&self, principal_id: PrincipalId) -> Result<(), ReadStateError> {
        if self.scope.owner_principal_id != principal_id {
            return Err(ReadStateError::OwnerMismatch);
        }
        Ok(())
    }

    fn require_complete_owner(&self, principal_id: PrincipalId) -> Result<(), ReadStateError> {
        self.require_owner(principal_id)?;
        if self.completeness != ReadStateCompleteness::Complete {
            return Err(ReadStateError::IncompleteLoad);
        }
        Ok(())
    }

    fn require_override_action(
        &self,
        principal_id: PrincipalId,
        context: &ReadContextId,
    ) -> Result<(), ReadStateError> {
        self.require_complete_owner(principal_id)?;
        if !context.supports_override() {
            return Err(ReadStateError::OverrideContextTooLong);
        }
        Ok(())
    }
}

impl fmt::Debug for ReadState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReadState")
            .field("completeness", &self.completeness)
            .field("frontier_count", &self.frontiers.len())
            .field("override_count", &self.overrides.len())
            .field("content", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadStateError {
    InvalidContext,
    OverrideContextTooLong,
    TooManyContexts,
    OverrideWithoutFrontier,
    OwnerMismatch,
    ScopeMismatch,
    IncompleteLoad,
    CounterExhausted,
}

impl fmt::Display for ReadStateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidContext => formatter.write_str("read context must contain 1..=256 bytes"),
            Self::OverrideContextTooLong => {
                formatter.write_str("manual-unread context exceeds the NIP-RS key limit")
            }
            Self::TooManyContexts => formatter.write_str("read state exceeds its context limit"),
            Self::OverrideWithoutFrontier => {
                formatter.write_str("manual-unread state requires a co-located frontier")
            }
            Self::OwnerMismatch => formatter.write_str("read state owner mismatch"),
            Self::ScopeMismatch => formatter.write_str("read state scope mismatch"),
            Self::IncompleteLoad => {
                formatter.write_str("manual read-state action requires a complete load")
            }
            Self::CounterExhausted => formatter.write_str("manual-unread counter is exhausted"),
        }
    }
}

impl Error for ReadStateError {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use uuid::Uuid;

    fn scope() -> ReadStateScope {
        ReadStateScope::new(
            CommunityId::from_uuid(Uuid::from_u128(1)),
            PrincipalId::from_uuid(Uuid::from_u128(2)),
        )
    }

    fn owner() -> PrincipalId {
        scope().owner_principal_id()
    }

    fn context(value: &str) -> ReadContextId {
        ReadContextId::new(value).expect("valid context")
    }

    fn replica(
        frontiers: impl IntoIterator<Item = (ReadContextId, u32)>,
        overrides: impl IntoIterator<Item = (ReadContextId, ManualUnreadRegister)>,
    ) -> OwnerReadStateReplica {
        OwnerReadStateReplica::new(scope(), owner(), frontiers, overrides)
            .expect("valid owner replica")
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            failure_persistence: None,
            ..ProptestConfig::default()
        })]

        #[test]
        fn frontier_merge_is_monotonic(values in prop::collection::vec(any::<u32>(), 1..64)) {
            let channel = context("channel-a");
            let replicas = values
                .iter()
                .map(|value| replica([(channel.clone(), *value)], []))
                .collect::<Vec<_>>();
            let state = ReadState::from_replicas(
                scope(),
                ReadStateCompleteness::Complete,
                replicas,
            ).expect("valid merged state");

            prop_assert_eq!(
                state
                    .effective_frontier(owner(), &channel, None)
                    .expect("owner frontier"),
                values.into_iter().max().unwrap_or_default()
            );
        }

        #[test]
        fn override_join_is_a_semilattice(
            first in (any::<u32>(), any::<u32>(), any::<u32>()),
            second in (any::<u32>(), any::<u32>(), any::<u32>()),
            third in (any::<u32>(), any::<u32>(), any::<u32>()),
        ) {
            let first = ManualUnreadRegister::new(first.0, first.1, first.2);
            let second = ManualUnreadRegister::new(second.0, second.1, second.2);
            let third = ManualUnreadRegister::new(third.0, third.1, third.2);

            prop_assert_eq!(first.merge(first), first);
            prop_assert_eq!(first.merge(second), second.merge(first));
            prop_assert_eq!(first.merge(second).merge(third), first.merge(second.merge(third)));
        }

        #[test]
        fn tombstone_floor_prevents_stale_override_resurrection(counter in 1_u32..u32::MAX, baseline in any::<u32>(), frontier in any::<u32>()) {
            let stale = ManualUnreadRegister::new(counter, 0, baseline);
            let joined = stale.merge(ManualUnreadRegister::tombstone(counter));

            prop_assert!(!joined.is_active(frontier));
            prop_assert_eq!(joined.canonical(frontier), ManualUnreadState::Tombstone { counter });
        }

        #[test]
        fn concurrent_devices_converge_independently_of_delivery_order(
            first_frontier in any::<u32>(),
            second_frontier in any::<u32>(),
            first_register in (any::<u32>(), any::<u32>(), any::<u32>()),
            second_register in (any::<u32>(), any::<u32>(), any::<u32>()),
        ) {
            let channel = context("channel-a");
            let first = replica(
                [(channel.clone(), first_frontier)],
                [(channel.clone(), ManualUnreadRegister::new(first_register.0, first_register.1, first_register.2))],
            );
            let second = replica(
                [(channel.clone(), second_frontier)],
                [(channel, ManualUnreadRegister::new(second_register.0, second_register.1, second_register.2))],
            );
            let forward = ReadState::from_replicas(
                scope(),
                ReadStateCompleteness::Complete,
                [first.clone(), second.clone()],
            ).expect("valid forward merge");
            let reverse = ReadState::from_replicas(
                scope(),
                ReadStateCompleteness::Complete,
                [second, first],
            ).expect("valid reverse merge");

            prop_assert_eq!(forward, reverse);
        }
    }

    #[test]
    fn manual_actions_require_complete_owner_state_and_preserve_frontier_progress() {
        let channel = context("channel-a");
        let other = PrincipalId::from_uuid(Uuid::from_u128(3));
        assert_eq!(
            OwnerReadStateReplica::new(scope(), other, [(channel.clone(), 10)], []),
            Err(ReadStateError::OwnerMismatch)
        );
        let mut incomplete = ReadState::from_replicas(
            scope(),
            ReadStateCompleteness::PotentiallyIncomplete,
            [replica([(channel.clone(), 10)], [])],
        )
        .expect("valid state");

        assert_eq!(
            incomplete.mark_unread(owner(), channel.clone(), None),
            Err(ReadStateError::IncompleteLoad)
        );
        assert_eq!(
            incomplete.effective_frontier(other, &channel, None),
            Err(ReadStateError::OwnerMismatch)
        );
        let foreign_scope =
            ReadStateScope::new(CommunityId::from_uuid(Uuid::from_u128(4)), owner());
        let foreign =
            OwnerReadStateReplica::new(foreign_scope, owner(), [(channel.clone(), 20)], [])
                .expect("valid foreign replica");
        assert_eq!(
            incomplete.merge(foreign),
            Err(ReadStateError::ScopeMismatch)
        );

        let mut complete = ReadState::from_replicas(
            scope(),
            ReadStateCompleteness::Complete,
            [replica([(channel.clone(), 10)], [])],
        )
        .expect("valid state");
        complete
            .mark_unread(owner(), channel.clone(), None)
            .expect("mark unread");
        assert!(
            complete
                .is_unread(owner(), &channel, None, 10)
                .expect("owner view")
        );
        complete
            .mark_read(owner(), channel.clone(), None, 11)
            .expect("mark read");
        assert!(
            !complete
                .is_unread(owner(), &channel, None, 11)
                .expect("owner view")
        );
        assert!(matches!(
            complete
                .manual_unread_state(owner(), &channel, None)
                .expect("owner view"),
            ManualUnreadState::Tombstone { .. }
        ));
    }

    #[test]
    fn natural_frontier_advance_deactivates_an_override_and_publication_is_canonical() {
        let channel = context("channel-a");
        let parent = context("parent");
        let mut state = ReadState::from_replicas(
            scope(),
            ReadStateCompleteness::Complete,
            [replica(
                [(channel.clone(), 5), (parent.clone(), 5)],
                [(channel.clone(), ManualUnreadRegister::new(1, 0, 5))],
            )],
        )
        .expect("valid state");

        state
            .advance_frontier(owner(), parent.clone(), 6)
            .expect("advance parent frontier");
        let parents = BTreeMap::from([(channel.clone(), parent)]);
        let canonical = state
            .canonical_replica(owner(), &parents)
            .expect("canonical replica");

        assert_eq!(
            canonical.overrides.get(&channel),
            Some(&ManualUnreadRegister::tombstone(1))
        );
    }

    #[test]
    fn exhausted_counters_never_wrap_or_report_a_live_override_as_read() {
        let channel = context("channel-a");
        let mut state = ReadState::from_replicas(
            scope(),
            ReadStateCompleteness::Complete,
            [replica(
                [(channel.clone(), 5)],
                [(channel.clone(), ManualUnreadRegister::new(u32::MAX, 0, 10))],
            )],
        )
        .expect("valid state");

        assert_eq!(
            state.mark_unread(owner(), channel.clone(), None),
            Err(ReadStateError::CounterExhausted)
        );
        assert_eq!(
            state.mark_read(owner(), channel.clone(), None, 10),
            Err(ReadStateError::CounterExhausted)
        );
        assert_eq!(
            state
                .effective_frontier(owner(), &channel, None)
                .expect("owner frontier"),
            10
        );
        assert!(
            state
                .is_unread(owner(), &channel, None, 10)
                .expect("owner view")
        );
    }

    #[test]
    fn debug_output_never_contains_owner_contexts() {
        let secret = context("secret-project-channel");
        let state = ReadState::from_replicas(
            scope(),
            ReadStateCompleteness::Complete,
            [replica([(secret, 42)], [])],
        )
        .expect("valid state");
        let debug = format!("{state:?}");

        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("secret-project-channel"));
        assert!(!debug.contains(&owner().to_string()));
    }
}
