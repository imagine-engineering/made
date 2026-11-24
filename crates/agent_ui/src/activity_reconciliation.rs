use std::{collections::HashMap, error::Error, fmt};

use crate::{
    activity_projection::{
        ActivityActorKind, ActivityContext, ActivityItem, ActivityItemId, ActivityLifecycle,
        ActivitySourceKind, ActivityVisibility,
    },
    activity_reducer::{ActivityReducer, ActivityReduction, ActivityReductionError},
};

const MAX_CORRELATION_ID_BYTES: usize = 1_024;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ActivityCorrelationId(String);

impl ActivityCorrelationId {
    pub fn new(value: impl Into<String>) -> Result<Self, ActivityReconciliationError> {
        let value = value.into();
        if value.trim().is_empty()
            || value.len() > MAX_CORRELATION_ID_BYTES
            || value.chars().any(char::is_control)
        {
            return Err(ActivityReconciliationError::InvalidCorrelationId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActivityProvenanceClass {
    Compatibility,
    Streaming,
    Authoritative,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivityReconciliationInput {
    pub correlation_id: ActivityCorrelationId,
    pub canonical_id: ActivityItemId,
    pub provenance: ActivityProvenanceClass,
    pub item: ActivityItem,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivitySourceProvenance {
    pub source_id: ActivityItemId,
    pub class: ActivityProvenanceClass,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActivityReconciliationReduction {
    Inserted { index: usize },
    Updated { index: usize },
    ProvenanceAdded { index: usize },
    SourceUpdated { index: usize },
    Duplicate { index: usize },
    IgnoredStale { index: usize },
}

struct ReconciliationGroup {
    canonical_id: ActivityItemId,
    actor_id: String,
    actor_kind: ActivityActorKind,
    context: ActivityContext,
    visibility: ActivityVisibility,
    sources: Vec<ActivityItemId>,
    provenance: HashMap<ActivityItemId, ActivityProvenanceClass>,
    display_version: u64,
}

#[derive(Default)]
pub struct ActivityReconciler {
    source_reducer: ActivityReducer,
    groups: Vec<ReconciliationGroup>,
    group_indices: HashMap<ActivityCorrelationId, usize>,
    source_bindings: HashMap<ActivityItemId, ActivityCorrelationId>,
    canonical_bindings: HashMap<ActivityItemId, ActivityCorrelationId>,
    items: Vec<ActivityItem>,
}

impl ActivityReconciler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn items(&self) -> &[ActivityItem] {
        &self.items
    }

    pub fn item(&self, correlation_id: &ActivityCorrelationId) -> Option<&ActivityItem> {
        self.group_indices
            .get(correlation_id)
            .map(|index| &self.items[*index])
    }

    pub fn provenance(
        &self,
        correlation_id: &ActivityCorrelationId,
    ) -> Option<Vec<ActivitySourceProvenance>> {
        let group = &self.groups[*self.group_indices.get(correlation_id)?];
        let mut provenance = group
            .sources
            .iter()
            .map(|source_id| ActivitySourceProvenance {
                source_id: source_id.clone(),
                class: group.provenance[source_id],
            })
            .collect::<Vec<_>>();
        provenance.sort_by(|left, right| {
            source_kind_rank(left.source_id.source_kind())
                .cmp(&source_kind_rank(right.source_id.source_kind()))
                .then_with(|| left.source_id.source_id().cmp(right.source_id.source_id()))
        });
        Some(provenance)
    }

    pub fn reduce(
        &mut self,
        input: ActivityReconciliationInput,
    ) -> Result<ActivityReconciliationReduction, ActivityReconciliationError> {
        let ActivityReconciliationInput {
            correlation_id,
            canonical_id,
            provenance,
            item,
        } = input;
        let source_id = item.id.clone();

        if let Some(bound_correlation_id) = self.source_bindings.get(&source_id)
            && bound_correlation_id != &correlation_id
        {
            return Err(ActivityReconciliationError::SourceBindingConflict {
                source_id,
                current: bound_correlation_id.clone(),
                incoming: correlation_id,
            });
        }
        if let Some(bound_correlation_id) = self.canonical_bindings.get(&source_id)
            && bound_correlation_id != &correlation_id
        {
            return Err(ActivityReconciliationError::SourceBindingConflict {
                source_id,
                current: bound_correlation_id.clone(),
                incoming: correlation_id,
            });
        }
        if let Some(bound_correlation_id) = self.canonical_bindings.get(&canonical_id)
            && bound_correlation_id != &correlation_id
        {
            return Err(ActivityReconciliationError::CanonicalBindingConflict {
                canonical_id,
                current: bound_correlation_id.clone(),
                incoming: correlation_id,
            });
        }
        if let Some(bound_correlation_id) = self.source_bindings.get(&canonical_id)
            && bound_correlation_id != &correlation_id
        {
            return Err(ActivityReconciliationError::CanonicalBindingConflict {
                canonical_id,
                current: bound_correlation_id.clone(),
                incoming: correlation_id,
            });
        }

        let Some(index) = self.group_indices.get(&correlation_id).copied() else {
            let reduction = self.source_reducer.reduce(item.clone())?;
            if !matches!(reduction, ActivityReduction::Inserted { .. }) {
                return Err(ActivityReconciliationError::InconsistentSourceState);
            }
            let mut provenance_by_source = HashMap::new();
            provenance_by_source.insert(source_id.clone(), provenance);
            let group = ReconciliationGroup {
                canonical_id: canonical_id.clone(),
                actor_id: item.actor.id.clone(),
                actor_kind: item.actor.kind,
                context: item.context.clone(),
                visibility: item.visibility,
                sources: vec![source_id.clone()],
                provenance: provenance_by_source,
                display_version: 1,
            };
            let mut display_item = item;
            display_item.id = canonical_id.clone();
            display_item.source_version = 1;
            let index = self.groups.len();
            self.groups.push(group);
            self.items.push(display_item);
            self.group_indices.insert(correlation_id.clone(), index);
            self.source_bindings
                .insert(source_id, correlation_id.clone());
            self.canonical_bindings.insert(canonical_id, correlation_id);
            return Ok(ActivityReconciliationReduction::Inserted { index });
        };

        {
            let group = &self.groups[index];
            if group.canonical_id != canonical_id {
                return Err(ActivityReconciliationError::CorrelationDefinitionConflict {
                    correlation_id,
                });
            }
            validate_actor(group, &item)?;
            merge_context(&group.context, &item.context)?;
            if let Some(current_provenance) = group.provenance.get(&source_id)
                && *current_provenance != provenance
            {
                return Err(ActivityReconciliationError::ProvenanceConflict { source_id });
            }
            if group.display_version == u64::MAX {
                return Err(ActivityReconciliationError::VersionExhausted);
            }
        }

        let is_new_source = !self.groups[index].provenance.contains_key(&source_id);
        let source_reduction = self.source_reducer.reduce(item)?;
        match source_reduction {
            ActivityReduction::Duplicate { .. } => {
                return Ok(ActivityReconciliationReduction::Duplicate { index });
            }
            ActivityReduction::IgnoredStale { .. } => {
                return Ok(ActivityReconciliationReduction::IgnoredStale { index });
            }
            ActivityReduction::Inserted { .. } | ActivityReduction::Updated { .. } => {}
        }

        {
            let group = &mut self.groups[index];
            if is_new_source {
                group.sources.push(source_id.clone());
                group.provenance.insert(source_id.clone(), provenance);
                self.source_bindings
                    .insert(source_id.clone(), correlation_id);
            }
            group.context = merge_context(
                &group.context,
                &self
                    .source_reducer
                    .item(&source_id)
                    .ok_or(ActivityReconciliationError::InconsistentSourceState)?
                    .context,
            )?;
            group.visibility = most_restrictive_visibility(
                group.visibility,
                self.source_reducer
                    .item(&source_id)
                    .ok_or(ActivityReconciliationError::InconsistentSourceState)?
                    .visibility,
            );
        }

        let winner_id = select_winner(&self.groups[index], &self.source_reducer)
            .ok_or(ActivityReconciliationError::InconsistentSourceState)?;
        let mut candidate = self
            .source_reducer
            .item(&winner_id)
            .cloned()
            .ok_or(ActivityReconciliationError::InconsistentSourceState)?;
        candidate.id = self.groups[index].canonical_id.clone();
        candidate.context = self.groups[index].context.clone();
        candidate.visibility = self.groups[index].visibility;

        let current_display_version = self.groups[index].display_version;
        candidate.source_version = current_display_version;
        if candidate == self.items[index] {
            return Ok(if is_new_source {
                ActivityReconciliationReduction::ProvenanceAdded { index }
            } else {
                ActivityReconciliationReduction::SourceUpdated { index }
            });
        }

        let next_display_version = current_display_version + 1;
        candidate.source_version = next_display_version;
        self.groups[index].display_version = next_display_version;
        self.items[index] = candidate;
        Ok(ActivityReconciliationReduction::Updated { index })
    }
}

fn validate_actor(
    group: &ReconciliationGroup,
    item: &ActivityItem,
) -> Result<(), ActivityReconciliationError> {
    if group.actor_id != item.actor.id || group.actor_kind != item.actor.kind {
        return Err(ActivityReconciliationError::ActorConflict {
            canonical_id: group.canonical_id.clone(),
        });
    }
    Ok(())
}

fn merge_context(
    current: &ActivityContext,
    incoming: &ActivityContext,
) -> Result<ActivityContext, ActivityReconciliationError> {
    Ok(ActivityContext {
        community_id: merge_context_field(
            "community_id",
            &current.community_id,
            &incoming.community_id,
        )?,
        project_id: merge_context_field("project_id", &current.project_id, &incoming.project_id)?,
        thread_id: merge_context_field("thread_id", &current.thread_id, &incoming.thread_id)?,
        session_id: merge_context_field("session_id", &current.session_id, &incoming.session_id)?,
    })
}

fn merge_context_field(
    field: &'static str,
    current: &Option<String>,
    incoming: &Option<String>,
) -> Result<Option<String>, ActivityReconciliationError> {
    match (current, incoming) {
        (Some(current), Some(incoming)) if current != incoming => {
            Err(ActivityReconciliationError::ContextConflict { field })
        }
        (Some(current), _) => Ok(Some(current.clone())),
        (_, Some(incoming)) => Ok(Some(incoming.clone())),
        (None, None) => Ok(None),
    }
}

fn select_winner(group: &ReconciliationGroup, reducer: &ActivityReducer) -> Option<ActivityItemId> {
    group
        .sources
        .iter()
        .filter_map(|source_id| {
            let item = reducer.item(source_id)?;
            let provenance = group.provenance.get(source_id)?;
            Some((winner_key(item, *provenance), source_id))
        })
        .max_by(|(left, _), (right, _)| left.cmp(right))
        .map(|(_, source_id)| source_id.clone())
}

fn winner_key(
    item: &ActivityItem,
    provenance: ActivityProvenanceClass,
) -> (
    u8,
    chrono::DateTime<chrono::Utc>,
    chrono::DateTime<chrono::Utc>,
    u8,
    u8,
    u8,
    &str,
) {
    let outcome_tier =
        if provenance == ActivityProvenanceClass::Authoritative && item.lifecycle.is_terminal() {
            3
        } else if item.lifecycle.is_terminal() {
            2
        } else {
            1
        };
    (
        outcome_tier,
        item.occurred_at,
        item.projected_at,
        lifecycle_urgency(item.lifecycle),
        provenance_rank(provenance),
        source_kind_rank(item.id.source_kind()),
        item.id.source_id(),
    )
}

const fn lifecycle_urgency(lifecycle: ActivityLifecycle) -> u8 {
    match lifecycle {
        ActivityLifecycle::WaitingForUser | ActivityLifecycle::Disconnected => 2,
        ActivityLifecycle::Idle => 1,
        _ => 0,
    }
}

const fn provenance_rank(provenance: ActivityProvenanceClass) -> u8 {
    match provenance {
        ActivityProvenanceClass::Compatibility => 0,
        ActivityProvenanceClass::Streaming => 1,
        ActivityProvenanceClass::Authoritative => 2,
    }
}

const fn source_kind_rank(source_kind: ActivitySourceKind) -> u8 {
    match source_kind {
        ActivitySourceKind::Acp => 0,
        ActivitySourceKind::NativeAction => 1,
        ActivitySourceKind::Nostr => 2,
        ActivitySourceKind::Git => 3,
        ActivitySourceKind::Workflow => 4,
        ActivitySourceKind::Ci => 5,
        ActivitySourceKind::Moderation => 6,
        ActivitySourceKind::System => 7,
    }
}

fn most_restrictive_visibility(
    current: ActivityVisibility,
    incoming: ActivityVisibility,
) -> ActivityVisibility {
    if visibility_rank(incoming) > visibility_rank(current) {
        incoming
    } else {
        current
    }
}

const fn visibility_rank(visibility: ActivityVisibility) -> u8 {
    match visibility {
        ActivityVisibility::Public => 0,
        ActivityVisibility::Community => 1,
        ActivityVisibility::Project => 2,
        ActivityVisibility::Participants => 3,
        ActivityVisibility::Private => 4,
    }
}

#[derive(Debug)]
pub enum ActivityReconciliationError {
    InvalidCorrelationId,
    CorrelationDefinitionConflict {
        correlation_id: ActivityCorrelationId,
    },
    SourceBindingConflict {
        source_id: ActivityItemId,
        current: ActivityCorrelationId,
        incoming: ActivityCorrelationId,
    },
    CanonicalBindingConflict {
        canonical_id: ActivityItemId,
        current: ActivityCorrelationId,
        incoming: ActivityCorrelationId,
    },
    ProvenanceConflict {
        source_id: ActivityItemId,
    },
    ActorConflict {
        canonical_id: ActivityItemId,
    },
    ContextConflict {
        field: &'static str,
    },
    VersionExhausted,
    InconsistentSourceState,
    Source(ActivityReductionError),
}

impl fmt::Display for ActivityReconciliationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCorrelationId => formatter.write_str("activity correlation ID is invalid"),
            Self::CorrelationDefinitionConflict { correlation_id } => write!(
                formatter,
                "activity correlation {} changed its canonical identity",
                correlation_id.as_str()
            ),
            Self::SourceBindingConflict { .. } => {
                formatter.write_str("activity source is already bound to another correlation")
            }
            Self::CanonicalBindingConflict { .. } => {
                formatter.write_str("canonical activity is already bound to another correlation")
            }
            Self::ProvenanceConflict { .. } => {
                formatter.write_str("activity source changed provenance class")
            }
            Self::ActorConflict { .. } => {
                formatter.write_str("correlated activity actors do not match")
            }
            Self::ContextConflict { field } => {
                write!(formatter, "correlated activity {field} values conflict")
            }
            Self::VersionExhausted => {
                formatter.write_str("reconciled activity version is exhausted")
            }
            Self::InconsistentSourceState => {
                formatter.write_str("activity reconciliation source state is inconsistent")
            }
            Self::Source(error) => error.fmt(formatter),
        }
    }
}

impl Error for ActivityReconciliationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Source(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ActivityReductionError> for ActivityReconciliationError {
    fn from(error: ActivityReductionError) -> Self {
        Self::Source(error)
    }
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone as _, Utc};

    use crate::activity_projection::{
        ActivityActor, ActivityObject, ActivityObjectKind, ActivityOutcome, ActivityOutcomeStatus,
        ActivitySemanticClass,
    };

    use super::*;

    fn correlation(value: &str) -> ActivityCorrelationId {
        ActivityCorrelationId::new(value).expect("valid correlation ID")
    }

    fn canonical(value: &str) -> ActivityItemId {
        ActivityItemId::new(ActivitySourceKind::Acp, value).expect("valid canonical ID")
    }

    fn item(
        source_kind: ActivitySourceKind,
        source_id: &str,
        source_version: u64,
        lifecycle: ActivityLifecycle,
        second: u32,
    ) -> ActivityItem {
        let timestamp = Utc
            .with_ymd_and_hms(2026, 8, 24, 12, 0, second)
            .single()
            .expect("valid timestamp");
        let outcome_status = match lifecycle {
            ActivityLifecycle::Succeeded => ActivityOutcomeStatus::Success,
            ActivityLifecycle::Failed => ActivityOutcomeStatus::Failure,
            ActivityLifecycle::Cancelled => ActivityOutcomeStatus::Cancelled,
            ActivityLifecycle::TimedOut => ActivityOutcomeStatus::TimedOut,
            ActivityLifecycle::Idle | ActivityLifecycle::Suppressed => {
                ActivityOutcomeStatus::NoChange
            }
            ActivityLifecycle::Disconnected => ActivityOutcomeStatus::Failure,
            ActivityLifecycle::Pending
            | ActivityLifecycle::Running
            | ActivityLifecycle::WaitingForUser => ActivityOutcomeStatus::Pending,
        };
        ActivityItem {
            id: ActivityItemId::new(source_kind, source_id).expect("valid source ID"),
            source_version,
            class: ActivitySemanticClass::Lifecycle,
            actor: ActivityActor {
                kind: ActivityActorKind::Agent,
                id: "agent-1".into(),
                label: "Agent".into(),
            },
            verb: format!("{lifecycle:?}"),
            object: ActivityObject {
                kind: ActivityObjectKind::Session,
                id: Some("turn-1".into()),
                label: "agent turn".into(),
            },
            outcome: ActivityOutcome {
                status: outcome_status,
                summary: None,
            },
            lifecycle,
            occurred_at: timestamp,
            projected_at: timestamp,
            context: ActivityContext {
                community_id: Some("community-1".into()),
                project_id: Some("project-1".into()),
                thread_id: None,
                session_id: Some("session-1".into()),
            },
            visibility: ActivityVisibility::Project,
            details: None,
            links: Vec::new(),
        }
    }

    fn input(
        correlation_id: &str,
        canonical_id: &str,
        provenance: ActivityProvenanceClass,
        item: ActivityItem,
    ) -> ActivityReconciliationInput {
        ActivityReconciliationInput {
            correlation_id: correlation(correlation_id),
            canonical_id: canonical(canonical_id),
            provenance,
            item,
        }
    }

    fn permutations<T: Clone>(values: &[T]) -> Vec<Vec<T>> {
        fn visit<T: Clone>(remaining: Vec<T>, prefix: Vec<T>, output: &mut Vec<Vec<T>>) {
            if remaining.is_empty() {
                output.push(prefix);
                return;
            }
            for index in 0..remaining.len() {
                let mut next_remaining = remaining.clone();
                let value = next_remaining.remove(index);
                let mut next_prefix = prefix.clone();
                next_prefix.push(value);
                visit(next_remaining, next_prefix, output);
            }
        }

        let mut output = Vec::new();
        visit(values.to_vec(), Vec::new(), &mut output);
        output
    }

    #[test]
    fn reconciliation_is_invariant_to_cross_source_reordering() {
        let updates = [
            input(
                "turn-1",
                "canonical-turn-1",
                ActivityProvenanceClass::Streaming,
                item(
                    ActivitySourceKind::Acp,
                    "stream-turn-1",
                    1,
                    ActivityLifecycle::Running,
                    1,
                ),
            ),
            input(
                "turn-1",
                "canonical-turn-1",
                ActivityProvenanceClass::Compatibility,
                item(
                    ActivitySourceKind::Nostr,
                    "observer-turn-1",
                    4,
                    ActivityLifecycle::TimedOut,
                    3,
                ),
            ),
            input(
                "turn-1",
                "canonical-turn-1",
                ActivityProvenanceClass::Authoritative,
                item(
                    ActivitySourceKind::System,
                    "authority-turn-1",
                    1,
                    ActivityLifecycle::Succeeded,
                    2,
                ),
            ),
        ];
        for permutation in permutations(&updates) {
            let mut reconciler = ActivityReconciler::new();
            for update in permutation {
                reconciler.reduce(update).expect("update should reconcile");
            }
            let item = reconciler
                .item(&correlation("turn-1"))
                .expect("correlated item");
            assert_eq!(item.id, canonical("canonical-turn-1"));
            assert_eq!(item.lifecycle, ActivityLifecycle::Succeeded);
            assert_eq!(reconciler.items().len(), 1);
        }
    }

    #[test]
    fn duplicate_and_reordered_source_frames_do_not_add_rows() {
        let running = input(
            "turn-1",
            "canonical-turn-1",
            ActivityProvenanceClass::Streaming,
            item(
                ActivitySourceKind::Acp,
                "stream-turn-1",
                2,
                ActivityLifecycle::Running,
                2,
            ),
        );
        let stale = input(
            "turn-1",
            "canonical-turn-1",
            ActivityProvenanceClass::Streaming,
            item(
                ActivitySourceKind::Acp,
                "stream-turn-1",
                1,
                ActivityLifecycle::Pending,
                1,
            ),
        );
        let mut reconciler = ActivityReconciler::new();
        reconciler
            .reduce(running.clone())
            .expect("running frame should insert");
        assert!(matches!(
            reconciler.reduce(running),
            Ok(ActivityReconciliationReduction::Duplicate { index: 0 })
        ));
        assert!(matches!(
            reconciler.reduce(stale),
            Ok(ActivityReconciliationReduction::IgnoredStale { index: 0 })
        ));
        assert_eq!(reconciler.items().len(), 1);
        assert_eq!(
            reconciler
                .provenance(&correlation("turn-1"))
                .expect("provenance")
                .len(),
            1
        );
    }

    #[test]
    fn timeout_disconnect_resume_and_late_authority_remain_visible() {
        let mut timeout = ActivityReconciler::new();
        timeout
            .reduce(input(
                "turn-timeout",
                "canonical-timeout",
                ActivityProvenanceClass::Streaming,
                item(
                    ActivitySourceKind::Acp,
                    "stream-timeout",
                    1,
                    ActivityLifecycle::Running,
                    1,
                ),
            ))
            .expect("running should insert");
        timeout
            .reduce(input(
                "turn-timeout",
                "canonical-timeout",
                ActivityProvenanceClass::Compatibility,
                item(
                    ActivitySourceKind::Nostr,
                    "observer-timeout",
                    2,
                    ActivityLifecycle::TimedOut,
                    3,
                ),
            ))
            .expect("timeout should reconcile");
        assert_eq!(timeout.items()[0].lifecycle, ActivityLifecycle::TimedOut);
        timeout
            .reduce(input(
                "turn-timeout",
                "canonical-timeout",
                ActivityProvenanceClass::Authoritative,
                item(
                    ActivitySourceKind::System,
                    "authority-timeout",
                    1,
                    ActivityLifecycle::Succeeded,
                    2,
                ),
            ))
            .expect("late authoritative terminal should supersede timeout");
        assert_eq!(timeout.items()[0].lifecycle, ActivityLifecycle::Succeeded);

        let mut connection = ActivityReconciler::new();
        for (source_id, provenance, lifecycle, second) in [
            (
                "stream-connection",
                ActivityProvenanceClass::Streaming,
                ActivityLifecycle::Running,
                1,
            ),
            (
                "observer-disconnected",
                ActivityProvenanceClass::Compatibility,
                ActivityLifecycle::Disconnected,
                2,
            ),
            (
                "authority-resumed",
                ActivityProvenanceClass::Authoritative,
                ActivityLifecycle::Running,
                3,
            ),
        ] {
            connection
                .reduce(input(
                    "connection-1",
                    "canonical-connection",
                    provenance,
                    item(
                        match provenance {
                            ActivityProvenanceClass::Streaming => ActivitySourceKind::Acp,
                            ActivityProvenanceClass::Compatibility => ActivitySourceKind::Nostr,
                            ActivityProvenanceClass::Authoritative => ActivitySourceKind::System,
                        },
                        source_id,
                        1,
                        lifecycle,
                        second,
                    ),
                ))
                .expect("connection update should reconcile");
            assert_eq!(connection.items()[0].lifecycle, lifecycle);
        }
    }

    #[test]
    fn explicit_provenance_keeps_similar_actions_distinct_and_private() {
        let mut reconciler = ActivityReconciler::new();
        for (correlation_id, canonical_id, source_id) in [
            ("turn-1", "canonical-turn-1", "stream-turn-1"),
            ("turn-2", "canonical-turn-2", "stream-turn-2"),
        ] {
            reconciler
                .reduce(input(
                    correlation_id,
                    canonical_id,
                    ActivityProvenanceClass::Streaming,
                    item(
                        ActivitySourceKind::Acp,
                        source_id,
                        1,
                        ActivityLifecycle::Running,
                        1,
                    ),
                ))
                .expect("distinct action should insert");
        }
        assert_eq!(reconciler.items().len(), 2);

        let mut private_observer = item(
            ActivitySourceKind::Nostr,
            "observer-turn-1",
            1,
            ActivityLifecycle::WaitingForUser,
            2,
        );
        private_observer.visibility = ActivityVisibility::Private;
        reconciler
            .reduce(input(
                "turn-1",
                "canonical-turn-1",
                ActivityProvenanceClass::Compatibility,
                private_observer,
            ))
            .expect("private observer should reconcile");
        assert_eq!(
            reconciler.items()[0].visibility,
            ActivityVisibility::Private
        );

        let conflict = input(
            "turn-2",
            "canonical-turn-2",
            ActivityProvenanceClass::Streaming,
            item(
                ActivitySourceKind::Acp,
                "stream-turn-1",
                2,
                ActivityLifecycle::Running,
                3,
            ),
        );
        assert!(matches!(
            reconciler.reduce(conflict),
            Err(ActivityReconciliationError::SourceBindingConflict { .. })
        ));

        let canonical_source_collision = input(
            "turn-2",
            "stream-turn-1",
            ActivityProvenanceClass::Streaming,
            item(
                ActivitySourceKind::System,
                "another-source",
                1,
                ActivityLifecycle::Running,
                3,
            ),
        );
        assert!(matches!(
            reconciler.reduce(canonical_source_collision),
            Err(ActivityReconciliationError::CanonicalBindingConflict { .. })
        ));
        assert_eq!(reconciler.items().len(), 2);
    }
}
