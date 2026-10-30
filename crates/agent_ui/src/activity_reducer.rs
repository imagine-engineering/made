use std::{collections::HashMap, error::Error, fmt};

use crate::activity_projection::{
    ActivityItem, ActivityItemId, ActivityLifecycle, ActivityOutcomeStatus,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActivityReduction {
    Inserted { index: usize },
    Updated { index: usize },
    Duplicate { index: usize },
    IgnoredStale { index: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ActivityReductionError {
    ConflictingVersion {
        id: ActivityItemId,
        source_version: u64,
    },
    TerminalRegression {
        id: ActivityItemId,
        from: ActivityLifecycle,
        to: ActivityLifecycle,
    },
    ConflictingTerminalOutcome {
        id: ActivityItemId,
        from_lifecycle: ActivityLifecycle,
        to_lifecycle: ActivityLifecycle,
        from_outcome: ActivityOutcomeStatus,
        to_outcome: ActivityOutcomeStatus,
    },
}

impl fmt::Display for ActivityReductionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConflictingVersion { id, source_version } => write!(
                formatter,
                "activity source {:?} produced different payloads at version {source_version}",
                id
            ),
            Self::TerminalRegression { id, from, to } => write!(
                formatter,
                "activity source {:?} cannot regress from terminal {from:?} to {to:?}",
                id
            ),
            Self::ConflictingTerminalOutcome {
                id,
                from_lifecycle,
                to_lifecycle,
                from_outcome,
                to_outcome,
            } => write!(
                formatter,
                "activity source {:?} cannot change terminal outcome from {from_lifecycle:?}/{from_outcome:?} to {to_lifecycle:?}/{to_outcome:?}",
                id
            ),
        }
    }
}

impl Error for ActivityReductionError {}

#[derive(Default)]
pub struct ActivityReducer {
    items: Vec<ActivityItem>,
    indices: HashMap<ActivityItemId, usize>,
}

impl ActivityReducer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn items(&self) -> &[ActivityItem] {
        &self.items
    }

    pub fn item(&self, id: &ActivityItemId) -> Option<&ActivityItem> {
        self.indices.get(id).map(|index| &self.items[*index])
    }

    pub fn reduce(
        &mut self,
        incoming: ActivityItem,
    ) -> Result<ActivityReduction, ActivityReductionError> {
        let Some(index) = self.indices.get(&incoming.id).copied() else {
            let index = self.items.len();
            self.indices.insert(incoming.id.clone(), index);
            self.items.push(incoming);
            return Ok(ActivityReduction::Inserted { index });
        };

        let current = &self.items[index];
        if incoming.source_version < current.source_version {
            return Ok(ActivityReduction::IgnoredStale { index });
        }
        if incoming.source_version == current.source_version {
            return if incoming == *current {
                Ok(ActivityReduction::Duplicate { index })
            } else {
                Err(ActivityReductionError::ConflictingVersion {
                    id: incoming.id,
                    source_version: incoming.source_version,
                })
            };
        }

        validate_lifecycle_transition(current, &incoming)?;
        self.items[index] = incoming;
        Ok(ActivityReduction::Updated { index })
    }
}

fn validate_lifecycle_transition(
    current: &ActivityItem,
    incoming: &ActivityItem,
) -> Result<(), ActivityReductionError> {
    if !current.lifecycle.is_terminal() {
        return Ok(());
    }
    if !incoming.lifecycle.is_terminal() {
        return Err(ActivityReductionError::TerminalRegression {
            id: incoming.id.clone(),
            from: current.lifecycle,
            to: incoming.lifecycle,
        });
    }
    if current.lifecycle != incoming.lifecycle || current.outcome.status != incoming.outcome.status
    {
        return Err(ActivityReductionError::ConflictingTerminalOutcome {
            id: incoming.id.clone(),
            from_lifecycle: current.lifecycle,
            to_lifecycle: incoming.lifecycle,
            from_outcome: current.outcome.status,
            to_outcome: incoming.outcome.status,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone as _, Utc};

    use super::*;
    use crate::activity_projection::{
        ActivityActor, ActivityActorKind, ActivityContext, ActivityObject, ActivityObjectKind,
        ActivityOutcome, ActivityOutcomeStatus, ActivitySemanticClass, ActivitySourceKind,
        ActivityVisibility,
    };

    fn item(
        source_id: &str,
        source_version: u64,
        lifecycle: ActivityLifecycle,
        label: &str,
    ) -> ActivityItem {
        let timestamp = Utc
            .with_ymd_and_hms(2026, 8, 14, 12, 0, 0)
            .single()
            .expect("test timestamp should be valid");
        let outcome_status = match lifecycle {
            ActivityLifecycle::Succeeded => ActivityOutcomeStatus::Success,
            ActivityLifecycle::Failed => ActivityOutcomeStatus::Failure,
            ActivityLifecycle::Cancelled => ActivityOutcomeStatus::Cancelled,
            ActivityLifecycle::TimedOut => ActivityOutcomeStatus::TimedOut,
            ActivityLifecycle::Pending
            | ActivityLifecycle::Running
            | ActivityLifecycle::WaitingForUser => ActivityOutcomeStatus::Pending,
            ActivityLifecycle::Idle => ActivityOutcomeStatus::NoChange,
            ActivityLifecycle::Disconnected => ActivityOutcomeStatus::Failure,
            ActivityLifecycle::Suppressed => ActivityOutcomeStatus::NoChange,
        };
        ActivityItem {
            id: ActivityItemId::new(ActivitySourceKind::Acp, source_id)
                .expect("test source id should be valid"),
            source_version,
            class: ActivitySemanticClass::Message,
            actor: ActivityActor {
                kind: ActivityActorKind::Agent,
                id: "agent-1".into(),
                label: "Agent".into(),
            },
            verb: "said".into(),
            object: ActivityObject {
                kind: ActivityObjectKind::Message,
                id: None,
                label: label.into(),
            },
            outcome: ActivityOutcome {
                status: outcome_status,
                summary: None,
            },
            lifecycle,
            occurred_at: timestamp,
            projected_at: timestamp,
            context: ActivityContext::default(),
            visibility: ActivityVisibility::Private,
            details: None,
            links: Vec::new(),
        }
    }

    #[test]
    fn activity_reducer_deduplicates_identical_source_versions() {
        let mut reducer = ActivityReducer::new();
        let update = item("message-1", 1, ActivityLifecycle::Running, "hello");
        assert_eq!(
            reducer.reduce(update.clone()),
            Ok(ActivityReduction::Inserted { index: 0 })
        );
        assert_eq!(
            reducer.reduce(update),
            Ok(ActivityReduction::Duplicate { index: 0 })
        );
        assert_eq!(reducer.items().len(), 1);
    }

    #[test]
    fn activity_reducer_ignores_reordered_stale_updates() {
        let mut reducer = ActivityReducer::new();
        reducer
            .reduce(item("message-1", 1, ActivityLifecycle::Running, "h"))
            .expect("initial update should insert");
        reducer
            .reduce(item("message-1", 3, ActivityLifecycle::Running, "hello"))
            .expect("newer update should replace");
        assert_eq!(
            reducer.reduce(item("message-1", 2, ActivityLifecycle::Running, "hel",)),
            Ok(ActivityReduction::IgnoredStale { index: 0 })
        );
        assert_eq!(reducer.items()[0].source_version, 3);
        assert_eq!(reducer.items()[0].object.label, "hello");
    }

    #[test]
    fn activity_reducer_replaces_streaming_fragments_in_place() {
        let mut reducer = ActivityReducer::new();
        for (source_version, label) in [(1, "h"), (2, "hel"), (3, "hello")] {
            reducer
                .reduce(item(
                    "message-1",
                    source_version,
                    ActivityLifecycle::Running,
                    label,
                ))
                .expect("streaming update should reduce");
        }
        assert_eq!(reducer.items().len(), 1);
        assert_eq!(reducer.items()[0].object.label, "hello");
    }

    #[test]
    fn activity_reducer_prevents_cancelled_item_resurrection() {
        let mut reducer = ActivityReducer::new();
        reducer
            .reduce(item(
                "command-1",
                1,
                ActivityLifecycle::Running,
                "cargo test",
            ))
            .expect("running action should insert");
        reducer
            .reduce(item(
                "command-1",
                2,
                ActivityLifecycle::Cancelled,
                "cargo test",
            ))
            .expect("cancellation should update");
        let error = reducer
            .reduce(item(
                "command-1",
                3,
                ActivityLifecycle::Running,
                "cargo test",
            ))
            .expect_err("terminal action must not restart in place");
        assert!(matches!(
            error,
            ActivityReductionError::TerminalRegression { .. }
        ));
        assert_eq!(reducer.items().len(), 1);
        assert_eq!(reducer.items()[0].lifecycle, ActivityLifecycle::Cancelled);
    }

    #[test]
    fn activity_reducer_keeps_one_timed_out_terminal_item() {
        let mut reducer = ActivityReducer::new();
        reducer
            .reduce(item(
                "command-1",
                1,
                ActivityLifecycle::Running,
                "cargo test",
            ))
            .expect("running action should insert");
        reducer
            .reduce(item(
                "command-1",
                2,
                ActivityLifecycle::TimedOut,
                "cargo test",
            ))
            .expect("timeout should update");
        reducer
            .reduce(item(
                "command-1",
                3,
                ActivityLifecycle::TimedOut,
                "cargo test timed out after 60s",
            ))
            .expect("same terminal outcome may gain detail");
        let mut inconsistent_timeout = item(
            "command-1",
            4,
            ActivityLifecycle::TimedOut,
            "cargo test timed out",
        );
        inconsistent_timeout.outcome.status = ActivityOutcomeStatus::Failure;
        let error = reducer
            .reduce(inconsistent_timeout)
            .expect_err("terminal lifecycle and outcome status must remain consistent");
        assert!(matches!(
            error,
            ActivityReductionError::ConflictingTerminalOutcome { .. }
        ));
        let error = reducer
            .reduce(item(
                "command-1",
                5,
                ActivityLifecycle::Failed,
                "cargo test failed",
            ))
            .expect_err("terminal outcome must not change");
        assert!(matches!(
            error,
            ActivityReductionError::ConflictingTerminalOutcome { .. }
        ));
        assert_eq!(reducer.items().len(), 1);
        assert_eq!(reducer.items()[0].source_version, 3);
        assert_eq!(reducer.items()[0].lifecycle, ActivityLifecycle::TimedOut);
    }

    #[test]
    fn activity_reducer_rejects_conflicting_same_version_payloads() {
        let mut reducer = ActivityReducer::new();
        reducer
            .reduce(item("message-1", 1, ActivityLifecycle::Running, "first"))
            .expect("initial update should insert");
        let error = reducer
            .reduce(item(
                "message-1",
                1,
                ActivityLifecycle::Running,
                "different",
            ))
            .expect_err("same version must be deterministic");
        assert!(matches!(
            error,
            ActivityReductionError::ConflictingVersion { .. }
        ));
        assert_eq!(reducer.items()[0].object.label, "first");
    }
}
