use std::{collections::HashSet, error::Error, fmt};

use crate::{
    collaborative_review::CollaborativeReviewSlot,
    collaborative_review_summary::CollaborativeReviewSummarySource,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CollaborativeReviewAction {
    Keep,
    Reject,
    Stage,
    Review,
}

impl CollaborativeReviewAction {
    fn is_valid_for(self, slot: CollaborativeReviewSlot) -> bool {
        matches!(
            (slot, self),
            (
                CollaborativeReviewSlot::AgentChanges,
                Self::Keep | Self::Reject
            ) | (
                CollaborativeReviewSlot::ProjectChanges,
                Self::Stage | Self::Review
            )
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborativeReviewActionState {
    Ready,
    Conflict,
    Rejected,
    Stale,
}

pub struct CollaborativeReviewActionContext {
    source: CollaborativeReviewSummarySource,
    state: CollaborativeReviewActionState,
    available_actions: HashSet<CollaborativeReviewAction>,
}

impl CollaborativeReviewActionContext {
    pub fn new(
        source: CollaborativeReviewSummarySource,
        state: CollaborativeReviewActionState,
        available_actions: impl IntoIterator<Item = CollaborativeReviewAction>,
    ) -> Self {
        Self {
            source,
            state,
            available_actions: available_actions.into_iter().collect(),
        }
    }

    pub fn source(&self) -> CollaborativeReviewSummarySource {
        self.source
    }

    pub fn state(&self) -> CollaborativeReviewActionState {
        self.state
    }

    pub fn is_available(&self, action: CollaborativeReviewAction) -> bool {
        action.is_valid_for(self.source.slot()) && self.available_actions.contains(&action)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollaborativeReviewActionRequest {
    source: CollaborativeReviewSummarySource,
    action: CollaborativeReviewAction,
}

impl CollaborativeReviewActionRequest {
    pub fn new(
        source: CollaborativeReviewSummarySource,
        action: CollaborativeReviewAction,
    ) -> Self {
        Self { source, action }
    }

    pub fn source(self) -> CollaborativeReviewSummarySource {
        self.source
    }

    pub fn action(self) -> CollaborativeReviewAction {
        self.action
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutedCollaborativeReviewAction {
    source: CollaborativeReviewSummarySource,
    action: CollaborativeReviewAction,
}

impl ExecutedCollaborativeReviewAction {
    pub fn source(self) -> CollaborativeReviewSummarySource {
        self.source
    }

    pub fn action(self) -> CollaborativeReviewAction {
        self.action
    }
}

pub fn route_collaborative_review_action(
    context: &CollaborativeReviewActionContext,
    request: CollaborativeReviewActionRequest,
    invoke_native: impl FnOnce(CollaborativeReviewAction) -> Result<(), String>,
) -> Result<ExecutedCollaborativeReviewAction, CollaborativeReviewActionError> {
    if request.source.slot() != context.source.slot()
        || request.source.provider_id() != context.source.provider_id()
    {
        return Err(CollaborativeReviewActionError::StaleProvider);
    }
    if request.source.revision() != context.source.revision() {
        return Err(CollaborativeReviewActionError::StaleRevision);
    }
    match context.state {
        CollaborativeReviewActionState::Ready => {}
        CollaborativeReviewActionState::Conflict => {
            return Err(CollaborativeReviewActionError::Conflict);
        }
        CollaborativeReviewActionState::Rejected => {
            return Err(CollaborativeReviewActionError::Rejected);
        }
        CollaborativeReviewActionState::Stale => {
            return Err(CollaborativeReviewActionError::StaleState);
        }
    }
    if !request.action.is_valid_for(context.source.slot()) {
        return Err(CollaborativeReviewActionError::InvalidForSlot);
    }
    if !context.available_actions.contains(&request.action) {
        return Err(CollaborativeReviewActionError::Unavailable);
    }

    invoke_native(request.action).map_err(CollaborativeReviewActionError::NativeFailure)?;
    Ok(ExecutedCollaborativeReviewAction {
        source: request.source,
        action: request.action,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollaborativeReviewActionError {
    StaleProvider,
    StaleRevision,
    Conflict,
    Rejected,
    StaleState,
    InvalidForSlot,
    Unavailable,
    NativeFailure(String),
}

impl fmt::Display for CollaborativeReviewActionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StaleProvider => {
                formatter.write_str("review action provider is no longer current")
            }
            Self::StaleRevision => {
                formatter.write_str("review action revision is no longer current")
            }
            Self::Conflict => formatter.write_str("review action is blocked by a conflict"),
            Self::Rejected => formatter.write_str("review action was rejected"),
            Self::StaleState => formatter.write_str("review action state is stale"),
            Self::InvalidForSlot => formatter.write_str("review action is invalid for this source"),
            Self::Unavailable => formatter.write_str("review action is unavailable"),
            Self::NativeFailure(message) => {
                write!(formatter, "native review action failed: {message}")
            }
        }
    }
}

impl Error for CollaborativeReviewActionError {}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc};

    use gpui::{AppContext as _, TestAppContext};

    use super::*;

    #[gpui::test]
    fn collaborative_review_actions(cx: &mut TestAppContext) {
        let (agent_provider, project_provider, replacement_provider) = cx.update(|cx| {
            (
                cx.new(|_| ()).entity_id(),
                cx.new(|_| ()).entity_id(),
                cx.new(|_| ()).entity_id(),
            )
        });
        let agent_source = CollaborativeReviewSummarySource::new(
            CollaborativeReviewSlot::AgentChanges,
            agent_provider,
            3,
        );
        let project_source = CollaborativeReviewSummarySource::new(
            CollaborativeReviewSlot::ProjectChanges,
            project_provider,
            7,
        );
        let invoked = Rc::new(Cell::new(0));

        let agent_context = CollaborativeReviewActionContext::new(
            agent_source,
            CollaborativeReviewActionState::Ready,
            [
                CollaborativeReviewAction::Keep,
                CollaborativeReviewAction::Reject,
            ],
        );
        let invoked_for_keep = invoked.clone();
        let executed = route_collaborative_review_action(
            &agent_context,
            CollaborativeReviewActionRequest::new(agent_source, CollaborativeReviewAction::Keep),
            move |action| {
                assert_eq!(action, CollaborativeReviewAction::Keep);
                invoked_for_keep.set(invoked_for_keep.get() + 1);
                Ok(())
            },
        )
        .expect("valid keep should invoke the native action");
        assert_eq!(executed.action(), CollaborativeReviewAction::Keep);
        assert_eq!(invoked.get(), 1);

        let project_context = CollaborativeReviewActionContext::new(
            project_source,
            CollaborativeReviewActionState::Ready,
            [
                CollaborativeReviewAction::Stage,
                CollaborativeReviewAction::Review,
            ],
        );
        for action in [
            CollaborativeReviewAction::Stage,
            CollaborativeReviewAction::Review,
        ] {
            let invoked = invoked.clone();
            route_collaborative_review_action(
                &project_context,
                CollaborativeReviewActionRequest::new(project_source, action),
                move |_| {
                    invoked.set(invoked.get() + 1);
                    Ok(())
                },
            )
            .expect("valid project action should invoke its native handler");
        }
        assert_eq!(invoked.get(), 3);

        let invalid_slot_invoked = Rc::new(Cell::new(false));
        let invalid_slot_flag = invalid_slot_invoked.clone();
        assert_eq!(
            route_collaborative_review_action(
                &agent_context,
                CollaborativeReviewActionRequest::new(
                    agent_source,
                    CollaborativeReviewAction::Stage,
                ),
                move |_| {
                    invalid_slot_flag.set(true);
                    Ok(())
                },
            ),
            Err(CollaborativeReviewActionError::InvalidForSlot)
        );
        assert!(!invalid_slot_invoked.get());

        let stale_source = CollaborativeReviewSummarySource::new(
            CollaborativeReviewSlot::AgentChanges,
            replacement_provider,
            3,
        );
        assert_eq!(
            route_collaborative_review_action(
                &agent_context,
                CollaborativeReviewActionRequest::new(
                    stale_source,
                    CollaborativeReviewAction::Keep,
                ),
                |_| Ok(()),
            ),
            Err(CollaborativeReviewActionError::StaleProvider)
        );
        let stale_revision = CollaborativeReviewSummarySource::new(
            CollaborativeReviewSlot::AgentChanges,
            agent_provider,
            2,
        );
        assert_eq!(
            route_collaborative_review_action(
                &agent_context,
                CollaborativeReviewActionRequest::new(
                    stale_revision,
                    CollaborativeReviewAction::Keep,
                ),
                |_| Ok(()),
            ),
            Err(CollaborativeReviewActionError::StaleRevision)
        );

        let unavailable_context = CollaborativeReviewActionContext::new(
            agent_source,
            CollaborativeReviewActionState::Ready,
            [CollaborativeReviewAction::Keep],
        );
        assert_eq!(
            route_collaborative_review_action(
                &unavailable_context,
                CollaborativeReviewActionRequest::new(
                    agent_source,
                    CollaborativeReviewAction::Reject,
                ),
                |_| Ok(()),
            ),
            Err(CollaborativeReviewActionError::Unavailable)
        );

        for (state, expected_error) in [
            (
                CollaborativeReviewActionState::Conflict,
                CollaborativeReviewActionError::Conflict,
            ),
            (
                CollaborativeReviewActionState::Rejected,
                CollaborativeReviewActionError::Rejected,
            ),
            (
                CollaborativeReviewActionState::Stale,
                CollaborativeReviewActionError::StaleState,
            ),
        ] {
            let context = CollaborativeReviewActionContext::new(
                agent_source,
                state,
                [CollaborativeReviewAction::Reject],
            );
            assert_eq!(
                route_collaborative_review_action(
                    &context,
                    CollaborativeReviewActionRequest::new(
                        agent_source,
                        CollaborativeReviewAction::Reject,
                    ),
                    |_| Ok(()),
                ),
                Err(expected_error)
            );
        }

        assert_eq!(
            route_collaborative_review_action(
                &agent_context,
                CollaborativeReviewActionRequest::new(
                    agent_source,
                    CollaborativeReviewAction::Reject,
                ),
                |_| Err("buffer changed during reject".into()),
            ),
            Err(CollaborativeReviewActionError::NativeFailure(
                "buffer changed during reject".into()
            ))
        );
    }
}
