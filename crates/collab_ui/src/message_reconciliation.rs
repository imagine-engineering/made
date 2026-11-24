use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MessageDeliveryState<EventId, Rejection> {
    Pending { attempt: u32 },
    Accepted { event_id: EventId },
    Rejected { reason: Rejection },
    Reconciled { event_id: EventId },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MessageReconciliationAction<OperationId, EventId> {
    InsertOptimistic {
        operation_id: OperationId,
    },
    RetryOptimistic {
        operation_id: OperationId,
        attempt: u32,
    },
    MarkRejected {
        operation_id: OperationId,
    },
    ReplaceOptimistic {
        operation_id: OperationId,
        event_id: EventId,
    },
    ReplaceAuthoritative {
        operation_id: OperationId,
        previous_event_id: EventId,
        event_id: EventId,
    },
    SuppressDuplicateEcho {
        operation_id: OperationId,
        event_id: EventId,
    },
    InsertAuthoritative {
        event_id: EventId,
    },
    Unchanged {
        operation_id: OperationId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MessageReconciliationError {
    DuplicateOperation,
    UnknownOperation,
    AttemptExhausted,
    EventOwnedByAnotherOperation,
}

impl std::fmt::Display for MessageReconciliationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::DuplicateOperation => "message operation is already pending",
            Self::UnknownOperation => "message operation is unknown",
            Self::AttemptExhausted => "message retry count is exhausted",
            Self::EventOwnedByAnotherOperation => {
                "authoritative message belongs to another operation"
            }
        })
    }
}

impl std::error::Error for MessageReconciliationError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageReconciler<OperationId, EventId, Rejection> {
    operations: BTreeMap<OperationId, MessageDeliveryState<EventId, Rejection>>,
    event_owners: BTreeMap<EventId, OperationId>,
}

impl<OperationId, EventId, Rejection> Default for MessageReconciler<OperationId, EventId, Rejection>
where
    OperationId: Clone + Ord,
    EventId: Clone + Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<OperationId, EventId, Rejection> MessageReconciler<OperationId, EventId, Rejection>
where
    OperationId: Clone + Ord,
    EventId: Clone + Ord,
{
    pub fn new() -> Self {
        Self {
            operations: BTreeMap::new(),
            event_owners: BTreeMap::new(),
        }
    }

    pub fn begin(
        &mut self,
        operation_id: OperationId,
    ) -> Result<MessageReconciliationAction<OperationId, EventId>, MessageReconciliationError> {
        if self.operations.contains_key(&operation_id) {
            return Err(MessageReconciliationError::DuplicateOperation);
        }
        self.operations.insert(
            operation_id.clone(),
            MessageDeliveryState::Pending { attempt: 1 },
        );
        Ok(MessageReconciliationAction::InsertOptimistic { operation_id })
    }

    pub fn retry(
        &mut self,
        operation_id: &OperationId,
    ) -> Result<MessageReconciliationAction<OperationId, EventId>, MessageReconciliationError> {
        let state = self
            .operations
            .get_mut(operation_id)
            .ok_or(MessageReconciliationError::UnknownOperation)?;
        let next_attempt = match state {
            MessageDeliveryState::Pending { attempt } => attempt
                .checked_add(1)
                .ok_or(MessageReconciliationError::AttemptExhausted)?,
            MessageDeliveryState::Rejected { .. } => 2,
            MessageDeliveryState::Accepted { .. } | MessageDeliveryState::Reconciled { .. } => {
                return Ok(MessageReconciliationAction::Unchanged {
                    operation_id: operation_id.clone(),
                });
            }
        };
        *state = MessageDeliveryState::Pending {
            attempt: next_attempt,
        };
        Ok(MessageReconciliationAction::RetryOptimistic {
            operation_id: operation_id.clone(),
            attempt: next_attempt,
        })
    }

    pub fn reject(
        &mut self,
        operation_id: &OperationId,
        reason: Rejection,
    ) -> Result<MessageReconciliationAction<OperationId, EventId>, MessageReconciliationError> {
        let state = self
            .operations
            .get_mut(operation_id)
            .ok_or(MessageReconciliationError::UnknownOperation)?;
        if matches!(
            state,
            MessageDeliveryState::Accepted { .. } | MessageDeliveryState::Reconciled { .. }
        ) {
            return Ok(MessageReconciliationAction::Unchanged {
                operation_id: operation_id.clone(),
            });
        }
        if matches!(state, MessageDeliveryState::Rejected { .. }) {
            return Ok(MessageReconciliationAction::Unchanged {
                operation_id: operation_id.clone(),
            });
        }
        *state = MessageDeliveryState::Rejected { reason };
        Ok(MessageReconciliationAction::MarkRejected {
            operation_id: operation_id.clone(),
        })
    }

    pub fn accept(
        &mut self,
        operation_id: &OperationId,
        event_id: EventId,
    ) -> Result<MessageReconciliationAction<OperationId, EventId>, MessageReconciliationError> {
        if !self.operations.contains_key(operation_id) {
            return Err(MessageReconciliationError::UnknownOperation);
        }
        self.claim_event(operation_id, &event_id)?;
        let state = self
            .operations
            .get_mut(operation_id)
            .ok_or(MessageReconciliationError::UnknownOperation)?;
        let action = match state {
            MessageDeliveryState::Pending { .. } | MessageDeliveryState::Rejected { .. } => {
                MessageReconciliationAction::ReplaceOptimistic {
                    operation_id: operation_id.clone(),
                    event_id: event_id.clone(),
                }
            }
            MessageDeliveryState::Accepted {
                event_id: previous_event_id,
            }
            | MessageDeliveryState::Reconciled {
                event_id: previous_event_id,
            } if previous_event_id != &event_id => {
                MessageReconciliationAction::ReplaceAuthoritative {
                    operation_id: operation_id.clone(),
                    previous_event_id: previous_event_id.clone(),
                    event_id: event_id.clone(),
                }
            }
            MessageDeliveryState::Accepted { .. } | MessageDeliveryState::Reconciled { .. } => {
                return Ok(MessageReconciliationAction::Unchanged {
                    operation_id: operation_id.clone(),
                });
            }
        };
        *state = MessageDeliveryState::Accepted { event_id };
        Ok(action)
    }

    pub fn observe_authoritative(
        &mut self,
        event_id: EventId,
        operation_id: Option<&OperationId>,
    ) -> Result<MessageReconciliationAction<OperationId, EventId>, MessageReconciliationError> {
        if let Some(owner) = self.event_owners.get(&event_id) {
            if operation_id.is_some_and(|operation_id| operation_id != owner) {
                return Err(MessageReconciliationError::EventOwnedByAnotherOperation);
            }
            let owner = owner.clone();
            if self
                .operations
                .get(&owner)
                .is_some_and(|state| matches!(state, MessageDeliveryState::Accepted { event_id: accepted } if accepted == &event_id))
            {
                self.operations.insert(
                    owner.clone(),
                    MessageDeliveryState::Reconciled {
                        event_id: event_id.clone(),
                    },
                );
            }
            return Ok(MessageReconciliationAction::SuppressDuplicateEcho {
                operation_id: owner,
                event_id,
            });
        }

        let Some(operation_id) = operation_id else {
            return Ok(MessageReconciliationAction::InsertAuthoritative { event_id });
        };
        self.claim_event(operation_id, &event_id)?;
        let previous =
            self.operations
                .get(operation_id)
                .and_then(|state| match state {
                    MessageDeliveryState::Accepted { event_id }
                    | MessageDeliveryState::Reconciled { event_id } => Some(event_id.clone()),
                    MessageDeliveryState::Pending { .. }
                    | MessageDeliveryState::Rejected { .. } => None,
                });
        let existed = self.operations.insert(
            operation_id.clone(),
            MessageDeliveryState::Reconciled {
                event_id: event_id.clone(),
            },
        );
        match (existed, previous) {
            (None, _) => Ok(MessageReconciliationAction::InsertAuthoritative { event_id }),
            (Some(_), Some(previous_event_id)) if previous_event_id != event_id => {
                Ok(MessageReconciliationAction::ReplaceAuthoritative {
                    operation_id: operation_id.clone(),
                    previous_event_id,
                    event_id,
                })
            }
            (Some(_), _) => Ok(MessageReconciliationAction::ReplaceOptimistic {
                operation_id: operation_id.clone(),
                event_id,
            }),
        }
    }

    pub fn state(
        &self,
        operation_id: &OperationId,
    ) -> Option<&MessageDeliveryState<EventId, Rejection>> {
        self.operations.get(operation_id)
    }

    pub fn forget(&mut self, operation_id: &OperationId) -> bool {
        if self.operations.remove(operation_id).is_none() {
            return false;
        }
        self.event_owners.retain(|_, owner| owner != operation_id);
        true
    }

    pub fn len(&self) -> usize {
        self.operations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    fn claim_event(
        &mut self,
        operation_id: &OperationId,
        event_id: &EventId,
    ) -> Result<(), MessageReconciliationError> {
        if self
            .event_owners
            .get(event_id)
            .is_some_and(|owner| owner != operation_id)
        {
            return Err(MessageReconciliationError::EventOwnedByAnotherOperation);
        }
        self.event_owners
            .insert(event_id.clone(), operation_id.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_preserves_one_optimistic_operation() {
        let mut reconciler = MessageReconciler::<u128, u8, &'static str>::new();
        assert_eq!(
            reconciler.begin(10),
            Ok(MessageReconciliationAction::InsertOptimistic { operation_id: 10 })
        );
        assert_eq!(
            reconciler.retry(&10),
            Ok(MessageReconciliationAction::RetryOptimistic {
                operation_id: 10,
                attempt: 2,
            })
        );
        assert_eq!(
            reconciler.state(&10),
            Some(&MessageDeliveryState::Pending { attempt: 2 })
        );
        assert_eq!(reconciler.len(), 1);
        assert_eq!(
            reconciler.begin(10),
            Err(MessageReconciliationError::DuplicateOperation)
        );
    }

    #[test]
    fn rejection_can_retry_with_the_same_operation_identity() {
        let mut reconciler = MessageReconciler::<u128, u8, &'static str>::new();
        reconciler.begin(20).expect("optimistic message");
        assert_eq!(
            reconciler.reject(&20, "denied"),
            Ok(MessageReconciliationAction::MarkRejected { operation_id: 20 })
        );
        assert_eq!(
            reconciler.state(&20),
            Some(&MessageDeliveryState::Rejected { reason: "denied" })
        );
        assert_eq!(
            reconciler.retry(&20),
            Ok(MessageReconciliationAction::RetryOptimistic {
                operation_id: 20,
                attempt: 2,
            })
        );
        assert_eq!(
            reconciler.state(&20),
            Some(&MessageDeliveryState::Pending { attempt: 2 })
        );

        assert_eq!(
            reconciler.observe_authoritative(2, Some(&20)),
            Ok(MessageReconciliationAction::ReplaceOptimistic {
                operation_id: 20,
                event_id: 2,
            })
        );
        assert_eq!(
            reconciler.state(&20),
            Some(&MessageDeliveryState::Reconciled { event_id: 2 })
        );
    }

    #[test]
    fn acceptance_replaces_local_state_and_suppresses_its_echo() {
        let mut reconciler = MessageReconciler::<u128, u8, &'static str>::new();
        reconciler.begin(30).expect("optimistic message");
        assert_eq!(
            reconciler.accept(&30, 3),
            Ok(MessageReconciliationAction::ReplaceOptimistic {
                operation_id: 30,
                event_id: 3,
            })
        );
        assert_eq!(
            reconciler.state(&30),
            Some(&MessageDeliveryState::Accepted { event_id: 3 })
        );
        assert_eq!(
            reconciler.observe_authoritative(3, None),
            Ok(MessageReconciliationAction::SuppressDuplicateEcho {
                operation_id: 30,
                event_id: 3,
            })
        );
        assert_eq!(
            reconciler.state(&30),
            Some(&MessageDeliveryState::Reconciled { event_id: 3 })
        );
        assert_eq!(
            reconciler.observe_authoritative(3, Some(&30)),
            Ok(MessageReconciliationAction::SuppressDuplicateEcho {
                operation_id: 30,
                event_id: 3,
            })
        );
        assert_eq!(reconciler.len(), 1);
    }

    #[test]
    fn reconnect_and_server_replacement_keep_one_operation_row() {
        let mut reconciler = MessageReconciler::<u128, u8, &'static str>::new();
        reconciler.begin(40).expect("optimistic message");
        reconciler.retry(&40).expect("uncertain retry");
        reconciler.accept(&40, 4).expect("accepted replacement");

        assert_eq!(
            reconciler.observe_authoritative(5, Some(&40)),
            Ok(MessageReconciliationAction::ReplaceAuthoritative {
                operation_id: 40,
                previous_event_id: 4,
                event_id: 5,
            })
        );
        assert_eq!(
            reconciler.state(&40),
            Some(&MessageDeliveryState::Reconciled { event_id: 5 })
        );
        for event_id in [4, 5] {
            assert_eq!(
                reconciler.observe_authoritative(event_id, None),
                Ok(MessageReconciliationAction::SuppressDuplicateEcho {
                    operation_id: 40,
                    event_id,
                })
            );
        }
        assert_eq!(reconciler.len(), 1);
        assert!(reconciler.forget(&40));
        assert!(reconciler.is_empty());
        assert_eq!(
            reconciler.observe_authoritative(5, None),
            Ok(MessageReconciliationAction::InsertAuthoritative { event_id: 5 })
        );
    }

    #[test]
    fn one_authoritative_event_cannot_claim_two_local_operations() {
        let mut reconciler = MessageReconciler::<u128, u8, &'static str>::new();
        reconciler.begin(50).expect("first optimistic message");
        reconciler.begin(51).expect("second optimistic message");
        reconciler.accept(&50, 5).expect("first accepted message");
        assert_eq!(
            reconciler.accept(&51, 5),
            Err(MessageReconciliationError::EventOwnedByAnotherOperation)
        );
        assert_eq!(
            reconciler.state(&51),
            Some(&MessageDeliveryState::Pending { attempt: 1 })
        );
    }
}
