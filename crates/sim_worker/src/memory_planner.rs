use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const MINIMUM_SAFETY_MARGIN_BYTES: u64 = 256 * 1024 * 1024;
pub const MAXIMUM_LOWER_MEMORY_RETRIES: u8 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryReservationKind {
    Weights,
    Patches,
    Workspace,
    Activations,
    Staging,
    Preview,
    Cache,
    Codec,
    Output,
    SafetyMargin,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MemoryReservation {
    pub kind: MemoryReservationKind,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct MemoryPlanRequest {
    pub weights_bytes: u64,
    pub patches_bytes: u64,
    pub workspace_bytes: u64,
    pub activations_bytes: u64,
    pub staging_bytes: u64,
    pub preview_bytes: u64,
    pub cache_bytes: u64,
    pub codec_bytes: u64,
    pub output_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MemoryPlan {
    pub capacity_bytes: u64,
    pub durable_baseline_bytes: u64,
    pub reservations: Vec<MemoryReservation>,
    pub workload_bytes: u64,
    pub safety_margin_bytes: u64,
    pub committed_target_bytes: u64,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum MemoryPlanError {
    #[error("memory accounting overflow while reserving {kind:?}")]
    Overflow { kind: MemoryReservationKind },
    #[error(
        "native memory plan requires {required_bytes} bytes after {kind:?}, but capacity is {capacity_bytes} bytes"
    )]
    OutOfMemory {
        kind: MemoryReservationKind,
        required_bytes: u64,
        capacity_bytes: u64,
    },
    #[error("retry target {proposed_bytes} must be lower than previous target {previous_bytes}")]
    RetryDidNotReduce {
        previous_bytes: u64,
        proposed_bytes: u64,
    },
    #[error("native memory retry budget of {MAXIMUM_LOWER_MEMORY_RETRIES} is exhausted")]
    RetryBudgetExhausted,
}

pub struct MemoryPlanner;

impl MemoryPlanner {
    pub fn plan(
        capacity_bytes: u64,
        durable_baseline_bytes: u64,
        request: MemoryPlanRequest,
    ) -> Result<MemoryPlan, MemoryPlanError> {
        let ordered = [
            (MemoryReservationKind::Weights, request.weights_bytes),
            (MemoryReservationKind::Patches, request.patches_bytes),
            (MemoryReservationKind::Workspace, request.workspace_bytes),
            (
                MemoryReservationKind::Activations,
                request.activations_bytes,
            ),
            (MemoryReservationKind::Staging, request.staging_bytes),
            (MemoryReservationKind::Preview, request.preview_bytes),
            (MemoryReservationKind::Cache, request.cache_bytes),
            (MemoryReservationKind::Codec, request.codec_bytes),
            (MemoryReservationKind::Output, request.output_bytes),
        ];

        let mut workload_bytes = 0_u64;
        let mut reservations = Vec::with_capacity(ordered.len() + 1);
        for (kind, bytes) in ordered {
            workload_bytes = workload_bytes
                .checked_add(bytes)
                .ok_or(MemoryPlanError::Overflow { kind })?;
            if bytes != 0 {
                reservations.push(MemoryReservation { kind, bytes });
            }
        }

        let safety_margin_bytes = workload_bytes.div_ceil(10).max(MINIMUM_SAFETY_MARGIN_BYTES);
        reservations.push(MemoryReservation {
            kind: MemoryReservationKind::SafetyMargin,
            bytes: safety_margin_bytes,
        });
        let committed_target_bytes = durable_baseline_bytes
            .checked_add(workload_bytes)
            .and_then(|total| total.checked_add(safety_margin_bytes))
            .ok_or(MemoryPlanError::Overflow {
                kind: MemoryReservationKind::SafetyMargin,
            })?;
        if committed_target_bytes > capacity_bytes {
            return Err(MemoryPlanError::OutOfMemory {
                kind: MemoryReservationKind::SafetyMargin,
                required_bytes: committed_target_bytes,
                capacity_bytes,
            });
        }

        Ok(MemoryPlan {
            capacity_bytes,
            durable_baseline_bytes,
            reservations,
            workload_bytes,
            safety_margin_bytes,
            committed_target_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MemoryRetryTracker {
    retries_used: u8,
    previous_target_bytes: u64,
}

impl MemoryRetryTracker {
    pub(super) fn new(initial_target_bytes: u64) -> Self {
        Self {
            retries_used: 0,
            previous_target_bytes: initial_target_bytes,
        }
    }

    pub(super) fn accept_lower_target(
        &mut self,
        proposed_bytes: u64,
    ) -> Result<u8, MemoryPlanError> {
        if self.retries_used >= MAXIMUM_LOWER_MEMORY_RETRIES {
            return Err(MemoryPlanError::RetryBudgetExhausted);
        }
        if proposed_bytes >= self.previous_target_bytes {
            return Err(MemoryPlanError::RetryDidNotReduce {
                previous_bytes: self.previous_target_bytes,
                proposed_bytes,
            });
        }
        self.retries_used = self
            .retries_used
            .checked_add(1)
            .ok_or(MemoryPlanError::RetryBudgetExhausted)?;
        self.previous_target_bytes = proposed_bytes;
        Ok(self.retries_used)
    }

    pub(super) fn retries_used(self) -> u8 {
        self.retries_used
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_orders_immutable_and_mandatory_reservations_first() {
        let plan = MemoryPlanner::plan(
            2 * 1024 * 1024 * 1024,
            64,
            MemoryPlanRequest {
                weights_bytes: 100,
                patches_bytes: 50,
                workspace_bytes: 200,
                activations_bytes: 300,
                output_bytes: 400,
                ..MemoryPlanRequest::default()
            },
        )
        .expect("plan fits");
        let kinds: Vec<_> = plan
            .reservations
            .iter()
            .map(|reservation| reservation.kind)
            .collect();
        assert_eq!(
            kinds,
            [
                MemoryReservationKind::Weights,
                MemoryReservationKind::Patches,
                MemoryReservationKind::Workspace,
                MemoryReservationKind::Activations,
                MemoryReservationKind::Output,
                MemoryReservationKind::SafetyMargin,
            ]
        );
        assert_eq!(plan.safety_margin_bytes, MINIMUM_SAFETY_MARGIN_BYTES);
    }

    #[test]
    fn plan_reports_oom_without_partial_reservation() {
        let error =
            MemoryPlanner::plan(MINIMUM_SAFETY_MARGIN_BYTES, 1, MemoryPlanRequest::default());
        assert!(matches!(error, Err(MemoryPlanError::OutOfMemory { .. })));
    }

    #[test]
    fn plan_rejects_checked_overflow() {
        let error = MemoryPlanner::plan(
            u64::MAX,
            0,
            MemoryPlanRequest {
                weights_bytes: u64::MAX,
                workspace_bytes: 1,
                ..MemoryPlanRequest::default()
            },
        );
        assert_eq!(
            error,
            Err(MemoryPlanError::Overflow {
                kind: MemoryReservationKind::Workspace
            })
        );
    }

    #[test]
    fn retries_are_bounded_and_strictly_reduce_committed_bytes() {
        let mut retries = MemoryRetryTracker::new(1_000);
        assert_eq!(retries.accept_lower_target(900), Ok(1));
        assert_eq!(
            retries.accept_lower_target(900),
            Err(MemoryPlanError::RetryDidNotReduce {
                previous_bytes: 900,
                proposed_bytes: 900
            })
        );
        assert_eq!(retries.accept_lower_target(800), Ok(2));
        assert_eq!(
            retries.accept_lower_target(700),
            Err(MemoryPlanError::RetryBudgetExhausted)
        );
    }
}
