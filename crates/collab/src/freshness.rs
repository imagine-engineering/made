use std::time::{Duration, Instant};

use collaboration_domain::{CommunityId, TenantContext};
use uuid::Uuid;

const REQUIRED_HEALTHY_RECOVERY_SAMPLES: u8 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FreshnessLimits {
    pub heartbeat_stale_after: Duration,
    pub maximum_projection_lag: u64,
}

impl FreshnessLimits {
    pub fn new(
        heartbeat_stale_after: Duration,
        maximum_projection_lag: u64,
    ) -> Result<Self, FreshnessError> {
        if heartbeat_stale_after.is_zero() {
            return Err(FreshnessError::InvalidLimits);
        }
        Ok(Self {
            heartbeat_stale_after,
            maximum_projection_lag,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeartbeatObservation {
    pub epoch: Uuid,
    pub token: u64,
    pub observed_at: Instant,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplicaObservation {
    pub heartbeat: Option<HeartbeatObservation>,
    pub authoritative_cursor: u64,
    pub projection_cursor: u64,
    pub pubsub_available: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplicaFreshnessState {
    Healthy,
    Lagging,
    Disconnected,
    Recovering,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FreshnessIssue {
    MissingHeartbeat,
    StaleHeartbeat,
    HeartbeatEpochChanged,
    HeartbeatTokenRegressed,
    PubSubUnavailable,
    ProjectionLag,
    RecoveryConfirmationPending,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplicaFreshnessSnapshot {
    pub state: ReplicaFreshnessState,
    pub issue: Option<FreshnessIssue>,
    pub heartbeat_age: Option<Duration>,
    pub heartbeat_epoch: Option<Uuid>,
    pub heartbeat_token: Option<u64>,
    pub authoritative_cursor: u64,
    pub projection_cursor: u64,
    pub projection_lag: u64,
    pub pubsub_available: bool,
    pub last_trustworthy_cursor: Option<u64>,
}

pub struct ReplicaFreshnessTracker {
    community_id: CommunityId,
    limits: FreshnessLimits,
    heartbeat: Option<HeartbeatObservation>,
    last_trustworthy_cursor: Option<u64>,
    recovery_required: bool,
    healthy_recovery_samples: u8,
}

impl ReplicaFreshnessTracker {
    pub fn new(community_id: CommunityId, limits: FreshnessLimits) -> Self {
        Self {
            community_id,
            limits,
            heartbeat: None,
            last_trustworthy_cursor: None,
            recovery_required: false,
            healthy_recovery_samples: 0,
        }
    }

    pub fn observe(
        &mut self,
        tenant: &TenantContext,
        observation: ReplicaObservation,
        now: Instant,
    ) -> Result<ReplicaFreshnessSnapshot, FreshnessError> {
        if tenant.community_id() != self.community_id {
            return Err(FreshnessError::TenantMismatch);
        }
        if observation.projection_cursor > observation.authoritative_cursor {
            return Err(FreshnessError::InvalidObservation);
        }
        let heartbeat_update = self.update_heartbeat(observation.heartbeat, now)?;
        let heartbeat_age = self
            .heartbeat
            .map(|heartbeat| now.duration_since(heartbeat.observed_at));
        let projection_lag = observation
            .authoritative_cursor
            .saturating_sub(observation.projection_cursor);

        let disconnected_issue = match heartbeat_update {
            HeartbeatUpdate::EpochChanged => Some(FreshnessIssue::HeartbeatEpochChanged),
            HeartbeatUpdate::TokenRegressed => Some(FreshnessIssue::HeartbeatTokenRegressed),
            HeartbeatUpdate::Accepted => match heartbeat_age {
                None => Some(FreshnessIssue::MissingHeartbeat),
                Some(age) if age > self.limits.heartbeat_stale_after => {
                    Some(FreshnessIssue::StaleHeartbeat)
                }
                Some(_) if !observation.pubsub_available => Some(FreshnessIssue::PubSubUnavailable),
                Some(_) => None,
            },
        };

        let (state, issue) = if let Some(issue) = disconnected_issue {
            self.recovery_required = true;
            self.healthy_recovery_samples = 0;
            (ReplicaFreshnessState::Disconnected, Some(issue))
        } else if projection_lag > self.limits.maximum_projection_lag {
            self.healthy_recovery_samples = 0;
            (
                ReplicaFreshnessState::Lagging,
                Some(FreshnessIssue::ProjectionLag),
            )
        } else if self.recovery_required {
            self.healthy_recovery_samples = self.healthy_recovery_samples.saturating_add(1);
            if self.healthy_recovery_samples < REQUIRED_HEALTHY_RECOVERY_SAMPLES {
                (
                    ReplicaFreshnessState::Recovering,
                    Some(FreshnessIssue::RecoveryConfirmationPending),
                )
            } else {
                self.recovery_required = false;
                self.healthy_recovery_samples = 0;
                (ReplicaFreshnessState::Healthy, None)
            }
        } else {
            (ReplicaFreshnessState::Healthy, None)
        };

        if matches!(
            state,
            ReplicaFreshnessState::Healthy | ReplicaFreshnessState::Recovering
        ) {
            self.last_trustworthy_cursor = Some(
                self.last_trustworthy_cursor
                    .unwrap_or_default()
                    .max(observation.projection_cursor),
            );
        }

        Ok(ReplicaFreshnessSnapshot {
            state,
            issue,
            heartbeat_age,
            heartbeat_epoch: self.heartbeat.map(|heartbeat| heartbeat.epoch),
            heartbeat_token: self.heartbeat.map(|heartbeat| heartbeat.token),
            authoritative_cursor: observation.authoritative_cursor,
            projection_cursor: observation.projection_cursor,
            projection_lag,
            pubsub_available: observation.pubsub_available,
            last_trustworthy_cursor: self.last_trustworthy_cursor,
        })
    }

    fn update_heartbeat(
        &mut self,
        heartbeat: Option<HeartbeatObservation>,
        now: Instant,
    ) -> Result<HeartbeatUpdate, FreshnessError> {
        let Some(heartbeat) = heartbeat else {
            return Ok(HeartbeatUpdate::Accepted);
        };
        if heartbeat.epoch.is_nil() || heartbeat.token == 0 || heartbeat.observed_at > now {
            return Err(FreshnessError::InvalidObservation);
        }
        let Some(current) = self.heartbeat else {
            self.heartbeat = Some(heartbeat);
            return Ok(HeartbeatUpdate::Accepted);
        };
        if heartbeat.observed_at < current.observed_at {
            return Err(FreshnessError::InvalidObservation);
        }
        if heartbeat.epoch != current.epoch {
            self.heartbeat = Some(heartbeat);
            return Ok(HeartbeatUpdate::EpochChanged);
        }
        if heartbeat.token < current.token {
            return Ok(HeartbeatUpdate::TokenRegressed);
        }
        if heartbeat.token > current.token {
            self.heartbeat = Some(heartbeat);
        }
        Ok(HeartbeatUpdate::Accepted)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HeartbeatUpdate {
    Accepted,
    EpochChanged,
    TokenRegressed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum FreshnessError {
    #[error("replica freshness limits are invalid")]
    InvalidLimits,
    #[error("replica freshness observation is invalid")]
    InvalidObservation,
    #[error("replica freshness observation crossed its tenant boundary")]
    TenantMismatch,
}
