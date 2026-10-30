use gpui::{Context, IntoElement, Render, Role, Window};
use remote::mesh::{
    advertisement::ProviderTrustClass,
    scheduler::{
        MeshCandidateIneligibility, MeshProviderSelection, MeshQueueReason, MeshScheduleOutcome,
    },
};
use thiserror::Error;
use ui::prelude::*;

const MAX_EXECUTION_LOCATION_LABEL_BYTES: usize = 128;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshComputeProviderDisplay {
    provider: MeshProviderSelection,
    location_label: String,
}

impl MeshComputeProviderDisplay {
    pub fn new(
        provider: MeshProviderSelection,
        location_label: impl Into<String>,
    ) -> Result<Self, MeshComputeUiError> {
        let location_label = location_label.into();
        let location_label = location_label.trim();
        if provider.community_id.as_uuid().is_nil()
            || provider.owner_principal_id.as_uuid().is_nil()
            || location_label.is_empty()
            || location_label.len() > MAX_EXECUTION_LOCATION_LABEL_BYTES
            || location_label.chars().any(char::is_control)
        {
            return Err(MeshComputeUiError::InvalidProviderDisplay);
        }
        Ok(Self {
            provider,
            location_label: location_label.to_string(),
        })
    }

    pub const fn provider(&self) -> &MeshProviderSelection {
        &self.provider
    }

    pub fn location_label(&self) -> &str {
        &self.location_label
    }

    fn trust_label(&self) -> &'static str {
        match self.provider.trust_class {
            ProviderTrustClass::DeploymentManaged => "Deployment-managed hardware",
            ProviderTrustClass::CommunityMemberOwned => "Community member hardware",
            ProviderTrustClass::ThirdParty => "Unapproved third-party hardware",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeshProviderFailureReason {
    Disconnected,
    ExecutionFailed,
    CapacityExceeded,
    UnknownOutcome,
}

impl MeshProviderFailureReason {
    const fn label(self) -> &'static str {
        match self {
            Self::Disconnected => "The shared-compute provider disconnected.",
            Self::ExecutionFailed => "The shared-compute provider reported a failure.",
            Self::CapacityExceeded => "The provider exceeded the approved resource lease.",
            Self::UnknownOutcome => {
                "The execution outcome is unknown. It was not retried on another provider."
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MeshComputeState {
    AwaitingAvailability,
    Available {
        provider: MeshComputeProviderDisplay,
        eligible_slots: u16,
    },
    Queued {
        provider: Option<MeshComputeProviderDisplay>,
        reason: MeshQueueReason,
    },
    Running {
        provider: MeshComputeProviderDisplay,
    },
    Stale {
        provider: MeshComputeProviderDisplay,
    },
    Revoked {
        provider: MeshComputeProviderDisplay,
    },
    NoCapacity {
        provider: Option<MeshComputeProviderDisplay>,
    },
    UnapprovedProvider {
        provider: Option<MeshComputeProviderDisplay>,
    },
    PolicyDenied {
        provider: Option<MeshComputeProviderDisplay>,
        reason: MeshCandidateIneligibility,
    },
    ProviderFailed {
        provider: MeshComputeProviderDisplay,
        reason: MeshProviderFailureReason,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeshComputeDisplayCopy {
    pub heading: String,
    pub detail: String,
    pub accessibility_label: String,
    pub is_alert: bool,
}

pub struct MeshComputeView {
    state: MeshComputeState,
}

impl Default for MeshComputeView {
    fn default() -> Self {
        Self {
            state: MeshComputeState::AwaitingAvailability,
        }
    }
}

impl MeshComputeView {
    pub fn state(&self) -> &MeshComputeState {
        &self.state
    }

    pub fn show_available(
        &mut self,
        provider: MeshComputeProviderDisplay,
        eligible_slots: u16,
        cx: &mut Context<Self>,
    ) -> Result<(), MeshComputeUiError> {
        if eligible_slots == 0 || provider.provider.trust_class == ProviderTrustClass::ThirdParty {
            return Err(MeshComputeUiError::InvalidAvailability);
        }
        self.state = MeshComputeState::Available {
            provider,
            eligible_slots,
        };
        cx.notify();
        Ok(())
    }

    pub fn show_stale(&mut self, provider: MeshComputeProviderDisplay, cx: &mut Context<Self>) {
        self.state = MeshComputeState::Stale { provider };
        cx.notify();
    }

    pub fn show_revoked(&mut self, provider: MeshComputeProviderDisplay, cx: &mut Context<Self>) {
        self.state = MeshComputeState::Revoked { provider };
        cx.notify();
    }

    pub fn show_provider_failure(
        &mut self,
        provider: MeshComputeProviderDisplay,
        reason: MeshProviderFailureReason,
        cx: &mut Context<Self>,
    ) {
        self.state = MeshComputeState::ProviderFailed { provider, reason };
        cx.notify();
    }

    pub fn apply_schedule_outcome(
        &mut self,
        outcome: MeshScheduleOutcome,
        provider_display: Option<MeshComputeProviderDisplay>,
        cx: &mut Context<Self>,
    ) -> Result<(), MeshComputeUiError> {
        let state = match outcome {
            MeshScheduleOutcome::Idle => MeshComputeState::AwaitingAvailability,
            MeshScheduleOutcome::Acquired(lease) => {
                let provider =
                    require_provider_display(&lease.request().provider, provider_display)?;
                MeshComputeState::Running { provider }
            }
            MeshScheduleOutcome::Queued { reason } => MeshComputeState::Queued {
                provider: provider_display,
                reason,
            },
            MeshScheduleOutcome::NoCapacity { provider } => MeshComputeState::NoCapacity {
                provider: match provider {
                    Some(provider) => Some(require_provider_display(&provider, provider_display)?),
                    None => {
                        if provider_display.is_some() {
                            return Err(MeshComputeUiError::ProviderMismatch);
                        }
                        None
                    }
                },
            },
            MeshScheduleOutcome::PolicyDenied { reason, provider } => {
                let provider = match provider {
                    Some(provider) => Some(require_provider_display(&provider, provider_display)?),
                    None => {
                        if provider_display.is_some() {
                            return Err(MeshComputeUiError::ProviderMismatch);
                        }
                        None
                    }
                };
                match reason {
                    MeshCandidateIneligibility::Trust => {
                        MeshComputeState::UnapprovedProvider { provider }
                    }
                    MeshCandidateIneligibility::Stale => {
                        let provider =
                            provider.ok_or(MeshComputeUiError::MissingProviderDisplay)?;
                        MeshComputeState::Stale { provider }
                    }
                    MeshCandidateIneligibility::Revoked => {
                        let provider =
                            provider.ok_or(MeshComputeUiError::MissingProviderDisplay)?;
                        MeshComputeState::Revoked { provider }
                    }
                    reason => MeshComputeState::PolicyDenied { provider, reason },
                }
            }
            MeshScheduleOutcome::ProviderUnavailable { provider } => MeshComputeState::Stale {
                provider: require_provider_display(&provider, provider_display)?,
            },
        };
        self.state = state;
        cx.notify();
        Ok(())
    }

    pub fn display_copy(&self) -> MeshComputeDisplayCopy {
        match &self.state {
            MeshComputeState::AwaitingAvailability => display_copy(
                "Shared compute unavailable",
                "No eligible community capacity is currently advertised.",
                false,
            ),
            MeshComputeState::Available {
                provider,
                eligible_slots,
            } => display_copy(
                "Shared compute available",
                format!(
                    "{} · {} · {} · {eligible_slots} eligible slot{}",
                    provider.location_label(),
                    provider.trust_label(),
                    provider.provider.model_id,
                    if *eligible_slots == 1 { "" } else { "s" }
                ),
                false,
            ),
            MeshComputeState::Queued { provider, reason } => display_copy(
                "Shared compute queued",
                format!(
                    "{}{}",
                    provider_detail(provider.as_ref()),
                    match reason {
                        MeshQueueReason::Fairness => " · Waiting for a fair scheduling turn",
                        MeshQueueReason::RequesterConcurrency => {
                            " · Waiting for the requester's concurrency limit"
                        }
                    }
                ),
                false,
            ),
            MeshComputeState::Running { provider } => display_copy(
                "Running on shared compute",
                format!(
                    "{} · {} · {}",
                    provider.location_label(),
                    provider.trust_label(),
                    provider.provider.model_id
                ),
                false,
            ),
            MeshComputeState::Stale { provider } => display_copy(
                "Shared-compute provider is stale",
                format!(
                    "{} is no longer fresh. This attempt was not moved to another provider.",
                    provider.location_label()
                ),
                true,
            ),
            MeshComputeState::Revoked { provider } => display_copy(
                "Shared-compute provider was revoked",
                format!(
                    "{} is no longer approved. New work is blocked and this attempt was not rerouted.",
                    provider.location_label()
                ),
                true,
            ),
            MeshComputeState::NoCapacity { provider } => display_copy(
                "No shared-compute capacity",
                match provider {
                    Some(provider) => format!(
                        "{} has no approved capacity for this attempt. No fallback was selected.",
                        provider.location_label()
                    ),
                    None => {
                        "No eligible provider has approved capacity. No fallback was selected."
                            .to_string()
                    }
                },
                false,
            ),
            MeshComputeState::UnapprovedProvider { provider } => display_copy(
                "Shared-compute provider is not approved",
                match provider {
                    Some(provider) => format!(
                        "{} does not satisfy the current trust policy. Execution was blocked.",
                        provider.location_label()
                    ),
                    None => {
                        "Available providers do not satisfy the current trust policy. Execution was blocked."
                            .to_string()
                    }
                },
                true,
            ),
            MeshComputeState::PolicyDenied { provider, reason } => display_copy(
                "Shared-compute request was denied",
                format!(
                    "{} · {}",
                    provider_detail(provider.as_ref()),
                    policy_denial_label(*reason)
                ),
                true,
            ),
            MeshComputeState::ProviderFailed { provider, reason } => display_copy(
                "Shared-compute execution failed",
                format!("{} · {}", provider.location_label(), reason.label()),
                true,
            ),
        }
    }
}

impl Render for MeshComputeView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let copy = self.display_copy();
        v_flex()
            .id("mesh-compute-status")
            .role(Role::Region)
            .aria_label("Shared compute status")
            .gap_1()
            .p_3()
            .child(div().text_ui(cx).child(copy.heading))
            .child(
                div()
                    .id("mesh-compute-status-detail")
                    .role(if copy.is_alert {
                        Role::Alert
                    } else {
                        Role::Status
                    })
                    .aria_label(copy.accessibility_label)
                    .text_sm()
                    .text_color(cx.theme().colors().text_muted)
                    .child(copy.detail),
            )
    }
}

fn require_provider_display(
    provider: &MeshProviderSelection,
    display: Option<MeshComputeProviderDisplay>,
) -> Result<MeshComputeProviderDisplay, MeshComputeUiError> {
    let display = display.ok_or(MeshComputeUiError::MissingProviderDisplay)?;
    if display.provider != *provider {
        return Err(MeshComputeUiError::ProviderMismatch);
    }
    Ok(display)
}

fn display_copy(
    heading: impl Into<String>,
    detail: impl Into<String>,
    is_alert: bool,
) -> MeshComputeDisplayCopy {
    let heading = heading.into();
    let detail = detail.into();
    MeshComputeDisplayCopy {
        accessibility_label: format!("{heading}. {detail}"),
        heading,
        detail,
        is_alert,
    }
}

fn provider_detail(provider: Option<&MeshComputeProviderDisplay>) -> String {
    provider.map_or_else(
        || "No provider selected".to_string(),
        |provider| format!("{} · {}", provider.location_label(), provider.trust_label()),
    )
}

const fn policy_denial_label(reason: MeshCandidateIneligibility) -> &'static str {
    match reason {
        MeshCandidateIneligibility::Consent => "Sharing consent is not current",
        MeshCandidateIneligibility::Membership => "Community membership is not current",
        MeshCandidateIneligibility::RequesterAuthorization => "The requester is not authorized",
        MeshCandidateIneligibility::Delegation => "Remote inference is outside the delegation",
        MeshCandidateIneligibility::Trust => "The provider trust class is not approved",
        MeshCandidateIneligibility::Model => "The approved model is unavailable",
        MeshCandidateIneligibility::Capability => "The requested capability is not approved",
        MeshCandidateIneligibility::Context => "The selected context cannot be transferred",
        MeshCandidateIneligibility::Revoked => "The provider was revoked",
        MeshCandidateIneligibility::Draining => "The provider is draining",
        MeshCandidateIneligibility::Quarantined => "The provider is quarantined",
        MeshCandidateIneligibility::Stale => "The provider state is stale",
        MeshCandidateIneligibility::Sandbox => "The approved inference sandbox is unavailable",
        MeshCandidateIneligibility::ResourcePolicy => "The resource request exceeds policy",
        MeshCandidateIneligibility::RequesterConcurrency => {
            "The requester concurrency limit is active"
        }
        MeshCandidateIneligibility::NoCapacity => "No approved capacity is available",
        MeshCandidateIneligibility::JobNotExecutable => {
            "The canonical job cannot acquire an executor lease"
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum MeshComputeUiError {
    #[error("shared-compute provider display is invalid")]
    InvalidProviderDisplay,
    #[error("shared-compute availability is invalid")]
    InvalidAvailability,
    #[error("shared-compute provider display is required")]
    MissingProviderDisplay,
    #[error("shared-compute provider display does not match the scheduler outcome")]
    ProviderMismatch,
}

#[cfg(test)]
mod tests {
    use collaboration_domain::{AggregateVersion, CommunityId, PrincipalId};
    use gpui::TestAppContext;
    use iroh_base::SecretKey;
    use remote::mesh::{
        advertisement::{MeshDeviceId, ModelArtifactDigest},
        scheduler::MeshScheduleOutcome,
    };
    use uuid::Uuid;

    use super::*;

    fn provider(trust_class: ProviderTrustClass) -> MeshProviderSelection {
        MeshProviderSelection {
            community_id: CommunityId::from_uuid(Uuid::from_u128(1)),
            owner_principal_id: PrincipalId::from_uuid(Uuid::from_u128(2)),
            device_id: MeshDeviceId::from_bytes([3; 32]).expect("valid device"),
            endpoint_id: SecretKey::from_bytes(&[4; 32]).public(),
            trust_class,
            membership_version: AggregateVersion::FIRST,
            runtime_generation: 1,
            sharing_generation: 1,
            advertisement_record_version: 1,
            advertisement_expires_at_millis: 60_000,
            model_id: "org/model@sha256".to_string(),
            artifact_digest: ModelArtifactDigest::from_bytes([5; 32]).expect("valid digest"),
        }
    }

    fn display(trust_class: ProviderTrustClass) -> MeshComputeProviderDisplay {
        MeshComputeProviderDisplay::new(provider(trust_class), "Avery's workstation")
            .expect("valid display")
    }

    #[gpui::test]
    fn mesh_compute_renders_eligible_capacity_and_execution_location(cx: &mut TestAppContext) {
        let view = cx.new(|_| MeshComputeView::default());
        view.update(cx, |view, cx| {
            view.show_available(display(ProviderTrustClass::CommunityMemberOwned), 2, cx)
        })
        .expect("available");
        let copy = view.read_with(cx, |view, _| view.display_copy());
        assert_eq!(copy.heading, "Shared compute available");
        assert!(copy.detail.contains("Avery's workstation"));
        assert!(copy.detail.contains("Community member hardware"));
        assert!(copy.detail.contains("2 eligible slots"));
        assert!(!copy.is_alert);
    }

    #[gpui::test]
    fn mesh_compute_renders_stale_provider_without_rerouting(cx: &mut TestAppContext) {
        let view = cx.new(|_| MeshComputeView::default());
        view.update(cx, |view, cx| {
            view.show_stale(display(ProviderTrustClass::CommunityMemberOwned), cx)
        });
        let copy = view.read_with(cx, |view, _| view.display_copy());
        assert_eq!(copy.heading, "Shared-compute provider is stale");
        assert!(copy.detail.contains("not moved to another provider"));
        assert!(copy.is_alert);
    }

    #[gpui::test]
    fn mesh_compute_renders_revoked_provider_as_blocked(cx: &mut TestAppContext) {
        let view = cx.new(|_| MeshComputeView::default());
        view.update(cx, |view, cx| {
            view.show_revoked(display(ProviderTrustClass::CommunityMemberOwned), cx)
        });
        let copy = view.read_with(cx, |view, _| view.display_copy());
        assert_eq!(copy.heading, "Shared-compute provider was revoked");
        assert!(copy.detail.contains("New work is blocked"));
        assert!(copy.detail.contains("not rerouted"));
    }

    #[gpui::test]
    fn mesh_compute_renders_no_capacity_without_fallback(cx: &mut TestAppContext) {
        let provider = provider(ProviderTrustClass::CommunityMemberOwned);
        let view = cx.new(|_| MeshComputeView::default());
        view.update(cx, |view, cx| {
            view.apply_schedule_outcome(
                MeshScheduleOutcome::NoCapacity {
                    provider: Some(provider),
                },
                Some(display(ProviderTrustClass::CommunityMemberOwned)),
                cx,
            )
        })
        .expect("no capacity");
        let copy = view.read_with(cx, |view, _| view.display_copy());
        assert_eq!(copy.heading, "No shared-compute capacity");
        assert!(copy.detail.contains("No fallback was selected"));
    }

    #[gpui::test]
    fn mesh_compute_renders_unapproved_provider_without_policy_controls(cx: &mut TestAppContext) {
        let view = cx.new(|_| MeshComputeView::default());
        view.update(cx, |view, cx| {
            view.apply_schedule_outcome(
                MeshScheduleOutcome::PolicyDenied {
                    reason: MeshCandidateIneligibility::Trust,
                    provider: None,
                },
                None,
                cx,
            )
        })
        .expect("policy denial");
        let copy = view.read_with(cx, |view, _| view.display_copy());
        assert_eq!(copy.heading, "Shared-compute provider is not approved");
        assert!(copy.detail.contains("Execution was blocked"));
        assert!(copy.is_alert);
    }

    #[gpui::test]
    fn mesh_compute_renders_provider_failure_and_unknown_outcome(cx: &mut TestAppContext) {
        let view = cx.new(|_| MeshComputeView::default());
        view.update(cx, |view, cx| {
            view.show_provider_failure(
                display(ProviderTrustClass::DeploymentManaged),
                MeshProviderFailureReason::UnknownOutcome,
                cx,
            )
        });
        let copy = view.read_with(cx, |view, _| view.display_copy());
        assert_eq!(copy.heading, "Shared-compute execution failed");
        assert!(copy.detail.contains("outcome is unknown"));
        assert!(copy.detail.contains("not retried on another provider"));
        assert!(copy.is_alert);
    }
}
