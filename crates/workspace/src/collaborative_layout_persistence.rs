use anyhow::Result;
use db::kvp::KeyValueStore;
use project::git_store::diff_buffer_list::DiffBase;
use serde::{Deserialize, Serialize};
use util::ResultExt;

use crate::{WorkspaceId, collaborative_review::CollaborativeReviewSlot, dock::DockPosition};

const COLLABORATIVE_LAYOUT_NAMESPACE: &str = "collaborative_workspace_layout";
const COLLABORATIVE_LAYOUT_VERSION: u32 = 7;

pub(crate) const DEFAULT_RAIL_WIDTH: f32 = 268.;
pub(crate) const DEFAULT_REVIEW_WIDTH: f32 = 354.5;
pub(crate) const MIN_RAIL_WIDTH: f32 = 180.;
pub(crate) const MAX_RAIL_WIDTH: f32 = 400.;
pub(crate) const MIN_REVIEW_WIDTH: f32 = 280.;
pub(crate) const MAX_REVIEW_WIDTH: f32 = 800.;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct CollaborativeLayoutState {
    #[serde(default = "collaborative_layout_version")]
    version: u32,
    #[serde(default = "default_review_requested")]
    review_requested: bool,
    #[serde(default = "default_review_width")]
    review_width: f32,
    #[serde(default)]
    review_fraction: Option<f64>,
    #[serde(default = "default_rail_width")]
    rail_width: f32,
    #[serde(default = "default_rail_requested")]
    rail_requested: bool,
    #[serde(default)]
    selected_review_slot: Option<CollaborativeReviewSlot>,
    #[serde(default)]
    project_review_base: Option<DiffBase>,
    #[serde(default)]
    native_editor_active: bool,
    #[serde(default)]
    requested_docks: CollaborativeDockRequests,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct CollaborativeDockRequests {
    left: bool,
    bottom: bool,
    right: bool,
}

impl Default for CollaborativeLayoutState {
    fn default() -> Self {
        Self {
            version: COLLABORATIVE_LAYOUT_VERSION,
            review_requested: true,
            review_width: DEFAULT_REVIEW_WIDTH,
            review_fraction: Some(0.5),
            rail_width: DEFAULT_RAIL_WIDTH,
            rail_requested: true,
            selected_review_slot: None,
            project_review_base: None,
            native_editor_active: false,
            requested_docks: CollaborativeDockRequests::default(),
        }
    }
}

impl CollaborativeLayoutState {
    pub(crate) fn dock_requested(&self, position: DockPosition) -> bool {
        match position {
            DockPosition::Left => self.requested_docks.left,
            DockPosition::Bottom => self.requested_docks.bottom,
            DockPosition::Right => self.requested_docks.right,
        }
    }

    pub(crate) fn with_dock_requested(mut self, position: DockPosition) -> Self {
        match position {
            DockPosition::Left => self.requested_docks.left = true,
            DockPosition::Bottom => self.requested_docks.bottom = true,
            DockPosition::Right => self.requested_docks.right = true,
        }
        self
    }

    pub(crate) fn native_editor_active(&self) -> bool {
        self.native_editor_active
    }

    pub(crate) fn with_native_editor_active(mut self, active: bool) -> Self {
        self.native_editor_active = active;
        self
    }

    pub(crate) fn review_requested(&self) -> bool {
        self.review_requested
    }

    pub(crate) fn review_width(&self) -> f32 {
        self.review_width
    }

    pub(crate) fn review_fraction(&self) -> Option<f64> {
        self.review_fraction
    }

    pub(crate) fn requested_review_width_for(&self, split_width: f32) -> f32 {
        self.review_fraction
            .map(|fraction| (f64::from(split_width.max(0.)) * fraction) as f32)
            .unwrap_or_else(|| self.review_width())
    }

    pub(crate) fn rail_width(&self) -> f32 {
        self.rail_width
    }

    pub(crate) fn rail_requested(&self) -> bool {
        self.rail_requested
    }

    pub(crate) fn with_rail_requested(mut self, requested: bool) -> Self {
        self.rail_requested = requested;
        self
    }

    pub(crate) fn selected_review_slot(&self) -> Option<CollaborativeReviewSlot> {
        self.selected_review_slot
    }

    pub(crate) fn project_review_base(&self) -> Option<&DiffBase> {
        self.project_review_base.as_ref()
    }

    pub(crate) fn with_project_review_base(mut self, base: DiffBase) -> Self {
        self.project_review_base = Some(base);
        self
    }

    pub(crate) fn with_review_requested(mut self, review_requested: bool) -> Self {
        self.review_requested = review_requested;
        self
    }

    pub(crate) fn with_review_width(mut self, review_width: f32) -> Self {
        self.review_width = review_width;
        self.review_fraction = None;
        self.normalized()
    }

    pub(crate) fn with_review_fraction(mut self, review_fraction: f64) -> Self {
        self.review_fraction = Some(review_fraction);
        self.normalized()
    }

    pub(crate) fn with_rail_width(mut self, rail_width: f32) -> Self {
        self.rail_width = rail_width;
        self.normalized()
    }

    pub(crate) fn with_selected_review_slot(
        mut self,
        slot: Option<CollaborativeReviewSlot>,
    ) -> Self {
        self.selected_review_slot = slot;
        self
    }

    pub(crate) fn reset_rail_width(mut self) -> Self {
        self.rail_width = DEFAULT_RAIL_WIDTH;
        self
    }

    fn normalized(self) -> Self {
        if self.version == 0 || self.version > COLLABORATIVE_LAYOUT_VERSION {
            return Self::default();
        }

        Self {
            version: COLLABORATIVE_LAYOUT_VERSION,
            review_requested: self.review_requested,
            review_width: finite_or_default(self.review_width, DEFAULT_REVIEW_WIDTH)
                .clamp(MIN_REVIEW_WIDTH, MAX_REVIEW_WIDTH),
            review_fraction: self
                .review_fraction
                .filter(|fraction| fraction.is_finite())
                .map(|fraction| fraction.clamp(0., 1.)),
            rail_width: finite_or_default(self.rail_width, DEFAULT_RAIL_WIDTH)
                .clamp(MIN_RAIL_WIDTH, MAX_RAIL_WIDTH),
            rail_requested: self.rail_requested,
            selected_review_slot: self.selected_review_slot,
            project_review_base: self.project_review_base,
            native_editor_active: self.native_editor_active,
            requested_docks: self.requested_docks,
        }
    }
}

pub(crate) fn read_collaborative_layout_state(
    key_value_store: &KeyValueStore,
    workspace_id: WorkspaceId,
) -> CollaborativeLayoutState {
    key_value_store
        .scoped(COLLABORATIVE_LAYOUT_NAMESPACE)
        .read(&workspace_id.0.to_string())
        .log_err()
        .flatten()
        .and_then(|serialized| {
            serde_json::from_str::<CollaborativeLayoutState>(&serialized).log_err()
        })
        .map(CollaborativeLayoutState::normalized)
        .unwrap_or_default()
}

pub(crate) async fn write_collaborative_layout_state(
    key_value_store: &KeyValueStore,
    workspace_id: WorkspaceId,
    state: CollaborativeLayoutState,
) -> Result<()> {
    let serialized = serde_json::to_string(&state.normalized())?;
    key_value_store
        .scoped(COLLABORATIVE_LAYOUT_NAMESPACE)
        .write(workspace_id.0.to_string(), serialized)
        .await
}

fn collaborative_layout_version() -> u32 {
    COLLABORATIVE_LAYOUT_VERSION
}

fn default_review_requested() -> bool {
    true
}

fn default_review_width() -> f32 {
    DEFAULT_REVIEW_WIDTH
}

fn default_rail_width() -> f32 {
    // Older records without a width must retain their original layout.
    226.
}

fn default_rail_requested() -> bool {
    true
}

fn finite_or_default(value: f32, default: f32) -> f32 {
    if value.is_finite() { value } else { default }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::init_test;

    #[test]
    fn collaborative_layout_preserves_earlier_supported_versions() {
        for version in 1..=COLLABORATIVE_LAYOUT_VERSION {
            let state: CollaborativeLayoutState = serde_json::from_value(serde_json::json!({
                "version": version,
                "review_requested": false,
                "review_width": 420.,
                "rail_width": 240.
            }))
            .expect("known earlier layout shape");
            let restored = state.normalized();
            assert_eq!(restored.version, COLLABORATIVE_LAYOUT_VERSION);
            assert!(!restored.review_requested());
            assert_eq!(restored.review_width(), 420.);
            assert_eq!(restored.review_fraction(), None);
            assert_eq!(restored.rail_width(), 240.);
            assert!(restored.rail_requested());
            assert!(!restored.native_editor_active());
            assert!(!restored.dock_requested(DockPosition::Left));
            assert!(!restored.dock_requested(DockPosition::Bottom));
            assert!(!restored.dock_requested(DockPosition::Right));
        }
    }

    #[test]
    fn collaborative_layout_default_split_preserves_explicit_widths() {
        let initial = CollaborativeLayoutState::default();
        assert_eq!(initial.review_fraction(), Some(0.5));
        assert_eq!(initial.rail_width(), 268.);
        let restored: CollaborativeLayoutState = serde_json::from_str(
            &serde_json::to_string(&initial).expect("serialize relative default"),
        )
        .expect("restore relative default");
        assert_eq!(restored.normalized(), initial);
        let dragged = initial.clone().with_review_fraction(1008. / 1917.);
        assert_eq!(dragged.requested_review_width_for(1917.), 1008.);
        let dragged_restored: CollaborativeLayoutState = serde_json::from_str(
            &serde_json::to_string(&dragged).expect("serialize dragged fraction"),
        )
        .expect("restore dragged fraction");
        assert_eq!(dragged_restored.requested_review_width_for(1917.), 1008.);
        let explicit = initial.with_review_width(420.);
        assert_eq!(explicit.review_fraction(), None);
        assert_eq!(explicit.review_width(), 420.);
        let legacy: CollaborativeLayoutState =
            serde_json::from_str(r#"{"version":6}"#).expect("earlier default layout");
        assert_eq!(legacy.normalized().rail_width(), 226.);
    }

    #[gpui::test]
    async fn collaborative_layout_restart(cx: &mut gpui::TestAppContext) {
        init_test(cx);

        let key_value_store = cx.update(|cx| KeyValueStore::global(cx));
        let editor_state_key = "editor-layout-window";
        let editor_state = r#"{"sidebar":{"width":517.0},"left_dock_visible":false}"#;
        key_value_store
            .scoped("multi_workspace_state")
            .write(editor_state_key.to_owned(), editor_state.to_owned())
            .await
            .expect("editor layout fixture should write");

        let workspace_id = WorkspaceId(7001);
        let requested = CollaborativeLayoutState::default()
            .with_rail_requested(false)
            .with_review_requested(false)
            .with_review_width(2500.)
            .with_rail_width(120.)
            .with_project_review_base(DiffBase::Staged)
            .with_native_editor_active(true)
            .with_dock_requested(DockPosition::Bottom)
            .with_selected_review_slot(Some(CollaborativeReviewSlot::ProjectChanges));
        write_collaborative_layout_state(&key_value_store, workspace_id, requested)
            .await
            .expect("collaborative layout should write");

        let restored = read_collaborative_layout_state(&key_value_store, workspace_id);
        assert!(!restored.review_requested());
        assert!(!restored.rail_requested());
        assert!(restored.native_editor_active());
        assert!(restored.dock_requested(DockPosition::Bottom));
        assert!(!restored.dock_requested(DockPosition::Left));
        assert!(!restored.dock_requested(DockPosition::Right));
        assert_eq!(restored.review_width(), MAX_REVIEW_WIDTH);
        assert_eq!(restored.rail_width(), MIN_RAIL_WIDTH);
        assert_eq!(restored.project_review_base(), Some(&DiffBase::Staged));
        assert_eq!(
            restored.selected_review_slot(),
            Some(CollaborativeReviewSlot::ProjectChanges)
        );
        assert_eq!(
            key_value_store
                .scoped("multi_workspace_state")
                .read(editor_state_key)
                .expect("editor layout fixture should read"),
            Some(editor_state.to_owned())
        );

        let relative_workspace_id = WorkspaceId(7003);
        let relative = CollaborativeLayoutState::default()
            .with_rail_requested(false)
            .with_review_fraction(0.6);
        write_collaborative_layout_state(&key_value_store, relative_workspace_id, relative.clone())
            .await
            .expect("relative layout should write");
        let restored_relative =
            read_collaborative_layout_state(&key_value_store, relative_workspace_id);
        assert_eq!(restored_relative, relative);
        assert_eq!(restored_relative.review_fraction(), Some(0.6));
        assert!(!restored_relative.rail_requested());
        assert_eq!(restored.review_fraction(), None);

        let other_workspace_id = WorkspaceId(7002);
        assert_eq!(
            read_collaborative_layout_state(&key_value_store, other_workspace_id),
            CollaborativeLayoutState::default()
        );

        key_value_store
            .scoped(COLLABORATIVE_LAYOUT_NAMESPACE)
            .write(
                other_workspace_id.0.to_string(),
                r#"{"version":999,"review_requested":false,"review_width":1000,"rail_width":700}"#
                    .to_owned(),
            )
            .await
            .expect("future-version fixture should write");
        assert_eq!(
            read_collaborative_layout_state(&key_value_store, other_workspace_id),
            CollaborativeLayoutState::default()
        );

        key_value_store
            .scoped(COLLABORATIVE_LAYOUT_NAMESPACE)
            .write(other_workspace_id.0.to_string(), "not-json".to_owned())
            .await
            .expect("malformed fixture should write");
        assert_eq!(
            read_collaborative_layout_state(&key_value_store, other_workspace_id),
            CollaborativeLayoutState::default()
        );
    }
}
