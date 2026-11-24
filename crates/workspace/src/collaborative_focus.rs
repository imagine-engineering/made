use gpui::actions;

actions!(
    workspace,
    [
        /// Moves focus to the next control in the Multiplayer Workspace.
        FocusNextCollaborativeRegion,
        /// Moves focus to the previous control in the Multiplayer Workspace.
        FocusPreviousCollaborativeRegion,
        /// Restores focus to the Multiplayer Workspace's first available control.
        RestoreCollaborativeFocus
    ]
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborativeFocusRegion {
    Navigation,
    Timeline,
    Composer,
    Review,
    Status,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CollaborativeFocusOrder {
    review_visible: bool,
    last_focused: Option<CollaborativeFocusRegion>,
}

impl CollaborativeFocusOrder {
    pub fn new(review_visible: bool) -> Self {
        Self {
            review_visible,
            last_focused: None,
        }
    }

    pub fn set_review_visible(&mut self, review_visible: bool) {
        self.review_visible = review_visible;
        if !review_visible && self.last_focused == Some(CollaborativeFocusRegion::Review) {
            self.last_focused = Some(CollaborativeFocusRegion::Timeline);
        }
    }

    pub fn record_focus(&mut self, region: CollaborativeFocusRegion) -> bool {
        if !self.regions().contains(&region) {
            return false;
        }
        self.last_focused = Some(region);
        true
    }

    pub fn restore(&self) -> CollaborativeFocusRegion {
        self.last_focused
            .filter(|region| self.regions().contains(region))
            .unwrap_or(CollaborativeFocusRegion::Timeline)
    }

    pub fn next(&self, current: CollaborativeFocusRegion) -> Option<CollaborativeFocusRegion> {
        let regions = self.regions();
        regions
            .iter()
            .position(|region| *region == current)
            .and_then(|index| regions.get(index.saturating_add(1)).copied())
    }

    pub fn previous(&self, current: CollaborativeFocusRegion) -> Option<CollaborativeFocusRegion> {
        let regions = self.regions();
        regions
            .iter()
            .position(|region| *region == current)
            .and_then(|index| index.checked_sub(1))
            .and_then(|index| regions.get(index).copied())
    }

    pub fn regions(&self) -> &'static [CollaborativeFocusRegion] {
        const COLLAPSED: &[CollaborativeFocusRegion] = &[
            CollaborativeFocusRegion::Navigation,
            CollaborativeFocusRegion::Timeline,
            CollaborativeFocusRegion::Composer,
            CollaborativeFocusRegion::Status,
        ];
        const EXPANDED: &[CollaborativeFocusRegion] = &[
            CollaborativeFocusRegion::Navigation,
            CollaborativeFocusRegion::Timeline,
            CollaborativeFocusRegion::Composer,
            CollaborativeFocusRegion::Review,
            CollaborativeFocusRegion::Status,
        ];
        if self.review_visible {
            EXPANDED
        } else {
            COLLAPSED
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collaborative_focus() {
        let mut focus = CollaborativeFocusOrder::new(true);
        assert_eq!(
            focus.regions(),
            &[
                CollaborativeFocusRegion::Navigation,
                CollaborativeFocusRegion::Timeline,
                CollaborativeFocusRegion::Composer,
                CollaborativeFocusRegion::Review,
                CollaborativeFocusRegion::Status,
            ]
        );
        assert_eq!(
            focus.next(CollaborativeFocusRegion::Composer),
            Some(CollaborativeFocusRegion::Review)
        );
        assert_eq!(
            focus.previous(CollaborativeFocusRegion::Composer),
            Some(CollaborativeFocusRegion::Timeline)
        );
        assert_eq!(focus.next(CollaborativeFocusRegion::Status), None);
        assert_eq!(focus.previous(CollaborativeFocusRegion::Navigation), None);

        assert!(focus.record_focus(CollaborativeFocusRegion::Review));
        assert_eq!(focus.restore(), CollaborativeFocusRegion::Review);
        focus.set_review_visible(false);
        assert_eq!(focus.restore(), CollaborativeFocusRegion::Timeline);
        assert_eq!(
            focus.next(CollaborativeFocusRegion::Composer),
            Some(CollaborativeFocusRegion::Status)
        );
        assert!(!focus.record_focus(CollaborativeFocusRegion::Review));
        assert_eq!(focus.next(CollaborativeFocusRegion::Status), None);
    }
}
