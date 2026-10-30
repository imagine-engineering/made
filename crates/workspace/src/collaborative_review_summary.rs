use gpui::EntityId;

use crate::collaborative_review::CollaborativeReviewSlot;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollaborativeReviewSummarySource {
    slot: CollaborativeReviewSlot,
    provider_id: EntityId,
    revision: u64,
}

impl CollaborativeReviewSummarySource {
    pub fn new(slot: CollaborativeReviewSlot, provider_id: EntityId, revision: u64) -> Self {
        Self {
            slot,
            provider_id,
            revision,
        }
    }

    pub fn slot(self) -> CollaborativeReviewSlot {
        self.slot
    }

    pub fn provider_id(self) -> EntityId {
        self.provider_id
    }

    pub fn revision(self) -> u64 {
        self.revision
    }
}
