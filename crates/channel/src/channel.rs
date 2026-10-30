mod channel_buffer;
mod channel_store;
#[cfg(feature = "multiplayer-tools")]
mod collaboration_store;

use client::{Client, UserStore};
use gpui::{App, Entity};
use std::sync::Arc;

pub use channel_buffer::{ACKNOWLEDGE_DEBOUNCE_INTERVAL, ChannelBuffer, ChannelBufferEvent};
pub use channel_store::{Channel, ChannelEvent, ChannelMembership, ChannelStore};
#[cfg(feature = "multiplayer-tools")]
pub use collaboration_store::{
    CanonicalChannelId, CollaborationChannelProjection, CollaborationCommunityProjection,
    CollaborationProjectionError, CollaborationProjectionOutcome, CollaborationSnapshot,
    CollaborationStore,
};

#[cfg(test)]
mod channel_store_tests;

pub fn init(client: &Arc<Client>, user_store: Entity<UserStore>, cx: &mut App) {
    channel_store::init(client, user_store, cx);
    channel_buffer::init(&client.clone().into());
}
