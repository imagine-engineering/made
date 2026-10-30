use anyhow::{Context as _, bail};
use collaboration_domain::{AggregateId, CommunityId, NostrEventId, PrincipalId};
use db::kvp::KeyValueStore;
use serde::{Deserialize, Serialize};
use std::fmt;

const NAMESPACE_PREFIX: &str = "collaborative_message_drafts.v1";
const CHANNEL_DRAFT_KEY: &str = "channel";
const STORED_DRAFT_VERSION: u8 = 1;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DraftLocation {
    community_id: CommunityId,
    channel_id: AggregateId,
    thread_root_event_id: Option<NostrEventId>,
}

impl DraftLocation {
    pub const fn channel(community_id: CommunityId, channel_id: AggregateId) -> Self {
        Self {
            community_id,
            channel_id,
            thread_root_event_id: None,
        }
    }

    pub const fn thread(
        community_id: CommunityId,
        channel_id: AggregateId,
        thread_root_event_id: NostrEventId,
    ) -> Self {
        Self {
            community_id,
            channel_id,
            thread_root_event_id: Some(thread_root_event_id),
        }
    }

    pub const fn community_id(self) -> CommunityId {
        self.community_id
    }

    pub const fn channel_id(self) -> AggregateId {
        self.channel_id
    }

    pub const fn thread_root_event_id(self) -> Option<NostrEventId> {
        self.thread_root_event_id
    }

    fn key(self) -> String {
        self.thread_root_event_id.map_or_else(
            || CHANNEL_DRAFT_KEY.to_owned(),
            |event_id| format!("thread:{}", encode_event_id(event_id)),
        )
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct LocalMessageDraft {
    content: String,
}

impl LocalMessageDraft {
    pub fn content(&self) -> &str {
        &self.content
    }
}

impl fmt::Debug for LocalMessageDraft {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalMessageDraft")
            .field("content", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DraftWriteOutcome {
    Saved,
    Cleared,
}

#[derive(Clone)]
pub struct DraftStore {
    database: KeyValueStore,
    owner_principal_id: PrincipalId,
}

impl DraftStore {
    pub const fn new(database: KeyValueStore, owner_principal_id: PrincipalId) -> Self {
        Self {
            database,
            owner_principal_id,
        }
    }

    pub const fn owner_principal_id(&self) -> PrincipalId {
        self.owner_principal_id
    }

    pub fn load(&self, location: DraftLocation) -> anyhow::Result<Option<LocalMessageDraft>> {
        let namespace = self.namespace(location.community_id, location.channel_id);
        let Some(raw) = self
            .database
            .scoped(&namespace)
            .read(&location.key())
            .context("reading local collaborative draft")?
        else {
            return Ok(None);
        };
        let stored = serde_json::from_str::<StoredDraft>(&raw)
            .context("decoding local collaborative draft")?;
        if stored.version != STORED_DRAFT_VERSION {
            bail!("unsupported local collaborative draft version");
        }
        if stored.content.trim().is_empty() {
            bail!("stored local collaborative draft is empty");
        }
        Ok(Some(LocalMessageDraft {
            content: stored.content,
        }))
    }

    pub async fn save(
        &self,
        location: DraftLocation,
        content: impl Into<String>,
    ) -> anyhow::Result<DraftWriteOutcome> {
        let content = content.into();
        if content.trim().is_empty() {
            self.clear(location).await?;
            return Ok(DraftWriteOutcome::Cleared);
        }
        let payload = serde_json::to_string(&StoredDraft {
            version: STORED_DRAFT_VERSION,
            content,
        })
        .context("encoding local collaborative draft")?;
        let namespace = self.namespace(location.community_id, location.channel_id);
        self.database
            .scoped(&namespace)
            .write(location.key(), payload)
            .await
            .context("persisting local collaborative draft")?;
        Ok(DraftWriteOutcome::Saved)
    }

    pub async fn clear(&self, location: DraftLocation) -> anyhow::Result<()> {
        let namespace = self.namespace(location.community_id, location.channel_id);
        self.database
            .scoped(&namespace)
            .delete(location.key())
            .await
            .context("clearing local collaborative draft")
    }

    pub async fn delete_channel(
        &self,
        community_id: CommunityId,
        channel_id: AggregateId,
    ) -> anyhow::Result<()> {
        let namespace = self.namespace(community_id, channel_id);
        self.database
            .scoped(&namespace)
            .delete_all()
            .await
            .context("clearing drafts for deleted collaborative channel")
    }

    fn namespace(&self, community_id: CommunityId, channel_id: AggregateId) -> String {
        format!(
            "{NAMESPACE_PREFIX}:{}:{community_id}:{channel_id}",
            self.owner_principal_id
        )
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StoredDraft {
    version: u8,
    content: String,
}

fn encode_event_id(event_id: NostrEventId) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let mut encoded = String::with_capacity(64);
    for byte in event_id.as_bytes() {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn principal(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn community(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn channel(value: u128) -> AggregateId {
        AggregateId::from_uuid(Uuid::from_u128(value))
    }

    fn thread(value: u8) -> NostrEventId {
        NostrEventId::from_bytes([value; 32])
    }

    #[gpui::test]
    async fn drafts_survive_restart_and_offline_thread_switches() {
        let database = KeyValueStore::open_test_db("collaborative_drafts_restart").await;
        let owner = principal(1);
        let channel_location = DraftLocation::channel(community(2), channel(3));
        let thread_location = DraftLocation::thread(community(2), channel(3), thread(4));
        {
            let store = DraftStore::new(database.clone(), owner);
            assert_eq!(
                store
                    .save(channel_location, "offline channel draft")
                    .await
                    .expect("save channel draft"),
                DraftWriteOutcome::Saved
            );
            store
                .save(thread_location, "offline thread draft")
                .await
                .expect("save thread draft");
        }

        let restarted = DraftStore::new(database, owner);
        assert_eq!(
            restarted
                .load(channel_location)
                .expect("load channel draft")
                .map(|draft| draft.content().to_owned()),
            Some("offline channel draft".to_owned())
        );
        assert_eq!(
            restarted
                .load(thread_location)
                .expect("load thread draft")
                .map(|draft| draft.content().to_owned()),
            Some("offline thread draft".to_owned())
        );
    }

    #[gpui::test]
    async fn channel_deletion_clears_only_that_channels_drafts() {
        let database = KeyValueStore::open_test_db("collaborative_drafts_delete_channel").await;
        let store = DraftStore::new(database, principal(1));
        let deleted_channel = DraftLocation::channel(community(2), channel(3));
        let deleted_thread = DraftLocation::thread(community(2), channel(3), thread(4));
        let retained_channel = DraftLocation::channel(community(2), channel(5));
        store
            .save(deleted_channel, "channel draft")
            .await
            .expect("save deleted channel draft");
        store
            .save(deleted_thread, "thread draft")
            .await
            .expect("save deleted thread draft");
        store
            .save(retained_channel, "retained")
            .await
            .expect("save retained draft");

        store
            .delete_channel(community(2), channel(3))
            .await
            .expect("delete channel drafts");

        assert_eq!(
            store.load(deleted_channel).expect("load deleted channel"),
            None
        );
        assert_eq!(
            store.load(deleted_thread).expect("load deleted thread"),
            None
        );
        assert_eq!(
            store
                .load(retained_channel)
                .expect("load retained channel")
                .map(|draft| draft.content().to_owned()),
            Some("retained".to_owned())
        );
    }

    #[gpui::test]
    async fn account_switch_never_serves_another_accounts_draft() {
        let database = KeyValueStore::open_test_db("collaborative_drafts_account_switch").await;
        let location = DraftLocation::channel(community(2), channel(3));
        let first = DraftStore::new(database.clone(), principal(1));
        first
            .save(location, "first account")
            .await
            .expect("save first account draft");

        let second = DraftStore::new(database.clone(), principal(4));
        assert_eq!(second.load(location).expect("load second account"), None);
        second
            .save(location, "second account")
            .await
            .expect("save second account draft");

        assert_eq!(
            DraftStore::new(database, principal(1))
                .load(location)
                .expect("reload first account")
                .map(|draft| draft.content().to_owned()),
            Some("first account".to_owned())
        );
    }

    #[gpui::test]
    async fn same_channel_id_never_reuses_a_cross_community_draft() {
        let database = KeyValueStore::open_test_db("collaborative_drafts_community_scope").await;
        let store = DraftStore::new(database, principal(1));
        let first = DraftLocation::channel(community(2), channel(3));
        let second = DraftLocation::channel(community(4), channel(3));
        store
            .save(first, "first community")
            .await
            .expect("save first community draft");

        assert_eq!(store.load(second).expect("load second community"), None);
        assert_eq!(
            store.save(first, "   ").await.expect("clear empty draft"),
            DraftWriteOutcome::Cleared
        );
        assert_eq!(store.load(first).expect("load cleared draft"), None);
    }

    #[test]
    fn debug_output_redacts_unsent_content() {
        let draft = LocalMessageDraft {
            content: "private unsent message".to_owned(),
        };
        let debug = format!("{draft:?}");

        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("private unsent message"));
    }
}
