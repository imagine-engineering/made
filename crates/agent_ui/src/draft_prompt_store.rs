//! Per-thread draft prompt persistence and display label rendering.
//!
//! Drafts are persisted in the thread metadata store with `session_id: None`,
//! but their unsent prompt text is kept separately here so we don't have to
//! plumb draft-prompt storage through the native agent's thread database.
//!
//! The display-label helpers ([`display_label_for_draft`] and friends) live
//! alongside the storage so the sidebar's preview rendering can't drift from
//! the format we persist.

use agent::ZED_AGENT_ID;
use agent_client_protocol::schema::v1 as acp;
use anyhow::Context as _;
use db::kvp::KeyValueStore;
use gpui::{App, AppContext as _, Entity, Task};
use itertools::Itertools;
use project::AgentId;
use ui::SharedString;
use util::ResultExt as _;
use workspace::Workspace;

use crate::AgentPanel;
use crate::thread_metadata_store::ThreadId;

const NAMESPACE: &str = "agent_draft_prompts";

/// Maximum length (in characters) of a draft label rendered in the sidebar.
const MAX_LABEL_CHARS: usize = 250;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
enum StoredDraft {
    Current {
        accepted: Option<Vec<acp::ContentBlock>>,
        live: Vec<acp::ContentBlock>,
    },
    Legacy(Vec<acp::ContentBlock>),
}

fn read_stored(thread_id: ThreadId, cx: &App) -> anyhow::Result<Option<StoredDraft>> {
    let kvp = KeyValueStore::global(cx);
    let raw = kvp
        .scoped(NAMESPACE)
        .read(&thread_id.to_key_string())
        .context("reading saved unsent draft")?;
    raw.map(|raw| serde_json::from_str(&raw).context("decoding saved unsent draft"))
        .transpose()
}

pub fn read(thread_id: ThreadId, cx: &App) -> Option<Vec<acp::ContentBlock>> {
    match read_stored(thread_id, cx).log_err().flatten()? {
        StoredDraft::Current { accepted, .. } => accepted,
        StoredDraft::Legacy(prompt) => Some(prompt),
    }
}

pub fn read_for_restore(thread_id: ThreadId, cx: &App) -> Option<Vec<acp::ContentBlock>> {
    try_read_for_restore(thread_id, cx).log_err().flatten()
}

pub fn try_read_for_restore(
    thread_id: ThreadId,
    cx: &App,
) -> anyhow::Result<Option<Vec<acp::ContentBlock>>> {
    Ok(read_stored(thread_id, cx)?.map(|stored| match stored {
        StoredDraft::Current { live, .. } => live,
        StoredDraft::Legacy(prompt) => prompt,
    }))
}

pub fn write_with_live_draft(
    thread_id: ThreadId,
    accepted: Option<&[acp::ContentBlock]>,
    live: &[acp::ContentBlock],
    cx: &App,
) -> Task<anyhow::Result<()>> {
    let accepted = accepted.map(<[_]>::to_vec);
    let live = live.to_vec();
    let kvp = KeyValueStore::global(cx);
    let key = thread_id.to_key_string();
    cx.background_spawn(async move {
        kvp.scoped(NAMESPACE)
            .update(key, move |raw| {
                // A release flush can wait behind a newer write; validate inside the write queue.
                let stored = raw
                    .map(|raw| {
                        serde_json::from_str::<StoredDraft>(&raw)
                            .context("decoding saved unsent draft")
                    })
                    .transpose()?;
                let draft = StoredDraft::Current {
                    accepted: accepted.or_else(|| {
                        stored.and_then(|stored| match stored {
                            StoredDraft::Current { accepted, .. } => accepted,
                            StoredDraft::Legacy(prompt) => Some(prompt),
                        })
                    }),
                    live,
                };
                serde_json::to_string(&draft)
                    .context("serializing draft prompt")
                    .map(Some)
            })
            .await
    })
}

pub fn write(
    thread_id: ThreadId,
    prompt: &[acp::ContentBlock],
    cx: &App,
) -> Task<anyhow::Result<()>> {
    write_serialized(thread_id, prompt, cx)
}

fn write_serialized(
    thread_id: ThreadId,
    prompt: &(impl serde::Serialize + ?Sized),
    cx: &App,
) -> Task<anyhow::Result<()>> {
    let kvp = KeyValueStore::global(cx);
    let key = thread_id.to_key_string();
    let payload = match serde_json::to_string(prompt).context("serializing draft prompt") {
        Ok(payload) => payload,
        Err(err) => return Task::ready(Err(err)),
    };
    cx.background_spawn(async move { kvp.scoped(NAMESPACE).write(key, payload).await })
}

pub fn delete(thread_id: ThreadId, cx: &App) -> Task<anyhow::Result<()>> {
    let kvp = KeyValueStore::global(cx);
    let key = thread_id.to_key_string();
    cx.background_spawn(async move { kvp.scoped(NAMESPACE).delete(key).await })
}

pub fn delete_if_readable(thread_id: ThreadId, cx: &App) -> Task<anyhow::Result<()>> {
    let kvp = KeyValueStore::global(cx);
    let key = thread_id.to_key_string();
    cx.background_spawn(async move {
        kvp.scoped(NAMESPACE)
            .update(key, |raw| {
                if let Some(raw) = raw {
                    serde_json::from_str::<StoredDraft>(&raw)
                        .context("decoding saved unsent draft before automatic deletion")?;
                }
                Ok(None)
            })
            .await
    })
}

pub fn draft_has_user_content<'a>(
    thread_id: ThreadId,
    workspaces: impl IntoIterator<Item = &'a Entity<Workspace>>,
    cx: &App,
) -> bool {
    let mut found_live_copy = false;
    for panel in workspaces
        .into_iter()
        .filter_map(|workspace| workspace.read(cx).panel::<AgentPanel>(cx))
    {
        let blocks = match panel
            .read(cx)
            .draft_prompt_blocks_if_in_memory(thread_id, cx)
        {
            Ok(Some(blocks)) => blocks,
            Ok(None) => continue,
            Err(error) => {
                log::error!(
                    "Cannot establish whether live draft {thread_id:?} is empty: {error:#}"
                );
                return true;
            }
        };
        found_live_copy = true;
        if blocks_have_user_content(&blocks) {
            return true;
        }
    }

    if found_live_copy {
        false
    } else {
        match try_read_for_restore(thread_id, cx) {
            Ok(blocks) => blocks.is_some_and(|blocks| blocks_have_user_content(&blocks)),
            Err(error) => {
                log::error!(
                    "Cannot establish whether saved draft {thread_id:?} is empty: {error:#}"
                );
                true
            }
        }
    }
}

fn blocks_have_user_content(blocks: &[acp::ContentBlock]) -> bool {
    blocks.iter().any(|block| match block {
        acp::ContentBlock::Text(text) => !text.text.trim().is_empty(),
        _ => true,
    })
}

/// Rewrites `[@Something](scheme://...)` mention links as `@Something` so the
/// sidebar's draft-title preview doesn't show raw markdown link syntax.
pub fn clean_mention_links(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut remaining = input;

    while let Some(start) = remaining.find("[@") {
        result.push_str(&remaining[..start]);
        let after_bracket = &remaining[start + 1..];
        if let Some(close_bracket) = after_bracket.find("](") {
            let mention = &after_bracket[..close_bracket];
            let after_link_start = &after_bracket[close_bracket + 2..];
            if let Some(close_paren) = after_link_start.find(')') {
                result.push_str(mention);
                remaining = &after_link_start[close_paren + 1..];
                continue;
            }
        }
        result.push_str("[@");
        remaining = &remaining[start + 2..];
    }
    result.push_str(remaining);
    result
}

/// Collapses whitespace and truncates raw editor text for display as a draft
/// label in the sidebar.
pub fn truncate_draft_label(raw: &str) -> Option<SharedString> {
    let first_line = raw.lines().next().unwrap_or("");
    let cleaned = clean_mention_links(first_line);
    let mut text: String = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.is_empty() {
        return None;
    }
    if let Some((truncate_at, _)) = text.char_indices().nth(MAX_LABEL_CHARS) {
        text.truncate(truncate_at);
    }
    Some(text.into())
}

/// Renders a draft thread's display label for sidebar rows and similar
/// preview UI.
///
/// Prefers the live message editor's text (when the thread's
/// `ConversationView` is loaded in the workspace's `AgentPanel`), and
/// otherwise falls back to the persisted draft prompt in the kvp store so
/// drafts restored from disk — but not yet opened — still show a meaningful
/// title instead of the generic default.
pub fn display_label_for_draft(
    workspace: Option<&Entity<Workspace>>,
    thread_id: ThreadId,
    cx: &App,
) -> Option<SharedString> {
    let in_memory = workspace
        .and_then(|ws| ws.read(cx).panel::<AgentPanel>(cx))
        .and_then(|panel| panel.read(cx).editor_text_if_in_memory(thread_id, cx));
    match in_memory {
        Some(Some(raw)) => return truncate_draft_label(&raw),
        Some(None) => return None,
        None => {}
    }

    let blocks = read_for_restore(thread_id, cx)?;
    let raw = blocks
        .iter()
        .filter_map(|block| match block {
            acp::ContentBlock::Text(text) => Some(text.text.as_str()),
            acp::ContentBlock::ResourceLink(link) => Some(link.uri.as_str()),
            _ => None,
        })
        .join(" ");
    truncate_draft_label(&raw)
}

pub fn empty_draft_placeholder_label(
    workspace: Option<&Entity<Workspace>>,
    agent_id: &AgentId,
    cx: &App,
) -> SharedString {
    let agent_name = if agent_id.as_ref() == ZED_AGENT_ID.as_ref() {
        SharedString::from(ZED_AGENT_ID.to_string())
    } else {
        workspace
            .map(|ws| ws.read(cx).project().read(cx).agent_server_store().clone())
            .and_then(|store| store.read(cx).agent_display_name(agent_id))
            .unwrap_or_else(|| SharedString::from(agent_id.to_string()))
    };

    format!("New {} Thread", agent_name).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[gpui::test]
    async fn test_automatic_save_and_empty_flush_preserve_unreadable_draft(
        cx: &mut gpui::TestAppContext,
    ) {
        crate::test_support::init_test(cx);
        let thread_id = ThreadId::new();
        let database = cx.read(KeyValueStore::global);
        let unreadable = "{unreadable draft";
        database
            .scoped(NAMESPACE)
            .write(thread_id.to_key_string(), unreadable.into())
            .await
            .expect("unreadable saved draft control");
        let live = vec![acp::ContentBlock::from("current live input")];
        let save = cx.read(|cx| write_with_live_draft(thread_id, None, &live, cx));
        assert!(save.await.is_err());
        let delete = cx.read(|cx| delete_if_readable(thread_id, cx));
        assert!(delete.await.is_err());
        assert_eq!(
            database
                .scoped(NAMESPACE)
                .read(&thread_id.to_key_string())
                .expect("retained draft"),
            Some(unreadable.into())
        );
    }

    #[test]
    fn test_clean_mention_links() {
        // Simple mention link.
        assert_eq!(
            clean_mention_links("check [@Button.tsx](file:///path/to/Button.tsx)"),
            "check @Button.tsx"
        );

        // Multiple mention links on one line.
        assert_eq!(
            clean_mention_links("look at [@foo.rs](file:///foo.rs) and [@bar.rs](file:///bar.rs)"),
            "look at @foo.rs and @bar.rs"
        );

        // Plain text without mentions is preserved.
        assert_eq!(
            clean_mention_links("plain text with no mentions"),
            "plain text with no mentions"
        );

        // Broken syntax (no closing bracket) is left alone.
        assert_eq!(
            clean_mention_links("broken [@mention without closing"),
            "broken [@mention without closing"
        );

        // Non-`@` markdown links are not touched.
        assert_eq!(
            clean_mention_links("see [docs](https://example.com)"),
            "see [docs](https://example.com)"
        );

        // Empty input.
        assert_eq!(clean_mention_links(""), "");
    }
}
