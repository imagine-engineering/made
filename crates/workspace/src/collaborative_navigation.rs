use std::{fmt, path::PathBuf};

use anyhow::Result;
use db::kvp::KeyValueStore;
use project::ProjectGroupKey;
use remote::{RemoteConnectionIdentity, remote_connection_identity};
use serde::{Deserialize, Serialize};
use url::Url;
use util::{ResultExt, path_list::PathList};

use crate::WorkspaceId;

const COLLABORATIVE_NAVIGATION_NAMESPACE: &str = "collaborative_workspace_navigation";
const COLLABORATIVE_NAVIGATION_VERSION: u32 = 1;
const MAX_HISTORY_ENTRIES: usize = 100;
const MAX_PINNED_ENTRIES: usize = 100;
const MAX_ENTITY_LINK_LENGTH: usize = 4096;
const MAX_OPAQUE_ID_LENGTH: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollaborativeProjectTarget {
    paths: Vec<PathBuf>,
    host: Option<CollaborativeRemoteTarget>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CollaborativeRemoteTarget {
    Ssh {
        host: String,
        username: Option<String>,
        port: Option<u16>,
    },
    Wsl {
        distro_name: String,
        user: Option<String>,
    },
    Docker {
        container_id: String,
        name: String,
        remote_user: String,
    },
    #[cfg(any(test, feature = "test-support"))]
    Mock { id: u64 },
}

impl From<RemoteConnectionIdentity> for CollaborativeRemoteTarget {
    fn from(identity: RemoteConnectionIdentity) -> Self {
        match identity {
            RemoteConnectionIdentity::Ssh {
                host,
                username,
                port,
            } => Self::Ssh {
                host,
                username,
                port,
            },
            RemoteConnectionIdentity::Wsl { distro_name, user } => Self::Wsl { distro_name, user },
            RemoteConnectionIdentity::Docker {
                container_id,
                name,
                remote_user,
            } => Self::Docker {
                container_id,
                name,
                remote_user,
            },
            #[cfg(any(test, feature = "test-support"))]
            RemoteConnectionIdentity::Mock { id } => Self::Mock { id },
        }
    }
}

impl CollaborativeProjectTarget {
    pub fn from_project_group_key(project: &ProjectGroupKey) -> Self {
        let host = project.host();
        Self {
            paths: project.path_list().ordered_paths().cloned().collect(),
            host: host
                .as_ref()
                .map(remote_connection_identity)
                .map(Into::into),
        }
    }

    pub fn matches(&self, project: &ProjectGroupKey) -> bool {
        let host = project.host();
        PathList::new(&self.paths) == *project.path_list()
            && self.host
                == host
                    .as_ref()
                    .map(remote_connection_identity)
                    .map(Into::into)
    }

    fn validate(&self) -> bool {
        self.paths.iter().all(|path| path.is_absolute())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CollaborativeNavigationTarget {
    Community {
        community_id: String,
    },
    Project {
        project: CollaborativeProjectTarget,
    },
    Repository {
        project: CollaborativeProjectTarget,
        work_directory: PathBuf,
    },
    Worktree {
        project: CollaborativeProjectTarget,
        worktree_id: u64,
        path: PathBuf,
    },
    Channel {
        channel_id: String,
    },
    DirectMessage {
        community_id: String,
        dm_id: String,
    },
    Forum {
        community_id: String,
        channel_id: String,
    },
    Thread {
        thread_id: String,
    },
    Message {
        channel_id: String,
        event_id: String,
        thread_root_id: Option<String>,
    },
    HostedRepository {
        owner: String,
        d_tag: String,
    },
    HostedProject {
        owner: String,
        d_tag: String,
    },
    PullRequest {
        event_id: String,
        owner: String,
        repository_d_tag: String,
    },
    Issue {
        event_id: String,
        owner: String,
        repository_d_tag: String,
    },
}

impl CollaborativeNavigationTarget {
    pub fn project(project: &ProjectGroupKey) -> Self {
        Self::Project {
            project: CollaborativeProjectTarget::from_project_group_key(project),
        }
    }

    pub fn repository(project: &ProjectGroupKey, work_directory: PathBuf) -> Self {
        Self::Repository {
            project: CollaborativeProjectTarget::from_project_group_key(project),
            work_directory,
        }
    }

    pub fn worktree(project: &ProjectGroupKey, worktree_id: u64, path: PathBuf) -> Self {
        Self::Worktree {
            project: CollaborativeProjectTarget::from_project_group_key(project),
            worktree_id,
            path,
        }
    }

    pub fn channel(channel_id: impl Into<String>) -> Self {
        Self::Channel {
            channel_id: channel_id.into(),
        }
    }

    pub fn thread(thread_id: impl Into<String>) -> Self {
        Self::Thread {
            thread_id: thread_id.into(),
        }
    }

    pub fn direct_message(community_id: impl Into<String>, dm_id: impl Into<String>) -> Self {
        Self::DirectMessage {
            community_id: community_id.into(),
            dm_id: dm_id.into(),
        }
    }

    pub fn forum(community_id: impl Into<String>, channel_id: impl Into<String>) -> Self {
        Self::Forum {
            community_id: community_id.into(),
            channel_id: channel_id.into(),
        }
    }

    fn validate(&self) -> bool {
        match self {
            Self::Community { community_id } => valid_opaque_id(community_id),
            Self::Project { project } => project.validate(),
            Self::Repository {
                project,
                work_directory,
            } => project.validate() && work_directory.is_absolute(),
            Self::Worktree { project, path, .. } => project.validate() && path.is_absolute(),
            Self::Channel { channel_id } => valid_opaque_id(channel_id),
            Self::DirectMessage {
                community_id,
                dm_id,
            } => valid_opaque_id(community_id) && valid_opaque_id(dm_id),
            Self::Forum {
                community_id,
                channel_id,
            } => valid_opaque_id(community_id) && valid_opaque_id(channel_id),
            Self::Thread { thread_id } => valid_opaque_id(thread_id),
            Self::Message {
                channel_id,
                event_id,
                thread_root_id,
            } => {
                valid_opaque_id(channel_id)
                    && valid_hex_identifier(event_id)
                    && thread_root_id
                        .as_ref()
                        .is_none_or(|thread_root_id| valid_hex_identifier(thread_root_id))
            }
            Self::HostedRepository { owner, d_tag } | Self::HostedProject { owner, d_tag } => {
                valid_hex_identifier(owner) && valid_d_tag(d_tag)
            }
            Self::PullRequest {
                event_id,
                owner,
                repository_d_tag,
            }
            | Self::Issue {
                event_id,
                owner,
                repository_d_tag,
            } => {
                valid_hex_identifier(event_id)
                    && valid_hex_identifier(owner)
                    && valid_d_tag(repository_d_tag)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CollaborativeNavigationError {
    InvalidTarget,
    MissingTarget(Box<CollaborativeNavigationTarget>),
    NoBackwardHistory,
    NoForwardHistory,
}

impl fmt::Display for CollaborativeNavigationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTarget => write!(formatter, "invalid collaborative navigation target"),
            Self::MissingTarget(_) => {
                write!(formatter, "collaborative navigation target is missing")
            }
            Self::NoBackwardHistory => {
                write!(formatter, "no backward collaborative navigation history")
            }
            Self::NoForwardHistory => {
                write!(formatter, "no forward collaborative navigation history")
            }
        }
    }
}

impl std::error::Error for CollaborativeNavigationError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CollaborativeEntityLinkError {
    TooLong,
    InvalidUrl,
    UnsupportedScheme,
    UnsupportedEntity,
    UnexpectedAuthority,
    UnexpectedPath,
    UnexpectedFragment,
    UnknownParameter,
    DuplicateParameter,
    MissingParameter,
    InvalidIdentifier,
}

impl fmt::Display for CollaborativeEntityLinkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "unsafe or unsupported collaborative entity link: {self:?}"
        )
    }
}

impl std::error::Error for CollaborativeEntityLinkError {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct PersistedCollaborativeNavigation {
    version: u32,
    current: Option<CollaborativeNavigationTarget>,
    backward: Vec<CollaborativeNavigationTarget>,
    forward: Vec<CollaborativeNavigationTarget>,
    #[serde(default)]
    pinned: Vec<CollaborativeNavigationTarget>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CollaborativeNavigation {
    current: Option<CollaborativeNavigationTarget>,
    backward: Vec<CollaborativeNavigationTarget>,
    forward: Vec<CollaborativeNavigationTarget>,
    pinned: Vec<CollaborativeNavigationTarget>,
}

impl CollaborativeNavigation {
    pub fn current(&self) -> Option<&CollaborativeNavigationTarget> {
        self.current.as_ref()
    }

    pub fn can_go_backward(&self) -> bool {
        !self.backward.is_empty()
    }

    pub fn can_go_forward(&self) -> bool {
        !self.forward.is_empty()
    }

    pub fn pinned(&self) -> &[CollaborativeNavigationTarget] {
        &self.pinned
    }

    pub fn is_pinned(&self, target: &CollaborativeNavigationTarget) -> bool {
        self.pinned.contains(target)
    }

    pub fn toggle_pin(
        &mut self,
        target: CollaborativeNavigationTarget,
    ) -> std::result::Result<bool, CollaborativeNavigationError> {
        if !target.validate() {
            return Err(CollaborativeNavigationError::InvalidTarget);
        }
        if let Some(index) = self.pinned.iter().position(|pinned| pinned == &target) {
            self.pinned.remove(index);
            return Ok(false);
        }
        if self.pinned.len() == MAX_PINNED_ENTRIES {
            self.pinned.remove(0);
        }
        self.pinned.push(target);
        Ok(true)
    }

    pub fn navigate_to(
        &mut self,
        target: CollaborativeNavigationTarget,
        is_available: impl FnOnce(&CollaborativeNavigationTarget) -> bool,
    ) -> std::result::Result<bool, CollaborativeNavigationError> {
        if !target.validate() {
            return Err(CollaborativeNavigationError::InvalidTarget);
        }
        if !is_available(&target) {
            return Err(CollaborativeNavigationError::MissingTarget(Box::new(
                target,
            )));
        }
        if self.current.as_ref() == Some(&target) {
            return Ok(false);
        }
        if let Some(current) = self.current.replace(target) {
            push_bounded(&mut self.backward, current);
        }
        self.forward.clear();
        Ok(true)
    }

    pub fn go_backward(
        &mut self,
        is_available: impl FnOnce(&CollaborativeNavigationTarget) -> bool,
    ) -> std::result::Result<&CollaborativeNavigationTarget, CollaborativeNavigationError> {
        let target = self
            .backward
            .last()
            .cloned()
            .ok_or(CollaborativeNavigationError::NoBackwardHistory)?;
        if !is_available(&target) {
            return Err(CollaborativeNavigationError::MissingTarget(Box::new(
                target,
            )));
        }
        self.backward.pop();
        if let Some(current) = self.current.replace(target) {
            push_bounded(&mut self.forward, current);
        }
        self.current
            .as_ref()
            .ok_or(CollaborativeNavigationError::NoBackwardHistory)
    }

    pub fn go_forward(
        &mut self,
        is_available: impl FnOnce(&CollaborativeNavigationTarget) -> bool,
    ) -> std::result::Result<&CollaborativeNavigationTarget, CollaborativeNavigationError> {
        let target = self
            .forward
            .last()
            .cloned()
            .ok_or(CollaborativeNavigationError::NoForwardHistory)?;
        if !is_available(&target) {
            return Err(CollaborativeNavigationError::MissingTarget(Box::new(
                target,
            )));
        }
        self.forward.pop();
        if let Some(current) = self.current.replace(target) {
            push_bounded(&mut self.backward, current);
        }
        self.current
            .as_ref()
            .ok_or(CollaborativeNavigationError::NoForwardHistory)
    }

    fn persisted(&self) -> PersistedCollaborativeNavigation {
        PersistedCollaborativeNavigation {
            version: COLLABORATIVE_NAVIGATION_VERSION,
            current: self.current.clone(),
            backward: self.backward.clone(),
            forward: self.forward.clone(),
            pinned: self.pinned.clone(),
        }
    }

    fn from_persisted(
        persisted: PersistedCollaborativeNavigation,
    ) -> std::result::Result<Self, CollaborativeNavigationError> {
        if persisted.version != COLLABORATIVE_NAVIGATION_VERSION
            || persisted.backward.len() > MAX_HISTORY_ENTRIES
            || persisted.forward.len() > MAX_HISTORY_ENTRIES
            || persisted.pinned.len() > MAX_PINNED_ENTRIES
            || persisted
                .current
                .as_ref()
                .is_some_and(|target| !target.validate())
            || persisted.backward.iter().any(|target| !target.validate())
            || persisted.forward.iter().any(|target| !target.validate())
            || persisted.pinned.iter().any(|target| !target.validate())
            || persisted
                .pinned
                .iter()
                .enumerate()
                .any(|(index, target)| persisted.pinned[index + 1..].contains(target))
        {
            return Err(CollaborativeNavigationError::InvalidTarget);
        }
        Ok(Self {
            current: persisted.current,
            backward: persisted.backward,
            forward: persisted.forward,
            pinned: persisted.pinned,
        })
    }
}

pub fn target_from_entity_link(
    entity_link: &str,
) -> std::result::Result<CollaborativeNavigationTarget, CollaborativeEntityLinkError> {
    if entity_link.len() > MAX_ENTITY_LINK_LENGTH {
        return Err(CollaborativeEntityLinkError::TooLong);
    }
    let url = Url::parse(entity_link).map_err(|_| CollaborativeEntityLinkError::InvalidUrl)?;
    if url.scheme() != "buzz" {
        return Err(CollaborativeEntityLinkError::UnsupportedScheme);
    }
    if !url.username().is_empty() || url.password().is_some() || url.port().is_some() {
        return Err(CollaborativeEntityLinkError::UnexpectedAuthority);
    }
    if !matches!(url.path(), "" | "/") {
        return Err(CollaborativeEntityLinkError::UnexpectedPath);
    }
    if url.fragment().is_some() {
        return Err(CollaborativeEntityLinkError::UnexpectedFragment);
    }

    let entity = url
        .host_str()
        .ok_or(CollaborativeEntityLinkError::UnsupportedEntity)?;
    let allowed_parameters: &[&str] = match entity {
        "message" => &["channel", "id", "thread"],
        "repo" | "project" => &["owner", "d"],
        "pr" | "issue" => &["id", "owner", "d"],
        _ => return Err(CollaborativeEntityLinkError::UnsupportedEntity),
    };
    let mut parameters = std::collections::HashMap::<String, String>::new();
    for (name, value) in url.query_pairs() {
        if !allowed_parameters.contains(&name.as_ref()) {
            return Err(CollaborativeEntityLinkError::UnknownParameter);
        }
        if parameters
            .insert(name.into_owned(), value.into_owned())
            .is_some()
        {
            return Err(CollaborativeEntityLinkError::DuplicateParameter);
        }
    }

    let required = |name: &str| {
        parameters
            .get(name)
            .filter(|value| !value.is_empty())
            .cloned()
            .ok_or(CollaborativeEntityLinkError::MissingParameter)
    };
    let target = match entity {
        "message" => CollaborativeNavigationTarget::Message {
            channel_id: required("channel")?,
            event_id: lowercase_hex(required("id")?)?,
            thread_root_id: parameters
                .get("thread")
                .filter(|value| !value.is_empty())
                .cloned()
                .map(lowercase_hex)
                .transpose()?,
        },
        "repo" => CollaborativeNavigationTarget::HostedRepository {
            owner: lowercase_hex(required("owner")?)?,
            d_tag: required("d")?,
        },
        "project" => CollaborativeNavigationTarget::HostedProject {
            owner: lowercase_hex(required("owner")?)?,
            d_tag: required("d")?,
        },
        "pr" => CollaborativeNavigationTarget::PullRequest {
            event_id: lowercase_hex(required("id")?)?,
            owner: lowercase_hex(required("owner")?)?,
            repository_d_tag: required("d")?,
        },
        "issue" => CollaborativeNavigationTarget::Issue {
            event_id: lowercase_hex(required("id")?)?,
            owner: lowercase_hex(required("owner")?)?,
            repository_d_tag: required("d")?,
        },
        _ => return Err(CollaborativeEntityLinkError::UnsupportedEntity),
    };
    target
        .validate()
        .then_some(target)
        .ok_or(CollaborativeEntityLinkError::InvalidIdentifier)
}

pub(crate) fn read_collaborative_navigation(
    key_value_store: &KeyValueStore,
    workspace_id: WorkspaceId,
) -> CollaborativeNavigation {
    key_value_store
        .scoped(COLLABORATIVE_NAVIGATION_NAMESPACE)
        .read(&workspace_id.0.to_string())
        .log_err()
        .flatten()
        .and_then(|serialized| {
            serde_json::from_str::<PersistedCollaborativeNavigation>(&serialized).log_err()
        })
        .and_then(|persisted| CollaborativeNavigation::from_persisted(persisted).log_err())
        .unwrap_or_default()
}

pub(crate) async fn write_collaborative_navigation(
    key_value_store: &KeyValueStore,
    workspace_id: WorkspaceId,
    navigation: &CollaborativeNavigation,
) -> Result<()> {
    let serialized = serde_json::to_string(&navigation.persisted())?;
    key_value_store
        .scoped(COLLABORATIVE_NAVIGATION_NAMESPACE)
        .write(workspace_id.0.to_string(), serialized)
        .await
}

fn push_bounded(
    history: &mut Vec<CollaborativeNavigationTarget>,
    target: CollaborativeNavigationTarget,
) {
    if history.len() == MAX_HISTORY_ENTRIES {
        history.remove(0);
    }
    history.push(target);
}

fn valid_opaque_id(identifier: &str) -> bool {
    !identifier.is_empty()
        && identifier.len() <= MAX_OPAQUE_ID_LENGTH
        && !identifier.chars().any(char::is_control)
}

fn valid_hex_identifier(identifier: &str) -> bool {
    identifier.len() == 64 && identifier.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn lowercase_hex(identifier: String) -> std::result::Result<String, CollaborativeEntityLinkError> {
    valid_hex_identifier(&identifier)
        .then(|| identifier.to_ascii_lowercase())
        .ok_or(CollaborativeEntityLinkError::InvalidIdentifier)
}

fn valid_d_tag(d_tag: &str) -> bool {
    !d_tag.is_empty()
        && d_tag.len() <= 64
        && !d_tag.starts_with('.')
        && !d_tag.contains("..")
        && d_tag
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::init_test;
    use remote::{RemoteConnectionOptions, SshConnectionOptions};

    fn thread_target(id: &str) -> CollaborativeNavigationTarget {
        CollaborativeNavigationTarget::thread(id)
    }

    #[test]
    fn collaborative_navigation_tracks_back_and_forward_without_duplicate_visits() {
        let mut navigation = CollaborativeNavigation::default();
        assert!(
            navigation
                .navigate_to(thread_target("one"), |_| true)
                .expect("first target should resolve")
        );
        assert!(
            navigation
                .navigate_to(thread_target("two"), |_| true)
                .expect("second target should resolve")
        );
        assert!(
            !navigation
                .navigate_to(thread_target("two"), |_| true)
                .expect("duplicate target should remain valid")
        );
        assert_eq!(
            navigation
                .go_backward(|_| true)
                .expect("back target should resolve"),
            &thread_target("one")
        );
        assert_eq!(
            navigation
                .go_forward(|_| true)
                .expect("forward target should resolve"),
            &thread_target("two")
        );
    }

    #[test]
    fn collaborative_navigation_rejects_missing_targets_without_mutation() {
        let mut navigation = CollaborativeNavigation::default();
        navigation
            .navigate_to(thread_target("available"), |_| true)
            .expect("available target should resolve");
        let before = navigation.clone();
        assert_eq!(
            navigation.navigate_to(thread_target("missing"), |_| false),
            Err(CollaborativeNavigationError::MissingTarget(Box::new(
                thread_target("missing")
            )))
        );
        assert_eq!(navigation, before);
    }

    #[gpui::test]
    async fn collaborative_navigation_restart(cx: &mut gpui::TestAppContext) {
        init_test(cx);
        let key_value_store = cx.update(|cx| KeyValueStore::global(cx));
        let workspace_id = WorkspaceId(8101);
        let mut navigation = CollaborativeNavigation::default();
        navigation
            .navigate_to(thread_target("one"), |_| true)
            .expect("first target should resolve");
        navigation
            .navigate_to(thread_target("two"), |_| true)
            .expect("second target should resolve");
        navigation
            .go_backward(|_| true)
            .expect("back target should resolve");
        let pinned_thread = thread_target("pinned-thread");
        assert!(
            navigation
                .toggle_pin(pinned_thread.clone())
                .expect("valid target should pin")
        );

        write_collaborative_navigation(&key_value_store, workspace_id, &navigation)
            .await
            .expect("navigation should persist");
        let restored = read_collaborative_navigation(&key_value_store, workspace_id);
        assert_eq!(restored, navigation);
        assert_eq!(restored.current(), Some(&thread_target("one")));
        assert!(restored.can_go_forward());
        assert_eq!(restored.pinned(), &[pinned_thread]);

        key_value_store
            .scoped(COLLABORATIVE_NAVIGATION_NAMESPACE)
            .write(workspace_id.0.to_string(), "not-json".to_owned())
            .await
            .expect("malformed fixture should write");
        assert_eq!(
            read_collaborative_navigation(&key_value_store, workspace_id),
            CollaborativeNavigation::default()
        );

        let mut future = navigation.persisted();
        future.version = COLLABORATIVE_NAVIGATION_VERSION + 1;
        key_value_store
            .scoped(COLLABORATIVE_NAVIGATION_NAMESPACE)
            .write(
                workspace_id.0.to_string(),
                serde_json::to_string(&future).expect("future fixture should serialize"),
            )
            .await
            .expect("future fixture should write");
        assert_eq!(
            read_collaborative_navigation(&key_value_store, workspace_id),
            CollaborativeNavigation::default()
        );
    }

    #[test]
    fn collaborative_navigation_toggles_and_bounds_pins() {
        let mut navigation = CollaborativeNavigation::default();
        let target = thread_target("thread-1");
        assert!(
            navigation
                .toggle_pin(target.clone())
                .expect("valid target should pin")
        );
        assert!(navigation.is_pinned(&target));
        assert!(
            !navigation
                .toggle_pin(target.clone())
                .expect("pinned target should unpin")
        );
        assert!(!navigation.is_pinned(&target));

        for index in 0..=MAX_PINNED_ENTRIES {
            navigation
                .toggle_pin(thread_target(&format!("thread-{index}")))
                .expect("bounded target should pin");
        }
        assert_eq!(navigation.pinned().len(), MAX_PINNED_ENTRIES);
        assert!(!navigation.is_pinned(&thread_target("thread-0")));
        assert!(navigation.is_pinned(&thread_target(&format!("thread-{MAX_PINNED_ENTRIES}"))));
    }

    #[test]
    fn collaborative_navigation_accepts_direct_message_and_forum_targets() {
        let mut navigation = CollaborativeNavigation::default();
        let dm = CollaborativeNavigationTarget::direct_message("community-1", "dm-1");
        let forum = CollaborativeNavigationTarget::forum("community-1", "forum-1");
        assert!(
            navigation
                .navigate_to(dm.clone(), |_| true)
                .expect("direct message target should resolve")
        );
        assert!(
            navigation
                .navigate_to(forum.clone(), |_| true)
                .expect("forum target should resolve")
        );
        assert_eq!(
            navigation
                .go_backward(|_| true)
                .expect("back target should resolve"),
            &dm
        );
        assert!(
            navigation
                .toggle_pin(dm.clone())
                .expect("valid direct message target should pin")
        );
        assert!(
            navigation
                .toggle_pin(forum.clone())
                .expect("valid forum target should pin")
        );
        assert_eq!(navigation.pinned(), &[dm, forum]);
    }

    #[test]
    fn collaborative_navigation_rejects_malformed_direct_message_and_forum_targets() {
        let mut navigation = CollaborativeNavigation::default();
        for target in [
            CollaborativeNavigationTarget::direct_message("", "dm-1"),
            CollaborativeNavigationTarget::direct_message("community-1", ""),
            CollaborativeNavigationTarget::direct_message("community-1", "bad\nid"),
            CollaborativeNavigationTarget::forum("", "forum-1"),
            CollaborativeNavigationTarget::forum("community-1", ""),
            CollaborativeNavigationTarget::forum("community-1", "\u{7}forum"),
        ] {
            assert_eq!(
                navigation.navigate_to(target, |_| true),
                Err(CollaborativeNavigationError::InvalidTarget),
                "malformed target should be rejected"
            );
        }
    }

    #[test]
    fn collaborative_navigation_round_trips_direct_message_and_forum_targets() {
        for (target, kind) in [
            (
                CollaborativeNavigationTarget::direct_message("community-1", "dm-1"),
                "direct_message",
            ),
            (
                CollaborativeNavigationTarget::forum("community-1", "forum-1"),
                "forum",
            ),
        ] {
            let serialized = serde_json::to_string(&target).expect("target should serialize");
            assert!(
                serialized.contains(&format!("\"kind\":\"{kind}\"")),
                "serialized target should use its discriminated kind: {serialized}"
            );
            let deserialized = serde_json::from_str::<CollaborativeNavigationTarget>(&serialized)
                .expect("serialized target should deserialize");
            assert_eq!(
                deserialized, target,
                "serialized target should round-trip: {serialized}"
            );
        }
    }

    #[test]
    fn collaborative_navigation_parses_supported_buzz_entity_links() {
        let owner = "A".repeat(64);
        let event_id = "B".repeat(64);
        assert_eq!(
            target_from_entity_link(&format!("buzz://pr?id={event_id}&owner={owner}&d=zed-main")),
            Ok(CollaborativeNavigationTarget::PullRequest {
                event_id: event_id.to_ascii_lowercase(),
                owner: owner.to_ascii_lowercase(),
                repository_d_tag: "zed-main".to_owned(),
            })
        );
        assert_eq!(
            target_from_entity_link(&format!(
                "buzz://message?channel=community&id={event_id}&thread={owner}"
            )),
            Ok(CollaborativeNavigationTarget::Message {
                channel_id: "community".to_owned(),
                event_id: event_id.to_ascii_lowercase(),
                thread_root_id: Some(owner.to_ascii_lowercase()),
            })
        );
    }

    #[test]
    fn collaborative_navigation_rejects_unsafe_entity_links() {
        let owner = "a".repeat(64);
        let event_id = "b".repeat(64);
        for link in [
            format!("https://repo?owner={owner}&d=zed"),
            format!("buzz://repo/extra?owner={owner}&d=zed"),
            format!("buzz://repo?owner={owner}&d=zed#fragment"),
            format!("buzz://repo?owner={owner}&owner={owner}&d=zed"),
            format!("buzz://repo?owner={owner}&d=../zed"),
            format!("buzz://pr?id=short&owner={owner}&d=zed"),
            format!("buzz://pr?id={event_id}&owner={owner}&d=zed&relay=wss://relay"),
        ] {
            assert!(
                target_from_entity_link(&link).is_err(),
                "unsafe link should be rejected: {link}"
            );
        }
    }

    #[test]
    fn collaborative_navigation_project_identity_never_persists_credentials() {
        let project = ProjectGroupKey::new(
            Some(RemoteConnectionOptions::Ssh(SshConnectionOptions {
                host: "example.test".into(),
                username: Some("builder".to_owned()),
                port: Some(2222),
                password: Some("do-not-persist".to_owned()),
                ..Default::default()
            })),
            PathList::new(&[PathBuf::from("/workspace/project")]),
        );
        let target = CollaborativeNavigationTarget::project(&project);
        let serialized = serde_json::to_string(&target).expect("target should serialize");

        assert!(!serialized.contains("do-not-persist"));
        assert!(!serialized.contains("password"));
        assert!(
            matches!(
                &target,
                CollaborativeNavigationTarget::Project { project: target_project }
                    if target_project.matches(&project)
            ),
            "safe persisted identity should still match the canonical project group"
        );
    }
}
