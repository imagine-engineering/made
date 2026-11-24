use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use crate::{ActiveReaction, CommunityId, MessageSource, NostrEventId, PrincipalId, ReactionGroup};

pub const MAX_CUSTOM_EMOJI_SHORTCODE_BYTES: usize = 64;
pub const MAX_CUSTOM_EMOJI_ASSET_URL_BYTES: usize = 2_048;
pub const MAX_CUSTOM_EMOJI_PER_SET: usize = 1_000;
pub const MAX_CUSTOM_EMOJI_SET_RECORDS: usize = 10_000;
pub const MAX_REACTION_CUSTOM_EMOJI_TAGS: usize = 10_000;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CustomEmojiShortcode(String);

impl CustomEmojiShortcode {
    pub fn new(value: impl AsRef<str>) -> Result<Self, CustomEmojiError> {
        let value = value.as_ref().trim().trim_matches(':').to_ascii_lowercase();
        if value.is_empty()
            || value.len() > MAX_CUSTOM_EMOJI_SHORTCODE_BYTES
            || !value.chars().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '-' | '_')
            })
        {
            return Err(CustomEmojiError::InvalidShortcode);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn reaction_value(&self) -> String {
        format!(":{}:", self.0)
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CustomEmojiAsset(String);

impl CustomEmojiAsset {
    pub fn new(value: impl Into<String>) -> Result<Self, CustomEmojiError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_CUSTOM_EMOJI_ASSET_URL_BYTES
            || value.chars().any(char::is_control)
            || (!value.starts_with("http://") && !value.starts_with("https://"))
        {
            return Err(CustomEmojiError::InvalidAsset);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CustomEmoji {
    pub shortcode: CustomEmojiShortcode,
    pub asset: CustomEmojiAsset,
}

impl CustomEmoji {
    pub fn new(
        shortcode: impl AsRef<str>,
        asset: impl Into<String>,
    ) -> Result<Self, CustomEmojiError> {
        Ok(Self {
            shortcode: CustomEmojiShortcode::new(shortcode)?,
            asset: CustomEmojiAsset::new(asset)?,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CustomEmojiSetRecord {
    community_id: CommunityId,
    owner_principal_id: PrincipalId,
    source: MessageSource,
    emoji: Vec<CustomEmoji>,
}

impl CustomEmojiSetRecord {
    pub fn new(
        community_id: CommunityId,
        owner_principal_id: PrincipalId,
        source: MessageSource,
        emoji: Vec<CustomEmoji>,
    ) -> Result<Self, CustomEmojiError> {
        if community_id.as_uuid().is_nil() || owner_principal_id.as_uuid().is_nil() {
            return Err(CustomEmojiError::InvalidIdentity);
        }
        source
            .validate()
            .map_err(|_| CustomEmojiError::InvalidSource)?;
        if emoji.len() > MAX_CUSTOM_EMOJI_PER_SET {
            return Err(CustomEmojiError::TooManyEmoji);
        }
        let mut shortcodes = BTreeSet::new();
        if emoji
            .iter()
            .any(|entry| !shortcodes.insert(entry.shortcode.clone()))
        {
            return Err(CustomEmojiError::DuplicateShortcode);
        }
        Ok(Self {
            community_id,
            owner_principal_id,
            source,
            emoji,
        })
    }

    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub const fn owner_principal_id(&self) -> PrincipalId {
        self.owner_principal_id
    }

    pub const fn source(&self) -> MessageSource {
        self.source
    }

    pub fn emoji(&self) -> &[CustomEmoji] {
        &self.emoji
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CustomEmojiPaletteEntry {
    pub emoji: CustomEmoji,
    pub owner_principal_id: PrincipalId,
    pub source: MessageSource,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CustomEmojiPalette {
    community_id: CommunityId,
    entries: BTreeMap<CustomEmojiShortcode, CustomEmojiPaletteEntry>,
}

impl CustomEmojiPalette {
    pub fn build(
        community_id: CommunityId,
        records: impl IntoIterator<Item = CustomEmojiSetRecord>,
    ) -> Result<Self, CustomEmojiError> {
        if community_id.as_uuid().is_nil() {
            return Err(CustomEmojiError::InvalidIdentity);
        }
        let mut records_by_source = BTreeMap::new();
        for (index, record) in records.into_iter().enumerate() {
            if index >= MAX_CUSTOM_EMOJI_SET_RECORDS {
                return Err(CustomEmojiError::TooManySetRecords);
            }
            if record.community_id != community_id {
                return Err(CustomEmojiError::CommunityMismatch);
            }
            match records_by_source.entry(record.source.event_id) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(record);
                }
                std::collections::btree_map::Entry::Occupied(entry) if entry.get() == &record => {}
                std::collections::btree_map::Entry::Occupied(_) => {
                    return Err(CustomEmojiError::ConflictingSource);
                }
            }
        }

        let mut latest_by_owner = BTreeMap::<PrincipalId, CustomEmojiSetRecord>::new();
        for record in records_by_source.into_values() {
            match latest_by_owner.entry(record.owner_principal_id) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(record);
                }
                std::collections::btree_map::Entry::Occupied(mut entry)
                    if set_record_replaces(&record, entry.get()) =>
                {
                    entry.insert(record);
                }
                std::collections::btree_map::Entry::Occupied(_) => {}
            }
        }

        let mut entries = BTreeMap::<CustomEmojiShortcode, CustomEmojiPaletteEntry>::new();
        for record in latest_by_owner.into_values() {
            for emoji in record.emoji {
                let candidate = CustomEmojiPaletteEntry {
                    emoji,
                    owner_principal_id: record.owner_principal_id,
                    source: record.source,
                };
                match entries.entry(candidate.emoji.shortcode.clone()) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(candidate);
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry)
                        if palette_entry_replaces(&candidate, entry.get()) =>
                    {
                        entry.insert(candidate);
                    }
                    std::collections::btree_map::Entry::Occupied(_) => {}
                }
            }
        }
        Ok(Self {
            community_id,
            entries,
        })
    }

    pub const fn community_id(&self) -> CommunityId {
        self.community_id
    }

    pub fn entries(&self) -> impl Iterator<Item = &CustomEmojiPaletteEntry> {
        self.entries.values()
    }

    pub fn get(&self, shortcode: &CustomEmojiShortcode) -> Option<&CustomEmojiPaletteEntry> {
        self.entries.get(shortcode)
    }

    pub fn resolve_reaction_group(
        &self,
        group: &ReactionGroup,
        tags: impl IntoIterator<Item = ReactionCustomEmojiTag>,
    ) -> Result<ResolvedReactionGroup, CustomEmojiError> {
        let Some(shortcode) = reaction_shortcode(group.value.as_str())? else {
            if tags.into_iter().next().is_some() {
                return Err(CustomEmojiError::UnexpectedReactionTag);
            }
            return Ok(ResolvedReactionGroup {
                value: group.value.as_str().to_owned(),
                count: group.count(),
                presentation: ResolvedReactionPresentation::Text,
            });
        };

        let active_sources = group
            .reactions
            .iter()
            .map(|reaction| reaction.added_source.event_id)
            .collect::<BTreeSet<_>>();
        let mut tags_by_source = BTreeMap::new();
        for (index, tag) in tags.into_iter().enumerate() {
            if index >= MAX_REACTION_CUSTOM_EMOJI_TAGS {
                return Err(CustomEmojiError::TooManyReactionTags);
            }
            if tag.shortcode != shortcode || !active_sources.contains(&tag.reaction_event_id) {
                return Err(CustomEmojiError::ReactionTagMismatch);
            }
            match tags_by_source.entry(tag.reaction_event_id) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(tag);
                }
                std::collections::btree_map::Entry::Occupied(entry) if entry.get() == &tag => {}
                std::collections::btree_map::Entry::Occupied(_) => {
                    return Err(CustomEmojiError::ConflictingReactionTag);
                }
            }
        }

        let embedded = earliest_tagged_reaction(&group.reactions, &tags_by_source);
        let (asset, source) = if let Some(tag) = embedded {
            (
                Some(tag.asset.clone()),
                CustomEmojiResolutionSource::ReactionEvent(tag.reaction_event_id),
            )
        } else if group.value.as_str().chars().count() > 64 {
            return Err(CustomEmojiError::MissingLongReactionTag);
        } else if let Some(entry) = self.entries.get(&shortcode) {
            (
                Some(entry.emoji.asset.clone()),
                CustomEmojiResolutionSource::CommunityPalette(entry.source.event_id),
            )
        } else {
            (None, CustomEmojiResolutionSource::Missing)
        };
        Ok(ResolvedReactionGroup {
            value: group.value.as_str().to_owned(),
            count: group.count(),
            presentation: ResolvedReactionPresentation::Custom {
                shortcode,
                asset,
                source,
            },
        })
    }
}

fn set_record_replaces(candidate: &CustomEmojiSetRecord, current: &CustomEmojiSetRecord) -> bool {
    candidate.source.event_created_at > current.source.event_created_at
        || (candidate.source.event_created_at == current.source.event_created_at
            && candidate.source.event_id < current.source.event_id)
}

fn palette_entry_replaces(
    candidate: &CustomEmojiPaletteEntry,
    current: &CustomEmojiPaletteEntry,
) -> bool {
    candidate.source.event_created_at > current.source.event_created_at
        || (candidate.source.event_created_at == current.source.event_created_at
            && candidate.emoji.asset < current.emoji.asset)
}

fn reaction_shortcode(value: &str) -> Result<Option<CustomEmojiShortcode>, CustomEmojiError> {
    if !value.starts_with(':') || !value.ends_with(':') {
        return Ok(None);
    }
    let shortcode = CustomEmojiShortcode::new(value)?;
    if shortcode.reaction_value() != value {
        return Err(CustomEmojiError::NonCanonicalReactionValue);
    }
    Ok(Some(shortcode))
}

fn earliest_tagged_reaction<'a>(
    reactions: &[ActiveReaction],
    tags: &'a BTreeMap<NostrEventId, ReactionCustomEmojiTag>,
) -> Option<&'a ReactionCustomEmojiTag> {
    reactions
        .iter()
        .filter_map(|reaction| {
            tags.get(&reaction.added_source.event_id)
                .map(|tag| (reaction.added_source, tag))
        })
        .min_by(|(left, _), (right, _)| {
            left.event_created_at
                .cmp(&right.event_created_at)
                .then_with(|| left.event_id.cmp(&right.event_id))
        })
        .map(|(_, tag)| tag)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReactionCustomEmojiTag {
    pub reaction_event_id: NostrEventId,
    pub shortcode: CustomEmojiShortcode,
    pub asset: CustomEmojiAsset,
}

impl ReactionCustomEmojiTag {
    pub fn new(
        reaction_event_id: NostrEventId,
        shortcode: impl AsRef<str>,
        asset: impl Into<String>,
    ) -> Result<Self, CustomEmojiError> {
        if reaction_event_id.as_bytes().iter().all(|byte| *byte == 0) {
            return Err(CustomEmojiError::InvalidSource);
        }
        Ok(Self {
            reaction_event_id,
            shortcode: CustomEmojiShortcode::new(shortcode)?,
            asset: CustomEmojiAsset::new(asset)?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CustomEmojiResolutionSource {
    ReactionEvent(NostrEventId),
    CommunityPalette(NostrEventId),
    Missing,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolvedReactionPresentation {
    Text,
    Custom {
        shortcode: CustomEmojiShortcode,
        asset: Option<CustomEmojiAsset>,
        source: CustomEmojiResolutionSource,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedReactionGroup {
    pub value: String,
    pub count: usize,
    pub presentation: ResolvedReactionPresentation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CustomEmojiError {
    InvalidShortcode,
    InvalidAsset,
    InvalidIdentity,
    InvalidSource,
    TooManyEmoji,
    DuplicateShortcode,
    TooManySetRecords,
    CommunityMismatch,
    ConflictingSource,
    UnexpectedReactionTag,
    TooManyReactionTags,
    ReactionTagMismatch,
    ConflictingReactionTag,
    NonCanonicalReactionValue,
    MissingLongReactionTag,
}

impl fmt::Display for CustomEmojiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShortcode => formatter.write_str("custom emoji shortcode is invalid"),
            Self::InvalidAsset => formatter.write_str("custom emoji asset is invalid"),
            Self::InvalidIdentity => formatter.write_str("custom emoji identity is invalid"),
            Self::InvalidSource => formatter.write_str("custom emoji source is invalid"),
            Self::TooManyEmoji => formatter.write_str("custom emoji set is too large"),
            Self::DuplicateShortcode => formatter.write_str("custom emoji set repeats a shortcode"),
            Self::TooManySetRecords => formatter.write_str("too many custom emoji set records"),
            Self::CommunityMismatch => formatter.write_str("custom emoji community does not match"),
            Self::ConflictingSource => formatter.write_str("custom emoji source conflicts"),
            Self::UnexpectedReactionTag => {
                formatter.write_str("text reaction carries a custom emoji tag")
            }
            Self::TooManyReactionTags => formatter.write_str("too many custom reaction tags"),
            Self::ReactionTagMismatch => formatter.write_str("custom reaction tag does not match"),
            Self::ConflictingReactionTag => formatter.write_str("custom reaction tag conflicts"),
            Self::NonCanonicalReactionValue => {
                formatter.write_str("custom reaction value is not canonical")
            }
            Self::MissingLongReactionTag => {
                formatter.write_str("long custom reaction has no embedded asset")
            }
        }
    }
}

impl Error for CustomEmojiError {}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;
    use crate::ReactionValue;

    fn community_id(value: u128) -> CommunityId {
        CommunityId::from_uuid(Uuid::from_u128(value))
    }

    fn principal_id(value: u128) -> PrincipalId {
        PrincipalId::from_uuid(Uuid::from_u128(value))
    }

    fn event_id(value: u8) -> NostrEventId {
        NostrEventId::from_bytes([value; 32])
    }

    fn source(value: u8, event_created_at: u64) -> MessageSource {
        MessageSource {
            event_id: event_id(value),
            event_created_at,
        }
    }

    fn emoji(shortcode: &str, asset: &str) -> CustomEmoji {
        CustomEmoji::new(shortcode, asset).expect("custom emoji")
    }

    fn record(
        owner: u128,
        source_value: u8,
        created_at: u64,
        emoji: Vec<CustomEmoji>,
    ) -> CustomEmojiSetRecord {
        CustomEmojiSetRecord::new(
            community_id(1),
            principal_id(owner),
            source(source_value, created_at),
            emoji,
        )
        .expect("custom emoji set")
    }

    #[test]
    fn duplicate_names_are_rejected_within_sets_and_collapsed_across_members() {
        assert_eq!(
            CustomEmojiSetRecord::new(
                community_id(1),
                principal_id(1),
                source(1, 10),
                vec![
                    emoji("Party_Parrot", "https://example.com/first.png"),
                    emoji(":party_parrot:", "https://example.com/second.png"),
                ],
            ),
            Err(CustomEmojiError::DuplicateShortcode)
        );

        let palette = CustomEmojiPalette::build(
            community_id(1),
            [
                record(1, 1, 10, vec![emoji("party", "https://example.com/z.png")]),
                record(2, 2, 10, vec![emoji("PARTY", "https://example.com/a.png")]),
            ],
        )
        .expect("community palette");
        let entries = palette.entries().collect::<Vec<_>>();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].emoji.shortcode.as_str(), "party");
        assert_eq!(entries[0].emoji.asset.as_str(), "https://example.com/a.png");
        assert_eq!(entries[0].owner_principal_id, principal_id(2));
    }

    #[test]
    fn invalid_media_references_fail_closed() {
        assert_eq!(
            CustomEmojiAsset::new(""),
            Err(CustomEmojiError::InvalidAsset)
        );
        assert_eq!(
            CustomEmojiAsset::new("file:///tmp/emoji.png"),
            Err(CustomEmojiError::InvalidAsset)
        );
        assert_eq!(
            CustomEmojiAsset::new("javascript:alert(1)"),
            Err(CustomEmojiError::InvalidAsset)
        );
        assert_eq!(
            CustomEmojiAsset::new("https://example.com/emoji\n.png"),
            Err(CustomEmojiError::InvalidAsset)
        );
        assert_eq!(
            CustomEmojiAsset::new(format!(
                "https://example.com/{}",
                "x".repeat(MAX_CUSTOM_EMOJI_ASSET_URL_BYTES)
            )),
            Err(CustomEmojiError::InvalidAsset)
        );
        assert!(CustomEmojiAsset::new("https://example.com/emoji.webp").is_ok());
        assert!(CustomEmojiAsset::new("http://relay.test/emoji.gif").is_ok());
    }

    #[test]
    fn a_newer_owner_set_removes_its_entry_without_hiding_another_owner() {
        let palette = CustomEmojiPalette::build(
            community_id(1),
            [
                record(
                    1,
                    1,
                    20,
                    vec![emoji("party", "https://example.com/owner-one.png")],
                ),
                record(1, 2, 30, Vec::new()),
                record(
                    2,
                    3,
                    10,
                    vec![emoji("party", "https://example.com/owner-two.png")],
                ),
            ],
        )
        .expect("community palette");
        let shortcode = CustomEmojiShortcode::new("party").expect("shortcode");
        let visible = palette.get(&shortcode).expect("other owner's emoji");
        assert_eq!(visible.owner_principal_id, principal_id(2));
        assert_eq!(
            visible.emoji.asset.as_str(),
            "https://example.com/owner-two.png"
        );

        let removed = CustomEmojiPalette::build(
            community_id(1),
            [
                record(
                    1,
                    1,
                    20,
                    vec![emoji("party", "https://example.com/owner-one.png")],
                ),
                record(1, 2, 30, Vec::new()),
            ],
        )
        .expect("removed palette");
        assert!(removed.get(&shortcode).is_none());
    }

    #[test]
    fn long_reactions_render_from_embedded_history_after_palette_removal() {
        let shortcode_text = "a".repeat(MAX_CUSTOM_EMOJI_SHORTCODE_BYTES);
        let shortcode = CustomEmojiShortcode::new(&shortcode_text).expect("shortcode");
        let value = ReactionValue::new(shortcode.reaction_value()).expect("long reaction");
        let reaction_source = source(9, 15);
        let group = ReactionGroup {
            value,
            reactions: vec![ActiveReaction {
                actor_principal_id: principal_id(3),
                added_source: reaction_source,
            }],
        };
        let original_group = group.clone();
        let palette = CustomEmojiPalette::build(
            community_id(1),
            [
                record(
                    1,
                    1,
                    10,
                    vec![emoji(&shortcode_text, "https://example.com/historical.png")],
                ),
                record(1, 2, 20, Vec::new()),
            ],
        )
        .expect("removed palette");
        let tag = ReactionCustomEmojiTag::new(
            reaction_source.event_id,
            &shortcode_text,
            "https://example.com/historical.png",
        )
        .expect("reaction tag");

        let resolved = palette
            .resolve_reaction_group(&group, [tag])
            .expect("historical reaction");
        assert_eq!(resolved.count, 1);
        assert!(matches!(
            resolved.presentation,
            ResolvedReactionPresentation::Custom {
                asset: Some(asset),
                source: CustomEmojiResolutionSource::ReactionEvent(event),
                ..
            } if asset.as_str() == "https://example.com/historical.png"
                && event == reaction_source.event_id
        ));
        assert_eq!(group, original_group);
        assert_eq!(
            palette.resolve_reaction_group(&group, []),
            Err(CustomEmojiError::MissingLongReactionTag)
        );
    }
}
