use std::path::Path;

use settings::{RegisterSetting, ScanSymlinksSetting, Settings};
use util::{
    paths::{PathMatcher, PathStyle},
    rel_path::RelPath,
};

#[derive(Clone, PartialEq, Eq, RegisterSetting)]
pub struct WorktreeSettings {
    /// Whether to prevent this project from being shared in public channels.
    pub prevent_sharing_in_public_channels: bool,
    pub file_scan_exclusions: PathMatcher,
    pub file_scan_inclusions: PathMatcher,
    /// This field contains all ancestors of the `file_scan_inclusions`. It's used to
    /// determine whether to terminate worktree scanning for a given dir.
    pub parent_dir_scan_inclusions: PathMatcher,
    pub scan_symlinks: ScanSymlinksSetting,
    pub private_files: PathMatcher,
    pub hidden_files: PathMatcher,
    pub read_only_files: PathMatcher,
}

impl WorktreeSettings {
    pub fn can_recover_excluded_settings_path(&self, path: &RelPath) -> bool {
        self.file_scan_exclusions.is_conservative()
            && path.ends_with(paths::local_settings_file_relative_path())
            && !path
                .ancestors()
                .any(|ancestor| self.file_scan_exclusions.is_match_accepted(ancestor))
    }

    pub fn is_path_private(&self, path: &RelPath) -> bool {
        path.ancestors()
            .any(|ancestor| self.private_files.is_match(ancestor))
    }

    pub fn is_path_excluded(&self, path: &RelPath) -> bool {
        path.ancestors()
            .any(|ancestor| self.file_scan_exclusions.is_match(ancestor))
    }

    pub fn is_path_always_included(&self, path: &RelPath, is_dir: bool) -> bool {
        if is_dir {
            self.parent_dir_scan_inclusions.is_match(path)
        } else {
            self.file_scan_inclusions.is_match(path)
        }
    }

    pub fn is_path_hidden(&self, path: &RelPath) -> bool {
        path.ancestors()
            .any(|ancestor| self.hidden_files.is_match(ancestor))
    }

    pub fn is_path_read_only(&self, path: &RelPath) -> bool {
        self.read_only_files.is_match(path)
    }

    pub fn is_std_path_read_only(&self, path: &Path) -> bool {
        self.read_only_files.is_match_std_path(path)
    }
}

impl Settings for WorktreeSettings {
    fn from_settings(content: &settings::SettingsContent) -> Self {
        let worktree = content.project.worktree.clone();
        let file_scan_exclusions = worktree.file_scan_exclusions.unwrap();
        let file_scan_inclusions = worktree.file_scan_inclusions.unwrap();
        let private_files = worktree.private_files.unwrap().0;
        let hidden_files = worktree.hidden_files.unwrap();
        let read_only_files = worktree.read_only_files.unwrap_or_default();
        let scan_symlinks = worktree.scan_symlinks.unwrap();
        let (file_scan_inclusions, parent_dir_scan_inclusions) =
            file_scan_inclusion_matchers(file_scan_inclusions);

        Self {
            prevent_sharing_in_public_channels: worktree.prevent_sharing_in_public_channels,
            file_scan_exclusions: valid_path_matchers(
                file_scan_exclusions,
                "file_scan_exclusions",
                true,
            ),
            parent_dir_scan_inclusions,
            file_scan_inclusions,
            private_files: valid_path_matchers(private_files, "private_files", true),
            hidden_files: valid_path_matchers(hidden_files, "hidden_files", false),
            read_only_files: valid_path_matchers(read_only_files, "read_only_files", true),
            scan_symlinks,
        }
    }
}

fn valid_path_matchers(
    mut values: Vec<String>,
    context: &'static str,
    protective: bool,
) -> PathMatcher {
    values.sort();
    match PathMatcher::new_lenient(values, PathStyle::local(), |error| {
        let source = error
            .glob()
            .map(|source| source.chars().take(120).collect::<String>());
        log::error!(
            "Invalid pattern in `{context}`: {source:?}: {}",
            error.kind()
        );
    }) {
        Ok(matcher) => matcher,
        Err(error) => {
            for failure in &error.failures {
                let source = failure.source.chars().take(120).collect::<String>();
                log::error!(
                    "Uncompiled pattern in `{context}` at parsed index {}: {source:?}: {}",
                    failure.index,
                    failure.error.kind()
                );
            }
            if protective {
                log::error!(
                    "Protecting all paths because `{context}` contains an uncompiled pattern; correct these settings to restore normal matching"
                );
                error.into_conservative_matcher()
            } else {
                *error.partial
            }
        }
    }
}

fn file_scan_inclusion_matchers(values: Vec<String>) -> (PathMatcher, PathMatcher) {
    let inclusions = valid_path_matchers(values, "file_scan_inclusions", false);
    let parents = inclusions
        .accepted_sources()
        .iter()
        .flat_map(|pattern| inclusion_parent_patterns(pattern, PathStyle::local()))
        .collect();
    let parent_inclusions = valid_path_matchers(parents, "file_scan_inclusions", false);
    (inclusions, parent_inclusions)
}

fn inclusion_parent_patterns(pattern: &str, path_style: PathStyle) -> Vec<String> {
    let mut parents = Vec::new();
    let mut open_braces = 0_usize;
    let mut class_start = None;
    let mut escaped = false;
    for (index, character) in pattern.char_indices() {
        let is_separator = path_style.separators_ch().contains(&character);
        let mut split_index = index;
        if class_start.is_none() {
            if escaped {
                escaped = false;
                if !is_separator {
                    continue;
                }
                split_index -= 1;
            } else if character == '\\' && path_style.is_posix() {
                escaped = true;
                continue;
            }
        }
        if let Some(start) = class_start {
            let first_content = start
                + 1
                + usize::from(matches!(
                    pattern.as_bytes().get(start + 1),
                    Some(b'!' | b'^')
                ));
            if character == ']' && index > first_content {
                class_start = None;
            }
        } else {
            match character {
                '[' => class_start = Some(index),
                '{' => open_braces += 1,
                '}' => open_braces = open_braces.saturating_sub(1),
                _ => {}
            }
        }
        if is_separator {
            let closing_braces = "}".repeat(open_braces);
            if let Some(start) = class_start {
                parents.push(format!("{}*{closing_braces}", &pattern[..start]));
            } else if open_braces > 0 {
                parents.push(format!("{}{closing_braces}", &pattern[..split_index]));
            } else if split_index > 0 {
                parents.push(pattern[..split_index].to_string());
            }
        }
    }
    parents.sort();
    parents.dedup();
    parents
}
