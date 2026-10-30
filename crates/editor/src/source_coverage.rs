use std::ops::Range;

use gpui::Context;
use language::{Bias, Point};
use project::{
    ProjectItem as _,
    source_coverage::{SourceCoverageFile, SourceCoverageRange, SourceCoverageStatus},
};
use theme::ActiveTheme as _;

use crate::Editor;

enum CoveredSourceRange {}
enum UncoveredSourceRange {}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SourceCoverageVisibleSummary {
    pub covered: usize,
    pub uncovered: usize,
}

pub fn summarize_visible_source_coverage(
    ranges: &[SourceCoverageRange],
    visible_lines: Range<u32>,
) -> SourceCoverageVisibleSummary {
    let mut summary = SourceCoverageVisibleSummary::default();
    let start = ranges.partition_point(|range| range.end.line < visible_lines.start);
    for range in &ranges[start..] {
        if range.start.line >= visible_lines.end {
            break;
        }
        if range.is_covered() {
            summary.covered += 1;
        } else {
            summary.uncovered += 1;
        }
    }
    summary
}

impl Editor {
    pub(crate) fn init_source_coverage(&mut self, cx: &mut Context<Self>) {
        let Some(project) = self.project.clone() else {
            return;
        };
        let store = project.read(cx).source_coverage_store().clone();
        self._subscriptions
            .push(cx.subscribe(&store, |editor, _, _, cx| {
                editor.refresh_source_coverage(cx)
            }));
        self._subscriptions
            .push(cx.subscribe(&project, |editor, project, event, cx| {
                let store = project.read(cx).source_coverage_store().clone();
                match event {
                    project::Event::DisconnectedFromHost
                    | project::Event::DisconnectedFromRemote { .. }
                    | project::Event::Closed => store.update(cx, |store, cx| {
                        store.invalidate(SourceCoverageStatus::Disconnected, cx)
                    }),
                    project::Event::WorktreeRemoved(_)
                    | project::Event::WorktreePathsChanged { .. }
                    | project::Event::RemoteIdChanged(_) => store.update(cx, |store, cx| {
                        store.invalidate(SourceCoverageStatus::Stale, cx)
                    }),
                    _ => {}
                }
                if matches!(
                    event,
                    project::Event::WorktreeUpdatedEntries(..)
                        | project::Event::WorktreeRemoved(_)
                        | project::Event::WorktreePathsChanged { .. }
                        | project::Event::DisconnectedFromHost
                        | project::Event::DisconnectedFromRemote { .. }
                        | project::Event::RemoteIdChanged(_)
                        | project::Event::Closed
                ) {
                    editor.refresh_source_coverage(cx);
                }
            }));
        if let Some(trust) = project::trusted_worktrees::TrustedWorktrees::try_get_global(cx) {
            self._subscriptions
                .push(cx.subscribe(&trust, |editor, _, event, cx| {
                    if let project::trusted_worktrees::TrustedWorktreesEvent::Restricted(
                        worktrees,
                        _,
                    ) = event
                        && let Some(project) = editor.project.as_ref()
                        && project.read(cx).worktree_store().downgrade() == *worktrees
                    {
                        let store = project.read(cx).source_coverage_store().clone();
                        store.update(cx, |store, cx| {
                            store.invalidate(SourceCoverageStatus::Restricted, cx)
                        });
                    }
                    editor.refresh_source_coverage(cx);
                }));
        }
        self.refresh_source_coverage(cx);
    }

    pub(crate) fn invalidate_edited_source_coverage(
        &mut self,
        buffer: &gpui::Entity<language::Buffer>,
        cx: &mut Context<Self>,
    ) {
        self.apply_source_coverage(None, cx);
        if let Some(project) = &self.project {
            let paths = buffer
                .read(cx)
                .project_path(cx)
                .into_iter()
                .collect::<Vec<_>>();
            let store = project.read(cx).source_coverage_store().clone();
            store.update(cx, |store, cx| store.invalidate_paths(&paths, cx));
        }
    }

    pub(crate) fn refresh_source_coverage(&mut self, cx: &mut Context<Self>) {
        let mut covered = Vec::new();
        let mut uncovered = Vec::new();
        if let Some(project) = self.project.as_ref() {
            let project = project.read(cx);
            let permitted = !project.is_disconnected(cx)
                && !project::trusted_worktrees::TrustedWorktrees::has_restricted_worktrees(
                    &project.worktree_store(),
                    cx,
                );
            if permitted
                && let Some(coverage) = project.source_coverage_store().read(cx).selected_snapshot()
                && matches!(
                    coverage.status,
                    SourceCoverageStatus::Current | SourceCoverageStatus::Partial
                )
            {
                let snapshot = self.buffer.read(cx).snapshot(cx);
                for buffer in self.buffer.read(cx).all_buffers() {
                    let buffer = buffer.read(cx);
                    let Some(path) = buffer.project_path(cx) else {
                        continue;
                    };
                    if buffer.is_dirty()
                        || !project
                            .visible_worktrees(cx)
                            .any(|tree| tree.read(cx).id() == path.worktree_id)
                        || !project.entry_for_path(&path, cx).is_some_and(|entry| {
                            entry.is_file()
                                && !entry.is_private
                                && !entry.is_fifo
                                && !entry.is_external
                        })
                    {
                        continue;
                    }
                    let Ok(index) = coverage.files.binary_search_by(|file| file.path.cmp(&path))
                    else {
                        continue;
                    };
                    let Some(file) = coverage.files.get(index) else {
                        continue;
                    };
                    let direct_mapping = snapshot.is_singleton() && !snapshot.has_diff_hunks();
                    for range in &file.ranges {
                        let start = buffer.clip_point(
                            Point::new(range.start.line, range.start.column),
                            Bias::Left,
                        );
                        let end = buffer
                            .clip_point(Point::new(range.end.line, range.end.column), Bias::Right);
                        if start >= end {
                            continue;
                        }
                        let anchors = buffer.anchor_before(start)..buffer.anchor_after(end);
                        let target = if range.is_covered() {
                            &mut covered
                        } else {
                            &mut uncovered
                        };
                        if direct_mapping {
                            if let Some(anchors) = snapshot.anchor_range_in_buffer(anchors.clone())
                            {
                                target.push(anchors);
                            } else {
                                target.extend(snapshot.buffer_range_to_excerpt_ranges(anchors));
                            }
                        } else {
                            target.extend(snapshot.buffer_range_to_excerpt_ranges(anchors));
                        }
                    }
                }
                for ranges in [&mut covered, &mut uncovered] {
                    ranges.sort_by(|left, right| left.start.cmp(&right.start, &snapshot));
                    let mut merged: Vec<std::ops::Range<crate::Anchor>> = Vec::new();
                    for range in ranges.drain(..) {
                        if let Some(previous) = merged.last_mut()
                            && previous.end.cmp(&range.start, &snapshot).is_ge()
                        {
                            if previous.end.cmp(&range.end, &snapshot).is_lt() {
                                previous.end = range.end;
                            }
                        } else {
                            merged.push(range);
                        }
                    }
                    *ranges = merged;
                }
            }
        }
        self.highlight_gutter::<CoveredSourceRange>(covered, |cx| cx.theme().status().success, cx);
        self.highlight_gutter::<UncoveredSourceRange>(
            uncovered,
            |cx| cx.theme().status().error,
            cx,
        );
    }

    pub fn apply_source_coverage(
        &mut self,
        coverage: Option<&SourceCoverageFile>,
        cx: &mut Context<Self>,
    ) {
        let Some(coverage) = coverage else {
            self.clear_gutter_highlights::<CoveredSourceRange>(cx);
            self.clear_gutter_highlights::<UncoveredSourceRange>(cx);
            return;
        };
        let snapshot = self.buffer().read(cx).snapshot(cx);
        let mut covered = Vec::new();
        let mut uncovered = Vec::new();
        for range in &coverage.ranges {
            let start =
                snapshot.clip_point(Point::new(range.start.line, range.start.column), Bias::Left);
            let end =
                snapshot.clip_point(Point::new(range.end.line, range.end.column), Bias::Right);
            let anchors = snapshot.anchor_before(start)..snapshot.anchor_after(end);
            if range.is_covered() {
                covered.push(anchors);
            } else {
                uncovered.push(anchors);
            }
        }
        self.highlight_gutter::<CoveredSourceRange>(covered, |cx| cx.theme().status().success, cx);
        self.highlight_gutter::<UncoveredSourceRange>(
            uncovered,
            |cx| cx.theme().status().error,
            cx,
        );
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn source_coverage_gutter_points(
        &self,
        cx: &gpui::App,
    ) -> (Vec<Range<Point>>, Vec<Range<Point>>) {
        use crate::ToPoint as _;
        let snapshot = self.buffer.read(cx).snapshot(cx);
        let points = |kind| {
            self.gutter_highlights
                .get(&kind)
                .map(|(_, ranges)| {
                    ranges
                        .iter()
                        .map(|range| range.start.to_point(&snapshot)..range.end.to_point(&snapshot))
                        .collect()
                })
                .unwrap_or_default()
        };
        (
            points(std::any::TypeId::of::<CoveredSourceRange>()),
            points(std::any::TypeId::of::<UncoveredSourceRange>()),
        )
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn source_coverage_gutter_counts(&self) -> (usize, usize) {
        let covered = self
            .gutter_highlights
            .get(&std::any::TypeId::of::<CoveredSourceRange>())
            .map(|(_, ranges)| ranges.len())
            .unwrap_or_default();
        let uncovered = self
            .gutter_highlights
            .get(&std::any::TypeId::of::<UncoveredSourceRange>())
            .map(|(_, ranges)| ranges.len())
            .unwrap_or_default();
        (covered, uncovered)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use buffer_diff::BufferDiff;
    use gpui::AppContext as _;
    use project::source_coverage::SourceCoveragePoint;
    use project::source_coverage::{SourceCoverageProviderId, SourceCoverageSnapshot};
    use project::{FakeFs, Project};
    use settings::WorktreeId;
    use util::rel_path::RelPath;

    use super::*;
    use crate::{EditorMode, MultiBuffer};

    fn range(line: u32, hit_count: u64) -> SourceCoverageRange {
        SourceCoverageRange {
            start: SourceCoveragePoint { line, column: 0 },
            end: SourceCoveragePoint { line, column: 1 },
            hit_count,
        }
    }

    #[test]
    fn source_coverage_large_visible_projection_is_range_bounded() {
        let ranges = (0..100_000)
            .map(|line| range(line, u64::from(line % 2)))
            .collect::<Vec<_>>();
        let summary = summarize_visible_source_coverage(&ranges, 50_000..50_100);
        assert_eq!(summary.covered, 50);
        assert_eq!(summary.uncovered, 50);
    }

    #[gpui::test]
    fn source_coverage_gutter_markers_replace_and_clear_without_stale_state(
        cx: &mut gpui::TestAppContext,
    ) {
        crate::editor_tests::init_test(cx, |_| {});
        let editor = cx.add_window(|window, cx| {
            let buffer = MultiBuffer::build_simple("covered\nuncovered\n", cx);
            Editor::new(EditorMode::full(), buffer, None, window, cx)
        });
        let file = SourceCoverageFile {
            path: project::ProjectPath {
                worktree_id: WorktreeId::from_usize(1),
                path: Arc::from(RelPath::from_unix_str("src/lib.rs").expect("valid fixture path")),
            },
            ranges: vec![range(0, 1), range(1, 0)],
            covered_lines: 1,
            uncovered_lines: 1,
            truncated: false,
        };
        editor
            .update(cx, |editor, _window, cx| {
                editor.apply_source_coverage(Some(&file), cx);
                assert_eq!(editor.source_coverage_gutter_counts(), (1, 1));
                editor.apply_source_coverage(None, cx);
                assert_eq!(editor.source_coverage_gutter_counts(), (0, 0));
            })
            .expect("update editor window");
    }

    #[gpui::test]
    async fn source_coverage_store_maps_disjoint_excerpts_around_deleted_hunks(
        cx: &mut gpui::TestAppContext,
    ) {
        crate::editor_tests::init_test(cx, |_| {});
        let current = "zero\none\nthree\nfive\n";
        let base = "zero\none\ntwo\nthree\nfour\nfive\n";
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/coverage", serde_json::json!({"a.py": current}))
            .await;
        let project = Project::test(fs.clone(), ["/coverage".as_ref()], cx).await;
        let path = project.read_with(cx, |project, cx| {
            project
                .find_project_path("/coverage/a.py", cx)
                .expect("coverage source")
        });
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(path.clone(), cx))
            .await
            .expect("source buffer");
        let multibuffer = cx.new(|cx| {
            let mut multibuffer = MultiBuffer::new(language::Capability::ReadWrite);
            multibuffer.set_excerpts_for_path(
                crate::PathKey::sorted(0),
                buffer.clone(),
                [
                    Point::new(1, 0)..Point::new(1, 3),
                    Point::new(3, 0)..Point::new(3, 4),
                ],
                0,
                cx,
            );
            let diff = cx.new(|cx| {
                BufferDiff::new_with_base_text(base, &buffer.read(cx).text_snapshot(), cx)
            });
            multibuffer.add_diff(diff, cx);
            multibuffer.set_all_diff_hunks_expanded(cx);
            multibuffer
        });
        let (editor, cx) = cx.add_window_view(|window, cx| {
            Editor::new(
                EditorMode::full(),
                multibuffer.clone(),
                Some(project.clone()),
                window,
                cx,
            )
        });
        let store = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        store
            .update(cx, |store, cx| {
                store.publish(
                    SourceCoverageSnapshot {
                        project_generation: 1,
                        provider_id: SourceCoverageProviderId("generic-deleted-hunks".into()),
                        generation: 1,
                        status: SourceCoverageStatus::Current,
                        truncated: false,
                        diagnostic: None,
                        files: vec![SourceCoverageFile {
                            path: path.clone(),
                            ranges: vec![range(1, 1), range(3, 0)],
                            covered_lines: 1,
                            uncovered_lines: 1,
                            truncated: false,
                        }],
                    },
                    cx,
                )
            })
            .expect("publish coverage");
        cx.run_until_parked();
        editor.read_with(cx, |editor, cx| {
            let snapshot = editor.buffer().read(cx).snapshot(cx);
            let lines = snapshot
                .text()
                .lines()
                .map(str::to_owned)
                .collect::<Vec<_>>();
            assert!(lines.iter().any(|line| line == "two" || line == "four"));
            let (covered, uncovered) = editor.source_coverage_gutter_points(cx);
            let covered_row = lines
                .iter()
                .position(|line| line == "one")
                .expect("first excerpt");
            let uncovered_row = lines
                .iter()
                .position(|line| line == "five")
                .expect("second excerpt");
            assert_eq!(covered.len(), 1);
            assert_eq!(uncovered.len(), 1);
            assert_eq!(covered[0].start.row, covered_row as u32);
            assert_eq!(uncovered[0].start.row, uncovered_row as u32);
            for range in covered.iter().chain(&uncovered) {
                assert!(!matches!(
                    lines.get(range.start.row as usize).map(String::as_str),
                    Some("two" | "four")
                ));
            }
        });
        fs.insert_file("/coverage/a.py", "zero\none changed\nthree\nfive\n".into())
            .await;
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (0, 0),
            "a watched source change invalidates both excerpt annotations"
        );
        assert_eq!(
            store.read_with(cx, |store, _| store
                .selected_snapshot()
                .map(|snapshot| snapshot.status)),
            Some(SourceCoverageStatus::Stale)
        );
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn source_coverage_real_watcher_clears_deleted_source(cx: &mut gpui::TestAppContext) {
        cx.executor().allow_parking();
        crate::editor_tests::init_test(cx, |_| {});
        let directory = tempfile::TempDir::new().expect("coverage fixture directory");
        let root = std::fs::canonicalize(directory.path()).expect("canonical fixture directory");
        let source = root.join("covered.rs");
        std::fs::write(&source, "fn covered() {}\nfn uncovered() {}\n")
            .expect("coverage source fixture");

        let fs = Arc::new(fs::RealFs::new(None, cx.executor()));
        let project = Project::test(fs, [root.as_path()], cx).await;
        let path = project.read_with(cx, |project, cx| {
            project
                .find_project_path(&source, cx)
                .expect("watched coverage source")
        });
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(path.clone(), cx))
            .await
            .expect("coverage source buffer");
        let multibuffer = cx.new(|cx| MultiBuffer::singleton(buffer, cx));
        let (editor, cx) = cx.add_window_view(|window, cx| {
            Editor::new(
                EditorMode::full(),
                multibuffer,
                Some(project.clone()),
                window,
                cx,
            )
        });
        let store = project.read_with(cx, |project, _| project.source_coverage_store().clone());
        store
            .update(cx, |store, cx| {
                store.publish(
                    SourceCoverageSnapshot {
                        project_generation: 1,
                        provider_id: SourceCoverageProviderId("real-watcher".into()),
                        generation: 1,
                        status: SourceCoverageStatus::Current,
                        truncated: false,
                        diagnostic: None,
                        files: vec![SourceCoverageFile {
                            path,
                            ranges: vec![range(0, 1), range(1, 0)],
                            covered_lines: 1,
                            uncovered_lines: 1,
                            truncated: false,
                        }],
                    },
                    cx,
                )
            })
            .expect("publish coverage");
        cx.run_until_parked();
        assert_eq!(
            editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()),
            (1, 1),
            "coverage must first reach the production Editor gutter"
        );

        std::fs::remove_file(&source).expect("delete covered source on disk");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while editor.read_with(cx, |editor, _| editor.source_coverage_gutter_counts()) != (0, 0) {
            assert!(
                std::time::Instant::now() < deadline,
                "real watcher did not clear stale coverage"
            );
            cx.background_executor
                .timer(std::time::Duration::from_millis(20))
                .await;
        }
        assert_eq!(
            store.read_with(cx, |store, _| store
                .selected_snapshot()
                .map(|snapshot| snapshot.status)),
            Some(SourceCoverageStatus::Stale)
        );
        cx.deactivate_window();
        editor.update_in(cx, |_, window, _| window.remove_window());
    }
}
