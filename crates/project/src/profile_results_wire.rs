use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context as _, Result, anyhow, ensure};
use rpc::proto::{self, Message as _};

use crate::{
    ProjectPath, WorktreeId,
    profile_results::{MAX_PROFILE_NODES, ProfileFrame, ProfileHost, ProfileNode, ProfileResults},
};

pub const PROFILE_PROTOCOL_VERSION: u32 = 1;
pub const MAX_PROFILE_PAGE_BYTES: usize = 512 * 1024;
pub const MAX_PROFILE_RESULT_BYTES: usize = 16 * 1024 * 1024;

pub fn valid_profile_task_run_id(identity: &str) -> bool {
    !identity.is_empty() && identity.len() <= 256 && !identity.chars().any(char::is_control)
}

pub fn profile_import_paths(
    request: &proto::ImportProfileResults,
    host: ProfileHost,
) -> std::result::Result<[ProjectPath; 2], proto::ProfileResultStatus> {
    use proto::ProfileResultStatus;
    if request.protocol_version != PROFILE_PROTOCOL_VERSION
        || request.artifact_format != "samply-0.13.1"
    {
        return Err(ProfileResultStatus::Mismatch);
    }
    if host != ProfileHost::MacOs {
        return Err(ProfileResultStatus::Unsupported);
    }
    if !request.consent_to_transfer {
        return Err(ProfileResultStatus::Restricted);
    }
    if request.import_id == 0
        || request
            .task_run_id
            .as_ref()
            .is_some_and(|identity| !valid_profile_task_run_id(identity))
    {
        return Err(ProfileResultStatus::Error);
    }
    let convert = |path: &Option<proto::ProjectPath>| {
        let path = path.as_ref().ok_or(ProfileResultStatus::Error)?;
        validate_path(&path.path).map_err(|_| ProfileResultStatus::Error)?;
        ProjectPath::from_proto(path.clone()).ok_or(ProfileResultStatus::Error)
    };
    let profile = convert(&request.profile_path)?;
    let symbols = convert(&request.symbols_path)?;
    if profile == symbols {
        return Err(ProfileResultStatus::Error);
    }
    Ok([profile, symbols])
}

fn check_cancelled(cancel: &AtomicBool) -> Result<()> {
    ensure!(
        !cancel.load(Ordering::Relaxed),
        "Profile transfer was cancelled"
    );
    Ok(())
}

fn validate_label(label: &str) -> Result<()> {
    ensure!(
        label.len() <= 512
            && !label.contains(['/', '\\', '"', '\''])
            && !label.chars().any(char::is_control),
        "Invalid normalized profile label"
    );
    Ok(())
}

fn validate_path(path: &str) -> Result<()> {
    ensure!(
        path.len() <= 4096
            && !path.starts_with('/')
            && !path.contains('\\')
            && !path.chars().any(char::is_control)
            && path.as_bytes().get(1) != Some(&b':')
            && (path.is_empty()
                || !path
                    .split('/')
                    .any(|component| matches!(component, ".." | "." | ""))),
        "Invalid normalized source path"
    );
    Ok(())
}

pub fn validate_profile_result(
    result: &ProfileResults,
    cancel: &AtomicBool,
    mut permitted: impl FnMut(&ProjectPath) -> bool,
) -> Result<()> {
    ensure!(
        result.frames.len() <= MAX_PROFILE_NODES
            && result.nodes.len() <= MAX_PROFILE_NODES
            && result.roots.len() <= MAX_PROFILE_NODES,
        "Profile result table exceeds the limit"
    );
    let mut retained_bytes = result.nodes.len() * 64;
    let mut unresolved = 0;
    for frame in &result.frames {
        check_cancelled(cancel)?;
        validate_label(&frame.label)?;
        retained_bytes += frame.label.len() + 64;
        unresolved += usize::from(!frame.resolved);
        if let Some((path, line)) = &frame.source {
            validate_path(path.path.as_unix_str())?;
            ensure!(
                *line > 0 && permitted(path),
                "Profile source is not permitted"
            );
            retained_bytes += path.path.as_unix_str().len();
        }
        ensure!(
            retained_bytes <= MAX_PROFILE_RESULT_BYTES,
            "Normalized profile exceeds the result limit"
        );
    }
    ensure!(
        unresolved == result.unresolved_frames,
        "Profile unresolved count mismatch"
    );
    let mut depths = vec![0usize; result.nodes.len()];
    let mut incoming = vec![false; result.nodes.len()];
    let mut roots = Vec::new();
    let mut root_samples = result.unattributed_samples;
    let mut edges = 0usize;
    for (index, node) in result.nodes.iter().enumerate() {
        check_cancelled(cancel)?;
        ensure!(
            node.frame < result.frames.len(),
            "Profile frame index is invalid"
        );
        let depth = if let Some(parent) = node.parent {
            ensure!(parent < index, "Profile parent is cyclic or out of order");
            depths
                .get(parent)
                .copied()
                .context("Invalid profile parent")?
                + 1
        } else {
            roots.push(index);
            root_samples = root_samples
                .checked_add(node.inclusive_samples)
                .context("Profile sample sum overflow")?;
            1
        };
        ensure!(depth <= 256, "Profile depth exceeds the limit");
        *depths.get_mut(index).context("Invalid profile node")? = depth;
        let mut samples = node.self_samples;
        edges = edges
            .checked_add(node.children.len())
            .context("Profile edge count overflow")?;
        ensure!(edges <= MAX_PROFILE_NODES, "Profile edge limit exceeded");
        for (child_index, child) in node.children.iter().enumerate() {
            if child_index % 256 == 0 {
                check_cancelled(cancel)?;
            }
            ensure!(*child > index, "Profile child is cyclic or out of order");
            let child_node = result.nodes.get(*child).context("Invalid profile child")?;
            let seen = incoming.get_mut(*child).context("Invalid profile child")?;
            ensure!(
                !*seen && child_node.parent == Some(index),
                "Profile child ownership mismatch"
            );
            *seen = true;
            samples = samples
                .checked_add(child_node.inclusive_samples)
                .context("Profile sample sum overflow")?;
        }
        ensure!(
            samples == node.inclusive_samples,
            "Profile aggregate sample mismatch"
        );
    }
    ensure!(
        roots == result.roots && root_samples == result.total_samples,
        "Profile root summary mismatch"
    );
    for (node, incoming) in result.nodes.iter().zip(incoming) {
        check_cancelled(cancel)?;
        ensure!(
            node.parent.is_some() == incoming,
            "Profile parent ownership mismatch"
        );
    }
    check_cancelled(cancel)
}

fn frame_to_wire(frame: &ProfileFrame) -> proto::ProfileResultFrame {
    proto::ProfileResultFrame {
        label: frame.label.clone(),
        resolved: frame.resolved,
        source: frame.source.as_ref().map(|(path, _)| path.to_proto()),
        source_line: frame.source.as_ref().map(|(_, line)| *line),
    }
}

fn indices_to_wire(indices: &[usize], cancel: &AtomicBool) -> Result<Vec<u32>> {
    indices
        .iter()
        .enumerate()
        .map(|(offset, index)| {
            if offset % 256 == 0 {
                check_cancelled(cancel)?;
            }
            u32::try_from(*index).context("Profile index exceeds the wire range")
        })
        .collect()
}

fn indices_from_wire(indices: Vec<u32>, cancel: &AtomicBool) -> Result<Vec<usize>> {
    indices
        .into_iter()
        .enumerate()
        .map(|(offset, index)| {
            if offset % 256 == 0 {
                check_cancelled(cancel)?;
            }
            ensure!(
                (index as usize) < MAX_PROFILE_NODES,
                "Profile index exceeds the table limit"
            );
            Ok(index as usize)
        })
        .collect()
}

fn node_to_wire(node: &ProfileNode, cancel: &AtomicBool) -> Result<proto::ProfileResultNode> {
    Ok(proto::ProfileResultNode {
        frame: node.frame as u32,
        parent: node.parent.map(|index| index as u32),
        children: indices_to_wire(&node.children, cancel)?,
        inclusive_samples: node.inclusive_samples,
        self_samples: node.self_samples,
    })
}

pub fn profile_result_pages(
    result: &ProfileResults,
    import_id: u64,
    generation: u64,
    cancel: &AtomicBool,
    permitted: impl FnMut(&ProjectPath) -> bool,
) -> Result<Vec<proto::ProfileResultPage>> {
    ensure!(
        import_id != 0 && generation != 0,
        "Profile transfer identity is invalid"
    );
    validate_profile_result(result, cancel, permitted)?;
    let mut frame_offset = 0;
    let mut node_offset = 0;
    let mut total_bytes = 0usize;
    let mut pages = Vec::new();
    loop {
        check_cancelled(cancel)?;
        let mut page = proto::ProfileResultPage {
            import_id,
            generation,
            protocol_version: PROFILE_PROTOCOL_VERSION,
            page_index: pages.len() as u32,
            status: proto::ProfileResultStatus::Ready as i32,
            summary: if pages.is_empty() {
                Some(proto::ProfileResultSummary {
                    frame_count: result.frames.len() as u32,
                    node_count: result.nodes.len() as u32,
                    roots: indices_to_wire(&result.roots, cancel)?,
                    total_samples: result.total_samples,
                    unattributed_samples: result.unattributed_samples,
                    unresolved_frames: result.unresolved_frames as u32,
                })
            } else {
                None
            },
            frame_offset: frame_offset as u32,
            node_offset: node_offset as u32,
            frames: Vec::new(),
            nodes: Vec::new(),
            complete: false,
        };
        let mut bytes = page.encoded_len() + 2;
        ensure!(
            bytes <= MAX_PROFILE_PAGE_BYTES,
            "Profile summary exceeds the page limit"
        );
        while let Some(frame) = result.frames.get(frame_offset) {
            check_cancelled(cancel)?;
            let frame = frame_to_wire(frame);
            let cost = frame.encoded_len() + 10;
            if bytes + cost > MAX_PROFILE_PAGE_BYTES {
                break;
            }
            bytes += cost;
            page.frames.push(frame);
            frame_offset += 1;
        }
        if frame_offset == result.frames.len() {
            while let Some(node) = result.nodes.get(node_offset) {
                check_cancelled(cancel)?;
                let node = node_to_wire(node, cancel)?;
                let cost = node.encoded_len() + 10;
                if bytes + cost > MAX_PROFILE_PAGE_BYTES {
                    break;
                }
                bytes += cost;
                page.nodes.push(node);
                node_offset += 1;
            }
        }
        page.complete = frame_offset == result.frames.len() && node_offset == result.nodes.len();
        ensure!(
            page.complete
                || page.summary.is_some()
                || !page.frames.is_empty()
                || !page.nodes.is_empty(),
            "Profile record exceeds the page limit"
        );
        let size = page.encoded_len();
        ensure!(
            size <= MAX_PROFILE_PAGE_BYTES,
            "Profile page exceeds the byte limit"
        );
        total_bytes = total_bytes
            .checked_add(size)
            .context("Profile transfer size overflow")?;
        ensure!(
            total_bytes <= MAX_PROFILE_RESULT_BYTES,
            "Profile transfer exceeds the result limit"
        );
        let complete = page.complete;
        pages.push(page);
        if complete {
            return Ok(pages);
        }
    }
}

pub struct ProfilePageAssembler {
    import_id: u64,
    generation: u64,
    next_page: u32,
    total_bytes: usize,
    edge_count: usize,
    summary: Option<proto::ProfileResultSummary>,
    frames: Vec<ProfileFrame>,
    nodes: Vec<ProfileNode>,
    complete: bool,
}

impl ProfilePageAssembler {
    pub fn new(import_id: u64, generation: u64) -> Result<Self> {
        ensure!(
            import_id != 0 && generation != 0,
            "Profile transfer identity is invalid"
        );
        Ok(Self {
            import_id,
            generation,
            next_page: 0,
            total_bytes: 0,
            edge_count: 0,
            summary: None,
            frames: Vec::new(),
            nodes: Vec::new(),
            complete: false,
        })
    }

    pub fn push(
        &mut self,
        page: proto::ProfileResultPage,
        cancel: &AtomicBool,
        mut permitted: impl FnMut(&ProjectPath) -> bool,
    ) -> Result<()> {
        check_cancelled(cancel)?;
        ensure!(
            !self.complete
                && page.protocol_version == PROFILE_PROTOCOL_VERSION
                && page.import_id == self.import_id
                && page.generation == self.generation
                && page.page_index == self.next_page,
            "Profile page identity or version mismatch"
        );
        ensure!(
            page.status == proto::ProfileResultStatus::Ready as i32,
            "Profile host did not return a ready result"
        );
        let size = page.encoded_len();
        ensure!(
            size <= MAX_PROFILE_PAGE_BYTES && self.total_bytes + size <= MAX_PROFILE_RESULT_BYTES,
            "Profile transfer exceeds the byte limit"
        );
        ensure!(
            page.frame_offset as usize == self.frames.len()
                && page.node_offset as usize == self.nodes.len(),
            "Profile page offsets are invalid"
        );
        ensure!(
            page.summary.is_some() == (self.next_page == 0),
            "Profile summary sequence is invalid"
        );
        let summary = page
            .summary
            .as_ref()
            .or(self.summary.as_ref())
            .context("Missing profile summary")?;
        ensure!(
            summary.frame_count as usize <= MAX_PROFILE_NODES
                && summary.node_count as usize <= MAX_PROFILE_NODES
                && summary.roots.len() <= MAX_PROFILE_NODES
                && summary.unresolved_frames <= summary.frame_count,
            "Profile summary exceeds the table limit"
        );
        ensure!(
            self.frames.len() + page.frames.len() <= summary.frame_count as usize
                && self.nodes.len() + page.nodes.len() <= summary.node_count as usize,
            "Profile page exceeds declared tables"
        );
        ensure!(
            page.nodes.is_empty()
                || self.frames.len() + page.frames.len() == summary.frame_count as usize,
            "Profile nodes precede frame completion"
        );
        ensure!(
            page.complete
                == (self.frames.len() + page.frames.len() == summary.frame_count as usize
                    && self.nodes.len() + page.nodes.len() == summary.node_count as usize),
            "Profile completion marker is invalid"
        );
        ensure!(
            page.complete
                || page.summary.is_some()
                || !page.frames.is_empty()
                || !page.nodes.is_empty(),
            "Profile page made no progress"
        );
        let edge_count = page.nodes.iter().try_fold(self.edge_count, |count, node| {
            check_cancelled(cancel)?;
            count
                .checked_add(node.children.len())
                .context("Profile edge count overflow")
        })?;
        ensure!(
            edge_count <= MAX_PROFILE_NODES,
            "Profile edge limit exceeded"
        );
        let frames = page
            .frames
            .into_iter()
            .map(|frame| {
                check_cancelled(cancel)?;
                validate_label(&frame.label)?;
                let source = match (frame.source, frame.source_line) {
                    (Some(source), Some(line)) => {
                        validate_path(&source.path)?;
                        let path = ProjectPath {
                            worktree_id: WorktreeId::from_proto(source.worktree_id),
                            path: util::rel_path::RelPath::from_unix_str(&source.path)
                                .map_err(|_| anyhow!("Invalid normalized source path"))?
                                .into(),
                        };
                        ensure!(
                            line > 0 && permitted(&path),
                            "Profile source is not permitted"
                        );
                        Some((path, line))
                    }
                    (None, None) => None,
                    _ => return Err(anyhow!("Profile source location is incomplete")),
                };
                Ok(ProfileFrame {
                    label: frame.label,
                    resolved: frame.resolved,
                    source,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let nodes = page
            .nodes
            .into_iter()
            .map(|node| {
                check_cancelled(cancel)?;
                Ok(ProfileNode {
                    frame: node.frame as usize,
                    parent: node.parent.map(|parent| parent as usize),
                    children: indices_from_wire(node.children, cancel)?,
                    inclusive_samples: node.inclusive_samples,
                    self_samples: node.self_samples,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.frames.extend(frames);
        self.nodes.extend(nodes);
        if page.summary.is_some() {
            self.summary = page.summary;
        }
        self.total_bytes += size;
        self.edge_count = edge_count;
        self.next_page += 1;
        self.complete = page.complete;
        Ok(())
    }

    pub fn finish(
        self,
        cancel: &AtomicBool,
        permitted: impl FnMut(&ProjectPath) -> bool,
    ) -> Result<ProfileResults> {
        ensure!(self.complete, "Profile transfer is incomplete");
        let summary = self.summary.context("Missing profile summary")?;
        let result = ProfileResults {
            frames: self.frames,
            nodes: self.nodes,
            roots: indices_from_wire(summary.roots, cancel)?,
            total_samples: summary.total_samples,
            unattributed_samples: summary.unattributed_samples,
            unresolved_frames: summary.unresolved_frames as usize,
        };
        validate_profile_result(&result, cancel, permitted)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile_results::parse_samply_profile;

    fn capture() -> ProfileResults {
        parse_samply_profile(
            include_bytes!("../../../crates/language_tools/test_fixtures/profiling/macos-samply-0.13.1.json"),
            include_bytes!("../../../crates/language_tools/test_fixtures/profiling/macos-samply-0.13.1.syms.json"),
            ProfileHost::MacOs,
            &AtomicBool::new(false),
            |_, _| Some(ProjectPath { worktree_id: WorktreeId::from_usize(3), path: util::rel_path::rel_path("workload.rs").into() }),
        ).expect("captured profile")
    }

    #[test]
    fn profile_results_wire_import_admission_requires_explicit_bounded_contract() {
        use proto::ProfileResultStatus;
        let request = proto::ImportProfileResults {
            project_id: 1,
            import_id: 2,
            protocol_version: PROFILE_PROTOCOL_VERSION,
            artifact_format: "samply-0.13.1".into(),
            consent_to_transfer: true,
            profile_path: Some(proto::ProjectPath {
                worktree_id: 1,
                path: "profile.json".into(),
            }),
            symbols_path: Some(proto::ProjectPath {
                worktree_id: 1,
                path: "profile.syms.json".into(),
            }),
            task_run_id: None,
        };
        let paths = profile_import_paths(&request, ProfileHost::MacOs).expect("explicit selection");
        assert_ne!(paths[0], paths[1]);
        for host in [ProfileHost::Linux, ProfileHost::Windows] {
            assert_eq!(
                profile_import_paths(&request, host),
                Err(ProfileResultStatus::Unsupported)
            );
        }
        let mut invalid = request.clone();
        invalid.consent_to_transfer = false;
        assert_eq!(
            profile_import_paths(&invalid, ProfileHost::MacOs),
            Err(ProfileResultStatus::Restricted)
        );
        for mutate in [
            |request: &mut proto::ImportProfileResults| request.protocol_version += 1,
            |request: &mut proto::ImportProfileResults| {
                request.artifact_format = "samply-latest".into()
            },
        ] {
            let mut invalid = request.clone();
            mutate(&mut invalid);
            assert_eq!(
                profile_import_paths(&invalid, ProfileHost::MacOs),
                Err(ProfileResultStatus::Mismatch)
            );
        }
        for mutate in [
            |request: &mut proto::ImportProfileResults| request.import_id = 0,
            |request: &mut proto::ImportProfileResults| request.profile_path = None,
            |request: &mut proto::ImportProfileResults| request.symbols_path = None,
            |request: &mut proto::ImportProfileResults| {
                request.symbols_path = request.profile_path.clone()
            },
            |request: &mut proto::ImportProfileResults| request.task_run_id = Some(String::new()),
            |request: &mut proto::ImportProfileResults| request.task_run_id = Some("x".repeat(257)),
            |request: &mut proto::ImportProfileResults| {
                request.task_run_id = Some("run\nprivate".into())
            },
        ] {
            let mut invalid = request.clone();
            mutate(&mut invalid);
            assert_eq!(
                profile_import_paths(&invalid, ProfileHost::MacOs),
                Err(ProfileResultStatus::Error)
            );
        }
        for path in [
            "../profile.json".into(),
            "/secret/profile.json".into(),
            "C:\\private.json".into(),
            "x".repeat(4097),
        ] {
            let mut invalid = request.clone();
            invalid.profile_path.as_mut().expect("fixture input").path = path;
            assert_eq!(
                profile_import_paths(&invalid, ProfileHost::MacOs),
                Err(ProfileResultStatus::Error)
            );
        }
        let mut metadata = request;
        metadata.task_run_id = Some("unproven-run-identity".into());
        assert_eq!(
            profile_import_paths(&metadata, ProfileHost::MacOs),
            Ok(paths)
        );
    }

    #[test]
    fn profile_results_wire_capture_round_trip_rechecks_source_permissions() {
        let result = capture();
        let cancel = AtomicBool::new(false);
        let pages =
            profile_result_pages(&result, 1, 2, &cancel, |_| true).expect("bounded encoding");
        let text = serde_json::to_string(&pages).expect("inspect wire facts");
        assert!(
            !text.contains("/fixture")
                && !text.contains("/Users/")
                && !text.contains("string_table")
        );
        let mut assembler = ProfilePageAssembler::new(1, 2).expect("identity");
        for page in pages.clone() {
            let encoded = page.encode_to_vec();
            assert!(encoded.len() <= MAX_PROFILE_PAGE_BYTES);
            let decoded =
                proto::ProfileResultPage::decode(encoded.as_slice()).expect("protobuf round trip");
            assembler
                .push(decoded, &cancel, |_| true)
                .expect("authorized page");
        }
        assert_eq!(
            assembler.finish(&cancel, |_| true).expect("complete model"),
            result
        );
        let mut assembler = ProfilePageAssembler::new(1, 2).expect("identity");
        for page in pages {
            assembler
                .push(page, &cancel, |_| true)
                .expect("initial permission");
        }
        assert!(
            assembler.finish(&cancel, |_| false).is_err(),
            "revocation must be rechecked at completion"
        );
        assert!(profile_result_pages(&result, 1, 2, &cancel, |_| false).is_err());
    }

    #[test]
    fn profile_results_wire_accepts_authorized_file_worktree_source() {
        let mut result = capture();
        for frame in &mut result.frames {
            if let Some((path, _)) = &mut frame.source {
                path.path = util::rel_path::RelPath::empty_arc();
            }
        }
        let cancel = AtomicBool::new(false);
        let pages = profile_result_pages(&result, 1, 2, &cancel, |path| path.path.is_empty())
            .expect("authorized file worktree root");
        let mut assembler = ProfilePageAssembler::new(1, 2).expect("identity");
        for page in pages {
            assembler
                .push(page, &cancel, |path| path.path.is_empty())
                .expect("file worktree page");
        }
        assert_eq!(
            assembler
                .finish(&cancel, |path| path.path.is_empty())
                .expect("file worktree result"),
            result
        );
    }

    #[test]
    fn profile_results_wire_large_model_uses_bounded_ordered_pages() {
        let count = 10_001;
        let result = ProfileResults {
            frames: (0..count)
                .map(|index| ProfileFrame {
                    label: format!("{}-{index}", "frame".repeat(80)),
                    source: None,
                    resolved: true,
                })
                .collect(),
            nodes: (0..count)
                .map(|index| ProfileNode {
                    frame: index,
                    parent: None,
                    children: Vec::new(),
                    inclusive_samples: 1,
                    self_samples: 1,
                })
                .collect(),
            roots: (0..count).collect(),
            total_samples: count as u64,
            unattributed_samples: 0,
            unresolved_frames: 0,
        };
        let cancel = AtomicBool::new(false);
        let pages = profile_result_pages(&result, 7, 8, &cancel, |_| true).expect("large model");
        assert!(pages.len() > 2);
        assert!(
            pages
                .iter()
                .all(|page| page.encoded_len() <= MAX_PROFILE_PAGE_BYTES)
        );
        assert!(
            pages.iter().map(|page| page.encoded_len()).sum::<usize>() <= MAX_PROFILE_RESULT_BYTES
        );
        let mut assembler = ProfilePageAssembler::new(7, 8).expect("identity");
        let first = pages.first().expect("first page").clone();
        assembler
            .push(first.clone(), &cancel, |_| true)
            .expect("first page");
        assert!(
            assembler.push(first, &cancel, |_| true).is_err(),
            "duplicate page"
        );
        for page in pages.into_iter().skip(1) {
            assembler.push(page, &cancel, |_| true).expect("next page");
        }
        assert_eq!(
            assembler
                .finish(&cancel, |_| true)
                .expect("complete large model"),
            result
        );
    }

    #[test]
    fn profile_results_wire_rejects_mismatch_paths_and_malformed_graphs() {
        let result = capture();
        let cancel = AtomicBool::new(false);
        let page = profile_result_pages(&result, 1, 2, &cancel, |_| true)
            .expect("capture page")
            .remove(0);
        for mutation in 0..12 {
            let mut invalid = page.clone();
            match mutation {
                0 => invalid.protocol_version += 1,
                1 => invalid.import_id += 1,
                2 => invalid.generation += 1,
                3 => invalid.frame_offset += 1,
                4 => invalid.complete = false,
                5 => invalid.status = proto::ProfileResultStatus::Restricted as i32,
                6 => invalid.summary = None,
                7 => {
                    invalid
                        .frames
                        .iter_mut()
                        .find(|frame| frame.source.is_some())
                        .expect("mapped frame")
                        .source
                        .as_mut()
                        .expect("source")
                        .path = "/private/secret.rs".into()
                }
                8 => invalid.frames.first_mut().expect("frame").label = "secret/path".into(),
                9 => {
                    invalid.summary.as_mut().expect("summary").frame_count =
                        MAX_PROFILE_NODES as u32 + 1
                }
                10 => {
                    invalid
                        .frames
                        .iter_mut()
                        .find(|frame| frame.source.is_some())
                        .expect("mapped frame")
                        .source_line = Some(0)
                }
                _ => {
                    invalid.nodes.first_mut().expect("node").children =
                        vec![0; MAX_PROFILE_NODES + 1]
                }
            }
            assert!(
                ProfilePageAssembler::new(1, 2)
                    .expect("identity")
                    .push(invalid, &cancel, |_| true)
                    .is_err(),
                "mutation {mutation}"
            );
        }
        let mut invalid = page;
        invalid.nodes.first_mut().expect("node").parent = Some(0);
        let mut assembler = ProfilePageAssembler::new(1, 2).expect("identity");
        assembler
            .push(invalid, &cancel, |_| true)
            .expect("graph validation occurs at completion");
        assert!(assembler.finish(&cancel, |_| true).is_err());
        let deep = ProfileResults {
            frames: vec![ProfileFrame {
                label: "frame".into(),
                source: None,
                resolved: true,
            }],
            nodes: (0usize..257)
                .map(|index| ProfileNode {
                    frame: 0,
                    parent: index.checked_sub(1),
                    children: if index < 256 {
                        vec![index + 1]
                    } else {
                        Vec::new()
                    },
                    inclusive_samples: 1,
                    self_samples: u64::from(index == 256),
                })
                .collect(),
            roots: vec![0],
            total_samples: 1,
            unattributed_samples: 0,
            unresolved_frames: 0,
        };
        assert!(
            validate_profile_result(&deep, &cancel, |_| true)
                .expect_err("depth cap")
                .to_string()
                .contains("depth")
        );
        let mut cycle = result.clone();
        cycle.nodes.first_mut().expect("node").parent = Some(0);
        assert!(validate_profile_result(&cycle, &cancel, |_| true).is_err());
        let mut weights = result.clone();
        weights.total_samples += 1;
        assert!(validate_profile_result(&weights, &cancel, |_| true).is_err());
        let mut ownership = result;
        ownership.nodes.first_mut().expect("root").children.push(0);
        assert!(validate_profile_result(&ownership, &cancel, |_| true).is_err());
    }

    #[test]
    fn profile_results_wire_rejects_byte_limits_and_cancelled_transfers() {
        let cancel = AtomicBool::new(false);
        let result = capture();
        let page = profile_result_pages(&result, 1, 2, &cancel, |_| true)
            .expect("capture")
            .remove(0);
        let mut oversized = page.clone();
        oversized.frames.first_mut().expect("frame").label = "a".repeat(MAX_PROFILE_PAGE_BYTES);
        assert!(
            ProfilePageAssembler::new(1, 2)
                .expect("identity")
                .push(oversized, &cancel, |_| true)
                .is_err()
        );
        let cancelled = AtomicBool::new(true);
        assert!(profile_result_pages(&result, 1, 2, &cancelled, |_| true).is_err());
        assert!(
            ProfilePageAssembler::new(1, 2)
                .expect("identity")
                .push(page, &cancelled, |_| true)
                .is_err()
        );
        assert!(
            ProfilePageAssembler::new(1, 2)
                .expect("identity")
                .finish(&cancel, |_| true)
                .is_err()
        );
        let mut assembler = ProfilePageAssembler::new(1, 2).expect("identity");
        let mut exceeded = false;
        for page_index in 0..34 {
            let page = proto::ProfileResultPage {
                import_id: 1,
                generation: 2,
                protocol_version: PROFILE_PROTOCOL_VERSION,
                page_index,
                status: proto::ProfileResultStatus::Ready as i32,
                summary: (page_index == 0).then(|| proto::ProfileResultSummary {
                    frame_count: 34_000,
                    ..Default::default()
                }),
                frame_offset: page_index * 1000,
                frames: (0..1000)
                    .map(|_| proto::ProfileResultFrame {
                        label: "f".repeat(512),
                        resolved: true,
                        source: None,
                        source_line: None,
                    })
                    .collect(),
                complete: page_index == 33,
                ..Default::default()
            };
            assert!(page.encoded_len() <= MAX_PROFILE_PAGE_BYTES);
            if let Err(error) = assembler.push(page, &cancel, |_| true) {
                assert!(error.to_string().contains("byte limit"), "{error:#}");
                exceeded = true;
                break;
            }
        }
        assert!(exceeded, "cumulative transfer must reject more than 16 MiB");
    }
}
