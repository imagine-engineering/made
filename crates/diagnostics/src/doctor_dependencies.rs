use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::doctor::{DiagnosticText, MAX_REPORT_BYTES};

pub const MAX_DEPENDENCY_NODES: usize = 4096;
pub const MAX_DEPENDENCY_EDGES: usize = 32768;
pub const MAX_DEPENDENCY_ROOTS: usize = 128;
pub const MAX_DEPENDENCY_GRAPH_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_DEPENDENCY_PATHS: usize = 8;
pub const MAX_DEPENDENCY_PATH_DEPTH: usize = 32;
pub const MAX_SELECTED_DEPENDENCIES: usize = 8;
const MAX_PATH_VISITS: usize = 4096;
const MAX_FIELD_BYTES: usize = 512;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum DependencyEcosystem {
    Cargo,
    Maven,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum DependencySourceKind {
    CratesIo,
    MavenCentral,
    Workspace,
    Path,
    Git,
    PrivateRegistry,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DependencyIdentity([u8; 32]);

impl DependencyIdentity {
    pub fn from_fields(fields: &[&str]) -> Result<Self> {
        ensure!(fields.len() <= 32, "Too many dependency identity fields");
        let mut digest = Sha256::new();
        for field in fields {
            ensure!(
                field.len() <= 16 * 1024,
                "Dependency identity input is too large"
            );
            digest.update(u64::try_from(field.len())?.to_le_bytes());
            digest.update(field.as_bytes());
        }
        Ok(Self(digest.finalize().into()))
    }

    pub fn hex(&self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DependencySource {
    kind: DependencySourceKind,
    identity: DependencyIdentity,
}

impl DependencySource {
    pub fn public_registry(ecosystem: DependencyEcosystem) -> Self {
        let (kind, identity) = match ecosystem {
            DependencyEcosystem::Cargo => (DependencySourceKind::CratesIo, "crates.io"),
            DependencyEcosystem::Maven => (DependencySourceKind::MavenCentral, "Maven Central"),
        };
        Self {
            kind,
            identity: DependencyIdentity(Sha256::digest(identity.as_bytes()).into()),
        }
    }

    pub fn non_public(kind: DependencySourceKind, owner_identity: &str) -> Result<Self> {
        ensure!(
            !matches!(
                kind,
                DependencySourceKind::CratesIo | DependencySourceKind::MavenCentral
            ),
            "Public registry identity requires verified ecosystem ownership"
        );
        ensure!(
            !owner_identity.is_empty(),
            "Missing dependency source identity"
        );
        Ok(Self {
            kind,
            identity: DependencyIdentity::from_fields(&[owner_identity])?,
        })
    }

    pub fn kind(&self) -> DependencySourceKind {
        self.kind
    }

    pub fn identity(&self) -> &DependencyIdentity {
        &self.identity
    }

    pub fn is_public_registry(&self) -> bool {
        matches!(
            self.kind,
            DependencySourceKind::CratesIo | DependencySourceKind::MavenCentral
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DependencyInstance {
    pub ecosystem: DependencyEcosystem,
    pub source: DependencySource,
    pub name: String,
    pub version: String,
    pub classifier: Option<String>,
    pub configuration: DependencyIdentity,
}

impl DependencyInstance {
    pub fn validate(&self) -> Result<()> {
        validate_field(&self.name)?;
        validate_field(&self.version)?;
        if let Some(classifier) = &self.classifier {
            validate_field(classifier)?;
        }
        ensure!(
            !matches!(
                (self.ecosystem, self.source.kind),
                (
                    DependencyEcosystem::Cargo,
                    DependencySourceKind::MavenCentral
                ) | (DependencyEcosystem::Maven, DependencySourceKind::CratesIo)
            ),
            "Dependency ecosystem does not match its registry"
        );
        if self.source.is_public_registry() {
            validate_public_coordinate(self.ecosystem, &self.name)?;
            ensure!(
                self.version
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b".-_+".contains(&byte)),
                "Invalid public dependency version"
            );
        }
        Ok(())
    }

    pub fn selected_label(&self) -> DiagnosticText {
        if self.source.is_public_registry() {
            DiagnosticText::new(format!("{} {}", self.name, self.version))
        } else {
            DiagnosticText::new(format!(
                "Non-public {:?} dependency {}",
                self.source.kind,
                self.source.identity.hex()
            ))
        }
    }
}

fn validate_field(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty() && value.len() <= MAX_FIELD_BYTES && !value.chars().any(char::is_control),
        "Invalid or excessive dependency field"
    );
    Ok(())
}

fn validate_public_coordinate(ecosystem: DependencyEcosystem, name: &str) -> Result<()> {
    let valid_component = |component: &str| {
        !component.is_empty()
            && component
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
    };
    ensure!(
        match ecosystem {
            DependencyEcosystem::Cargo => valid_component(name) && !name.contains('.'),
            DependencyEcosystem::Maven => name.split_once(':').is_some_and(|(group, artifact)| {
                valid_component(group) && valid_component(artifact)
            }),
        },
        "Dependency is not a supported public coordinate"
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DependencyCoverage {
    Complete,
    Partial,
    Unavailable,
    Unsupported,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependencyProvenance {
    pub host: DependencyIdentity,
    pub inputs: DependencyIdentity,
    pub provider: DiagnosticText,
    pub observed_at: u64,
    pub coverage: DependencyCoverage,
    pub limitations: Vec<DiagnosticText>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependencyEdge {
    pub parent: usize,
    pub child: usize,
    pub alias: String,
    pub scope: String,
    pub requirement: Option<String>,
    pub target: Option<String>,
    pub workspace_inherited: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependencyPath {
    pub nodes: Vec<usize>,
    pub edges: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependencyIntroducingPaths {
    pub paths: Vec<DependencyPath>,
    pub truncated: bool,
    pub cycle_observed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependencyGraph {
    provenance: DependencyProvenance,
    nodes: Vec<DependencyInstance>,
    edges: Vec<DependencyEdge>,
    roots: Vec<usize>,
}

impl DependencyGraph {
    pub fn new(
        provenance: DependencyProvenance,
        nodes: Vec<DependencyInstance>,
        edges: Vec<DependencyEdge>,
        roots: Vec<usize>,
    ) -> Result<Self> {
        ensure!(
            nodes.len() <= MAX_DEPENDENCY_NODES,
            "Dependency node limit exceeded"
        );
        ensure!(
            edges.len() <= MAX_DEPENDENCY_EDGES,
            "Dependency edge limit exceeded"
        );
        ensure!(
            roots.len() <= MAX_DEPENDENCY_ROOTS,
            "Dependency root limit exceeded"
        );
        ensure!(
            provenance.limitations.len() <= 32,
            "Dependency limitation limit exceeded"
        );
        ensure!(
            provenance.coverage != DependencyCoverage::Complete
                || (!nodes.is_empty() && !roots.is_empty()),
            "An empty resolved graph does not establish complete inventory"
        );
        let mut identities = BTreeSet::new();
        let mut bytes = provenance.provider.as_str().len();
        for limitation in &provenance.limitations {
            bytes = bytes
                .checked_add(limitation.as_str().len())
                .ok_or_else(|| anyhow::anyhow!("Dependency byte count overflow"))?;
        }
        for node in &nodes {
            node.validate()?;
            ensure!(
                identities.insert(node),
                "Duplicate resolved dependency instance"
            );
            for field in [
                Some(node.name.as_str()),
                Some(node.version.as_str()),
                node.classifier.as_deref(),
            ] {
                bytes = bytes
                    .checked_add(field.map_or(0, str::len) + 128)
                    .ok_or_else(|| anyhow::anyhow!("Dependency byte count overflow"))?;
            }
        }
        let mut edge_identities = BTreeSet::new();
        for edge in &edges {
            ensure!(
                edge.parent < nodes.len() && edge.child < nodes.len(),
                "Dependency edge references an absent instance"
            );
            ensure!(
                edge_identities.insert((
                    edge.parent,
                    edge.child,
                    &edge.alias,
                    &edge.scope,
                    &edge.requirement,
                    &edge.target,
                    edge.workspace_inherited
                )),
                "Duplicate dependency edge"
            );
            for field in [
                Some(edge.alias.as_str()),
                Some(edge.scope.as_str()),
                edge.requirement.as_deref(),
                edge.target.as_deref(),
            ]
            .into_iter()
            .flatten()
            {
                validate_field(field)?;
                bytes = bytes
                    .checked_add(field.len() + 32)
                    .ok_or_else(|| anyhow::anyhow!("Dependency byte count overflow"))?;
            }
        }
        ensure!(
            bytes <= MAX_DEPENDENCY_GRAPH_BYTES,
            "Dependency aggregate byte limit exceeded"
        );
        let unique_roots: BTreeSet<_> = roots.iter().copied().collect();
        ensure!(
            unique_roots.len() == roots.len() && roots.iter().all(|root| *root < nodes.len()),
            "Invalid or duplicated dependency root"
        );
        Ok(Self {
            provenance,
            nodes,
            edges,
            roots,
        })
    }

    pub fn provenance(&self) -> &DependencyProvenance {
        &self.provenance
    }

    pub fn nodes(&self) -> &[DependencyInstance] {
        &self.nodes
    }

    pub fn edges(&self) -> &[DependencyEdge] {
        &self.edges
    }

    pub fn roots(&self) -> &[usize] {
        &self.roots
    }

    pub fn introducing_paths(&self, target: usize) -> Result<DependencyIntroducingPaths> {
        ensure!(target < self.nodes.len(), "Unknown dependency instance");
        let mut incoming = BTreeMap::<usize, Vec<usize>>::new();
        for (index, edge) in self.edges.iter().enumerate() {
            incoming.entry(edge.child).or_default().push(index);
        }
        let mut queue = VecDeque::from([DependencyPath {
            nodes: vec![target],
            edges: Vec::new(),
        }]);
        let roots: BTreeSet<_> = self.roots.iter().copied().collect();
        let mut result = DependencyIntroducingPaths {
            paths: Vec::new(),
            truncated: false,
            cycle_observed: false,
        };
        let mut visits = 0;
        while let Some(mut path) = queue.pop_front() {
            visits += 1;
            if visits > MAX_PATH_VISITS {
                result.truncated = true;
                break;
            }
            let Some(parent) = path.nodes.last().copied() else {
                continue;
            };
            if roots.contains(&parent) {
                path.nodes.reverse();
                path.edges.reverse();
                result.paths.push(path);
                if result.paths.len() == MAX_DEPENDENCY_PATHS {
                    result.truncated |= !queue.is_empty();
                    break;
                }
                continue;
            }
            if path.nodes.len() >= MAX_DEPENDENCY_PATH_DEPTH {
                result.truncated |= incoming.contains_key(&parent);
                continue;
            }
            for edge_index in incoming.get(&parent).into_iter().flatten() {
                let Some(edge) = self.edges.get(*edge_index) else {
                    continue;
                };
                if path.nodes.contains(&edge.parent) {
                    result.cycle_observed = true;
                    continue;
                }
                if queue.len() >= MAX_PATH_VISITS {
                    result.truncated = true;
                    continue;
                }
                let mut next = path.clone();
                next.nodes.push(edge.parent);
                next.edges.push(*edge_index);
                queue.push_back(next);
            }
        }
        result.truncated |=
            self.provenance.coverage != DependencyCoverage::Complete || result.paths.is_empty();
        Ok(result)
    }

    pub fn selected_summaries(
        &self,
        selection: &[usize],
    ) -> Result<Vec<DependencySelectedSummary>> {
        ensure!(
            selection.len() <= MAX_SELECTED_DEPENDENCIES,
            "Select at most eight dependency instances"
        );
        let mut selected = BTreeSet::new();
        let mut summaries = Vec::new();
        for index in selection {
            ensure!(
                selected.insert(*index),
                "Duplicate selected dependency instance"
            );
            let node = self
                .nodes
                .get(*index)
                .ok_or_else(|| anyhow::anyhow!("Unknown selected dependency instance"))?;
            let paths = self.introducing_paths(*index)?;
            summaries.push(DependencySelectedSummary {
                package: node.selected_label(),
                source: DiagnosticText::new(format!("{:?}", node.source.kind())),
                provider: self.provenance.provider.clone(),
                observed_at: self.provenance.observed_at,
                coverage: DiagnosticText::new(format!("{:?}; {} bounded introducing paths; truncated={}; cycle={}", self.provenance.coverage, paths.paths.len(), paths.truncated, paths.cycle_observed)),
                application_exploitability: DiagnosticText::new("Unknown: a resolved affected package is not evidence that this application is exploitable"),
            });
        }
        ensure!(
            serde_json::to_vec(&summaries)?.len() <= MAX_REPORT_BYTES,
            "Selected dependency evidence is too large"
        );
        Ok(summaries)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DependencySelectedSummary {
    pub package: DiagnosticText,
    pub source: DiagnosticText,
    pub provider: DiagnosticText,
    pub observed_at: u64,
    pub coverage: DiagnosticText,
    pub application_exploitability: DiagnosticText,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DependencyAdvisoryAuthority {
    RustSec,
    OsvMaven,
    GitHub,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DependencyAdvisoryState {
    KnownAffectedDependency,
    Informational,
    Withdrawn,
    UnknownVersionSemantics,
    UnverifiedSource,
    UnknownApplicability,
    NotAffectedByThisAdvisory,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependencyAdvisoryEvidence {
    pub package: DependencyInstance,
    pub authority: DependencyAdvisoryAuthority,
    pub identifier: DiagnosticText,
    pub aliases: Vec<DiagnosticText>,
    pub severity: Option<DiagnosticText>,
    pub affected_versions: DiagnosticText,
    pub fixed_versions: Vec<DiagnosticText>,
    pub state: DependencyAdvisoryState,
    pub provenance: DependencyProvenance,
}

impl DependencyAdvisoryEvidence {
    pub fn validate(&self) -> Result<()> {
        self.package.validate()?;
        ensure!(
            !self.identifier.as_str().is_empty(),
            "Missing advisory identifier"
        );
        ensure!(
            self.aliases.len() <= 32 && self.fixed_versions.len() <= 32,
            "Advisory evidence limit exceeded"
        );
        ensure!(
            self.provenance.limitations.len() <= 32,
            "Advisory provenance limit exceeded"
        );
        ensure!(
            match self.authority {
                DependencyAdvisoryAuthority::RustSec => {
                    self.package.ecosystem == DependencyEcosystem::Cargo
                        && (self.package.source.kind() == DependencySourceKind::CratesIo
                            || (self.state == DependencyAdvisoryState::UnverifiedSource
                                && self.package.source.kind() == DependencySourceKind::Unknown))
                }
                DependencyAdvisoryAuthority::OsvMaven => {
                    self.package.ecosystem == DependencyEcosystem::Maven
                        && (self.package.source.kind() == DependencySourceKind::MavenCentral
                            || (self.state == DependencyAdvisoryState::UnverifiedSource
                                && self.package.source.kind() == DependencySourceKind::Unknown))
                }
                DependencyAdvisoryAuthority::GitHub => true,
            },
            "Advisory authority does not match the resolved package source"
        );
        Ok(())
    }

    pub fn application_exploitability(&self) -> &'static str {
        "Unknown: dependency matching does not establish application exploitability"
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DependencyUpdateKind {
    AdvisoryFix,
    ConstraintCompatible,
    PotentiallyBreaking,
    Replacement,
    UnknownCompatibility,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependencyUpdateEvidence {
    pub package: DependencyInstance,
    pub candidate: DependencyInstance,
    pub kind: DependencyUpdateKind,
    pub advisory_identifiers: Vec<DiagnosticText>,
    pub provenance: DependencyProvenance,
}

impl DependencyUpdateEvidence {
    pub fn validate(&self) -> Result<()> {
        self.package.validate()?;
        self.candidate.validate()?;
        ensure!(
            self.advisory_identifiers.len() <= 32,
            "Update advisory limit exceeded"
        );
        ensure!(
            self.provenance.limitations.len() <= 32,
            "Update provenance limit exceeded"
        );
        ensure!(
            self.kind == DependencyUpdateKind::Replacement
                || (self.package.ecosystem == self.candidate.ecosystem
                    && self.package.source == self.candidate.source
                    && self.package.name == self.candidate.name
                    && self.package.classifier == self.candidate.classifier
                    && self.package.configuration == self.candidate.configuration),
            "A replacement dependency must be explicitly classified"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "jvm-tools")]
    #[test]
    fn doctor_dependency_collection_jvm_disposable_input_has_no_preapproval_writes_or_overwrite() {
        let adapter = DoctorGradleAdapter::planned(&uuid::Uuid::new_v4().to_string())
            .expect("Planned private input");
        let path = adapter.path().to_path_buf();
        assert!(!path.exists());
        drop(adapter);
        assert!(!path.exists(), "Cancelled proposal creates no input");
        let identity = uuid::Uuid::new_v4().to_string();
        let first = DoctorGradleAdapter::planned(&identity).expect("First equivalent proposal");
        let duplicate = DoctorGradleAdapter::planned(&identity).expect("Repeated proposal");
        assert_eq!(first.path(), duplicate.path());
        first.prepare().expect("First approved input");
        assert!(duplicate.prepare().is_err());
        drop(duplicate);
        assert!(
            first.path().exists(),
            "A rejected duplicate cannot delete active input"
        );
        first.close().expect("First input cleanup");
        let adapter = DoctorGradleAdapter::planned(&uuid::Uuid::new_v4().to_string())
            .expect("Approved input fixture");
        let path = adapter.path().to_path_buf();
        adapter.prepare().expect("Approved creation fixture");
        assert_eq!(
            std::fs::read(&path).expect("Exact input"),
            include_bytes!("../../../tooling/jvm/gradle-model/doctor-dependencies.gradle")
        );
        assert!(adapter.prepare().is_err(), "No repeated preparation/replay");
        adapter.close().expect("Explicit terminal cleanup");
        assert!(!path.exists());
        adapter.close().expect("Idempotent cleanup");
        let adapter = DoctorGradleAdapter::planned(&uuid::Uuid::new_v4().to_string())
            .expect("Existing path rejection");
        let path = adapter.path().to_path_buf();
        std::fs::write(&path, b"unrelated sentinel").expect("Foreign test input");
        assert!(adapter.prepare().is_err());
        drop(adapter);
        assert_eq!(
            std::fs::read(&path).expect("Preserved foreign input"),
            b"unrelated sentinel"
        );
        std::fs::remove_file(path).expect("Remove only created test sentinel");
    }

    fn instance(name: &str, version: &str) -> DependencyInstance {
        DependencyInstance {
            ecosystem: DependencyEcosystem::Cargo,
            source: DependencySource::public_registry(DependencyEcosystem::Cargo),
            name: name.into(),
            version: version.into(),
            classifier: None,
            configuration: DependencyIdentity::from_fields(&["locked", "default", "host-target"])
                .expect("Configuration"),
        }
    }

    fn provenance(coverage: DependencyCoverage) -> DependencyProvenance {
        DependencyProvenance {
            host: DependencyIdentity::from_fields(&["selected-host"]).expect("Host"),
            inputs: DependencyIdentity::from_fields(&["metadata", "manifest", "lock"])
                .expect("Inputs"),
            provider: DiagnosticText::new("Cargo owner; TOKEN=PRIVATE_PROVIDER_SECRET"),
            observed_at: 100,
            coverage,
            limitations: Vec::new(),
        }
    }

    fn edge(parent: usize, child: usize, alias: &str) -> DependencyEdge {
        DependencyEdge {
            parent,
            child,
            alias: alias.into(),
            scope: "normal".into(),
            requirement: Some("^1".into()),
            target: None,
            workspace_inherited: false,
        }
    }

    #[test]
    fn doctor_dependency_evidence_preserves_requirement_and_inheritance_edge_identity() {
        let direct = edge(0, 1, "selected");
        let mut inherited = direct.clone();
        inherited.workspace_inherited = true;
        let mut different_request = direct.clone();
        different_request.requirement = Some("^2".into());
        let graph = DependencyGraph::new(
            provenance(DependencyCoverage::Complete),
            vec![
                instance("workspace", "1.0.0"),
                instance("selected", "2.0.0"),
            ],
            vec![direct.clone(), inherited, different_request],
            vec![0],
        )
        .expect("Represented distinct declarations");
        assert_eq!(
            graph
                .introducing_paths(1)
                .expect("Introducing paths")
                .paths
                .len(),
            3
        );
        assert!(
            DependencyGraph::new(
                provenance(DependencyCoverage::Complete),
                graph.nodes().to_vec(),
                vec![direct.clone(), direct],
                vec![0],
            )
            .is_err()
        );
    }

    #[test]
    fn doctor_dependency_evidence_distinguishes_source_version_classifier_and_configuration() {
        let public = instance("shared", "1.0.0");
        let mut private = public.clone();
        private.source = DependencySource::non_public(
            DependencySourceKind::PrivateRegistry,
            "https://PRIVATE_SOURCE_TOKEN@private.example/repository",
        )
        .expect("Opaque source");
        let mut second_version = public.clone();
        second_version.version = "2.0.0".into();
        let mut classifier = public.clone();
        classifier.classifier = Some("tests".into());
        let mut configuration = public.clone();
        configuration.configuration =
            DependencyIdentity::from_fields(&["other-feature"]).expect("Configuration");
        let graph = DependencyGraph::new(
            provenance(DependencyCoverage::Complete),
            vec![
                public.clone(),
                private.clone(),
                second_version,
                classifier,
                configuration,
            ],
            Vec::new(),
            vec![0],
        )
        .expect("Distinct instances");
        assert_eq!(graph.nodes().len(), 5);
        assert_ne!(private.source.identity(), public.source.identity());
        assert!(!format!("{:?}", private.source).contains("PRIVATE_SOURCE_TOKEN"));
        assert!(
            DependencyGraph::new(
                provenance(DependencyCoverage::Complete),
                vec![public.clone(), public],
                Vec::new(),
                vec![0]
            )
            .is_err()
        );
        assert_ne!(
            DependencyIdentity::from_fields(&["ab", "c"]).expect("Framing"),
            DependencyIdentity::from_fields(&["a", "bc"]).expect("Framing")
        );
    }

    #[test]
    fn doctor_dependency_evidence_paths_preserve_alias_scope_and_bound_cycles() {
        let mut inherited = edge(0, 1, "renamed_direct");
        inherited.workspace_inherited = true;
        inherited.scope = "build".into();
        inherited.target = Some("cfg(unix)".into());
        let graph = DependencyGraph::new(
            provenance(DependencyCoverage::Complete),
            vec![
                instance("workspace", "1.0.0"),
                instance("direct", "1.0.0"),
                instance("transitive", "1.0.0"),
            ],
            vec![
                inherited,
                edge(1, 2, "transitive_alias"),
                edge(2, 1, "cycle"),
            ],
            vec![0],
        )
        .expect("Graph");
        let paths = graph.introducing_paths(2).expect("Introducing path");
        assert!(paths.cycle_observed);
        assert_eq!(
            paths.paths,
            vec![DependencyPath {
                nodes: vec![0, 1, 2],
                edges: vec![0, 1]
            }]
        );
        assert!(
            graph
                .edges()
                .first()
                .expect("Declaration")
                .workspace_inherited
        );
        assert_eq!(
            graph.edges().get(1).expect("Alias").alias,
            "transitive_alias"
        );
        assert!(graph.introducing_paths(3).is_err());
    }

    #[test]
    fn doctor_dependency_evidence_path_limit_and_partial_graph_remain_incomplete() {
        let nodes = (0..11)
            .map(|index| instance(&format!("package-{index}"), "1.0.0"))
            .collect();
        let edges = (0..10).map(|index| edge(index, 10, "dependency")).collect();
        let graph = DependencyGraph::new(
            provenance(DependencyCoverage::Complete),
            nodes,
            edges,
            (0..10).collect(),
        )
        .expect("Graph");
        let paths = graph.introducing_paths(10).expect("Paths");
        assert_eq!(paths.paths.len(), MAX_DEPENDENCY_PATHS);
        assert!(paths.truncated);
        let partial = DependencyGraph::new(
            provenance(DependencyCoverage::Partial),
            vec![instance("only", "1.0.0")],
            Vec::new(),
            vec![0],
        )
        .expect("Partial");
        assert!(partial.introducing_paths(0).expect("Paths").truncated);
        assert!(
            DependencyGraph::new(
                provenance(DependencyCoverage::Complete),
                Vec::new(),
                Vec::new(),
                Vec::new()
            )
            .is_err()
        );
    }

    #[test]
    fn doctor_dependency_evidence_depth_disconnected_and_aggregate_bounds() {
        let nodes = (0..34)
            .map(|index| instance(&format!("package-{index}"), "1.0.0"))
            .collect();
        let edges = (0..33)
            .map(|index| edge(index, index + 1, "dependency"))
            .collect();
        let graph = DependencyGraph::new(
            provenance(DependencyCoverage::Complete),
            nodes,
            edges,
            vec![0],
        )
        .expect("Deep graph");
        let paths = graph.introducing_paths(33).expect("Bounded search");
        assert!(paths.truncated);
        assert!(paths.paths.is_empty());
        let disconnected = DependencyGraph::new(
            provenance(DependencyCoverage::Complete),
            vec![instance("root", "1.0.0"), instance("unreachable", "1.0.0")],
            Vec::new(),
            vec![0],
        )
        .expect("Unreachable graph");
        assert!(
            disconnected
                .introducing_paths(1)
                .expect("Unavailable path")
                .truncated
        );
        let nodes = (0..100)
            .map(|index| instance(&format!("package-{index}"), "1.0.0"))
            .collect();
        let mut edges = Vec::new();
        for parent in 0..100 {
            for child in 0..100 {
                let mut dependency = edge(parent, child, "dependency");
                dependency.requirement = Some("x".repeat(MAX_FIELD_BYTES));
                dependency.target = Some("y".repeat(MAX_FIELD_BYTES));
                edges.push(dependency);
            }
        }
        let error = DependencyGraph::new(
            provenance(DependencyCoverage::Complete),
            nodes,
            edges,
            vec![0],
        )
        .expect_err("Aggregate payload must be bounded independently of counts");
        assert!(error.to_string().contains("aggregate byte limit"));
        assert!(
            DependencyGraph::new(
                provenance(DependencyCoverage::Partial),
                vec![instance("root", "1.0.0")],
                vec![edge(0, 0, "edge"); MAX_DEPENDENCY_EDGES + 1],
                vec![0]
            )
            .is_err()
        );
    }

    #[test]
    fn doctor_dependency_evidence_selected_summary_excludes_private_coordinates_and_graphs() {
        let mut private = instance("PRIVATE_COORDINATE", "PRIVATE_VERSION");
        private.source =
            DependencySource::non_public(DependencySourceKind::Git, "/Users/private/SECRET_PATH")
                .expect("Source");
        let graph = DependencyGraph::new(
            provenance(DependencyCoverage::Partial),
            vec![private, instance("public", "1.0.0")],
            vec![edge(0, 1, "PRIVATE_ALIAS")],
            vec![0],
        )
        .expect("Graph");
        let summaries = graph.selected_summaries(&[0, 1]).expect("Selected only");
        let serialized = serde_json::to_string(&summaries).expect("Summary");
        for sentinel in [
            "PRIVATE_COORDINATE",
            "PRIVATE_VERSION",
            "SECRET_PATH",
            "PRIVATE_ALIAS",
            "PRIVATE_PROVIDER_SECRET",
        ] {
            assert!(!serialized.contains(sentinel), "Leaked {sentinel}");
        }
        assert!(serialized.contains("Unknown: a resolved affected package"));
        assert!(graph.selected_summaries(&[0, 0]).is_err());
        assert!(graph.selected_summaries(&[2]).is_err());
        assert!(graph.selected_summaries(&[0; 9]).is_err());
        let restored: Vec<DependencySelectedSummary> =
            serde_json::from_str(&serialized).expect("Redacted history shape");
        assert_eq!(summaries, restored);
    }

    #[test]
    fn doctor_dependency_evidence_rejects_bad_graphs_and_public_disclosure_shapes() {
        let mut node = instance("https://TOKEN@private.example", "1.0.0");
        assert!(node.validate().is_err());
        node.name = "valid".into();
        node.version = "1.0.0?token=SECRET".into();
        assert!(node.validate().is_err());
        assert!(
            DependencyGraph::new(
                provenance(DependencyCoverage::Complete),
                vec![instance("root", "1.0.0")],
                vec![edge(0, 1, "missing")],
                vec![0]
            )
            .is_err()
        );
        assert!(
            DependencyGraph::new(
                provenance(DependencyCoverage::Complete),
                vec![instance("root", "1.0.0")],
                Vec::new(),
                vec![0, 0]
            )
            .is_err()
        );
        assert!(
            DependencyGraph::new(
                provenance(DependencyCoverage::Partial),
                vec![instance("root", "1.0.0"); MAX_DEPENDENCY_NODES + 1],
                Vec::new(),
                vec![0]
            )
            .is_err()
        );
        let mut maven = instance("org.example:library", "1.0-rc1");
        maven.ecosystem = DependencyEcosystem::Maven;
        maven.source = DependencySource::public_registry(DependencyEcosystem::Maven);
        assert!(maven.validate().is_ok());
        maven.name = "org.example:library:private".into();
        assert!(maven.validate().is_err());
    }

    #[test]
    fn doctor_dependency_evidence_advisory_and_update_labels_never_prove_application_fix() {
        let package = instance("affected", "1.0.0");
        let advisory = DependencyAdvisoryEvidence {
            package: package.clone(),
            authority: DependencyAdvisoryAuthority::RustSec,
            identifier: DiagnosticText::new("RUSTSEC-2099-0001"),
            aliases: vec![DiagnosticText::new("CVE-2099-0001")],
            severity: None,
            affected_versions: DiagnosticText::new("<1.0.1"),
            fixed_versions: vec![DiagnosticText::new("1.0.1")],
            state: DependencyAdvisoryState::KnownAffectedDependency,
            provenance: provenance(DependencyCoverage::Complete),
        };
        advisory.validate().expect("Advisory evidence");
        let mut wrong_source = advisory.clone();
        wrong_source.package.source =
            DependencySource::non_public(DependencySourceKind::PrivateRegistry, "private-owner")
                .expect("Private source");
        assert!(wrong_source.validate().is_err());
        wrong_source = advisory.clone();
        wrong_source.authority = DependencyAdvisoryAuthority::OsvMaven;
        assert!(wrong_source.validate().is_err());
        assert!(advisory.application_exploitability().starts_with("Unknown"));
        let mut update = DependencyUpdateEvidence {
            package,
            candidate: instance("replacement", "2.0.0"),
            kind: DependencyUpdateKind::ConstraintCompatible,
            advisory_identifiers: vec![advisory.identifier],
            provenance: provenance(DependencyCoverage::Complete),
        };
        assert!(update.validate().is_err());
        update.kind = DependencyUpdateKind::Replacement;
        assert!(update.validate().is_ok());
        update.kind = DependencyUpdateKind::AdvisoryFix;
        update.candidate = instance("affected", "1.0.1");
        assert!(update.validate().is_ok());
        update.advisory_identifiers = vec![DiagnosticText::new("CVE"); 33];
        assert!(update.validate().is_err());
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DoctorDependencyReportFormat {
    CargoMetadata,
    MavenSelectedTree,
    GradleResolution {
        project_path: String,
        configuration: String,
    },
}

#[cfg(feature = "jvm-tools")]
pub(crate) struct DoctorGradleAdapter {
    path: std::path::PathBuf,
    file: std::sync::Mutex<Option<tempfile::TempPath>>,
}

#[cfg(feature = "jvm-tools")]
impl DoctorGradleAdapter {
    pub(crate) fn planned(identity: &str) -> Result<std::sync::Arc<Self>> {
        let root = std::env::temp_dir()
            .canonicalize()
            .map_err(|_| anyhow::anyhow!("Private Task temporary directory unavailable"))?;
        Ok(std::sync::Arc::new(Self {
            path: root.join(format!(
                "made-doctor-gradle-{:x}.gradle",
                Sha256::digest(identity.as_bytes())
            )),
            file: std::sync::Mutex::new(None),
        }))
    }
    pub(crate) fn path(&self) -> &std::path::Path {
        &self.path
    }
    fn prepare(&self) -> Result<()> {
        use std::io::Write as _;
        let mut owned = self
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("Bundled adapter input lock unavailable"))?;
        ensure!(
            owned.is_none(),
            "Bundled adapter was already prepared; no operation replay"
        );
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let mut file = options.open(&self.path).map_err(|_| {
            anyhow::anyhow!(
                "New private adapter input cannot be created; existing files are untouched"
            )
        })?;
        *owned = Some(tempfile::TempPath::from_path(self.path.clone()));
        file.write_all(include_bytes!(
            "../../../tooling/jvm/gradle-model/doctor-dependencies.gradle"
        ))?;
        file.sync_all()?;
        Ok(())
    }
    fn close(&self) -> Result<()> {
        if let Some(file) = self
            .file
            .lock()
            .map_err(|_| anyhow::anyhow!("Bundled adapter cleanup lock unavailable"))?
            .take()
        {
            file.close()?;
        }
        Ok(())
    }
    fn preview(&self) -> String {
        format!(
            " After explicit approval, create one private disposable local Task input {:?} containing only the bundled Made Gradle adapter (SHA-256 {:x}); no project script is installed or overwritten. Remove that owned file when the operation ends, including cancellation; reached project/cache writes remain. Installed Gradle 8.13/JDK required; no wrapper/bootstrap download is invoked.",
            self.path,
            Sha256::digest(include_bytes!(
                "../../../tooling/jvm/gradle-model/doctor-dependencies.gradle"
            ))
        )
    }
}

#[cfg(feature = "jvm-tools")]
impl Drop for DoctorGradleAdapter {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            log::warn!("Doctor disposable Gradle input cleanup failed: {error}");
        }
    }
}

#[derive(Clone)]
pub(crate) struct DoctorDependencyCollection {
    artifact: task::TaskArtifact,
    path: project::ProjectPath,
    source_identity: crate::doctor::DoctorInvestigationInputIdentity,
    format: DoctorDependencyReportFormat,
    #[cfg(feature = "rust-tools")]
    cargo_inputs: Option<project::cargo_workspace_store::CargoReviewedMetadataImport>,
    #[cfg(feature = "jvm-tools")]
    jvm_inputs: Option<project::jvm_project::JvmReviewedDependencyImport>,
    #[cfg(feature = "jvm-tools")]
    gradle_adapter: Option<std::sync::Arc<DoctorGradleAdapter>>,
}

pub struct DoctorCollectedDependencyReport {
    pub(crate) path: project::ProjectPath,
    pub(crate) format: DoctorDependencyReportFormat,
    pub(crate) bytes: Vec<u8>,
    pub(crate) source_identity: crate::doctor::DoctorInvestigationInputIdentity,
    pub(crate) binding: crate::doctor::DoctorRepairBinding,
    pub(crate) task_receipt: uuid::Uuid,
    #[cfg(feature = "rust-tools")]
    cargo_inputs: Option<project::cargo_workspace_store::CargoReviewedMetadataImport>,
    #[cfg(feature = "jvm-tools")]
    jvm_inputs: Option<project::jvm_project::JvmReviewedDependencyImport>,
}

impl DoctorCollectedDependencyReport {
    #[cfg(feature = "jvm-tools")]
    pub fn admit_jvm(
        mut self,
        project: gpui::Entity<project::Project>,
        cx: &mut gpui::App,
    ) -> gpui::Task<Result<project::jvm_project::JvmDependencyGraph>> {
        let prepared = self.revalidate_jvm_result(&project, cx).and_then(|()| {
            let inputs = self
                .jvm_inputs
                .take()
                .ok_or_else(|| anyhow::anyhow!("Report lacks reviewed JVM owner inputs"))?;
            let store = project.read(cx).lsp_store();
            Ok(store.update(cx, |store, cx| {
                store.admit_reviewed_jvm_dependency_import(
                    inputs,
                    std::mem::take(&mut self.bytes),
                    cx,
                )
            }))
        });
        match prepared {
            Err(error) => gpui::Task::ready(Err(error)),
            Ok(pending) => cx.spawn(async move |cx| {
                let graph = pending.await?;
                cx.update(|cx| self.revalidate_jvm_result(&project, cx))?;
                Ok(graph)
            }),
        }
    }

    #[cfg(feature = "jvm-tools")]
    fn revalidate_jvm_result(
        &self,
        project: &gpui::Entity<project::Project>,
        cx: &gpui::App,
    ) -> Result<()> {
        ensure!(
            matches!(
                self.format,
                DoctorDependencyReportFormat::MavenSelectedTree
                    | DoctorDependencyReportFormat::GradleResolution { .. }
            ),
            "This report is not a reviewed JVM dependency graph"
        );
        self.binding.revalidate_project(project, cx)?;
        ensure!(
            crate::doctor::comparison_source_identity_excluding(
                project,
                self.binding.directory(),
                Some(&self.path),
                cx
            )? == self.source_identity,
            "JVM result source inputs changed"
        );
        ensure!(
            !project.read(cx).is_shared(),
            "JVM result host became shared"
        );
        Ok(())
    }
    #[cfg(feature = "rust-tools")]
    pub fn admit_cargo(
        mut self,
        project: gpui::Entity<project::Project>,
        cx: &mut gpui::App,
    ) -> gpui::Task<Result<project::cargo_workspace::CargoWorkspaceModel>> {
        let prepared = self.revalidate_cargo_result(&project, cx).and_then(|()| {
            let inputs = self
                .cargo_inputs
                .take()
                .ok_or_else(|| anyhow::anyhow!("Report lacks reviewed Cargo owner inputs"))?;
            let store = project.read(cx).cargo_workspace_store().clone();
            Ok(store.update(cx, |store, cx| {
                store.admit_reviewed_metadata_import(inputs, std::mem::take(&mut self.bytes), cx)
            }))
        });
        match prepared {
            Err(error) => gpui::Task::ready(Err(error)),
            Ok(pending) => cx.spawn(async move |cx| {
                let workspace = pending.await?;
                cx.update(|cx| self.revalidate_cargo_result(&project, cx))?;
                Ok(workspace)
            }),
        }
    }

    #[cfg(feature = "rust-tools")]
    fn revalidate_cargo_result(
        &self,
        project: &gpui::Entity<project::Project>,
        cx: &gpui::App,
    ) -> Result<()> {
        ensure!(
            self.format == DoctorDependencyReportFormat::CargoMetadata,
            "This report is not reviewed Cargo metadata"
        );
        self.binding.revalidate_project(project, cx)?;
        ensure!(
            crate::doctor::comparison_source_identity_excluding(
                project,
                self.binding.directory(),
                Some(&self.path),
                cx
            )? == self.source_identity,
            "Cargo result source inputs changed"
        );
        ensure!(
            !project.read(cx).is_shared(),
            "Cargo result host became shared"
        );
        Ok(())
    }

    pub fn byte_count(&self) -> usize {
        self.bytes.len()
    }

    pub fn report_identity(&self) -> DependencyIdentity {
        DependencyIdentity(Sha256::digest(&self.bytes).into())
    }

    pub fn path(&self) -> &project::ProjectPath {
        &self.path
    }

    pub fn format(&self) -> &DoctorDependencyReportFormat {
        &self.format
    }

    pub fn source_identity(&self) -> &crate::doctor::DoctorInvestigationInputIdentity {
        &self.source_identity
    }

    pub fn binding(&self) -> &crate::doctor::DoctorRepairBinding {
        &self.binding
    }

    pub fn task_receipt(&self) -> uuid::Uuid {
        self.task_receipt
    }
}

impl DoctorDependencyCollection {
    #[cfg(all(test, feature = "jvm-tools"))]
    pub(crate) fn runtime_input_exists_for_test(&self) -> bool {
        self.gradle_adapter
            .as_ref()
            .is_some_and(|adapter| adapter.path().exists())
    }

    #[cfg(feature = "jvm-tools")]
    pub(crate) fn bind_gradle_adapter(&mut self, adapter: std::sync::Arc<DoctorGradleAdapter>) {
        self.gradle_adapter = Some(adapter);
    }

    #[cfg(feature = "jvm-tools")]
    pub(crate) fn cleanup_runtime_inputs(&self) -> Result<()> {
        if let Some(adapter) = &self.gradle_adapter {
            adapter.close()?;
        }
        Ok(())
    }

    #[cfg(feature = "jvm-tools")]
    pub(crate) fn bind_jvm_import(
        &mut self,
        inputs: project::jvm_project::JvmReviewedDependencyImport,
    ) {
        self.jvm_inputs = Some(inputs);
    }

    #[cfg(feature = "rust-tools")]
    pub(crate) fn bind_cargo_import(
        &mut self,
        inputs: project::cargo_workspace_store::CargoReviewedMetadataImport,
    ) {
        self.cargo_inputs = Some(inputs);
    }

    pub(crate) fn new(
        project: &gpui::Entity<project::Project>,
        directory: &project::ProjectPath,
        artifact: task::TaskArtifact,
        format: DoctorDependencyReportFormat,
        cx: &gpui::App,
    ) -> Result<Self> {
        let artifact = artifact.with_resolved_path(artifact.path.clone())?;
        ensure!(
            artifact.kind == task::TaskArtifactKind::Data
                && artifact.max_bytes <= MAX_DEPENDENCY_GRAPH_BYTES as u64,
            "Dependency collection requires one bounded data report of at most eight MiB"
        );
        let path = project::ProjectPath {
            worktree_id: directory.worktree_id,
            path: std::sync::Arc::from(
                directory
                    .path
                    .join(util::rel_path::RelPath::from_unix_str(&artifact.path)?),
            ),
        };
        let owner = project.read(cx);
        ensure!(
            owner.is_local() && !owner.is_shared(),
            "Bounded dependency report collection requires a local owned development host; remote collection is unsupported without host controls"
        );
        ensure!(
            owner.entry_for_path(&path, cx).is_none(),
            "Choose a new dependency report; existing artifacts are not overwritten or reused"
        );
        let collection = Self {
            source_identity: crate::doctor::comparison_source_identity_excluding(
                project,
                directory,
                Some(&path),
                cx,
            )?,
            artifact,
            path,
            format,
            #[cfg(feature = "rust-tools")]
            cargo_inputs: None,
            #[cfg(feature = "jvm-tools")]
            jvm_inputs: None,
            #[cfg(feature = "jvm-tools")]
            gradle_adapter: None,
        };
        collection.revalidate(project, directory, cx)?;
        Ok(collection)
    }

    pub(crate) fn exact_preview(&self) -> String {
        let preview = format!(
            "Collect only the new {:?} dependency report {:?}, at most {} bytes, through the existing confined project filesystem after successful Task completion. Repository/advisory conclusions require existing owner validation; no raw report enters Doctor history, export or model handoff. Reached writes remain with the selected Task; interruption never replays the command.",
            self.format, self.artifact.path, self.artifact.max_bytes
        );
        #[cfg(feature = "jvm-tools")]
        if let Some(adapter) = &self.gradle_adapter {
            return format!("{preview}{}", adapter.preview());
        }
        preview
    }

    pub(crate) fn revalidate(
        &self,
        project: &gpui::Entity<project::Project>,
        directory: &project::ProjectPath,
        cx: &gpui::App,
    ) -> Result<()> {
        use settings::Settings as _;
        let owner = project.read(cx);
        ensure!(
            owner.is_local()
                && !owner.is_shared()
                && self.path.worktree_id == directory.worktree_id,
            "Dependency collection host or project authority changed"
        );
        let settings = project::WorktreeSettings::get(
            Some(settings::SettingsLocation {
                worktree_id: directory.worktree_id,
                path: self.path.path.as_ref(),
            }),
            cx,
        );
        ensure!(
            !settings.is_path_private(&self.path.path)
                && !settings.is_path_excluded(&self.path.path)
                && !settings.is_path_read_only(&self.path.path),
            "Selected dependency report path is private, excluded or read-only"
        );
        let worktree = owner
            .worktree_for_id(directory.worktree_id, cx)
            .ok_or_else(|| anyhow::anyhow!("Dependency worktree is unavailable"))?;
        ensure!(
            worktree.read(cx).as_local().is_some(),
            "Dependency host has no local confined filesystem"
        );
        ensure!(
            !owner.opened_buffers(cx).into_iter().any(|buffer| {
                let buffer = buffer.read(cx);
                buffer.is_dirty()
                    && project::File::from_dyn(buffer.file())
                        .is_some_and(|file| file.worktree_id(cx) == directory.worktree_id)
            }),
            "Save or resolve pending project edits separately before collecting actual dependency inputs"
        );
        ensure!(
            crate::doctor::comparison_source_identity_excluding(
                project,
                directory,
                Some(&self.path),
                cx
            )? == self.source_identity,
            "Reviewed dependency source inputs changed; collect and review a fresh operation"
        );
        #[cfg(feature = "rust-tools")]
        if let Some(inputs) = &self.cargo_inputs {
            project
                .read(cx)
                .cargo_workspace_store()
                .read(cx)
                .revalidate_reviewed_metadata_import(inputs, cx)?;
        }
        #[cfg(feature = "jvm-tools")]
        if let Some(inputs) = &self.jvm_inputs {
            project
                .read(cx)
                .lsp_store()
                .read(cx)
                .revalidate_reviewed_jvm_dependency_import(inputs, cx)?;
        }
        Ok(())
    }

    pub(crate) fn preflight(
        &self,
        project: &gpui::Entity<project::Project>,
        directory: &project::ProjectPath,
        cx: &gpui::App,
    ) -> Result<gpui::Task<Result<()>>> {
        self.revalidate(project, directory, cx)?;
        let (filesystem, root, root_inode) = self.filesystem(project, cx)?;
        let relative = self.path.path.as_std_path().to_path_buf();
        #[cfg(feature = "rust-tools")]
        let cargo_inputs = self.cargo_inputs.clone();
        #[cfg(feature = "jvm-tools")]
        let jvm_inputs = self.jvm_inputs.clone();
        #[cfg(feature = "jvm-tools")]
        let gradle_adapter = self.gradle_adapter.clone();
        Ok(cx.background_executor().spawn(async move {
            #[cfg(feature = "rust-tools")]
            if let Some(inputs) = cargo_inputs {
                inputs.preflight_filesystem(filesystem.as_ref()).await?;
            }
            #[cfg(feature = "jvm-tools")]
            if let Some(inputs) = jvm_inputs {
                inputs.preflight_filesystem(filesystem.as_ref()).await?;
            }
            #[cfg(feature = "jvm-tools")]
            if let Some(adapter) = gradle_adapter {
                adapter.prepare()?;
            }
            let metadata = filesystem
                .metadata(&root)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Dependency root is unavailable"))?;
            ensure!(
                metadata.is_dir && !metadata.is_symlink && metadata.inode == root_inode,
                "Dependency root changed or is not a canonical directory"
            );
            let mut parent = root.clone();
            if let Some(relative_parent) = relative.parent() {
                for component in relative_parent.components() {
                    ensure!(
                        matches!(component, std::path::Component::Normal(_)),
                        "Dependency report parent is not a confined project path"
                    );
                    parent.push(component);
                    let metadata = filesystem.metadata(&parent).await?.ok_or_else(|| {
                        anyhow::anyhow!("Choose an existing dependency report directory")
                    })?;
                    ensure!(
                        metadata.is_dir && !metadata.is_symlink,
                        "Dependency report parent is not an ordinary contained directory"
                    );
                }
            }
            ensure!(
                filesystem.metadata(&root.join(&relative)).await?.is_none(),
                "The reviewed new dependency report already exists; no Task was dispatched"
            );
            Ok(())
        }))
    }

    fn filesystem(
        &self,
        project: &gpui::Entity<project::Project>,
        cx: &gpui::App,
    ) -> Result<(std::sync::Arc<dyn fs::Fs>, std::path::PathBuf, u64)> {
        let owner = project.read(cx);
        let worktree = owner
            .worktree_for_id(self.path.worktree_id, cx)
            .ok_or_else(|| anyhow::anyhow!("Dependency worktree is unavailable"))?;
        let worktree = worktree.read(cx);
        let root_inode = worktree
            .snapshot()
            .root_entry()
            .ok_or_else(|| anyhow::anyhow!("Dependency worktree root is unavailable"))?
            .inode;
        Ok((
            owner.fs().clone(),
            worktree.abs_path().as_ref().to_path_buf(),
            root_inode,
        ))
    }

    pub(crate) fn capture(
        &self,
        project: &gpui::Entity<project::Project>,
        directory: &project::ProjectPath,
        binding: crate::doctor::DoctorRepairBinding,
        task_receipt: uuid::Uuid,
        cx: &gpui::App,
    ) -> Result<gpui::Task<Result<DoctorCollectedDependencyReport>>> {
        self.revalidate(project, directory, cx)?;
        let (filesystem, root, root_inode) = self.filesystem(project, cx)?;
        let collection = self.clone();
        Ok(cx.background_executor().spawn(async move {
            let relative = collection.path.path.as_std_path();
            let metadata = filesystem.metadata(&root.join(relative)).await?
                .ok_or_else(|| anyhow::anyhow!("Successful Task did not produce its reviewed dependency report"))?;
            let bytes = crate::doctor::read_confined_declaration(filesystem.as_ref(), &root, root_inode,
                relative, (metadata.inode, Some(metadata.mtime), metadata.len), collection.artifact.max_bytes).await?.into_bytes();
            ensure!(!bytes.is_empty(), "Dependency report is empty");
            let value: serde_json::Value = serde_json::from_slice(&bytes)?;
            ensure!(value.is_object(), "Dependency report is not a structured object");
            match &collection.format {
                DoctorDependencyReportFormat::CargoMetadata => ensure!(value.get("version").and_then(serde_json::Value::as_u64) == Some(1)
                    && value.get("packages").is_some_and(serde_json::Value::is_array), "Not a supported Cargo metadata report"),
                DoctorDependencyReportFormat::MavenSelectedTree => ensure!(value.get("groupId").is_some_and(serde_json::Value::is_string)
                    && value.get("artifactId").is_some_and(serde_json::Value::is_string), "Not a Maven dependency tree; effective selected mode still requires owner validation"),
                DoctorDependencyReportFormat::GradleResolution { project_path, configuration } => ensure!(value.get("version").and_then(serde_json::Value::as_u64) == Some(1)
                    && value.get("projectPath").and_then(serde_json::Value::as_str) == Some(project_path.as_str())
                    && value.get("configuration").and_then(serde_json::Value::as_str) == Some(configuration.as_str()), "Gradle report differs from the reviewed format/project/configuration"),
            }
            Ok(DoctorCollectedDependencyReport {
                path: collection.path,
                format: collection.format,
                bytes,
                source_identity: collection.source_identity,
                binding,
                task_receipt,
                #[cfg(feature = "rust-tools")]
                cargo_inputs: collection.cargo_inputs,
                #[cfg(feature = "jvm-tools")]
                jvm_inputs: collection.jvm_inputs,
            })
        }))
    }
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
pub use advisory_cache::{
    AdvisoryArchive, AdvisoryArchiveKind, AdvisoryArchiveReceipt, AdvisoryCache,
    AdvisoryCacheState, AdvisoryDataset, AdvisoryFreshness, AdvisoryRecord, AdvisorySnapshot,
};

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
mod advisory_cache {
    use super::*;
    use futures::io::{AsyncReadExt as _, BufReader, Cursor};
    use std::{
        io::Read as _,
        path::{Component, Path, PathBuf},
        sync::{Arc, LazyLock, Weak},
    };

    pub const MAX_ADVISORY_CACHE_BYTES: usize = 256 * 1024 * 1024;
    pub const MAX_ADVISORY_ARCHIVE_BYTES: usize = 128 * 1024 * 1024;
    const CACHE_LOCK_FILENAME: &str = ".made-doctor-advisories.lock";
    const MAX_RECORD_BYTES: usize = 1024 * 1024;
    const MAX_ARCHIVE_ENTRIES: usize = 65534;
    const MAX_RECORDS: usize = 65534;
    const FRESH_SECONDS: u64 = 24 * 60 * 60;
    const RETENTION_SECONDS: u64 = 30 * FRESH_SECONDS;
    type CacheOperationLock = Arc<async_lock::Mutex<()>>;
    static CACHE_OPERATIONS: LazyLock<
        std::sync::Mutex<std::collections::HashMap<(usize, PathBuf), Weak<async_lock::Mutex<()>>>>,
    > = LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
    pub enum AdvisoryDataset {
        RustSec,
        OsvMaven,
    }

    impl AdvisoryDataset {
        pub(super) fn filename(self) -> &'static str {
            match self {
                Self::RustSec => "rustsec-v1.json",
                Self::OsvMaven => "osv-maven-v1.json",
            }
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
    pub enum AdvisoryArchiveKind {
        RustSec,
        OsvMaven,
        OsvWithdrawals,
    }

    impl AdvisoryArchiveKind {
        pub fn source_url(self, revision: &str) -> String {
            match self {
                Self::RustSec => {
                    format!("https://codeload.github.com/RustSec/advisory-db/zip/{revision}")
                }
                Self::OsvMaven => {
                    "https://storage.googleapis.com/osv-vulnerabilities/Maven/all.zip".into()
                }
                Self::OsvWithdrawals => {
                    "https://storage.googleapis.com/osv-vulnerabilities/%5BEMPTY%5D/all.zip".into()
                }
            }
        }
    }

    pub struct AdvisoryArchive {
        pub kind: AdvisoryArchiveKind,
        pub revision: String,
        pub bytes: Vec<u8>,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct AdvisoryArchiveReceipt {
        pub kind: AdvisoryArchiveKind,
        pub source_url: String,
        pub revision: String,
        pub sha256: String,
        pub downloaded_bytes: u64,
        pub expanded_bytes: u64,
        pub records: usize,
        pub records_sha256: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub source_modified_at: Option<u64>,
    }

    #[derive(Clone, Eq, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct AdvisoryRecord {
        pub kind: AdvisoryArchiveKind,
        pub identifier: String,
        pub document: String,
    }

    #[derive(Clone, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct AdvisorySnapshot {
        format: u32,
        pub dataset: AdvisoryDataset,
        host: String,
        pub fetched_at: u64,
        pub sources: Vec<AdvisoryArchiveReceipt>,
        pub records: Vec<AdvisoryRecord>,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum AdvisoryFreshness {
        Fresh,
        Stale,
        Expired,
        ClockUncertain,
    }

    impl AdvisorySnapshot {
        pub fn freshness(&self, now: u64) -> AdvisoryFreshness {
            let Some(age) = now.checked_sub(self.fetched_at) else {
                return AdvisoryFreshness::ClockUncertain;
            };
            if age >= RETENTION_SECONDS {
                AdvisoryFreshness::Expired
            } else if age >= FRESH_SECONDS {
                AdvisoryFreshness::Stale
            } else {
                AdvisoryFreshness::Fresh
            }
        }

        pub fn upstream_freshness(&self, now: u64) -> Option<AdvisoryFreshness> {
            let mut unknown = false;
            for source in &self.sources {
                let Some(published) = source.source_modified_at else {
                    unknown = true;
                    continue;
                };
                let Some(age) = now.checked_sub(published) else {
                    return Some(AdvisoryFreshness::ClockUncertain);
                };
                if age >= FRESH_SECONDS {
                    return Some(AdvisoryFreshness::Stale);
                }
            }
            if unknown {
                None
            } else {
                Some(AdvisoryFreshness::Fresh)
            }
        }

        pub fn identity(&self) -> Result<DependencyIdentity> {
            Ok(DependencyIdentity(
                Sha256::digest(serde_json::to_vec(self)?).into(),
            ))
        }

        pub(crate) fn validate_for_matching(&self) -> Result<()> {
            self.validate(&self.host)
        }

        fn validate(&self, host: &str) -> Result<()> {
            ensure!(
                self.format == 1 && self.host == host,
                "Advisory cache identity mismatch"
            );
            ensure!(
                self.records.len() <= MAX_RECORDS,
                "Advisory record limit exceeded"
            );
            let expected: &[AdvisoryArchiveKind] = match self.dataset {
                AdvisoryDataset::RustSec => &[AdvisoryArchiveKind::RustSec],
                AdvisoryDataset::OsvMaven => &[
                    AdvisoryArchiveKind::OsvMaven,
                    AdvisoryArchiveKind::OsvWithdrawals,
                ],
            };
            ensure!(
                self.sources.len() == expected.len()
                    && expected
                        .iter()
                        .zip(&self.sources)
                        .all(|(kind, source)| *kind == source.kind),
                "Advisory source or withdrawal coverage is incomplete"
            );
            let mut identifiers = BTreeSet::new();
            let mut counts = BTreeMap::new();
            let mut total_bytes = 0usize;
            for source in &self.sources {
                validate_revision(source.kind, &source.revision)?;
                ensure!(
                    source.source_url == source.kind.source_url(&source.revision)
                        && valid_digest(&source.sha256)
                        && source.downloaded_bytes <= MAX_ADVISORY_ARCHIVE_BYTES as u64
                        && source.expanded_bytes <= MAX_ADVISORY_CACHE_BYTES as u64,
                    "Invalid advisory source receipt"
                );
            }
            for record in &self.records {
                ensure!(
                    expected.contains(&record.kind),
                    "Unexpected advisory record source"
                );
                ensure!(
                    identifiers.insert((record.kind, record.identifier.as_str())),
                    "Duplicate advisory record"
                );
                validate_document(record.kind, &record.identifier, &record.document)?;
                total_bytes = total_bytes
                    .checked_add(record.document.len())
                    .ok_or_else(|| anyhow::anyhow!("Advisory size overflow"))?;
                ensure!(
                    total_bytes <= MAX_ADVISORY_CACHE_BYTES,
                    "Advisory expansion limit exceeded"
                );
                *counts.entry(record.kind).or_insert(0usize) += 1;
            }
            for source in &self.sources {
                ensure!(
                    source.records_sha256 == records_digest(source.kind, &self.records),
                    "Advisory record integrity mismatch"
                );
                ensure!(
                    source.records == *counts.get(&source.kind).unwrap_or(&0),
                    "Advisory receipt count mismatch"
                );
                ensure!(
                    source.kind == AdvisoryArchiveKind::OsvWithdrawals || source.records > 0,
                    "Empty authoritative advisory dataset"
                );
            }
            Ok(())
        }
    }

    pub struct AdvisoryCacheState {
        pub revision: Option<DependencyIdentity>,
        pub snapshot: Option<AdvisorySnapshot>,
        pub problem: Option<DiagnosticText>,
    }

    fn shared_cache_root(product_cache: &Path) -> Result<PathBuf> {
        ensure!(
            product_cache.is_absolute(),
            "Native cache base is not absolute"
        );
        Ok(product_cache
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Native cache base is unavailable"))?
            .join("made-doctor-advisories"))
    }

    #[cfg(unix)]
    fn acquire_native_cache_guard(root: &Path, inode: u64) -> Result<std::fs::File> {
        use std::os::unix::fs::OpenOptionsExt as _;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(root.join(CACHE_LOCK_FILENAME))
            .map_err(|_| {
                anyhow::anyhow!("Public advisory cache admission is unsafe or unavailable")
            })?;
        file.try_lock().map_err(|_| anyhow::anyhow!("Another Made, Copper or Groovy operation owns the public advisory cache; review again after it finishes"))?;
        verify_native_cache_guard(root, inode, &file)?;
        Ok(file)
    }

    #[cfg(unix)]
    fn verify_native_cache_guard(root: &Path, inode: u64, file: &std::fs::File) -> Result<()> {
        use std::os::unix::fs::MetadataExt as _;
        let metadata = file.metadata()?;
        let path = root.join(CACHE_LOCK_FILENAME);
        let named = std::fs::symlink_metadata(&path)?;
        let directory = std::fs::symlink_metadata(root)?;
        ensure!(
            metadata.is_file()
                && metadata.len() == 0
                && metadata.nlink() == 1
                && metadata.mode() & 0o077 == 0
                && metadata.uid() == unsafe { libc::geteuid() }
                && named.is_file()
                && !named.is_symlink()
                && metadata.ino() == named.ino()
                && metadata.dev() == named.dev()
                && directory.is_dir()
                && !directory.is_symlink()
                && directory.ino() == inode
                && std::fs::canonicalize(&path)? == path,
            "Public advisory cache admission changed or is not an owned empty regular lock"
        );
        Ok(())
    }

    #[cfg(not(unix))]
    fn acquire_native_cache_guard(_root: &Path, _inode: u64) -> Result<std::fs::File> {
        anyhow::bail!("Native public advisory cache transactions are unsupported on this platform")
    }

    #[cfg(not(unix))]
    fn verify_native_cache_guard(_root: &Path, _inode: u64, _file: &std::fs::File) -> Result<()> {
        anyhow::bail!("Native public advisory cache transactions are unsupported on this platform")
    }

    pub struct AdvisoryCache {
        fs: Arc<dyn fs::Fs>,
        root: PathBuf,
        host: String,
        maximum_bytes: usize,
        operations: CacheOperationLock,
    }

    impl AdvisoryCache {
        pub(crate) fn root_for_preview(&self) -> &std::path::Path {
            &self.root
        }

        #[cfg(all(test, feature = "rust-tools", feature = "jvm-tools"))]
        pub(super) async fn hold_operations_for_test(&self) -> async_lock::MutexGuard<'_, ()> {
            self.operations.lock().await
        }

        pub fn for_local_project(project: &project::Project) -> Result<Self> {
            ensure!(
                project.is_local(),
                "Advisory cache on this remote execution host is unsupported"
            );
            Self::new(
                project.fs().clone(),
                shared_cache_root(paths::temp_dir())?,
                &DependencyIdentity::from_fields(&["selected-local-application-host"])?,
            )
        }

        fn new(fs: Arc<dyn fs::Fs>, root: PathBuf, host: &DependencyIdentity) -> Result<Self> {
            ensure!(
                root.is_absolute()
                    && root.components().all(|component| matches!(
                        component,
                        Component::RootDir | Component::Prefix(_) | Component::Normal(_)
                    )),
                "Invalid advisory cache root"
            );
            let key = (Arc::as_ptr(&fs) as *const () as usize, root.clone());
            let mut locks = CACHE_OPERATIONS
                .lock()
                .map_err(|_| anyhow::anyhow!("Advisory cache lock is unavailable"))?;
            locks.retain(|_, lock| lock.strong_count() > 0);
            let operations = locks
                .get(&key)
                .and_then(Weak::upgrade)
                .unwrap_or_else(|| Arc::new(async_lock::Mutex::new(())));
            locks.insert(key, Arc::downgrade(&operations));
            Ok(Self {
                fs,
                root,
                host: host.hex(),
                maximum_bytes: MAX_ADVISORY_CACHE_BYTES,
                operations,
            })
        }

        pub async fn validate_archives(
            &self,
            dataset: AdvisoryDataset,
            archives: Vec<AdvisoryArchive>,
            fetched_at: u64,
        ) -> Result<AdvisorySnapshot> {
            let expected: &[AdvisoryArchiveKind] = match dataset {
                AdvisoryDataset::RustSec => &[AdvisoryArchiveKind::RustSec],
                AdvisoryDataset::OsvMaven => &[
                    AdvisoryArchiveKind::OsvMaven,
                    AdvisoryArchiveKind::OsvWithdrawals,
                ],
            };
            ensure!(
                archives.len() == expected.len()
                    && archives
                        .iter()
                        .zip(expected)
                        .all(|(archive, kind)| archive.kind == *kind),
                "Advisory source or withdrawal coverage is incomplete"
            );
            let mut sources = Vec::new();
            let mut records = Vec::new();
            let mut total_expansion = 0usize;
            for archive in archives {
                validate_revision(archive.kind, &archive.revision)?;
                let (receipt, mut entries) = read_archive(archive).await?;
                total_expansion = total_expansion
                    .checked_add(usize::try_from(receipt.expanded_bytes)?)
                    .ok_or_else(|| anyhow::anyhow!("Advisory expansion overflow"))?;
                ensure!(
                    total_expansion <= MAX_ADVISORY_CACHE_BYTES
                        && records.len().saturating_add(entries.len()) <= MAX_RECORDS,
                    "Advisory expansion or record limit exceeded"
                );
                sources.push(receipt);
                records.append(&mut entries);
            }
            let snapshot = AdvisorySnapshot {
                format: 1,
                dataset,
                host: self.host.clone(),
                fetched_at,
                sources,
                records,
            };
            snapshot.validate(&self.host)?;
            Ok(snapshot)
        }

        async fn ordinary_root(&self) -> Result<u64> {
            let mut path = PathBuf::new();
            for component in self.root.components() {
                path.push(component);
                if self.fs.metadata(&path).await?.is_none() {
                    self.fs.create_dir(&path).await?;
                }
                let metadata =
                    self.fs.metadata(&path).await?.ok_or_else(|| {
                        anyhow::anyhow!("Advisory cache directory is unavailable")
                    })?;
                ensure!(
                    metadata.is_dir && !metadata.is_symlink,
                    "Advisory cache directory is not ordinary"
                );
            }
            ensure!(
                self.fs.canonicalize(&self.root).await? == self.root,
                "Advisory cache path changed"
            );
            Ok(self
                .fs
                .metadata(&self.root)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Advisory cache directory disappeared"))?
                .inode)
        }

        fn native_guard(&self, inode: u64) -> Result<Option<std::fs::File>> {
            if self.fs.is_fake() {
                return Ok(None);
            }
            acquire_native_cache_guard(&self.root, inode).map(Some)
        }

        fn verify_guard(&self, inode: u64, guard: &Option<std::fs::File>) -> Result<()> {
            if let Some(guard) = guard {
                verify_native_cache_guard(&self.root, inode, guard)?;
            }
            Ok(())
        }

        async fn retained_bytes(
            &self,
            dataset: AdvisoryDataset,
            root_inode: u64,
        ) -> Result<Option<Vec<u8>>> {
            let path = self.root.join(dataset.filename());
            let Some(metadata) = self.fs.metadata(&path).await? else {
                return Ok(None);
            };
            ensure!(
                !metadata.is_dir
                    && !metadata.is_symlink
                    && !metadata.is_fifo
                    && metadata.len <= self.maximum_bytes as u64,
                "Advisory cache file is unsafe or exceeds its limit"
            );
            let mut file = self
                .fs
                .open_sync_confined_file(
                    &self.root,
                    Path::new(dataset.filename()),
                    root_inode,
                    metadata.inode,
                )
                .await?;
            let mut bytes = Vec::new();
            file.by_ref()
                .take(self.maximum_bytes as u64 + 1)
                .read_to_end(&mut bytes)?;
            let after = self
                .fs
                .metadata(&path)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Advisory cache file disappeared"))?;
            ensure!(
                bytes.len() <= self.maximum_bytes
                    && after.inode == metadata.inode
                    && after.mtime == metadata.mtime
                    && after.len == metadata.len
                    && bytes.len() as u64 == metadata.len
                    && !after.is_symlink,
                "Advisory cache file changed"
            );
            Ok(Some(bytes))
        }

        pub async fn load(&self, dataset: AdvisoryDataset, now: u64) -> Result<AdvisoryCacheState> {
            let _guard = self.operations.lock().await;
            if self.fs.metadata(&self.root).await?.is_none() {
                return Ok(AdvisoryCacheState {
                    revision: None,
                    snapshot: None,
                    problem: None,
                });
            }
            let root_inode = self.ordinary_root().await?;
            let native_guard = self.native_guard(root_inode)?;
            let Some(bytes) = self.retained_bytes(dataset, root_inode).await? else {
                return Ok(AdvisoryCacheState {
                    revision: None,
                    snapshot: None,
                    problem: None,
                });
            };
            let revision = Some(DependencyIdentity(Sha256::digest(&bytes).into()));
            let snapshot = serde_json::from_slice::<AdvisorySnapshot>(&bytes)
                .map_err(|_| anyhow::anyhow!("Invalid advisory cache data"))
                .and_then(|snapshot| {
                    snapshot.validate(&self.host)?;
                    Ok(snapshot)
                });
            match snapshot {
                Ok(snapshot) if snapshot.freshness(now) != AdvisoryFreshness::Expired => {
                    Ok(AdvisoryCacheState {
                        revision,
                        snapshot: Some(snapshot),
                        problem: None,
                    })
                }
                Ok(_) => {
                    self.ensure_root(root_inode).await?;
                    self.verify_guard(root_inode, &native_guard)?;
                    self.fs
                        .remove_file(
                            &self.root.join(dataset.filename()),
                            fs::RemoveOptions::default(),
                        )
                        .await?;
                    Ok(AdvisoryCacheState {
                        revision: None,
                        snapshot: None,
                        problem: Some(DiagnosticText::new(
                            "Advisory cache expired after thirty days; no current scan evidence",
                        )),
                    })
                }
                Err(error) => Ok(AdvisoryCacheState {
                    revision,
                    snapshot: None,
                    problem: Some(DiagnosticText::new(error.to_string())),
                }),
            }
        }

        pub async fn replace(
            &self,
            snapshot: AdvisorySnapshot,
            expected: Option<&DependencyIdentity>,
            now: u64,
        ) -> Result<DependencyIdentity> {
            self.replace_when_authorized(snapshot, expected, now, || Ok(()))
                .await
        }

        pub(crate) async fn replace_when_authorized(
            &self,
            snapshot: AdvisorySnapshot,
            expected: Option<&DependencyIdentity>,
            now: u64,
            still_authorized: impl Fn() -> Result<()>,
        ) -> Result<DependencyIdentity> {
            still_authorized()?;
            snapshot.validate(&self.host)?;
            ensure!(
                snapshot.freshness(now) == AdvisoryFreshness::Fresh,
                "Advisory refresh timestamp is stale or uncertain"
            );
            let serialized = serde_json::to_string(&snapshot)?;
            ensure!(
                serialized.len() <= self.maximum_bytes,
                "Advisory cache size limit exceeded"
            );
            let _guard = self.operations.lock().await;
            let root_inode = self.ordinary_root().await?;
            let native_guard = self.native_guard(root_inode)?;
            let current = self.retained_bytes(snapshot.dataset, root_inode).await?;
            let current_revision = current
                .as_ref()
                .map(|bytes| DependencyIdentity(Sha256::digest(bytes).into()));
            ensure!(
                current_revision.as_ref() == expected,
                "Advisory cache changed after review"
            );
            let mut total = serialized.len() as u64;
            let mut entries = self.fs.read_dir(&self.root).await?;
            use futures::StreamExt as _;
            while let Some(path) = entries.next().await.transpose()? {
                let metadata = self
                    .fs
                    .metadata(&path)
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("Advisory cache entry changed"))?;
                if path == self.root.join(CACHE_LOCK_FILENAME) {
                    ensure!(
                        !metadata.is_symlink
                            && !metadata.is_dir
                            && !metadata.is_fifo
                            && metadata.len == 0,
                        "Unsafe public advisory cache lock entry"
                    );
                    continue;
                }
                ensure!(
                    !metadata.is_symlink
                        && !metadata.is_dir
                        && !metadata.is_fifo
                        && [
                            AdvisoryDataset::RustSec.filename(),
                            AdvisoryDataset::OsvMaven.filename()
                        ]
                        .iter()
                        .any(|name| path == self.root.join(name)),
                    "Unexpected advisory cache entry; preserved without modification"
                );
                if path != self.root.join(snapshot.dataset.filename()) {
                    total = total
                        .checked_add(metadata.len)
                        .ok_or_else(|| anyhow::anyhow!("Advisory cache size overflow"))?;
                }
            }
            ensure!(
                total <= self.maximum_bytes as u64,
                "Aggregate advisory cache limit exceeded; previous data preserved"
            );
            self.ensure_root(root_inode).await?;
            self.verify_guard(root_inode, &native_guard)?;
            still_authorized()?;
            let revision = DependencyIdentity(Sha256::digest(serialized.as_bytes()).into());
            let expected_length = serialized.len();
            self.fs
                .atomic_write(self.root.join(snapshot.dataset.filename()), serialized)
                .await?;
            self.ensure_root(root_inode).await?;
            self.verify_guard(root_inode, &native_guard)?;
            let retained = self
                .retained_bytes(snapshot.dataset, root_inode)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Advisory cache replacement is unavailable"))?;
            ensure!(
                retained.len() == expected_length
                    && DependencyIdentity(Sha256::digest(retained).into()) == revision,
                "Advisory cache replacement changed"
            );
            Ok(revision)
        }

        async fn ensure_root(&self, inode: u64) -> Result<()> {
            let metadata = self
                .fs
                .metadata(&self.root)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Advisory cache root disappeared"))?;
            ensure!(
                metadata.is_dir
                    && !metadata.is_symlink
                    && metadata.inode == inode
                    && self.fs.canonicalize(&self.root).await? == self.root,
                "Advisory cache root changed"
            );
            Ok(())
        }

        pub(crate) async fn clear_selected_when_authorized(
            &self,
            dataset: AdvisoryDataset,
            expected: Option<&DependencyIdentity>,
            mut still_authorized: impl FnMut() -> Result<()>,
        ) -> Result<bool> {
            still_authorized()?;
            let _guard = self.operations.lock().await;
            still_authorized()?;
            let root = self.fs.metadata(&self.root).await?;
            still_authorized()?;
            if root.is_none() {
                ensure!(expected.is_none(), "Reviewed advisory cache disappeared");
                return Ok(false);
            }
            let inode = self.ordinary_root().await?;
            let native_guard = self.native_guard(inode)?;
            still_authorized()?;
            let bytes = self.retained_bytes(dataset, inode).await?;
            still_authorized()?;
            let revision = bytes
                .as_ref()
                .map(|bytes| DependencyIdentity(Sha256::digest(bytes).into()));
            ensure!(
                revision.as_ref() == expected,
                "Selected advisory cache changed after clear review"
            );
            self.ensure_root(inode).await?;
            self.verify_guard(inode, &native_guard)?;
            still_authorized()?;
            let removed = bytes.is_some();
            let path = self.root.join(dataset.filename());
            if removed {
                self.fs
                    .remove_file(&path, fs::RemoveOptions::default())
                    .await?;
            }
            still_authorized()?;
            self.ensure_root(inode).await?;
            self.verify_guard(inode, &native_guard)?;
            still_authorized()?;
            ensure!(
                self.fs.metadata(&path).await?.is_none(),
                "Selected public cache absence was not observed; reached deletion effects may remain"
            );
            still_authorized()?;
            Ok(removed)
        }

        pub async fn clear(&self) -> Result<()> {
            let _guard = self.operations.lock().await;
            if self.fs.metadata(&self.root).await?.is_none() {
                return Ok(());
            }
            let inode = self.ordinary_root().await?;
            let native_guard = self.native_guard(inode)?;
            for dataset in [AdvisoryDataset::RustSec, AdvisoryDataset::OsvMaven] {
                self.ensure_root(inode).await?;
                self.verify_guard(inode, &native_guard)?;
                let path = self.root.join(dataset.filename());
                if let Some(metadata) = self.fs.metadata(&path).await? {
                    ensure!(
                        !metadata.is_symlink && !metadata.is_dir && !metadata.is_fifo,
                        "Unsafe advisory cache entry; clear blocked"
                    );
                    self.fs
                        .remove_file(&path, fs::RemoveOptions::default())
                        .await?;
                }
            }
            Ok(())
        }
    }

    #[cfg(test)]
    pub(super) fn matching_fixture(
        dataset: AdvisoryDataset,
        records: Vec<AdvisoryRecord>,
        now: u64,
    ) -> AdvisorySnapshot {
        let kinds = match dataset {
            AdvisoryDataset::RustSec => vec![AdvisoryArchiveKind::RustSec],
            AdvisoryDataset::OsvMaven => vec![
                AdvisoryArchiveKind::OsvMaven,
                AdvisoryArchiveKind::OsvWithdrawals,
            ],
        };
        let sources = kinds
            .into_iter()
            .map(|kind| {
                let revision = if kind == AdvisoryArchiveKind::RustSec {
                    "a".repeat(40)
                } else {
                    "generation:123".into()
                };
                AdvisoryArchiveReceipt {
                    source_url: kind.source_url(&revision),
                    kind,
                    revision,
                    sha256: "a".repeat(64),
                    downloaded_bytes: 1,
                    expanded_bytes: records
                        .iter()
                        .filter(|record| record.kind == kind)
                        .map(|record| record.document.len() as u64)
                        .sum(),
                    records: records.iter().filter(|record| record.kind == kind).count(),
                    records_sha256: records_digest(kind, &records),
                    source_modified_at: Some(now),
                }
            })
            .collect();
        AdvisorySnapshot {
            format: 1,
            dataset,
            host: "matching-fixture-host".into(),
            fetched_at: now,
            sources,
            records,
        }
    }

    #[cfg(test)]
    pub(super) fn matching_fixture_for_cache(
        cache: &AdvisoryCache,
        dataset: AdvisoryDataset,
        records: Vec<AdvisoryRecord>,
        now: u64,
    ) -> AdvisorySnapshot {
        let mut snapshot = matching_fixture(dataset, records, now);
        snapshot.host = cache.host.clone();
        snapshot
    }

    fn records_digest(kind: AdvisoryArchiveKind, records: &[AdvisoryRecord]) -> String {
        let mut digest = Sha256::new();
        for record in records.iter().filter(|record| record.kind == kind) {
            for value in [record.identifier.as_str(), record.document.as_str()] {
                digest.update((value.len() as u64).to_le_bytes());
                digest.update(value.as_bytes());
            }
        }
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn valid_digest(value: &str) -> bool {
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
    }

    fn validate_revision(kind: AdvisoryArchiveKind, revision: &str) -> Result<()> {
        ensure!(
            !revision.is_empty()
                && revision.len() <= 256
                && revision
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"-_:.\"".contains(&byte)),
            "Invalid advisory source revision"
        );
        if kind == AdvisoryArchiveKind::RustSec {
            ensure!(
                revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()),
                "RustSec revision must be an exact commit"
            );
        }
        Ok(())
    }

    fn validate_document(
        kind: AdvisoryArchiveKind,
        identifier: &str,
        document: &str,
    ) -> Result<()> {
        ensure!(
            !identifier.is_empty()
                && identifier.len() <= 128
                && (identifier
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
                    || (kind != AdvisoryArchiveKind::RustSec
                        && qualified_osv_identifier(identifier))),
            "Invalid advisory identifier"
        );
        ensure!(
            document.len() <= MAX_RECORD_BYTES,
            "Advisory record exceeds its limit"
        );
        match kind {
            AdvisoryArchiveKind::RustSec => {
                #[cfg(feature = "rust-tools")]
                {
                    let header = document
                        .strip_prefix("```toml\n")
                        .and_then(|value| value.split_once("\n```"))
                        .map(|(header, _)| header)
                        .ok_or_else(|| anyhow::anyhow!("Unsupported RustSec document format"))?;
                    let parsed = header
                        .parse::<toml::Value>()
                        .map_err(|_| anyhow::anyhow!("Invalid RustSec advisory metadata"))?;
                    ensure!(
                        parsed
                            .get("advisory")
                            .and_then(|value| value.get("id"))
                            .and_then(toml::Value::as_str)
                            == Some(identifier)
                            && parsed
                                .get("advisory")
                                .and_then(|value| value.get("package"))
                                .and_then(toml::Value::as_str)
                                .is_some_and(|value| !value.is_empty() && value.len() <= 128)
                            && parsed
                                .get("versions")
                                .and_then(|value| value.get("patched"))
                                .is_some_and(toml::Value::is_array),
                        "Incomplete RustSec advisory metadata"
                    );
                }
                #[cfg(not(feature = "rust-tools"))]
                anyhow::bail!("RustSec interpretation requires rust-tools");
            }
            AdvisoryArchiveKind::OsvMaven | AdvisoryArchiveKind::OsvWithdrawals => {
                #[cfg(feature = "jvm-tools")]
                {
                    let parsed: serde_json::Value = serde_json::from_str(document)
                        .map_err(|_| anyhow::anyhow!("Invalid OSV advisory data"))?;
                    ensure!(
                        parsed.get("id").and_then(serde_json::Value::as_str) == Some(identifier),
                        "OSV advisory identifier mismatch"
                    );
                    let modified = parsed
                        .get("modified")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| anyhow::anyhow!("OSV modification time is missing"))?;
                    time::OffsetDateTime::parse(
                        modified,
                        &time::format_description::well_known::Rfc3339,
                    )
                    .map_err(|_| anyhow::anyhow!("Invalid OSV modification time"))?;
                    if let Some(withdrawn) = parsed.get("withdrawn") {
                        let withdrawn = withdrawn
                            .as_str()
                            .ok_or_else(|| anyhow::anyhow!("Invalid OSV withdrawal time"))?;
                        time::OffsetDateTime::parse(
                            withdrawn,
                            &time::format_description::well_known::Rfc3339,
                        )
                        .map_err(|_| anyhow::anyhow!("Invalid OSV withdrawal time"))?;
                    }
                    if kind == AdvisoryArchiveKind::OsvMaven {
                        ensure!(
                            parsed
                                .get("affected")
                                .and_then(serde_json::Value::as_array)
                                .is_some_and(|affected| affected.iter().any(|value| value
                                    .get("package")
                                    .and_then(|value| value.get("ecosystem"))
                                    .and_then(serde_json::Value::as_str)
                                    == Some("Maven")))
                                || parsed.get("withdrawn").is_some(),
                            "OSV Maven dataset has no matching ecosystem or withdrawal"
                        );
                    }
                }
                #[cfg(not(feature = "jvm-tools"))]
                anyhow::bail!("OSV Maven interpretation requires jvm-tools");
            }
        }
        Ok(())
    }

    fn qualified_osv_identifier(value: &str) -> bool {
        value.split_once(':').is_some_and(|(prefix, suffix)| {
            prefix.len() > 1
                && !suffix.is_empty()
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"-_.:".contains(&byte))
        })
    }

    fn validate_archive_path(name: &str) -> Result<()> {
        ensure!(
            !name.is_empty()
                && name.len() <= 512
                && !name.starts_with('/')
                && !name.contains(['\\', ':'])
                && name
                    .trim_end_matches('/')
                    .split('/')
                    .all(|part| !part.is_empty()
                        && part != "."
                        && part != ".."
                        && !part.chars().any(char::is_control)),
            "Unsafe advisory archive entry path"
        );
        Ok(())
    }

    fn validate_zip_directory(bytes: &[u8]) -> Result<()> {
        // async_zip allocates from the directory count before reading entries.
        let start = bytes.len().saturating_sub(65557);
        let position = (start..bytes.len().saturating_sub(21))
            .rev()
            .find(|position| {
                bytes.get(*position..position.saturating_add(4)) == Some(b"PK\x05\x06")
                    && bytes
                        .get(position.saturating_add(20)..position.saturating_add(22))
                        .and_then(|value| value.try_into().ok())
                        .is_some_and(|value| {
                            position
                                .saturating_add(22)
                                .saturating_add(u16::from_le_bytes(value) as usize)
                                == bytes.len()
                        })
            })
            .ok_or_else(|| anyhow::anyhow!("Invalid or unsupported advisory ZIP directory"))?;
        let field = |offset| -> Result<u16> {
            Ok(u16::from_le_bytes(
                bytes
                    .get(position + offset..position + offset + 2)
                    .ok_or_else(|| anyhow::anyhow!("Incomplete advisory ZIP directory"))?
                    .try_into()?,
            ))
        };
        let entries = field(10)?;
        let directory_field = |offset| -> Result<u32> {
            Ok(u32::from_le_bytes(
                bytes
                    .get(position + offset..position + offset + 4)
                    .ok_or_else(|| anyhow::anyhow!("Incomplete advisory ZIP directory"))?
                    .try_into()?,
            ))
        };
        let size = directory_field(12)?;
        let offset = directory_field(16)?;
        ensure!(
            size != u32::MAX
                && offset != u32::MAX
                && u64::from(size) + u64::from(offset) <= position as u64
                && position
                    .checked_sub(20)
                    .and_then(|offset| bytes.get(offset..offset + 4))
                    != Some(b"PK\x06\x07"),
            "ZIP64 or out-of-bounds advisory directory is unsupported"
        );
        ensure!(
            field(4)? == 0
                && field(6)? == 0
                && field(8)? == entries
                && usize::from(entries) <= MAX_ARCHIVE_ENTRIES,
            "Unsupported multi-volume or ZIP64 advisory archive"
        );
        Ok(())
    }

    async fn read_archive(
        archive: AdvisoryArchive,
    ) -> Result<(AdvisoryArchiveReceipt, Vec<AdvisoryRecord>)> {
        ensure!(
            archive.bytes.len() <= MAX_ADVISORY_ARCHIVE_BYTES,
            "Advisory archive download limit exceeded"
        );
        validate_zip_directory(&archive.bytes)?;
        let digest = Sha256::digest(&archive.bytes);
        let mut reader = async_zip::base::read::seek::ZipFileReader::new(BufReader::new(
            Cursor::new(archive.bytes.as_slice()),
        ))
        .await
        .map_err(|_| anyhow::anyhow!("Invalid advisory ZIP archive"))?;
        ensure!(
            reader.file().entries().len() <= MAX_ARCHIVE_ENTRIES,
            "Advisory archive entry limit exceeded"
        );
        let mut paths = BTreeSet::new();
        let mut records = Vec::new();
        let mut expansion = 0u64;
        let mut rustsec_prefix = None;
        for index in 0..reader.file().entries().len() {
            let entry = reader
                .file()
                .entries()
                .get(index)
                .ok_or_else(|| anyhow::anyhow!("Advisory ZIP entry is unavailable"))?;
            let name = entry
                .filename()
                .as_str()
                .map_err(|_| anyhow::anyhow!("Invalid advisory ZIP filename"))?
                .to_owned();
            if archive.kind != AdvisoryArchiveKind::RustSec
                && name
                    .strip_suffix(".json")
                    .is_some_and(qualified_osv_identifier)
            {
                // Qualified OSV IDs are logical keys; no entry name reaches Fs.
                ensure!(
                    !name.contains(['/', '\\']),
                    "OSV logical key contains a path separator"
                );
            } else {
                validate_archive_path(&name)?;
            }
            ensure!(
                name.bytes().all(|byte| byte.is_ascii()),
                "Unsupported non-ASCII advisory archive path"
            );
            ensure!(
                paths.insert(name.trim_end_matches('/').to_owned()),
                "Duplicate advisory archive path"
            );
            let directory = entry
                .dir()
                .map_err(|_| anyhow::anyhow!("Invalid advisory ZIP entry"))?;
            let kind = entry
                .unix_permissions()
                .map(|mode| mode & 0o170000)
                .unwrap_or(0);
            ensure!(
                kind == 0 || kind == 0o100000 || (directory && kind == 0o040000),
                "Advisory archive link or special file is forbidden"
            );
            ensure!(
                entry.uncompressed_size() <= MAX_RECORD_BYTES as u64,
                "Advisory archive entry exceeds its limit"
            );
            expansion = expansion
                .checked_add(entry.uncompressed_size())
                .ok_or_else(|| anyhow::anyhow!("Advisory expansion overflow"))?;
            ensure!(
                expansion <= MAX_ADVISORY_CACHE_BYTES as u64,
                "Advisory expansion limit exceeded"
            );
            let expected_size = entry.uncompressed_size();
            let expected_crc = entry.crc32();
            let mut content = Vec::new();
            let mut entry_reader = reader
                .reader_with_entry(index)
                .await
                .map_err(|_| anyhow::anyhow!("Invalid advisory ZIP entry"))?;
            (&mut entry_reader)
                .take(MAX_RECORD_BYTES as u64 + 1)
                .read_to_end(&mut content)
                .await
                .map_err(|_| anyhow::anyhow!("Advisory decompression failed"))?;
            ensure!(
                content.len() as u64 == expected_size
                    && content.len() <= MAX_RECORD_BYTES
                    && entry_reader.compute_hash() == expected_crc,
                "Advisory entry size or checksum mismatch"
            );
            if directory {
                ensure!(content.is_empty(), "Advisory directory contains data");
                continue;
            }
            let identifier = match archive.kind {
                AdvisoryArchiveKind::RustSec => {
                    let (prefix, relative) = name
                        .split_once('/')
                        .ok_or_else(|| anyhow::anyhow!("RustSec archive root is missing"))?;
                    if let Some(expected) = &rustsec_prefix {
                        ensure!(expected == prefix, "RustSec archive has multiple roots");
                    } else {
                        rustsec_prefix = Some(prefix.to_owned());
                    }
                    let Some(relative) = relative.strip_prefix("crates/") else {
                        continue;
                    };
                    let Some((package, filename)) = relative.split_once('/') else {
                        continue;
                    };
                    if !filename.starts_with("RUSTSEC-") || !filename.ends_with(".md") {
                        continue;
                    }
                    ensure!(
                        !filename.contains('/') && !package.is_empty(),
                        "Invalid RustSec advisory path"
                    );
                    filename.trim_end_matches(".md").to_owned()
                }
                AdvisoryArchiveKind::OsvMaven | AdvisoryArchiveKind::OsvWithdrawals => {
                    ensure!(
                        !name.contains('/') && name.ends_with(".json"),
                        "Unexpected OSV archive entry"
                    );
                    name.trim_end_matches(".json").to_owned()
                }
            };
            let document = String::from_utf8(content)
                .map_err(|_| anyhow::anyhow!("Advisory record is not UTF-8"))?;
            validate_document(archive.kind, &identifier, &document)?;
            #[cfg(feature = "rust-tools")]
            if archive.kind == AdvisoryArchiveKind::RustSec {
                let package = name
                    .split('/')
                    .nth(2)
                    .ok_or_else(|| anyhow::anyhow!("RustSec package path is missing"))?;
                let header = document
                    .strip_prefix("```toml\n")
                    .and_then(|value| value.split_once("\n```"))
                    .map(|(header, _)| header)
                    .ok_or_else(|| anyhow::anyhow!("Unsupported RustSec document format"))?;
                let parsed = header
                    .parse::<toml::Value>()
                    .map_err(|_| anyhow::anyhow!("Invalid RustSec advisory metadata"))?;
                ensure!(
                    parsed
                        .get("advisory")
                        .and_then(|value| value.get("package"))
                        .and_then(toml::Value::as_str)
                        == Some(package),
                    "RustSec archive package identity mismatch"
                );
            }
            records.push(AdvisoryRecord {
                kind: archive.kind,
                identifier,
                document,
            });
        }
        let receipt = AdvisoryArchiveReceipt {
            kind: archive.kind,
            source_url: archive.kind.source_url(&archive.revision),
            revision: archive.revision,
            sha256: digest.iter().map(|byte| format!("{byte:02x}")).collect(),
            downloaded_bytes: archive.bytes.len() as u64,
            expanded_bytes: expansion,
            records: records.len(),
            records_sha256: records_digest(archive.kind, &records),
            source_modified_at: None,
        };
        Ok((receipt, records))
    }

    #[cfg(all(test, feature = "rust-tools", feature = "jvm-tools"))]
    mod tests {
        use super::*;
        use fs::Fs as _;
        use gpui::TestAppContext;

        async fn archive(
            kind: AdvisoryArchiveKind,
            entries: &[(&str, &str, u16)],
        ) -> AdvisoryArchive {
            let mut writer = async_zip::base::write::ZipFileWriter::new(Vec::new());
            for (name, document, permissions) in entries {
                let builder = async_zip::ZipEntryBuilder::new(
                    (*name).into(),
                    async_zip::Compression::Deflate,
                )
                .attribute_compatibility(async_zip::AttributeCompatibility::Unix)
                .unix_permissions(*permissions);
                writer
                    .write_entry_whole(builder, document.as_bytes())
                    .await
                    .expect("ZIP fixture entry");
            }
            AdvisoryArchive {
                kind,
                revision: if kind == AdvisoryArchiveKind::RustSec {
                    "a".repeat(40)
                } else {
                    "generation-123".into()
                },
                bytes: writer.close().await.expect("ZIP fixture"),
            }
        }

        const RUSTSEC: &str = "```toml\n[advisory]\nid = \"RUSTSEC-2026-0001\"\npackage = \"example\"\ndate = \"2026-10-01\"\n[versions]\npatched = [\">=1.2\"]\n```\n# Example advisory\nPublic advisory details";
        const MAVEN: &str = r#"{"schema_version":"1.9.1","id":"GHSA-aaaa-bbbb-cccc","modified":"2026-10-01T00:00:00Z","affected":[{"package":{"ecosystem":"Maven","name":"org.example:library"},"versions":["1.0"]}]}"#;
        const WITHDRAWN: &str = r#"{"id":"GHSA-dddd-eeee-ffff","modified":"2026-10-01T00:00:00Z","withdrawn":"2026-10-01T00:00:00Z"}"#;

        fn cache(filesystem: Arc<fs::FakeFs>, root: &str, host: &str) -> AdvisoryCache {
            AdvisoryCache::new(
                filesystem,
                root.into(),
                &DependencyIdentity::from_fields(&[host]).expect("Host fixture"),
            )
            .expect("Cache fixture")
        }

        async fn rust_snapshot(cache: &AdvisoryCache, now: u64) -> AdvisorySnapshot {
            cache
                .validate_archives(
                    AdvisoryDataset::RustSec,
                    vec![
                        archive(
                            AdvisoryArchiveKind::RustSec,
                            &[(
                                "advisory-db-fixture/crates/example/RUSTSEC-2026-0001.md",
                                RUSTSEC,
                                0o100644,
                            )],
                        )
                        .await,
                    ],
                    now,
                )
                .await
                .expect("Validated RustSec fixture")
        }

        async fn maven_snapshot(cache: &AdvisoryCache, now: u64) -> AdvisorySnapshot {
            cache
                .validate_archives(
                    AdvisoryDataset::OsvMaven,
                    vec![
                        archive(
                            AdvisoryArchiveKind::OsvMaven,
                            &[("GHSA-aaaa-bbbb-cccc.json", MAVEN, 0o100644)],
                        )
                        .await,
                        archive(
                            AdvisoryArchiveKind::OsvWithdrawals,
                            &[("GHSA-dddd-eeee-ffff.json", WITHDRAWN, 0o100644)],
                        )
                        .await,
                    ],
                    now,
                )
                .await
                .expect("Validated Maven plus withdrawals")
        }

        #[gpui::test]
        async fn doctor_advisory_clear_selected_revision_authority_partial_and_foreign_preservation(
            cx: &mut TestAppContext,
        ) {
            for scenario in ["success", "stale", "before", "after", "symlink"] {
                let filesystem = fs::FakeFs::new(cx.background_executor.clone());
                let owner = cache(filesystem.clone(), "/cache/doctor", "selected-host");
                let revision = owner
                    .replace(rust_snapshot(&owner, 100).await, None, 100)
                    .await
                    .expect("Rust retained data");
                let other_revision = owner
                    .replace(maven_snapshot(&owner, 100).await, None, 100)
                    .await
                    .expect("Other dataset");
                filesystem
                    .insert_file(
                        Path::new("/cache/doctor/foreign.txt"),
                        "preserved unrelated data".into(),
                    )
                    .await;
                if scenario == "symlink" {
                    filesystem
                        .remove_file(
                            Path::new("/cache/doctor/rustsec-v1.json"),
                            fs::RemoveOptions::default(),
                        )
                        .await
                        .expect("Replace fixture file");
                    filesystem
                        .insert_symlink(
                            Path::new("/cache/doctor/rustsec-v1.json"),
                            Path::new("/cache/doctor/foreign.txt").to_path_buf(),
                        )
                        .await;
                }
                let mut checks = 0;
                let result = owner
                    .clear_selected_when_authorized(
                        AdvisoryDataset::RustSec,
                        if scenario == "stale" {
                            Some(&other_revision)
                        } else {
                            Some(&revision)
                        },
                        || {
                            checks += 1;
                            ensure!(
                                !(scenario == "before" || scenario == "after" && checks == 7),
                                "Explicit revocation fixture; completed effects may remain"
                            );
                            Ok(())
                        },
                    )
                    .await;
                assert_eq!(result.is_ok(), scenario == "success");
                assert_eq!(
                    filesystem
                        .metadata(Path::new("/cache/doctor/rustsec-v1.json"))
                        .await
                        .expect("Selected state")
                        .is_none(),
                    matches!(scenario, "success" | "after")
                );
                assert_eq!(
                    owner
                        .load(AdvisoryDataset::OsvMaven, 100)
                        .await
                        .expect("Preserved other dataset")
                        .revision,
                    Some(other_revision)
                );
                assert_eq!(
                    filesystem
                        .load(Path::new("/cache/doctor/foreign.txt"))
                        .await
                        .expect("Foreign file"),
                    "preserved unrelated data"
                );
            }
        }

        #[gpui::test]
        async fn doctor_advisory_cache_reopen_freshness_retention_and_clear(
            cx: &mut TestAppContext,
        ) {
            let filesystem = fs::FakeFs::new(cx.background_executor.clone());
            let owner = cache(filesystem.clone(), "/cache/doctor", "host-a");
            let missing = owner
                .load(AdvisoryDataset::RustSec, 100)
                .await
                .expect("Missing cache");
            assert!(missing.snapshot.is_none() && missing.problem.is_none());
            assert!(
                filesystem
                    .metadata(Path::new("/cache"))
                    .await
                    .expect("No passive writes")
                    .is_none()
            );
            let snapshot = rust_snapshot(&owner, 100).await;
            let revision = owner
                .replace(snapshot, None, 100)
                .await
                .expect("Approved replacement");
            let reopened = cache(filesystem.clone(), "/cache/doctor", "host-a");
            let retained = reopened
                .load(AdvisoryDataset::RustSec, 100)
                .await
                .expect("Reopen");
            assert!(retained.revision == Some(revision));
            let retained = retained.snapshot.expect("Offline retained snapshot");
            assert_eq!(retained.freshness(100), AdvisoryFreshness::Fresh);
            assert_eq!(
                retained.freshness(100 + FRESH_SECONDS - 1),
                AdvisoryFreshness::Fresh
            );
            assert_eq!(
                retained.freshness(100 + FRESH_SECONDS),
                AdvisoryFreshness::Stale
            );
            assert_eq!(retained.freshness(99), AdvisoryFreshness::ClockUncertain);
            assert_eq!(
                retained.freshness(100 + RETENTION_SECONDS),
                AdvisoryFreshness::Expired
            );
            let expired = reopened
                .load(AdvisoryDataset::RustSec, 100 + RETENTION_SECONDS)
                .await
                .expect("Retention expiry");
            assert!(
                expired.snapshot.is_none()
                    && expired.problem.is_some()
                    && expired.revision.is_none()
            );
            reopened
                .replace(maven_snapshot(&reopened, 200).await, None, 200)
                .await
                .expect("Maven data");
            filesystem.insert_tree("/unrelated", serde_json::json!({"source.rs":"preserved edits", "credential":"outside cache"})).await;
            reopened.clear().await.expect("Explicit clear");
            assert!(
                reopened
                    .load(AdvisoryDataset::OsvMaven, 200)
                    .await
                    .expect("Cleared")
                    .snapshot
                    .is_none()
            );
            assert_eq!(
                filesystem
                    .load(Path::new("/unrelated/source.rs"))
                    .await
                    .expect("Unrelated source"),
                "preserved edits"
            );
            assert_eq!(
                filesystem
                    .load(Path::new("/unrelated/credential"))
                    .await
                    .expect("Unrelated credential"),
                "outside cache"
            );
        }

        #[gpui::test]
        async fn doctor_advisory_cache_failed_stale_and_duplicate_refresh_preserve_previous(
            cx: &mut TestAppContext,
        ) {
            let filesystem = fs::FakeFs::new(cx.background_executor.clone());
            let owner = cache(filesystem, "/cache/doctor", "host-a");
            let previous = owner
                .replace(rust_snapshot(&owner, 100).await, None, 100)
                .await
                .expect("Previous snapshot");
            let mut invalid = rust_snapshot(&owner, 200).await;
            invalid.sources.first_mut().expect("Receipt").source_url =
                "https://unapproved.example/dataset".into();
            assert!(owner.replace(invalid, Some(&previous), 200).await.is_err());
            let snapshot = rust_snapshot(&owner, 200).await;
            assert!(owner.replace(snapshot.clone(), None, 200).await.is_err());
            assert!(
                owner
                    .replace(snapshot.clone(), Some(&previous), 199)
                    .await
                    .is_err()
            );
            assert!(
                owner
                    .replace(snapshot.clone(), Some(&previous), 200 + FRESH_SECONDS)
                    .await
                    .is_err()
            );
            assert!(
                owner
                    .load(AdvisoryDataset::RustSec, 200)
                    .await
                    .expect("Preserved")
                    .revision
                    == Some(previous.clone())
            );
            let next = owner
                .replace(snapshot, Some(&previous), 200)
                .await
                .expect("Reviewed current replacement");
            assert!(next != previous);
            assert!(
                owner
                    .replace(rust_snapshot(&owner, 300).await, Some(&previous), 300)
                    .await
                    .is_err()
            );
            assert!(
                owner
                    .load(AdvisoryDataset::RustSec, 300)
                    .await
                    .expect("Latest remains")
                    .revision
                    == Some(next)
            );
        }

        #[gpui::test]
        async fn doctor_advisory_cache_host_separation_corruption_and_withdrawal_coverage(
            cx: &mut TestAppContext,
        ) {
            let filesystem = fs::FakeFs::new(cx.background_executor.clone());
            let owner = cache(filesystem.clone(), "/cache/doctor", "host-a");
            let snapshot = maven_snapshot(&owner, 100).await;
            assert_eq!(snapshot.sources.len(), 2);
            assert!(snapshot.sources.iter().all(|receipt| receipt.records == 1));
            assert!(
                snapshot
                    .records
                    .iter()
                    .any(|record| record.kind == AdvisoryArchiveKind::OsvWithdrawals)
            );
            let mut incomplete = snapshot.clone();
            incomplete.sources.pop();
            assert!(owner.replace(incomplete, None, 100).await.is_err());
            owner
                .replace(snapshot, None, 100)
                .await
                .expect("Complete cache");
            let foreign = cache(filesystem.clone(), "/cache/doctor", "host-b")
                .load(AdvisoryDataset::OsvMaven, 100)
                .await
                .expect("Host mismatch status");
            assert!(
                foreign.snapshot.is_none()
                    && foreign.problem.is_some()
                    && foreign.revision.is_some()
            );
            filesystem
                .atomic_write(
                    PathBuf::from("/cache/doctor/osv-maven-v1.json"),
                    "{invalid SOURCE_CONTENT_NEVER_IN_ERROR".into(),
                )
                .await
                .expect("Corruption fixture");
            let corrupt = owner
                .load(AdvisoryDataset::OsvMaven, 100)
                .await
                .expect("Corrupt cache status");
            assert!(corrupt.snapshot.is_none() && corrupt.revision.is_some());
            assert!(
                !corrupt
                    .problem
                    .expect("Bounded generic error")
                    .as_str()
                    .contains("SOURCE_CONTENT_NEVER_IN_ERROR")
            );
            owner
                .replace(
                    maven_snapshot(&owner, 200).await,
                    corrupt.revision.as_ref(),
                    200,
                )
                .await
                .expect("Explicit reviewed corruption replacement");
        }

        #[gpui::test]
        async fn doctor_advisory_cache_aggregate_limit_and_foreign_entries_are_preserved(
            cx: &mut TestAppContext,
        ) {
            let filesystem = fs::FakeFs::new(cx.background_executor.clone());
            let mut owner = cache(filesystem.clone(), "/cache/doctor", "host-a");
            let rust = rust_snapshot(&owner, 100).await;
            let maven = maven_snapshot(&owner, 100).await;
            owner.maximum_bytes = serde_json::to_vec(&rust)
                .expect("Size")
                .len()
                .max(serde_json::to_vec(&maven).expect("Size").len());
            let retained = owner
                .replace(rust, None, 100)
                .await
                .expect("Within single-source limit");
            assert!(owner.replace(maven, None, 100).await.is_err());
            assert!(
                owner
                    .load(AdvisoryDataset::RustSec, 100)
                    .await
                    .expect("Previous remains")
                    .revision
                    == Some(retained.clone())
            );
            filesystem
                .insert_tree(
                    "/cache/doctor",
                    serde_json::json!({"foreign-source.txt":"preserved"}),
                )
                .await;
            assert!(
                owner
                    .replace(rust_snapshot(&owner, 200).await, Some(&retained), 200)
                    .await
                    .is_err()
            );
            owner.clear().await.expect("Clear only own data");
            assert_eq!(
                filesystem
                    .load(Path::new("/cache/doctor/foreign-source.txt"))
                    .await
                    .expect("Foreign entry"),
                "preserved"
            );
        }

        #[gpui::test]
        async fn doctor_advisory_cache_symlink_containment_and_special_paths(
            cx: &mut TestAppContext,
        ) {
            let filesystem = fs::FakeFs::new(cx.background_executor.clone());
            filesystem
                .insert_tree("/outside", serde_json::json!({"data":"unrelated"}))
                .await;
            filesystem
                .create_symlink(Path::new("/linked"), PathBuf::from("/outside"))
                .await
                .expect("Root symlink fixture");
            let owner = cache(filesystem.clone(), "/linked/doctor", "host-a");
            assert!(
                owner
                    .replace(rust_snapshot(&owner, 100).await, None, 100)
                    .await
                    .is_err()
            );
            assert!(
                filesystem
                    .metadata(Path::new("/outside/doctor"))
                    .await
                    .expect("No escape")
                    .is_none()
            );
            let owner = cache(filesystem.clone(), "/cache/doctor", "host-a");
            owner
                .replace(rust_snapshot(&owner, 100).await, None, 100)
                .await
                .expect("Ordinary root");
            filesystem
                .remove_file(
                    Path::new("/cache/doctor/rustsec-v1.json"),
                    fs::RemoveOptions::default(),
                )
                .await
                .expect("Fixture replacement");
            filesystem
                .create_symlink(
                    Path::new("/cache/doctor/rustsec-v1.json"),
                    PathBuf::from("/outside/data"),
                )
                .await
                .expect("File symlink");
            assert!(owner.load(AdvisoryDataset::RustSec, 100).await.is_err());
            assert!(owner.clear().await.is_err());
            assert_eq!(
                filesystem
                    .load(Path::new("/outside/data"))
                    .await
                    .expect("Unrelated retained"),
                "unrelated"
            );
            assert!(
                AdvisoryCache::new(
                    filesystem.clone(),
                    "/cache/../outside".into(),
                    &DependencyIdentity::from_fields(&["host-a"]).expect("Host")
                )
                .is_err()
            );
            assert!(
                AdvisoryCache::new(
                    filesystem,
                    "relative-cache".into(),
                    &DependencyIdentity::from_fields(&["host-a"]).expect("Host")
                )
                .is_err()
            );
        }

        #[gpui::test]
        async fn doctor_advisory_cache_archive_paths_links_duplicates_and_expansion(
            cx: &mut TestAppContext,
        ) {
            let filesystem = fs::FakeFs::new(cx.background_executor.clone());
            let owner = cache(filesystem, "/cache/doctor", "host-a");
            for (name, mode) in [
                ("../GHSA-aaaa-bbbb-cccc.json", 0o100644),
                ("/GHSA-aaaa-bbbb-cccc.json", 0o100644),
                ("folder\\GHSA-aaaa-bbbb-cccc.json", 0o100644),
                ("C:GHSA-aaaa-bbbb-cccc.json", 0o100644),
                ("GHSA-aaaa-bbbb-cccc.json", 0o120777),
                ("GHSA-aaaa-bbbb-cccc.json", 0o010644),
            ] {
                assert!(
                    owner
                        .validate_archives(
                            AdvisoryDataset::OsvMaven,
                            vec![
                                archive(AdvisoryArchiveKind::OsvMaven, &[(name, MAVEN, mode)])
                                    .await
                            ],
                            100
                        )
                        .await
                        .is_err()
                );
            }
            let duplicate = archive(
                AdvisoryArchiveKind::OsvMaven,
                &[
                    ("GHSA-aaaa-bbbb-cccc.json", MAVEN, 0o100644),
                    ("GHSA-aaaa-bbbb-cccc.json", MAVEN, 0o100644),
                ],
            )
            .await;
            assert!(
                owner
                    .validate_archives(AdvisoryDataset::OsvMaven, vec![duplicate], 100)
                    .await
                    .is_err()
            );
            let too_large = "x".repeat(MAX_RECORD_BYTES + 1);
            assert!(
                owner
                    .validate_archives(
                        AdvisoryDataset::OsvMaven,
                        vec![
                            archive(
                                AdvisoryArchiveKind::OsvMaven,
                                &[("GHSA-aaaa-bbbb-cccc.json", &too_large, 0o100644)]
                            )
                            .await
                        ],
                        100
                    )
                    .await
                    .is_err()
            );
            let mut corrupt = archive(
                AdvisoryArchiveKind::OsvMaven,
                &[("GHSA-aaaa-bbbb-cccc.json", MAVEN, 0o100644)],
            )
            .await;
            corrupt.bytes.truncate(10);
            assert!(
                owner
                    .validate_archives(AdvisoryDataset::OsvMaven, vec![corrupt], 100)
                    .await
                    .is_err()
            );
            assert!(
                owner
                    .load(AdvisoryDataset::OsvMaven, 100)
                    .await
                    .expect("No failed-import writes")
                    .snapshot
                    .is_none()
            );
        }

        #[cfg(unix)]
        #[gpui::test]
        async fn doctor_advisory_cache_host_native_shared_admission_budget_and_clear(
            cx: &mut TestAppContext,
        ) {
            // RealFs completes native filesystem I/O outside the deterministic dispatcher.
            cx.executor().allow_parking();
            assert_eq!(
                shared_cache_root(Path::new("/native/cache/ide-rust")).expect("Copper cache"),
                shared_cache_root(Path::new("/native/cache/ide-jvm")).expect("Groovy cache")
            );
            assert_eq!(
                shared_cache_root(Path::new("/native/cache/made")).expect("Made cache"),
                PathBuf::from("/native/cache/made-doctor-advisories")
            );
            let directory = tempfile::tempdir().expect("Owned public cache fixture");
            let root = directory
                .path()
                .canonicalize()
                .expect("Ordinary cache root");
            let host = DependencyIdentity::from_fields(&["same-native-host"]).expect("Host");
            let first = AdvisoryCache::new(
                Arc::new(fs::RealFs::new(None, cx.background_executor.clone())),
                root.clone(),
                &host,
            )
            .expect("First product owner");
            let mut second = AdvisoryCache::new(
                Arc::new(fs::RealFs::new(None, cx.background_executor.clone())),
                root.clone(),
                &host,
            )
            .expect("Independent product owner");
            let snapshot = rust_snapshot(&first, 100).await;
            let revision = first
                .replace(snapshot.clone(), None, 100)
                .await
                .expect("First product commit");
            assert_eq!(
                second
                    .load(AdvisoryDataset::RustSec, 100)
                    .await
                    .expect("Other product reopens shared public data")
                    .revision
                    .as_ref(),
                Some(&revision)
            );
            assert!(
                second.replace(snapshot, None, 100).await.is_err(),
                "Cross-owner expected revision blocks stale proposal"
            );
            use std::os::unix::fs::MetadataExt as _;
            let inode = std::fs::metadata(&root).expect("Root identity").ino();
            let guard =
                acquire_native_cache_guard(&root, inode).expect("Independent native operation");
            assert!(second.load(AdvisoryDataset::RustSec, 100).await.is_err());
            assert!(
                second.clear().await.is_err(),
                "Busy cache is not queued or cleared"
            );
            drop(guard);
            let snapshot = maven_snapshot(&second, 100).await;
            second.maximum_bytes = serde_json::to_vec(&snapshot)
                .expect("Serialized Maven fixture")
                .len();
            assert!(
                second.replace(snapshot, None, 100).await.is_err(),
                "One aggregate host budget includes the other product's Rust data"
            );
            assert_eq!(
                first
                    .load(AdvisoryDataset::RustSec, 100)
                    .await
                    .expect("Prior data preserved")
                    .revision
                    .as_ref(),
                Some(&revision)
            );
            second.maximum_bytes = MAX_ADVISORY_CACHE_BYTES;
            std::fs::write(root.join("unrelated"), "preserved").expect("Foreign file");
            second.clear().await.expect("Explicit scoped clear");
            assert!(
                first
                    .load(AdvisoryDataset::RustSec, 100)
                    .await
                    .expect("Shared clear observed")
                    .snapshot
                    .is_none()
            );
            assert_eq!(
                std::fs::read_to_string(root.join("unrelated")).expect("Foreign file retained"),
                "preserved"
            );
            assert!(
                root.join(CACHE_LOCK_FILENAME).is_file(),
                "Keep the stable admission inode across clear"
            );
        }

        #[cfg(unix)]
        #[test]
        fn doctor_advisory_cache_host_native_guard_rejects_foreign_links_and_replacement() {
            use std::os::unix::fs::MetadataExt as _;
            let directory = tempfile::tempdir().expect("Owned guard fixture");
            let root = directory.path().canonicalize().expect("Ordinary root");
            let inode = std::fs::metadata(&root).expect("Root identity").ino();
            let guard = acquire_native_cache_guard(&root, inode).expect("First guard");
            assert!(acquire_native_cache_guard(&root, inode).is_err());
            let path = root.join(CACHE_LOCK_FILENAME);
            std::fs::rename(&path, root.join("former-lock")).expect("Only test fixture mutation");
            assert!(verify_native_cache_guard(&root, inode, &guard).is_err());
            drop(guard);
            let sentinel = root.join("sentinel");
            std::fs::write(&sentinel, "preserved").expect("Foreign sentinel");
            std::os::unix::fs::symlink(&sentinel, &path).expect("Guard alias");
            assert!(acquire_native_cache_guard(&root, inode).is_err());
            assert_eq!(
                std::fs::read_to_string(&sentinel).expect("Alias target retained"),
                "preserved"
            );
            std::fs::remove_file(&path).expect("Remove owned fixture link");
            std::fs::write(&path, "foreign guard contents").expect("Foreign guard");
            assert!(acquire_native_cache_guard(&root, inode).is_err());
            assert_eq!(
                std::fs::read_to_string(&path).expect("Foreign data retained"),
                "foreign guard contents"
            );
        }

        #[gpui::test]
        async fn doctor_advisory_cache_qualified_ids_are_data_never_paths(cx: &mut TestAppContext) {
            let filesystem = fs::FakeFs::new(cx.background_executor.clone());
            let owner = cache(filesystem.clone(), "/cache/doctor", "host-a");
            let document = r#"{"id":"RLSA-2023:2097","modified":"2026-10-01T00:00:00Z","withdrawn":"2026-10-01T00:00:00Z"}"#;
            let snapshot = owner
                .validate_archives(
                    AdvisoryDataset::OsvMaven,
                    vec![
                        archive(
                            AdvisoryArchiveKind::OsvMaven,
                            &[("GHSA-aaaa-bbbb-cccc.json", MAVEN, 0o100644)],
                        )
                        .await,
                        archive(
                            AdvisoryArchiveKind::OsvWithdrawals,
                            &[("RLSA-2023:2097.json", document, 0o100644)],
                        )
                        .await,
                    ],
                    100,
                )
                .await
                .expect("Qualified logical identity");
            assert!(
                snapshot
                    .records
                    .iter()
                    .any(|record| record.identifier == "RLSA-2023:2097")
            );
            owner
                .replace(snapshot, None, 100)
                .await
                .expect("One owned JSON cache file");
            assert!(
                filesystem
                    .metadata(Path::new("/cache/doctor/RLSA-2023:2097.json"))
                    .await
                    .expect("No extraction destination")
                    .is_none()
            );
            let reopened = owner
                .load(AdvisoryDataset::OsvMaven, 100)
                .await
                .expect("Reopen")
                .snapshot
                .expect("Retained snapshot");
            assert!(
                reopened
                    .records
                    .iter()
                    .any(|record| record.identifier == "RLSA-2023:2097")
            );
            for name in [
                "C:GHSA-aaaa-bbbb-cccc.json",
                "RLSA-2023:../outside.json",
                "RLSA-2023:folder\\outside.json",
            ] {
                assert!(
                    owner
                        .validate_archives(
                            AdvisoryDataset::OsvMaven,
                            vec![
                                archive(
                                    AdvisoryArchiveKind::OsvMaven,
                                    &[("GHSA-aaaa-bbbb-cccc.json", MAVEN, 0o100644)]
                                )
                                .await,
                                archive(
                                    AdvisoryArchiveKind::OsvWithdrawals,
                                    &[(name, document, 0o100644)]
                                )
                                .await,
                            ],
                            100
                        )
                        .await
                        .is_err()
                );
            }
        }

        #[gpui::test]
        #[ignore = "Requires explicitly supplied downloaded official public source fixtures"]
        async fn doctor_advisory_cache_qualified_published_datasets(cx: &mut TestAppContext) {
            let directory = std::env::var_os("MADE_DOCTOR_PUBLIC_ADVISORY_DIR")
                .expect("Select the downloaded official public corpus explicitly");
            let directory = PathBuf::from(directory);
            let filesystem = fs::FakeFs::new(cx.background_executor.clone());
            let owner = cache(filesystem, "/cache/doctor", "published-source-fixture");
            let archives = [
                ("maven.zip", AdvisoryArchiveKind::OsvMaven),
                ("withdrawals.zip", AdvisoryArchiveKind::OsvWithdrawals),
            ]
            .into_iter()
            .map(|(filename, kind)| {
                let bytes = std::fs::read(directory.join(filename))
                    .expect("Explicitly selected downloaded public fixture");
                assert!(bytes.len() <= MAX_ADVISORY_ARCHIVE_BYTES);
                AdvisoryArchive {
                    kind,
                    revision: "selected-public-fixture".into(),
                    bytes,
                }
            })
            .collect();
            let snapshot = owner
                .validate_archives(AdvisoryDataset::OsvMaven, archives, 100)
                .await
                .expect("Published public sources accepted without extraction or execution");
            assert!(snapshot.sources.iter().all(|source| source.records > 0));
            assert!(
                snapshot
                    .records
                    .iter()
                    .any(|record| record.identifier.contains(':'))
            );
            println!(
                "Published dataset source records: {:?}",
                snapshot
                    .sources
                    .iter()
                    .map(|source| (source.kind, source.records, source.expanded_bytes))
                    .collect::<Vec<_>>()
            );
        }

        #[test]
        fn doctor_advisory_cache_rejects_zip64_before_directory_allocation_and_invalid_records() {
            let mut footer = b"PK\x05\x06".to_vec();
            footer.extend([0u8; 18]);
            assert!(validate_zip_directory(&footer).is_ok());
            footer
                .get_mut(10..12)
                .expect("Count fixture")
                .copy_from_slice(&u16::MAX.to_le_bytes());
            assert!(validate_zip_directory(&footer).is_err());
            assert!(
                validate_document(AdvisoryArchiveKind::OsvMaven, "GHSA-mismatch", MAVEN).is_err()
            );
            assert!(
                validate_document(
                    AdvisoryArchiveKind::RustSec,
                    "RUSTSEC-2026-0001",
                    "```toml\nSOURCE_CONTENT_SECRET[invalid\n```\n"
                )
                .expect_err("Invalid metadata")
                .to_string()
                .find("SOURCE_CONTENT_SECRET")
                .is_none()
            );
            assert!(validate_revision(AdvisoryArchiveKind::RustSec, "main").is_err());
            assert!(validate_revision(AdvisoryArchiveKind::RustSec, "../main").is_err());
        }
    }
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
pub use advisory_fetch::{
    AdvisoryRefreshAction, AdvisoryRefreshProgress, AdvisoryRefreshRequest, AdvisoryRefreshResult,
    launch_advisory_refresh, prepare_advisory_refresh,
};

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
mod advisory_fetch {
    use super::*;
    use crate::doctor::{
        CheckOutcome, DoctorCheck, DoctorHistoryStore, DoctorProduct, DoctorRepairOperationOutcome,
        DoctorRepairReview, DoctorReport, project_repair_binding,
    };
    use futures::{FutureExt as _, io::AsyncReadExt as _};
    use http_client::{AsyncBody, HttpClient, HttpRequestExt as _, RedirectPolicy, Request};
    use std::{sync::Arc, time::Duration};

    const REFRESH_DEADLINE: Duration = Duration::from_secs(60);
    const MAX_REVISION_BYTES: usize = 16 * 1024;

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub enum AdvisoryRefreshRequest {
        ResolveRustSecRevision,
        RustSec { commit: String },
        OsvMaven,
        RegistryVersions(RegistryQuery),
        Clear { dataset: AdvisoryDataset },
        MatchRetained { dataset: AdvisoryDataset },
    }

    impl AdvisoryRefreshRequest {
        fn validate(&self) -> Result<()> {
            match self {
                Self::ResolveRustSecRevision | Self::RustSec { .. } => {
                    ensure!(
                        cfg!(feature = "rust-tools"),
                        "Rust advisory integration is disabled"
                    );
                    if let Self::RustSec { commit } = self {
                        ensure!(
                            commit.len() == 40
                                && commit.bytes().all(|byte| byte.is_ascii_hexdigit()),
                            "RustSec refresh requires an exact forty-hex commit"
                        );
                    }
                }
                Self::OsvMaven => ensure!(
                    cfg!(feature = "jvm-tools"),
                    "Maven advisory integration is disabled"
                ),
                Self::RegistryVersions(query) => {
                    query.exact_url()?;
                }
                Self::Clear { dataset } | Self::MatchRetained { dataset } => ensure!(
                    match dataset {
                        AdvisoryDataset::RustSec => cfg!(feature = "rust-tools"),
                        AdvisoryDataset::OsvMaven => cfg!(feature = "jvm-tools"),
                    },
                    "Selected advisory integration is disabled"
                ),
            }
            Ok(())
        }

        pub fn exact_requests(&self) -> Result<Vec<String>> {
            self.validate()?;
            Ok(match self {
                Self::Clear { .. } | Self::MatchRetained { .. } => Vec::new(),
                Self::RegistryVersions(query) => vec![query.exact_url()?],
                Self::ResolveRustSecRevision => {
                    vec!["https://api.github.com/repos/RustSec/advisory-db/commits/main".into()]
                }
                Self::RustSec { commit } => vec![AdvisoryArchiveKind::RustSec.source_url(commit)],
                Self::OsvMaven => vec![
                    AdvisoryArchiveKind::OsvMaven.source_url(""),
                    AdvisoryArchiveKind::OsvWithdrawals.source_url(""),
                ],
            })
        }

        fn dataset(&self) -> Option<AdvisoryDataset> {
            match self {
                Self::ResolveRustSecRevision | Self::RegistryVersions(_) => None,
                Self::RustSec { .. } => Some(AdvisoryDataset::RustSec),
                Self::OsvMaven => Some(AdvisoryDataset::OsvMaven),
                Self::Clear { dataset } | Self::MatchRetained { dataset } => Some(*dataset),
            }
        }
    }

    #[derive(Clone)]
    pub struct AdvisoryRefreshAction {
        request: AdvisoryRefreshRequest,
        cache: Arc<AdvisoryCache>,
        expected_revision: Option<DependencyIdentity>,
        source_inputs: Option<(usize, Vec<String>)>,
        #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
        pub(super) projection: Option<Arc<DoctorDependencyProjection>>,
    }

    impl AdvisoryRefreshAction {
        pub fn request(&self) -> &AdvisoryRefreshRequest {
            &self.request
        }

        pub(super) fn revalidate_inputs(
            &self,
            project: &gpui::Entity<project::Project>,
            directory: &project::ProjectPath,
            cx: &gpui::App,
        ) -> Result<()> {
            if matches!(self.request, AdvisoryRefreshRequest::MatchRetained { .. }) {
                ensure!(
                    self.projection.is_some(),
                    "Retained matching requires current selected owner evidence"
                );
            }
            #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
            if let Some(projection) = &self.projection {
                ensure!(
                    projection.binding().directory() == directory,
                    "Selected dependency scope changed"
                );
                projection.revalidate(project, cx)?;
                if let AdvisoryRefreshRequest::RegistryVersions(query) = &self.request {
                    query.revalidate_graph(projection.graph())?;
                }
            }
            if let Some((scan_id, buffers)) = &self.source_inputs {
                ensure!(
                    project
                        .read(cx)
                        .worktree_for_id(directory.worktree_id, cx)
                        .is_some_and(|worktree| worktree.read(cx).scan_id() == *scan_id)
                        && crate::doctor::reproduction_buffer_versions(
                            project,
                            directory.worktree_id,
                            cx
                        ) == *buffers,
                    "Reviewed dependency inputs changed; no registry result may be admitted"
                );
            }
            Ok(())
        }
        pub fn exact_preview(&self) -> Result<Vec<String>> {
            if let AdvisoryRefreshRequest::MatchRetained { dataset } = &self.request {
                let projection = self.projection.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("Current selected dependency evidence is missing")
                })?;
                return Ok(vec![
                    format!("Evaluate selected {dataset:?} retained public advisory data at {:?}; exact cache revision {}; selected graph {} and coverage {:?}", self.cache.root_for_preview().join(dataset.filename()), self.expected_revision.as_ref().map(DependencyIdentity::hex).unwrap_or_else(|| "absent".into()), projection.graph().identity().hex(), projection.graph().provenance().coverage),
                    "Local selected-host evaluation only. No network, dependency fetch/resolution, subprocesses, source/lock/settings/credential changes or permission/scheduling consent. Public cache retention still expires data after thirty days on access.".into(),
                    "Sixty-second total deadline, bounded findings, cooperative cancellation and current Project/graph/cache checks through late admission. Missing/replaced/corrupt data is incomplete; stale/publication-unknown/partial coverage stays uncertain. No findings never proves security or application exploitability. Only redacted bounded summaries and receipts persist; full graphs and advisory bodies remain transient.".into(),
                ]);
            }
            if let AdvisoryRefreshRequest::Clear { dataset } = &self.request {
                return Ok(vec![format!("Clear selected {dataset:?} public advisory data at {:?}; reviewed revision {}", self.cache.root_for_preview().join(dataset.filename()), self.expected_revision.as_ref().map(DependencyIdentity::hex).unwrap_or_else(|| "absent".into())),
                    "Shared selected local-host public cache for Made, Copper and Groovy. No network, dependency fetch, source/history/session/settings/credential deletion or permission changes. Other datasets and foreign files remain preserved.".into(),
                    "Sixty-second deadline; cancellation and current trust/host/settings checks before deletion and recheck. Interrupted deletion can leave the selected dataset absent; no rollback or automatic replay. Recheck proves only observed absence, never a vulnerability fix or project security.".into()]);
            }
            if let AdvisoryRefreshRequest::RegistryVersions(query) = &self.request {
                return Ok(vec![format!("GET {}; disclose only the separately selected public coordinate {:?}; no authorization, cookies, body, redirects or project/source/lockfile upload", query.exact_url()?, query.package().name), "Version metadata only: one-MiB response, sixty-second total deadline, no packages, resolution, cache replacement, repository/default/lock changes or unattended consent. Responses remain transient; redacted operation metadata uses normal Doctor history".into(), format!("Selected graph {}, node {}, source {:?}, resolved version {:?}; catalog-only confirmation {}. Unknown-origin catalog candidates never establish this resolved artifact's repository, a verified update or a security fix. No API/runtime compatibility promise", query.graph_revision().hex(), query.node(), query.package().source.kind(), query.package().version, query.is_catalog_only())]);
            }
            let mut preview = self
                .request
                .exact_requests()?
                .into_iter()
                .map(|url| {
                    format!("GET {url}; no authorization/cookies/project data; redirects rejected")
                })
                .collect::<Vec<_>>();
            preview.push(format!("Public data only; sixty-second total deadline; at most {} bytes per archive (unknown Content-Length uses that cap), {} retained bytes in the selected local application-host cache; no extraction, execution, dependency fetch or background refresh", advisory_cache::MAX_ADVISORY_ARCHIVE_BYTES, advisory_cache::MAX_ADVISORY_CACHE_BYTES));
            preview.push("This is the shared public offline advisory cache for Made, Copper and Groovy on the selected native host. Refresh/replacement and an explicitly reviewed clear affect that shared public data; private settings, history, credentials and project graphs remain separate".into());
            preview.push(format!("Cache destination: {:?}; existing cache revision {}. Stale after twenty-four hours, expire on access after thirty days; explicit clear. Missing/failed/withdrawal-incomplete data cannot establish no findings", self.cache.root_for_preview(), self.expected_revision.as_ref().map(DependencyIdentity::hex).unwrap_or_else(|| "absent".into())));
            if self.request == AdvisoryRefreshRequest::ResolveRustSecRevision {
                preview.push("Read at most sixteen KiB of official repository commit metadata. Resolving a revision does not download an archive: the returned exact commit needs a separately reviewed download action".into());
            }
            Ok(preview)
        }
    }

    #[derive(Clone, Debug)]
    pub struct AdvisoryRefreshProgress {
        pub stage: DiagnosticText,
        pub received_bytes: u64,
    }

    pub struct AdvisoryRefreshResult {
        pub review: DoctorRepairReview<AdvisoryRefreshAction>,
        pub resolved_commit: Option<String>,
        pub snapshot: Option<AdvisorySnapshot>,
        pub registry_metadata: Option<RegistryMetadata>,
        pub cleared_dataset: Option<AdvisoryDataset>,
        pub scan: Option<AdvisoryScan>,
    }

    pub fn prepare_advisory_refresh(
        project: &gpui::Entity<project::Project>,
        directory: project::ProjectPath,
        generation: u64,
        request: AdvisoryRefreshRequest,
        now: u64,
        cx: &mut gpui::App,
    ) -> gpui::Task<Result<DoctorRepairReview<AdvisoryRefreshAction>>> {
        let project = project.clone();
        cx.spawn(async move |cx| {
            request.validate()?;
            let (binding, cache) = cx.update(|cx| {
                ensure!(project.read(cx).is_local(), "Remote advisory refresh is unsupported; no local fallback");
                Ok::<_, anyhow::Error>((project_repair_binding(&project, directory.clone(), generation, cx)?, Arc::new(AdvisoryCache::for_local_project(project.read(cx))?)))
            })?;
            let expected_revision = if let Some(dataset) = request.dataset() { cx.background_executor().spawn({ let cache = cache.clone(); async move { cache.load(dataset, now).await } }).await?.revision } else { None };
            cx.update(|cx| {
                ensure!(project_repair_binding(&project, directory.clone(), generation, cx)? == binding, "Doctor authority changed during advisory planning");
                Ok::<_, anyhow::Error>(())
            })?;
            let requests = request.exact_requests()?;
            let operation = match &request { AdvisoryRefreshRequest::Clear { dataset } => format!("clear:{dataset:?}"), AdvisoryRefreshRequest::MatchRetained { dataset } => format!("match:{dataset:?}"), _ => "retrieve".into() };
            let mut identity = requests.iter().map(String::as_str).collect::<Vec<_>>();
            identity.push(&operation);
            let revision = expected_revision.as_ref().map(DependencyIdentity::hex).unwrap_or_else(|| "absent".into());
            identity.push(&revision);
            let query_identity = if let AdvisoryRefreshRequest::RegistryVersions(query) = &request {
                Some(query.identity()?.hex())
            } else { None };
            if let Some(query_identity) = &query_identity { identity.push(query_identity); }
            let identity = DependencyIdentity::from_fields(&identity)?.hex();
            let source_inputs = if matches!(&request, AdvisoryRefreshRequest::RegistryVersions(_)) {
                Some(cx.update(|cx| {
                    let scan = project.read(cx).worktree_for_id(directory.worktree_id, cx).ok_or_else(|| anyhow::anyhow!("Selected worktree closed"))?.read(cx).scan_id();
                    Ok::<_, anyhow::Error>((scan, crate::doctor::reproduction_buffer_versions(&project, directory.worktree_id, cx)))
                })?)
            } else { None };
            let effects = if let AdvisoryRefreshRequest::MatchRetained { dataset } = &request {
                DiagnosticText::new(format!("Evaluate only exact selected {dataset:?} retained public advisory data against current selected owner evidence. No network, dependency fetch/resolution, subprocess/source/lock/settings/credential changes, permissions or schedules. Sixty-second total deadline and bounded cooperative matching; current graph/cache/project authority at dispatch and late admission. Existing thirty-day public-cache expiry remains in force. Stale, incomplete, absent or unsupported data never proves no vulnerabilities; full graph/advisory bodies remain transient and history contains only bounded redacted summaries."))
            } else if let AdvisoryRefreshRequest::Clear { dataset } = &request {
                DiagnosticText::new(format!("Delete only the selected {dataset:?} shared local-host public advisory dataset at the reviewed revision. No network, source/lock/configuration/history/session/credential or other dataset changes; preserve foreign files and native containment guards. Sixty-second deadline, current authority/cancellation before deletion and result admission. Interruption can leave completed deletion effects; no rollback or automatic replay. Observe retained absence only; clearing is not a security fix."))
            } else if let AdvisoryRefreshRequest::RegistryVersions(query) = &request {
                DiagnosticText::new(format!("Exact GET {}; only the explicitly selected and confirmed public coordinate is disclosed. No credentials, project/source/lock data, redirects, resolution, package downloads, cache replacement or lock/repository changes. One-MiB response and sixty-second deadline, transient results with bounded redacted operation history. Catalog-only {}: resolved artifact source remains {:?}, candidates are not verified updates or security fixes. Freshness, graph/source revision and compatibility uncertainty remain explicit", query.exact_url()?, query.is_catalog_only(), query.package().source.kind()))
            } else { DiagnosticText::new("Read only the exact separately displayed official source URLs, with no credential/project upload or redirects. Sixty-second total deadline, sixteen-KiB revision metadata or 128-MiB per archive, 256-MiB expanded/retained cache cap. Validate paths/checksums/source and required withdrawals before replacing this selected host's shared Made/Copper/Groovy public advisory cache; preserve previous data on incomplete refresh. No scripts, package installation, graph resolution, source changes, permissions or schedules. Revision lookup and archive download require separate actions. Only dataset availability is rechecked; it proves neither absence of vulnerabilities nor application security.") };
            DoctorRepairReview::new(
                AdvisoryRefreshAction { request, cache, expected_revision, source_inputs, #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))] projection: None },
                binding,
                DoctorCheck::new(&format!("dependency.advisory.refresh.{identity}"), "Selected official advisory data", CheckOutcome::Untested, "No current reviewed retrieval/recheck has been observed. Cached data may be absent, stale or incomplete", now),
                effects,
            )
        })
    }

    #[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
    pub fn prepare_project_registry_query(
        project: &gpui::Entity<project::Project>,
        projection: DoctorDependencyProjection,
        query: RegistryQuery,
        generation: u64,
        now: u64,
        cx: &mut gpui::App,
    ) -> gpui::Task<Result<DoctorRepairReview<AdvisoryRefreshAction>>> {
        let project = project.clone();
        cx.spawn(async move |cx| {
            let directory = cx.update(|cx| {
                projection.revalidate(&project, cx)?;
                query.revalidate_graph(projection.graph())?;
                let directory = projection.binding().directory().clone();
                ensure!(
                    project_repair_binding(&project, directory.clone(), generation, cx)?
                        == *projection.binding(),
                    "Dependency view or review generation changed"
                );
                Ok::<_, anyhow::Error>(directory)
            })?;
            let review = cx
                .update(|cx| {
                    prepare_advisory_refresh(
                        &project,
                        directory,
                        generation,
                        AdvisoryRefreshRequest::RegistryVersions(query),
                        now,
                        cx,
                    )
                })
                .await?;
            cx.update(|cx| projection.revalidate(&project, cx))?;
            let mut action = review.action().clone();
            action.projection = Some(Arc::new(projection));
            DoctorRepairReview::new(
                action,
                review.binding().clone(),
                review.before().clone(),
                review.effects().clone(),
            )
        })
    }

    pub fn prepare_project_advisory_scan(
        project: &gpui::Entity<project::Project>,
        projection: DoctorDependencyProjection,
        dataset: AdvisoryDataset,
        generation: u64,
        now: u64,
        cx: &mut gpui::App,
    ) -> gpui::Task<Result<DoctorRepairReview<AdvisoryRefreshAction>>> {
        let project = project.clone();
        cx.spawn(async move |cx| {
            let directory = cx.update(|cx| {
                projection.revalidate(&project, cx)?;
                let ecosystem = match dataset { AdvisoryDataset::RustSec => DependencyEcosystem::Cargo, AdvisoryDataset::OsvMaven => DependencyEcosystem::Maven };
                ensure!(projection.graph().nodes().iter().any(|node| node.ecosystem == ecosystem), "Selected dataset does not cover the selected dependency ecosystem");
                let directory = projection.binding().directory().clone();
                ensure!(project_repair_binding(&project, directory.clone(), generation, cx)? == *projection.binding(), "Dependency view or review generation changed");
                Ok::<_, anyhow::Error>(directory)
            })?;
            let review = cx.update(|cx| prepare_advisory_refresh(&project, directory, generation, AdvisoryRefreshRequest::MatchRetained { dataset }, now, cx)).await?;
            cx.update(|cx| projection.revalidate(&project, cx))?;
            ensure!(review.action().expected_revision.is_some(), "Retained advisory data is missing; explicitly review any refresh before matching");
            let mut action = review.action().clone();
            let graph_revision = projection.graph().identity().hex();
            action.projection = Some(Arc::new(projection));
            let identity = DependencyIdentity::from_fields(&[review.before().id.as_str(), &graph_revision])?.hex();
            DoctorRepairReview::new(action, review.binding().clone(), DoctorCheck::new(&format!("dependency.advisory.match.{identity}"), "Selected retained advisory evaluation", CheckOutcome::Untested, "No current reviewed match is admitted; retained source and graph coverage/freshness remain explicit", now), review.effects().clone())
        })
    }

    enum Retrieval {
        Revision(String),
        Snapshot(AdvisorySnapshot),
        Registry(RegistryMetadata),
        Clear(AdvisoryDataset),
        Scan(AdvisoryScan),
    }

    fn progress(
        sender: &async_channel::Sender<AdvisoryRefreshProgress>,
        stage: &str,
        received_bytes: u64,
    ) -> Result<()> {
        match sender.try_send(AdvisoryRefreshProgress {
            stage: DiagnosticText::new(stage),
            received_bytes,
        }) {
            Ok(()) | Err(async_channel::TrySendError::Full(_)) => Ok(()),
            Err(async_channel::TrySendError::Closed(_)) => {
                anyhow::bail!("Advisory progress consumer closed; operation cancelled")
            }
        }
    }

    fn validate_official_url(value: &str) -> Result<()> {
        let url =
            url::Url::parse(value).map_err(|_| anyhow::anyhow!("Invalid advisory source URL"))?;
        ensure!(
            url.scheme() == "https"
                && url.username().is_empty()
                && url.password().is_none()
                && url.port().is_none()
                && url.query().is_none()
                && url.fragment().is_none(),
            "Unapproved advisory source authority"
        );
        let fixed = [
            "https://api.github.com/repos/RustSec/advisory-db/commits/main".to_owned(),
            AdvisoryArchiveKind::OsvMaven.source_url(""),
            AdvisoryArchiveKind::OsvWithdrawals.source_url(""),
        ];
        let commit = value.strip_prefix("https://codeload.github.com/RustSec/advisory-db/zip/");
        ensure!(
            fixed.iter().any(|approved| approved == value)
                || registry_updates::is_metadata_url(value)
                || commit.is_some_and(|commit| commit.len() == 40
                    && commit.bytes().all(|byte| byte.is_ascii_hexdigit())),
            "Unapproved advisory source destination"
        );
        Ok(())
    }

    struct OfficialBody {
        bytes: Vec<u8>,
        source_revision: Option<String>,
        source_modified_at: Option<u64>,
        response_date: Option<u64>,
        response_age: Option<u64>,
    }

    #[cfg(all(test, feature = "rust-tools", feature = "jvm-tools"))]
    async fn read_official_body(
        client: &Arc<dyn HttpClient>,
        url: &str,
        maximum: usize,
    ) -> Result<Vec<u8>> {
        Ok(read_official_response(client, url, maximum).await?.bytes)
    }

    async fn read_official_response(
        client: &Arc<dyn HttpClient>,
        url: &str,
        maximum: usize,
    ) -> Result<OfficialBody> {
        validate_official_url(url)?;
        ensure!(
            maximum <= advisory_cache::MAX_ADVISORY_ARCHIVE_BYTES,
            "Advisory transport byte bound exceeds approved cap"
        );
        let request = Request::builder()
            .method("GET")
            .uri(url)
            .header(
                "Accept",
                if url.starts_with("https://api.github.com/")
                    || url.starts_with("https://crates.io/")
                {
                    "application/vnd.github+json"
                } else {
                    "application/octet-stream"
                },
            )
            .header("Accept-Encoding", "identity")
            .header("User-Agent", "Made-Doctor")
            .follow_redirects(RedirectPolicy::NoFollow)
            .timeout(REFRESH_DEADLINE)
            .body(AsyncBody::empty())?;
        let response = client.send(request).await.map_err(|_| {
            anyhow::anyhow!("Official advisory request failed; no current evidence")
        })?;
        ensure!(
            response.status() == http_client::StatusCode::OK,
            "Official advisory request returned status {}; no current evidence",
            response.status().as_u16()
        );
        ensure!(
            response
                .headers()
                .get_all("Content-Encoding")
                .iter()
                .all(|value| value.as_bytes() == b"identity"),
            "Encoded advisory HTTP response is unsupported"
        );
        let lengths = response
            .headers()
            .get_all("Content-Length")
            .iter()
            .collect::<Vec<_>>();
        ensure!(lengths.len() <= 1, "Ambiguous advisory response length");
        let expected = lengths
            .first()
            .map(|value| {
                value
                    .to_str()
                    .map_err(|_| anyhow::anyhow!("Invalid advisory response length"))
                    .and_then(|value| {
                        value
                            .parse::<usize>()
                            .map_err(|_| anyhow::anyhow!("Invalid advisory response length"))
                    })
            })
            .transpose()?;
        ensure!(
            expected.is_none_or(|length| length <= maximum),
            "Advisory download exceeds its reviewed byte cap"
        );
        let generations = response
            .headers()
            .get_all("x-goog-generation")
            .iter()
            .collect::<Vec<_>>();
        let modified = response
            .headers()
            .get_all("Last-Modified")
            .iter()
            .collect::<Vec<_>>();
        ensure!(
            generations.len() <= 1 && modified.len() <= 1,
            "Ambiguous official-source version metadata"
        );
        let source_revision = generations
            .first()
            .map(|value| -> Result<String> {
                let value = value
                    .to_str()
                    .map_err(|_| anyhow::anyhow!("Invalid official-source generation"))?;
                ensure!(
                    !value.is_empty()
                        && value.len() <= 64
                        && value.bytes().all(|byte| byte.is_ascii_digit()),
                    "Invalid official-source generation"
                );
                Ok(format!("generation:{value}"))
            })
            .transpose()?;
        let source_modified_at = modified
            .first()
            .map(|value| -> Result<u64> {
                let value = value
                    .to_str()
                    .map_err(|_| anyhow::anyhow!("Invalid official-source publication time"))?;
                let timestamp = time::OffsetDateTime::parse(
                    value,
                    &time::format_description::well_known::Rfc2822,
                )
                .map_err(|_| anyhow::anyhow!("Invalid official-source publication time"))?;
                u64::try_from(timestamp.unix_timestamp())
                    .map_err(|_| anyhow::anyhow!("Invalid official-source publication time"))
            })
            .transpose()?;
        let parse_time_header = |name: &str| -> Result<Option<u64>> {
            let values = response.headers().get_all(name).iter().collect::<Vec<_>>();
            ensure!(values.len() <= 1, "Ambiguous public metadata time");
            values
                .first()
                .map(|value| {
                    let value = value
                        .to_str()
                        .map_err(|_| anyhow::anyhow!("Invalid public metadata time"))?;
                    let parsed = time::OffsetDateTime::parse(
                        value,
                        &time::format_description::well_known::Rfc2822,
                    )
                    .map_err(|_| anyhow::anyhow!("Invalid public metadata time"))?;
                    u64::try_from(parsed.unix_timestamp())
                        .map_err(|_| anyhow::anyhow!("Invalid public metadata time"))
                })
                .transpose()
        };
        let response_date = parse_time_header("Date")?;
        let ages = response.headers().get_all("Age").iter().collect::<Vec<_>>();
        ensure!(ages.len() <= 1, "Ambiguous public metadata age");
        let response_age = ages
            .first()
            .map(|value| {
                value
                    .to_str()
                    .ok()
                    .and_then(|value| value.parse::<u64>().ok())
                    .ok_or_else(|| anyhow::anyhow!("Invalid public metadata age"))
            })
            .transpose()?;
        let mut bytes = Vec::new();
        response
            .into_body()
            .take(maximum as u64 + 1)
            .read_to_end(&mut bytes)
            .await
            .map_err(|_| {
                anyhow::anyhow!("Advisory response collection failed; current data is incomplete")
            })?;
        ensure!(
            bytes.len() <= maximum && expected.is_none_or(|length| bytes.len() == length),
            "Advisory response is oversized or incomplete"
        );
        Ok(OfficialBody {
            bytes,
            source_revision,
            source_modified_at,
            response_date,
            response_age,
        })
    }

    async fn retrieve(
        action: &AdvisoryRefreshAction,
        client: Arc<dyn HttpClient>,
        observed_at: u64,
        sender: &async_channel::Sender<AdvisoryRefreshProgress>,
        cancelled: &async_channel::Receiver<()>,
        stopped: &std::sync::atomic::AtomicBool,
    ) -> Result<Retrieval> {
        let requests = action.request.exact_requests()?;
        if let AdvisoryRefreshRequest::MatchRetained { dataset } = action.request {
            progress(
                sender,
                "Evaluating exact retained advisory revision locally; no HTTP",
                0,
            )?;
            let retained = action.cache.load(dataset, observed_at).await?;
            ensure!(
                retained.problem.is_none() && retained.revision == action.expected_revision,
                "Retained advisory data changed or is corrupt/incomplete; no scan result"
            );
            let snapshot = retained.snapshot.ok_or_else(|| {
                anyhow::anyhow!("Retained advisory data is absent; evaluation incomplete")
            })?;
            let projection = action
                .projection
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Current selected graph is missing"))?;
            let scan = match_advisories(projection.graph(), &snapshot, observed_at, || {
                ensure!(
                    !cancelled.is_closed()
                        && cancelled.is_empty()
                        && !stopped.load(std::sync::atomic::Ordering::Acquire),
                    "Retained advisory matching cancelled or timed out; remaining records unevaluated"
                );
                Ok(())
            })?;
            progress(
                sender,
                "Bounded local matching finished; revalidating graph/cache before admission",
                0,
            )?;
            return Ok(Retrieval::Scan(scan));
        }
        if let AdvisoryRefreshRequest::Clear { dataset } = action.request {
            progress(
                sender,
                "Preparing reviewed selected-cache deletion; no HTTP request",
                0,
            )?;
            return Ok(Retrieval::Clear(dataset));
        }
        if let AdvisoryRefreshRequest::RegistryVersions(query) = &action.request {
            progress(
                sender,
                "Reading exact reviewed public registry version metadata",
                0,
            )?;
            let body = read_official_response(
                &client,
                &query.exact_url()?,
                registry_updates::MAX_REGISTRY_BYTES,
            )
            .await?;
            progress(
                sender,
                "Validating bounded version metadata; no package download or resolution",
                body.bytes.len() as u64,
            )?;
            return Ok(Retrieval::Registry(registry_updates::parse_metadata(
                query.clone(),
                &body.bytes,
                observed_at,
                body.response_date,
                body.response_age,
                body.source_modified_at,
            )?));
        }
        if action.request == AdvisoryRefreshRequest::ResolveRustSecRevision {
            progress(sender, "Retrieving official RustSec commit metadata", 0)?;
            let url = requests
                .first()
                .ok_or_else(|| anyhow::anyhow!("Missing revision metadata request"))?;
            let bytes = read_official_response(&client, url, MAX_REVISION_BYTES)
                .await?
                .bytes;
            let parsed: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|_| anyhow::anyhow!("Invalid official RustSec revision response"))?;
            let commit = parsed
                .get("sha")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("Official RustSec response has no commit"))?;
            AdvisoryRefreshRequest::RustSec {
                commit: commit.to_owned(),
            }
            .validate()?;
            progress(
                sender,
                "Exact RustSec commit resolved; separately review archive download",
                bytes.len() as u64,
            )?;
            return Ok(Retrieval::Revision(commit.to_owned()));
        }
        let dataset = action
            .request
            .dataset()
            .ok_or_else(|| anyhow::anyhow!("Missing advisory dataset"))?;
        let kinds: &[AdvisoryArchiveKind] = match dataset {
            AdvisoryDataset::RustSec => &[AdvisoryArchiveKind::RustSec],
            AdvisoryDataset::OsvMaven => &[
                AdvisoryArchiveKind::OsvMaven,
                AdvisoryArchiveKind::OsvWithdrawals,
            ],
        };
        let mut archives = Vec::new();
        let mut publication_times = Vec::new();
        let mut total = 0u64;
        for (url, kind) in requests.iter().zip(kinds) {
            progress(
                sender,
                match kind {
                    AdvisoryArchiveKind::RustSec => "Downloading exact reviewed RustSec revision",
                    AdvisoryArchiveKind::OsvMaven => "Downloading official OSV Maven archive",
                    AdvisoryArchiveKind::OsvWithdrawals => {
                        "Downloading required OSV withdrawal archive"
                    }
                },
                total,
            )?;
            let body =
                read_official_response(&client, url, advisory_cache::MAX_ADVISORY_ARCHIVE_BYTES)
                    .await?;
            publication_times.push(body.source_modified_at);
            let bytes = body.bytes;
            total = total
                .checked_add(bytes.len() as u64)
                .ok_or_else(|| anyhow::anyhow!("Advisory transfer byte overflow"))?;
            let revision = match &action.request {
                AdvisoryRefreshRequest::RustSec { commit } => commit.clone(),
                _ => body.source_revision.unwrap_or_else(|| {
                    format!(
                        "sha256:{}",
                        Sha256::digest(&bytes)
                            .iter()
                            .map(|byte| format!("{byte:02x}"))
                            .collect::<String>()
                    )
                }),
            };
            archives.push(AdvisoryArchive {
                kind: *kind,
                revision,
                bytes,
            });
        }
        progress(
            sender,
            "Validating source, record, withdrawal, path and expansion coverage",
            total,
        )?;
        let mut snapshot = action
            .cache
            .validate_archives(dataset, archives, observed_at)
            .await?;
        for (source, modified) in snapshot.sources.iter_mut().zip(publication_times) {
            source.source_modified_at = modified;
        }
        Ok(Retrieval::Snapshot(snapshot))
    }

    pub fn launch_advisory_refresh(
        mut review: DoctorRepairReview<AdvisoryRefreshAction>,
        project: &gpui::Entity<project::Project>,
        directory: project::ProjectPath,
        generation: u64,
        report: DoctorReport,
        product: DoctorProduct,
        history: DoctorHistoryStore,
        now: u64,
        cancellation: async_channel::Sender<()>,
        cancelled: async_channel::Receiver<()>,
        updates: async_channel::Sender<AdvisoryRefreshProgress>,
        cx: &mut gpui::App,
    ) -> gpui::Task<Result<AdvisoryRefreshResult>> {
        let project = project.downgrade();
        cx.spawn(async move |cx| {
            let binding = cx.update(|cx| {
                let owner = project.upgrade().ok_or_else(|| anyhow::anyhow!("Doctor project closed"))?;
                review.action().revalidate_inputs(&owner, &directory, cx)?;
                project_repair_binding(&owner, directory.clone(), generation, cx)
            })?;
            review.record_before_execution(&binding, report, product, &history, now, &cancelled).await?;
            let executor = cx.background_executor().clone();
            let started = executor.now();
            let revoked = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let monitor = cx.spawn({
                let project = project.clone(); let directory = directory.clone(); let binding = binding.clone(); let executor = executor.clone(); let revoked = revoked.clone(); let cancellation = cancellation.clone(); let action = review.action().clone();
                async move |cx| {
                    loop {
                        let current = cx.update(|cx| {
                            let owner = project.upgrade().ok_or_else(|| anyhow::anyhow!("Doctor project closed"))?;
                            action.revalidate_inputs(&owner, &directory, cx)?;
                            project_repair_binding(&owner, directory.clone(), generation, cx)
                        });
                        if !matches!(&current, Ok(current) if current == &binding) {
                            let reason = current.err().map(|error| DiagnosticText::new(error.to_string())).unwrap_or_else(|| DiagnosticText::new("Reviewed Project/host/settings binding changed"));
                            log::warn!("Doctor advisory authority changed: {}", reason.as_str());
                            revoked.store(true, std::sync::atomic::Ordering::Release);
                            cancellation.close();
                            return;
                        }
                        executor.timer(Duration::from_millis(100)).await;
                    }
                }
            });
            let operation = async {
                cx.update(|cx| {
                    let current = project_repair_binding(&project.upgrade().ok_or_else(|| anyhow::anyhow!("Doctor project closed"))?, directory.clone(), generation, cx)?;
                    review.revalidate(&current)?;
                    review.action().revalidate_inputs(&project.upgrade().ok_or_else(|| anyhow::anyhow!("Doctor project closed"))?, &directory, cx)?;
                    Ok::<_, anyhow::Error>(())
                })?;
                let client: Arc<dyn HttpClient> = cx.update(|cx| project.upgrade().ok_or_else(|| anyhow::anyhow!("Doctor project closed")).map(|owner| owner.read(cx).client().http_client()))?;
                executor.spawn({
                    let action = review.action().clone();
                    let updates = updates.clone();
                    let cancelled = cancelled.clone();
                    let stopped = stopped.clone();
                    async move { retrieve(&action, client, now, &updates, &cancelled, &stopped).await }
                }).await
            };
            let stop = async {
                futures::future::select(cancelled.recv().boxed_local(), executor.timer(REFRESH_DEADLINE).boxed_local()).await
            };
            let result = match futures::future::select(operation.boxed_local(), stop.boxed_local()).await {
                futures::future::Either::Left((result, _)) => result,
                futures::future::Either::Right((futures::future::Either::Left(_), _)) => Err(anyhow::anyhow!("Advisory refresh cancelled; downloaded data was not admitted")),
                futures::future::Either::Right((futures::future::Either::Right(_), _)) => Err(anyhow::anyhow!("Advisory refresh timed out; data remains incomplete")),
            };
            let observed_at = now.saturating_add(executor.now().duration_since(started).as_secs());
            let mut resolved_commit = None;
            let mut snapshot = None;
            let mut registry_metadata = None;
            if result.is_err() { stopped.store(true, std::sync::atomic::Ordering::Release); }
            let mut cleared_dataset = None;
            let mut scan = None;
            let completion = async {
                ensure!(!cancelled.is_closed() && cancelled.is_empty() && !revoked.load(std::sync::atomic::Ordering::Acquire), "Advisory authority was cancelled or revoked; result discarded");
                let retrieval = result?;
                cx.update(|cx| review.revalidate(&project_repair_binding(&project.upgrade().ok_or_else(|| anyhow::anyhow!("Doctor project closed"))?, directory.clone(), generation, cx)?))?;
                cx.update(|cx| review.action().revalidate_inputs(&project.upgrade().ok_or_else(|| anyhow::anyhow!("Doctor project closed"))?, &directory, cx))?;
                match retrieval {
                    Retrieval::Scan(mut candidate) => {
                        progress(&updates, "Reopening retained revision and checking current selected graph", 0)?;
                        let dataset = review.action().request.dataset().ok_or_else(|| anyhow::anyhow!("Selected dataset disappeared"))?;
                        let retained = executor.spawn({ let cache = review.action().cache.clone(); async move { cache.load(dataset, observed_at).await } }).await?;
                        ensure!(retained.problem.is_none() && retained.revision == review.action().expected_revision, "Retained advisory data changed before result admission");
                        let retained_snapshot = retained.snapshot.ok_or_else(|| anyhow::anyhow!("Retained dataset is absent at recheck"))?;
                        ensure!(candidate.advisory_revision == retained_snapshot.identity()?, "Matched advisory contents changed before admission");
                        cx.update(|cx| {
                            let owner = project.upgrade().ok_or_else(|| anyhow::anyhow!("Selected project closed"))?;
                            review.revalidate(&project_repair_binding(&owner, directory.clone(), generation, cx)?)?;
                            review.action().revalidate_inputs(&owner, &directory, cx)
                        })?;
                        let admission_at = now.saturating_add(executor.now().duration_since(started).as_secs());
                        candidate.observed_at = admission_at;
                        candidate.freshness = retained_snapshot.freshness(admission_at);
                        candidate.upstream_freshness = retained_snapshot.upstream_freshness(admission_at);
                        if candidate.freshness != AdvisoryFreshness::Fresh || candidate.upstream_freshness != Some(AdvisoryFreshness::Fresh) {
                            candidate.incomplete("Retained publication/download age is stale or uncertain at admission");
                        }
                        let detail = format!("{} Selected graph {}, advisory {}, download freshness {:?}, publication freshness {:?}. Local retained evaluation only; no network/dependency/source effects", candidate.conclusion().as_str(), candidate.graph_revision.hex(), candidate.advisory_revision.hex(), candidate.freshness, candidate.upstream_freshness);
                        snapshot = Some(retained_snapshot);
                        scan = Some(candidate);
                        Ok(detail)
                    }
                    Retrieval::Clear(dataset) => {
                        progress(&updates, "Revalidating reviewed deletion and observing selected cache absence", 0)?;
                        let action = review.action().clone();
                        let removed = action.cache.clear_selected_when_authorized(dataset, action.expected_revision.as_ref(), || {
                            ensure!(!cancelled.is_closed() && cancelled.is_empty() && !revoked.load(std::sync::atomic::Ordering::Acquire), "Clear cancelled or revoked; completed deletion effects may remain and recheck is incomplete");
                            cx.update(|cx| {
                                let owner = project.upgrade().ok_or_else(|| anyhow::anyhow!("Doctor project closed; completed deletion effects may remain"))?;
                                review.revalidate(&project_repair_binding(&owner, directory.clone(), generation, cx)?)?;
                                action.revalidate_inputs(&owner, &directory, cx)
                            })
                        }).await?;
                        cleared_dataset = Some(dataset);
                        Ok(format!("Selected {dataset:?} public cache {} and absence independently observed. Other datasets/foreign files preserved. This proves cache state only, not a vulnerability fix or project security", if removed { "removed" } else { "already absent" }))
                    }
                    Retrieval::Registry(metadata) => {
                        let detail = format!("Bounded public catalog metadata observed; coverage {:?}, freshness {:?}. No dependency was downloaded/resolved/updated, no lock/repository changed and no security fix was verified. Catalog-only {}; artifact source {:?}", metadata.coverage, metadata.freshness(observed_at), metadata.query.is_catalog_only(), metadata.query.package().source.kind());
                        registry_metadata = Some(metadata);
                        Ok(detail)
                    }
                    Retrieval::Revision(commit) => { resolved_commit = Some(commit); Ok("Official exact commit metadata was observed. Archive retrieval requires a separate approval; no cache was replaced".to_owned()) }
                    Retrieval::Snapshot(candidate) => {
                        progress(&updates, "Admitting current reviewed data and reopening retained cache", 0)?;
                        let dataset = candidate.dataset;
                        snapshot = executor.spawn({
                            let action = review.action().clone();
                            let cancelled = cancelled.clone();
                            let revoked = revoked.clone();
                            async move {
                                let authorized = || -> Result<()> {
                                    ensure!(!cancelled.is_closed() && cancelled.is_empty() && !revoked.load(std::sync::atomic::Ordering::Acquire), "Advisory authority changed before admission/recheck; reached completed cache writes may remain but current evidence is incomplete");
                                    Ok(())
                                };
                                let expected = action.cache.replace_when_authorized(candidate, action.expected_revision.as_ref(), observed_at, &authorized).await?;
                                authorized()?;
                                let retained = action.cache.load(dataset, observed_at).await?;
                                authorized()?;
                                ensure!(retained.revision == Some(expected) && retained.snapshot.as_ref().is_some_and(|snapshot| snapshot.freshness(observed_at) == AdvisoryFreshness::Fresh), "Advisory cache recheck did not observe this valid fresh replacement");
                                Ok::<_, anyhow::Error>(retained.snapshot)
                            }
                        }).await?;
                        cx.update(|cx| review.revalidate(&project_repair_binding(&project.upgrade().ok_or_else(|| anyhow::anyhow!("Doctor project closed"))?, directory.clone(), generation, cx)?))?;
                        Ok("Validated official-source data was retained and independently reopened as current. This establishes dataset availability only; local matching/coverage and application exploitability remain untested".to_owned())
                    }
                }
            };
            let remaining = REFRESH_DEADLINE.saturating_sub(executor.now().duration_since(started));
            let stop = async {
                futures::future::select(cancelled.recv().boxed_local(), executor.timer(remaining).boxed_local()).await
            };
            let completion = match futures::future::select(completion.boxed_local(), stop.boxed_local()).await {
                futures::future::Either::Left((result, _)) => result,
                futures::future::Either::Right((futures::future::Either::Left(_), _)) => Err(anyhow::anyhow!("Advisory operation cancelled during admission/recheck; completed public-cache effects may remain; no rollback or replay")),
                futures::future::Either::Right((futures::future::Either::Right(_), _)) => Err(anyhow::anyhow!("Advisory operation timed out during admission/recheck; completed public-cache effects may remain; no rollback or replay")),
            };
            if completion.is_err() { stopped.store(true, std::sync::atomic::Ordering::Release); }
            let observed_at = now.saturating_add(executor.now().duration_since(started).as_secs());
            drop(monitor);
            let is_cancelled = cancelled.is_closed() || !cancelled.is_empty();
            let (outcome, detail, check) = match completion {
                Ok(detail) => {
                    let check = if registry_metadata.as_ref().is_some_and(|metadata| metadata.coverage != DependencyCoverage::Complete || metadata.freshness(observed_at) != Some(AdvisoryFreshness::Fresh)) { CheckOutcome::Untested } else { CheckOutcome::Ready };
                    let check = if let Some(scan) = &scan {
                        if scan.findings.iter().any(|finding| finding.evidence.state == DependencyAdvisoryState::KnownAffectedDependency) { CheckOutcome::Warning }
                        else if scan.coverage != DependencyCoverage::Complete || scan.freshness != AdvisoryFreshness::Fresh || scan.upstream_freshness != Some(AdvisoryFreshness::Fresh) { CheckOutcome::Untested }
                        else { CheckOutcome::Ready }
                    } else { check };
                    (DoctorRepairOperationOutcome::Completed, detail, check)
                },
                Err(error) => {
                    resolved_commit = None; snapshot = None; registry_metadata = None; cleared_dataset = None; scan = None;
                    (if is_cancelled { DoctorRepairOperationOutcome::Cancelled } else { DoctorRepairOperationOutcome::Failed }, format!("Incomplete advisory operation: {error}. Previous retained data is not newly established as fresh; no absence-of-findings/security conclusion"), if is_cancelled { CheckOutcome::Cancelled } else { CheckOutcome::Untested })
                }
            };
            review.operation_finished(outcome, DiagnosticText::new(&detail))?;
            if outcome != DoctorRepairOperationOutcome::Cancelled {
                let current = cx.update(|cx| project_repair_binding(&project.upgrade().ok_or_else(|| anyhow::anyhow!("Doctor project closed"))?, directory, generation, cx));
                match current {
                    Ok(current) if &current == review.binding() => review.observe_recheck(&current, DoctorCheck::new(review.before().id.as_str(), "Observed advisory operation and recheck", check, &detail, observed_at))?,
                    _ => review.recheck_unavailable(DiagnosticText::new("Selected project/host/settings authority changed; downloaded evidence is incomplete"))?,
                }
            }
            review.persist_outcome(product, &history, None, observed_at).await?;
            updates.close();
            Ok(AdvisoryRefreshResult { review, resolved_commit, snapshot, registry_metadata, cleared_dataset, scan })
        })
    }

    #[cfg(all(test, feature = "rust-tools", feature = "jvm-tools"))]
    mod tests {
        use super::*;
        use gpui::{BorrowAppContext as _, TestAppContext};
        use http_client::{FakeHttpClient, Response};
        use std::{
            path::Path,
            sync::atomic::{AtomicUsize, Ordering},
        };

        async fn zip(kind: AdvisoryArchiveKind) -> Vec<u8> {
            let (name, document) = match kind {
                AdvisoryArchiveKind::RustSec => (
                    "advisory-db-fixture/crates/example/RUSTSEC-2026-0001.md",
                    "```toml\n[advisory]\nid = \"RUSTSEC-2026-0001\"\npackage = \"example\"\ndate = \"2026-10-01\"\n[versions]\npatched = [\">=1.2\"]\n```\n# RAW_PUBLIC_ADVISORY_NOT_HISTORY",
                ),
                AdvisoryArchiveKind::OsvMaven => (
                    "GHSA-aaaa-bbbb-cccc.json",
                    r#"{"id":"GHSA-aaaa-bbbb-cccc","modified":"2026-10-01T00:00:00Z","summary":"RAW_PUBLIC_ADVISORY_NOT_HISTORY","affected":[{"package":{"ecosystem":"Maven","name":"org.example:library"},"versions":["1.0"]}]}"#,
                ),
                AdvisoryArchiveKind::OsvWithdrawals => (
                    "GHSA-dddd-eeee-ffff.json",
                    r#"{"id":"GHSA-dddd-eeee-ffff","modified":"2026-10-01T00:00:00Z","withdrawn":"2026-10-01T00:00:00Z"}"#,
                ),
            };
            let mut writer = async_zip::base::write::ZipFileWriter::new(Vec::new());
            writer
                .write_entry_whole(
                    async_zip::ZipEntryBuilder::new(name.into(), async_zip::Compression::Deflate),
                    document.as_bytes(),
                )
                .await
                .expect("Public ZIP fixture");
            writer.close().await.expect("ZIP fixture closure")
        }

        async fn fixture(
            cx: &mut TestAppContext,
        ) -> (gpui::Entity<project::Project>, project::ProjectPath) {
            use project::trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees};
            cx.update(|cx| {
                let settings = settings::SettingsStore::test(cx);
                cx.set_global(settings);
                trusted_worktrees::init(DbTrustedPaths::default(), cx);
            });
            let filesystem = fs::FakeFs::new(cx.background_executor.clone());
            filesystem
                .insert_tree(
                    "/doctor-fetch",
                    serde_json::json!({"source.txt":"unrelated project edits"}),
                )
                .await;
            let owner = project::Project::test_with_worktree_trust(
                filesystem,
                [Path::new("/doctor-fetch")],
                cx,
            )
            .await;
            let directory = owner.read_with(cx, |owner, cx| project::ProjectPath {
                worktree_id: owner
                    .visible_worktrees(cx)
                    .next()
                    .expect("Selected root")
                    .read(cx)
                    .id(),
                path: util::rel_path::RelPath::empty_arc(),
            });
            let worktrees = owner.read_with(cx, |owner, _| owner.worktree_store());
            let trust =
                cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("Existing trust owner"));
            trust.update(cx, |trust, cx| {
                trust.trust(
                    &worktrees,
                    collections::HashSet::from_iter([PathTrust::Worktree(directory.worktree_id)]),
                    cx,
                )
            });
            cx.run_until_parked();
            (owner, directory)
        }

        fn response(bytes: Vec<u8>) -> Result<http_client::Response<AsyncBody>> {
            Ok(Response::builder()
                .status(200)
                .header("Content-Length", bytes.len())
                .body(bytes.into())?)
        }

        fn report(
            review: &DoctorRepairReview<AdvisoryRefreshAction>,
            product: &str,
            now: u64,
        ) -> DoctorReport {
            let mut report = DoctorReport::new(
                crate::doctor::DoctorScope {
                    product: DiagnosticText::new(product),
                    project: DiagnosticText::new("selected project"),
                    worktree: DiagnosticText::new("selected worktree"),
                    host: DiagnosticText::new("Local"),
                },
                1,
                now,
            );
            report
                .push(review.before().clone())
                .expect("Relevant check");
            report
        }

        async fn approved(
            cx: &mut TestAppContext,
            owner: &gpui::Entity<project::Project>,
            directory: &project::ProjectPath,
            request: AdvisoryRefreshRequest,
            now: u64,
        ) -> DoctorRepairReview<AdvisoryRefreshAction> {
            let mut review = cx
                .update(|cx| {
                    prepare_advisory_refresh(owner, directory.clone(), 1, request, now, cx)
                })
                .await
                .expect("Proposed exact requests");
            review
                .approve(
                    &cx.update(|cx| project_repair_binding(owner, directory.clone(), 1, cx))
                        .expect("Current authority"),
                )
                .expect("Explicit fixture consent");
            review
        }

        #[gpui::test]
        async fn doctor_advisory_clear_native_review_receipts_cancel_revocation_and_no_http(
            cx: &mut TestAppContext,
        ) {
            for scenario in [
                "unapproved",
                "cancel",
                "revoke",
                "changed",
                "success",
                "duplicate",
            ] {
                let (owner, directory) = fixture(cx).await;
                let calls = Arc::new(AtomicUsize::new(0));
                owner
                    .read_with(cx, |owner, _| owner.client().http_client())
                    .as_fake()
                    .replace_handler({
                        let calls = calls.clone();
                        move |_, _| {
                            calls.fetch_add(1, Ordering::SeqCst);
                            async { anyhow::bail!("Clear must never make HTTP requests") }
                        }
                    });
                let cache = owner
                    .read_with(cx, |owner, _| AdvisoryCache::for_local_project(owner))
                    .expect("Existing cache owner");
                let snapshot = super::super::advisory_cache::matching_fixture_for_cache(&cache, AdvisoryDataset::RustSec,
                    vec![AdvisoryRecord { kind: AdvisoryArchiveKind::RustSec, identifier: "RUSTSEC-2026-0001".into(), document: "```toml\n[advisory]\nid = \"RUSTSEC-2026-0001\"\npackage = \"example\"\ndate = \"2026-10-01\"\n[versions]\npatched = [\">= 1.2\"]\n```\nRAW_NOT_HISTORY".into() }], 1000);
                let revision = cache
                    .replace(snapshot.clone(), None, 1000)
                    .await
                    .expect("Retained dataset");
                let request = AdvisoryRefreshRequest::Clear {
                    dataset: AdvisoryDataset::RustSec,
                };
                let mut review = cx
                    .update(|cx| {
                        prepare_advisory_refresh(
                            &owner,
                            directory.clone(),
                            1,
                            request.clone(),
                            1000,
                            cx,
                        )
                    })
                    .await
                    .expect("Review no effects");
                assert!(
                    review
                        .action()
                        .exact_preview()
                        .expect("Exact preview")
                        .iter()
                        .any(|line| line.contains("No network"))
                );
                assert_eq!(
                    cache
                        .load(AdvisoryDataset::RustSec, 1000)
                        .await
                        .expect("Unchanged before approval")
                        .revision,
                    Some(revision.clone())
                );
                if scenario != "unapproved" {
                    review
                        .approve(
                            &cx.update(|cx| {
                                project_repair_binding(&owner, directory.clone(), 1, cx)
                            })
                            .expect("Binding"),
                        )
                        .expect("Explicit action approval");
                }
                let duplicate = if scenario == "duplicate" {
                    Some(approved(cx, &owner, &directory, request, 1000).await)
                } else {
                    None
                };
                if scenario == "changed" {
                    let mut updated = snapshot;
                    updated.fetched_at = 1001;
                    cache
                        .replace(updated, Some(&revision), 1001)
                        .await
                        .expect("Changed shared public data");
                }
                if scenario == "revoke" {
                    cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                        settings
                            .set_user_settings(r#"{"tab_size":3}"#, cx)
                            .expect("Revoked review inputs");
                    });
                }
                let history = DoctorHistoryStore::for_test_database(&db::AppDatabase::test_new());
                let source = report(&review, "Copper", 1000);
                let (cancel, cancelled) = async_channel::bounded(1);
                if scenario == "cancel" {
                    cancel.close();
                }
                let (updates, _progress) = async_channel::bounded(16);
                let run = cx.update(|cx| {
                    launch_advisory_refresh(
                        review,
                        &owner,
                        directory.clone(),
                        1,
                        source,
                        DoctorProduct::Copper,
                        history.clone(),
                        1000,
                        cancel,
                        cancelled,
                        updates,
                        cx,
                    )
                });
                let duplicate_run = duplicate.map(|review| {
                    let source = report(&review, "Copper", 1000);
                    let (cancel, cancelled) = async_channel::bounded(1);
                    let (updates, _progress) = async_channel::bounded(16);
                    cx.update(|cx| {
                        launch_advisory_refresh(
                            review,
                            &owner,
                            directory.clone(),
                            1,
                            source,
                            DoctorProduct::Copper,
                            history.clone(),
                            1000,
                            cancel,
                            cancelled,
                            updates,
                            cx,
                        )
                    })
                });
                let mut result = run.await;
                if let Some(duplicate) = duplicate_run {
                    let duplicate = duplicate.await;
                    let cleared = |result: &Result<AdvisoryRefreshResult>| {
                        result.as_ref().is_ok_and(|result| {
                            result.cleared_dataset == Some(AdvisoryDataset::RustSec)
                        })
                    };
                    assert_eq!(
                        usize::from(cleared(&result)) + usize::from(cleared(&duplicate)),
                        1,
                        "Equivalent overlapping or stale approved clears establish successful deletion only once"
                    );
                    if cleared(&duplicate) {
                        result = duplicate;
                    }
                }
                let success = matches!(scenario, "success" | "duplicate");
                assert_eq!(
                    result.as_ref().is_ok_and(
                        |result| result.cleared_dataset == Some(AdvisoryDataset::RustSec)
                    ),
                    success
                );
                assert_eq!(
                    cache
                        .load(AdvisoryDataset::RustSec, 1001)
                        .await
                        .expect("Observed cache state")
                        .snapshot
                        .is_none(),
                    success
                );
                if success {
                    let result = result.expect("Reviewed clear");
                    assert_eq!(
                        result.review.phase(),
                        crate::doctor::DoctorRepairPhase::Verified
                    );
                    let records = history
                        .records(DoctorProduct::Copper, 1001)
                        .await
                        .expect("Durable receipt");
                    assert_eq!(
                        records
                            .iter()
                            .filter(
                                |record| record.outcome == crate::doctor::ReceiptOutcome::Completed
                            )
                            .count(),
                        1,
                        "Only one successful clear receipt; active duplicates are rejected and reached stale actions remain failed"
                    );
                    assert!(records.len() <= if scenario == "duplicate" { 2 } else { 1 });
                    let serialized = serde_json::to_string(&records).expect("Bounded metadata");
                    assert!(
                        serialized.contains("absence") && !serialized.contains("RAW_NOT_HISTORY")
                    );
                }
                assert_eq!(calls.load(Ordering::SeqCst), 0);
                assert_eq!(
                    owner
                        .read_with(cx, |owner, _| owner.fs().clone())
                        .load(Path::new("/doctor-fetch/source.txt"))
                        .await
                        .expect("Preserved source"),
                    "unrelated project edits"
                );
            }
        }

        fn registry_request() -> AdvisoryRefreshRequest {
            let graph = DependencyGraph::new(
                DependencyProvenance {
                    host: DependencyIdentity::from_fields(&["selected-host"]).expect("Host"),
                    inputs: DependencyIdentity::from_fields(&["current-source"]).expect("Inputs"),
                    provider: DiagnosticText::new("Native resolved fixture"),
                    observed_at: 1000,
                    coverage: DependencyCoverage::Complete,
                    limitations: vec![],
                },
                vec![DependencyInstance {
                    ecosystem: DependencyEcosystem::Cargo,
                    name: "example".into(),
                    version: "1.0.0".into(),
                    classifier: None,
                    source: DependencySource::public_registry(DependencyEcosystem::Cargo),
                    configuration: DependencyIdentity::from_fields(&["selected"])
                        .expect("Configuration"),
                }],
                vec![],
                vec![0],
            )
            .expect("Graph");
            AdvisoryRefreshRequest::RegistryVersions(
                RegistryQuery::for_resolved(&graph, 0).expect("Public selected instance"),
            )
        }

        #[gpui::test]
        async fn doctor_dependency_updates_native_review_freshness_partial_history_and_no_package_effects(
            cx: &mut TestAppContext,
        ) {
            for scenario in ["fresh", "stale", "unknown", "partial"] {
                let (owner, directory) = fixture(cx).await;
                let calls = Arc::new(AtomicUsize::new(0));
                owner.read_with(cx,|owner,_|owner.client().http_client()).as_fake().replace_handler({
                    let calls=calls.clone();
                    move |_,request| {
                        calls.fetch_add(1,Ordering::SeqCst);
                        async move {
                            assert_eq!(request.uri().to_string(),"https://crates.io/api/v1/crates/example/versions");
                            assert_eq!(request.method(),"GET");
                            assert!(request.headers().get("Authorization").is_none() && request.headers().get("Cookie").is_none());
                            assert_eq!(request.extensions().get::<RedirectPolicy>(),Some(&RedirectPolicy::NoFollow));
                            let bytes=serde_json::to_vec(&serde_json::json!({"versions":[{"crate":"example","num":"1.1.0","yanked":false}],"meta":{"total":1,"next_page":if scenario=="partial" { Some("NEVER_FOLLOW_THIS_PAGE") } else {None}}})).expect("Metadata");
                            let mut builder=Response::builder().status(200).header("Content-Length",bytes.len());
                            if scenario!="unknown" { builder=builder.header("Date","Thu, 01 Jan 1970 00:16:40 GMT"); }
                            if scenario=="stale" { builder=builder.header("Age","86400"); }
                            Ok(builder.body(bytes.into())?)
                        }
                    }
                });
                let proposed = cx
                    .update(|cx| {
                        prepare_advisory_refresh(
                            &owner,
                            directory.clone(),
                            1,
                            registry_request(),
                            1000,
                            cx,
                        )
                    })
                    .await
                    .expect("Proposal");
                assert_eq!(proposed.phase(), crate::doctor::DoctorRepairPhase::Proposed);
                assert_eq!(calls.load(Ordering::SeqCst), 0, "Proposal is not consent");
                let history = DoctorHistoryStore::for_test_database(&db::AppDatabase::test_new());
                let source = report(&proposed, "Copper", 1000);
                let (cancel, cancelled) = async_channel::bounded(1);
                let (updates, _progress) = async_channel::bounded(16);
                assert!(
                    cx.update(|cx| launch_advisory_refresh(
                        proposed,
                        &owner,
                        directory.clone(),
                        1,
                        source,
                        DoctorProduct::Copper,
                        history.clone(),
                        1000,
                        cancel,
                        cancelled,
                        updates,
                        cx
                    ))
                    .await
                    .is_err()
                );
                assert_eq!(
                    calls.load(Ordering::SeqCst),
                    0,
                    "Unapproved operation never reaches HTTP"
                );
                let review = approved(cx, &owner, &directory, registry_request(), 1000).await;
                let source = report(&review, "Copper", 1000);
                let (cancel, cancelled) = async_channel::bounded(1);
                let (updates, _progress) = async_channel::bounded(16);
                let result = cx
                    .update(|cx| {
                        launch_advisory_refresh(
                            review,
                            &owner,
                            directory.clone(),
                            1,
                            source,
                            DoctorProduct::Copper,
                            history.clone(),
                            1000,
                            cancel,
                            cancelled,
                            updates,
                            cx,
                        )
                    })
                    .await
                    .expect("Observed request");
                assert_eq!(calls.load(Ordering::SeqCst), 1);
                assert!(result.snapshot.is_none() && result.resolved_commit.is_none());
                let metadata = result
                    .registry_metadata
                    .expect("Transient catalog evidence");
                assert_eq!(metadata.versions.len(), 1);
                assert_eq!(
                    result.review.phase() == crate::doctor::DoctorRepairPhase::Verified,
                    scenario == "fresh"
                );
                let records = history
                    .records(DoctorProduct::Copper, 1000)
                    .await
                    .expect("History");
                assert_eq!(records.len(), 1);
                let serialized = serde_json::to_string(&records).expect("Redacted receipts");
                assert!(
                    !serialized.contains("NEVER_FOLLOW_THIS_PAGE")
                        && !serialized.contains("\"num\"")
                        && !serialized.contains("unrelated project edits")
                );
                assert_eq!(
                    owner
                        .read_with(cx, |owner, _| owner.fs().clone())
                        .load(Path::new("/doctor-fetch/source.txt"))
                        .await
                        .expect("Original source"),
                    "unrelated project edits"
                );
            }
        }

        #[gpui::test]
        async fn doctor_dependency_updates_native_cancel_revocation_and_changed_input(
            cx: &mut TestAppContext,
        ) {
            for scenario in ["cancel", "revocation", "inputs", "duplicate"] {
                let (owner, directory) = fixture(cx).await;
                let (started, began) = async_channel::bounded(1);
                owner
                    .read_with(cx, |owner, _| owner.client().http_client())
                    .as_fake()
                    .replace_handler(move |_, _| {
                        let started = started.clone();
                        async move {
                            started.send(()).await.expect("Dispatched");
                            futures::future::pending::<Result<http_client::Response<AsyncBody>>>()
                                .await
                        }
                    });
                let review = approved(cx, &owner, &directory, registry_request(), 1000).await;
                let source = report(&review, "Copper", 1000);
                let history = DoctorHistoryStore::for_test_database(&db::AppDatabase::test_new());
                let (cancel, cancelled) = async_channel::bounded(1);
                let (updates, _progress) = async_channel::bounded(16);
                let run = cx.update(|cx| {
                    launch_advisory_refresh(
                        review,
                        &owner,
                        directory.clone(),
                        1,
                        source,
                        DoctorProduct::Copper,
                        history.clone(),
                        1000,
                        cancel.clone(),
                        cancelled,
                        updates,
                        cx,
                    )
                });
                began.recv().await.expect("Real owner dispatch");
                if scenario == "duplicate" {
                    let duplicate =
                        approved(cx, &owner, &directory, registry_request(), 1000).await;
                    let source = report(&duplicate, "Copper", 1000);
                    let (duplicate_cancel, duplicate_cancelled) = async_channel::bounded(1);
                    let (updates, _progress) = async_channel::bounded(16);
                    assert!(
                        cx.update(|cx| launch_advisory_refresh(
                            duplicate,
                            &owner,
                            directory.clone(),
                            1,
                            source,
                            DoctorProduct::Copper,
                            history.clone(),
                            1000,
                            duplicate_cancel,
                            duplicate_cancelled,
                            updates,
                            cx
                        ))
                        .await
                        .is_err(),
                        "Equivalent active operation never dispatches twice"
                    );
                    cancel.close();
                }
                if scenario == "cancel" {
                    cancel.close();
                }
                if scenario == "revocation" {
                    cx.update(|cx| {
                        cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                            settings
                                .set_user_settings(r#"{"vim_mode":true}"#, cx)
                                .expect("Changed authority")
                        })
                    });
                }
                if scenario == "inputs" {
                    owner
                        .read_with(cx, |owner, _| owner.fs().clone())
                        .as_fake()
                        .insert_file(
                            Path::new("/doctor-fetch/source.txt"),
                            "changed graph input".into(),
                        )
                        .await;
                    cx.run_until_parked();
                }
                cx.background_executor
                    .advance_clock(Duration::from_millis(200));
                let result = run.await.expect("Stopped operation receipt");
                assert!(result.registry_metadata.is_none());
                assert_ne!(
                    result.review.phase(),
                    crate::doctor::DoctorRepairPhase::Verified
                );
                assert_eq!(
                    history
                        .records(DoctorProduct::Copper, 1001)
                        .await
                        .expect("Durable stop")
                        .len(),
                    1
                );
            }
        }

        #[gpui::test]
        async fn doctor_advisory_fetch_transport_origin_headers_limits_and_no_body_disclosure(
            _cx: &mut TestAppContext,
        ) {
            let requests = Arc::new(AtomicUsize::new(0));
            let client: Arc<dyn HttpClient> = FakeHttpClient::create({
                let requests = requests.clone();
                move |mut request| {
                    requests.fetch_add(1, Ordering::SeqCst);
                    async move {
                        assert_eq!(request.method(), "GET");
                        assert_eq!(
                            request.extensions().get::<RedirectPolicy>(),
                            Some(&RedirectPolicy::NoFollow)
                        );
                        assert_eq!(
                            request
                                .extensions()
                                .get::<http_client::RequestTimeout>()
                                .expect("Deadline")
                                .0,
                            REFRESH_DEADLINE
                        );
                        assert!(
                            request.headers().get("Authorization").is_none()
                                && request.headers().get("Cookie").is_none()
                        );
                        let mut body = Vec::new();
                        request
                            .body_mut()
                            .read_to_end(&mut body)
                            .await
                            .expect("Request body");
                        assert!(body.is_empty());
                        response(b"data".to_vec())
                    }
                }
            });
            let official = AdvisoryArchiveKind::OsvMaven.source_url("");
            assert_eq!(
                read_official_body(&client, &official, 4)
                    .await
                    .expect("Bounded exact body"),
                b"data"
            );
            for url in [
                "http://storage.googleapis.com/osv-vulnerabilities/Maven/all.zip",
                "https://unapproved.example/Maven/all.zip",
                "https://storage.googleapis.com/osv-vulnerabilities/npm/all.zip",
                "https://storage.googleapis.com/osv-vulnerabilities/Maven/all.zip?project=private",
                "https://TOKEN@storage.googleapis.com/osv-vulnerabilities/Maven/all.zip",
                "https://codeload.github.com/RustSec/advisory-db/zip/main",
            ] {
                assert!(read_official_body(&client, url, 4).await.is_err());
            }
            assert_eq!(
                requests.load(Ordering::SeqCst),
                1,
                "Unapproved requests never reach HTTP owner"
            );
            for (status, length, encoding, bytes) in [
                (302, None, None, "PRIVATE_RESPONSE_SECRET"),
                (206, None, None, "PRIVATE_RESPONSE_SECRET"),
                (403, None, None, "PRIVATE_RESPONSE_SECRET"),
                (200, Some("100"), None, "data"),
                (200, Some("3"), None, "data"),
                (200, Some("invalid"), None, "data"),
                (200, None, Some("gzip"), "data"),
                (200, None, None, "too many bytes"),
            ] {
                let client: Arc<dyn HttpClient> = FakeHttpClient::create(move |_| async move {
                    let mut response = Response::builder()
                        .status(status)
                        .header("Location", "https://unapproved.example");
                    if let Some(length) = length {
                        response = response.header("Content-Length", length);
                    }
                    if let Some(encoding) = encoding {
                        response = response.header("Content-Encoding", encoding);
                    }
                    Ok(response.body(AsyncBody::from(bytes))?)
                });
                let error = read_official_body(&client, &official, 4)
                    .await
                    .expect_err("Incomplete body rejected");
                assert!(!error.to_string().contains("PRIVATE_RESPONSE_SECRET"));
            }
            let client: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async move {
                Ok(Response::builder()
                    .status(200)
                    .body(AsyncBody::from("data"))?)
            });
            assert_eq!(
                read_official_body(&client, &official, 4)
                    .await
                    .expect("Unknown length uses actual cap"),
                b"data"
            );
        }

        #[gpui::test]
        async fn doctor_advisory_fetch_source_publication_age_is_not_refreshed_by_download(
            cx: &mut TestAppContext,
        ) {
            let official = AdvisoryArchiveKind::OsvMaven.source_url("");
            let client: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async move {
                Ok(Response::builder()
                    .status(200)
                    .header("Last-Modified", "Fri, 02 Oct 2026 23:32:40 GMT")
                    .header("x-goog-generation", "123456789")
                    .body(AsyncBody::from("data"))?)
            });
            let body = read_official_response(&client, &official, 4)
                .await
                .expect("Official publication metadata");
            assert_eq!(
                body.source_revision.as_deref(),
                Some("generation:123456789")
            );
            let published = body.source_modified_at.expect("Publication timestamp");
            let now = published + 48 * 60 * 60;
            let (project, _) = fixture(cx).await;
            let cache = project
                .read_with(cx, |project, _| AdvisoryCache::for_local_project(project))
                .expect("Existing owner");
            let mut snapshot = cache
                .validate_archives(
                    AdvisoryDataset::OsvMaven,
                    vec![
                        AdvisoryArchive {
                            kind: AdvisoryArchiveKind::OsvMaven,
                            revision: "generation:123456789".into(),
                            bytes: zip(AdvisoryArchiveKind::OsvMaven).await,
                        },
                        AdvisoryArchive {
                            kind: AdvisoryArchiveKind::OsvWithdrawals,
                            revision: "generation:123456788".into(),
                            bytes: zip(AdvisoryArchiveKind::OsvWithdrawals).await,
                        },
                    ],
                    now,
                )
                .await
                .expect("Complete source fixture");
            assert_eq!(
                snapshot.upstream_freshness(now),
                None,
                "Missing publication provenance remains unknown"
            );
            for source in &mut snapshot.sources {
                source.source_modified_at = Some(published);
            }
            assert_eq!(snapshot.freshness(now), AdvisoryFreshness::Fresh);
            assert_eq!(
                snapshot.upstream_freshness(now),
                Some(AdvisoryFreshness::Stale),
                "A new download cannot hide an old upstream publication"
            );
            for source in &mut snapshot.sources {
                source.source_modified_at = Some(now);
            }
            assert_eq!(
                snapshot.upstream_freshness(now),
                Some(AdvisoryFreshness::Fresh)
            );
            snapshot
                .sources
                .first_mut()
                .expect("Source fixture")
                .source_modified_at = Some(now + 1);
            assert_eq!(
                snapshot.upstream_freshness(now),
                Some(AdvisoryFreshness::ClockUncertain)
            );
            for (generation, modified) in [
                ("not-a-generation", "Fri, 02 Oct 2026 23:32:40 GMT"),
                ("123", "invalid publication"),
            ] {
                let client: Arc<dyn HttpClient> = FakeHttpClient::create(move |_| async move {
                    Ok(Response::builder()
                        .status(200)
                        .header("x-goog-generation", generation)
                        .header("Last-Modified", modified)
                        .body(AsyncBody::from("data"))?)
                });
                assert!(read_official_response(&client, &official, 4).await.is_err());
            }
            let client: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async move {
                Ok(Response::builder()
                    .status(200)
                    .header("x-goog-generation", "123")
                    .header("x-goog-generation", "456")
                    .body(AsyncBody::from("data"))?)
            });
            assert!(
                read_official_response(&client, &official, 4).await.is_err(),
                "Ambiguous source identity is rejected"
            );
        }

        #[gpui::test]
        async fn doctor_advisory_fetch_native_review_recheck_history_and_no_implicit_requests(
            cx: &mut TestAppContext,
        ) {
            let (owner, directory) = fixture(cx).await;
            let maven = zip(AdvisoryArchiveKind::OsvMaven).await;
            let withdrawn = zip(AdvisoryArchiveKind::OsvWithdrawals).await;
            let calls = Arc::new(AtomicUsize::new(0));
            owner
                .read_with(cx, |owner, _| owner.client().http_client())
                .as_fake()
                .replace_handler({
                    let calls = calls.clone();
                    move |_, request| {
                        calls.fetch_add(1, Ordering::SeqCst);
                        let bytes = if request.uri().to_string()
                            == AdvisoryArchiveKind::OsvMaven.source_url("")
                        {
                            maven.clone()
                        } else {
                            assert_eq!(
                                request.uri().to_string(),
                                AdvisoryArchiveKind::OsvWithdrawals.source_url("")
                            );
                            withdrawn.clone()
                        };
                        async move {
                            assert!(request.headers().get("Authorization").is_none());
                            response(bytes)
                        }
                    }
                });
            let proposal = cx
                .update(|cx| {
                    prepare_advisory_refresh(
                        &owner,
                        directory.clone(),
                        1,
                        AdvisoryRefreshRequest::OsvMaven,
                        100,
                        cx,
                    )
                })
                .await
                .expect("Draft without requests");
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            assert_eq!(
                proposal
                    .action()
                    .exact_preview()
                    .expect("Exact preview")
                    .len(),
                5
            );
            let database = db::AppDatabase::test_new();
            let history = DoctorHistoryStore::for_test_database(&database);
            let (cancel, cancelled) = async_channel::bounded(1);
            let (updates, _progress) = async_channel::bounded(16);
            let source = report(&proposal, "Groovy", 100);
            assert!(
                cx.update(|cx| launch_advisory_refresh(
                    proposal,
                    &owner,
                    directory.clone(),
                    1,
                    source,
                    DoctorProduct::Groovy,
                    history.clone(),
                    100,
                    cancel,
                    cancelled,
                    updates,
                    cx
                ))
                .await
                .is_err()
            );
            assert_eq!(
                calls.load(Ordering::SeqCst),
                0,
                "A model/proposal cannot approve itself"
            );
            let review = approved(
                cx,
                &owner,
                &directory,
                AdvisoryRefreshRequest::OsvMaven,
                100,
            )
            .await;
            let source = report(&review, "Groovy", 100);
            let (cancel, cancelled) = async_channel::bounded(1);
            let (updates, progress) = async_channel::bounded(16);
            let result = cx
                .update(|cx| {
                    launch_advisory_refresh(
                        review,
                        &owner,
                        directory.clone(),
                        1,
                        source,
                        DoctorProduct::Groovy,
                        history.clone(),
                        100,
                        cancel,
                        cancelled,
                        updates,
                        cx,
                    )
                })
                .await
                .expect("Observed native HTTP/cache flow");
            assert_eq!(calls.load(Ordering::SeqCst), 2);
            assert_eq!(
                result.review.phase(),
                crate::doctor::DoctorRepairPhase::Verified
            );
            assert!(
                result
                    .review
                    .operation_detail()
                    .expect("Availability conclusion")
                    .as_str()
                    .contains("availability only")
            );
            assert!(result.resolved_commit.is_none());
            assert_eq!(
                result
                    .snapshot
                    .expect("Rechecked complete source snapshot")
                    .sources
                    .len(),
                2
            );
            let mut stages = Vec::new();
            while let Ok(progress) = progress.try_recv() {
                stages.push(progress.stage);
            }
            assert!(
                stages
                    .iter()
                    .any(|stage| stage.as_str().contains("withdrawal"))
            );
            let records = history
                .records(DoctorProduct::Groovy, 100)
                .await
                .expect("Redacted durable history");
            assert_eq!(records.len(), 1);
            let serialized = serde_json::to_string(&records).expect("History serialization");
            assert!(
                !serialized.contains("RAW_PUBLIC_ADVISORY_NOT_HISTORY")
                    && !serialized.contains("org.example:library")
                    && !serialized.contains("unrelated project edits")
            );
            assert_eq!(
                owner
                    .read_with(cx, |owner, _| owner.fs().clone())
                    .load(Path::new("/doctor-fetch/source.txt"))
                    .await
                    .expect("Unrelated source"),
                "unrelated project edits"
            );
        }

        #[gpui::test]
        async fn doctor_advisory_fetch_revision_lookup_requires_separate_pinned_download(
            cx: &mut TestAppContext,
        ) {
            let (owner, directory) = fixture(cx).await;
            let calls = Arc::new(AtomicUsize::new(0));
            owner
                .read_with(cx, |owner, _| owner.client().http_client())
                .as_fake()
                .replace_handler({
                    let calls = calls.clone();
                    move |_, request| {
                        calls.fetch_add(1, Ordering::SeqCst);
                        async move {
                            assert_eq!(
                                request.uri().to_string(),
                                "https://api.github.com/repos/RustSec/advisory-db/commits/main"
                            );
                            response(format!("{{\"sha\":\"{}\"}}", "a".repeat(40)).into_bytes())
                        }
                    }
                });
            let review = approved(
                cx,
                &owner,
                &directory,
                AdvisoryRefreshRequest::ResolveRustSecRevision,
                100,
            )
            .await;
            let source = report(&review, "Copper", 100);
            let history = DoctorHistoryStore::for_test_database(&db::AppDatabase::test_new());
            let (cancel, cancelled) = async_channel::bounded(1);
            let (updates, _progress) = async_channel::bounded(16);
            let result = cx
                .update(|cx| {
                    launch_advisory_refresh(
                        review,
                        &owner,
                        directory.clone(),
                        1,
                        source,
                        DoctorProduct::Copper,
                        history,
                        100,
                        cancel,
                        cancelled,
                        updates,
                        cx,
                    )
                })
                .await
                .expect("Metadata lookup only");
            let commit = result.resolved_commit.expect("Exact observed commit");
            assert_eq!(commit, "a".repeat(40));
            assert!(result.snapshot.is_none());
            let proposal = cx
                .update(|cx| {
                    prepare_advisory_refresh(
                        &owner,
                        directory,
                        1,
                        AdvisoryRefreshRequest::RustSec {
                            commit: commit.clone(),
                        },
                        100,
                        cx,
                    )
                })
                .await
                .expect("Separate concrete archive proposal");
            assert_eq!(proposal.phase(), crate::doctor::DoctorRepairPhase::Proposed);
            assert_eq!(
                proposal
                    .action()
                    .request()
                    .exact_requests()
                    .expect("Pinned request"),
                vec![AdvisoryArchiveKind::RustSec.source_url(&commit)]
            );
            assert_eq!(
                calls.load(Ordering::SeqCst),
                1,
                "No implicit archive or branch-latest request"
            );
        }

        #[gpui::test]
        async fn doctor_advisory_fetch_cancel_deadline_revocation_and_interrupted_history(
            cx: &mut TestAppContext,
        ) {
            for scenario in ["cancel", "deadline", "revocation", "drop"] {
                let (owner, directory) = fixture(cx).await;
                let (started, began) = async_channel::bounded(1);
                let calls = Arc::new(AtomicUsize::new(0));
                owner.read_with(cx, |owner, _| owner.client().http_client()).as_fake().replace_handler({
                    let calls = calls.clone();
                    move |_, _| {
                        calls.fetch_add(1, Ordering::SeqCst);
                        let started = started.clone();
                        async move { started.send(()).await.expect("Started request"); futures::future::pending::<Result<http_client::Response<AsyncBody>>>().await }
                    }
                });
                let review = approved(
                    cx,
                    &owner,
                    &directory,
                    AdvisoryRefreshRequest::OsvMaven,
                    100,
                )
                .await;
                let source = report(&review, "Groovy", 100);
                let database = db::AppDatabase::test_new();
                let history = DoctorHistoryStore::for_test_database(&database);
                let (cancel, cancelled) = async_channel::bounded(1);
                let (updates, _progress) = async_channel::bounded(16);
                let run = cx.update(|cx| {
                    launch_advisory_refresh(
                        review,
                        &owner,
                        directory.clone(),
                        1,
                        source,
                        DoctorProduct::Groovy,
                        history.clone(),
                        100,
                        cancel.clone(),
                        cancelled,
                        updates,
                        cx,
                    )
                });
                began
                    .recv()
                    .await
                    .expect("Actual reviewed HTTP owner dispatch");
                if scenario == "drop" {
                    drop(run);
                    cx.run_until_parked();
                } else {
                    if scenario == "cancel" {
                        cancel.close();
                    }
                    if scenario == "deadline" {
                        cx.background_executor
                            .advance_clock(Duration::from_secs(61));
                    }
                    if scenario == "revocation" {
                        cx.update(|cx| {
                            cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                                settings
                                    .set_user_settings(r#"{"vim_mode":true}"#, cx)
                                    .expect("Revoked reviewed settings")
                            })
                        });
                        cx.background_executor
                            .advance_clock(Duration::from_millis(200));
                    }
                    let result = run.await.expect("Observed stopped operation");
                    assert!(result.snapshot.is_none() && result.resolved_commit.is_none());
                    assert_ne!(
                        result.review.phase(),
                        crate::doctor::DoctorRepairPhase::Verified
                    );
                }
                let cache = cx
                    .update(|cx| AdvisoryCache::for_local_project(owner.read(cx)))
                    .expect("Same selected cache owner");
                assert!(
                    cache
                        .load(AdvisoryDataset::OsvMaven, 101)
                        .await
                        .expect("No incomplete replacement")
                        .snapshot
                        .is_none()
                );
                let reopened = DoctorHistoryStore::for_test_database(&database);
                let records = reopened
                    .records(DoctorProduct::Groovy, 101)
                    .await
                    .expect("Durable stop/interruption");
                assert_eq!(records.len(), 1);
                assert_ne!(
                    records.first().expect("Receipt").outcome,
                    crate::doctor::ReceiptOutcome::Completed
                );
                assert_eq!(
                    calls.load(Ordering::SeqCst),
                    1,
                    "History never replays HTTP"
                );
            }
        }

        #[gpui::test]
        async fn doctor_advisory_fetch_failed_withdrawal_preserves_previous_and_late_cache_change(
            cx: &mut TestAppContext,
        ) {
            let (owner, directory) = fixture(cx).await;
            let maven = zip(AdvisoryArchiveKind::OsvMaven).await;
            let withdrawn = zip(AdvisoryArchiveKind::OsvWithdrawals).await;
            let cache = cx
                .update(|cx| AdvisoryCache::for_local_project(owner.read(cx)))
                .expect("Selected cache");
            let previous = cache
                .validate_archives(
                    AdvisoryDataset::OsvMaven,
                    vec![
                        AdvisoryArchive {
                            kind: AdvisoryArchiveKind::OsvMaven,
                            revision: "original-1".into(),
                            bytes: maven.clone(),
                        },
                        AdvisoryArchive {
                            kind: AdvisoryArchiveKind::OsvWithdrawals,
                            revision: "original-2".into(),
                            bytes: withdrawn,
                        },
                    ],
                    100,
                )
                .await
                .expect("Prior valid public fixture");
            let revision = cache
                .replace(previous, None, 100)
                .await
                .expect("Prior cache fixture");
            let calls = Arc::new(AtomicUsize::new(0));
            owner
                .read_with(cx, |owner, _| owner.client().http_client())
                .as_fake()
                .replace_handler({
                    let calls = calls.clone();
                    move |_, request| {
                        calls.fetch_add(1, Ordering::SeqCst);
                        let bytes = maven.clone();
                        async move {
                            if request.uri().to_string()
                                == AdvisoryArchiveKind::OsvMaven.source_url("")
                            {
                                response(bytes)
                            } else {
                                Ok(Response::builder()
                                    .status(404)
                                    .body(AsyncBody::from("PRIVATE_RESPONSE_SECRET"))?)
                            }
                        }
                    }
                });
            let review = approved(
                cx,
                &owner,
                &directory,
                AdvisoryRefreshRequest::OsvMaven,
                200,
            )
            .await;
            let source = report(&review, "Groovy", 200);
            let history = DoctorHistoryStore::for_test_database(&db::AppDatabase::test_new());
            let (cancel, cancelled) = async_channel::bounded(1);
            let (updates, _progress) = async_channel::bounded(16);
            let result = cx
                .update(|cx| {
                    launch_advisory_refresh(
                        review,
                        &owner,
                        directory.clone(),
                        1,
                        source,
                        DoctorProduct::Groovy,
                        history,
                        200,
                        cancel,
                        cancelled,
                        updates,
                        cx,
                    )
                })
                .await
                .expect("Incomplete withdrawal result");
            assert_eq!(calls.load(Ordering::SeqCst), 2);
            assert!(result.snapshot.is_none());
            assert_ne!(
                result.review.phase(),
                crate::doctor::DoctorRepairPhase::Verified
            );
            let retained = cache
                .load(AdvisoryDataset::OsvMaven, 200)
                .await
                .expect("Previous data retained");
            assert!(retained.revision == Some(revision));
            assert_eq!(
                retained.snapshot.expect("Old valid snapshot").fetched_at,
                100
            );
            assert!(
                !result
                    .review
                    .operation_detail()
                    .expect("Generic result")
                    .as_str()
                    .contains("PRIVATE_RESPONSE_SECRET")
            );
            let review = approved(
                cx,
                &owner,
                &directory,
                AdvisoryRefreshRequest::OsvMaven,
                300,
            )
            .await;
            cache
                .clear()
                .await
                .expect("Independent explicitly reviewed clear");
            assert!(
                review
                    .action()
                    .cache
                    .load(AdvisoryDataset::OsvMaven, 300)
                    .await
                    .expect("Current cache absence")
                    .revision
                    .is_none()
            );
            let candidate = review
                .action()
                .cache
                .validate_archives(
                    AdvisoryDataset::OsvMaven,
                    vec![
                        AdvisoryArchive {
                            kind: AdvisoryArchiveKind::OsvMaven,
                            revision: "replacement-1".into(),
                            bytes: zip(AdvisoryArchiveKind::OsvMaven).await,
                        },
                        AdvisoryArchive {
                            kind: AdvisoryArchiveKind::OsvWithdrawals,
                            revision: "replacement-2".into(),
                            bytes: zip(AdvisoryArchiveKind::OsvWithdrawals).await,
                        },
                    ],
                    300,
                )
                .await
                .expect("Candidate");
            assert!(
                review
                    .action()
                    .cache
                    .replace_when_authorized(
                        candidate,
                        review.action().expected_revision.as_ref(),
                        300,
                        || Ok(())
                    )
                    .await
                    .is_err(),
                "Stale expected cache cannot undo explicit clear"
            );
        }
    }
}

#[cfg(feature = "jvm-tools")]
pub use advisory_match::compare_maven_versions;
#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
pub use advisory_match::{AdvisoryFinding, AdvisoryScan, match_advisories};

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
mod advisory_match {
    use super::*;
    use std::cmp::Ordering;

    const MAX_FINDINGS: usize = 512;
    const MAX_MATCH_METADATA: usize = 1024 * 1024;

    pub struct AdvisoryFinding {
        pub node: usize,
        pub evidence: DependencyAdvisoryEvidence,
        pub source: AdvisoryArchiveReceipt,
        pub source_url: String,
        pub modified_at: Option<u64>,
        pub withdrawn_at: Option<u64>,
        pub conditions: Vec<DiagnosticText>,
        pub paths: DependencyIntroducingPaths,
    }

    pub struct AdvisoryScan {
        pub graph_revision: DependencyIdentity,
        pub advisory_revision: DependencyIdentity,
        pub fetched_at: u64,
        pub observed_at: u64,
        pub freshness: AdvisoryFreshness,
        pub upstream_freshness: Option<AdvisoryFreshness>,
        pub coverage: DependencyCoverage,
        pub findings: Vec<AdvisoryFinding>,
        pub limitations: Vec<DiagnosticText>,
    }

    impl AdvisoryScan {
        pub fn conclusion(&self) -> DiagnosticText {
            let affected = self
                .findings
                .iter()
                .filter(|finding| {
                    finding.evidence.state == DependencyAdvisoryState::KnownAffectedDependency
                })
                .count();
            DiagnosticText::new(format!(
                "{affected} known affected resolved dependency observations; coverage {:?}. No findings is not proof of security. Application exploitability remains unknown; withdrawn, informational, source-uncertain and unsupported observations retain their own states",
                self.coverage
            ))
        }

        pub(super) fn incomplete(&mut self, reason: &str) {
            if self.coverage == DependencyCoverage::Complete {
                self.coverage = DependencyCoverage::Partial;
            }
            let reason = DiagnosticText::new(reason);
            if self.limitations.len() < 32 && !self.limitations.contains(&reason) {
                self.limitations.push(reason);
            }
        }
    }

    impl DependencyGraph {
        pub fn identity(&self) -> DependencyIdentity {
            let mut hash = Sha256::new();
            let mut field = |value: &str| {
                hash.update((value.len() as u64).to_le_bytes());
                hash.update(value.as_bytes());
            };
            field(&self.provenance.host.hex());
            field(&self.provenance.inputs.hex());
            field(self.provenance.provider.as_str());
            field(&format!("{:?}", self.provenance.coverage));
            for node in &self.nodes {
                field(&format!("{node:?}"));
            }
            for edge in &self.edges {
                field(&format!("{edge:?}"));
            }
            field(&format!("{:?}", self.roots));
            DependencyIdentity(hash.finalize().into())
        }
    }

    fn fields(value: Option<&serde_json::Value>) -> Result<Vec<String>> {
        let Some(value) = value else {
            return Ok(Vec::new());
        };
        let values = value
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Unsupported advisory list"))?;
        ensure!(
            values.len() <= 32,
            "Advisory metadata list exceeds its bound"
        );
        values
            .iter()
            .map(|value| {
                let text = value
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("Invalid advisory list item"))?;
                validate_field(text)?;
                Ok(text.to_owned())
            })
            .collect()
    }

    fn timestamp(value: Option<&serde_json::Value>) -> Result<Option<u64>> {
        value
            .map(|value| {
                let value = value
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("Invalid advisory timestamp"))?;
                let value = time::OffsetDateTime::parse(
                    value,
                    &time::format_description::well_known::Rfc3339,
                )
                .map_err(|_| anyhow::anyhow!("Unsupported advisory timestamp"))?;
                u64::try_from(value.unix_timestamp())
                    .map_err(|_| anyhow::anyhow!("Invalid advisory timestamp"))
            })
            .transpose()
    }

    fn text_list(values: Vec<String>) -> Vec<DiagnosticText> {
        values.into_iter().map(DiagnosticText::new).collect()
    }

    #[derive(Default)]
    struct Interpretation {
        aliases: Vec<String>,
        severity: Option<String>,
        affected: String,
        fixed: Vec<String>,
        modified_at: Option<u64>,
        withdrawn_at: Option<u64>,
        conditions: Vec<String>,
        state: Option<DependencyAdvisoryState>,
    }

    fn osv_interpret(
        value: &serde_json::Value,
        affected: &serde_json::Value,
        version: &str,
    ) -> Result<Interpretation> {
        let schema = value
            .get("schema_version")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("1.0.0");
        let schema = semver::Version::parse(schema)
            .map_err(|_| anyhow::anyhow!("Invalid OSV schema version"))?;
        ensure!(
            schema.major == 1 && schema.minor <= 9,
            "Unsupported OSV schema version"
        );
        let aliases = fields(value.get("aliases"))?;
        let modified_at = timestamp(value.get("modified"))?;
        let withdrawn_at = timestamp(value.get("withdrawn"))?;
        let mut result = Interpretation {
            aliases,
            modified_at,
            withdrawn_at,
            ..Default::default()
        };
        let severity = affected.get("severity").or_else(|| value.get("severity"));
        if let Some(values) = severity {
            let values = values
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("Invalid OSV severity"))?;
            ensure!(values.len() <= 8, "OSV severity bound exceeded");
            let mut scores = Vec::new();
            for value in values {
                let kind = value
                    .get("type")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| anyhow::anyhow!("Severity type is missing"))?;
                let score = value
                    .get("score")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| anyhow::anyhow!("Severity score is missing"))?;
                validate_field(kind)?;
                validate_field(score)?;
                scores.push(format!("{kind}: {score}"));
            }
            if !scores.is_empty() {
                result.severity = Some(scores.join("; "));
            }
        }
        if result.severity.is_none() {
            if let Some(severity) = value
                .get("database_specific")
                .and_then(|value| value.get("severity"))
                .and_then(serde_json::Value::as_str)
            {
                validate_field(severity)?;
                result.severity = Some(format!("Source severity: {severity}"));
            }
        }
        let versions = affected
            .get("versions")
            .map(|value| {
                let versions = value
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("Invalid OSV version list"))?;
                ensure!(versions.len() <= 65534, "OSV version list bound exceeded");
                versions
                    .iter()
                    .map(|value| {
                        let version = value
                            .as_str()
                            .ok_or_else(|| anyhow::anyhow!("Invalid OSV version"))?;
                        validate_field(version)?;
                        Ok(version)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        let explicitly_affected = versions.contains(&version);
        let mut ranges_affected = false;
        let mut unsupported = false;
        if let Some(ranges) = affected.get("ranges") {
            let ranges = ranges
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("Invalid OSV ranges"))?;
            ensure!(ranges.len() <= 128, "OSV range count exceeded");
            for range in ranges {
                match osv_range(range, version) {
                    Ok((matches, fixed)) => {
                        ranges_affected |= matches;
                        for version in fixed {
                            if !result.fixed.contains(&version) && result.fixed.len() < 32 {
                                result.fixed.push(version);
                            }
                        }
                    }
                    Err(_) => unsupported = true,
                }
            }
        }
        let preview = affected
            .get("ranges")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .take(8)
            .map(|range| {
                let kind = range
                    .get("type")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown");
                let events = range
                    .get("events")
                    .and_then(serde_json::Value::as_array)
                    .into_iter()
                    .flatten()
                    .take(8)
                    .filter_map(|event| event.as_object())
                    .flat_map(|event| event.iter())
                    .filter_map(|(kind, value)| {
                        value.as_str().map(|value| format!("{kind} {value}"))
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{kind}: {events}")
            })
            .collect::<Vec<_>>()
            .join("; ");
        result.affected = format!(
            "Explicit versions [{}] ({} total); ranges [{}]; bounded preview, see source for complete versions. Interpretation {}; selected resolved version {}",
            versions
                .iter()
                .take(8)
                .copied()
                .collect::<Vec<_>>()
                .join(", "),
            versions.len(),
            preview,
            if unsupported {
                "incomplete"
            } else {
                "supported"
            },
            if explicitly_affected || ranges_affected {
                "matches"
            } else {
                "does not match supported evidence"
            }
        );
        result.state = Some(if withdrawn_at.is_some() {
            DependencyAdvisoryState::Withdrawn
        } else if explicitly_affected || ranges_affected {
            DependencyAdvisoryState::KnownAffectedDependency
        } else if unsupported
            || (versions.is_empty()
                && !affected
                    .get("ranges")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|ranges| !ranges.is_empty()))
        {
            DependencyAdvisoryState::UnknownVersionSemantics
        } else {
            DependencyAdvisoryState::NotAffectedByThisAdvisory
        });
        if unsupported {
            result.conditions.push("At least one advisory range has unsupported or malformed semantics; exact positive evidence is retained, but negative coverage is incomplete".into());
        }
        Ok(result)
    }

    fn osv_range(value: &serde_json::Value, version: &str) -> Result<(bool, Vec<String>)> {
        let kind = value
            .get("type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("OSV range type is missing"))?;
        ensure!(
            matches!(kind, "SEMVER" | "ECOSYSTEM"),
            "Unsupported OSV range type"
        );
        let events = value
            .get("events")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("OSV events missing"))?;
        ensure!(
            !events.is_empty() && events.len() <= 128,
            "OSV event count invalid"
        );
        let compare =
            |left: &str, right: &str| -> Result<Ordering> {
                match kind {
                    "SEMVER" => Ok(semver::Version::parse(left)?
                        .cmp_precedence(&semver::Version::parse(right)?)),
                    "ECOSYSTEM" => {
                        #[cfg(feature = "jvm-tools")]
                        {
                            compare_maven_versions(left, right)
                        }
                        #[cfg(not(feature = "jvm-tools"))]
                        {
                            anyhow::bail!("Maven range support is disabled");
                        }
                    }
                    _ => anyhow::bail!("Unsupported OSV range type"),
                }
            };
        compare(version, version)?;
        let mut start: Option<&str> = None;
        let mut active = false;
        let mut matches = false;
        let mut fixed = Vec::new();
        let mut previous: Option<&str> = None;
        for (index, event) in events.iter().enumerate() {
            let event = event
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("Invalid OSV event"))?;
            ensure!(event.len() == 1, "Ambiguous OSV event");
            let (key, boundary) = event
                .iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("Empty OSV event"))?;
            let boundary = boundary
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Invalid OSV boundary"))?;
            validate_field(boundary)?;
            ensure!(
                boundary != "0" || key == "introduced",
                "Invalid OSV zero boundary"
            );
            let infinite = key == "limit" && boundary == "*";
            ensure!(
                boundary != "0" || index == 0,
                "Unordered OSV zero introduction"
            );
            ensure!(
                !infinite || index + 1 == events.len(),
                "OSV infinity must close the final interval"
            );
            if boundary != "0" && !infinite {
                compare(boundary, boundary)?;
                if let Some(previous) = previous {
                    ensure!(
                        compare(previous, boundary)? != Ordering::Greater,
                        "Unordered OSV events"
                    );
                }
                previous = Some(boundary);
            }
            match key.as_str() {
                "introduced" => {
                    ensure!(!active, "Overlapping OSV introductions");
                    active = true;
                    start = if boundary == "0" {
                        None
                    } else {
                        Some(boundary)
                    };
                }
                "fixed" | "last_affected" | "limit" => {
                    ensure!(active, "OSV closing event without introduction");
                    let above_start = start
                        .map(|start| compare(version, start).map(|order| order != Ordering::Less))
                        .transpose()?
                        .unwrap_or(true);
                    let below_end = infinite
                        || (if key == "last_affected" {
                            compare(version, boundary)? != Ordering::Greater
                        } else {
                            compare(version, boundary)? == Ordering::Less
                        });
                    matches |= above_start && below_end;
                    if key == "fixed" {
                        fixed.push(boundary.to_owned());
                    }
                    active = false;
                    start = None;
                }
                _ => anyhow::bail!("Unsupported OSV event"),
            }
        }
        if active {
            matches |= start
                .map(|start| compare(version, start).map(|order| order != Ordering::Less))
                .transpose()?
                .unwrap_or(true);
        }
        Ok((matches, fixed))
    }

    #[cfg(feature = "rust-tools")]
    #[derive(Clone)]
    struct RustInterval {
        lower: Option<(semver::Version, bool)>,
        upper: Option<(semver::Version, bool)>,
    }

    #[cfg(feature = "rust-tools")]
    impl RustInterval {
        fn contains(&self, version: &semver::Version) -> bool {
            self.lower.as_ref().is_none_or(|(bound, inclusive)| {
                match version.cmp_precedence(bound) {
                    Ordering::Greater => true,
                    Ordering::Equal => *inclusive,
                    Ordering::Less => false,
                }
            }) && self.upper.as_ref().is_none_or(|(bound, inclusive)| {
                match version.cmp_precedence(bound) {
                    Ordering::Less => true,
                    Ordering::Equal => *inclusive,
                    Ordering::Greater => false,
                }
            })
        }

        fn parse(value: &str) -> Result<Self> {
            use semver::{Op, Prerelease, Version, VersionReq};
            let requirement = VersionReq::parse(value)?;
            ensure!(
                requirement.comparators.len() <= 2,
                "Unsupported RustSec selector complexity"
            );
            let mut interval = Self {
                lower: None,
                upper: None,
            };
            for comparator in &requirement.comparators {
                let mut bound = Version::new(
                    comparator.major,
                    comparator.minor.unwrap_or(0),
                    comparator.patch.unwrap_or(0),
                );
                bound.pre = comparator.pre.clone();
                match comparator.op {
                    Op::Greater | Op::GreaterEq => {
                        ensure!(interval.lower.is_none(), "Duplicate RustSec lower bound");
                        interval.lower = Some((bound, comparator.op == Op::GreaterEq));
                    }
                    Op::Less | Op::LessEq => {
                        ensure!(interval.upper.is_none(), "Duplicate RustSec upper bound");
                        interval.upper = Some((bound, comparator.op == Op::LessEq));
                    }
                    Op::Exact => {
                        ensure!(
                            requirement.comparators.len() == 1,
                            "Combined exact RustSec selector"
                        );
                        interval.lower = Some((bound.clone(), true));
                        interval.upper = Some((bound, true));
                    }
                    Op::Caret | Op::Tilde => {
                        ensure!(
                            requirement.comparators.len() == 1,
                            "Combined RustSec caret/tilde selector"
                        );
                        let increment = |value: u64| {
                            value
                                .checked_add(1)
                                .ok_or_else(|| anyhow::anyhow!("RustSec selector overflow"))
                        };
                        let mut end = match comparator.op {
                            Op::Caret if bound.major > 0 || comparator.minor.is_none() => {
                                Version::new(increment(bound.major)?, 0, 0)
                            }
                            Op::Caret if bound.minor == 0 && comparator.patch.is_some() => {
                                Version::new(0, 0, increment(bound.patch)?)
                            }
                            Op::Caret => Version::new(0, increment(bound.minor)?, 0),
                            Op::Tilde if comparator.minor.is_none() => {
                                Version::new(increment(bound.major)?, 0, 0)
                            }
                            Op::Tilde => Version::new(bound.major, increment(bound.minor)?, 0),
                            _ => anyhow::bail!("Unsupported RustSec selector"),
                        };
                        end.pre = Prerelease::new("0")?;
                        interval.lower = Some((bound, true));
                        interval.upper = Some((end, false));
                    }
                    _ => anyhow::bail!("Unsupported RustSec selector"),
                }
            }
            if let (Some((lower, lower_inclusive)), Some((upper, upper_inclusive))) =
                (&interval.lower, &interval.upper)
            {
                ensure!(
                    lower.cmp_precedence(upper) == Ordering::Less
                        || (lower.cmp_precedence(upper) == Ordering::Equal
                            && *lower_inclusive
                            && *upper_inclusive),
                    "Invalid RustSec interval"
                );
            }
            Ok(interval)
        }

        fn overlaps(&self, other: &Self) -> bool {
            let separated = |upper: &Option<(semver::Version, bool)>,
                             lower: &Option<(semver::Version, bool)>| {
                match (upper, lower) {
                    (Some((end, end_inclusive)), Some((start, start_inclusive))) => {
                        end.cmp_precedence(start) == Ordering::Less
                            || (end.cmp_precedence(start) == Ordering::Equal
                                && !(*end_inclusive && *start_inclusive))
                    }
                    _ => false,
                }
            };
            !separated(&self.upper, &other.lower) && !separated(&other.upper, &self.lower)
        }
    }

    #[cfg(feature = "rust-tools")]
    fn rust_interpret(value: &toml::Value, version: &str) -> Result<Interpretation> {
        let advisory = value
            .get("advisory")
            .ok_or_else(|| anyhow::anyhow!("RustSec advisory missing"))?;
        let version = semver::Version::parse(version)?;
        let list = |value: Option<&toml::Value>| -> Result<Vec<String>> {
            let Some(value) = value else {
                return Ok(Vec::new());
            };
            let values = value
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("Invalid RustSec list"))?;
            ensure!(values.len() <= 32, "RustSec list exceeds its bound");
            values
                .iter()
                .map(|value| {
                    let value = value
                        .as_str()
                        .ok_or_else(|| anyhow::anyhow!("Invalid RustSec item"))?;
                    validate_field(value)?;
                    Ok(value.to_owned())
                })
                .collect()
        };
        let date = |value: Option<&toml::Value>| -> Result<Option<u64>> {
            value
                .map(|value| {
                    let text = match value {
                        toml::Value::Datetime(value) => value.to_string(),
                        toml::Value::String(value) => value.clone(),
                        _ => anyhow::bail!("Unsupported RustSec date"),
                    };
                    let format = time::format_description::parse("[year]-[month]-[day]")?;
                    let date = time::Date::parse(&text, &format)
                        .map_err(|_| anyhow::anyhow!("Unsupported RustSec date"))?;
                    u64::try_from(date.midnight().assume_utc().unix_timestamp())
                        .map_err(|_| anyhow::anyhow!("Invalid RustSec date"))
                })
                .transpose()
        };
        let patched = list(value.get("versions").and_then(|value| value.get("patched")))?;
        let unaffected = list(
            value
                .get("versions")
                .and_then(|value| value.get("unaffected")),
        )?;
        let intervals = patched
            .iter()
            .chain(&unaffected)
            .map(|value| RustInterval::parse(value))
            .collect::<Result<Vec<_>>>()?;
        for (index, interval) in intervals.iter().enumerate() {
            for other in intervals.iter().skip(index + 1) {
                ensure!(
                    !interval.overlaps(other),
                    "Overlapping RustSec unaffected/patched selectors"
                );
            }
        }
        let state = if advisory.get("withdrawn").is_some() {
            DependencyAdvisoryState::Withdrawn
        } else if advisory.get("informational").is_some() {
            DependencyAdvisoryState::Informational
        } else if intervals.iter().any(|interval| interval.contains(&version)) {
            DependencyAdvisoryState::NotAffectedByThisAdvisory
        } else {
            DependencyAdvisoryState::KnownAffectedDependency
        };
        let mut conditions = Vec::new();
        if let Some(affected) = value.get("affected").and_then(toml::Value::as_table) {
            for name in ["arch", "os", "functions"] {
                if affected.contains_key(name) {
                    conditions.push(format!("RustSec {name} conditions are declared; current application reachability/platform applicability is not established"));
                }
            }
        }
        Ok(Interpretation {
            aliases: list(advisory.get("aliases"))?,
            severity: advisory
                .get("cvss")
                .and_then(toml::Value::as_str)
                .map(str::to_owned),
            modified_at: date(advisory.get("date"))?,
            withdrawn_at: date(advisory.get("withdrawn"))?,
            affected: format!(
                "Affected outside patched/unaffected selectors: patched [{}], unaffected [{}]; prereleases use SemVer precedence, build metadata does not alter precedence",
                patched.join(", "),
                unaffected.join(", ")
            ),
            fixed: patched,
            state: Some(state),
            conditions,
        })
    }

    #[cfg(feature = "jvm-tools")]
    #[derive(Clone)]
    enum MavenItem {
        Number(String),
        Qualifier(String),
        List(Vec<MavenItem>),
    }

    #[cfg(feature = "jvm-tools")]
    impl MavenItem {
        fn qualifier(value: &str, followed_by_digit: bool) -> Self {
            let value = match value {
                "a" if followed_by_digit => "alpha",
                "b" if followed_by_digit => "beta",
                "m" if followed_by_digit => "milestone",
                "cr" => "rc",
                "ga" | "final" | "release" => "",
                _ => value,
            };
            Self::Qualifier(value.into())
        }
        fn number(value: &str) -> Self {
            Self::Number(
                {
                    let value = value.trim_start_matches('0');
                    if value.is_empty() { "0" } else { value }
                }
                .into(),
            )
        }
        fn qualifier_order(value: &str) -> (usize, &str) {
            match value {
                "alpha" => (0, ""),
                "beta" => (1, ""),
                "milestone" => (2, ""),
                "rc" => (3, ""),
                "snapshot" => (4, ""),
                "" => (5, ""),
                "sp" => (6, ""),
                _ => (7, value),
            }
        }
        fn compare(&self, other: Option<&Self>) -> Ordering {
            match (self, other) {
                (Self::Number(left), Some(Self::Number(right))) => {
                    left.len().cmp(&right.len()).then_with(|| left.cmp(right))
                }
                (Self::Number(value), None) => {
                    if value == "0" {
                        Ordering::Equal
                    } else {
                        Ordering::Greater
                    }
                }
                (Self::Number(_), _) => Ordering::Greater,
                (Self::Qualifier(left), Some(Self::Qualifier(right))) => {
                    Self::qualifier_order(left).cmp(&Self::qualifier_order(right))
                }
                (Self::Qualifier(value), None) => {
                    Self::qualifier_order(value).cmp(&Self::qualifier_order(""))
                }
                (Self::Qualifier(_), _) => Ordering::Less,
                (Self::List(left), Some(Self::List(right))) => {
                    for index in 0..left.len().max(right.len()) {
                        let order = match (left.get(index), right.get(index)) {
                            (Some(left), right) => left.compare(right),
                            (None, Some(right)) => right.compare(None).reverse(),
                            (None, None) => Ordering::Equal,
                        };
                        if order != Ordering::Equal {
                            return order;
                        }
                    }
                    Ordering::Equal
                }
                (Self::List(values), None) => values
                    .iter()
                    .map(|value| value.compare(None))
                    .find(|order| *order != Ordering::Equal)
                    .unwrap_or(Ordering::Equal),
                (Self::List(_), Some(Self::Number(_))) => Ordering::Less,
                (Self::List(_), Some(Self::Qualifier(_))) => Ordering::Greater,
            }
        }
        fn is_null(&self) -> bool {
            match self {
                Self::Number(value) => value == "0",
                Self::Qualifier(value) => value.is_empty(),
                Self::List(value) => value.is_empty(),
            }
        }
        fn normalize(values: &mut Vec<Self>) {
            for index in (0..values.len()).rev() {
                match values.get(index) {
                    Some(value) if value.is_null() => {
                        values.remove(index);
                    }
                    Some(Self::List(_)) => {}
                    _ => break,
                }
            }
        }
        fn current(stack: &mut [Vec<Self>]) -> Result<&mut Vec<Self>> {
            stack
                .last_mut()
                .ok_or_else(|| anyhow::anyhow!("Invalid Maven version state"))
        }
        fn parse(version: &str) -> Result<Self> {
            ensure!(
                !version.is_empty()
                    && version.len() <= MAX_FIELD_BYTES
                    && version
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || b".-_+".contains(&byte)),
                "Unsupported Maven version syntax"
            );
            let version = version.to_ascii_lowercase();
            let mut stack = vec![Vec::new()];
            let mut numeric = false;
            let mut start = 0;
            for (index, byte) in version.bytes().enumerate() {
                ensure!(stack.len() <= 128, "Maven version nesting bound exceeded");
                if byte == b'.' || byte == b'-' {
                    let text = version
                        .get(start..index)
                        .ok_or_else(|| anyhow::anyhow!("Invalid Maven token"))?;
                    let item = if text.is_empty() {
                        Self::number("0")
                    } else if numeric {
                        Self::number(text)
                    } else {
                        Self::qualifier(text, false)
                    };
                    Self::current(&mut stack)?.push(item);
                    start = index + 1;
                    if byte == b'-' {
                        stack.push(Vec::new());
                    }
                } else if byte.is_ascii_digit() {
                    if !numeric && index > start {
                        if !Self::current(&mut stack)?.is_empty() {
                            stack.push(Vec::new());
                        }
                        Self::current(&mut stack)?.push(Self::qualifier(
                            version
                                .get(start..index)
                                .ok_or_else(|| anyhow::anyhow!("Invalid Maven token"))?,
                            true,
                        ));
                        stack.push(Vec::new());
                        start = index;
                    }
                    numeric = true;
                } else {
                    if numeric && index > start {
                        Self::current(&mut stack)?.push(Self::number(
                            version
                                .get(start..index)
                                .ok_or_else(|| anyhow::anyhow!("Invalid Maven token"))?,
                        ));
                        stack.push(Vec::new());
                        start = index;
                    }
                    numeric = false;
                }
            }
            if let Some(text) = version.get(start..).filter(|text| !text.is_empty()) {
                if !numeric && !Self::current(&mut stack)?.is_empty() {
                    stack.push(Vec::new());
                }
                Self::current(&mut stack)?.push(if numeric {
                    Self::number(text)
                } else {
                    Self::qualifier(text, false)
                });
            }
            while stack.len() > 1 {
                let mut values = stack
                    .pop()
                    .ok_or_else(|| anyhow::anyhow!("Invalid Maven version state"))?;
                Self::normalize(&mut values);
                Self::current(&mut stack)?.push(Self::List(values));
            }
            let mut values = stack
                .pop()
                .ok_or_else(|| anyhow::anyhow!("Invalid Maven version state"))?;
            Self::normalize(&mut values);
            Ok(Self::List(values))
        }
    }

    #[cfg(feature = "jvm-tools")]
    pub fn compare_maven_versions(left: &str, right: &str) -> Result<Ordering> {
        Ok(MavenItem::parse(left)?.compare(Some(&MavenItem::parse(right)?)))
    }

    pub(super) fn is_documented_fix(
        snapshot: &AdvisorySnapshot,
        finding: &AdvisoryFinding,
        baseline: &str,
        version: &str,
    ) -> Result<bool> {
        #[cfg(not(feature = "jvm-tools"))]
        let _ = (snapshot, baseline);
        if finding.evidence.state != DependencyAdvisoryState::NotAffectedByThisAdvisory
            || finding.withdrawn_at.is_some()
        {
            return Ok(false);
        }
        match finding.evidence.package.ecosystem {
            DependencyEcosystem::Cargo => {
                #[cfg(feature = "rust-tools")]
                {
                    let version = semver::Version::parse(version)?;
                    for selector in &finding.evidence.fixed_versions {
                        if RustInterval::parse(selector.as_str())?.contains(&version) {
                            return Ok(true);
                        }
                    }
                    Ok(false)
                }
                #[cfg(not(feature = "rust-tools"))]
                {
                    anyhow::bail!("Rust fixed-version semantics are disabled")
                }
            }
            DependencyEcosystem::Maven => {
                #[cfg(feature = "jvm-tools")]
                {
                    let record = snapshot
                        .records
                        .iter()
                        .find(|record| {
                            record.kind == AdvisoryArchiveKind::OsvMaven
                                && record.identifier == finding.evidence.identifier.as_str()
                        })
                        .ok_or_else(|| {
                            anyhow::anyhow!("Documented fixed-version source is unavailable")
                        })?;
                    let value: serde_json::Value = serde_json::from_str(&record.document)
                        .map_err(|_| anyhow::anyhow!("Invalid fixed-version source"))?;
                    for affected in value
                        .get("affected")
                        .and_then(serde_json::Value::as_array)
                        .into_iter()
                        .flatten()
                    {
                        let name = affected
                            .get("package")
                            .and_then(|package| package.get("name"))
                            .and_then(serde_json::Value::as_str);
                        if name != Some(finding.evidence.package.name.as_str()) && name != Some("*")
                        {
                            continue;
                        }
                        for range in affected
                            .get("ranges")
                            .and_then(serde_json::Value::as_array)
                            .into_iter()
                            .flatten()
                        {
                            let kind = range.get("type").and_then(serde_json::Value::as_str);
                            if !matches!(kind, Some("ECOSYSTEM" | "SEMVER")) {
                                continue;
                            }
                            let compare = |left: &str, right: &str| -> Result<Ordering> {
                                if kind == Some("ECOSYSTEM") {
                                    compare_maven_versions(left, right)
                                } else {
                                    Ok(semver::Version::parse(left)?
                                        .cmp_precedence(&semver::Version::parse(right)?))
                                }
                            };
                            let mut introduced = None;
                            for event in range
                                .get("events")
                                .and_then(serde_json::Value::as_array)
                                .into_iter()
                                .flatten()
                            {
                                if let Some(start) =
                                    event.get("introduced").and_then(serde_json::Value::as_str)
                                {
                                    introduced = Some(start);
                                }
                                if let Some(fixed) =
                                    event.get("fixed").and_then(serde_json::Value::as_str)
                                {
                                    if let Some(start) = introduced {
                                        let baseline_after_start = start == "0"
                                            || compare(baseline, start)? != Ordering::Less;
                                        if baseline_after_start
                                            && compare(baseline, fixed)? == Ordering::Less
                                            && compare(version, fixed)? != Ordering::Less
                                        {
                                            return Ok(true);
                                        }
                                    }
                                    introduced = None;
                                }
                                if event.get("last_affected").is_some()
                                    || event.get("limit").is_some()
                                {
                                    introduced = None;
                                }
                            }
                        }
                    }
                    Ok(false)
                }
                #[cfg(not(feature = "jvm-tools"))]
                {
                    anyhow::bail!("JVM fixed-version semantics are disabled")
                }
            }
        }
    }

    struct Matcher<'a> {
        graph: &'a DependencyGraph,
        snapshot: &'a AdvisorySnapshot,
        scan: AdvisoryScan,
        bytes: usize,
    }

    impl Matcher<'_> {
        fn admit(
            &mut self,
            node: usize,
            record: &AdvisoryRecord,
            interpretation: Result<Interpretation>,
        ) -> Result<()> {
            if self.scan.findings.len() >= MAX_FINDINGS || self.bytes >= MAX_MATCH_METADATA {
                self.scan.incomplete(
                    "Advisory finding bound reached; omitted observations remain unevaluated",
                );
                return Ok(());
            }
            let mut value = match interpretation {
                Ok(value) => value,
                Err(_) => {
                    self.scan.incomplete("A relevant advisory has malformed or unsupported semantics; raw source errors are excluded");
                    Interpretation {
                        state: Some(DependencyAdvisoryState::UnknownVersionSemantics),
                        affected: "Unsupported advisory interpretation".into(),
                        ..Default::default()
                    }
                }
            };
            let package = self
                .graph
                .nodes
                .get(node)
                .ok_or_else(|| anyhow::anyhow!("Advisory matched an absent graph node"))?
                .clone();
            let mut state = value
                .state
                .unwrap_or(DependencyAdvisoryState::UnknownVersionSemantics);
            if package.source.kind() == DependencySourceKind::Unknown {
                state = DependencyAdvisoryState::UnverifiedSource;
                self.scan.incomplete("Coordinate/version correspondence was observed, but the resolved package's repository origin is unknown; this is not a known vulnerable dependency conclusion");
            }
            if package.classifier.is_some() && state != DependencyAdvisoryState::UnverifiedSource {
                state = DependencyAdvisoryState::UnknownApplicability;
                self.scan.incomplete(
                    "Advisory applicability to the selected classified artifact is not established",
                );
            }
            if value
                .modified_at
                .is_some_and(|modified| modified > self.scan.observed_at)
            {
                self.scan.incomplete("Advisory modification time is later than the observation clock; current coverage is uncertain");
            }
            if state == DependencyAdvisoryState::UnknownVersionSemantics {
                self.scan
                    .incomplete("Relevant advisory version semantics remain incomplete");
            }
            if !value.conditions.is_empty() {
                self.scan.incomplete("Advisory range/platform/application conditions remain incomplete; inspect the retained observation");
            }
            let source = self
                .snapshot
                .sources
                .iter()
                .find(|source| source.kind == record.kind)
                .ok_or_else(|| anyhow::anyhow!("Advisory source receipt missing"))?
                .clone();
            let mut provenance = self.graph.provenance.clone();
            provenance.provider = DiagnosticText::new(format!(
                "{:?}; source revision {}; source digest {}",
                record.kind, source.revision, source.sha256
            ));
            provenance.observed_at = self.scan.observed_at;
            provenance.coverage = self.scan.coverage;
            provenance.limitations = self.scan.limitations.clone();
            let evidence = DependencyAdvisoryEvidence {
                package,
                authority: if record.kind == AdvisoryArchiveKind::RustSec {
                    DependencyAdvisoryAuthority::RustSec
                } else {
                    DependencyAdvisoryAuthority::OsvMaven
                },
                identifier: DiagnosticText::new(&record.identifier),
                aliases: text_list(value.aliases),
                severity: value.severity.map(DiagnosticText::new),
                affected_versions: DiagnosticText::new(&value.affected),
                fixed_versions: text_list(value.fixed),
                state,
                provenance,
            };
            evidence.validate()?;
            let base = if record.kind == AdvisoryArchiveKind::RustSec {
                "https://rustsec.org/advisories/"
            } else {
                "https://osv.dev/vulnerability/"
            };
            let mut source_url = url::Url::parse(base)?;
            source_url
                .path_segments_mut()
                .map_err(|_| anyhow::anyhow!("Invalid advisory link base"))?
                .pop_if_empty()
                .push(&if record.kind == AdvisoryArchiveKind::RustSec {
                    format!("{}.html", record.identifier)
                } else {
                    record.identifier.clone()
                });
            value.conditions.truncate(32);
            let paths = self.graph.introducing_paths(node)?;
            if paths.truncated {
                self.scan
                    .incomplete("Introducing paths or dependency graph coverage are incomplete");
            }
            self.bytes = self.bytes.saturating_add(
                value.affected.len()
                    + record.identifier.len()
                    + value.conditions.iter().map(String::len).sum::<usize>()
                    + evidence
                        .aliases
                        .iter()
                        .map(|value| value.as_str().len())
                        .sum::<usize>()
                    + evidence
                        .fixed_versions
                        .iter()
                        .map(|value| value.as_str().len())
                        .sum::<usize>(),
            );
            if self.bytes > MAX_MATCH_METADATA {
                self.scan.incomplete(
                    "Advisory metadata bound reached; further observations are omitted",
                );
                return Ok(());
            }
            self.scan.findings.push(AdvisoryFinding {
                node,
                evidence,
                source,
                source_url: source_url.into(),
                modified_at: value.modified_at,
                withdrawn_at: value.withdrawn_at,
                conditions: text_list(value.conditions),
                paths,
            });
            Ok(())
        }
    }

    pub fn match_advisories(
        graph: &DependencyGraph,
        snapshot: &AdvisorySnapshot,
        now: u64,
        check_cancelled: impl Fn() -> Result<()>,
    ) -> Result<AdvisoryScan> {
        check_cancelled()?;
        snapshot.validate_for_matching()?;
        check_cancelled()?;
        let mut matcher = Matcher {
            graph,
            snapshot,
            bytes: 0,
            scan: AdvisoryScan {
                graph_revision: graph.identity(),
                advisory_revision: snapshot.identity()?,
                fetched_at: snapshot.fetched_at,
                observed_at: now,
                freshness: snapshot.freshness(now),
                upstream_freshness: snapshot.upstream_freshness(now),
                coverage: graph.provenance.coverage,
                findings: Vec::new(),
                limitations: graph.provenance.limitations.clone(),
            },
        };
        if matcher.scan.freshness != AdvisoryFreshness::Fresh {
            matcher
                .scan
                .incomplete("Retained advisory data is stale, expired or clock-uncertain");
        }
        if matcher.scan.upstream_freshness != Some(AdvisoryFreshness::Fresh) {
            matcher.scan.incomplete("Upstream publication age is stale, unknown or clock-uncertain independently of download age");
        }
        let ecosystem = if snapshot.dataset == AdvisoryDataset::RustSec {
            DependencyEcosystem::Cargo
        } else {
            DependencyEcosystem::Maven
        };
        let mut packages = BTreeMap::<&str, Vec<usize>>::new();
        for (index, node) in graph.nodes.iter().enumerate() {
            if node.ecosystem != ecosystem {
                continue;
            }
            if !node.source.is_public_registry()
                && node.source.kind() != DependencySourceKind::Unknown
            {
                matcher.scan.incomplete("Non-public/workspace/path/Git dependencies are outside this official public dataset's authority");
                continue;
            }
            if validate_public_coordinate(node.ecosystem, &node.name).is_err() {
                matcher
                    .scan
                    .incomplete("Unsupported dependency coordinates remain unevaluated");
                continue;
            }
            packages.entry(&node.name).or_default().push(index);
        }
        let mut observed_ids = BTreeMap::<String, BTreeSet<usize>>::new();
        let mut observed_aliases = BTreeMap::<String, BTreeSet<usize>>::new();
        for record in snapshot
            .records
            .iter()
            .filter(|record| record.kind != AdvisoryArchiveKind::OsvWithdrawals)
        {
            check_cancelled()?;
            if matcher.scan.findings.len() >= MAX_FINDINGS || matcher.bytes >= MAX_MATCH_METADATA {
                matcher.scan.incomplete(
                    "Advisory output bound reached; remaining observations are unevaluated",
                );
                break;
            }
            match record.kind {
                AdvisoryArchiveKind::RustSec => {
                    #[cfg(feature = "rust-tools")]
                    {
                        let header = record
                            .document
                            .strip_prefix("```toml\n")
                            .and_then(|value| value.split_once("\n```"))
                            .map(|(value, _)| value)
                            .ok_or_else(|| anyhow::anyhow!("RustSec front matter missing"))?;
                        let value: toml::Value = header
                            .parse()
                            .map_err(|_| anyhow::anyhow!("Invalid RustSec front matter"))?;
                        let name = value
                            .get("advisory")
                            .and_then(|value| value.get("package"))
                            .and_then(toml::Value::as_str)
                            .ok_or_else(|| anyhow::anyhow!("RustSec package missing"))?;
                        for node in packages.get(name).into_iter().flatten() {
                            check_cancelled()?;
                            matcher.admit(
                                *node,
                                record,
                                rust_interpret(
                                    &value,
                                    &graph
                                        .nodes
                                        .get(*node)
                                        .ok_or_else(|| anyhow::anyhow!("Missing dependency"))?
                                        .version,
                                ),
                            )?;
                        }
                    }
                    #[cfg(not(feature = "rust-tools"))]
                    anyhow::bail!("Rust advisory matching is disabled");
                }
                AdvisoryArchiveKind::OsvMaven => {
                    let value: serde_json::Value = serde_json::from_str(&record.document)
                        .map_err(|_| anyhow::anyhow!("Invalid OSV advisory"))?;
                    let schema_supported = value
                        .get("schema_version")
                        .and_then(serde_json::Value::as_str)
                        .map(|version| {
                            semver::Version::parse(version)
                                .is_ok_and(|version| version.major == 1 && version.minor <= 9)
                        })
                        .unwrap_or(true);
                    if !schema_supported {
                        matcher.scan.incomplete("An official source record has an unsupported or malformed OSV schema version; source coverage remains incomplete");
                    }

                    let Some(affected) =
                        value.get("affected").and_then(serde_json::Value::as_array)
                    else {
                        if value.get("withdrawn").is_none() {
                            matcher.scan.incomplete("OSV affected coverage missing");
                        }
                        continue;
                    };
                    for affected in affected {
                        if affected
                            .get("package")
                            .and_then(|value| value.get("ecosystem"))
                            .and_then(serde_json::Value::as_str)
                            != Some("Maven")
                        {
                            continue;
                        }
                        let Some(name) = affected
                            .get("package")
                            .and_then(|value| value.get("name"))
                            .and_then(serde_json::Value::as_str)
                        else {
                            matcher
                                .scan
                                .incomplete("OSV Maven package identity missing");
                            continue;
                        };
                        let selection = if name == "*" {
                            packages.values().flatten().copied().collect::<Vec<_>>()
                        } else {
                            packages.get(name).cloned().unwrap_or_default()
                        };
                        for node in selection {
                            check_cancelled()?;
                            matcher.admit(
                                node,
                                record,
                                osv_interpret(
                                    &value,
                                    affected,
                                    &graph
                                        .nodes
                                        .get(node)
                                        .ok_or_else(|| anyhow::anyhow!("Missing dependency"))?
                                        .version,
                                ),
                            )?;
                            if matcher.scan.findings.len() >= MAX_FINDINGS
                                || matcher.bytes >= MAX_MATCH_METADATA
                            {
                                break;
                            }
                        }
                    }
                }
                AdvisoryArchiveKind::OsvWithdrawals => {}
            }
        }
        for finding in &matcher.scan.findings {
            if finding.source.kind == AdvisoryArchiveKind::OsvMaven {
                observed_ids
                    .entry(finding.evidence.identifier.as_str().to_owned())
                    .or_default()
                    .insert(finding.node);
                for alias in &finding.evidence.aliases {
                    observed_aliases
                        .entry(alias.as_str().to_owned())
                        .or_default()
                        .insert(finding.node);
                }
            }
        }
        for record in snapshot
            .records
            .iter()
            .filter(|record| record.kind == AdvisoryArchiveKind::OsvWithdrawals)
        {
            check_cancelled()?;
            let value: serde_json::Value = serde_json::from_str(&record.document)
                .map_err(|_| anyhow::anyhow!("Invalid OSV withdrawal source"))?;
            let Some(withdrawn_at) = timestamp(value.get("withdrawn"))? else {
                continue;
            };
            let aliases = fields(value.get("aliases"))?;
            let exact = observed_ids
                .get(&record.identifier)
                .cloned()
                .unwrap_or_default();
            let mut related = exact.clone();
            for identifier in std::iter::once(&record.identifier).chain(aliases.iter()) {
                related.extend(
                    observed_aliases
                        .get(identifier)
                        .into_iter()
                        .flatten()
                        .copied(),
                );
                related.extend(observed_ids.get(identifier).into_iter().flatten().copied());
            }
            for node in related {
                let exact_identity = exact.contains(&node);
                matcher.scan.incomplete("An official withdrawal observation accompanies this advisory identity/alias; both source observations are preserved, not silently reconciled into a security conclusion");
                matcher.admit(node, record, Ok(Interpretation { state: Some(if exact_identity { DependencyAdvisoryState::Withdrawn } else { DependencyAdvisoryState::UnknownApplicability }), withdrawn_at: Some(withdrawn_at), modified_at: timestamp(value.get("modified"))?, aliases: aliases.clone(), affected: "Official withdrawal/tombstone observation; does not erase another source's evidence".into(), conditions: if exact_identity { Vec::new() } else { vec!["Alias correspondence only; withdrawal applicability to the other package/advisory identity is unverified".into()] }, ..Default::default() }))?;
            }
        }
        check_cancelled()?;
        for finding in &mut matcher.scan.findings {
            finding.evidence.provenance.coverage = matcher.scan.coverage;
            finding.evidence.provenance.limitations = matcher.scan.limitations.clone();
        }
        Ok(matcher.scan)
    }
    #[cfg(all(test, feature = "rust-tools", feature = "jvm-tools"))]
    mod tests {
        use super::*;
        const NOW: u64 = 1791187200;

        fn package(
            ecosystem: DependencyEcosystem,
            name: &str,
            version: &str,
        ) -> DependencyInstance {
            DependencyInstance {
                ecosystem,
                source: DependencySource::public_registry(ecosystem),
                name: name.into(),
                version: version.into(),
                classifier: None,
                configuration: DependencyIdentity::from_fields(&["fixture-config"])
                    .expect("Configuration"),
            }
        }
        fn graph(nodes: Vec<DependencyInstance>, edges: Vec<DependencyEdge>) -> DependencyGraph {
            DependencyGraph::new(
                DependencyProvenance {
                    host: DependencyIdentity::from_fields(&["local-host"]).expect("Host"),
                    inputs: DependencyIdentity::from_fields(&["manifest-and-lock"])
                        .expect("Inputs"),
                    provider: DiagnosticText::new("Selected owner fixture"),
                    observed_at: NOW,
                    coverage: DependencyCoverage::Complete,
                    limitations: Vec::new(),
                },
                nodes,
                edges,
                vec![0],
            )
            .expect("Bounded graph")
        }
        fn rust_record(extra: &str, patched: &str, unaffected: &str) -> AdvisoryRecord {
            AdvisoryRecord {
                kind: AdvisoryArchiveKind::RustSec,
                identifier: "RUSTSEC-2026-0001".into(),
                document: format!(
                    "```toml\n[advisory]\nid = \"RUSTSEC-2026-0001\"\npackage = \"example\"\ndate = \"2026-10-01\"\naliases = [\"CVE-2026-1234\"]\ncvss = \"CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H\"\n{extra}\n[versions]\npatched = [{patched}]\nunaffected = [{unaffected}]\n```\n# UNTRUSTED_ADVISORY_INSTRUCTIONS_NOT_MODEL_INPUT"
                ),
            }
        }
        fn osv_record(
            identifier: &str,
            ranges: serde_json::Value,
            versions: serde_json::Value,
        ) -> AdvisoryRecord {
            AdvisoryRecord { kind: AdvisoryArchiveKind::OsvMaven, identifier: identifier.into(), document: serde_json::json!({ "schema_version":"1.9.1", "id":identifier, "aliases":["CVE-2026-1234"], "modified":"2026-10-01T00:00:00Z", "severity":[{"type":"CVSS_V3","score":"CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}], "affected":[{"package":{"ecosystem":"Maven","name":"org.example:library"},"ranges":ranges,"versions":versions}], "details":"UNTRUSTED_ADVISORY_INSTRUCTIONS_NOT_MODEL_INPUT" }).to_string() }
        }
        fn source_snapshot(
            dataset: AdvisoryDataset,
            records: Vec<AdvisoryRecord>,
        ) -> AdvisorySnapshot {
            super::super::advisory_cache::matching_fixture(dataset, records, NOW)
        }

        #[test]
        fn doctor_advisory_match_rust_selectors_prerelease_metadata_and_conditions() {
            let record = rust_record("", "\">= 1.2.0\"", "\"< 1.0.0\"");
            let snapshot = source_snapshot(AdvisoryDataset::RustSec, vec![record]);
            let mut graph = graph(
                vec![
                    package(DependencyEcosystem::Cargo, "example", "1.1.0"),
                    package(DependencyEcosystem::Cargo, "example", "1.2.0+build"),
                    package(DependencyEcosystem::Cargo, "example", "1.2.0-alpha.1"),
                    package(DependencyEcosystem::Cargo, "example", "0.9.0"),
                ],
                vec![DependencyEdge {
                    parent: 0,
                    child: 2,
                    alias: "renamed".into(),
                    scope: "normal".into(),
                    requirement: Some("^1.1".into()),
                    target: Some("cfg(unix)".into()),
                    workspace_inherited: true,
                }],
            );
            graph.roots = vec![0, 1, 3];
            let scan = match_advisories(&graph, &snapshot, NOW, || Ok(())).expect("Local matching");
            let states = scan
                .findings
                .iter()
                .map(|finding| finding.evidence.state)
                .collect::<Vec<_>>();
            assert_eq!(
                states,
                vec![
                    DependencyAdvisoryState::KnownAffectedDependency,
                    DependencyAdvisoryState::NotAffectedByThisAdvisory,
                    DependencyAdvisoryState::KnownAffectedDependency,
                    DependencyAdvisoryState::NotAffectedByThisAdvisory
                ]
            );
            assert_eq!(
                scan.findings
                    .get(2)
                    .expect("Transitive prerelease")
                    .paths
                    .paths
                    .first()
                    .expect("Introducing path")
                    .nodes,
                vec![0, 2]
            );
            assert!(scan.findings.iter().all(|finding| {
                finding
                    .evidence
                    .application_exploitability()
                    .contains("Unknown")
            }));
            assert!(
                scan.findings
                    .first()
                    .expect("Finding")
                    .source_url
                    .ends_with("RUSTSEC-2026-0001.html")
            );
            assert!(
                scan.findings
                    .first()
                    .expect("Severity")
                    .evidence
                    .severity
                    .is_some()
            );
            assert_eq!(scan.coverage, DependencyCoverage::Complete);
            assert!(!scan.conclusion().as_str().contains("UNTRUSTED_ADVISORY"));
            for (selector, matches, misses) in [
                ("^0.2.3", "0.2.4-alpha", "0.3.0-alpha"),
                ("~1.2", "1.2.9-alpha", "1.3.0-alpha"),
                ("=1.0", "1.0.0+build", "1.0.0-alpha"),
            ] {
                let interval = RustInterval::parse(selector).expect("RustSec interval");
                assert!(
                    interval.contains(&semver::Version::parse(matches).expect("Version")),
                    "{selector} includes {matches}"
                );
                assert!(
                    !interval.contains(&semver::Version::parse(misses).expect("Version")),
                    "{selector} excludes {misses}"
                );
            }
            assert!(RustInterval::parse(">=1, >=2").is_err());
            assert!(RustInterval::parse(">2, <1").is_err());
            assert!(RustInterval::parse("1.*").is_err());
        }

        #[test]
        fn doctor_advisory_match_rust_withdrawn_informational_unsupported_and_platforms() {
            let graph = graph(
                vec![package(DependencyEcosystem::Cargo, "example", "1.1.0")],
                Vec::new(),
            );
            for (extra, expected) in [
                (
                    "withdrawn = \"2026-10-02\"",
                    DependencyAdvisoryState::Withdrawn,
                ),
                (
                    "informational = \"unmaintained\"",
                    DependencyAdvisoryState::Informational,
                ),
            ] {
                let snapshot = source_snapshot(
                    AdvisoryDataset::RustSec,
                    vec![rust_record(extra, "\">=1.2.0\"", "")],
                );
                let scan = match_advisories(&graph, &snapshot, NOW, || Ok(())).expect("Notice");
                assert_eq!(
                    scan.findings
                        .first()
                        .expect("Notice finding")
                        .evidence
                        .state,
                    expected
                );
                if expected == DependencyAdvisoryState::Withdrawn {
                    assert!(
                        scan.findings
                            .first()
                            .expect("Withdrawal")
                            .withdrawn_at
                            .is_some()
                    );
                }
            }
            let snapshot = source_snapshot(
                AdvisoryDataset::RustSec,
                vec![rust_record("", "\">=1.0\"", "\">=1.1\"")],
            );
            let scan = match_advisories(&graph, &snapshot, NOW, || Ok(()))
                .expect("Malformed ranges are observations");
            assert_eq!(scan.coverage, DependencyCoverage::Partial);
            assert_eq!(
                scan.findings.first().expect("Unsupported").evidence.state,
                DependencyAdvisoryState::UnknownVersionSemantics
            );
            let mut record = rust_record("", "\">=1.2.0\"", "");
            record.document = record.document.replace(
                "\n```",
                "\n[affected]\narch = [\"x86_64\"]\nos = [\"linux\"]\n```",
            );
            let snapshot = source_snapshot(AdvisoryDataset::RustSec, vec![record]);
            let scan = match_advisories(&graph, &snapshot, NOW, || Ok(()))
                .expect("Conditional observation");
            assert_eq!(scan.coverage, DependencyCoverage::Partial);
            assert!(
                scan.findings
                    .first()
                    .expect("Platform condition")
                    .conditions
                    .len()
                    >= 2
            );
        }

        #[test]
        fn doctor_advisory_match_maven_comparable_qualifiers_and_nested_tokens() {
            for (left, right, expected) in [
                ("1", "1.0.0", Ordering::Equal),
                ("1-ga", "1-final", Ordering::Equal),
                ("1-release", "1", Ordering::Equal),
                ("1-a1", "1-alpha-1", Ordering::Equal),
                ("1-cr2", "1-rc2", Ordering::Equal),
                ("1-beta", "1-milestone", Ordering::Less),
                ("1-rc", "1-SNAPSHOT", Ordering::Less),
                ("1-SNAPSHOT", "1", Ordering::Less),
                ("1", "1-sp", Ordering::Less),
                ("1-sp", "1-unknown", Ordering::Less),
                ("1.0.RC2", "1.0-RC3", Ordering::Less),
                ("1.0-X2", "1.0.X3", Ordering::Less),
                ("1-1", "1.1", Ordering::Less),
                ("1.0-1", "1.0.1", Ordering::Less),
                ("1.0.0-foo.0.0", "1-foo", Ordering::Equal),
                ("1.000000000000000000000002", "1.2", Ordering::Equal),
                (
                    "1.99999999999999999999999999",
                    "1.100000000000000000000000000",
                    Ordering::Less,
                ),
                ("1.0.0", "1.0.0-Final", Ordering::Equal),
            ] {
                assert_eq!(
                    compare_maven_versions(left, right).expect("Maven semantics"),
                    expected,
                    "{left} versus {right}"
                );
                assert_eq!(
                    compare_maven_versions(right, left).expect("Reverse order"),
                    expected.reverse()
                );
            }
            assert!(compare_maven_versions("${project.version}", "1.0").is_err());
            assert!(compare_maven_versions("é", "1.0").is_err());
            assert!(compare_maven_versions(&"1-".repeat(256), "1").is_err());
        }

        #[test]
        fn doctor_advisory_match_osv_event_spans_explicit_versions_and_unknown_ranges() {
            let ranges = serde_json::json!([{"type":"ECOSYSTEM","events":[{"introduced":"0"},{"fixed":"1.0"},{"introduced":"2.0-alpha"},{"last_affected":"2.0-rc1"}]}]);
            for (version, expected) in [
                (
                    "1.0-alpha",
                    DependencyAdvisoryState::KnownAffectedDependency,
                ),
                ("1.0", DependencyAdvisoryState::NotAffectedByThisAdvisory),
                ("1.5", DependencyAdvisoryState::NotAffectedByThisAdvisory),
                (
                    "2.0-beta1",
                    DependencyAdvisoryState::KnownAffectedDependency,
                ),
                ("2.0-rc1", DependencyAdvisoryState::KnownAffectedDependency),
                ("2.0", DependencyAdvisoryState::NotAffectedByThisAdvisory),
            ] {
                let snapshot = source_snapshot(
                    AdvisoryDataset::OsvMaven,
                    vec![osv_record(
                        "GHSA-aaaa-bbbb-cccc",
                        ranges.clone(),
                        serde_json::json!([]),
                    )],
                );
                let graph = graph(
                    vec![package(
                        DependencyEcosystem::Maven,
                        "org.example:library",
                        version,
                    )],
                    Vec::new(),
                );
                let scan =
                    match_advisories(&graph, &snapshot, NOW, || Ok(())).expect("OSV Maven range");
                assert_eq!(
                    scan.findings.first().expect("Observation").evidence.state,
                    expected,
                    "{version}"
                );
                assert_eq!(
                    scan.findings.first().expect("Fix").evidence.fixed_versions,
                    vec![DiagnosticText::new("1.0")]
                );
            }
            let unknown = serde_json::json!([{"type":"GIT","repo":"https://unsupported.example/repo","events":[{"introduced":"0"},{"fixed":"abcdef"}]}]);
            for (versions, expected) in [
                (
                    serde_json::json!(["1.0"]),
                    DependencyAdvisoryState::KnownAffectedDependency,
                ),
                (
                    serde_json::json!([]),
                    DependencyAdvisoryState::UnknownVersionSemantics,
                ),
            ] {
                let snapshot = source_snapshot(
                    AdvisoryDataset::OsvMaven,
                    vec![osv_record("GHSA-aaaa-bbbb-cccc", unknown.clone(), versions)],
                );
                let graph = graph(
                    vec![package(
                        DependencyEcosystem::Maven,
                        "org.example:library",
                        "1.0",
                    )],
                    Vec::new(),
                );
                let scan = match_advisories(&graph, &snapshot, NOW, || Ok(()))
                    .expect("Explicit version evidence independent of unsupported range");
                assert_eq!(
                    scan.findings.first().expect("Observation").evidence.state,
                    expected
                );
                assert_eq!(scan.coverage, DependencyCoverage::Partial);
            }
            let semver = serde_json::json!({"type":"SEMVER","events":[{"introduced":"1.0.0-alpha.1"},{"fixed":"1.0.0"}]});
            assert!(
                osv_range(&semver, "1.0.0-beta+build")
                    .expect("SEMVER prerelease")
                    .0
            );
            assert!(
                !osv_range(&semver, "1.0.0+build")
                    .expect("SEMVER build metadata")
                    .0
            );
            for events in [
                serde_json::json!([{"fixed":"1.0"}]),
                serde_json::json!([{"introduced":"2"},{"fixed":"1"}]),
                serde_json::json!([{"introduced":"0"},{"introduced":"1"}]),
                serde_json::json!([{"introduced":"0"},{"fixed":"1"},{"introduced":"0"}]),
            ] {
                assert!(
                    osv_range(
                        &serde_json::json!({"type":"ECOSYSTEM","events":events}),
                        "1.0"
                    )
                    .is_err()
                );
            }
        }

        #[test]
        fn doctor_advisory_match_empty_ranges_future_schema_and_clock_are_incomplete() {
            let graph = graph(
                vec![package(
                    DependencyEcosystem::Maven,
                    "org.example:library",
                    "1.0",
                )],
                Vec::new(),
            );
            let empty = source_snapshot(
                AdvisoryDataset::OsvMaven,
                vec![osv_record(
                    "GHSA-aaaa-bbbb-cccc",
                    serde_json::json!([]),
                    serde_json::json!([]),
                )],
            );
            let scan =
                match_advisories(&graph, &empty, NOW, || Ok(())).expect("Empty range semantics");
            assert_eq!(scan.coverage, DependencyCoverage::Partial);
            assert_eq!(
                scan.findings
                    .first()
                    .expect("Unknown version observation")
                    .evidence
                    .state,
                DependencyAdvisoryState::UnknownVersionSemantics
            );
            let mut record = osv_record(
                "GHSA-aaaa-bbbb-cccc",
                serde_json::json!([]),
                serde_json::json!(["1.0"]),
            );
            record.document = record.document.replace("1.9.1", "2.0.0");
            let unsupported = source_snapshot(AdvisoryDataset::OsvMaven, vec![record]);
            let scan = match_advisories(&graph, &unsupported, NOW, || Ok(()))
                .expect("Unsupported schema observation");
            assert_eq!(scan.coverage, DependencyCoverage::Partial);
            assert_eq!(
                scan.findings
                    .first()
                    .expect("Unsupported schema")
                    .evidence
                    .state,
                DependencyAdvisoryState::UnknownVersionSemantics
            );
            let scan =
                match_advisories(&graph, &empty, NOW - 1, || Ok(())).expect("Clock rollback");
            assert_eq!(scan.freshness, AdvisoryFreshness::ClockUncertain);
            assert_eq!(scan.coverage, DependencyCoverage::Partial);
        }

        #[test]
        fn doctor_advisory_match_unknown_origin_classifier_withdrawal_and_private_are_not_authority()
         {
            let record = osv_record(
                "GHSA-aaaa-bbbb-cccc",
                serde_json::json!([]),
                serde_json::json!(["1.0"]),
            );
            let withdrawal = AdvisoryRecord { kind: AdvisoryArchiveKind::OsvWithdrawals, identifier: "GHSA-aaaa-bbbb-cccc".into(), document: r#"{"id":"GHSA-aaaa-bbbb-cccc","modified":"2026-10-02T00:00:00Z","withdrawn":"2026-10-02T00:00:00Z"}"#.into() };
            let snapshot = source_snapshot(AdvisoryDataset::OsvMaven, vec![record, withdrawal]);
            let mut unknown = package(DependencyEcosystem::Maven, "org.example:library", "1.0");
            unknown.source =
                DependencySource::non_public(DependencySourceKind::Unknown, "unknown-origin")
                    .expect("Owner uncertainty");
            let scan = match_advisories(
                &graph(vec![unknown.clone()], Vec::new()),
                &snapshot,
                NOW,
                || Ok(()),
            )
            .expect("Unverified correspondence");
            assert!(
                scan.findings
                    .iter()
                    .all(|finding| finding.evidence.state
                        == DependencyAdvisoryState::UnverifiedSource)
            );
            let mut false_authority = scan
                .findings
                .first()
                .expect("Unverified observation")
                .evidence
                .clone();
            false_authority.state = DependencyAdvisoryState::KnownAffectedDependency;
            assert!(
                false_authority.validate().is_err(),
                "Original proven-origin invariant is preserved"
            );
            false_authority.state = DependencyAdvisoryState::NotAffectedByThisAdvisory;
            assert!(false_authority.validate().is_err());
            let public = package(DependencyEcosystem::Maven, "org.example:library", "1.0");
            let scan = match_advisories(
                &graph(vec![public.clone()], Vec::new()),
                &snapshot,
                NOW,
                || Ok(()),
            )
            .expect("Both sources");
            assert_eq!(scan.findings.len(), 2);
            assert_eq!(
                scan.findings
                    .first()
                    .expect("Active advisory")
                    .evidence
                    .state,
                DependencyAdvisoryState::KnownAffectedDependency
            );
            assert_eq!(
                scan.findings.get(1).expect("Tombstone").evidence.state,
                DependencyAdvisoryState::Withdrawn
            );
            assert_ne!(
                scan.findings.first().expect("Source").source.kind,
                scan.findings.get(1).expect("Source").source.kind
            );
            assert_eq!(scan.coverage, DependencyCoverage::Partial);
            let mut private = unknown;
            private.source = DependencySource::non_public(
                DependencySourceKind::PrivateRegistry,
                "private-registry",
            )
            .expect("Private owner");
            let scan =
                match_advisories(&graph(vec![private], Vec::new()), &snapshot, NOW, || Ok(()))
                    .expect("Unsupported private package");
            assert!(scan.findings.is_empty());
            assert_eq!(scan.coverage, DependencyCoverage::Partial);
            let mut classified = public;
            classified.classifier = Some("tests".into());
            let scan =
                match_advisories(&graph(vec![classified], Vec::new()), &snapshot, NOW, || {
                    Ok(())
                })
                .expect("Classifier correspondence");
            assert!(scan.findings.iter().all(|finding| finding.evidence.state
                != DependencyAdvisoryState::KnownAffectedDependency));
        }

        #[test]
        fn doctor_advisory_match_freshness_identity_cancellation_bounds_and_safe_wording() {
            let graph = graph(
                vec![package(DependencyEcosystem::Cargo, "example", "1.1.0")],
                Vec::new(),
            );
            let snapshot = source_snapshot(
                AdvisoryDataset::RustSec,
                vec![rust_record("", "\">=1.2.0\"", "")],
            );
            let scan = match_advisories(&graph, &snapshot, NOW + 24 * 60 * 60, || Ok(()))
                .expect("Stale retained evidence");
            assert_eq!(scan.freshness, AdvisoryFreshness::Stale);
            assert_eq!(scan.coverage, DependencyCoverage::Partial);
            assert!(scan.conclusion().as_str().contains("not proof of security"));
            let mut changed = graph.clone();
            changed.nodes.first_mut().expect("Node").version = "1.2.0".into();
            assert_ne!(graph.identity(), changed.identity());
            assert!(
                match_advisories(&graph, &snapshot, NOW, || anyhow::bail!(
                    "Cancelled by the existing operation owner"
                ))
                .is_err()
            );
            let calls = std::cell::Cell::new(0);
            assert!(
                match_advisories(&graph, &snapshot, NOW, || {
                    calls.set(calls.get() + 1);
                    ensure!(calls.get() < 4, "Cancellation during matching");
                    Ok(())
                })
                .is_err()
            );
            let records = (0..600)
                .map(|index| {
                    let mut record = rust_record("", "\">=1.2.0\"", "");
                    let identifier = format!("RUSTSEC-2026-{index:04}");
                    record.document = record.document.replace("RUSTSEC-2026-0001", &identifier);
                    record.identifier = identifier;
                    record
                })
                .collect();
            let snapshot = source_snapshot(AdvisoryDataset::RustSec, records);
            let scan =
                match_advisories(&graph, &snapshot, NOW, || Ok(())).expect("Bounded findings");
            assert_eq!(scan.findings.len(), MAX_FINDINGS);
            assert_eq!(scan.coverage, DependencyCoverage::Partial);
            assert!(
                scan.limitations
                    .iter()
                    .any(|value| value.as_str().contains("bound"))
            );
            let mut corrupt = snapshot;
            corrupt.records.first_mut().expect("Record").identifier = "RUSTSEC-2026-9999".into();
            assert!(match_advisories(&graph, &corrupt, NOW, || Ok(())).is_err());
        }

        #[test]
        fn doctor_advisory_match_alias_withdrawals_and_wildcard_keep_uncertainty_and_metadata() {
            let mut active = osv_record(
                "GHSA-aaaa-bbbb-cccc",
                serde_json::json!([{ "type":"ECOSYSTEM", "events":[{"introduced":"0"},{"fixed":"2.0"}] }]),
                serde_json::json!(["1.0"]),
            );
            let mut parsed: serde_json::Value =
                serde_json::from_str(&active.document).expect("Public source fixture");
            parsed["affected"][0]["package"]["name"] = serde_json::json!("*");
            parsed.as_object_mut().expect("Fixture").remove("severity");
            parsed["database_specific"] = serde_json::json!({"severity":"HIGH"});
            active.document = parsed.to_string();
            let withdrawal = AdvisoryRecord { kind: AdvisoryArchiveKind::OsvWithdrawals, identifier: "GHSA-eeee-ffff-gggg".into(), document: r#"{"id":"GHSA-eeee-ffff-gggg","aliases":["CVE-2026-1234"],"modified":"2026-10-02T00:00:00Z","withdrawn":"2026-10-02T00:00:00Z"}"#.into() };
            let snapshot = source_snapshot(AdvisoryDataset::OsvMaven, vec![active, withdrawal]);
            let graph = graph(
                vec![package(
                    DependencyEcosystem::Maven,
                    "org.example:library",
                    "1.0",
                )],
                Vec::new(),
            );
            let scan = match_advisories(&graph, &snapshot, NOW, || Ok(()))
                .expect("Scoped wildcard and alias observation");
            assert_eq!(scan.findings.len(), 2);
            let active = scan.findings.first().expect("Active source");
            assert_eq!(
                active.evidence.state,
                DependencyAdvisoryState::KnownAffectedDependency
            );
            assert!(
                active
                    .evidence
                    .affected_versions
                    .as_str()
                    .contains("introduced 0")
            );
            assert!(
                active
                    .evidence
                    .severity
                    .as_ref()
                    .expect("Source severity")
                    .as_str()
                    .contains("HIGH")
            );
            let tombstone = scan
                .findings
                .get(1)
                .expect("Alias-linked withdrawal source");
            assert_eq!(
                tombstone.evidence.state,
                DependencyAdvisoryState::UnknownApplicability
            );
            assert!(tombstone.withdrawn_at.is_some());
            assert!(
                tombstone
                    .conditions
                    .iter()
                    .any(|value| value.as_str().contains("Alias correspondence"))
            );
            assert_eq!(scan.coverage, DependencyCoverage::Partial);
        }

        #[test]
        #[ignore = "Requires explicitly supplied output from the installed Maven ComparableVersion"]
        fn doctor_advisory_match_maven_installed_differential() {
            let path = std::env::var_os("MADE_DOCTOR_MAVEN_COMPARISONS")
                .expect("Select actual Maven comparison evidence explicitly");
            let bytes = std::fs::read(path).expect("Owned actual-runtime comparison output");
            assert!(bytes.len() <= 1024 * 1024);
            let value: serde_json::Value = serde_json::from_slice(&bytes).expect("Runtime receipt");
            let versions = value
                .get("versions")
                .and_then(serde_json::Value::as_array)
                .expect("Versions");
            let results = value
                .get("comparisons")
                .and_then(serde_json::Value::as_array)
                .expect("Comparisons");
            assert!(versions.len() > 30 && versions.len() < 200);
            assert_eq!(results.len(), versions.len() * versions.len());
            for (index, result) in results.iter().enumerate() {
                let left = versions
                    .get(index / versions.len())
                    .and_then(serde_json::Value::as_str)
                    .expect("Left version");
                let right = versions
                    .get(index % versions.len())
                    .and_then(serde_json::Value::as_str)
                    .expect("Right version");
                let expected = result.as_i64().expect("Maven ordering").cmp(&0);
                assert_eq!(
                    compare_maven_versions(left, right).expect("Native Maven interpretation"),
                    expected,
                    "{left} versus {right}"
                );
            }
        }
    }
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
pub use registry_updates::{
    RegistryMetadata, RegistryQuery, RegistryVersion, UpdateRecommendation, recommend_updates,
};

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
mod registry_updates {
    use super::*;
    use std::cmp::Ordering;

    pub(crate) const MAX_REGISTRY_BYTES: usize = 1024 * 1024;
    const MAX_VERSIONS: usize = 4096;
    const MAX_RECOMMENDATIONS: usize = 32;

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct RegistryQuery {
        package: DependencyInstance,
        graph: DependencyIdentity,
        node: usize,
        confirmed_public_catalog: bool,
    }

    impl RegistryQuery {
        pub fn for_resolved(graph: &DependencyGraph, node: usize) -> Result<Self> {
            let package = graph
                .nodes()
                .get(node)
                .ok_or_else(|| anyhow::anyhow!("Selected dependency is unavailable"))?
                .clone();
            ensure!(
                package.source.is_public_registry(),
                "Private, path, Git or unknown repository dependencies cannot authorize public registry disclosure"
            );
            let query = Self {
                package,
                graph: graph.identity(),
                node,
                confirmed_public_catalog: false,
            };
            query.exact_url()?;
            Ok(query)
        }
        pub fn for_confirmed_public_catalog(
            graph: &DependencyGraph,
            node: usize,
            confirmed_coordinate: &str,
        ) -> Result<Self> {
            let package = graph
                .nodes()
                .get(node)
                .ok_or_else(|| anyhow::anyhow!("Selected dependency is unavailable"))?
                .clone();
            ensure!(
                package.ecosystem == DependencyEcosystem::Maven
                    && package.source.kind() == DependencySourceKind::Unknown
                    && package.name == confirmed_coordinate,
                "Public catalog confirmation must select the exact unknown-origin Maven coordinate; known private/path/Git sources remain blocked"
            );
            let query = Self {
                package,
                graph: graph.identity(),
                node,
                confirmed_public_catalog: true,
            };
            query.exact_url()?;
            Ok(query)
        }
        pub fn is_catalog_only(&self) -> bool {
            self.confirmed_public_catalog
        }
        pub fn package(&self) -> &DependencyInstance {
            &self.package
        }
        pub fn graph_revision(&self) -> &DependencyIdentity {
            &self.graph
        }
        pub fn node(&self) -> usize {
            self.node
        }
        pub fn revalidate_graph(&self, graph: &DependencyGraph) -> Result<()> {
            ensure!(
                graph.identity() == self.graph
                    && graph.nodes().get(self.node) == Some(&self.package),
                "Selected dependency graph/source/version changed; review a fresh public request"
            );
            Ok(())
        }
        pub(super) fn identity(&self) -> Result<DependencyIdentity> {
            DependencyIdentity::from_fields(&[
                &self.graph.hex(),
                &self.node.to_string(),
                &self.package.name,
                &self.package.version,
                &self.package.source.identity().hex(),
                &self.package.configuration.hex(),
                if self.confirmed_public_catalog {
                    "confirmed-public-catalog"
                } else {
                    "proven-public-registry"
                },
            ])
        }
        pub fn exact_url(&self) -> Result<String> {
            self.package.validate()?;
            ensure!(
                (self.package.source.is_public_registry()
                    || (self.confirmed_public_catalog
                        && self.package.ecosystem == DependencyEcosystem::Maven
                        && self.package.source.kind() == DependencySourceKind::Unknown))
                    && self.package.classifier.is_none(),
                "Public version queries require a proven public registry and an unclassified artifact"
            );
            metadata_url(self.package.ecosystem, &self.package.name)
        }
    }

    pub(crate) fn metadata_url(ecosystem: DependencyEcosystem, coordinate: &str) -> Result<String> {
        validate_public_coordinate(ecosystem, coordinate)?;
        match ecosystem {
            DependencyEcosystem::Cargo => {
                ensure!(
                    cfg!(feature = "rust-tools"),
                    "Rust registry queries are disabled"
                );
                Ok(format!(
                    "https://crates.io/api/v1/crates/{coordinate}/versions"
                ))
            }
            DependencyEcosystem::Maven => {
                ensure!(
                    cfg!(feature = "jvm-tools"),
                    "JVM registry queries are disabled"
                );
                let (group, artifact) = coordinate
                    .split_once(':')
                    .ok_or_else(|| anyhow::anyhow!("Invalid public Maven coordinate"))?;
                ensure!(
                    group.split('.').all(|part| !part.is_empty())
                        && artifact != "."
                        && artifact != "..",
                    "Ambiguous Maven metadata path"
                );
                Ok(format!(
                    "https://repo.maven.apache.org/maven2/{}/{artifact}/maven-metadata.xml",
                    group.replace('.', "/")
                ))
            }
        }
    }

    pub(crate) fn is_metadata_url(value: &str) -> bool {
        if let Some(name) = value
            .strip_prefix("https://crates.io/api/v1/crates/")
            .and_then(|path| path.strip_suffix("/versions"))
        {
            return metadata_url(DependencyEcosystem::Cargo, name).is_ok_and(|url| url == value);
        }
        if let Some(path) = value
            .strip_prefix("https://repo.maven.apache.org/maven2/")
            .and_then(|path| path.strip_suffix("/maven-metadata.xml"))
        {
            if let Some((group, artifact)) = path.rsplit_once('/') {
                return metadata_url(
                    DependencyEcosystem::Maven,
                    &format!("{}:{artifact}", group.replace('/', ".")),
                )
                .is_ok_and(|url| url == value);
            }
        }
        false
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct RegistryVersion {
        pub version: String,
        pub yanked: bool,
        pub required_rust: Option<DiagnosticText>,
    }

    #[derive(Clone, Debug)]
    pub struct RegistryMetadata {
        pub query: RegistryQuery,
        pub versions: Vec<RegistryVersion>,
        pub fetched_at: u64,
        pub response_date: Option<u64>,
        pub response_age: Option<u64>,
        pub modified_at: Option<u64>,
        pub body_identity: DependencyIdentity,
        pub coverage: DependencyCoverage,
        pub limitations: Vec<DiagnosticText>,
    }

    impl RegistryMetadata {
        pub fn freshness(&self, now: u64) -> Option<AdvisoryFreshness> {
            let elapsed = now.checked_sub(self.fetched_at)?;
            let date = self.response_date?;
            let age = self.response_age.unwrap_or(0);
            let server_elapsed = now.checked_sub(date)?;
            Some(
                if elapsed >= 86400 || server_elapsed.saturating_add(age) >= 86400 {
                    AdvisoryFreshness::Stale
                } else {
                    AdvisoryFreshness::Fresh
                },
            )
        }
    }

    fn validate_version(version: &str) -> Result<()> {
        validate_field(version)?;
        ensure!(
            version
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b".-_+".contains(&byte)),
            "Unsupported registry version encoding"
        );
        Ok(())
    }

    pub(crate) fn parse_metadata(
        query: RegistryQuery,
        bytes: &[u8],
        fetched_at: u64,
        response_date: Option<u64>,
        response_age: Option<u64>,
        modified_at: Option<u64>,
    ) -> Result<RegistryMetadata> {
        query.exact_url()?;
        ensure!(
            bytes.len() <= MAX_REGISTRY_BYTES,
            "Version metadata exceeds reviewed byte limit"
        );
        let (versions, coverage, limitations) = match query.package.ecosystem {
            DependencyEcosystem::Cargo => parse_cargo(&query.package.name, bytes)?,
            DependencyEcosystem::Maven => {
                #[cfg(feature = "jvm-tools")]
                {
                    (
                        parse_maven(&query.package.name, bytes)?,
                        DependencyCoverage::Complete,
                        Vec::new(),
                    )
                }
                #[cfg(not(feature = "jvm-tools"))]
                {
                    anyhow::bail!("JVM metadata parsing is disabled")
                }
            }
        };
        let metadata = RegistryMetadata {
            query,
            versions,
            fetched_at,
            response_date,
            response_age,
            modified_at,
            body_identity: DependencyIdentity(Sha256::digest(bytes).into()),
            coverage,
            limitations,
        };
        Ok(metadata)
    }

    fn parse_cargo(
        name: &str,
        bytes: &[u8],
    ) -> Result<(
        Vec<RegistryVersion>,
        DependencyCoverage,
        Vec<DiagnosticText>,
    )> {
        #[derive(Deserialize)]
        struct Version {
            #[serde(rename = "crate")]
            name: String,
            num: String,
            yanked: bool,
            rust_version: Option<String>,
        }
        #[derive(Deserialize)]
        struct Body {
            versions: Vec<Version>,
            meta: serde_json::Value,
        }
        let body: Body = serde_json::from_slice(bytes)
            .map_err(|_| anyhow::anyhow!("Invalid crates.io version metadata"))?;
        ensure!(
            body.versions.len() <= MAX_VERSIONS,
            "Registry version count exceeds bound"
        );
        let mut seen = BTreeSet::new();
        let mut versions = Vec::new();
        for version in body.versions {
            ensure!(
                version.name == name && seen.insert(version.num.clone()),
                "Registry response contains a different or duplicate package version"
            );
            validate_version(&version.num)?;
            semver::Version::parse(&version.num)
                .map_err(|_| anyhow::anyhow!("Unsupported Cargo registry version"))?;
            if let Some(required) = &version.rust_version {
                validate_version(required)?;
            }
            versions.push(RegistryVersion {
                version: version.num,
                yanked: version.yanked,
                required_rust: version.rust_version.map(DiagnosticText::new),
            });
        }
        let meta = body
            .meta
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Invalid crates.io pagination metadata"))?;
        let next = meta
            .get("next_page")
            .ok_or_else(|| anyhow::anyhow!("Missing crates.io pagination coverage"))?;
        ensure!(
            next.is_null()
                || next
                    .as_str()
                    .is_some_and(|value| !value.is_empty() && value.len() <= MAX_FIELD_BYTES),
            "Invalid crates.io next-page metadata"
        );
        let total = meta.get("total").and_then(serde_json::Value::as_u64);
        ensure!(
            total.is_none_or(|total| total >= versions.len() as u64),
            "Crates.io total version count contradicts the response"
        );
        let partial = !next.is_null() || total != Some(versions.len() as u64);
        Ok((
            versions,
            if partial {
                DependencyCoverage::Partial
            } else {
                DependencyCoverage::Complete
            },
            if partial {
                vec![DiagnosticText::new(
                    "Registry metadata indicates additional unreviewed pages; candidate coverage is incomplete",
                )]
            } else {
                Vec::new()
            },
        ))
    }

    #[cfg(feature = "jvm-tools")]
    fn parse_maven(coordinate: &str, bytes: &[u8]) -> Result<Vec<RegistryVersion>> {
        use quick_xml::{Reader, events::Event};
        let (group, artifact) = coordinate
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("Invalid Maven coordinate"))?;
        let mut reader = Reader::from_reader(bytes);
        reader.config_mut().trim_text(true);
        let mut stack = Vec::<Vec<u8>>::new();
        let mut fields = BTreeMap::<String, String>::new();
        let mut versions = Vec::new();
        let mut seen = BTreeSet::new();
        let mut containers = BTreeSet::new();
        let mut root_seen = false;
        let mut root_closed = false;
        let mut text = String::new();
        loop {
            match reader
                .read_event()
                .map_err(|_| anyhow::anyhow!("Malformed Maven version metadata"))?
            {
                Event::Start(start) => {
                    ensure!(
                        !root_closed && stack.len() < 16 && start.attributes().next().is_none(),
                        "Unsupported Maven metadata structure"
                    );
                    if stack.is_empty() {
                        ensure!(
                            !root_seen && start.name().as_ref() == b"metadata",
                            "Invalid Maven metadata root"
                        );
                        root_seen = true;
                    }
                    let name = start.name().as_ref().to_vec();
                    let parent = stack.last().map(Vec::as_slice);
                    let valid = match parent {
                        None => name == b"metadata",
                        Some(b"metadata") => {
                            matches!(name.as_slice(), b"groupId" | b"artifactId" | b"versioning")
                        }
                        Some(b"versioning") => matches!(
                            name.as_slice(),
                            b"latest" | b"release" | b"versions" | b"lastUpdated"
                        ),
                        Some(b"versions") => name == b"version",
                        _ => false,
                    };
                    ensure!(valid, "Unsupported Maven metadata schema");
                    if name == b"versioning" || name == b"versions" {
                        ensure!(
                            containers.insert(name.clone()),
                            "Duplicate Maven metadata container"
                        );
                    }
                    text.clear();
                    stack.push(name);
                }
                Event::End(end) => {
                    ensure!(
                        stack
                            .last()
                            .is_some_and(|name| name.as_slice() == end.name().as_ref()),
                        "Mismatched Maven metadata element"
                    );
                    let path = stack
                        .iter()
                        .map(|value| {
                            std::str::from_utf8(value)
                                .map_err(|_| anyhow::anyhow!("Invalid metadata element name"))
                        })
                        .collect::<Result<Vec<_>>>()?
                        .join("/");
                    match path.as_str() {
                        "metadata/groupId" | "metadata/artifactId" => {
                            ensure!(
                                fields.insert(path, text.clone()).is_none(),
                                "Duplicate Maven metadata identity"
                            );
                        }
                        "metadata/versioning/versions/version" => {
                            validate_version(&text)?;
                            compare_maven_versions(&text, &text)?;
                            ensure!(
                                versions.len() < MAX_VERSIONS && seen.insert(text.clone()),
                                "Excessive or duplicate Maven versions"
                            );
                            versions.push(RegistryVersion {
                                version: text.clone(),
                                yanked: false,
                                required_rust: None,
                            });
                        }
                        _ => {}
                    }
                    stack.pop();
                    text.clear();
                    if stack.is_empty() {
                        root_closed = true;
                    }
                }
                Event::Text(value) => {
                    let value = value
                        .decode()
                        .map_err(|_| anyhow::anyhow!("Invalid Maven metadata text"))?;
                    ensure!(
                        text.len().saturating_add(value.len()) <= MAX_FIELD_BYTES,
                        "Maven metadata text exceeds bound"
                    );
                    ensure!(
                        !stack.is_empty() || value.trim().is_empty(),
                        "Text outside Maven metadata root"
                    );
                    text.push_str(&value);
                }
                Event::DocType(_) | Event::CData(_) | Event::GeneralRef(_) | Event::Empty(_) => {
                    anyhow::bail!("Unsupported Maven metadata entity/empty structure")
                }
                Event::Decl(_) | Event::Comment(_) => {}
                Event::PI(_) => {
                    anyhow::bail!("Maven metadata processing instructions are unsupported")
                }
                Event::Eof => break,
            }
        }
        ensure!(
            root_seen
                && root_closed
                && stack.is_empty()
                && containers.contains(b"versions".as_slice())
                && fields
                    .get("metadata/groupId")
                    .is_some_and(|value| value == group)
                && fields
                    .get("metadata/artifactId")
                    .is_some_and(|value| value == artifact),
            "Maven metadata does not establish the selected coordinate"
        );
        Ok(versions)
    }

    #[derive(Clone, Debug)]
    pub struct UpdateRecommendation {
        pub evidence: DependencyUpdateEvidence,
        pub constraint_compatibility: DependencyUpdateKind,
        pub registry_revision: DependencyIdentity,
        pub required_rust: Option<DiagnosticText>,
        pub advisory_revision: Option<DependencyIdentity>,
        pub catalog_only: bool,
        pub limitations: Vec<DiagnosticText>,
    }

    pub fn recommend_updates(
        graph: &DependencyGraph,
        metadata: &RegistryMetadata,
        advisory: Option<&AdvisorySnapshot>,
        observed_at: u64,
        check_cancelled: impl Fn() -> Result<()>,
    ) -> Result<Vec<UpdateRecommendation>> {
        check_cancelled()?;
        metadata.query.revalidate_graph(graph)?;
        let current = metadata.query.package();
        let mut recommendations = Vec::new();
        let mut limitations = metadata.limitations.clone();
        if metadata.freshness(observed_at) != Some(AdvisoryFreshness::Fresh) {
            limitations.push(DiagnosticText::new("Registry response freshness is stale or uncertain; this is not a current update guarantee"));
        }
        limitations.push(DiagnosticText::new("At most thirty-two candidate versions are interpreted, reserving sixteen places for each constraint-compatible and other candidate group; omitted alternatives remain unassessed"));
        limitations.push(DiagnosticText::new("Version constraints and advisory matching do not prove API compatibility, successful resolution, toolchain support or an application security fix"));
        let requirements = graph
            .edges()
            .iter()
            .filter(|edge| edge.child == metadata.query.node())
            .map(|edge| edge.requirement.as_deref())
            .collect::<Vec<_>>();
        if metadata.query.is_catalog_only() {
            limitations.push(DiagnosticText::new("Explicitly confirmed public catalog lookup only: the resolved artifact's repository remains unknown, so these are not verified updates or security fixes for that artifact"));
        }
        for version in &metadata.versions {
            validate_version(&version.version)?;
            version_order(current.ecosystem, &version.version, &version.version)?;
        }
        if advisory.is_some_and(|snapshot| {
            snapshot.freshness(observed_at) != AdvisoryFreshness::Fresh
                || snapshot.upstream_freshness(observed_at) != Some(AdvisoryFreshness::Fresh)
        }) {
            limitations.push(DiagnosticText::new("Advisory data is stale or its publication freshness is uncertain; security-fix candidates are relative to that historical source, not a current security guarantee"));
        }
        let before = advisory
            .map(|snapshot| match_advisories(graph, snapshot, observed_at, &check_cancelled))
            .transpose()?;
        let mut versions = Vec::<&RegistryVersion>::new();
        for version in metadata.versions.iter().filter(|version| !version.yanked) {
            check_cancelled()?;
            let mut lower = 0usize;
            let mut upper = versions.len();
            while lower < upper {
                let middle = lower + (upper - lower) / 2;
                let other = versions
                    .get(middle)
                    .ok_or_else(|| anyhow::anyhow!("Version ordering bound changed"))?;
                if version_order(current.ecosystem, &version.version, &other.version)?
                    == Ordering::Greater
                {
                    upper = middle;
                } else {
                    lower = middle + 1;
                }
            }
            versions.insert(lower, version);
        }
        let mut compatible_count = 0usize;
        let mut other_count = 0usize;
        for version in versions {
            check_cancelled()?;
            if version_order(current.ecosystem, &version.version, &current.version)?
                != Ordering::Greater
            {
                continue;
            }
            let constraint = match current.ecosystem {
                DependencyEcosystem::Cargo => {
                    let candidate = semver::Version::parse(&version.version)
                        .map_err(|_| anyhow::anyhow!("Unsupported candidate SemVer"))?;
                    let parsed = requirements
                        .iter()
                        .map(|requirement| {
                            requirement.and_then(|value| semver::VersionReq::parse(value).ok())
                        })
                        .collect::<Option<Vec<_>>>();
                    match parsed.filter(|values| !values.is_empty()) {
                        Some(values)
                            if values
                                .iter()
                                .all(|requirement| requirement.matches(&candidate)) =>
                        {
                            DependencyUpdateKind::ConstraintCompatible
                        }
                        Some(_) => DependencyUpdateKind::PotentiallyBreaking,
                        None => DependencyUpdateKind::UnknownCompatibility,
                    }
                }
                DependencyEcosystem::Maven => DependencyUpdateKind::UnknownCompatibility,
            };
            let count = if constraint == DependencyUpdateKind::ConstraintCompatible {
                &mut compatible_count
            } else {
                &mut other_count
            };
            if *count >= MAX_RECOMMENDATIONS / 2 {
                continue;
            }
            *count += 1;
            let mut candidate = current.clone();
            candidate.version = version.version.clone();
            let mut fixed = Vec::new();
            if let (Some(snapshot), Some(before)) = (advisory, &before) {
                if !metadata.query.is_catalog_only() {
                    let mut nodes = graph.nodes().to_vec();
                    *nodes
                        .get_mut(metadata.query.node())
                        .ok_or_else(|| anyhow::anyhow!("Selected node disappeared"))? =
                        candidate.clone();
                    let candidate_graph = DependencyGraph::new(
                        graph.provenance().clone(),
                        nodes,
                        graph.edges().to_vec(),
                        graph.roots().to_vec(),
                    )?;
                    let after = match_advisories(
                        &candidate_graph,
                        snapshot,
                        observed_at,
                        &check_cancelled,
                    )?;
                    for finding in before.findings.iter().filter(|finding| {
                        finding.node == metadata.query.node()
                            && finding.evidence.state
                                == DependencyAdvisoryState::KnownAffectedDependency
                    }) {
                        for other in after.findings.iter().filter(|other| {
                            other.node == finding.node
                                && other.evidence.identifier == finding.evidence.identifier
                        }) {
                            if advisory_match::is_documented_fix(
                                snapshot,
                                other,
                                &current.version,
                                &version.version,
                            )? {
                                if !fixed.contains(&finding.evidence.identifier) {
                                    fixed.push(finding.evidence.identifier.clone());
                                }
                                break;
                            }
                        }
                    }
                }
            }
            let kind = if fixed.is_empty() {
                constraint
            } else {
                DependencyUpdateKind::AdvisoryFix
            };
            let evidence = DependencyUpdateEvidence {
                package: current.clone(),
                candidate,
                kind,
                advisory_identifiers: fixed,
                provenance: DependencyProvenance {
                    host: graph.provenance().host.clone(),
                    inputs: graph.provenance().inputs.clone(),
                    provider: DiagnosticText::new(metadata.query.exact_url()?),
                    observed_at,
                    coverage: if metadata.coverage == DependencyCoverage::Complete
                        && metadata.freshness(observed_at) == Some(AdvisoryFreshness::Fresh)
                    {
                        graph.provenance().coverage
                    } else {
                        DependencyCoverage::Partial
                    },
                    limitations: limitations.clone(),
                },
            };
            evidence.validate()?;
            recommendations.push(UpdateRecommendation {
                evidence,
                constraint_compatibility: constraint,
                registry_revision: metadata.body_identity.clone(),
                required_rust: version.required_rust.clone(),
                advisory_revision: advisory.map(AdvisorySnapshot::identity).transpose()?,
                catalog_only: metadata.query.is_catalog_only(),
                limitations: limitations.clone(),
            });
            if recommendations.len() == MAX_RECOMMENDATIONS {
                break;
            }
        }
        check_cancelled()?;
        Ok(recommendations)
    }

    fn version_order(ecosystem: DependencyEcosystem, left: &str, right: &str) -> Result<Ordering> {
        match ecosystem {
            DependencyEcosystem::Cargo => Ok(semver::Version::parse(left)
                .map_err(|_| anyhow::anyhow!("Unsupported Cargo version"))?
                .cmp_precedence(
                    &semver::Version::parse(right)
                        .map_err(|_| anyhow::anyhow!("Unsupported Cargo version"))?,
                )),
            DependencyEcosystem::Maven => {
                #[cfg(feature = "jvm-tools")]
                {
                    compare_maven_versions(left, right)
                }
                #[cfg(not(feature = "jvm-tools"))]
                {
                    anyhow::bail!("JVM version comparison is disabled")
                }
            }
        }
    }
    #[cfg(all(test, feature = "rust-tools", feature = "jvm-tools"))]
    mod tests {
        use super::*;
        fn graph(
            ecosystem: DependencyEcosystem,
            version: &str,
            requirement: Option<&str>,
        ) -> DependencyGraph {
            let node = DependencyInstance {
                ecosystem,
                source: DependencySource::public_registry(ecosystem),
                name: if ecosystem == DependencyEcosystem::Cargo {
                    "example".into()
                } else {
                    "org.example:library".into()
                },
                version: version.into(),
                classifier: None,
                configuration: DependencyIdentity::from_fields(&["selected"]).expect("Identity"),
            };
            let root = DependencyInstance {
                name: "workspace".into(),
                source: DependencySource::non_public(DependencySourceKind::Workspace, "root")
                    .expect("Workspace"),
                ..node.clone()
            };
            DependencyGraph::new(
                DependencyProvenance {
                    host: DependencyIdentity::from_fields(&["host"]).expect("Host"),
                    inputs: DependencyIdentity::from_fields(&["inputs"]).expect("Inputs"),
                    provider: DiagnosticText::new("existing resolved owner"),
                    observed_at: 1000,
                    coverage: DependencyCoverage::Complete,
                    limitations: vec![],
                },
                vec![root, node],
                vec![DependencyEdge {
                    parent: 0,
                    child: 1,
                    alias: "library".into(),
                    scope: "normal".into(),
                    requirement: requirement.map(str::to_owned),
                    target: None,
                    workspace_inherited: true,
                }],
                vec![0],
            )
            .expect("Selected graph")
        }
        fn cargo_bytes(versions: &[(&str, bool)]) -> Vec<u8> {
            serde_json::to_vec(&serde_json::json!({"versions": versions.iter().map(|(version, yanked)| serde_json::json!({"crate":"example", "num":version,"yanked":yanked,"rust_version":"1.90"})).collect::<Vec<_>>(), "meta":{"next_page":null,"total":versions.len()}})).expect("Official shape")
        }
        #[test]
        fn doctor_dependency_updates_unknown_catalog_requires_exact_confirmation_and_preserves_origin()
         {
            let original = graph(DependencyEcosystem::Maven, "1.0", None);
            for kind in [
                DependencySourceKind::Unknown,
                DependencySourceKind::PrivateRegistry,
                DependencySourceKind::Path,
                DependencySourceKind::Git,
                DependencySourceKind::Workspace,
            ] {
                let mut nodes = original.nodes().to_vec();
                nodes.get_mut(1).expect("Node").source =
                    DependencySource::non_public(kind, "original-origin").expect("Source");
                let selected = DependencyGraph::new(
                    original.provenance().clone(),
                    nodes,
                    original.edges().to_vec(),
                    original.roots().to_vec(),
                )
                .expect("Graph");
                assert!(RegistryQuery::for_resolved(&selected, 1).is_err());
                assert!(
                    RegistryQuery::for_confirmed_public_catalog(&selected, 1, "org.other:library")
                        .is_err()
                );
                let query = RegistryQuery::for_confirmed_public_catalog(
                    &selected,
                    1,
                    "org.example:library",
                );
                if kind != DependencySourceKind::Unknown {
                    assert!(query.is_err());
                    continue;
                }
                let query = query.expect("Explicit exact public coordinate confirmation");
                assert!(query.is_catalog_only());
                assert_eq!(query.package().source.kind(), DependencySourceKind::Unknown);
                assert_ne!(
                    query.identity().expect("Identity"),
                    RegistryQuery::for_resolved(&original, 1)
                        .expect("Public")
                        .identity()
                        .expect("Identity")
                );
                let bytes = br#"<metadata><groupId>org.example</groupId><artifactId>library</artifactId><versioning><versions><version>2.0</version></versions></versioning></metadata>"#;
                let metadata =
                    parse_metadata(query, bytes, 1000, Some(1000), None, None).expect("Catalog");
                let findings = recommend_updates(&selected, &metadata, None, 1000, || Ok(()))
                    .expect("Candidates");
                assert_eq!(findings.len(), 1);
                let candidate = findings.first().expect("Candidate");
                assert!(candidate.catalog_only);
                assert_eq!(
                    candidate.evidence.candidate.source.kind(),
                    DependencySourceKind::Unknown
                );
                assert_ne!(candidate.evidence.kind, DependencyUpdateKind::AdvisoryFix);
                assert!(
                    candidate
                        .limitations
                        .iter()
                        .any(|text| text.as_str().contains("repository remains unknown"))
                );
                assert!(metadata.query.revalidate_graph(&original).is_err());
            }
        }

        #[test]
        fn doctor_dependency_updates_preserves_compatible_candidates_and_requires_documented_fix() {
            let graph = graph(DependencyEcosystem::Cargo, "1.0.0", Some("^1"));
            let mut versions = (2..42)
                .map(|major| (format!("{major}.0.0"), false))
                .collect::<Vec<_>>();
            versions.push(("1.1.0".into(), false));
            versions.push(("1.2.0".into(), false));
            let refs = versions
                .iter()
                .map(|(version, yanked)| (version.as_str(), *yanked))
                .collect::<Vec<_>>();
            let metadata = parse_metadata(
                RegistryQuery::for_resolved(&graph, 1).expect("Query"),
                &cargo_bytes(&refs),
                1791187200,
                Some(1791187200),
                None,
                None,
            )
            .expect("Catalog");
            let source = |patched: &str, unaffected: &str| {
                advisory_cache::matching_fixture(
                    AdvisoryDataset::RustSec,
                    vec![AdvisoryRecord {
                        kind: AdvisoryArchiveKind::RustSec,
                        identifier: "RUSTSEC-2026-0001".into(),
                        document: format!(
                            "```toml\n[advisory]\nid = \"RUSTSEC-2026-0001\"\npackage = \"example\"\ndate = \"2026-10-01\"\n[versions]\npatched = [{patched}]\nunaffected = [{unaffected}]\n```"
                        ),
                    }],
                    1791187200,
                )
            };
            let unaffected = source("", "\">= 1.1.0\"");
            let candidates =
                recommend_updates(&graph, &metadata, Some(&unaffected), 1791187200, || Ok(()))
                    .expect("Unaffected candidates");
            assert!(
                candidates
                    .iter()
                    .all(|candidate| candidate.evidence.kind != DependencyUpdateKind::AdvisoryFix)
            );
            assert!(
                candidates
                    .iter()
                    .any(|candidate| candidate.evidence.candidate.version == "1.1.0")
            );
            assert!(candidates.len() <= MAX_RECOMMENDATIONS);
            let patched = source("\">= 1.2.0\"", "\">= 1.1.0, < 1.2.0\"");
            let candidates =
                recommend_updates(&graph, &metadata, Some(&patched), 1791187200, || Ok(()))
                    .expect("Fixed candidates");
            assert_eq!(
                candidates
                    .iter()
                    .find(|candidate| candidate.evidence.candidate.version == "1.1.0")
                    .expect("Unaffected")
                    .evidence
                    .kind,
                DependencyUpdateKind::ConstraintCompatible
            );
            assert_eq!(
                candidates
                    .iter()
                    .find(|candidate| candidate.evidence.candidate.version == "1.2.0")
                    .expect("Patched")
                    .evidence
                    .kind,
                DependencyUpdateKind::AdvisoryFix
            );
        }

        #[test]
        fn doctor_dependency_updates_public_source_graph_and_destination_guards() {
            let public = graph(DependencyEcosystem::Cargo, "1.0.0", Some("^1"));
            let query = RegistryQuery::for_resolved(&public, 1).expect("Proven public instance");
            assert_eq!(
                query.exact_url().expect("URL"),
                "https://crates.io/api/v1/crates/example/versions"
            );
            let mut nodes = public.nodes().to_vec();
            nodes.get_mut(1).expect("Node").source =
                DependencySource::non_public(DependencySourceKind::PrivateRegistry, "private")
                    .expect("Private");
            let private = DependencyGraph::new(
                public.provenance().clone(),
                nodes,
                public.edges().to_vec(),
                public.roots().to_vec(),
            )
            .expect("Private graph");
            assert!(RegistryQuery::for_resolved(&private, 1).is_err());
            assert!(query.revalidate_graph(&private).is_err());
            for forbidden in [
                "http://crates.io/api/v1/crates/example/versions",
                "https://crates.io.evil/api/v1/crates/example/versions",
                "https://crates.io/api/v1/crates/example/1.0/download",
                "https://user@crates.io/api/v1/crates/example/versions",
                "https://repo.maven.apache.org/maven2/org/../library/maven-metadata.xml",
                "https://repo.maven.apache.org/maven2/org/example/library/maven-metadata.xml?token=SECRET",
            ] {
                assert!(!is_metadata_url(forbidden), "{forbidden}");
            }
            assert!(metadata_url(DependencyEcosystem::Maven, "org..example:library").is_err());
            assert!(is_metadata_url(
                "https://repo.maven.apache.org/maven2/org/example/library/maven-metadata.xml"
            ));
        }
        #[test]
        fn doctor_dependency_updates_cargo_zero_prerelease_yank_constraints_and_cancellation() {
            let graph = graph(DependencyEcosystem::Cargo, "0.2.1", Some("^0.2"));
            let metadata = parse_metadata(
                RegistryQuery::for_resolved(&graph, 1).expect("Query"),
                &cargo_bytes(&[
                    ("0.2.2", false),
                    ("0.3.0", false),
                    ("0.2.3", true),
                    ("0.2.4-rc.1", false),
                ]),
                1000,
                Some(1000),
                Some(0),
                None,
            )
            .expect("Metadata");
            let recommendations =
                recommend_updates(&graph, &metadata, None, 1000, || Ok(())).expect("Candidates");
            assert_eq!(recommendations.len(), 3);
            assert!(
                recommendations
                    .iter()
                    .any(|item| item.evidence.candidate.version == "0.2.2"
                        && item.constraint_compatibility
                            == DependencyUpdateKind::ConstraintCompatible)
            );
            assert!(
                recommendations
                    .iter()
                    .filter(|item| item.evidence.candidate.version != "0.2.2")
                    .all(|item| item.constraint_compatibility
                        == DependencyUpdateKind::PotentiallyBreaking)
            );
            assert!(
                recommend_updates(&graph, &metadata, None, 1000, || anyhow::bail!("Cancelled"))
                    .is_err()
            );
            let stale = recommend_updates(&graph, &metadata, None, 1000 + 86400, || Ok(()))
                .expect("Historical candidates");
            assert!(
                stale
                    .iter()
                    .all(|item| item.evidence.provenance.coverage == DependencyCoverage::Partial)
            );
            assert_eq!(metadata.freshness(999), None);
        }
        #[test]
        fn doctor_dependency_updates_maven_identity_entities_qualifiers_and_unknown_compatibility()
        {
            let graph = graph(DependencyEcosystem::Maven, "1.0-rc1", Some("1.0-rc1"));
            let query = RegistryQuery::for_resolved(&graph, 1).expect("Public fixture");
            let bytes=br#"<?xml version="1.0"?><metadata><groupId>org.example</groupId><artifactId>library</artifactId><versioning><versions><version>1.0-alpha1</version><version>1.0</version><version>1.0-sp1</version><version>2.0</version></versions></versioning></metadata>"#;
            let metadata = parse_metadata(query.clone(), bytes, 1000, Some(1000), None, Some(100))
                .expect("Bounded Maven metadata");
            let recommendations = recommend_updates(&graph, &metadata, None, 1000, || Ok(()))
                .expect("Ordered candidates");
            assert_eq!(recommendations.len(), 3);
            assert_eq!(
                recommendations
                    .first()
                    .expect("Latest")
                    .evidence
                    .candidate
                    .version,
                "2.0"
            );
            assert!(
                recommendations
                    .iter()
                    .all(|item| item.constraint_compatibility
                        == DependencyUpdateKind::UnknownCompatibility)
            );
            for bytes in [br#"<!DOCTYPE metadata SYSTEM "https://evil"><metadata/>"#.as_slice(),br#"<metadata><groupId>org.other</groupId><artifactId>library</artifactId></metadata>"#.as_slice(), br#"<metadata><groupId>org.example</groupId><artifactId>library</artifactId><versioning><versions><version>&secret;</version></versions></versioning></metadata>"#.as_slice()] { assert!(parse_metadata(query.clone(),bytes,1000,Some(1000),None,None).is_err()); }
        }
        #[test]
        fn doctor_dependency_updates_maven_fixed_interval_must_include_the_baseline() {
            let now = 1791187200;
            let fixture = |current: &str, terminal: &str| {
                let graph = graph(DependencyEcosystem::Maven, current, None);
                let metadata=parse_metadata(RegistryQuery::for_resolved(&graph,1).expect("Query"),br#"<metadata><groupId>org.example</groupId><artifactId>library</artifactId><versioning><versions><version>3.0</version></versions></versioning></metadata>"#,now,Some(now),None,None).expect("Metadata");
                let record=AdvisoryRecord { kind:AdvisoryArchiveKind::OsvMaven,identifier:"GHSA-aaaa-bbbb-cccc".into(),document:serde_json::json!({"schema_version":"1.9.1","id":"GHSA-aaaa-bbbb-cccc","modified":"2026-10-01T00:00:00Z","affected":[{"package":{"ecosystem":"Maven","name":"org.example:library"},"ranges":[{"type":"ECOSYSTEM","events":[{"introduced":"0"},{"fixed":"1.1"},{"introduced":"2.0"},{terminal:"2.5"}]}]}]}).to_string() };
                let snapshot =
                    advisory_cache::matching_fixture(AdvisoryDataset::OsvMaven, vec![record], now);
                recommend_updates(&graph, &metadata, Some(&snapshot), now, || Ok(()))
                    .expect("Recommendations")
                    .first()
                    .expect("Candidate")
                    .evidence
                    .kind
            };
            assert_eq!(
                fixture("1.0", "limit"),
                DependencyUpdateKind::AdvisoryFix,
                "Documented fix closes the observed baseline interval"
            );
            assert_eq!(
                fixture("2.0", "limit"),
                DependencyUpdateKind::UnknownCompatibility,
                "An older fixed interval must not masquerade as a fix for a later affected baseline"
            );
            assert_eq!(fixture("2.0", "fixed"), DependencyUpdateKind::AdvisoryFix);
        }

        #[test]
        fn doctor_dependency_updates_bounds_partial_metadata_and_source_mismatch() {
            let graph = graph(DependencyEcosystem::Cargo, "1.0.0", Some("^1"));
            let query = RegistryQuery::for_resolved(&graph, 1).expect("Query");
            let mut body: serde_json::Value =
                serde_json::from_slice(&cargo_bytes(&[("1.1.0", false)])).expect("Body");
            body["meta"]["next_page"] = serde_json::json!("UNREVIEWED_DESTINATION");
            let metadata = parse_metadata(
                query.clone(),
                &serde_json::to_vec(&body).expect("Partial body"),
                1000,
                None,
                None,
                None,
            )
            .expect("Explicit partial result");
            assert_eq!(metadata.coverage, DependencyCoverage::Partial);
            assert_eq!(metadata.freshness(1000), None);
            for invalid in [
                serde_json::Value::Null,
                serde_json::json!({}),
                serde_json::json!({"next_page":null,"total":0}),
                serde_json::json!({"next_page":42,"total":1}),
            ] {
                let mut malformed = body.clone();
                malformed["meta"] = invalid;
                assert!(
                    parse_metadata(
                        query.clone(),
                        &serde_json::to_vec(&malformed).expect("Malformed fixture"),
                        1000,
                        Some(1000),
                        None,
                        None
                    )
                    .is_err()
                );
            }
            let mut unknown_count = body.clone();
            unknown_count["meta"] = serde_json::json!({"next_page":null});
            assert_eq!(
                parse_metadata(
                    query.clone(),
                    &serde_json::to_vec(&unknown_count).expect("Unknown count"),
                    1000,
                    Some(1000),
                    None,
                    None
                )
                .expect("Explicit incomplete coverage")
                .coverage,
                DependencyCoverage::Partial
            );
            body["versions"][0]["crate"] = serde_json::json!("other");
            assert!(
                parse_metadata(
                    query.clone(),
                    &serde_json::to_vec(&body).expect("Wrong name"),
                    1000,
                    None,
                    None,
                    None
                )
                .is_err()
            );
            assert!(
                parse_metadata(
                    query,
                    &vec![b' '; MAX_REGISTRY_BYTES + 1],
                    1000,
                    None,
                    None,
                    None
                )
                .is_err()
            );
        }
    }
}

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
pub use advisory_fetch::{prepare_project_advisory_scan, prepare_project_registry_query};

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
pub use owner_projection::{
    DoctorDependencyProjection, DoctorDependencyRepairProposal, DoctorDependencyVerification,
    DoctorDependencyVerificationBaseline,
};

#[cfg(any(feature = "rust-tools", feature = "jvm-tools"))]
mod owner_projection {
    use super::*;
    use crate::doctor::{
        DoctorInvestigationInputIdentity, DoctorRepairBinding, comparison_source_identity,
        project_repair_binding,
    };
    use gpui::{App, Entity};
    use project::{Project, ProjectPath};

    #[derive(Clone)]
    enum OwnerRevision {
        #[cfg(feature = "rust-tools")]
        Cargo {
            root: ProjectPath,
            revision: u64,
            fingerprint: u64,
            metadata: String,
        },
        #[cfg(feature = "jvm-tools")]
        Jvm {
            root: ProjectPath,
            format: project::jvm_project::JvmReviewedDependencyFormat,
            report: String,
        },
    }

    #[derive(Clone)]
    pub struct DoctorDependencyProjection {
        graph: DependencyGraph,
        binding: DoctorRepairBinding,
        inputs: DoctorInvestigationInputIdentity,
        owner: OwnerRevision,
    }

    impl DoctorDependencyProjection {
        pub fn graph(&self) -> &DependencyGraph {
            &self.graph
        }
        pub fn binding(&self) -> &DoctorRepairBinding {
            &self.binding
        }

        pub fn revalidate(&self, project: &Entity<Project>, cx: &App) -> Result<()> {
            self.binding.revalidate_project(project, cx)?;
            ensure_saved_inputs(project, self.binding.directory(), cx)?;
            ensure!(
                comparison_source_identity(project, self.binding.directory(), cx)? == self.inputs,
                "Dependency source inputs changed; collect and review current owner evidence"
            );
            match &self.owner {
                #[cfg(feature = "rust-tools")]
                OwnerRevision::Cargo {
                    root,
                    revision,
                    fingerprint,
                    metadata,
                } => {
                    let store = project.read(cx).cargo_workspace_store().read(cx);
                    let (snapshot, current) = store
                        .cached_snapshot(cx)
                        .ok_or_else(|| anyhow::anyhow!("Cargo owner has no current graph"))?;
                    ensure!(
                        current
                            && snapshot.revision == *revision
                            && snapshot.input_fingerprint == *fingerprint,
                        "Cargo owner revision changed"
                    );
                    let workspace = snapshot
                        .workspaces
                        .iter()
                        .find(|workspace| workspace.key.root == *root)
                        .ok_or_else(|| {
                            anyhow::anyhow!("Selected Cargo workspace is unavailable")
                        })?;
                    ensure!(
                        workspace
                            .dependency_graph
                            .as_ref()
                            .is_some_and(|graph| graph.metadata_identity == *metadata),
                        "Selected Cargo resolution changed"
                    );
                }
                #[cfg(feature = "jvm-tools")]
                OwnerRevision::Jvm {
                    root,
                    format,
                    report,
                } => {
                    let store = project.read(cx).lsp_store();
                    let store = store.read(cx);
                    ensure!(
                        store
                            .reviewed_jvm_dependency_snapshot(root, format, cx)
                            .is_some_and(|graph| graph.report_identity == *report),
                        "Selected JVM dependency report/provider authority changed or is unavailable"
                    );
                }
            }
            Ok(())
        }

        #[cfg(feature = "rust-tools")]
        pub fn cargo(
            project: &Entity<Project>,
            directory: ProjectPath,
            root: ProjectPath,
            generation: u64,
            observed_at: u64,
            cx: &App,
        ) -> Result<Self> {
            ensure_scope(&directory, &root)?;
            let binding = project_repair_binding(project, directory.clone(), generation, cx)?;
            ensure_saved_inputs(project, &directory, cx)?;
            let inputs = comparison_source_identity(project, &directory, cx)?;
            let store = project.read(cx).cargo_workspace_store().read(cx);
            let (snapshot,current)=store.cached_snapshot(cx).ok_or_else(||anyhow::anyhow!("No cached Cargo resolution; separately review an existing dependency collection Task"))?;
            ensure!(
                current,
                "Cached Cargo resolution is stale; no resolver was invoked"
            );
            let mut selected = snapshot
                .workspaces
                .iter()
                .filter(|workspace| workspace.key.root == root);
            let workspace = selected
                .next()
                .ok_or_else(|| anyhow::anyhow!("Select a current Cargo workspace root"))?;
            ensure!(
                selected.next().is_none(),
                "Ambiguous Cargo owner workspace selection"
            );
            let resolved = workspace
                .dependency_graph
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Selected Cargo workspace has no resolved graph"))?;
            let provenance = provenance(
                project,
                &directory,
                &inputs,
                "Existing cached Cargo owner",
                observed_at,
                cx,
            )?;
            let graph = project_cargo(workspace, provenance)?;
            let projection = Self {
                graph,
                binding,
                inputs,
                owner: OwnerRevision::Cargo {
                    root,
                    revision: snapshot.revision,
                    fingerprint: snapshot.input_fingerprint,
                    metadata: resolved.metadata_identity.clone(),
                },
            };
            projection.revalidate(project, cx)?;
            Ok(projection)
        }

        #[cfg(feature = "jvm-tools")]
        pub fn jvm(
            project: &Entity<Project>,
            directory: ProjectPath,
            root: ProjectPath,
            format: project::jvm_project::JvmReviewedDependencyFormat,
            generation: u64,
            observed_at: u64,
            cx: &App,
        ) -> Result<Self> {
            ensure_scope(&directory, &root)?;
            let binding = project_repair_binding(project, directory.clone(), generation, cx)?;
            ensure_saved_inputs(project, &directory, cx)?;
            let inputs = comparison_source_identity(project, &directory, cx)?;
            let store = project.read(cx).lsp_store();
            let store = store.read(cx);
            let resolved=store.reviewed_jvm_dependency_snapshot(&root,&format,cx).ok_or_else(||anyhow::anyhow!("No current reviewed JVM dependency graph for this module/configuration; separately review an existing collection Task"))?;
            let provenance = provenance(
                project,
                &directory,
                &inputs,
                "Existing reviewed JVM owner",
                observed_at,
                cx,
            )?;
            let graph = project_jvm(resolved, provenance)?;
            let projection = Self {
                graph,
                binding,
                inputs,
                owner: OwnerRevision::Jvm {
                    root,
                    format,
                    report: resolved.report_identity.clone(),
                },
            };
            projection.revalidate(project, cx)?;
            Ok(projection)
        }
    }

    #[derive(Clone)]
    pub struct DoctorDependencyRepairProposal {
        projection: DoctorDependencyProjection,
        evidence: String,
        summary: DiagnosticText,
        identity: DependencyIdentity,
        node: usize,
        candidate: DependencyInstance,
        catalog_only: bool,
        advisory_revision: Option<DependencyIdentity>,
        advisory_identifiers: Vec<DiagnosticText>,
    }

    impl DoctorDependencyRepairProposal {
        pub fn update(
            projection: DoctorDependencyProjection,
            node: usize,
            choice: &UpdateRecommendation,
        ) -> Result<Self> {
            choice.evidence.validate()?;
            ensure!(
                choice.limitations.len() <= 32,
                "Selected recommendation limitation bound exceeded"
            );
            ensure!(
                choice.evidence.package != choice.evidence.candidate,
                "Selected update does not change the dependency"
            );
            let package = projection
                .graph()
                .nodes()
                .get(node)
                .ok_or_else(|| anyhow::anyhow!("Selected dependency is unavailable"))?;
            ensure!(
                package == &choice.evidence.package
                    && choice.evidence.provenance.host == projection.graph().provenance().host
                    && choice.evidence.provenance.inputs == projection.graph().provenance().inputs,
                "Selected update belongs to changed dependency evidence"
            );
            ensure!(
                choice.evidence.kind != DependencyUpdateKind::Replacement,
                "Use a separately selected replacement proposal"
            );
            ensure!(
                package.source.is_public_registry()
                    || (choice.catalog_only
                        && package.ecosystem == DependencyEcosystem::Maven
                        && package.source.kind() == DependencySourceKind::Unknown),
                "Unconfirmed non-public dependency cannot become a registry update"
            );
            ensure!(
                !choice.catalog_only || choice.evidence.kind != DependencyUpdateKind::AdvisoryFix,
                "Unknown-origin catalog candidates cannot establish a security fix"
            );
            ensure!(
                choice.evidence.kind != DependencyUpdateKind::AdvisoryFix
                    || (choice.advisory_revision.is_some()
                        && !choice.evidence.advisory_identifiers.is_empty()),
                "Security-fix candidate lacks selected advisory provenance"
            );
            let intent = serde_json::json!({
                "choice":"explicitly selected update candidate",
                "package":DiagnosticText::new(format!("{} {}",package.name,package.version)),
                "candidate":DiagnosticText::new(format!("{} {}",choice.evidence.candidate.name,choice.evidence.candidate.version)),
                "classifier":choice.evidence.candidate.classifier.as_ref().map(DiagnosticText::new),
                "configuration":choice.evidence.candidate.configuration.hex(),
                "registry_source":choice.evidence.provenance.provider,
                "advisory_source":choice.advisory_revision.as_ref().map(|_|if package.ecosystem==DependencyEcosystem::Cargo{"RustSec"}else{"OSV Maven"}),
                "classification":format!("{:?}",choice.evidence.kind),
                "constraint_compatibility":format!("{:?}",choice.constraint_compatibility),
                "catalog_only":choice.catalog_only,
                "registry_revision":choice.registry_revision.hex(),
                "advisory_revision":choice.advisory_revision.as_ref().map(DependencyIdentity::hex),
                "advisory_identifiers":choice.evidence.advisory_identifiers,
                "required_rust":choice.required_rust,
                "observed_at":choice.evidence.provenance.observed_at,
                "limitations":choice.limitations,
            });
            let summary = DiagnosticText::new(format!(
                "Unverified selected dependency proposal: {} {} to {}; {:?}; catalog-only={}. Relevant fresh advisory and build/test verification remains required",
                package.name,
                package.version,
                choice.evidence.candidate.version,
                choice.evidence.kind,
                choice.catalog_only
            ));
            let identity = DependencyIdentity::from_fields(&[
                "update",
                &format!("{package:?}"),
                &format!("{:?}", choice.evidence.candidate),
            ])?;
            Self::selected(
                projection,
                node,
                intent,
                summary,
                identity,
                choice.evidence.candidate.clone(),
                choice.catalog_only,
                choice.advisory_revision.clone(),
                choice.evidence.advisory_identifiers.clone(),
            )
        }

        pub fn replacement(
            projection: DoctorDependencyProjection,
            node: usize,
            candidate: DependencyInstance,
        ) -> Result<Self> {
            candidate.validate()?;
            let package = projection
                .graph()
                .nodes()
                .get(node)
                .ok_or_else(|| anyhow::anyhow!("Selected dependency is unavailable"))?;
            ensure!(
                candidate.ecosystem == package.ecosystem && candidate.source.is_public_registry(),
                "Replacement must be an explicitly selected public coordinate in the same ecosystem; repository changes need separate review"
            );
            ensure!(
                candidate.name != package.name,
                "Same-name version changes are updates, not replacement packages"
            );
            let intent = serde_json::json!({
                "choice":"explicitly selected replacement package",
                "package":package.selected_label(),
                "candidate":candidate.selected_label(),
                "classifier":candidate.classifier.as_ref().map(DiagnosticText::new),
                "configuration":candidate.configuration.hex(),
                "classification":"Replacement",
                "availability":"Unverified manual selection; no catalog request or dependency resolution was performed",
                "compatibility":"Unknown; manifest, lockfile and necessary compatibility changes require normal change review and relevant verification",
            });
            let summary = DiagnosticText::new(format!(
                "Unverified selected replacement proposal: {} to {}; availability and compatibility remain unknown; relevant fresh advisory and build/test verification remains required",
                package.selected_label().as_str(),
                candidate.selected_label().as_str()
            ));
            let identity = DependencyIdentity::from_fields(&[
                "replacement",
                &format!("{package:?}"),
                &format!("{candidate:?}"),
            ])?;
            Self::selected(
                projection,
                node,
                intent,
                summary,
                identity,
                candidate,
                false,
                None,
                vec![],
            )
        }

        fn selected(
            projection: DoctorDependencyProjection,
            node: usize,
            intent: serde_json::Value,
            summary: DiagnosticText,
            identity: DependencyIdentity,
            candidate: DependencyInstance,
            catalog_only: bool,
            advisory_revision: Option<DependencyIdentity>,
            advisory_identifiers: Vec<DiagnosticText>,
        ) -> Result<Self> {
            let graph = projection.graph();
            let paths = graph.introducing_paths(node)?;
            let selected_paths=paths.paths.iter().take(2).map(|path| {
                let nodes=path.nodes.iter().map(|index| {
                    let instance=graph.nodes().get(*index).ok_or_else(||anyhow::anyhow!("Selected introducing path changed"))?;
                    Ok(serde_json::json!({"node":index,"package":instance.selected_label(),"source":format!("{:?}",instance.source.kind())}))
                }).collect::<Result<Vec<_>>>()?;
                let edges=path.edges.iter().map(|index| {
                    let edge=graph.edges().get(*index).ok_or_else(||anyhow::anyhow!("Selected dependency constraint changed"))?;
                    let child=graph.nodes().get(edge.child).ok_or_else(||anyhow::anyhow!("Selected constraint target changed"))?;
                    Ok(serde_json::json!({"alias":if child.source.is_public_registry(){Some(DiagnosticText::new(&edge.alias))}else{None},"scope":DiagnosticText::new(&edge.scope),"requirement":if child.source.is_public_registry(){edge.requirement.as_ref().map(DiagnosticText::new)}else{None},"target":edge.target.as_ref().map(DiagnosticText::new),"workspace_inherited":edge.workspace_inherited}))
                }).collect::<Result<Vec<_>>>()?;
                Ok(serde_json::json!({"nodes":nodes,"edges":edges}))
            }).collect::<Result<Vec<_>>>()?;
            let evidence = serde_json::to_string(&serde_json::json!({
                "selected_graph":graph.identity().hex(),"selected_node":node,"source":format!("{:?}",graph.nodes().get(node).ok_or_else(||anyhow::anyhow!("Selected instance disappeared"))?.source.kind()),
                "selected_intent":intent,"introducing_paths":selected_paths,
                "paths_omitted":paths.truncated || paths.paths.len()>2,"cycle_observed":paths.cycle_observed,
                "coverage":format!("{:?}",graph.provenance().coverage),"limitations":graph.provenance().limitations,
                "verification":"Selected dependency/advisory evidence is not application exploitability or a verified fix. Collect the changed resolved graph, rerun current advisory matching and relevant explicitly reviewed build/tests after change review; unrelated tests and model assertions are insufficient.",
            }))?;
            ensure!(
                evidence.len() <= 8 * 1024,
                "Selected repair evidence exceeds the model-handoff bound; choose a smaller introducing path"
            );
            Ok(Self {
                projection,
                evidence,
                summary,
                identity,
                node,
                candidate,
                catalog_only,
                advisory_revision,
                advisory_identifiers,
            })
        }
        pub fn summary(&self) -> &DiagnosticText {
            &self.summary
        }
        pub fn evidence(&self) -> &str {
            &self.evidence
        }
        pub fn identity(&self) -> &DependencyIdentity {
            &self.identity
        }
        pub fn revalidate(&self, project: &Entity<Project>, cx: &App) -> Result<()> {
            self.projection.revalidate(project, cx)
        }
        pub fn binding(&self) -> &DoctorRepairBinding {
            self.projection.binding()
        }
    }

    #[derive(Clone)]
    pub struct DoctorDependencyVerificationBaseline {
        proposal: DoctorDependencyRepairProposal,
        validation: crate::doctor::DoctorComparisonSnapshot,
        advisory_revision: DependencyIdentity,
        advisory_fetched_at: u64,
        dataset: AdvisoryDataset,
        affected: Vec<DiagnosticText>,
        observed_at: u64,
        complete: bool,
    }

    pub struct DoctorDependencyVerification {
        pub outcome: crate::doctor::DoctorComparisonOutcome,
        pub detail: DiagnosticText,
        pub before_graph: DependencyIdentity,
        pub after_graph: DependencyIdentity,
        pub before_advisory: DependencyIdentity,
        pub after_advisory: DependencyIdentity,
        pub verification_evidence: uuid::Uuid,
        pub change: crate::doctor::DoctorReviewedChange,
        pub observed_at: u64,
    }

    impl DoctorDependencyVerification {
        pub fn report_check(&self) -> crate::doctor::DoctorCheck {
            use crate::doctor::{CheckOutcome, DoctorComparisonOutcome};
            let outcome = match self.outcome {
                DoctorComparisonOutcome::VerifiedSelectedTarget => CheckOutcome::Ready,
                DoctorComparisonOutcome::CriterionNotMet => CheckOutcome::Failed,
                DoctorComparisonOutcome::Incomplete => CheckOutcome::Warning,
                DoctorComparisonOutcome::Cancelled => CheckOutcome::Cancelled,
            };
            crate::doctor::DoctorCheck::new(
                "doctor.dependency.verification",
                "Selected dependency verification",
                outcome,
                &format!(
                    "{} Before/after graph: {} / {}; advisory: {} / {}; relevant test receipt: {}; reviewed session: {}",
                    self.detail.as_str(),
                    self.before_graph.hex(),
                    self.after_graph.hex(),
                    self.before_advisory.hex(),
                    self.after_advisory.hex(),
                    self.verification_evidence,
                    self.change
                        .session_id
                        .as_ref()
                        .map(DiagnosticText::as_str)
                        .unwrap_or("none")
                ),
                self.observed_at,
            )
        }
    }

    fn verification_slice(
        graph: &DependencyGraph,
        original: &DependencyInstance,
        candidate: &DependencyInstance,
    ) -> Result<DependencyGraph> {
        let nodes = graph
            .nodes()
            .iter()
            .filter(|node| {
                node.ecosystem == original.ecosystem
                    && (node.name == original.name || node.name == candidate.name)
            })
            .cloned()
            .collect::<Vec<_>>();
        let roots = (0..nodes.len()).collect();
        let mut provenance = graph.provenance().clone();
        provenance.limitations = vec![DiagnosticText::new(
            "Selected package instances only; this is not a whole-project security assessment",
        )];
        DependencyGraph::new(provenance, nodes, vec![], roots)
    }

    impl DoctorDependencyVerificationBaseline {
        pub fn selection(&self) -> &crate::doctor::DoctorInvestigationSelection {
            self.validation.selection()
        }

        pub fn capture(
            proposal: DoctorDependencyRepairProposal,
            validation: &crate::doctor::DoctorComparisonSnapshot,
            advisory: &AdvisorySnapshot,
            observed_at: u64,
            project: &Entity<Project>,
            cx: &App,
        ) -> Result<Self> {
            proposal.revalidate(project, cx)?;
            validation.verify_dependency_inputs_current(
                &proposal.projection.inputs,
                project,
                cx,
            )?;
            let selection = validation.selection().clone();
            selection.validate()?;
            ensure!(
                selection.scope == crate::doctor::DoctorInvestigationScope::Development
                    && selection.problem == crate::doctor::DoctorInvestigationProblem::Dependencies,
                "Select the live dependency inputs and relevant validation target before repair"
            );
            let original = proposal
                .projection
                .graph()
                .nodes()
                .get(proposal.node)
                .ok_or_else(|| anyhow::anyhow!("Selected baseline dependency is unavailable"))?;
            ensure!(
                (original.ecosystem == DependencyEcosystem::Cargo
                    && advisory.dataset == AdvisoryDataset::RustSec)
                    || (original.ecosystem == DependencyEcosystem::Maven
                        && advisory.dataset == AdvisoryDataset::OsvMaven),
                "Selected advisory dataset belongs to another ecosystem"
            );
            let graph =
                verification_slice(proposal.projection.graph(), original, &proposal.candidate)?;
            let scan = match_advisories(&graph, advisory, observed_at, || Ok(()))?;
            let affected = scan
                .findings
                .iter()
                .filter(|finding| {
                    finding.evidence.package == *original
                        && finding.evidence.state
                            == DependencyAdvisoryState::KnownAffectedDependency
                })
                .map(|finding| finding.evidence.identifier.clone())
                .collect::<Vec<_>>();
            ensure!(
                affected.len() <= 32,
                "Selected baseline advisory count exceeds its bound"
            );
            if let Some(revision) = &proposal.advisory_revision {
                ensure!(
                    revision == &scan.advisory_revision
                        && proposal
                            .advisory_identifiers
                            .iter()
                            .all(|identifier| affected.contains(identifier)),
                    "Selected advisory-fix proposal no longer matches its recorded security baseline"
                );
            }
            let complete = scan.coverage == DependencyCoverage::Complete
                && scan.freshness == AdvisoryFreshness::Fresh
                && scan.upstream_freshness == Some(AdvisoryFreshness::Fresh);
            Ok(Self {
                proposal,
                validation: validation.clone(),
                advisory_revision: scan.advisory_revision,
                advisory_fetched_at: advisory.fetched_at,
                dataset: advisory.dataset,
                affected,
                observed_at,
                complete,
            })
        }

        pub fn verify_retained_current(
            &self,
            current: DoctorDependencyProjection,
            tests: crate::doctor::DoctorComparisonSnapshot,
            evidence_id: uuid::Uuid,
            change: crate::doctor::DoctorReviewedChange,
            observed_at: u64,
            cancelled: async_channel::Receiver<()>,
            project: &Entity<Project>,
            cx: &mut App,
        ) -> gpui::Task<Result<DoctorDependencyVerification>> {
            let baseline = self.clone();
            let project = project.clone();
            cx.spawn(async move |cx| {
                let executor = cx.background_executor().clone();
                let started = executor.now();
                let cache = cx.update(|cx| {
                    current.revalidate(&project, cx)?;
                    ensure!(project.read(cx).is_local(), "Remote advisory cache verification is unsupported; no local-host substitution");
                    AdvisoryCache::for_local_project(project.read(cx))
                })?;
                let retained = cache.load(baseline.dataset, observed_at).await?;
                ensure!(retained.problem.is_none(), "Retained advisory data is unavailable or incomplete; inspect cache status and explicitly review any refresh");
                let advisory = retained.snapshot.ok_or_else(|| anyhow::anyhow!("Retained advisory data is absent; verification is incomplete, not no findings"))?;
                let revision = retained.revision.ok_or_else(|| anyhow::anyhow!("Retained advisory revision is unavailable"))?;
                let (functional, detail) = cx.update(|cx| {
                    current.revalidate(&project, cx)?;
                    tests.verify_dependency_inputs_current(&current.inputs, &project, cx)?;
                    tests.verify_dependency_change_current(&baseline.validation, &change, evidence_id, observed_at, &project, cx)
                })?;
                let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(cancelled.is_closed() || !cancelled.is_empty()));
                let matching_at = observed_at.saturating_add(executor.now().duration_since(started).as_secs());
                let result = executor.spawn({
                    let baseline = baseline.clone(); let current = current.clone(); let stop = stop.clone(); let change = change.clone();
                    async move { baseline.evaluate_guarded(current.graph(), &advisory, functional, detail, evidence_id, change, matching_at, &stop) }
                });
                let deadline = async {
                    futures::future::select(Box::pin(cancelled.recv()), Box::pin(executor.timer(std::time::Duration::from_secs(60)))).await
                };
                let mut result = match futures::future::select(result, Box::pin(deadline)).await {
                    futures::future::Either::Left((result, _)) => result?,
                    futures::future::Either::Right((reason, _)) => {
                        stop.store(true, std::sync::atomic::Ordering::Release);
                        return Err(anyhow::anyhow!(match reason {
                            futures::future::Either::Left(_) => "Dependency verification cancelled; no completed advisory match or verified repair",
                            futures::future::Either::Right(_) => "Dependency verification timed out at sixty seconds; remaining records are unevaluated and no repair is verified",
                        }));
                    }
                };
                let admission_at = observed_at.saturating_add(executor.now().duration_since(started).as_secs());
                let latest = cache.load(baseline.dataset, admission_at).await?;
                ensure!(latest.problem.is_none() && latest.revision.as_ref() == Some(&revision)
                    && !cancelled.is_closed() && cancelled.is_empty(), "Advisory data changed or verification was cancelled before result admission; rerun the selected check");
                ensure!(result.outcome != crate::doctor::DoctorComparisonOutcome::VerifiedSelectedTarget
                    || latest.snapshot.as_ref().is_some_and(|snapshot| snapshot.freshness(admission_at) == AdvisoryFreshness::Fresh && snapshot.upstream_freshness(admission_at) == Some(AdvisoryFreshness::Fresh)),
                    "Advisory publication/download data became stale before verification admission; result is incomplete");
                cx.update(|cx| {
                    current.revalidate(&project, cx)?;
                    tests.verify_dependency_inputs_current(&current.inputs, &project, cx)?;
                    tests.verify_dependency_change_current(&baseline.validation, &change, evidence_id, observed_at, &project, cx)?;
                    Ok::<_, anyhow::Error>(())
                })?;
                result.observed_at = admission_at;
                Ok(result)
            })
        }

        #[cfg(test)]
        fn evaluate(
            &self,
            current: &DependencyGraph,
            advisory: &AdvisorySnapshot,
            functional: crate::doctor::DoctorComparisonOutcome,
            functional_detail: DiagnosticText,
            evidence_id: uuid::Uuid,
            change: crate::doctor::DoctorReviewedChange,
            observed_at: u64,
            cancelled: bool,
        ) -> Result<DoctorDependencyVerification> {
            let cancelled = std::sync::atomic::AtomicBool::new(cancelled);
            self.evaluate_guarded(
                current,
                advisory,
                functional,
                functional_detail,
                evidence_id,
                change,
                observed_at,
                &cancelled,
            )
        }

        fn evaluate_guarded(
            &self,
            current: &DependencyGraph,
            advisory: &AdvisorySnapshot,
            functional: crate::doctor::DoctorComparisonOutcome,
            functional_detail: DiagnosticText,
            evidence_id: uuid::Uuid,
            change: crate::doctor::DoctorReviewedChange,
            observed_at: u64,
            cancelled: &std::sync::atomic::AtomicBool,
        ) -> Result<DoctorDependencyVerification> {
            use crate::doctor::DoctorComparisonOutcome as Outcome;
            let mut result = DoctorDependencyVerification {
                outcome: Outcome::Incomplete,
                detail: functional_detail,
                before_graph: self.proposal.projection.graph().identity(),
                after_graph: current.identity(),
                before_advisory: self.advisory_revision.clone(),
                after_advisory: advisory.identity()?,
                verification_evidence: evidence_id,
                change,
                observed_at,
            };
            if cancelled.load(std::sync::atomic::Ordering::Acquire)
                || functional == Outcome::Cancelled
            {
                result.outcome = Outcome::Cancelled;
                result.detail = DiagnosticText::new(
                    "Cancelled: selected advisory/graph/test revalidation did not finish; no verified repair",
                );
                return Ok(result);
            }
            if functional != Outcome::VerifiedSelectedTarget {
                result.outcome = functional;
                return Ok(result);
            }
            let before = self.proposal.projection.graph();
            let original = before
                .nodes()
                .get(self.proposal.node)
                .ok_or_else(|| anyhow::anyhow!("Selected baseline instance disappeared"))?;
            if advisory.dataset != self.dataset
                || !self.complete
                || self.proposal.catalog_only
                || observed_at < self.observed_at
                || current.provenance().observed_at <= self.observed_at
                || current.provenance().observed_at > observed_at
                || current.provenance().host != before.provenance().host
                || current.provenance().inputs == before.provenance().inputs
                || current.identity() == before.identity()
                || current.provenance().coverage != DependencyCoverage::Complete
                || advisory.fetched_at < self.advisory_fetched_at
            {
                result.detail = DiagnosticText::new(
                    "Incomplete: baseline/current graph, advisory or artifact source is partial, stale, unchanged, catalog-only or belongs to another host",
                );
                return Ok(result);
            }
            let Some(candidate_node) = current
                .nodes()
                .iter()
                .position(|node| node == &self.proposal.candidate)
            else {
                result.detail = DiagnosticText::new(
                    "Incomplete: the exact selected candidate/source/classifier/configuration was not observed in the changed resolved graph",
                );
                return Ok(result);
            };
            let paths = current.introducing_paths(candidate_node)?;
            if paths.paths.is_empty() || paths.truncated || paths.cycle_observed {
                result.detail = DiagnosticText::new(
                    "Incomplete: selected candidate introducing paths are unresolved, cyclic or truncated",
                );
                return Ok(result);
            }
            let selected = verification_slice(current, original, &self.proposal.candidate)?;
            let scan = match_advisories(&selected, advisory, observed_at, || {
                ensure!(
                    !cancelled.load(std::sync::atomic::Ordering::Acquire),
                    "Dependency advisory matching cancelled; remaining records are unevaluated"
                );
                Ok(())
            })?;
            if scan.coverage != DependencyCoverage::Complete
                || scan.freshness != AdvisoryFreshness::Fresh
                || scan.upstream_freshness != Some(AdvisoryFreshness::Fresh)
            {
                result.detail = DiagnosticText::new(
                    "Incomplete: current selected advisory matching is stale, unsupported, source-uncertain or incomplete",
                );
                return Ok(result);
            }
            if scan.findings.iter().any(|finding| {
                finding.evidence.state == DependencyAdvisoryState::KnownAffectedDependency
            }) {
                result.outcome = Outcome::CriterionNotMet;
                result.detail = DiagnosticText::new(
                    "Still failing: a known affected selected package instance remains in the changed resolved graph, including another transitive version",
                );
                return Ok(result);
            }
            if current.nodes().iter().any(|node| node == original) {
                result.detail = DiagnosticText::new(
                    "Incomplete: the original selected instance remains; the chosen replacement/update was not established for all introducing paths",
                );
                return Ok(result);
            }
            if original.name == self.proposal.candidate.name {
                for identifier in &self.affected {
                    let finding = scan.findings.iter().find(|finding| {
                        finding.evidence.package == self.proposal.candidate
                            && &finding.evidence.identifier == identifier
                    });
                    let Some(finding) = finding else {
                        result.detail = DiagnosticText::new(
                            "Incomplete: a baseline advisory is missing from current matching; absence is not proof of a fix",
                        );
                        return Ok(result);
                    };
                    if !advisory_match::is_documented_fix(
                        advisory,
                        finding,
                        &original.version,
                        &self.proposal.candidate.version,
                    )? {
                        result.detail = DiagnosticText::new(
                            "Incomplete: the selected advisory is withdrawn, unsupported or does not document this candidate as a fixed version",
                        );
                        return Ok(result);
                    }
                }
            } else if advisory.dataset
                != if original.ecosystem == DependencyEcosystem::Cargo {
                    AdvisoryDataset::RustSec
                } else {
                    AdvisoryDataset::OsvMaven
                }
            {
                result.detail = DiagnosticText::new(
                    "Incomplete: replacement advisory dataset does not match the selected ecosystem",
                );
                return Ok(result);
            }
            result.outcome = Outcome::VerifiedSelectedTarget;
            result.detail = DiagnosticText::new(if original.name == self.proposal.candidate.name {
                if self.affected.is_empty() {
                    "Verified selected update observation: exact candidate resolved, original instances removed, current selected-package advisory matching and relevant owner-observed tests. No findings is not proof of security; overall security, exploitability and general compatibility remain unknown"
                } else {
                    "Verified selected dependency change: exact candidate resolved, original instances removed, current documented advisory fixes and relevant owner-observed tests. This establishes only selected observations, not application exploitability, overall security or general compatibility"
                }
            } else {
                "Verified selected replacement observation: candidate resolved, original instances removed, current selected-package advisory matching and relevant owner-observed tests. A replacement is not a documented fix for the old package; overall security, exploitability and general compatibility remain unknown"
            });
            Ok(result)
        }
    }

    fn ensure_scope(directory: &ProjectPath, root: &ProjectPath) -> Result<()> {
        ensure!(
            directory.worktree_id == root.worktree_id && root.path.starts_with(&directory.path),
            "Selected dependency owner root belongs to a different worktree or scope"
        );
        Ok(())
    }
    fn ensure_saved_inputs(
        project: &Entity<Project>,
        directory: &ProjectPath,
        cx: &App,
    ) -> Result<()> {
        ensure!(
            !project
                .read(cx)
                .opened_buffers(cx)
                .into_iter()
                .any(|buffer| {
                    let buffer = buffer.read(cx);
                    buffer.is_dirty()
                        && project::File::from_dyn(buffer.file())
                            .is_some_and(|file| file.worktree_id(cx) == directory.worktree_id)
                }),
            "Unsaved selected-worktree inputs were not resolved; explicitly save relevant changes before current dependency evidence"
        );
        Ok(())
    }
    fn provenance(
        project: &Entity<Project>,
        directory: &ProjectPath,
        inputs: &DoctorInvestigationInputIdentity,
        provider: &str,
        observed_at: u64,
        cx: &App,
    ) -> Result<DependencyProvenance> {
        let owner = project.read(cx);
        Ok(DependencyProvenance {
            host: DependencyIdentity::from_fields(&[
                &format!(
                    "{:?}",
                    owner.doctor_observation(directory.worktree_id, cx).host
                ),
                &format!("{:?}", project.entity_id()),
                &format!("{:?}", directory.worktree_id),
                &format!(
                    "{:?}",
                    owner.remote_client().map(|client| client.entity_id())
                ),
            ])?,
            inputs: DependencyIdentity::from_fields(&[&format!("{inputs:?}")])?,
            provider: DiagnosticText::new(provider),
            observed_at,
            coverage: DependencyCoverage::Complete,
            limitations: vec![],
        })
    }
    fn configuration(fields: impl IntoIterator<Item = String>) -> Result<DependencyIdentity> {
        let mut digest = Sha256::new();
        let mut bytes = 0usize;
        for field in fields {
            bytes = bytes
                .checked_add(field.len())
                .ok_or_else(|| anyhow::anyhow!("Configuration identity overflow"))?;
            ensure!(
                bytes <= MAX_DEPENDENCY_GRAPH_BYTES,
                "Configuration identity exceeds bound"
            );
            digest.update((field.len() as u64).to_le_bytes());
            digest.update(field.as_bytes());
        }
        Ok(DependencyIdentity(digest.finalize().into()))
    }

    #[cfg(feature = "rust-tools")]
    fn project_cargo(
        workspace: &project::cargo_workspace::CargoWorkspaceModel,
        mut provenance: DependencyProvenance,
    ) -> Result<DependencyGraph> {
        use project::cargo_workspace::{
            CargoDependencyKind, CargoDependencyLockStatus, CargoDependencySourceKind,
            CargoSnapshotCompleteness,
        };
        let resolved = workspace
            .dependency_graph
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Cargo owner lacks resolved dependency evidence"))?;
        resolved.validate()?;
        if resolved.completeness != CargoSnapshotCompleteness::Complete {
            provenance.coverage = DependencyCoverage::Partial;
            provenance.limitations.push(DiagnosticText::new("Cargo owner resolution is incomplete; unobserved instances and paths remain unknown"));
        }
        if resolved.lock_identity.is_none()
            || resolved
                .nodes
                .iter()
                .any(|node| node.lock_status != CargoDependencyLockStatus::Locked)
        {
            provenance.limitations.push(DiagnosticText::new("Some resolved instances are not proven against a lockfile; this observation is not a reproducibility guarantee"));
        }
        provenance.limitations.push(DiagnosticText::new("Dependency version/feature inventory is distinct from toolchain readiness and API/runtime compatibility"));
        let context = format!("{:?}:{:?}", workspace.key.root, workspace.configuration);
        let nodes = resolved
            .nodes
            .iter()
            .map(|node| {
                let source = if node.crates_io {
                    DependencySource::public_registry(DependencyEcosystem::Cargo)
                } else {
                    let kind = match node.source_kind {
                        CargoDependencySourceKind::Registry => {
                            DependencySourceKind::PrivateRegistry
                        }
                        CargoDependencySourceKind::Git => DependencySourceKind::Git,
                        CargoDependencySourceKind::Path if node.workspace_manifest.is_some() => {
                            DependencySourceKind::Workspace
                        }
                        CargoDependencySourceKind::Path => DependencySourceKind::Path,
                        CargoDependencySourceKind::Other => DependencySourceKind::Unknown,
                    };
                    DependencySource::non_public(kind, &node.source_identity)?
                };
                let mut features = node.enabled_features.clone();
                features.sort();
                Ok(DependencyInstance {
                    ecosystem: DependencyEcosystem::Cargo,
                    source,
                    name: node.name.clone(),
                    version: node.version.clone(),
                    classifier: None,
                    configuration: configuration(std::iter::once(context.clone()).chain(features))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let edges = resolved
            .edges
            .iter()
            .map(|edge| {
                Ok(DependencyEdge {
                    parent: usize::try_from(edge.parent)?,
                    child: usize::try_from(edge.child)?,
                    alias: edge.alias.clone(),
                    scope: match edge.kind {
                        CargoDependencyKind::Normal => "normal",
                        CargoDependencyKind::Development => "development",
                        CargoDependencyKind::Build => "build",
                        CargoDependencyKind::Unknown => "unknown",
                    }
                    .into(),
                    requirement: edge.version_requirement.clone(),
                    target: edge.target.clone(),
                    workspace_inherited: edge.workspace_inherited,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let roots = resolved
            .roots
            .iter()
            .map(|root| usize::try_from(*root).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;
        DependencyGraph::new(provenance, nodes, edges, roots)
    }

    #[cfg(feature = "jvm-tools")]
    fn project_jvm(
        resolved: &project::jvm_project::JvmDependencyGraph,
        mut provenance: DependencyProvenance,
    ) -> Result<DependencyGraph> {
        use project::jvm_project::JvmDependencyComponentKind;
        resolved.validate()?;
        provenance.coverage = DependencyCoverage::Partial;
        provenance.limitations.push(DiagnosticText::new("JVM owner preserves selected versions and introducing paths, but repository origins and declarations are unknown; Maven coordinates do not prove Maven Central provenance"));
        if resolved.unresolved_dependencies > 0
            || !resolved.introducing_paths_complete
            || !resolved.selected_versions_known
        {
            provenance.limitations.push(DiagnosticText::new(
                "JVM owner reports unresolved versions or incomplete introducing paths",
            ));
        }
        if resolved.nodes.iter().any(|node| node.version.is_empty()) {
            provenance.limitations.push(DiagnosticText::new("Some workspace/unknown components have no observed version; their explicit unknown-version label is not a resolved release"));
        }
        let context = format!(
            "{:?}:{:?}:{}",
            resolved.module_root, resolved.build_system, resolved.configuration
        );
        let nodes = resolved
            .nodes
            .iter()
            .map(|node| {
                let name = if node.group_id.is_empty() {
                    node.artifact_id.clone()
                } else {
                    format!("{}:{}", node.group_id, node.artifact_id)
                };
                let kind = match node.kind {
                    JvmDependencyComponentKind::Workspace => DependencySourceKind::Workspace,
                    JvmDependencyComponentKind::Module | JvmDependencyComponentKind::Unknown => {
                        DependencySourceKind::Unknown
                    }
                };
                let source = DependencySource::non_public(kind, &format!("{context}:{name}"))?;
                let mut variants = node.variants.clone();
                variants.sort();
                Ok(DependencyInstance {
                    ecosystem: DependencyEcosystem::Maven,
                    source,
                    name,
                    version: if node.version.is_empty() {
                        "<unknown workspace version>".into()
                    } else {
                        node.version.clone()
                    },
                    classifier: node.classifier.clone(),
                    configuration: configuration(
                        std::iter::once(context.clone())
                            .chain(node.project_path.iter().cloned())
                            .chain(node.artifact_type.iter().cloned())
                            .chain(variants),
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let edges = resolved
            .edges
            .iter()
            .map(|edge| {
                let child = usize::try_from(edge.child)?;
                let alias = nodes
                    .get(child)
                    .ok_or_else(|| anyhow::anyhow!("JVM child instance is unavailable"))?
                    .name
                    .clone();
                Ok(DependencyEdge {
                    parent: usize::try_from(edge.parent)?,
                    child,
                    alias,
                    scope: format!(
                        "{}; optional={}; constraint={}",
                        if edge.scope.is_empty() {
                            "unknown"
                        } else {
                            &edge.scope
                        },
                        edge.optional
                            .map(|value| value.to_string())
                            .unwrap_or_else(|| "unknown".into()),
                        edge.constraint
                    ),
                    requirement: edge.requested.clone(),
                    target: None,
                    workspace_inherited: false,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let roots = resolved
            .roots
            .iter()
            .map(|root| usize::try_from(*root).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;
        DependencyGraph::new(provenance, nodes, edges, roots)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use gpui::BorrowAppContext as _;
        use util::rel_path::RelPath;

        fn path(value: &str) -> ProjectPath {
            ProjectPath {
                worktree_id: project::WorktreeId::from_usize(1),
                path: std::sync::Arc::from(RelPath::from_unix_str(value).expect("Fixture path")),
            }
        }
        fn evidence() -> DependencyProvenance {
            DependencyProvenance {
                host: DependencyIdentity::from_fields(&["host"]).expect("Host"),
                inputs: DependencyIdentity::from_fields(&["inputs"]).expect("Inputs"),
                provider: DiagnosticText::new("Existing owner fixture"),
                observed_at: 100,
                coverage: DependencyCoverage::Complete,
                limitations: vec![],
            }
        }

        #[cfg(feature = "rust-tools")]
        fn cargo_model() -> project::cargo_workspace::CargoWorkspaceModel {
            use project::cargo_workspace::{parse_metadata, workspace_from_metadata};
            let metadata = parse_metadata(include_bytes!(
                "../../project/test_data/cargo_workspace/workspace-v1.json"
            ))
            .expect("Actual Cargo metadata fixture");
            workspace_from_metadata(&metadata, |absolute| {
                absolute
                    .strip_prefix("/workspace")
                    .ok()
                    .and_then(|relative| relative.to_str())
                    .map(path)
            })
            .expect("Owner parsed workspace")
        }

        #[cfg(feature = "rust-tools")]
        #[test]
        fn doctor_dependency_projection_cargo_instances_sources_paths_and_comparison() {
            use project::cargo_workspace::*;
            let mut workspace = cargo_model();
            let resolved = workspace.dependency_graph.as_mut().expect("Resolved graph");
            let original_count = resolved.nodes.len();
            let source =
                cargo_evidence_identity(b"registry+https://github.com/rust-lang/crates.io-index");
            for (offset, version) in ["1.0.0", "2.0.0"].into_iter().enumerate() {
                let child = u32::try_from(original_count + offset).expect("Index");
                resolved.nodes.push(CargoDependencyGraphNode {
                    identity: cargo_evidence_identity(version.as_bytes()),
                    name: "public-example".into(),
                    version: version.into(),
                    source_kind: CargoDependencySourceKind::Registry,
                    source_identity: source.clone(),
                    crates_io: true,
                    workspace_manifest: None,
                    enabled_features: vec!["derive".into()],
                    lock_status: CargoDependencyLockStatus::Locked,
                });
                resolved.edges.push(CargoDependencyGraphEdge {
                    parent: 0,
                    child,
                    alias: "renamed_example".into(),
                    kind: CargoDependencyKind::Development,
                    target: Some("cfg(unix)".into()),
                    version_requirement: Some(format!("^{}", offset + 1)),
                    workspace_inherited: true,
                });
            }
            let projected = project_cargo(&workspace, evidence()).expect("Bounded projection");
            assert_eq!(projected.nodes().len(), original_count + 2);
            assert_eq!(
                projected.nodes()[original_count].source.kind(),
                DependencySourceKind::CratesIo
            );
            assert_eq!(projected.nodes()[original_count + 1].version, "2.0.0");
            assert_eq!(
                projected.nodes()[0].source.kind(),
                DependencySourceKind::Workspace
            );
            assert_eq!(
                projected
                    .introducing_paths(original_count)
                    .expect("Paths")
                    .paths
                    .len(),
                1
            );
            let edge = projected.edges().last().expect("Edge");
            assert!(edge.workspace_inherited);
            assert_eq!(edge.requirement.as_deref(), Some("^2"));
            assert_eq!(edge.target.as_deref(), Some("cfg(unix)"));
            let configuration = projected.nodes()[original_count].configuration.clone();
            workspace.dependency_graph.as_mut().expect("Graph").nodes[original_count].version =
                "1.1.0".into();
            let changed = project_cargo(&workspace, evidence()).expect("Changed version");
            assert_eq!(
                changed.nodes()[original_count].configuration,
                configuration,
                "Version-only changes can be compared under the same configuration"
            );
            assert_ne!(changed.identity(), projected.identity());
            workspace.dependency_graph.as_mut().expect("Graph").nodes[original_count]
                .source_identity = cargo_evidence_identity(b"private registry");
            assert!(
                project_cargo(&workspace, evidence()).is_err(),
                "Owner cannot promote private source as public"
            );
            let node =
                &mut workspace.dependency_graph.as_mut().expect("Graph").nodes[original_count];
            node.crates_io = false;
            let private = project_cargo(&workspace, evidence()).expect("Private source preserved");
            assert_eq!(
                private.nodes()[original_count].source.kind(),
                DependencySourceKind::PrivateRegistry
            );
            assert!(RegistryQuery::for_resolved(&private, original_count).is_err());
        }

        #[cfg(feature = "rust-tools")]
        #[test]
        fn doctor_dependency_projection_cargo_partial_and_excessive_facts_fail_closed() {
            use project::cargo_workspace::CargoSnapshotCompleteness;
            let mut workspace = cargo_model();
            workspace
                .dependency_graph
                .as_mut()
                .expect("Graph")
                .completeness = CargoSnapshotCompleteness::Partial;
            let projected = project_cargo(&workspace, evidence()).expect("Partial evidence");
            assert_eq!(projected.provenance().coverage, DependencyCoverage::Partial);
            assert!(
                projected
                    .provenance()
                    .limitations
                    .iter()
                    .any(|text| text.as_str().contains("lockfile"))
            );
            workspace
                .dependency_graph
                .as_mut()
                .expect("Graph")
                .nodes
                .first_mut()
                .expect("Node")
                .enabled_features = vec!["x".repeat(MAX_DEPENDENCY_GRAPH_BYTES + 1)];
            assert!(
                project_cargo(&workspace, evidence()).is_err(),
                "No unbounded configuration or silent truncation"
            );
            assert!(
                ensure_scope(
                    &path(""),
                    &ProjectPath {
                        worktree_id: project::WorktreeId::from_usize(2),
                        path: RelPath::empty_arc()
                    }
                )
                .is_err()
            );
        }

        #[cfg(feature = "jvm-tools")]
        #[test]
        fn doctor_dependency_projection_gradle_parallel_requests_paths_and_duplicates() {
            let mut resolved = project::jvm_project::parse_gradle_dependency_report(
                include_bytes!(
                    "../../project/test_data/groovy_jvm_gradle/doctor-selected-dependencies-v1.json"
                ),
                path(""),
                ":",
                "runtimeClasspath",
            )
            .expect("Existing reviewed Gradle owner parser");
            let graph = project_jvm(&resolved, evidence())
                .expect("Distinct requests resolving to one selected component remain valid");
            assert_eq!(graph.edges().len(), resolved.edges.len());
            let requests = graph
                .edges()
                .iter()
                .filter(|edge| edge.parent == 0 && edge.child == 1)
                .filter_map(|edge| edge.requirement.as_deref())
                .collect::<BTreeSet<_>>();
            assert_eq!(
                requests,
                BTreeSet::from(["evidence:shared:1.0", "evidence:shared:2.0"])
            );
            let paths = graph.introducing_paths(3).expect("Transitive paths");
            assert_eq!(paths.paths.len(), 2);
            assert!(
                paths.truncated,
                "Unknown JVM provenance keeps total coverage partial"
            );
            assert!(!paths.cycle_observed);
            assert_ne!(paths.paths[0].edges, paths.paths[1].edges);
            assert_eq!(graph.provenance().coverage, DependencyCoverage::Partial);
            assert_eq!(
                graph.nodes().get(1).expect("Resolved module").source.kind(),
                DependencySourceKind::Unknown
            );
            let duplicate = resolved.edges.first().expect("Existing edge").clone();
            resolved.edges.push(duplicate);
            assert!(
                project_jvm(&resolved, evidence()).is_err(),
                "Exactly duplicate requests remain rejected"
            );
        }

        #[cfg(feature = "jvm-tools")]
        #[test]
        fn doctor_dependency_projection_jvm_unknown_sources_constraints_and_catalog_confirmation() {
            use project::jvm_project::*;
            let mode = MavenDependencyReportMode {
                plugin_version: "3.7.0".into(),
                verbose: false,
                scope: None,
                includes: vec![],
                excludes: vec![],
            };
            let mut resolved = parse_maven_dependency_report(
                include_bytes!(
                    "../../project/test_data/groovy_jvm_maven/doctor-selected-dependencies-v1.json"
                ),
                path(""),
                &mode,
            )
            .expect("Existing owner parser");
            let mut additional = resolved.edges.first().expect("Edge").clone();
            additional.optional = Some(true);
            additional.constraint = true;
            additional.requested = Some("[5.0,6.0)".into());
            resolved.edges.push(additional);
            let graph = project_jvm(&resolved, evidence())
                .expect("Projection preserves distinct edge facts");
            assert_eq!(graph.provenance().coverage, DependencyCoverage::Partial);
            assert_eq!(graph.edges().len(), resolved.edges.len());
            assert!(
                graph
                    .edges()
                    .last()
                    .expect("Constraint")
                    .scope
                    .contains("optional=true; constraint=true")
            );
            for (index, node) in graph.nodes().iter().enumerate() {
                if node.source.kind() == DependencySourceKind::Unknown {
                    assert!(RegistryQuery::for_resolved(&graph, index).is_err());
                    let query =
                        RegistryQuery::for_confirmed_public_catalog(&graph, index, &node.name)
                            .expect("Separate exact public coordinate confirmation");
                    assert!(query.is_catalog_only());
                    assert_eq!(query.package().source.kind(), DependencySourceKind::Unknown);
                }
            }
            resolved.nodes.first_mut().expect("Root").version.clear();
            let unknown =
                project_jvm(&resolved, evidence()).expect("Unknown version remains explicit");
            assert!(
                unknown
                    .nodes()
                    .iter()
                    .any(|node| node.version == "<unknown workspace version>")
            );
            assert!(
                unknown
                    .provenance()
                    .limitations
                    .iter()
                    .any(|text| text.as_str().contains("no observed version"))
            );
            resolved.repository_origins_known = true;
            assert!(
                project_jvm(&resolved, evidence()).is_err(),
                "Report cannot establish registry provenance"
            );
        }

        #[cfg(feature = "rust-tools")]
        fn public_metadata_fixture(version: &str) -> serde_json::Value {
            let mut metadata: serde_json::Value = serde_json::from_slice(include_bytes!(
                "../../project/test_data/cargo_workspace/workspace-v1.json"
            ))
            .expect("Cargo metadata schema");
            let mut package = metadata["packages"][1].clone();
            let public_id = format!(
                "registry+https://github.com/rust-lang/crates.io-index#public-example@{version}"
            );
            let public_id = public_id.as_str();
            package["name"] = "public-example".into();
            package["version"] = version.into();
            package["id"] = public_id.into();
            package["source"] = "registry+https://github.com/rust-lang/crates.io-index".into();
            package["manifest_path"] = "/cargo-cache/public-example/Cargo.toml".into();
            metadata["packages"]
                .as_array_mut()
                .expect("Packages")
                .push(package);
            metadata["resolve"]["nodes"]
                .as_array_mut()
                .expect("Resolved nodes")
                .push(
                    serde_json::json!({"id":public_id,"dependencies":[],"deps":[],"features":[]}),
                );
            metadata["resolve"]["nodes"][0]["deps"].as_array_mut().expect("Direct dependencies").push(serde_json::json!({"name":"public_example","pkg":public_id,"dep_kinds":[{"kind":null,"target":null}]}));
            metadata
        }

        #[cfg(feature = "rust-tools")]
        async fn current_projection_fixture(
            cx: &mut gpui::TestAppContext,
        ) -> (Entity<Project>, ProjectPath, DoctorDependencyProjection) {
            use project::trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees};
            cx.update(|cx| {
                let settings = settings::SettingsStore::test(cx);
                cx.set_global(settings);
                trusted_worktrees::init(DbTrustedPaths::default(), cx);
                cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                    settings
                        .set_user_settings("{}", cx)
                        .expect("Observed settings")
                });
            });
            let filesystem = fs::FakeFs::new(cx.executor());
            filesystem.insert_tree("/workspace",serde_json::json!({
                "Cargo.toml":"[workspace]\nmembers = [\"member-one\", \"member-two\"]\nresolver = \"2\"\n",
                "member-one":{"Cargo.toml":"[package]\nname = \"member-one\"\nversion = \"0.1.0\"\n","src":{"lib.rs":"preserved source"}},
                "member-two":{"Cargo.toml":"[package]\nname = \"member-two\"\nversion = \"0.2.0\"\n","src":{"lib.rs":"other source"}}
            })).await;
            let project = Project::test_with_worktree_trust(
                filesystem.clone(),
                [std::path::Path::new("/workspace")],
                cx,
            )
            .await;
            let directory = project.read_with(cx, |project, cx| ProjectPath {
                worktree_id: project
                    .visible_worktrees(cx)
                    .next()
                    .expect("Root")
                    .read(cx)
                    .id(),
                path: RelPath::empty_arc(),
            });
            let trees = project.read_with(cx, |project, _| project.worktree_store());
            let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("Trust"));
            trust.update(cx, |trust, cx| {
                trust.trust(
                    &trees,
                    collections::HashSet::from_iter([PathTrust::Worktree(directory.worktree_id)]),
                    cx,
                )
            });
            cx.run_until_parked();
            // Cancel initial automatic test discovery through its existing Task
            // before sealing metadata, so advancing time does not start Cargo.
            let process_calls =
                project::cargo_workspace_store::local_cargo_process_calls_for_test();
            let tests = project.read_with(cx, |owner, _| owner.rust_test_provider_store().clone());
            let pending_discovery = tests.update(cx, |store, cx| store.refresh(cx));
            drop(pending_discovery);
            cx.run_until_parked();
            assert_eq!(
                project::cargo_workspace_store::local_cargo_process_calls_for_test(),
                process_calls,
                "Cancelled initial discovery never executes Cargo"
            );
            assert!(
                cx.update(|cx| DoctorDependencyProjection::cargo(
                    &project,
                    directory.clone(),
                    directory.clone(),
                    1,
                    100,
                    cx
                ))
                .is_err(),
                "Missing graph cannot trigger collection"
            );
            let store = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
            let manifest = ProjectPath {
                worktree_id: directory.worktree_id,
                path: std::sync::Arc::from(RelPath::from_unix_str("Cargo.toml").expect("Manifest")),
            };
            let inputs = store
                .read_with(cx, |store, cx| {
                    store.prepare_reviewed_metadata_import(manifest, cx)
                })
                .expect("Existing importer review");
            inputs
                .preflight_filesystem(filesystem.as_ref())
                .await
                .expect("Existing path checks");
            let metadata = public_metadata_fixture("1.0.0");
            store
                .update(cx, |store, cx| {
                    store.admit_reviewed_metadata_import(
                        inputs,
                        serde_json::to_vec(&metadata).expect("Metadata bytes"),
                        cx,
                    )
                })
                .await
                .expect("Existing owner admission");
            let projection = cx
                .update(|cx| {
                    DoctorDependencyProjection::cargo(
                        &project,
                        directory.clone(),
                        directory.clone(),
                        1,
                        100,
                        cx,
                    )
                })
                .expect("Current graph");
            (project, directory, projection)
        }

        #[cfg(all(feature = "rust-tools", feature = "jvm-tools"))]
        #[gpui::test]
        async fn doctor_advisory_scan_review_current_cache_cancel_deadline_and_no_http(
            cx: &mut gpui::TestAppContext,
        ) {
            use crate::doctor::*;
            use std::sync::{
                Arc,
                atomic::{AtomicUsize, Ordering},
            };
            const NOW: u64 = 1_791_187_200;
            for scenario in [
                "unapproved",
                "cancel",
                "revoked",
                "cache_changed",
                "cache_missing",
                "stale_graph",
                "stale_data",
                "unknown_publication",
                "success",
                "duplicate",
                "timeout",
                "cancel_during",
                "revoke_during",
            ] {
                // Keep each scenario's settings, trust and history owners independent.
                let mut application = cx.new_app();
                let cx = &mut application;
                let (project, directory, projection) = current_projection_fixture(cx).await;
                let calls = Arc::new(AtomicUsize::new(0));
                project
                    .read_with(cx, |owner, _| owner.client().http_client())
                    .as_fake()
                    .replace_handler({
                        let calls = calls.clone();
                        move |_, _| {
                            calls.fetch_add(1, Ordering::SeqCst);
                            async { anyhow::bail!("Local evaluation must not make HTTP requests") }
                        }
                    });
                let cache = project
                    .read_with(cx, |owner, _| AdvisoryCache::for_local_project(owner))
                    .expect("Existing cache");
                assert!(
                    cx.update(|cx| prepare_project_advisory_scan(
                        &project,
                        projection.clone(),
                        AdvisoryDataset::RustSec,
                        1,
                        NOW,
                        cx
                    ))
                    .await
                    .is_err(),
                    "Missing data cannot be a negative scan"
                );
                let record = AdvisoryRecord {
                    kind: AdvisoryArchiveKind::RustSec,
                    identifier: "RUSTSEC-2026-0001".into(),
                    document: "```toml\n[advisory]\nid = \"RUSTSEC-2026-0001\"\npackage = \"public-example\"\ndate = \"2026-10-01\"\n[versions]\npatched = [\">= 2.0.0\"]\n```\nRAW_BODY_NOT_HISTORY".into(),
                };
                let mut snapshot = super::super::advisory_cache::matching_fixture_for_cache(
                    &cache,
                    AdvisoryDataset::RustSec,
                    vec![record],
                    NOW,
                );
                if scenario == "stale_data" {
                    snapshot.fetched_at = NOW - 86401;
                    snapshot
                        .sources
                        .first_mut()
                        .expect("Source")
                        .source_modified_at = Some(snapshot.fetched_at);
                }
                if scenario == "unknown_publication" {
                    snapshot
                        .sources
                        .first_mut()
                        .expect("Source")
                        .source_modified_at = None;
                }
                let retained_revision = cache
                    .replace(snapshot.clone(), None, snapshot.fetched_at)
                    .await
                    .expect("Retained source fixture");
                let mut review = cx
                    .update(|cx| {
                        prepare_project_advisory_scan(
                            &project,
                            projection.clone(),
                            AdvisoryDataset::RustSec,
                            1,
                            NOW,
                            cx,
                        )
                    })
                    .await
                    .expect("Current exact reviewed local match");
                assert!(
                    review
                        .action()
                        .request()
                        .exact_requests()
                        .expect("No HTTP")
                        .is_empty()
                );
                assert!(
                    review
                        .action()
                        .exact_preview()
                        .expect("Concrete preview")
                        .iter()
                        .any(|line| line.contains("No network"))
                );
                assert_eq!(
                    cache
                        .load(AdvisoryDataset::RustSec, NOW)
                        .await
                        .expect("No planning replacement")
                        .revision,
                    Some(retained_revision.clone())
                );
                if scenario != "unapproved" {
                    review
                        .approve(
                            &cx.update(|cx| {
                                project_repair_binding(&project, directory.clone(), 1, cx)
                            })
                            .expect("Current binding"),
                        )
                        .expect("One explicit action approval");
                }
                let duplicate = if scenario == "duplicate" {
                    let mut review = cx
                        .update(|cx| {
                            prepare_project_advisory_scan(
                                &project,
                                projection.clone(),
                                AdvisoryDataset::RustSec,
                                1,
                                NOW,
                                cx,
                            )
                        })
                        .await
                        .expect("Equivalent proposal");
                    review
                        .approve(
                            &cx.update(|cx| {
                                project_repair_binding(&project, directory.clone(), 1, cx)
                            })
                            .expect("Binding"),
                        )
                        .expect("Approval");
                    Some(review)
                } else {
                    None
                };
                if scenario == "cache_changed" {
                    snapshot.fetched_at += 1;
                    cache
                        .replace(snapshot, Some(&retained_revision), NOW + 1)
                        .await
                        .expect("Changed retained revision");
                }
                if scenario == "cache_missing" {
                    cache
                        .clear_selected_when_authorized(
                            AdvisoryDataset::RustSec,
                            Some(&retained_revision),
                            || Ok(()),
                        )
                        .await
                        .expect("Separate explicit fixture deletion");
                }
                if scenario == "revoked" {
                    cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                        settings
                            .set_user_settings(r#"{"tab_size":3}"#, cx)
                            .expect("Changed authority")
                    });
                }
                if scenario == "stale_graph" {
                    project
                        .read_with(cx, |owner, _| owner.fs().as_fake())
                        .insert_file(
                            std::path::Path::new("/workspace/member-one/src/lib.rs"),
                            "intervening source edit".into(),
                        )
                        .await;
                    cx.run_until_parked();
                }
                let history = DoctorHistoryStore::for_test_database(&db::AppDatabase::test_new());
                let source = |review: &DoctorRepairReview<AdvisoryRefreshAction>| {
                    let mut report = DoctorReport::new(
                        DoctorScope {
                            product: DiagnosticText::new("Copper"),
                            project: DiagnosticText::new("Selected project"),
                            worktree: DiagnosticText::new("Selected worktree"),
                            host: DiagnosticText::new("Local"),
                        },
                        1,
                        NOW,
                    );
                    report
                        .push(review.before().clone())
                        .expect("Selected action");
                    report
                };
                let (cancel, cancelled) = async_channel::bounded(1);
                if scenario == "cancel" {
                    cancel.close();
                }
                let held = if matches!(scenario, "timeout" | "cancel_during" | "revoke_during") {
                    Some(cache.hold_operations_for_test().await)
                } else {
                    None
                };
                let report = source(&review);
                let (updates, progress) = async_channel::bounded(16);
                let run = cx.update(|cx| {
                    launch_advisory_refresh(
                        review,
                        &project,
                        directory.clone(),
                        1,
                        report,
                        DoctorProduct::Copper,
                        history.clone(),
                        NOW,
                        cancel.clone(),
                        cancelled,
                        updates,
                        cx,
                    )
                });
                let duplicate = duplicate.map(|review| {
                    let report = source(&review);
                    let (cancel, cancelled) = async_channel::bounded(1);
                    let (updates, _progress) = async_channel::bounded(16);
                    cx.update(|cx| {
                        launch_advisory_refresh(
                            review,
                            &project,
                            directory.clone(),
                            1,
                            report,
                            DoctorProduct::Copper,
                            history.clone(),
                            NOW,
                            cancel,
                            cancelled,
                            updates,
                            cx,
                        )
                    })
                });
                if held.is_some() {
                    let started = progress
                        .recv()
                        .await
                        .expect("Actual reviewed retained-data dispatch");
                    assert!(started.stage.as_str().contains("Evaluating exact retained"));
                    match scenario {
                        "timeout" => cx
                            .background_executor
                            .advance_clock(std::time::Duration::from_secs(61)),
                        "cancel_during" => {
                            cancel.close();
                        }
                        "revoke_during" => {
                            cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                                settings
                                    .set_user_settings(r#"{"tab_size":3}"#, cx)
                                    .expect("Revoke during wait")
                            });
                            cx.background_executor
                                .advance_clock(std::time::Duration::from_millis(200));
                        }
                        _ => unreachable!(),
                    }
                }
                let mut result = run.await;
                drop(held);
                if let Some(duplicate) = duplicate {
                    let duplicate = duplicate.await;
                    let admitted = |result: &Result<AdvisoryRefreshResult>| {
                        result.as_ref().is_ok_and(|result| result.scan.is_some())
                    };
                    assert_eq!(
                        usize::from(admitted(&result)) + usize::from(admitted(&duplicate)),
                        1,
                        "Equivalent overlapping evaluation is admitted once"
                    );
                    if admitted(&duplicate) {
                        result = duplicate;
                    }
                }
                let successes =
                    usize::from(result.as_ref().is_ok_and(|result| result.scan.is_some()));
                let success = matches!(
                    scenario,
                    "success" | "duplicate" | "stale_data" | "unknown_publication"
                );
                assert_eq!(successes, usize::from(success), "Scenario {scenario}");
                let failure = result
                    .as_ref()
                    .err()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "returned result".into());
                if success {
                    let result = result.expect("Admitted selected observations");
                    let scan = result.scan.expect("Structured transient scan");
                    assert!(scan.findings.iter().any(|finding| finding.evidence.state
                        == DependencyAdvisoryState::KnownAffectedDependency));
                    assert_ne!(
                        result.review.phase(),
                        DoctorRepairPhase::Verified,
                        "Known affected observations do not become a verified fix"
                    );
                    assert!(scan.conclusion().as_str().contains("not proof of security"));
                    if scenario == "stale_data" {
                        assert_eq!(scan.freshness, AdvisoryFreshness::Stale);
                        assert_ne!(scan.coverage, DependencyCoverage::Complete);
                    }
                    if scenario == "unknown_publication" {
                        assert_eq!(scan.upstream_freshness, None);
                        assert_ne!(scan.coverage, DependencyCoverage::Complete);
                    }
                }
                assert_eq!(calls.load(Ordering::SeqCst), 0);
                let records = history
                    .records(DoctorProduct::Copper, NOW + 61)
                    .await
                    .expect("Durable bounded receipts");
                let persisted = serde_json::to_string(&records).expect("History data");
                assert!(!persisted.contains("RAW_BODY_NOT_HISTORY"));
                assert!(!persisted.contains("preserved source"));
                if scenario == "timeout" {
                    assert!(
                        persisted.contains("timed out"),
                        "Timeout receipt: {persisted}; observed operation {failure}"
                    );
                }
                if matches!(scenario, "cancel_during" | "revoke_during") {
                    assert!(
                        records
                            .iter()
                            .any(|record| record.outcome == ReceiptOutcome::Cancelled)
                    );
                }
                if matches!(scenario, "success" | "duplicate") {
                    assert!(progress.try_recv().is_ok());
                }
            }
        }

        #[cfg(all(feature = "rust-tools", feature = "jvm-tools"))]
        #[gpui::test]
        async fn doctor_dependency_projection_native_current_stale_revoked_and_review_bound(
            cx: &mut gpui::TestAppContext,
        ) {
            use project::trusted_worktrees::{PathTrust, TrustedWorktrees};
            let (project, directory, projection) = current_projection_fixture(cx).await;
            let trees = project.read_with(cx, |project, _| project.worktree_store());
            let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("Trust"));
            assert!(cx.update(|cx| projection.revalidate(&project, cx)).is_ok());
            let absent = ProjectPath {
                worktree_id: directory.worktree_id,
                path: std::sync::Arc::from(RelPath::from_unix_str("missing").expect("Module")),
            };
            assert!(
                cx.update(|cx| DoctorDependencyProjection::cargo(
                    &project,
                    directory.clone(),
                    absent,
                    1,
                    100,
                    cx
                ))
                .is_err()
            );
            let public_node = projection
                .graph()
                .nodes()
                .iter()
                .position(|node| node.source.kind() == DependencySourceKind::CratesIo)
                .expect("Public resolved instance");
            let query = RegistryQuery::for_resolved(projection.graph(), public_node)
                .expect("Exact public query");
            assert!(
                cx.update(|cx| prepare_project_registry_query(
                    &project,
                    projection.clone(),
                    query.clone(),
                    2,
                    100,
                    cx
                ))
                .await
                .is_err(),
                "Changed view generation rejects before disclosure"
            );
            let review = cx
                .update(|cx| {
                    prepare_project_registry_query(&project, projection.clone(), query, 1, 100, cx)
                })
                .await
                .expect("Same transport review with current owner seal");
            assert_eq!(review.phase(), crate::doctor::DoctorRepairPhase::Proposed);
            assert!(
                review
                    .action()
                    .exact_preview()
                    .expect("Exact preview")
                    .first()
                    .expect("Request")
                    .contains("/public-example/versions")
            );
            let action = review.action().clone();
            assert!(
                cx.update(|cx| action.revalidate_inputs(&project, &directory, cx))
                    .is_ok()
            );
            cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                settings
                    .set_user_settings("{\"tab_size\":3}", cx)
                    .expect("Changed settings")
            });
            assert!(cx.update(|cx| projection.revalidate(&project, cx)).is_err());
            assert!(
                cx.update(|cx| action.revalidate_inputs(&project, &directory, cx))
                    .is_err()
            );
            trust.update(cx, |trust, cx| {
                trust.restrict(
                    trees.downgrade(),
                    collections::HashSet::from_iter([PathTrust::Worktree(directory.worktree_id)]),
                    cx,
                )
            });
            assert!(cx.update(|cx| projection.revalidate(&project, cx)).is_err());
        }

        #[cfg(feature = "rust-tools")]
        #[gpui::test]
        async fn doctor_dependency_verify_advisories_changed_graph_paths_freshness_and_receipts(
            cx: &mut gpui::TestAppContext,
        ) {
            use crate::doctor::*;
            const NOW: u64 = 1_791_187_200;
            let (project, _directory, projection) = current_projection_fixture(cx).await;
            let node = projection
                .graph()
                .nodes()
                .iter()
                .position(|node| node.source.kind() == DependencySourceKind::CratesIo)
                .expect("Selected public instance");
            let record = |extra: &str| AdvisoryRecord {
                kind: AdvisoryArchiveKind::RustSec,
                identifier: "RUSTSEC-2026-0001".into(),
                document: format!(
                    "```toml\n[advisory]\nid = \"RUSTSEC-2026-0001\"\npackage = \"public-example\"\ndate = \"2026-10-01\"\n{extra}\n[versions]\npatched = [\">= 2.0.0\"]\nunaffected = []\n```\nUntrusted body not diagnostic content"
                ),
            };
            let advisory = super::super::advisory_cache::matching_fixture(
                AdvisoryDataset::RustSec,
                vec![record("")],
                NOW,
            );
            let query =
                RegistryQuery::for_resolved(projection.graph(), node).expect("Selected query");
            let metadata = super::super::registry_updates::parse_metadata(query, br#"{"versions":[{"crate":"public-example","num":"2.0.0","yanked":false}],"meta":{"total":1,"next_page":null}}"#, NOW, Some(NOW), None, None).expect("Reviewed source fixture");
            let choice =
                recommend_updates(projection.graph(), &metadata, Some(&advisory), NOW, || {
                    Ok(())
                })
                .expect("Recommendations")
                .into_iter()
                .next()
                .expect("Candidate");
            assert_eq!(choice.evidence.kind, DependencyUpdateKind::AdvisoryFix);
            let proposal =
                DoctorDependencyRepairProposal::update(projection.clone(), node, &choice)
                    .expect("Selected update");
            let selection = DoctorInvestigationSelection {
                input_identity: projection.inputs.clone(),
                scope: DoctorInvestigationScope::Development,
                problem: DoctorInvestigationProblem::Dependencies,
                symptom: DiagnosticText::new("Selected affected dependency"),
                validation_target: DiagnosticText::new("selected regression tests"),
                workload: DiagnosticText::new("selected existing test workload"),
                configuration: DiagnosticText::new("exact owner configuration"),
                source_revision: DiagnosticText::new("before reviewed dependency change"),
                cache_conditions: DiagnosticText::new("observed caches"),
            };
            let validation = cx
                .update(|cx| {
                    DoctorInvestigation::new(
                        selection.clone(),
                        DoctorScope {
                            product: DiagnosticText::new("Copper"),
                            project: DiagnosticText::new("selected project"),
                            worktree: DiagnosticText::new("selected worktree"),
                            host: DiagnosticText::new("local"),
                        },
                        projection.binding().clone(),
                        NOW,
                    )
                    .expect("Selected validation review")
                    .comparison_snapshot_at(&project, cx)
                })
                .expect("Saved current validation target");
            let baseline = cx
                .update(|cx| {
                    DoctorDependencyVerificationBaseline::capture(
                        proposal.clone(),
                        &validation,
                        &advisory,
                        NOW,
                        &project,
                        cx,
                    )
                })
                .expect("Current baseline capture");
            assert_eq!(
                baseline.affected,
                vec![DiagnosticText::new("RUSTSEC-2026-0001")]
            );
            let mut changed_nodes = projection.graph().nodes().to_vec();
            *changed_nodes.get_mut(node).expect("Changed instance") =
                choice.evidence.candidate.clone();
            let mut provenance = projection.graph().provenance().clone();
            provenance.inputs = DependencyIdentity::from_fields(&[
                "actual reviewed graph inputs after change fixture",
            ])
            .expect("After identity");
            provenance.observed_at = NOW + 2;
            let graph = DependencyGraph::new(
                provenance.clone(),
                changed_nodes.clone(),
                projection.graph().edges().to_vec(),
                projection.graph().roots().to_vec(),
            )
            .expect("Changed graph");
            let after = super::super::advisory_cache::matching_fixture(
                AdvisoryDataset::RustSec,
                vec![record("")],
                NOW + 1,
            );
            let change = DoctorReviewedChange {
                before_input: selection.input_identity,
                after_input: DoctorInvestigationInputIdentity::from_fields(&["after saved inputs"])
                    .expect("After"),
                description: DiagnosticText::new("Exact reviewed update"),
                session_id: Some(DiagnosticText::new("native-session")),
            };
            let evidence_id = uuid::Uuid::new_v4();
            let evaluate =
                |graph: &DependencyGraph, data: &AdvisorySnapshot, outcome, cancelled| {
                    baseline
                        .evaluate(
                            graph,
                            data,
                            outcome,
                            DiagnosticText::new("Selected owner test fixture"),
                            evidence_id,
                            change.clone(),
                            NOW + 3,
                            cancelled,
                        )
                        .expect("Bounded verification")
                };
            let verified = evaluate(
                &graph,
                &after,
                DoctorComparisonOutcome::VerifiedSelectedTarget,
                false,
            );
            assert_eq!(
                verified.outcome,
                DoctorComparisonOutcome::VerifiedSelectedTarget
            );
            assert!(
                verified
                    .detail
                    .as_str()
                    .contains("not application exploitability")
            );
            let encoded =
                serde_json::to_string(&verified.report_check()).expect("Redacted report metadata");
            assert!(encoded.contains(&graph.identity().hex()));
            assert!(encoded.contains(&after.identity().expect("Advisory identity").hex()));
            assert!(encoded.contains("native-session"));
            assert!(!encoded.contains("Untrusted body"));
            assert!(!encoded.contains("member-one"));
            assert_eq!(
                evaluate(&graph, &after, DoctorComparisonOutcome::Incomplete, false).outcome,
                DoctorComparisonOutcome::Incomplete,
                "Unrelated or absent tests cannot verify a dependency change"
            );
            assert_eq!(
                evaluate(
                    &graph,
                    &after,
                    DoctorComparisonOutcome::CriterionNotMet,
                    false
                )
                .outcome,
                DoctorComparisonOutcome::CriterionNotMet
            );
            assert_eq!(
                evaluate(
                    &graph,
                    &after,
                    DoctorComparisonOutcome::VerifiedSelectedTarget,
                    true
                )
                .outcome,
                DoctorComparisonOutcome::Cancelled
            );
            assert_eq!(
                evaluate(
                    projection.graph(),
                    &after,
                    DoctorComparisonOutcome::VerifiedSelectedTarget,
                    false
                )
                .outcome,
                DoctorComparisonOutcome::Incomplete,
                "Historical unchanged graph cannot verify"
            );
            let mut stale = after.clone();
            stale.fetched_at = NOW - 86_400;
            assert_eq!(
                evaluate(
                    &graph,
                    &stale,
                    DoctorComparisonOutcome::VerifiedSelectedTarget,
                    false
                )
                .outcome,
                DoctorComparisonOutcome::Incomplete
            );
            let withdrawn = super::super::advisory_cache::matching_fixture(
                AdvisoryDataset::RustSec,
                vec![record("withdrawn = \"2026-10-02\"")],
                NOW + 1,
            );
            assert_eq!(
                evaluate(
                    &graph,
                    &withdrawn,
                    DoctorComparisonOutcome::VerifiedSelectedTarget,
                    false
                )
                .outcome,
                DoctorComparisonOutcome::Incomplete
            );
            let missing = super::super::advisory_cache::matching_fixture(
                AdvisoryDataset::RustSec,
                vec![AdvisoryRecord {
                    document: record("")
                        .document
                        .replace("public-example", "other-public-package"),
                    ..record("")
                }],
                NOW + 1,
            );
            assert_eq!(
                evaluate(
                    &graph,
                    &missing,
                    DoctorComparisonOutcome::VerifiedSelectedTarget,
                    false
                )
                .outcome,
                DoctorComparisonOutcome::Incomplete,
                "Missing baseline advisory is not a fix"
            );
            let mut partial_provenance = provenance.clone();
            partial_provenance.coverage = DependencyCoverage::Partial;
            let partial = DependencyGraph::new(
                partial_provenance,
                changed_nodes.clone(),
                graph.edges().to_vec(),
                graph.roots().to_vec(),
            )
            .expect("Partial graph");
            assert_eq!(
                evaluate(
                    &partial,
                    &after,
                    DoctorComparisonOutcome::VerifiedSelectedTarget,
                    false
                )
                .outcome,
                DoctorComparisonOutcome::Incomplete
            );
            for (label, roots, edges) in [
                ("unresolved", graph.roots().to_vec(), Vec::new()),
                (
                    "cyclic",
                    graph.roots().to_vec(),
                    vec![DependencyEdge {
                        parent: node,
                        child: node,
                        alias: "cycle".into(),
                        scope: "normal".into(),
                        requirement: None,
                        target: None,
                        workspace_inherited: false,
                    }],
                ),
            ] {
                let unresolved =
                    DependencyGraph::new(provenance.clone(), changed_nodes.clone(), edges, roots)
                        .expect("Bounded unresolved path fixture");
                assert_eq!(
                    evaluate(
                        &unresolved,
                        &after,
                        DoctorComparisonOutcome::VerifiedSelectedTarget,
                        false
                    )
                    .outcome,
                    DoctorComparisonOutcome::Incomplete,
                    "{label} introducing paths cannot verify a dependency repair"
                );
            }
            let old_node = changed_nodes.len();
            changed_nodes.push(projection.graph().nodes()[node].clone());
            let mut edges = graph.edges().to_vec();
            edges.push(DependencyEdge {
                parent: 0,
                child: old_node,
                alias: "old_transitive_version".into(),
                scope: "normal".into(),
                requirement: Some("^1".into()),
                target: None,
                workspace_inherited: false,
            });
            let remaining =
                DependencyGraph::new(provenance, changed_nodes, edges, graph.roots().to_vec())
                    .expect("Remaining transitive version");
            assert_eq!(
                evaluate(
                    &remaining,
                    &after,
                    DoctorComparisonOutcome::VerifiedSelectedTarget,
                    false
                )
                .outcome,
                DoctorComparisonOutcome::CriterionNotMet,
                "A fixed direct candidate cannot hide a vulnerable transitive instance"
            );
            #[cfg(feature = "jvm-tools")]
            {
                let original = DependencyInstance {
                    ecosystem: DependencyEcosystem::Maven,
                    source: DependencySource::public_registry(DependencyEcosystem::Maven),
                    name: "org.example:library".into(),
                    version: "1.0".into(),
                    classifier: None,
                    configuration: DependencyIdentity::from_fields(&[
                        "selected JVM configuration fixture",
                    ])
                    .expect("Configuration"),
                };
                let candidate = DependencyInstance {
                    version: "2.0".into(),
                    ..original.clone()
                };
                let before = DependencyGraph::new(
                    projection.graph().provenance().clone(),
                    vec![original],
                    vec![],
                    vec![0],
                )
                .expect("JVM before fixture");
                let mut after_provenance = before.provenance().clone();
                after_provenance.inputs =
                    DependencyIdentity::from_fields(&["JVM after saved inputs fixture"])
                        .expect("After identity");
                after_provenance.observed_at = NOW + 2;
                let after_graph = DependencyGraph::new(
                    after_provenance.clone(),
                    vec![candidate.clone()],
                    vec![],
                    vec![0],
                )
                .expect("JVM changed graph fixture");
                let maven_record = AdvisoryRecord { kind: AdvisoryArchiveKind::OsvMaven, identifier: "GHSA-aaaa-bbbb-cccc".into(), document: serde_json::json!({"schema_version":"1.9.1", "id":"GHSA-aaaa-bbbb-cccc", "modified":"2026-10-05T00:00:00Z", "affected":[{"package":{"ecosystem":"Maven","name":"org.example:library"},"ranges":[{"type":"ECOSYSTEM","events":[{"introduced":"0"},{"fixed":"2.0"}]}]}]}).to_string() };
                let before_data = super::super::advisory_cache::matching_fixture(
                    AdvisoryDataset::OsvMaven,
                    vec![maven_record.clone()],
                    NOW,
                );
                let data = super::super::advisory_cache::matching_fixture(
                    AdvisoryDataset::OsvMaven,
                    vec![maven_record],
                    NOW + 1,
                );
                let mut maven = baseline.clone();
                maven.proposal.projection.graph = before;
                maven.proposal.node = 0;
                maven.proposal.candidate = candidate.clone();
                maven.proposal.advisory_revision =
                    Some(before_data.identity().expect("Before revision"));
                maven.dataset = AdvisoryDataset::OsvMaven;
                maven.advisory_revision = before_data.identity().expect("Before revision");
                maven.affected = vec![DiagnosticText::new("GHSA-aaaa-bbbb-cccc")];
                let evaluate = |graph: &DependencyGraph, dataset: &AdvisorySnapshot| {
                    maven
                        .evaluate(
                            graph,
                            dataset,
                            DoctorComparisonOutcome::VerifiedSelectedTarget,
                            DiagnosticText::new("Selected JVM owner-test fixture"),
                            evidence_id,
                            change.clone(),
                            NOW + 3,
                            false,
                        )
                        .expect("JVM matching fixture")
                };
                assert_eq!(
                    evaluate(&after_graph, &data).outcome,
                    DoctorComparisonOutcome::VerifiedSelectedTarget,
                    "Maven fixed intervals use ecosystem semantics"
                );
                let mut unknown = candidate;
                unknown.source = DependencySource::non_public(
                    DependencySourceKind::Unknown,
                    "actual repository origin remains unknown",
                )
                .expect("Unknown source");
                let unknown_graph =
                    DependencyGraph::new(after_provenance, vec![unknown.clone()], vec![], vec![0])
                        .expect("Unknown source graph");
                maven.proposal.candidate = unknown;
                assert_eq!(
                    maven
                        .evaluate(
                            &unknown_graph,
                            &data,
                            DoctorComparisonOutcome::VerifiedSelectedTarget,
                            DiagnosticText::new("Fixture test"),
                            evidence_id,
                            change.clone(),
                            NOW + 3,
                            false
                        )
                        .expect("Inspectable unknown origin")
                        .outcome,
                    DoctorComparisonOutcome::Incomplete
                );
                maven.proposal.catalog_only = true;
                assert_eq!(
                    maven
                        .evaluate(
                            &unknown_graph,
                            &data,
                            DoctorComparisonOutcome::VerifiedSelectedTarget,
                            DiagnosticText::new("Fixture test"),
                            evidence_id,
                            change.clone(),
                            NOW + 3,
                            false
                        )
                        .expect("Inspectable catalog candidate")
                        .outcome,
                    DoctorComparisonOutcome::Incomplete,
                    "User public-coordinate confirmation cannot promote artifact provenance"
                );
            }
            let cache = project
                .read_with(cx, |project, _| AdvisoryCache::for_local_project(project))
                .expect("Existing native cache owner");
            let (_, no_cancel) = async_channel::unbounded();
            assert!(
                cx.update(|cx| baseline.verify_retained_current(
                    projection.clone(),
                    validation.clone(),
                    evidence_id,
                    change.clone(),
                    NOW + 4,
                    no_cancel,
                    &project,
                    cx
                ))
                .await
                .is_err(),
                "Missing retained data stays incomplete, never no findings"
            );
            let retained = super::super::advisory_cache::matching_fixture_for_cache(
                &cache,
                AdvisoryDataset::RustSec,
                vec![record("")],
                NOW + 1,
            );
            cache
                .replace(retained, None, NOW + 1)
                .await
                .expect("Reviewed fixture data retention; no live network");
            let filesystem = project.read_with(cx, |project, _| project.fs().clone());
            filesystem.atomic_write(std::path::PathBuf::from("/workspace/Cargo.toml"), "[workspace]\nmembers = [\"member-one\", \"member-two\"]\nresolver = \"2\"\n# explicitly reviewed dependency compatibility fixture change\n".to_owned()).await.expect("Explicit saved change fixture");
            cx.run_until_parked();
            let store = project.read_with(cx, |project, _| project.cargo_workspace_store().clone());
            let manifest = ProjectPath {
                worktree_id: projection.binding().directory().worktree_id,
                path: std::sync::Arc::from(RelPath::from_unix_str("Cargo.toml").expect("Manifest")),
            };
            let import = store
                .read_with(cx, |store, cx| {
                    store.prepare_reviewed_metadata_import(manifest, cx)
                })
                .expect("Existing owner reviewed changed metadata");
            import
                .preflight_filesystem(filesystem.as_ref())
                .await
                .expect("Existing path/security checks");
            store
                .update(cx, |store, cx| {
                    store.admit_reviewed_metadata_import(
                        import,
                        serde_json::to_vec(&public_metadata_fixture("2.0.0"))
                            .expect("Changed metadata fixture"),
                        cx,
                    )
                })
                .await
                .expect("Current resolved owner admission");
            let current = cx
                .update(|cx| {
                    DoctorDependencyProjection::cargo(
                        &project,
                        projection.binding().directory().clone(),
                        projection.binding().directory().clone(),
                        1,
                        NOW + 2,
                        cx,
                    )
                })
                .expect("Fresh actual Cargo owner projection");
            let mut after_selection = validation.selection().clone();
            after_selection.input_identity = DoctorInvestigationInputIdentity::from_fields(&[
                &format!("{:?}", current.inputs),
                "current selected Task execution context",
            ])
            .expect("Actual Task context identity differs from source identity");
            after_selection.source_revision =
                DiagnosticText::new("current saved dependency compatibility change");
            let mut investigation = DoctorInvestigation::new(
                after_selection.clone(),
                validation.summary().scope.clone(),
                current.binding().clone(),
                NOW + 2,
            )
            .expect("Current relevant owner fixture");
            let functional_id = investigation.record(current.binding(), &after_selection, DoctorEvidenceInput {
                owner: DoctorEvidenceOwner::Test, owner_reference: DiagnosticText::new("exact current selected provider test fixture"), classification: DoctorEvidenceClass::Observed,
                observation: DoctorEvidenceObservation::FunctionalSuccess { assertion: DiagnosticText::new("selected provider assertion identity") }, detail: DiagnosticText::new("Deterministic owner observation fixture, not a live Rust harness claim"), observed_at: NOW + 3, historical: false,
            }).expect("Selected bounded test observation");
            let tests = cx
                .update(|cx| investigation.comparison_snapshot_at(&project, cx))
                .expect("Sealed current test and source inputs");
            let reviewed = DoctorReviewedChange {
                before_input: validation.selection().input_identity.clone(),
                after_input: after_selection.input_identity,
                description: DiagnosticText::new("Explicitly reviewed selected update to 2.0.0"),
                session_id: Some(DiagnosticText::new("native-selected-update-session")),
            };
            let (cancellation, no_cancel) = async_channel::unbounded();
            let verified = cx
                .update(|cx| {
                    baseline.verify_retained_current(
                        current.clone(),
                        tests.clone(),
                        functional_id,
                        reviewed.clone(),
                        NOW + 4,
                        no_cancel,
                        &project,
                        cx,
                    )
                })
                .await
                .expect("Actual current cache/graph/comparison bridges");
            assert_eq!(
                verified.outcome,
                DoctorComparisonOutcome::VerifiedSelectedTarget
            );
            assert_eq!(verified.after_graph, current.graph().identity());
            cancellation.close();
            let (cancelled, receiver) = async_channel::unbounded();
            cancelled.close();
            assert!(
                cx.update(|cx| baseline.verify_retained_current(
                    current.clone(),
                    tests.clone(),
                    functional_id,
                    reviewed.clone(),
                    NOW + 4,
                    receiver,
                    &project,
                    cx
                ))
                .await
                .is_err(),
                "Cancelled matching cannot admit verified evidence"
            );
            cache.clear().await.expect("Explicit fixture cache removal");
            let (_cancellation, receiver) = async_channel::unbounded();
            assert!(
                cx.update(|cx| baseline.verify_retained_current(
                    current,
                    tests,
                    functional_id,
                    reviewed,
                    NOW + 4,
                    receiver,
                    &project,
                    cx
                ))
                .await
                .is_err(),
                "Missing current cache after clear cannot reuse historical source data"
            );
            cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                settings
                    .set_user_settings("{\"tab_size\":4}", cx)
                    .expect("Changed authority")
            });
            assert!(
                cx.update(|cx| DoctorDependencyVerificationBaseline::capture(
                    proposal,
                    &validation,
                    &advisory,
                    NOW,
                    &project,
                    cx
                ))
                .is_err()
            );
        }

        #[cfg(all(feature = "agentic", feature = "rust-tools"))]
        #[gpui::test]
        async fn doctor_dependency_repair_exact_intent_stale_revoked_and_once_only(
            cx: &mut gpui::TestAppContext,
        ) {
            use crate::doctor::*;
            let (model, authenticated) = crate::doctor_agent::tests::doctor_probe_fixture(cx);
            model.forbid_requests();
            let (project, directory, projection) = current_projection_fixture(cx).await;
            let graph = projection.graph();
            let node = graph
                .nodes()
                .iter()
                .position(|node| node.source.kind() == DependencySourceKind::CratesIo)
                .expect("Public instance");
            let query =
                RegistryQuery::for_resolved(graph, node).expect("Reviewed public query fixture");
            let metadata=super::super::registry_updates::parse_metadata(query,br#"{"versions":[{"crate":"public-example","num":"2.0.0","yanked":false}],"meta":{"total":1,"next_page":null}}"#,100,Some(100),None,None).expect("Approved metadata fixture, not live retrieval");
            let choice = recommend_updates(graph, &metadata, None, 100, || Ok(()))
                .expect("Owner recommendation")
                .into_iter()
                .next()
                .expect("Major candidate");
            let proposal =
                DoctorDependencyRepairProposal::update(projection.clone(), node, &choice)
                    .expect("Exact selected update");
            let mut refreshed_choice = choice.clone();
            refreshed_choice.registry_revision = DependencyIdentity::from_fields(&[
                "Newly reviewed metadata for the same selected candidate",
            ])
            .expect("New data revision");
            let refreshed =
                DoctorDependencyRepairProposal::update(projection.clone(), node, &refreshed_choice)
                    .expect("Separately reviewed metadata");
            assert_eq!(
                refreshed.identity(),
                proposal.identity(),
                "Metadata refresh cannot bypass active equivalent-action admission"
            );
            assert_ne!(
                refreshed.evidence(),
                proposal.evidence(),
                "Each review still discloses its exact data provenance"
            );
            assert!(proposal.evidence().contains("public-example 2.0.0"));
            assert!(proposal.evidence().contains("registry_revision"));
            assert!(
                !proposal.evidence().contains("/workspace")
                    && !proposal.evidence().contains("packages")
                    && proposal.evidence().len() < 8 * 1024
            );
            let report = DoctorReport::new(
                DoctorScope {
                    product: DiagnosticText::new("Copper"),
                    project: DiagnosticText::new("Selected project"),
                    worktree: DiagnosticText::new("Selected root"),
                    host: DiagnosticText::new("Local"),
                },
                1,
                100,
            );
            let mut review = cx
                .update(|cx| {
                    crate::doctor_agent::prepare_dependency_native_repair(
                        &project,
                        directory.clone(),
                        &report,
                        proposal.clone(),
                        "selected Cargo workspace build and affected tests",
                        cx,
                    )
                })
                .expect("Concrete native proposal without dispatch");
            assert_eq!(review.phase(), DoctorRepairPhase::Proposed);
            assert!(
                review
                    .action()
                    .prompt()
                    .contains("only the exact explicitly selected dependency")
            );
            assert!(
                review.action().prompt().contains("2.0.0")
                    && review
                        .action()
                        .prompt()
                        .contains("separately reviewed Doctor actions")
            );
            let repeat = cx
                .update(|cx| {
                    crate::doctor_agent::prepare_dependency_native_repair(
                        &project,
                        directory.clone(),
                        &report,
                        proposal.clone(),
                        "selected Cargo workspace build and affected tests",
                        cx,
                    )
                })
                .expect("Same explicit selection");
            assert_eq!(
                review.before().id,
                repeat.before().id,
                "Equivalent active proposals share normal admission identity"
            );
            let mut candidate = choice.evidence.candidate.clone();
            candidate.name = "replacement-example".into();
            let replacement =
                DoctorDependencyRepairProposal::replacement(projection.clone(), node, candidate)
                    .expect("Explicit manual alternative");
            let alternate = cx
                .update(|cx| {
                    crate::doctor_agent::prepare_dependency_native_repair(
                        &project,
                        directory.clone(),
                        &report,
                        replacement,
                        "selected Cargo workspace build and affected tests",
                        cx,
                    )
                })
                .expect("Replacement proposal");
            assert_ne!(
                review.before().id,
                alternate.before().id,
                "Distinct concrete choices cannot collapse into one generic action"
            );
            assert!(
                alternate
                    .action()
                    .prompt()
                    .contains("Unverified manual selection")
            );
            assert!(
                cx.update(|cx| crate::doctor_agent::prepare_dependency_native_repair(
                    &project,
                    directory.clone(),
                    &report,
                    proposal.clone(),
                    "",
                    cx
                ))
                .is_err()
            );
            let mut changed = choice.clone();
            changed.evidence.package.version = "9.9.9".into();
            assert!(
                DoctorDependencyRepairProposal::update(projection.clone(), node, &changed).is_err()
            );
            changed = choice.clone();
            changed.evidence.candidate = changed.evidence.package.clone();
            assert!(
                DoctorDependencyRepairProposal::update(projection.clone(), node, &changed).is_err()
            );
            changed = choice;
            changed.evidence.kind = DependencyUpdateKind::AdvisoryFix;
            assert!(
                DoctorDependencyRepairProposal::update(projection.clone(), node, &changed).is_err(),
                "A label without advisory provenance cannot become security evidence"
            );
            authenticated.store(false, std::sync::atomic::Ordering::SeqCst);
            assert!(
                cx.update(|cx| review.action().revalidate(cx)).is_err(),
                "Existing credential owner revocation wins"
            );
            authenticated.store(true, std::sync::atomic::Ordering::SeqCst);
            review
                .approve(projection.binding())
                .expect("Explicit approval fixture");
            cx.update(|cx| review.action().mark_prompt_submitted(cx))
                .expect("Once-only prompt admission");
            assert!(
                cx.update(|cx| review.action().mark_prompt_submitted(cx))
                    .is_err()
            );
            cx.update_global::<settings::SettingsStore, _>(|settings, cx| {
                settings
                    .set_user_settings("{\"tab_size\":3}", cx)
                    .expect("Changed current inputs")
            });
            assert!(
                cx.update(|cx| repeat.action().revalidate(cx)).is_err(),
                "Unsubmitted stale proposal stays blocked"
            );
            assert!(
                cx.update(|cx| crate::doctor_agent::prepare_dependency_native_repair(
                    &project,
                    directory,
                    &report,
                    proposal,
                    "selected Cargo tests",
                    cx
                ))
                .is_err()
            );
        }
    }
}

#[cfg(feature = "agentic")]
pub use github_evidence::{
    GithubCheckoutEvidence, GithubCheckoutRelevance, GithubCorrelatedEvidence,
    GithubDependencyCorrelation, GithubEvidenceSnapshot, GithubFindingCorrelation,
    correlate_github_evidence,
};

#[cfg(feature = "agentic")]
mod github_evidence {
    use super::*;
    use http_client::github_security::{
        GithubCodeScanningAnalysis, GithubCodeScanningEvidence, GithubDependabotEvidence,
        GithubDependabotPullRequest, GithubSecurityCollection, GithubSecurityRepository,
        GithubSecurityVerifiedMetadata,
    };

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct GithubCheckoutEvidence {
        pub reference: Option<String>,
        pub commit: Option<String>,
        pub dirty: Option<bool>,
    }
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum GithubCheckoutRelevance {
        ExactCommit,
        BranchMismatch,
        CommitMismatch,
        Dirty,
        Unknown,
    }
    impl GithubCheckoutEvidence {
        pub fn relevance(
            &self,
            reference: Option<&str>,
            commit: Option<&str>,
        ) -> GithubCheckoutRelevance {
            if let (Some(selected), Some(observed)) = (self.reference.as_deref(), reference) {
                if selected != observed {
                    return GithubCheckoutRelevance::BranchMismatch;
                }
            }
            if let (Some(selected), Some(observed)) = (self.commit.as_deref(), commit) {
                if !selected.eq_ignore_ascii_case(observed) {
                    return GithubCheckoutRelevance::CommitMismatch;
                }
            }
            if self.dirty == Some(true) {
                return GithubCheckoutRelevance::Dirty;
            }
            if self.dirty == Some(false)
                && self.commit.is_some()
                && commit.is_some()
                && self.reference.is_some()
                && reference.is_some()
            {
                GithubCheckoutRelevance::ExactCommit
            } else {
                GithubCheckoutRelevance::Unknown
            }
        }
    }
    #[derive(Clone, Debug)]
    pub struct GithubEvidenceSnapshot {
        pub repository: GithubSecurityRepository,
        pub authority: GithubSecurityVerifiedMetadata,
        pub checkout: GithubCheckoutEvidence,
        pub fetched_at: u64,
        pub dependabot: GithubSecurityCollection<GithubDependabotEvidence>,
        pub pull_requests: GithubSecurityCollection<GithubDependabotPullRequest>,
        pub code_scanning: GithubSecurityCollection<GithubCodeScanningEvidence>,
        pub analyses: GithubSecurityCollection<GithubCodeScanningAnalysis>,
    }
    impl GithubEvidenceSnapshot {
        pub fn source_freshness(value: Option<&str>, now: u64) -> DiagnosticText {
            freshness(value, now)
        }
    }
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct GithubDependencyCorrelation {
        pub local_observation: usize,
        pub github_alert: u64,
        pub exact_resolved_version: bool,
        pub checkout_relevance: GithubCheckoutRelevance,
        pub limitation: DiagnosticText,
    }
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct GithubFindingCorrelation {
        pub alert_numbers: Vec<u64>,
        pub checkout_relevance: GithubCheckoutRelevance,
        pub matching_analysis_ids: Vec<u64>,
    }
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct GithubCorrelatedEvidence {
        pub dependency_links: Vec<GithubDependencyCorrelation>,
        pub code_groups: Vec<GithubFindingCorrelation>,
        pub local_occurrences: usize,
        pub github_occurrences: usize,
        pub verified_dependency_groups: usize,
        pub limitations: Vec<DiagnosticText>,
    }
    fn timestamp(value: Option<&str>) -> Option<u64> {
        time::OffsetDateTime::parse(value?, &time::format_description::well_known::Rfc3339)
            .ok()
            .and_then(|value| u64::try_from(value.unix_timestamp()).ok())
    }
    pub(super) fn freshness(value: Option<&str>, now: u64) -> DiagnosticText {
        DiagnosticText::new(match timestamp(value) {
            Some(observed) if observed <= now && now - observed <= 86400 => {
                "Fresh source timestamp within 24 hours"
            }
            Some(observed) if observed <= now => "Stale source timestamp older than 24 hours",
            _ => "Unknown or future source timestamp",
        })
    }
    fn affected_version(
        alert: &GithubDependabotEvidence,
        package: &DependencyInstance,
    ) -> Option<bool> {
        #[cfg(feature = "rust-tools")]
        if package.ecosystem == DependencyEcosystem::Cargo {
            return semver::VersionReq::parse(alert.affected_range.as_deref()?)
                .ok()
                .zip(semver::Version::parse(&package.version).ok())
                .map(|(range, version)| range.matches(&version));
        }
        // GitHub's free-form Maven ranges are not the OSV interval schema. An
        // alias/name match alone cannot assign a resolved artifact version.
        let _ = (alert, package);
        None
    }
    pub fn correlate_github_evidence(
        snapshot: &GithubEvidenceSnapshot,
        local: &[&DependencyAdvisoryEvidence],
        now: u64,
    ) -> Result<GithubCorrelatedEvidence> {
        ensure!(
            local.len() <= 512,
            "Local finding correlation exceeds its bound"
        );
        ensure!(
            snapshot.dependabot.observations.len() <= 1000
                && snapshot.code_scanning.observations.len() <= 1000
                && snapshot.analyses.observations.len() <= 1000,
            "GitHub finding correlation exceeds its bound"
        );
        let mut result = GithubCorrelatedEvidence {
            dependency_links: vec![],
            code_groups: vec![],
            local_occurrences: local.len(),
            github_occurrences: snapshot.dependabot.observations.len()
                + snapshot.code_scanning.observations.len(),
            verified_dependency_groups: 0,
            limitations: vec![
                DiagnosticText::new(
                    "Source occurrences remain separate. Correlation is not proof of application exploitability or a verified repair",
                ),
                DiagnosticText::new(
                    "Dependabot does not establish the resolved version or analyzed commit; package/advisory links retain unknown checkout applicability",
                ),
            ],
        };
        if snapshot.fetched_at > now || now.saturating_sub(snapshot.fetched_at) > 86400 {
            result.limitations.push(DiagnosticText::new(
                "GitHub retrieval is stale or clock-uncertain",
            ));
        }
        for (index, evidence) in local.iter().enumerate() {
            evidence.validate()?;
            if !evidence.package.source.is_public_registry() {
                continue;
            }
            let aliases = evidence
                .aliases
                .iter()
                .map(|value| value.as_str())
                .chain(std::iter::once(evidence.identifier.as_str()))
                .collect::<BTreeSet<_>>();
            for alert in &snapshot.dependabot.observations {
                let ecosystem = match alert.ecosystem.as_str() {
                    "rust" => DependencyEcosystem::Cargo,
                    "maven" => DependencyEcosystem::Maven,
                    _ => continue,
                };
                if ecosystem != evidence.package.ecosystem
                    || alert.package != evidence.package.name
                    || !alert
                        .aliases
                        .iter()
                        .chain(std::iter::once(&alert.advisory_id))
                        .any(|alias| aliases.contains(alias.as_str()))
                {
                    continue;
                }
                if result.dependency_links.len() == 512 {
                    result.limitations.push(DiagnosticText::new(
                        "Correlation link limit reached; remaining associations are unevaluated",
                    ));
                    break;
                }
                let affected = affected_version(alert, &evidence.package);
                result.dependency_links.push(GithubDependencyCorrelation{local_observation:index,github_alert:alert.number,exact_resolved_version:false,
                    checkout_relevance:snapshot.checkout.relevance(snapshot.authority.default_branch.as_ref().map(|branch|format!("refs/heads/{branch}")).as_deref(),None),
                    limitation:DiagnosticText::new(format!("Package/advisory candidate link only; resolved version is absent from GitHub. Local range evaluation {:?}; local state {:?}, GitHub state {:?}, withdrawn {}, advisory time {}. No contradictory source is discarded",affected,evidence.state,alert.state,alert.withdrawn_at.is_some(),freshness(alert.advisory_updated_at.as_deref(),now).as_str()))});
            }
        }
        let mut grouped = BTreeMap::<
            (
                String,
                Option<String>,
                Option<String>,
                String,
                String,
                u64,
                Option<u64>,
                String,
                String,
            ),
            usize,
        >::new();
        for alert in &snapshot.code_scanning.observations {
            let relevance = snapshot
                .checkout
                .relevance(alert.reference.as_deref(), alert.commit.as_deref());
            let key = match (
                &alert.rule_id,
                &alert.path,
                alert.start_line,
                &alert.reference,
                &alert.commit,
            ) {
                (Some(rule), Some(path), Some(line), Some(reference), Some(commit)) => Some((
                    alert.tool.name.clone(),
                    alert.tool.version.clone(),
                    alert.tool.guid.clone(),
                    rule.clone(),
                    path.clone(),
                    line,
                    alert.end_line,
                    reference.clone(),
                    commit.clone(),
                )),
                _ => None,
            };
            if let Some(index) = key.as_ref().and_then(|key| grouped.get(key)).copied() {
                result
                    .code_groups
                    .get_mut(index)
                    .ok_or_else(|| anyhow::anyhow!("Missing correlated group"))?
                    .alert_numbers
                    .push(alert.number);
                continue;
            }
            let analyses = snapshot
                .analyses
                .observations
                .iter()
                .filter(|analysis| {
                    analysis.tool == alert.tool
                        && Some(analysis.reference.as_str()) == alert.reference.as_deref()
                        && Some(analysis.commit.as_str()) == alert.commit.as_deref()
                })
                .map(|analysis| analysis.id)
                .collect();
            let index = result.code_groups.len();
            result.code_groups.push(GithubFindingCorrelation {
                alert_numbers: vec![alert.number],
                checkout_relevance: relevance,
                matching_analysis_ids: analyses,
            });
            if let Some(key) = key {
                grouped.insert(key, index);
            }
        }
        Ok(result)
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        use http_client::github_security::{
            GithubCodeScanningTool, GithubDependabotState, GithubSecurityCoverage,
        };
        const NOW: u64 = 1_791_158_400;
        fn collection<T>(observations: Vec<T>) -> GithubSecurityCollection<T> {
            GithubSecurityCollection {
                observations,
                coverage: GithubSecurityCoverage::Complete,
                pages_completed: 1,
                bytes_collected: 100,
            }
        }
        fn snapshot() -> GithubEvidenceSnapshot {
            let alert = GithubDependabotEvidence {
                number: 1,
                state: GithubDependabotState::Open,
                ecosystem: "rust".into(),
                package: "public-example".into(),
                manifest_path: Some("Cargo.lock".into()),
                relationship: Some("indirect".into()),
                scope: Some("runtime".into()),
                advisory_id: "GHSA-aaaa-bbbb-cccc".into(),
                aliases: vec!["CVE-2026-1234".into()],
                severity: Some("high".into()),
                affected_range: Some("< 2.0.0".into()),
                first_fixed_version: Some("2.0.0".into()),
                advisory_published_at: Some("2026-10-01T00:00:00Z".into()),
                advisory_updated_at: Some("2026-10-01T00:00:00Z".into()),
                withdrawn_at: None,
                created_at: None,
                updated_at: None,
                url: "https://github.com/owner/repository/security/dependabot/1".into(),
            };
            let tool = GithubCodeScanningTool {
                name: "Scanner".into(),
                version: Some("1".into()),
                guid: None,
            };
            let code = GithubCodeScanningEvidence {
                number: 2,
                state: "open".into(),
                rule_id: Some("rust/rule".into()),
                severity: Some("warning".into()),
                security_severity: None,
                tool: tool.clone(),
                reference: Some("refs/heads/dev".into()),
                commit: Some("a".repeat(40)),
                path: Some("src/main.rs".into()),
                start_line: Some(1),
                end_line: Some(2),
                created_at: None,
                updated_at: None,
                url: "https://github.com/owner/repository/security/code-scanning/2".into(),
            };
            let analysis = GithubCodeScanningAnalysis {
                id: 3,
                tool,
                reference: "refs/heads/dev".into(),
                commit: "a".repeat(40),
                created_at: None,
                results_count: Some(1),
                rules_count: Some(10),
                category: None,
                observed_error: false,
                observed_warning: false,
            };
            GithubEvidenceSnapshot {
                repository: GithubSecurityRepository::new("owner", "repository")
                    .expect("Repository"),
                authority: GithubSecurityVerifiedMetadata {
                    account_id: 1,
                    repository_id: 2,
                    private_repository: true,
                    default_branch: Some("dev".into()),
                },
                checkout: GithubCheckoutEvidence {
                    reference: Some("refs/heads/dev".into()),
                    commit: Some("a".repeat(40)),
                    dirty: Some(false),
                },
                fetched_at: NOW,
                dependabot: collection(vec![alert]),
                pull_requests: collection(vec![]),
                code_scanning: collection(vec![code]),
                analyses: collection(vec![analysis]),
            }
        }
        fn local(version: &str) -> DependencyAdvisoryEvidence {
            DependencyAdvisoryEvidence {
                package: DependencyInstance {
                    ecosystem: DependencyEcosystem::Cargo,
                    source: DependencySource::public_registry(DependencyEcosystem::Cargo),
                    name: "public-example".into(),
                    version: version.into(),
                    classifier: None,
                    configuration: DependencyIdentity::from_fields(&["selected-config"])
                        .expect("Identity"),
                },
                authority: DependencyAdvisoryAuthority::RustSec,
                identifier: DiagnosticText::new("RUSTSEC-2026-0001"),
                aliases: vec![DiagnosticText::new("CVE-2026-1234")],
                severity: Some(DiagnosticText::new("high")),
                affected_versions: DiagnosticText::new("<2.0"),
                fixed_versions: vec![DiagnosticText::new("2.0")],
                state: DependencyAdvisoryState::KnownAffectedDependency,
                provenance: DependencyProvenance {
                    host: DependencyIdentity::from_fields(&["host"]).expect("Host"),
                    inputs: DependencyIdentity::from_fields(&["inputs"]).expect("Inputs"),
                    provider: DiagnosticText::new("RustSec"),
                    observed_at: NOW,
                    coverage: DependencyCoverage::Complete,
                    limitations: vec![],
                },
            }
        }
        #[test]
        fn doctor_github_correlate_aliases_preserve_instances_sources_and_unknown_versions() {
            let snapshot = snapshot();
            let one = local("1.0.0");
            let two = local("1.1.0");
            let result =
                correlate_github_evidence(&snapshot, &[&one, &two], NOW).expect("Correlation");
            assert_eq!(result.dependency_links.len(), 2);
            assert_eq!(result.local_occurrences, 2);
            assert_eq!(result.github_occurrences, 2);
            assert_eq!(result.verified_dependency_groups, 0);
            for link in &result.dependency_links {
                assert!(!link.exact_resolved_version);
                assert_eq!(link.checkout_relevance, GithubCheckoutRelevance::Unknown);
                assert!(link.limitation.as_str().contains("candidate"));
            }
            assert_eq!(
                result
                    .code_groups
                    .first()
                    .expect("Group")
                    .checkout_relevance,
                GithubCheckoutRelevance::ExactCommit
            );
            assert_eq!(
                result
                    .code_groups
                    .first()
                    .expect("Group")
                    .matching_analysis_ids,
                vec![3]
            );
        }
        #[test]
        fn doctor_github_correlate_private_unknown_sources_false_aliases_and_ecosystems_do_not_merge()
         {
            for kind in [
                DependencySourceKind::PrivateRegistry,
                DependencySourceKind::Unknown,
                DependencySourceKind::Git,
                DependencySourceKind::Path,
            ] {
                let mut evidence = local("1.0.0");
                evidence.authority = DependencyAdvisoryAuthority::GitHub;
                evidence.package.source =
                    DependencySource::non_public(kind, "private-owner").expect("Source");
                assert!(
                    correlate_github_evidence(&snapshot(), &[&evidence], NOW)
                        .expect("Correlation")
                        .dependency_links
                        .is_empty()
                );
            }
            let mut evidence = local("1.0.0");
            evidence.aliases = vec![DiagnosticText::new("CVE-2026-12340")];
            assert!(
                correlate_github_evidence(&snapshot(), &[&evidence], NOW)
                    .expect("Correlation")
                    .dependency_links
                    .is_empty()
            );
            let mut snapshot = snapshot();
            snapshot
                .dependabot
                .observations
                .first_mut()
                .expect("Alert")
                .ecosystem = "maven".into();
            assert!(
                correlate_github_evidence(&snapshot, &[&local("1.0.0")], NOW)
                    .expect("Correlation")
                    .dependency_links
                    .is_empty()
            );
        }
        #[test]
        fn doctor_github_correlate_contradictions_withdrawal_stale_and_dirty_evidence_remain() {
            let mut snapshot = snapshot();
            snapshot.fetched_at = NOW - 86401;
            snapshot.checkout.dirty = Some(true);
            let alert = snapshot.dependabot.observations.first_mut().expect("Alert");
            alert.state = GithubDependabotState::Fixed;
            alert.withdrawn_at = Some("2026-10-01T00:00:00Z".into());
            let mut evidence = local("1.0.0");
            evidence.state = DependencyAdvisoryState::Withdrawn;
            let result =
                correlate_github_evidence(&snapshot, &[&evidence], NOW).expect("Correlation");
            let link = result.dependency_links.first().expect("Source link");
            assert!(link.limitation.as_str().contains("Withdrawn"));
            assert!(link.limitation.as_str().contains("Fixed"));
            assert!(link.limitation.as_str().contains("withdrawn true"));
            assert_eq!(
                result
                    .code_groups
                    .first()
                    .expect("Code group")
                    .checkout_relevance,
                GithubCheckoutRelevance::Dirty
            );
            assert!(
                result
                    .limitations
                    .iter()
                    .any(|value| value.as_str().contains("stale"))
            );
            assert_eq!(
                snapshot.dependabot.observations.len(),
                1,
                "Source evidence preserved"
            );
        }
        #[test]
        fn doctor_github_correlate_code_groups_require_rule_location_ref_commit_and_keep_occurrences()
         {
            let mut snapshot = snapshot();
            let mut duplicate = snapshot
                .code_scanning
                .observations
                .first()
                .expect("Alert")
                .clone();
            duplicate.number = 4;
            duplicate.state = "fixed".into();
            snapshot.code_scanning.observations.push(duplicate.clone());
            duplicate.number = 5;
            duplicate.commit = Some("b".repeat(40));
            snapshot.code_scanning.observations.push(duplicate.clone());
            duplicate.number = 6;
            duplicate.reference = Some("refs/heads/other".into());
            snapshot.code_scanning.observations.push(duplicate.clone());
            duplicate.number = 7;
            duplicate.rule_id = None;
            snapshot.code_scanning.observations.push(duplicate);
            let result = correlate_github_evidence(&snapshot, &[], NOW).expect("Correlation");
            assert_eq!(result.code_groups.len(), 4);
            assert_eq!(result.github_occurrences, 6);
            assert_eq!(
                result.code_groups.first().expect("Exact").alert_numbers,
                vec![2, 4]
            );
            assert_eq!(
                result
                    .code_groups
                    .get(1)
                    .expect("Commit")
                    .checkout_relevance,
                GithubCheckoutRelevance::CommitMismatch
            );
            assert_eq!(
                result
                    .code_groups
                    .get(2)
                    .expect("Branch")
                    .checkout_relevance,
                GithubCheckoutRelevance::BranchMismatch
            );
            assert!(
                result
                    .code_groups
                    .get(1)
                    .expect("Different analysis")
                    .matching_analysis_ids
                    .is_empty()
            );
        }
        #[test]
        fn doctor_github_correlate_missing_authority_timestamps_and_bounds_are_unknown() {
            let mut snapshot = snapshot();
            snapshot.checkout.commit = None;
            snapshot.checkout.dirty = None;
            let result = correlate_github_evidence(&snapshot, &[], NOW).expect("Unknown");
            assert_eq!(
                result
                    .code_groups
                    .first()
                    .expect("Group")
                    .checkout_relevance,
                GithubCheckoutRelevance::Unknown
            );
            assert!(freshness(None, NOW).as_str().contains("Unknown"));
            assert!(
                freshness(Some("2027-01-01T00:00:00Z"), NOW)
                    .as_str()
                    .contains("future")
            );
            let evidence = local("1.0.0");
            assert!(correlate_github_evidence(&snapshot, &vec![&evidence; 513], NOW).is_err());
        }
    }
}

#[cfg(all(feature = "agentic", any(test, feature = "test-support")))]
pub use github_review::prepare_github_evidence_fixture;
#[cfg(feature = "agentic")]
pub use github_review::{
    DoctorGithubAction, DoctorGithubResult, DoctorGithubSources, launch_github_evidence,
    prepare_github_evidence,
};

#[cfg(all(test, feature = "agentic"))]
pub(crate) use github_review::tests::{
    fixture as doctor_github_project_fixture, install_transport as doctor_github_transport_fixture,
};

#[cfg(feature = "agentic")]
mod github_review {
    use super::*;
    use crate::doctor::{
        CheckOutcome, DoctorCheck, DoctorHistoryStore, DoctorProduct, DoctorRepairOperationOutcome,
        DoctorRepairReview, DoctorReport, comparison_source_identity, project_repair_binding,
    };
    use futures::FutureExt as _;
    use git::{GitHostingProvider as _, RemoteUrl};
    use gpui::{App, Entity};
    use http_client::github_security::{
        GithubSecurityBudget, GithubSecurityCollection, GithubSecurityCoverage,
        GithubSecurityError, GithubSecurityRepository, GithubSecuritySelection,
        SelectedGithubSecurityAuthority,
    };
    use project::{Project, ProjectPath};
    use std::{str::FromStr as _, time::Duration};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct DoctorGithubSources {
        pub dependabot: bool,
        pub code_scanning: bool,
    }
    #[derive(Clone)]
    pub struct DoctorGithubAction {
        selected: SelectedGithubSecurityAuthority,
        sources: DoctorGithubSources,
        project_identity: DependencyIdentity,
        checkout: GithubCheckoutEvidence,
    }
    impl DoctorGithubAction {
        pub fn selection(&self) -> &GithubSecuritySelection {
            self.selected.selection()
        }
        pub fn checkout(&self) -> &GithubCheckoutEvidence {
            &self.checkout
        }
        pub fn sources(&self) -> DoctorGithubSources {
            self.sources
        }
        pub fn exact_preview(&self) -> Result<Vec<String>> {
            let mut result = self.selection().reviewed_evidence_requests(
                self.checkout.reference.as_deref(),
                self.sources.dependabot,
                self.sources.code_scanning,
            )?;
            result.push(format!("Existing GITHUB_TOKEN; expected account {}; exact selected GitHub.com repository {}. Repository Metadata read, Dependabot alerts and Pull requests read{}, Code scanning alerts read{}. No new authentication or permission grants",self.selection().account(),self.selection().repository().full_name(),if self.sources.dependabot {" required"}else{" not requested"},if self.sources.code_scanning{" required"}else{" not requested"}));
            result.push(format!("Current checkout: ref {:?}, commit {:?}, dirty {:?}. Sixty-second operation; ten collection pages, one MiB per page, ten MiB total including account verification. Read-only bounded metadata; no SARIF, source, secret values, dependency submissions, service changes or workflows. Pagination must preserve exact selected endpoint/ref/filters",self.checkout.reference,self.checkout.commit,self.checkout.dirty));
            Ok(result)
        }
        pub fn revalidate_credential(&self) -> Result<()> {
            self.selected.revalidate_credential().map_err(Into::into)
        }
        pub fn revalidate(
            &self,
            project: &Entity<Project>,
            directory: &ProjectPath,
            cx: &App,
        ) -> Result<()> {
            self.selected.revalidate_credential()?;
            let (identity, checkout, repository) = project_authority(project, directory, cx)?;
            ensure!(
                identity == self.project_identity
                    && checkout == self.checkout
                    && repository
                        .full_name()
                        .eq_ignore_ascii_case(&self.selection().repository().full_name()),
                "Selected repository/origin/ref/HEAD/source or host changed; review a new GitHub read"
            );
            Ok(())
        }
    }
    fn exact_repository(origin: &str) -> Result<GithubSecurityRepository> {
        ensure!(
            origin.len() <= 4096,
            "Selected Git origin exceeds its bound"
        );
        let remote = RemoteUrl::from_str(origin)
            .map_err(|_| anyhow::anyhow!("Unsupported selected Git origin"))?;
        ensure!(
            remote.host_str() == Some("github.com")
                && matches!(remote.scheme(), "https" | "ssh")
                && remote.port().is_none()
                && remote.password().is_none()
                && (remote.scheme() == "https" && remote.username().is_empty()
                    || remote.scheme() == "ssh" && remote.username() == "git")
                && remote.query().is_none()
                && remote.fragment().is_none(),
            "Select an exact GitHub.com repository origin; custom hosts, embedded credentials and origin substitutions are unsupported"
        );
        let segments = remote
            .path_segments()
            .ok_or_else(|| anyhow::anyhow!("Missing selected Git repository path"))?
            .collect::<Vec<_>>();
        ensure!(
            segments.len() == 2,
            "Selected Git origin must contain exactly one owner/repository"
        );
        let parsed = git_hosting_providers::Github::public_instance()
            .parse_remote_url(origin)
            .ok_or_else(|| {
                anyhow::anyhow!("Existing GitHub owner could not parse selected origin")
            })?;
        let owner = segments
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Missing selected owner"))?;
        let name = segments
            .get(1)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Missing selected repository"))?;
        let name = name.strip_suffix(".git").unwrap_or(name);
        let strict = GithubSecurityRepository::new(owner, name)?;
        ensure!(
            parsed.owner.as_ref() == owner && parsed.repo.as_ref() == name,
            "Existing remote parser and exact GitHub authority differ"
        );
        Ok(strict)
    }
    fn project_authority(
        project: &Entity<Project>,
        directory: &ProjectPath,
        cx: &App,
    ) -> Result<(
        DependencyIdentity,
        GithubCheckoutEvidence,
        GithubSecurityRepository,
    )> {
        let owner = project.read(cx);
        ensure!(
            !owner.is_isolated_recipe_test(),
            "Isolated recipe tests cannot access GitHub credentials or security metadata"
        );
        let observation = owner.doctor_observation(directory.worktree_id, cx);
        ensure!(
            observation.host == project::DoctorExecutionHost::Local
                && observation.supports_reviewed_probe(),
            "GitHub evidence requires selected authorized local-host credentials. Remote/collaboration credential transport is unsupported; no local fallback"
        );
        ensure!(
            directory.path.is_empty(),
            "Select a worktree root for GitHub evidence"
        );
        let repository = owner
            .git_store()
            .read(cx)
            .repository_and_path_for_project_path(directory, cx)
            .map(|(repository, _)| repository)
            .ok_or_else(|| anyhow::anyhow!("Selected project has no observed Git repository"))?;
        let repository_owner = repository.read(cx);
        ensure!(
            repository_owner.pending_ops().next().is_none(),
            "Selected repository has an active Git operation; wait and review current authority"
        );
        let snapshot = repository_owner.snapshot();
        let origin = snapshot
            .remote_origin_url
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Selected repository has no observed origin"))?;
        let selected = exact_repository(origin)?;
        let reference = snapshot
            .branch
            .as_ref()
            .map(|branch| branch.ref_name.to_string());
        let commit = snapshot
            .head_commit
            .as_ref()
            .map(|commit| commit.sha.to_string());
        let dirty_buffers = owner.opened_buffers(cx).into_iter().any(|buffer| {
            let buffer = buffer.read(cx);
            buffer.is_dirty()
                && project::File::from_dyn(buffer.file())
                    .is_some_and(|file| file.worktree_id(cx) == directory.worktree_id)
        });
        let checkout = GithubCheckoutEvidence {
            reference,
            commit,
            dirty: Some(!snapshot.statuses_by_path.is_empty() || dirty_buffers),
        };
        let source = comparison_source_identity(project, directory, cx)?;
        let identity = DependencyIdentity::from_fields(&[
            &format!("{:?}", repository.entity_id()),
            &format!("{:?}", snapshot.id),
            origin,
            &snapshot.scan_id.to_string(),
            &format!("{checkout:?}"),
            &format!("{source:?}"),
        ])?;
        Ok((identity, checkout, selected))
    }
    pub fn prepare_github_evidence(
        project: &Entity<Project>,
        directory: ProjectPath,
        generation: u64,
        selection: GithubSecuritySelection,
        sources: DoctorGithubSources,
        now: u64,
        cx: &App,
    ) -> Result<DoctorRepairReview<DoctorGithubAction>> {
        let selected = SelectedGithubSecurityAuthority::capture_selected_environment(selection)?;
        prepare_selected(project, directory, generation, selected, sources, now, cx)
    }
    #[cfg(any(test, feature = "test-support"))]
    pub fn prepare_github_evidence_fixture(
        project: &Entity<Project>,
        directory: ProjectPath,
        generation: u64,
        selection: GithubSecuritySelection,
        sources: DoctorGithubSources,
        now: u64,
        reader: std::sync::Arc<dyn Fn() -> Option<String> + Send + Sync>,
        cx: &App,
    ) -> Result<DoctorRepairReview<DoctorGithubAction>> {
        let selected =
            SelectedGithubSecurityAuthority::capture_fixture_credential(selection, reader)?;
        prepare_selected(project, directory, generation, selected, sources, now, cx)
    }
    fn prepare_selected(
        project: &Entity<Project>,
        directory: ProjectPath,
        generation: u64,
        selected: SelectedGithubSecurityAuthority,
        sources: DoctorGithubSources,
        now: u64,
        cx: &App,
    ) -> Result<DoctorRepairReview<DoctorGithubAction>> {
        let binding = project_repair_binding(project, directory.clone(), generation, cx)?;
        let (project_identity, checkout, repository) = project_authority(project, &directory, cx)?;
        ensure!(
            repository
                .full_name()
                .eq_ignore_ascii_case(&selected.selection().repository().full_name()),
            "Explicitly selected repository differs from this Project's origin"
        );
        let action = DoctorGithubAction {
            selected,
            sources,
            project_identity,
            checkout,
        };
        let preview = action.exact_preview()?;
        let identity = DependencyIdentity::from_fields(
            &preview.iter().map(String::as_str).collect::<Vec<_>>(),
        )?
        .hex();
        DoctorRepairReview::new(
            action,
            binding,
            DoctorCheck::new(
                &format!("github.evidence.{identity}"),
                "Optional selected GitHub security evidence",
                CheckOutcome::Untested,
                "No reviewed GitHub read or account verification has run. Local/offline Doctor checks remain available",
                now,
            ),
            DiagnosticText::new(
                "Read selected existing GitHub account/repository and requested bounded security metadata. Network and existing credential use are explicit; no permission, dependency, source, service, workflow or scheduling changes. Current authority is checked before requests and late admission; raw responses are excluded from history and model handoffs",
            ),
        )
    }
    pub struct DoctorGithubResult {
        pub review: DoctorRepairReview<DoctorGithubAction>,
        pub snapshot: Option<GithubEvidenceSnapshot>,
    }
    fn not_requested<T>() -> GithubSecurityCollection<T> {
        GithubSecurityCollection {
            observations: vec![],
            coverage: GithubSecurityCoverage::NotRequested,
            pages_completed: 0,
            bytes_collected: 0,
        }
    }
    pub fn launch_github_evidence(
        mut review: DoctorRepairReview<DoctorGithubAction>,
        project: &Entity<Project>,
        directory: ProjectPath,
        generation: u64,
        report: DoctorReport,
        product: DoctorProduct,
        history: DoctorHistoryStore,
        now: u64,
        cancellation: async_channel::Sender<()>,
        cancelled: async_channel::Receiver<()>,
        updates: async_channel::Sender<DiagnosticText>,
        cx: &mut App,
    ) -> gpui::Task<Result<DoctorGithubResult>> {
        let project = project.downgrade();
        cx.spawn(async move |cx| {
            let binding=cx.update(|cx| {
                let project=project.upgrade().ok_or_else(||anyhow::anyhow!("Selected project closed"))?;
                review.action().revalidate(&project,&directory,cx)?;
                project_repair_binding(&project,directory.clone(),generation,cx)
            })?;
            review.record_before_execution(&binding,report,product,&history,now,&cancelled).await?;
            let executor=cx.background_executor().clone();let started=executor.now();
            let action=review.action().clone();
            let monitor=cx.spawn({let project=project.clone();let directory=directory.clone();let binding=binding.clone();let action=action.clone();let cancellation=cancellation.clone();let executor=executor.clone();async move|cx| {
                loop {
                    let authorized=cx.update(|cx| {let owner=project.upgrade().ok_or_else(||anyhow::anyhow!("Selected project closed"))?;action.revalidate(&owner,&directory,cx)?;ensure!(project_repair_binding(&owner,directory.clone(),generation,cx)?==binding,"GitHub review inputs changed");Ok::<_,anyhow::Error>(())});
                    if authorized.is_err() {cancellation.close();return;}
                    executor.timer(Duration::from_millis(100)).await;
                }
            }});
            let client=cx.update(|cx|project.upgrade().ok_or_else(||anyhow::anyhow!("Selected project closed")).map(|owner|owner.read(cx).client().http_client()))?;
            let mut checkpoint={let app=cx.clone();let project=project.clone();let action=action.clone();let directory=directory.clone();let binding=binding.clone();let cancelled=cancelled.clone();move|| {
                let app=app.clone();let project=project.clone();let action=action.clone();let directory=directory.clone();let binding=binding.clone();let cancelled=cancelled.clone();async move {
                    if cancelled.is_closed() || !cancelled.is_empty() {return Err(GithubSecurityError::Cancelled);}
                    app.update(|cx| {let owner=project.upgrade().ok_or(GithubSecurityError::AuthorityChanged)?;action.revalidate(&owner,&directory,cx).map_err(|_|GithubSecurityError::AuthorityChanged)?;let current=project_repair_binding(&owner,directory,generation,cx).map_err(|_|GithubSecurityError::AuthorityChanged)?;if current!=binding{return Err(GithubSecurityError::AuthorityChanged);}Ok(())})
                }
            }};
            let mut snapshot=None;
            let operation=async {
                updates.try_send(DiagnosticText::new("Verifying explicitly selected GitHub account and repository")).map_err(|_|anyhow::anyhow!("GitHub progress view closed or full"))?;
                let verified=action.selected.clone().verify_with_checkpoint(client.clone(),&mut checkpoint).await?;
                let mut budget=GithubSecurityBudget::after_verification(&verified)?;
                snapshot=Some(GithubEvidenceSnapshot{repository:verified.selection().repository().clone(),authority:verified.metadata().clone(),checkout:action.checkout.clone(),fetched_at:now,dependabot:not_requested(),pull_requests:not_requested(),code_scanning:not_requested(),analyses:not_requested()});
                let snapshot=snapshot.as_mut().ok_or_else(||anyhow::anyhow!("Missing verified GitHub receipt"))?;
                if action.sources.dependabot {
                    updates.try_send(DiagnosticText::new("Reading selected Dependabot alerts and update-PR candidates")).map_err(|_|anyhow::anyhow!("GitHub progress view closed or full"))?;
                    snapshot.dependabot=verified.read_dependabot_with_checkpoint(client.clone(),&mut budget,&mut checkpoint).await;
                    snapshot.pull_requests=verified.read_dependabot_pull_requests_with_checkpoint(client.clone(),&mut budget,&mut checkpoint).await;
                }
                if action.sources.code_scanning {
                    updates.try_send(DiagnosticText::new("Reading selected-ref code-scanning alerts and analysis provenance")).map_err(|_|anyhow::anyhow!("GitHub progress view closed or full"))?;
                    let reference=action.checkout.reference.as_deref().ok_or_else(||anyhow::anyhow!("Selected checkout has no observed branch reference"))?;
                    snapshot.code_scanning=verified.read_code_scanning_with_checkpoint(reference,client.clone(),&mut budget,&mut checkpoint).await;
                    snapshot.analyses=verified.read_code_scanning_analyses_with_checkpoint(reference,client.clone(),&mut budget,&mut checkpoint).await;
                }
                checkpoint().await?;
                Ok::<_,anyhow::Error>(())
            };
            let stop=async {futures::future::select(cancelled.recv().boxed_local(),executor.timer(Duration::from_secs(60)).boxed_local()).await};
            let result=match futures::future::select(operation.boxed_local(),stop.boxed_local()).await {
                futures::future::Either::Left((result,_))=>result,
                futures::future::Either::Right((futures::future::Either::Left(_),_))=>Err(GithubSecurityError::Cancelled.into()),
                futures::future::Either::Right((futures::future::Either::Right(_),_))=>Err(GithubSecurityError::TimedOut.into()),
            };
            drop(monitor);
            let observed_at=now.saturating_add(executor.now().duration_since(started).as_secs());
            let current=cx.update(|cx| {let project=project.upgrade().ok_or_else(||anyhow::anyhow!("Selected project closed"))?;action.revalidate(&project,&directory,cx)?;let current=project_repair_binding(&project,directory,generation,cx)?;ensure!(current==binding,"Reviewed authority changed at admission");Ok(current)});
            let is_cancelled=cancelled.is_closed() || !cancelled.is_empty();
            let reached=snapshot.as_ref().map(|snapshot|snapshot.dependabot.pages_completed+snapshot.pull_requests.pages_completed+snapshot.code_scanning.pages_completed+snapshot.analyses.pages_completed).unwrap_or(0);
            let (outcome,detail)=if current.is_err() {snapshot=None;(DoctorRepairOperationOutcome::Failed,format!("GitHub authority changed; reached {reached} pages are not admitted as current evidence. Review current scope; no replay or remote mutation"))}
                else if result.is_err() {snapshot=None;(if is_cancelled {DoctorRepairOperationOutcome::Cancelled}else{DoctorRepairOperationOutcome::Failed},format!("GitHub read incomplete: {}. Reached {reached} pages; no absence-of-findings or verified-fix claim",result.err().map(|error|error.to_string()).unwrap_or_else(||"Unknown retrieval failure".into())))}
                else {if let Some(snapshot)=snapshot.as_mut(){snapshot.fetched_at=observed_at;}
                    let detail=snapshot.as_ref().map(|snapshot|format!("Read-only GitHub metadata observed: Dependabot {:?}, PR candidates {:?}, code alerts {:?}, analyses {:?}; {reached} pages. Missing commit, analysis coverage, stale/failed sources and checkout relevance remain explicit. Retrieval never proves project security or a repair",snapshot.dependabot.coverage,snapshot.pull_requests.coverage,snapshot.code_scanning.coverage,snapshot.analyses.coverage)).unwrap_or_else(||"No verified repository metadata was admitted".into());
                    (DoctorRepairOperationOutcome::Completed,detail)};
            review.operation_finished(outcome,DiagnosticText::new(&detail))?;
            if outcome!=DoctorRepairOperationOutcome::Cancelled {
                match current {Ok(current)=>review.observe_recheck(&current,DoctorCheck::new(review.before().id.as_str(),"Observed read-only GitHub evidence",CheckOutcome::Untested,&detail,observed_at))?,Err(_)=>review.recheck_unavailable(DiagnosticText::new(&detail))?}
            }
            review.persist_outcome(product,&history,None,observed_at).await?;updates.close();
            Ok(DoctorGithubResult{review,snapshot})
        })
    }
    #[cfg(test)]
    pub(crate) mod tests {
        use super::*;
        use fs::Fs as _;
        use gpui::{BorrowAppContext as _, TestAppContext};
        use std::{
            path::Path,
            sync::{
                Arc, Mutex,
                atomic::{AtomicUsize, Ordering},
            },
        };
        const NOW: u64 = 1_791_158_400;
        pub(crate) async fn fixture(
            cx: &mut TestAppContext,
        ) -> (Entity<Project>, ProjectPath, Arc<fs::FakeFs>) {
            use project::trusted_worktrees::{self, DbTrustedPaths, PathTrust, TrustedWorktrees};
            cx.update(|cx| {
                let settings = settings::SettingsStore::test(cx);
                cx.set_global(settings);
                trusted_worktrees::init(DbTrustedPaths::default(), cx);
            });
            let filesystem = fs::FakeFs::new(cx.background_executor.clone());
            filesystem
                .insert_tree(
                    "/github-review",
                    serde_json::json!({".git":{},"source.txt":"unrelated edits"}),
                )
                .await;
            filesystem.set_remote_for_repo(
                Path::new("/github-review/.git"),
                "origin",
                "git@github.com:selected-owner/selected-repository.git",
            );
            filesystem.set_branch_name(Path::new("/github-review/.git"), Some("dev"));
            filesystem.set_head_for_repo(
                Path::new("/github-review/.git"),
                &[("source.txt", "unrelated edits".into())],
                &"a".repeat(40),
            );
            filesystem.set_index_for_repo(
                Path::new("/github-review/.git"),
                &[("source.txt", "unrelated edits".into())],
            );
            let project = Project::test_with_worktree_trust(
                filesystem.clone(),
                [Path::new("/github-review")],
                cx,
            )
            .await;
            let directory = project.read_with(cx, |owner, cx| ProjectPath {
                worktree_id: owner
                    .visible_worktrees(cx)
                    .next()
                    .expect("Root")
                    .read(cx)
                    .id(),
                path: util::rel_path::RelPath::empty_arc(),
            });
            let worktrees = project.read_with(cx, |owner, _| owner.worktree_store());
            let trust = cx.update(|cx| TrustedWorktrees::try_get_global(cx).expect("Trust owner"));
            trust.update(cx, |trust, cx| {
                trust.trust(
                    &worktrees,
                    collections::HashSet::from_iter([PathTrust::Worktree(directory.worktree_id)]),
                    cx,
                )
            });
            project
                .update(cx, |project, cx| project.git_scans_complete(cx))
                .await;
            cx.run_until_parked();
            (project, directory, filesystem)
        }
        #[gpui::test]
        async fn doctor_github_review_native_reference_preserves_repository_identity(
            cx: &mut TestAppContext,
        ) {
            for branch in ["refs/heads/dev", "refs/heads/feature/security-review"] {
                let mut application = cx.new_app();
                let (project, directory, filesystem) = fixture(&mut application).await;
                filesystem.set_branch_name(Path::new("/github-review/.git"), Some(branch));
                project
                    .update(&mut application, |project, cx| {
                        project.git_scans_complete(cx)
                    })
                    .await;
                application.run_until_parked();
                let credential = Arc::new(Mutex::new(Some("FIXTURE_SELECTED_GITHUB_TOKEN".into())));
                let review = application
                    .update(|cx| {
                        prepare_selected(
                            &project,
                            directory,
                            1,
                            selected(credential),
                            DoctorGithubSources {
                                dependabot: false,
                                code_scanning: true,
                            },
                            NOW,
                            cx,
                        )
                    })
                    .expect("Current native ref review");
                assert_eq!(
                    review.action().checkout().reference.as_deref(),
                    Some(branch)
                );
                let preview = review
                    .action()
                    .exact_preview()
                    .expect("Exact pre-approval requests")
                    .join(" ");
                assert!(!preview.contains("refs%2Fheads%2Frefs%2Fheads"));
                let expected =
                    url::Url::parse("https://api.github.com/").expect("Fixed fixture origin");
                let mut expected = expected;
                expected.query_pairs_mut().append_pair("ref", branch);
                assert!(
                    preview.contains(expected.query().expect("Reviewed native ref")),
                    "Exact preview preserves canonical ref"
                );
                assert_eq!(review.phase(), crate::doctor::DoctorRepairPhase::Proposed);
            }
        }
        fn selected(credential: Arc<Mutex<Option<String>>>) -> SelectedGithubSecurityAuthority {
            let selection = GithubSecuritySelection::new(
                "selected-user",
                GithubSecurityRepository::new("selected-owner", "selected-repository")
                    .expect("Repo"),
            )
            .expect("Account");
            SelectedGithubSecurityAuthority::capture_fixture_credential(
                selection,
                Arc::new(move || credential.lock().expect("Credential fixture").clone()),
            )
            .expect("Explicit selected fixture authority")
        }
        fn proposal(
            project: &Entity<Project>,
            directory: &ProjectPath,
            credential: Arc<Mutex<Option<String>>>,
            cx: &App,
        ) -> DoctorRepairReview<DoctorGithubAction> {
            prepare_selected(
                project,
                directory.clone(),
                1,
                selected(credential),
                DoctorGithubSources {
                    dependabot: true,
                    code_scanning: true,
                },
                NOW,
                cx,
            )
            .expect("Native proposal")
        }
        fn report(review: &DoctorRepairReview<DoctorGithubAction>) -> DoctorReport {
            let mut report = DoctorReport::new(
                crate::doctor::DoctorScope {
                    product: DiagnosticText::new("Copper"),
                    project: DiagnosticText::new("selected project"),
                    worktree: DiagnosticText::new("selected worktree"),
                    host: DiagnosticText::new("Local"),
                },
                1,
                NOW,
            );
            report
                .push(review.before().clone())
                .expect("Selected report");
            report
        }
        fn response(
            value: serde_json::Value,
        ) -> Result<http_client::Response<http_client::AsyncBody>> {
            Ok(http_client::Response::builder()
                .status(200)
                .header("Content-Type", "application/json")
                .body(serde_json::to_vec(&value)?.into())?)
        }
        pub(crate) fn install_transport(
            project: &Entity<Project>,
            calls: Arc<AtomicUsize>,
            pending: bool,
            cx: &App,
        ) {
            project.read(cx).client().http_client().as_fake().replace_handler(move|_,request| {
                let calls=calls.clone();async move {
                    calls.fetch_add(1,Ordering::SeqCst);assert_eq!(request.method(),http_client::Method::GET);
                    assert_eq!(request.extensions().get::<http_client::RedirectPolicy>(),Some(&http_client::RedirectPolicy::NoFollow));
                    if pending {return futures::future::pending().await;}
                    let path=request.uri().path();
                    if path=="/user" {response(serde_json::json!({"id":42,"login":"selected-user","private":"RAW_CREDENTIAL_NOT_HISTORY"}))}
                    else if path=="/repos/selected-owner/selected-repository" {response(serde_json::json!({"id":123,"full_name":"selected-owner/selected-repository","private":true,"default_branch":"dev","raw":"RAW_SOURCE_NOT_HISTORY"}))}
                    else {assert!(path.starts_with("/repos/selected-owner/selected-repository/"));assert!(!path.contains("secret-scanning"));response(serde_json::json!([]))}
                }
            });
        }
        #[test]
        fn doctor_github_review_remote_parser_requires_exact_existing_github_authority() {
            for origin in [
                "git@github.com:owner/repository.git",
                "ssh://git@github.com/owner/repository.git",
                "https://github.com/owner/repository.git",
            ] {
                assert_eq!(
                    exact_repository(origin)
                        .expect("Supported existing parser")
                        .full_name(),
                    "owner/repository"
                );
            }
            for origin in [
                "git@github.enterprise.example:owner/repository.git",
                "https://github.com/owner/repository/extra",
                "https://github.com/owner/repository?token=x",
                "https://secret@github.com/owner/repository.git",
                "https://github.com/owner%2fother/repository",
                "file:///owner/repository",
            ] {
                assert!(exact_repository(origin).is_err(), "{origin}");
            }
        }
        #[gpui::test]
        async fn doctor_github_review_explicit_approval_current_authority_receipts_and_no_security_claim(
            cx: &mut TestAppContext,
        ) {
            for scenario in [
                "unapproved",
                "success",
                "precancel",
                "revoked",
                "origin_changed",
                "settings_changed",
                "source_changed",
            ] {
                let mut application = cx.new_app();
                let cx = &mut application;
                let (project, directory, filesystem) = fixture(cx).await;
                let credential =
                    Arc::new(Mutex::new(Some("FIXTURE_CREDENTIAL_NOT_HISTORY".into())));
                let calls = Arc::new(AtomicUsize::new(0));
                cx.update(|cx| install_transport(&project, calls.clone(), false, cx));
                let mut review =
                    cx.update(|cx| proposal(&project, &directory, credential.clone(), cx));
                assert_eq!(calls.load(Ordering::SeqCst), 0);
                let report = report(&review);
                if scenario != "unapproved" {
                    review
                        .approve(
                            &cx.update(|cx| {
                                project_repair_binding(&project, directory.clone(), 1, cx)
                            })
                            .expect("Current scope"),
                        )
                        .expect("Explicit approval");
                }
                let database = db::AppDatabase::test_new();
                let history = DoctorHistoryStore::for_test_database(&database);
                let (cancel, cancelled) = async_channel::bounded(1);
                match scenario {
                    "precancel" => {
                        cancel.close();
                    }
                    "revoked" => {
                        *credential.lock().expect("Credential fixture") = None;
                    }
                    "origin_changed" => {
                        filesystem.set_remote_for_repo(
                            Path::new("/github-review/.git"),
                            "origin",
                            "git@github.com:other-owner/other-repository.git",
                        );
                        project
                            .update(cx, |project, cx| project.git_scans_complete(cx))
                            .await;
                        cx.run_until_parked();
                    }
                    "settings_changed" => cx.update(|cx| {
                        cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                            store
                                .set_user_settings("{\"vim_mode\":true}", cx)
                                .expect("Changed settings");
                        })
                    }),
                    "source_changed" => {
                        filesystem
                            .insert_file(
                                Path::new("/github-review/source.txt"),
                                "changed unrelated edits".into(),
                            )
                            .await;
                        cx.run_until_parked();
                    }
                    _ => {}
                }
                let (updates, _progress) = async_channel::bounded(32);
                let result = cx
                    .update(|cx| {
                        launch_github_evidence(
                            review,
                            &project,
                            directory.clone(),
                            1,
                            report,
                            DoctorProduct::Copper,
                            history.clone(),
                            NOW,
                            cancel,
                            cancelled,
                            updates,
                            cx,
                        )
                    })
                    .await;
                if scenario == "success" {
                    let result = result.expect("Observed read");
                    assert!(result.snapshot.is_some());
                    assert_eq!(calls.load(Ordering::SeqCst), 6);
                    assert_eq!(
                        result.review.phase(),
                        crate::doctor::DoctorRepairPhase::Incomplete,
                        "Read success is not a fixed/security conclusion"
                    );
                    let snapshot = result.snapshot.expect("Current selected metadata");
                    assert_eq!(snapshot.analyses.coverage, GithubSecurityCoverage::Complete);
                    assert!(
                        snapshot.analyses.observations.is_empty(),
                        "No existing analyses cannot prove coverage"
                    );
                    let reopened = DoctorHistoryStore::for_test_database(&database);
                    let records = reopened
                        .records(DoctorProduct::Copper, NOW)
                        .await
                        .expect("Reopen history");
                    assert_eq!(records.len(), 1);
                    let text = serde_json::to_string(&records).expect("Redacted durable history");
                    for sentinel in [
                        "FIXTURE_CREDENTIAL",
                        "RAW_CREDENTIAL",
                        "RAW_SOURCE",
                        "unrelated edits",
                    ] {
                        assert!(!text.contains(sentinel));
                    }
                } else {
                    assert!(result.is_err(), "{scenario}");
                    assert_eq!(
                        calls.load(Ordering::SeqCst),
                        0,
                        "{scenario}: no unapproved or stale request"
                    );
                    assert!(
                        history
                            .records(DoctorProduct::Copper, NOW)
                            .await
                            .expect("No receipt")
                            .is_empty()
                    );
                }
                assert_eq!(
                    filesystem
                        .load(Path::new("/github-review/source.txt"))
                        .await
                        .expect("Preserved source"),
                    if scenario == "source_changed" {
                        "changed unrelated edits"
                    } else {
                        "unrelated edits"
                    }
                );
            }
        }
        #[gpui::test]
        async fn doctor_github_review_pending_duplicate_cancellation_revocation_timeout_and_interruption(
            cx: &mut TestAppContext,
        ) {
            for scenario in [
                "cancel",
                "revoked",
                "settings",
                "timeout",
                "drop",
                "duplicate",
            ] {
                let mut application = cx.new_app();
                let cx = &mut application;
                let (project, directory, _) = fixture(cx).await;
                let credential =
                    Arc::new(Mutex::new(Some("FIXTURE_CREDENTIAL_NOT_HISTORY".into())));
                let calls = Arc::new(AtomicUsize::new(0));
                cx.update(|cx| install_transport(&project, calls.clone(), true, cx));
                let mut review =
                    cx.update(|cx| proposal(&project, &directory, credential.clone(), cx));
                let selected_report = report(&review);
                review
                    .approve(
                        &cx.update(|cx| project_repair_binding(&project, directory.clone(), 1, cx))
                            .expect("Scope"),
                    )
                    .expect("Approval");
                let database = db::AppDatabase::test_new();
                let history = DoctorHistoryStore::for_test_database(&database);
                let (cancel, cancelled) = async_channel::bounded(1);
                let (updates, _progress) = async_channel::bounded(32);
                let mut running = Some(cx.update(|cx| {
                    launch_github_evidence(
                        review,
                        &project,
                        directory.clone(),
                        1,
                        selected_report,
                        DoctorProduct::Copper,
                        history.clone(),
                        NOW,
                        cancel.clone(),
                        cancelled,
                        updates,
                        cx,
                    )
                }));
                cx.run_until_parked();
                assert_eq!(
                    calls.load(Ordering::SeqCst),
                    1,
                    "Only selected account verification is pending"
                );
                if scenario == "duplicate" {
                    let mut duplicate =
                        cx.update(|cx| proposal(&project, &directory, credential.clone(), cx));
                    let report = report(&duplicate);
                    duplicate
                        .approve(
                            &cx.update(|cx| {
                                project_repair_binding(&project, directory.clone(), 1, cx)
                            })
                            .expect("Scope"),
                        )
                        .expect("Separate approval");
                    let (other_cancel, other_cancelled) = async_channel::bounded(1);
                    let (updates, _progress) = async_channel::bounded(32);
                    let result = cx
                        .update(|cx| {
                            launch_github_evidence(
                                duplicate,
                                &project,
                                directory.clone(),
                                1,
                                report,
                                DoctorProduct::Copper,
                                history.clone(),
                                NOW,
                                other_cancel,
                                other_cancelled,
                                updates,
                                cx,
                            )
                        })
                        .await;
                    assert!(result.is_err());
                    assert_eq!(calls.load(Ordering::SeqCst), 1);
                    cancel.close();
                }
                match scenario {
                    "cancel" => {
                        cancel.close();
                    }
                    "revoked" => {
                        *credential.lock().expect("Credential fixture") = None;
                    }
                    "settings" => cx.update(|cx| {
                        cx.update_global::<settings::SettingsStore, _>(|store, cx| {
                            store
                                .set_user_settings("{\"vim_mode\":true}", cx)
                                .expect("Changed settings");
                        })
                    }),
                    "timeout" => cx.executor().advance_clock(Duration::from_secs(61)),
                    "drop" => drop(running.take()),
                    _ => {}
                }
                cx.executor().advance_clock(Duration::from_millis(100));
                cx.run_until_parked();
                if let Some(running) = running {
                    let result = running.await.expect("Observed terminal read");
                    assert!(result.snapshot.is_none());
                    assert_ne!(
                        result.review.phase(),
                        crate::doctor::DoctorRepairPhase::Verified
                    );
                }
                assert_eq!(
                    calls.load(Ordering::SeqCst),
                    1,
                    "No replay or extra source read"
                );
                let reopened = DoctorHistoryStore::for_test_database(&database);
                let records = reopened
                    .records(DoctorProduct::Copper, NOW + 62)
                    .await
                    .expect("Reopen durable receipt");
                assert_eq!(records.len(), 1);
                assert_ne!(
                    records.first().expect("Receipt").outcome,
                    crate::doctor::ReceiptOutcome::Running
                );
                let text = serde_json::to_string(&records).expect("Redaction");
                assert!(!text.contains("FIXTURE_CREDENTIAL"));
            }
        }
    }
}
