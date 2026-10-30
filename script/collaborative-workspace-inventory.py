#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


CAPABILITY_PATTERN = re.compile(r"CAP-\d{3}")
REQUIREMENT_PATTERN = re.compile(r"\d+\.\d+")
LEAF_HEADER_PATTERN = re.compile(r"^  - \[[ x~]\] (\d+\.\d+)\.", re.MULTILINE)
CHECKBOX_PATTERN = re.compile(r"^\s*- \[[ x~]\] \d+(?:\.\d+)?\.", re.MULTILINE)
CONSTANT_PATTERN = re.compile(
    r"^pub const ([A-Z][A-Z0-9_]*): u32 = (\d+);$", re.MULTILINE
)
RANGE_CONSTANTS = {
    "PARAM_REPLACEABLE_KIND_MIN",
    "PARAM_REPLACEABLE_KIND_MAX",
    "EPHEMERAL_KIND_MIN",
    "EPHEMERAL_KIND_MAX",
}
CATALOG_HEADERS = {
    "packages": [
        "package_id",
        "member_path",
        "manifest_path",
        "package_name",
        "package_version",
        "targets",
        "feature_flags",
        "capability_ids",
        "final_disposition",
    ],
    "protocol": [
        "catalog_id",
        "record_type",
        "name",
        "numeric_value",
        "protocol_references",
        "source_paths",
        "capability_ids",
        "status",
        "summary",
    ],
    "data": [
        "data_source_id",
        "record_type",
        "name",
        "backend",
        "scope",
        "durability",
        "authority",
        "source_paths",
        "capability_ids",
        "migration_requirement",
        "status",
        "summary",
    ],
    "surfaces": [
        "surface_id",
        "surface_type",
        "name",
        "source_paths",
        "dependencies",
        "capability_ids",
        "status",
        "final_disposition",
        "summary",
    ],
}
CATALOG_FILES = {
    "packages": "buzz-packages.csv",
    "protocol": "protocol.csv",
    "data": "data-sources.csv",
    "surfaces": "surfaces.csv",
}


@dataclass(frozen=True)
class CapabilityOwner:
    zed_owner: str
    disposition: str


@dataclass(frozen=True)
class LeafTask:
    state: str
    requirements: frozenset[str]
    capabilities: frozenset[str]


@dataclass(frozen=True)
class Catalog:
    name: str
    path: Path
    identifier_field: str
    rows: tuple[dict[str, str], ...]


class InventoryChecker:
    def __init__(self, repository_root: Path, selected_catalog: str) -> None:
        self.repository_root = repository_root
        self.spec_root = repository_root / "docs/collaboration/collaborative-workspace"
        self.buzz_root = repository_root / "projects/buzz"
        self.selected_catalog = selected_catalog
        self.errors: list[str] = []
        self.catalogs: dict[str, Catalog] = {}
        self.owners: dict[str, CapabilityOwner] = {}
        self.requirements: set[str] = set()
        self.leaves: dict[str, LeafTask] = {}
        self.source_inventory_capabilities: set[str] = set()

    def run(self, fixture_source: str | None) -> bool:
        self._load_spec_references()
        self._load_catalogs()
        self._check_reference_graph()
        self._check_adr_task_references()
        self._check_final_coverage()
        self._check_remediation_checklist()

        selected = (
            tuple(CATALOG_FILES)
            if self.selected_catalog == "all"
            else (self.selected_catalog,)
        )
        for catalog_name in selected:
            {
                "packages": self._check_packages,
                "protocol": self._check_protocol,
                "data": self._check_data,
                "surfaces": self._check_surfaces,
            }[catalog_name]()

        if fixture_source is not None:
            self._check_unmapped_fixture(fixture_source)

        return not self.errors

    def _check_remediation_checklist(self) -> None:
        path = self.spec_root / "remediation-checklist.json"
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            self._error(f"{self._relative(path)}: unreadable remediation checklist: {error}")
            return
        rows = document.get("tasks") if isinstance(document, dict) else None
        if not isinstance(rows, list):
            self._error(f"{self._relative(path)}: tasks must be a list")
            return
        criterion_text = dict(re.findall(
            r"^\d+\. \*\*(\d+\.\d+)\*\* (.+)$",
            (self.spec_root / "requirements.md").read_text(encoding="utf-8"),
            re.MULTILINE,
        ))
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                self._error(f"{self._relative(path)}: task must be an object")
                continue
            task = row.get("task_id")
            location = f"{self._relative(path)}: task {task}"
            if not isinstance(task, str) or task not in self.leaves or task in seen:
                self._error(f"{location}: unknown or duplicate task")
                continue
            seen.add(task)
            leaf = self.leaves[task]
            criteria = row.get("acceptance_criteria")
            if not isinstance(criteria, dict) or set(criteria) != leaf.requirements or any(
                value != criterion_text.get(identifier) for identifier, value in criteria.items()
            ):
                self._error(f"{location}: acceptance criteria differ from the canonical task")
            if row.get("spec_state") != leaf.state:
                self._error(f"{location}: stale specification state")
            if not row.get("affected_files") or not row.get("required_validation"):
                self._error(f"{location}: missing affected files or required validation")
            affected_files = row.get("affected_files")
            if isinstance(affected_files, str):
                valid_affected_files = len(affected_files.strip()) > 1
            elif isinstance(affected_files, list):
                valid_affected_files = bool(affected_files) and all(
                    isinstance(path, str) and len(path.strip()) > 1 for path in affected_files
                )
            else:
                valid_affected_files = False
            if not valid_affected_files:
                self._error(f"{location}: invalid affected files manifest or path list")
            findings = row.get("findings")
            if not isinstance(findings, list) or not findings:
                self._error(f"{location}: missing finding reconciliation")
                continue
            identifiers = set()
            for finding in findings:
                if not isinstance(finding, dict):
                    self._error(f"{location}: finding must be an object")
                    continue
                identifier = finding.get("id")
                if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                    self._error(f"{location}: missing or duplicate finding ID")
                    continue
                identifiers.add(identifier)
                if not finding.get("historical_finding") or not finding.get("source") or not finding.get("current_status"):
                    self._error(f"{location}: finding {identifier} lacks source, description or status")
                if finding.get("current_status", "").startswith("resolved") and not all(
                    finding.get(field) for field in ("implementation", "validation", "evidence")
                ):
                    self._error(f"{location}: finding {identifier} claims resolution without implementation, validation and evidence")
                if finding.get("current_status") == "externally_blocked" and not finding.get("external_blocker"):
                    self._error(f"{location}: finding {identifier} lacks its specific external dependency")
        for task in self.leaves.keys() - seen:
            self._error(f"{self._relative(path)}: missing canonical task {task}")

    def _check_adr_task_references(self) -> None:
        for path in sorted((self.spec_root / "decisions").glob("adr-*.md")):
            in_trace = False
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line == "## Implementation and validation trace":
                    in_trace = True
                    continue
                if not in_trace or not re.search(r"\b[Tt]asks?\b", line):
                    continue
                location = f"{self._relative(path)}:{line_number}"
                for match in re.finditer(r"(?<!\d)(\d+\.\d+)(?:[–-](\d+\.\d+))?", line):
                    first, last = match.groups()
                    if last is None:
                        references = [first]
                    else:
                        first_epic, first_leaf = map(int, first.split("."))
                        last_epic, last_leaf = map(int, last.split("."))
                        if first_epic != last_epic or first_leaf > last_leaf:
                            self._error(f"{location}: invalid ADR task range {match.group()}")
                            continue
                        if first not in self.leaves or last not in self.leaves:
                            self._error(f"{location}: unknown ADR task range endpoint {match.group()}")
                            continue
                        references = [f"{first_epic}.{leaf}" for leaf in range(first_leaf, last_leaf + 1)]
                    for reference in references:
                        if reference not in self.leaves:
                            self._error(f"{location}: unknown ADR leaf-task reference {reference}")
            if not in_trace:
                self._error(f"{self._relative(path)}: missing ADR implementation and validation trace")

    def _load_spec_references(self) -> None:
        reuse_audit_path = self.spec_root / "reuse-audit.md"
        for line_number, line in enumerate(
            reuse_audit_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not re.match(r"^\|\s*CAP-\d{3}\s*\|", line):
                continue
            columns = [column.strip() for column in line.strip().strip("|").split("|")]
            if len(columns) != 7:
                self._error(
                    f"{self._relative(reuse_audit_path)}:{line_number}: "
                    "capability ownership row must contain seven columns"
                )
                continue
            capability_id = columns[0]
            if capability_id in self.owners:
                self._error(
                    f"{self._relative(reuse_audit_path)}:{line_number}: "
                    f"duplicate ownership row {capability_id}"
                )
                continue
            self.owners[capability_id] = CapabilityOwner(
                zed_owner=columns[3], disposition=columns[4]
            )

        requirements_path = self.spec_root / "requirements.md"
        self.requirements = set(
            re.findall(
                r"^\d+\. \*\*(\d+\.\d+)\*\*",
                requirements_path.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
        )

        tasks_path = self.spec_root / "tasks.md"
        task_text = tasks_path.read_text(encoding="utf-8")
        matches = list(LEAF_HEADER_PATTERN.finditer(task_text))
        for match in matches:
            leaf_id = match.group(1)
            block_end = len(task_text)
            next_checkbox = CHECKBOX_PATTERN.search(task_text, match.end())
            if next_checkbox is not None:
                block_end = next_checkbox.start()
            block = task_text[match.start() : block_end]
            requirement_line = re.search(r"_Requirements: ([^\n]+)_", block)
            capability_line = re.search(r"_Capability IDs: ([^\n]+)_", block)
            leaf_requirements = frozenset(
                REQUIREMENT_PATTERN.findall(
                    requirement_line.group(1) if requirement_line else ""
                )
            )
            leaf_capabilities = frozenset(
                CAPABILITY_PATTERN.findall(
                    capability_line.group(1) if capability_line else ""
                )
            )
            if leaf_id in self.leaves:
                self._error(
                    f"{self._relative(tasks_path)}: duplicate leaf task {leaf_id}"
                )
            self.leaves[leaf_id] = LeafTask(
                state=re.search(r"\[([ x~])\]", block).group(1), requirements=leaf_requirements, capabilities=leaf_capabilities
            )

        source_inventory_path = self.spec_root / "source-inventory.md"
        self.source_inventory_capabilities = set(
            re.findall(
                r"^\|\s*(CAP-\d{3})\s*\|",
                source_inventory_path.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
        )

    def _load_catalogs(self) -> None:
        catalogs_root = self.spec_root / "catalogs"
        for name, filename in CATALOG_FILES.items():
            path = catalogs_root / filename
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                actual_headers = reader.fieldnames or []
                if actual_headers != CATALOG_HEADERS[name]:
                    self._error(
                        f"{self._relative(path)}: expected headers "
                        f"{CATALOG_HEADERS[name]!r}, found {actual_headers!r}"
                    )
                rows = tuple(dict(row) for row in reader)
            identifier_field = CATALOG_HEADERS[name][0]
            self.catalogs[name] = Catalog(
                name=name,
                path=path,
                identifier_field=identifier_field,
                rows=rows,
            )

    def _check_reference_graph(self) -> None:
        expected_capabilities = {f"CAP-{number:03d}" for number in range(1, 46)}
        self._compare_sets(
            "reuse-audit capability ownership rows",
            expected_capabilities,
            set(self.owners),
        )
        self._compare_sets(
            "source-inventory capability rows",
            expected_capabilities,
            self.source_inventory_capabilities,
        )

        capability_leaves: dict[str, set[str]] = defaultdict(set)
        capability_requirements: dict[str, set[str]] = defaultdict(set)
        requirement_leaves: dict[str, set[str]] = defaultdict(set)
        for leaf_id, leaf in self.leaves.items():
            if not leaf.requirements:
                self._error(f"tasks.md leaf {leaf_id}: missing requirement references")
            if not leaf.capabilities:
                self._error(f"tasks.md leaf {leaf_id}: missing capability references")
            for requirement_id in leaf.requirements:
                if requirement_id not in self.requirements:
                    self._error(
                        f"tasks.md leaf {leaf_id}: unknown requirement {requirement_id}"
                    )
                requirement_leaves[requirement_id].add(leaf_id)
            for capability_id in leaf.capabilities:
                if capability_id not in expected_capabilities:
                    self._error(
                        f"tasks.md leaf {leaf_id}: unknown capability {capability_id}"
                    )
                capability_leaves[capability_id].add(leaf_id)
                capability_requirements[capability_id].update(leaf.requirements)

        for requirement_id in sorted(self.requirements):
            if not requirement_leaves[requirement_id]:
                self._error(
                    f"requirements.md acceptance criterion {requirement_id}: "
                    "missing leaf-task reference"
                )

        catalog_capabilities: dict[str, set[str]] = defaultdict(set)
        global_identifiers: dict[str, str] = {}
        for catalog in self.catalogs.values():
            for row_number, row in enumerate(catalog.rows, start=2):
                identifier = row.get(catalog.identifier_field, "").strip()
                location = f"{self._relative(catalog.path)}:{row_number}"
                if not identifier:
                    self._error(f"{location}: missing stable identifier")
                    identifier = f"row-{row_number}"
                previous = global_identifiers.get(identifier)
                if previous is not None:
                    self._error(
                        f"{location}: duplicate stable identifier {identifier}; "
                        f"first used by {previous}"
                    )
                else:
                    global_identifiers[identifier] = location

                capabilities = self._parse_capability_field(
                    row.get("capability_ids", ""), f"{location} ({identifier})"
                )
                if not capabilities:
                    self._error(
                        f"{location} ({identifier}): missing capability ID reference"
                    )
                for capability_id in capabilities:
                    catalog_capabilities[capability_id].add(identifier)
                    if capability_id not in expected_capabilities:
                        self._error(
                            f"{location} ({identifier}): unknown capability "
                            f"{capability_id}"
                        )
                        continue
                    owner = self.owners.get(capability_id)
                    if owner is None or not owner.zed_owner or not owner.disposition:
                        self._error(
                            f"{location} ({identifier}): {capability_id} is missing "
                            "canonical owner/disposition reference"
                        )
                    if not capability_requirements[capability_id]:
                        self._error(
                            f"{location} ({identifier}): {capability_id} is missing "
                            "requirement references"
                        )
                    if not capability_leaves[capability_id]:
                        self._error(
                            f"{location} ({identifier}): {capability_id} is missing "
                            "leaf-task references"
                        )

                self._check_source_paths(catalog, row, row_number, identifier)

        for capability_id in sorted(expected_capabilities):
            owner = self.owners.get(capability_id)
            if owner is not None:
                if not owner.zed_owner:
                    self._error(
                        f"reuse-audit.md {capability_id}: missing existing Zed owner/gap"
                    )
                if not owner.disposition:
                    self._error(
                        f"reuse-audit.md {capability_id}: missing canonical "
                        "owner/disposition"
                    )
            if not catalog_capabilities[capability_id]:
                self._error(f"{capability_id}: missing catalog reference")
            if not capability_requirements[capability_id]:
                self._error(f"{capability_id}: missing requirement reference")
            if not capability_leaves[capability_id]:
                self._error(f"{capability_id}: missing leaf-task reference")

    def _check_packages(self) -> None:
        catalog = self.catalogs["packages"]
        workspace_manifest = self.buzz_root / "Cargo.toml"
        workspace = self._read_toml(workspace_manifest)
        workspace_members = set(workspace["workspace"]["members"])
        rows_by_member = {row["member_path"]: row for row in catalog.rows}
        self._compare_sets(
            "Buzz Cargo workspace members", workspace_members, set(rows_by_member)
        )

        workspace_version = str(workspace["workspace"]["package"]["version"])
        for member_path in sorted(workspace_members & set(rows_by_member)):
            row = rows_by_member[member_path]
            identifier = row["package_id"]
            expected_manifest = f"projects/buzz/{member_path}/Cargo.toml"
            if row["manifest_path"] != expected_manifest:
                self._error(
                    f"{identifier}: manifest path is {row['manifest_path']!r}; "
                    f"expected {expected_manifest!r}"
                )
                continue
            manifest_path = self.repository_root / expected_manifest
            manifest = self._read_toml(manifest_path)
            package = manifest["package"]
            if row["package_name"] != package["name"]:
                self._error(
                    f"{identifier}: package name is {row['package_name']!r}; "
                    f"expected {package['name']!r}"
                )
            raw_version = package["version"]
            expected_version = (
                f"workspace:{workspace_version}"
                if isinstance(raw_version, dict) and raw_version.get("workspace") is True
                else str(raw_version)
            )
            if row["package_version"] != expected_version:
                self._error(
                    f"{identifier}: package version is {row['package_version']!r}; "
                    f"expected {expected_version!r}"
                )

            expected_features = set(manifest.get("features", {}))
            actual_features = self._split_values(row["feature_flags"])
            self._compare_sets(
                f"{identifier} feature flags", expected_features, actual_features
            )

            member_root = manifest_path.parent
            expected_targets = self._discover_cargo_targets(member_root, manifest)
            actual_targets = self._split_values(row["targets"])
            self._compare_sets(
                f"{identifier} library/binary targets", expected_targets, actual_targets
            )

            if not row["final_disposition"].strip():
                self._error(f"{identifier}: missing final disposition")

    def _check_protocol(self) -> None:
        catalog = self.catalogs["protocol"]
        kind_path = self.buzz_root / "crates/buzz-core/src/kind.rs"
        source_constants = {
            name: value
            for name, value in CONSTANT_PATTERN.findall(
                kind_path.read_text(encoding="utf-8")
            )
        }
        catalog_constants: dict[str, str] = {}
        for row in catalog.rows:
            if row["record_type"] not in {"event_kind", "kind_range"}:
                continue
            name = row["name"]
            expected_type = "kind_range" if name in RANGE_CONSTANTS else "event_kind"
            if row["record_type"] != expected_type:
                self._error(
                    f"{row['catalog_id']}: {name} must use record type {expected_type}"
                )
            if name in catalog_constants:
                self._error(f"protocol catalog: duplicate scalar constant {name}")
            catalog_constants[name] = row["numeric_value"]
        self._compare_sets(
            "Buzz scalar u32 kind constants",
            set(source_constants),
            set(catalog_constants),
        )
        for name in sorted(set(source_constants) & set(catalog_constants)):
            if source_constants[name] != catalog_constants[name]:
                self._error(
                    f"protocol constant {name}: catalog value "
                    f"{catalog_constants[name]!r}; source value {source_constants[name]!r}"
                )

        custom_documents = {
            self._relative(path)
            for path in (self.buzz_root / "docs/nips").glob("NIP-*.md")
        }
        catalog_documents = self._paths_for_record_type(
            catalog, "custom_nip_document"
        )
        self._compare_sets(
            "Buzz custom NIP documents", custom_documents, catalog_documents
        )

        protocol_fixtures = {
            self._relative(path)
            for path in (self.buzz_root / "docs/nips").glob("NIP-*.json")
        }
        catalog_fixtures = self._paths_for_record_type(catalog, "protocol_fixture")
        self._compare_sets(
            "Buzz protocol fixtures", protocol_fixtures, catalog_fixtures
        )

        expected_guides = {"projects/buzz/NOSTR.md"}
        catalog_guides = self._paths_for_record_type(catalog, "protocol_guide")
        self._compare_sets("Buzz protocol guides", expected_guides, catalog_guides)

        referenced_standard_nips: set[str] = set()
        for row in catalog.rows:
            if row["record_type"] in {"event_kind", "kind_range", "protocol_guide"}:
                referenced_standard_nips.update(
                    re.findall(r"NIP-\d{2}", row["protocol_references"])
                )
        catalog_standard_nips = {
            row["name"]
            for row in catalog.rows
            if row["record_type"] == "standard_nip"
        }
        self._compare_sets(
            "referenced standard NIPs",
            referenced_standard_nips,
            catalog_standard_nips,
        )

    def _check_data(self) -> None:
        catalog = self.catalogs["data"]
        source_migrations = {
            self._relative(path)
            for path in (self.buzz_root / "migrations").glob("*.sql")
        }
        catalog_migrations = self._paths_for_record_type(catalog, "sql_migration")
        self._compare_sets(
            "Buzz SQL migrations", source_migrations, catalog_migrations
        )

        source_schemas = {
            self._relative(path)
            for path in (self.buzz_root / "schema").rglob("*")
            if path.is_file()
        }
        catalog_schemas = self._paths_for_record_type(catalog, "schema_snapshot")
        self._compare_sets("Buzz schema snapshots", source_schemas, catalog_schemas)

        required_types = {
            "sql_migration",
            "schema_snapshot",
            "canonical_store",
            "redis_state",
            "derived_cache",
            "object_store",
            "desktop_store",
            "secret_store",
            "operational_store",
            "migration_bridge",
            "declared_gap",
        }
        actual_types = {row["record_type"] for row in catalog.rows}
        missing_types = required_types - actual_types
        for record_type in sorted(missing_types):
            self._error(f"data catalog: missing required record type {record_type}")

        declared_gaps = {
            row["data_source_id"]
            for row in catalog.rows
            if row["record_type"] == "declared_gap"
        }
        if "REDIS-TYPING-GAP-001" not in declared_gaps:
            self._error(
                "data catalog: missing REDIS-TYPING-GAP-001 known-gap record"
            )

    def _check_surfaces(self) -> None:
        catalog = self.catalogs["surfaces"]

        tauri_source = self.buzz_root / "desktop/src-tauri/src/lib.rs"
        tauri_modules = set(
            re.findall(
                r"^\s*(?:pub\s+)?mod\s+([A-Za-z0-9_]+)\s*;",
                tauri_source.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
        )
        catalog_tauri_modules = {
            row["name"]
            for row in catalog.rows
            if row["surface_type"] == "tauri_module"
        }
        self._compare_sets(
            "declared Tauri modules", tauri_modules, catalog_tauri_modules
        )

        desktop_features = {
            self._relative(path)
            for path in (self.buzz_root / "desktop/src/features").iterdir()
            if path.is_dir()
        }
        self._compare_sets(
            "desktop feature directories",
            desktop_features,
            self._paths_for_surface_type(catalog, "desktop_feature"),
        )

        desktop_routes = {
            self._relative(path)
            for path in (self.buzz_root / "desktop/src/app/routes").glob("*.tsx")
            if re.search(
                r"create(?:File|Root)Route",
                path.read_text(encoding="utf-8"),
            )
        }
        self._compare_sets(
            "desktop route files",
            desktop_routes,
            self._paths_for_surface_type(catalog, "desktop_route"),
        )

        web_routes = {
            self._relative(path)
            for path in (self.buzz_root / "web/src/app/routes").glob("*.tsx")
            if re.search(
                r"create(?:File|Root)Route",
                path.read_text(encoding="utf-8"),
            )
        }
        self._compare_sets(
            "web route files",
            web_routes,
            self._paths_for_surface_type(catalog, "web_route"),
        )

        mobile_components = {
            "projects/buzz/mobile/lib/app.dart",
            "projects/buzz/mobile/lib/shared/deeplink",
        }
        mobile_components.update(
            self._relative(path)
            for path in (self.buzz_root / "mobile/lib/features").iterdir()
            if path.is_dir()
        )
        mobile_rows = [
            row for row in catalog.rows if row["surface_type"] == "mobile_surface"
        ]
        self._check_logical_surface_coverage(
            "mobile surfaces", mobile_components, mobile_rows
        )

        admin_source = (self.buzz_root / "admin-web/src/App.tsx").read_text(
            encoding="utf-8"
        )
        admin_routes = set(re.findall(r'href="(/[a-z-]+)"', admin_source))
        admin_routes.update(
            f"/{name}/:id"
            for name in re.findall(
                r"path\.match\(/\^\\/([a-z-]+)\\/\(\[\^/\]\+\)\$/\)",
                admin_source,
            )
        )
        catalog_admin_routes = {
            row["name"]
            for row in catalog.rows
            if row["surface_type"] == "admin_route"
        }
        self._compare_sets("admin routes", admin_routes, catalog_admin_routes)

        deployment_components = {
            self._relative(path)
            for path in (self.buzz_root / "deploy/charts").iterdir()
            if path.is_dir()
        }
        deployment_components.update(
            self._relative(path)
            for path in (self.buzz_root / "deploy").iterdir()
            if path.is_dir() and path.name != "charts"
        )
        self._compare_sets(
            "deployment components",
            deployment_components,
            self._paths_for_surface_type(catalog, "deployment_component"),
        )

        workflows = {
            self._relative(path)
            for pattern in ("*.yml", "*.yaml")
            for path in (self.buzz_root / ".github/workflows").glob(pattern)
        }
        self._compare_sets(
            "GitHub workflows",
            workflows,
            self._paths_for_surface_type(catalog, "ci_workflow"),
        )

        scripts = {
            self._relative(path)
            for path in (self.buzz_root / "scripts").rglob("*")
            if path.is_file()
        }
        self._compare_sets(
            "operational scripts",
            scripts,
            self._paths_for_surface_type(catalog, "operational_script"),
        )

        examples = {
            self._relative(path)
            for path in (self.buzz_root / "examples").iterdir()
            if path.is_dir()
        }
        self._compare_sets(
            "Buzz examples", examples, self._paths_for_surface_type(catalog, "example")
        )

        benchmark_components = {
            self._relative(path)
            for path in (self.buzz_root / "benchmarks").iterdir()
            if path.is_dir()
        }
        if (self.buzz_root / "perf").is_dir():
            benchmark_components.add("projects/buzz/perf")
        benchmark_rows = [
            row for row in catalog.rows if row["surface_type"] == "benchmark"
        ]
        self._check_logical_surface_coverage(
            "benchmark suites", benchmark_components, benchmark_rows
        )

        test_surfaces = {
            "projects/buzz/desktop/tests/e2e",
            "projects/buzz/desktop/src-tauri/tests",
            "projects/buzz/mobile/test",
            "projects/buzz/web/tests/e2e",
            "projects/buzz/admin-web/tests",
        }
        existing_test_surfaces = {
            path
            for path in test_surfaces
            if (self.repository_root / path).exists()
        }
        self._compare_sets(
            "client test surfaces",
            existing_test_surfaces,
            self._paths_for_surface_type(catalog, "test_surface"),
        )

    def _check_final_coverage(self) -> None:
        path = self.spec_root / "catalogs/final-coverage.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        seen: set[str] = set()
        for row in rows:
            capability = row.get("capability_id", "")
            location = f"{self._relative(path)} ({capability})"
            if capability in seen:
                self._error(f"{location}: duplicate capability")
            seen.add(capability)
            leaves = {
                leaf_id: leaf for leaf_id, leaf in self.leaves.items()
                if capability in leaf.capabilities
            }
            self._compare_sets(
                f"{location} implementation tasks",
                set(leaves), self._split_values(row.get("implementation_tasks", "")),
            )
            self._compare_sets(
                f"{location} requirements",
                {requirement for leaf in leaves.values() for requirement in leaf.requirements},
                self._split_values(row.get("requirement_ids", "")),
            )
            expected_sources = {
                f"{name}:{record[catalog.identifier_field]}"
                for name, catalog in self.catalogs.items() for record in catalog.rows
                if capability in self._split_values(record.get("capability_ids", ""))
            }
            self._compare_sets(
                f"{location} source rows", expected_sources,
                self._split_values(row.get("source_rows", "")),
            )
            if row.get("source_row_count") != str(len(expected_sources)):
                self._error(f"{location}: source row count is stale")
            status = row.get("component_status", "")
            if status not in {"complete", "partial", "missing", "blocked", "unverified"}:
                self._error(f"{location}: invalid component status {status!r}")
            if status == "complete":
                unfinished = sorted(leaf_id for leaf_id, leaf in leaves.items() if leaf.state != "x")
                if unfinished:
                    self._error(f"{location}: complete claim has unfinished tasks {unfinished}")
            if not row.get("validation_evidence", "").strip():
                self._error(f"{location}: missing validation evidence or explicit evidence gap")
        self._compare_sets("final coverage capabilities", set(self.owners), seen)

    def _check_unmapped_fixture(self, fixture_source: str) -> None:
        fixture_path = Path(fixture_source)
        if not fixture_path.exists():
            self._error(f"fixture {fixture_source}: source path does not exist")
        self._error(f"fixture {fixture_source}: missing capability ID references")
        self._error(
            f"fixture {fixture_source}: missing canonical owner/disposition references"
        )
        self._error(f"fixture {fixture_source}: missing requirement references")
        self._error(f"fixture {fixture_source}: missing leaf-task references")

    def _check_source_paths(
        self,
        catalog: Catalog,
        row: dict[str, str],
        row_number: int,
        identifier: str,
    ) -> None:
        field_name = "manifest_path" if catalog.name == "packages" else "source_paths"
        source_paths = self._split_values(row.get(field_name, ""))
        location = f"{self._relative(catalog.path)}:{row_number} ({identifier})"
        if not source_paths:
            self._error(f"{location}: missing source path")
            return
        for source_path in sorted(source_paths):
            path = Path(source_path)
            if path.is_absolute() or ".." in path.parts:
                self._error(
                    f"{location}: source path must be repository-relative: {source_path}"
                )
                continue
            if not (self.repository_root / path).exists():
                self._error(f"{location}: source path does not exist: {source_path}")

    def _parse_capability_field(self, value: str, location: str) -> set[str]:
        values = self._split_values(value)
        parsed = set(CAPABILITY_PATTERN.findall(value))
        if values != parsed:
            invalid = sorted(values - parsed)
            if invalid:
                self._error(
                    f"{location}: invalid capability reference(s): {', '.join(invalid)}"
                )
        return parsed

    def _discover_cargo_targets(
        self, member_root: Path, manifest: dict[str, object]
    ) -> set[str]:
        package = manifest["package"]
        if not isinstance(package, dict):
            return set()
        package_name = str(package["name"])
        targets: set[str] = set()
        explicit_paths: set[Path] = set()

        raw_library = manifest.get("lib")
        library_path = member_root / "src/lib.rs"
        if isinstance(raw_library, dict):
            library_name = str(raw_library.get("name", package_name.replace("-", "_")))
            targets.add(f"lib:{library_name}")
            explicit_paths.add(member_root / str(raw_library.get("path", "src/lib.rs")))
        elif library_path.exists():
            targets.add(f"lib:{package_name.replace('-', '_')}")

        raw_bins = manifest.get("bin", [])
        if isinstance(raw_bins, list):
            for raw_bin in raw_bins:
                if not isinstance(raw_bin, dict) or "name" not in raw_bin:
                    continue
                targets.add(f"bin:{raw_bin['name']}")
                default_path = f"src/bin/{raw_bin['name']}.rs"
                explicit_paths.add(member_root / str(raw_bin.get("path", default_path)))

        if package.get("autobins", True):
            main_path = member_root / "src/main.rs"
            if main_path.exists() and main_path not in explicit_paths:
                targets.add(f"bin:{package_name}")
            bins_root = member_root / "src/bin"
            if bins_root.is_dir():
                for bin_path in bins_root.glob("*.rs"):
                    if bin_path not in explicit_paths:
                        targets.add(f"bin:{bin_path.stem}")
                for bin_directory in bins_root.iterdir():
                    main = bin_directory / "main.rs"
                    if bin_directory.is_dir() and main.exists() and main not in explicit_paths:
                        targets.add(f"bin:{bin_directory.name}")
        return targets

    def _check_logical_surface_coverage(
        self,
        label: str,
        expected_components: set[str],
        rows: list[dict[str, str]],
    ) -> None:
        if len(rows) != len(expected_components):
            self._error(
                f"{label}: expected {len(expected_components)} catalog rows, "
                f"found {len(rows)}"
            )
        covered: set[str] = set()
        for row in rows:
            row_paths = self._split_values(row["source_paths"])
            matches = {
                component
                for component in expected_components
                if any(
                    source_path == component
                    or source_path.startswith(f"{component}/")
                    for source_path in row_paths
                )
            }
            if not matches:
                self._error(
                    f"{row['surface_id']}: does not map a discovered {label} component"
                )
            covered.update(matches)
        self._compare_sets(label, expected_components, covered)

    def _paths_for_record_type(self, catalog: Catalog, record_type: str) -> set[str]:
        paths: set[str] = set()
        for row in catalog.rows:
            if row["record_type"] == record_type:
                paths.update(self._split_values(row["source_paths"]))
        return paths

    def _paths_for_surface_type(self, catalog: Catalog, surface_type: str) -> set[str]:
        paths: set[str] = set()
        for row in catalog.rows:
            if row["surface_type"] == surface_type:
                paths.update(self._split_values(row["source_paths"]))
        return paths

    def _compare_sets(
        self, label: str, expected: set[str], actual: set[str]
    ) -> None:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing:
            self._error(f"{label}: missing {', '.join(missing)}")
        if unexpected:
            self._error(f"{label}: unexpected {', '.join(unexpected)}")

    def _read_toml(self, path: Path) -> dict[str, object]:
        with path.open("rb") as handle:
            return tomllib.load(handle)

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.repository_root).as_posix()

    @staticmethod
    def _split_values(value: str) -> set[str]:
        return {item.strip() for item in value.split(";") if item.strip()}

    def _error(self, message: str) -> None:
        self.errors.append(message)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Collaborative Workspace Buzz inventory coverage and drift."
    )
    parser.add_argument(
        "--catalog",
        choices=("all", "packages", "protocol", "data", "surfaces"),
        default="all",
        help="run source drift checks for one catalog or all catalogs",
    )
    parser.add_argument(
        "--fixture-source",
        metavar="PATH",
        help="inject an intentionally unmapped source fixture for negative validation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checker = InventoryChecker(repository_root(), args.catalog)
    passed = checker.run(args.fixture_source)
    if not passed:
        print(f"Inventory check failed ({len(checker.errors)} error(s)):", file=sys.stderr)
        for error in sorted(set(checker.errors)):
            print(f"- {error}", file=sys.stderr)
        return 1

    counts = " ".join(
        f"{name}={len(checker.catalogs[name].rows)}"
        for name in ("packages", "protocol", "data", "surfaces")
    )
    print(
        f"Inventory check passed: {counts} capabilities={len(checker.owners)} "
        f"requirements={len(checker.requirements)} leaves={len(checker.leaves)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
