#!/usr/bin/env python3

import argparse
import posixpath
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


REQUIREMENT_ID = re.compile(r"^\s*\d+\.\s+\*\*(\d+\.\d+)\*\*", re.MULTILINE)
REFERENCE_ID = re.compile(r"\b\d+\.\d+\b")
TASK_ID_REFERENCE = re.compile(r"\b\d+(?:\.\d+)?\b")
TASK_HEADER = re.compile(
    r"^(?P<indent>[ \t]*)-\s+\[(?P<state>[ xX~-])\]\s+(?P<id>\d+(?:\.\d+)*)\.\s+",
    re.MULTILINE,
)
HEADING = re.compile(r"^#{2,6}\s+(?P<title>.+?)\s*$", re.MULTILINE)
METADATA_ENTRY = re.compile(
    r"^(?P<indent>[ \t]*)-\s+_"
    r"(?P<label>Requirements|Depends on|Reads|Writes|Validation|Evidence):\s*"
    r"(?P<value>.*?)_\s*$",
    re.IGNORECASE | re.MULTILINE,
)
FEATURE_NAME_SEGMENT = r"(?:[a-z0-9]+|v[0-9]+(?:\.[0-9]+)+)"
KEBAB_CASE = re.compile(rf"^{FEATURE_NAME_SEGMENT}(?:-{FEATURE_NAME_SEGMENT})*$")
REQUIRED_METADATA = ("Requirements", "Depends on", "Reads", "Writes", "Validation")
CANONICAL_METADATA = REQUIRED_METADATA + ("Evidence",)
LEGACY_DIALECT_MARKER = re.compile(r"^\s*-\s+_id:\s*\S", re.MULTILINE)
CONTAINER_ROOTS = {
    "apps",
    "components",
    "crates",
    "libs",
    "modules",
    "packages",
    "plugins",
    "services",
}


def referenced_ids(value):
    return set(REFERENCE_ID.findall(value))


def design_references(design):
    references = set()
    traceability = re.search(
        r"^## Requirements traceability\s*$\n(?P<body>.*?)(?=^##\s|\Z)",
        design,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if traceability:
        for line in traceability.group("body").splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 3:
                references.update(referenced_ids(cells[0]))

    for match in re.finditer(
        r"(?:Validates|Addresses):\s*(?:Requirements?\s*)?([^\n]+)",
        design,
        re.IGNORECASE,
    ):
        references.update(referenced_ids(match.group(1)))
    return references


def task_blocks(tasks):
    headers = list(TASK_HEADER.finditer(tasks))
    blocks = []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(tasks)
        blocks.append((header.group("id"), tasks[header.start():end]))
    return blocks


def task_metadata(tasks, headers):
    owned = defaultdict(lambda: defaultdict(list))
    orphaned = []
    noncanonical = []
    for entry in METADATA_ENTRY.finditer(tasks):
        entry_indent = len(entry.group("indent").expandtabs(4))
        owners = [
            header
            for header in headers
            if header.start() < entry.start()
            and len(header.group("indent").expandtabs(4)) < entry_indent
        ]
        raw_label = entry.group("label")
        label = next(
            canonical
            for canonical in CANONICAL_METADATA
            if canonical.casefold() == raw_label.casefold()
        )
        if raw_label != label:
            noncanonical.append(raw_label)
        if not owners:
            orphaned.append(label)
            continue
        owner = owners[-1].group("id")
        owned[owner][label].append(entry.group("value").strip())
    return owned, orphaned, noncanonical


def dependency_ids(value):
    if value.strip().lower() == "none":
        return set(), None
    dependency_list = r"\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)*"
    if not re.fullmatch(dependency_list, value.strip()):
        return set(), "use 'none' or a comma-separated list of task IDs"
    return set(TASK_ID_REFERENCE.findall(value)), None


def path_items(value):
    return [
        item.strip().strip("`")
        for item in value.split(",")
        if item.strip().strip("`").lower() not in {"", "none", "n/a"}
    ]


def broad_write_glob(path):
    normalized = path.replace("\\", "/").removeprefix("./")
    parts = [part for part in normalized.split("/") if part]
    return "**" in normalized or any(
        "*" in part or "?" in part or "[" in part for part in parts[:-1]
    ) or normalized in {"*", "**"}


def subsystem(path):
    parts = [
        part
        for part in path.replace("\\", "/").removeprefix("./").split("/")
        if part
    ]
    if not parts:
        return None
    if len(parts) > 1 and parts[0] in CONTAINER_ROOTS:
        return "/".join(parts[:2])
    if len(parts) == 1:
        return "<repository root>"
    return parts[0]


def validate_contents(feature_name, requirements, design, tasks):
    errors = []
    warnings = []

    if not KEBAB_CASE.fullmatch(feature_name):
        errors.append(f"feature directory must be kebab-case: {feature_name}")

    requirement_list = REQUIREMENT_ID.findall(requirements)
    requirement_counts = Counter(requirement_list)
    requirement_ids = set(requirement_list)
    if not requirement_ids:
        errors.append("requirements.md contains no explicit acceptance criterion IDs")
    duplicates = sorted(item for item, count in requirement_counts.items() if count > 1)
    if duplicates:
        errors.append(f"duplicate acceptance criterion IDs: {', '.join(duplicates)}")

    design_ids = design_references(design)
    unknown_design = sorted(design_ids - requirement_ids)
    if unknown_design:
        errors.append(f"design.md references unknown requirements: {', '.join(unknown_design)}")
    missing_design = sorted(requirement_ids - design_ids)
    if missing_design:
        errors.append(f"design.md does not trace requirements: {', '.join(missing_design)}")

    headers = list(TASK_HEADER.finditer(tasks))
    blocks = task_blocks(tasks)
    task_ids = [header.group("id") for header in headers]
    known_task_ids = set(task_ids)
    duplicate_tasks = sorted(item for item, count in Counter(task_ids).items() if count > 1)
    if duplicate_tasks:
        errors.append(f"duplicate task IDs: {', '.join(duplicate_tasks)}")
    if not blocks:
        errors.append("tasks.md contains no checkbox tasks with numeric IDs")

    epic_ids = {task_id for task_id in task_ids if "." not in task_id}
    leaf_ids = {task_id for task_id in task_ids if task_id.count(".") == 1}
    deep_ids = sorted(task_id for task_id in task_ids if task_id.count(".") > 1)
    if deep_ids:
        errors.append(
            "tasks use unsupported nesting below implementation leaves: "
            + ", ".join(deep_ids)
        )

    headings = list(HEADING.finditer(tasks))
    for header in headers:
        task_id = header.group("id")
        indent = header.group("indent")
        if task_id in epic_ids:
            if indent:
                errors.append(f"epic task {task_id} must be a top-level checkbox")
            if not any(leaf.startswith(f"{task_id}.") for leaf in leaf_ids):
                errors.append(f"epic task {task_id} has no implementation leaves")
            preceding_headings = [
                heading for heading in headings if heading.start() < header.start()
            ]
            if not preceding_headings or not preceding_headings[-1].group(
                "title"
            ).lower().startswith("milestone"):
                errors.append(f"epic task {task_id} must appear under a milestone heading")
        elif task_id in leaf_ids:
            parent_id = task_id.split(".", 1)[0]
            if not indent:
                errors.append(
                    f"implementation leaf {task_id} must be indented under epic {parent_id}"
                )
            if parent_id not in epic_ids:
                errors.append(f"implementation leaf {task_id} has no parent epic {parent_id}")
            preceding_epics = [
                candidate
                for candidate in headers
                if candidate.start() < header.start()
                and candidate.group("id") in epic_ids
            ]
            if not preceding_epics or preceding_epics[-1].group("id") != parent_id:
                errors.append(
                    f"implementation leaf {task_id} is not nested under epic {parent_id}"
                )

    owned_metadata, orphaned_metadata, noncanonical_metadata = task_metadata(tasks, headers)
    if noncanonical_metadata:
        errors.append(
            "task metadata must use exact canonical capitalization: "
            + ", ".join(sorted(set(noncanonical_metadata)))
        )
    if orphaned_metadata:
        errors.append(
            "task metadata is not nested under a task: "
            + ", ".join(sorted(set(orphaned_metadata)))
        )
    for task_id in sorted(epic_ids):
        labels = sorted(owned_metadata.get(task_id, {}))
        if labels:
            errors.append(
                f"metadata must belong to implementation leaves, but epic {task_id} has: "
                + ", ".join(labels)
            )

    task_requirement_ids = set()
    dependencies = {}
    write_owners = defaultdict(list)
    headers_by_id = {header.group("id"): header for header in headers}

    for task_id in sorted(leaf_ids):
        task_values = owned_metadata.get(task_id, {})
        duplicate_metadata = sorted(
            label for label, values in task_values.items() if len(values) > 1
        )
        if duplicate_metadata:
            errors.append(
                f"task {task_id} has duplicate metadata: {', '.join(duplicate_metadata)}"
            )
        values = {
            label: task_values[label][0] if len(task_values.get(label, [])) == 1 else None
            for label in REQUIRED_METADATA
        }
        missing = [label for label, value in values.items() if value is None]
        if missing:
            errors.append(f"task {task_id} is missing metadata: {', '.join(missing)}")
            continue
        empty = [label for label, value in values.items() if not value]
        if empty:
            errors.append(f"task {task_id} has empty metadata: {', '.join(empty)}")
            continue

        state = headers_by_id[task_id].group("state")
        if state in {"x", "X", "-"} and not task_values.get("Evidence"):
            warnings.append(
                f"task {task_id} is complete or superseded without _Evidence; "
                "preserved for compatibility but future transitions must record evidence"
            )

        task_requirement_ids.update(referenced_ids(values["Requirements"]))
        parsed_dependencies, dependency_error = dependency_ids(values["Depends on"])
        if dependency_error:
            errors.append(f"task {task_id} has invalid dependencies: {dependency_error}")
        dependencies[task_id] = parsed_dependencies
        unknown_dependencies = sorted(parsed_dependencies - known_task_ids)
        if unknown_dependencies:
            errors.append(
                f"task {task_id} depends on unknown tasks: {', '.join(unknown_dependencies)}"
            )
        epic_dependencies = sorted(parsed_dependencies & epic_ids)
        if epic_dependencies:
            errors.append(
                f"task {task_id} depends on parent epics instead of leaves: "
                + ", ".join(epic_dependencies)
            )
        if task_id in parsed_dependencies:
            errors.append(f"task {task_id} depends on itself")
        writes = path_items(values["Writes"])
        broad_globs = sorted(path for path in writes if broad_write_glob(path))
        if broad_globs:
            warnings.append(
                f"task {task_id} uses broad write globs: {', '.join(broad_globs)}"
            )
        if len(writes) > 5:
            warnings.append(
                f"task {task_id} has {len(writes)} write targets; review whether outcomes should split"
            )
        subsystems = sorted(
            item for item in {subsystem(path) for path in writes} if item is not None
        )
        if len(subsystems) > 1:
            warnings.append(
                f"task {task_id} spans top-level subsystems: {', '.join(subsystems)}"
            )
        for path in writes:
            if task_id not in write_owners[path]:
                write_owners[path].append(task_id)

    unknown_tasks = sorted(task_requirement_ids - requirement_ids)
    if unknown_tasks:
        errors.append(f"tasks.md references unknown requirements: {', '.join(unknown_tasks)}")
    missing_tasks = sorted(requirement_ids - task_requirement_ids)
    if missing_tasks:
        errors.append(f"tasks.md does not cover requirements: {', '.join(missing_tasks)}")

    visiting = set()
    visited = set()

    def visit(task_id):
        if task_id in visiting:
            return True
        if task_id in visited:
            return False
        visiting.add(task_id)
        if any(visit(dependency) for dependency in dependencies.get(task_id, set())):
            return True
        visiting.remove(task_id)
        visited.add(task_id)
        return False

    if any(visit(task_id) for task_id in dependencies if task_id not in visited):
        errors.append("tasks.md contains a dependency cycle")

    def transitively_depends_on(task_id, expected_dependency, seen=None):
        if seen is None:
            seen = set()
        if task_id in seen:
            return False
        seen.add(task_id)
        direct = dependencies.get(task_id, set())
        return expected_dependency in direct or any(
            dependency in leaf_ids
            and transitively_depends_on(dependency, expected_dependency, seen)
            for dependency in direct
        )

    for path, owners in sorted(write_owners.items()):
        unsequenced = []
        for index, owner in enumerate(owners):
            for other in owners[index + 1 :]:
                if not transitively_depends_on(owner, other) and not transitively_depends_on(
                    other, owner
                ):
                    unsequenced.extend((owner, other))
        if unsequenced:
            warnings.append(
                f"sequence overlapping write '{path}' in tasks "
                + ", ".join(sorted(set(unsequenced)))
            )

    return errors, warnings


LEGACY_REQUIREMENT_HEADING = re.compile(r"^### Requirement\s+([0-9]+)\b")
LEGACY_ACCEPTANCE_CRITERION = re.compile(r"^\s*([0-9]+)\.\s+\S")
LEGACY_DECISION_HEADING = re.compile(r"^###\s+(D[0-9]+):\s+\S")
LEGACY_PROPERTY_HEADING = re.compile(r"^### Property\s+([0-9]+)\b")
LEGACY_PROPERTY_VALIDATES = re.compile(
    r"Validates:\s*Requirement\s+([0-9]+\.[0-9]+)\b"
)
LEGACY_CRITERION_ID = re.compile(r"^[0-9]+\.[0-9]+$")
LEGACY_DECISION_IN_TEXT = re.compile(r"\bD[0-9]+\b")
LEGACY_TASK_HEADING = re.compile(
    r"^- \[([ xX~-])\]\s+(?:(\d+(?:\.\d+)*)[.)]\s+)?(.+)$"
)
LEGACY_NESTED_TASK = re.compile(r"^\s+- \[[ xX~-]\]\s+")
LEGACY_TASK_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
LEGACY_NO_TASK = re.compile(
    r"^\s*-\s+No task:\s+([0-9]+\.[0-9]+)\s+(?:-|—)\s+\S", re.IGNORECASE
)
LEGACY_PLACEHOLDERS = {"none", "tbd", "todo", "unknown", "n/a"}


@dataclass
class LegacyTask:
    title: str
    marker: str
    line: int
    body: list[str]
    metadata: dict[str, str]

    @property
    def task_id(self):
        return self.metadata.get("_id")


@dataclass
class LegacyDesignCoverage:
    criteria: set[str]
    decisions: set[str]


def structural_lines(text):
    result = []
    in_fence = False
    in_comment = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            result.append("")
            continue
        if in_fence:
            result.append("")
            continue
        visible = line
        while True:
            if in_comment:
                end = visible.find("-->")
                if end == -1:
                    visible = ""
                    break
                visible = visible[end + 3 :]
                in_comment = False
                continue
            start = visible.find("<!--")
            if start == -1:
                break
            end = visible.find("-->", start + 4)
            if end == -1:
                visible = visible[:start]
                in_comment = True
                break
            visible = visible[:start] + visible[end + 3 :]
        result.append(visible)
    return result


def legacy_parse_requirements(text, errors):
    headings = set()
    criteria = set()
    criteria_by_heading = {}
    current_heading = None
    in_acceptance_criteria = False

    for line_number, line in enumerate(structural_lines(text), 1):
        heading_match = LEGACY_REQUIREMENT_HEADING.match(line)
        if heading_match:
            current_heading = heading_match.group(1)
            if current_heading in headings:
                errors.append(
                    f"requirements.md:{line_number}: duplicate requirement {current_heading}"
                )
            headings.add(current_heading)
            criteria_by_heading.setdefault(current_heading, set())
            in_acceptance_criteria = False
            continue

        if line.casefold() == "#### acceptance criteria":
            in_acceptance_criteria = True
            continue
        if line.startswith("#"):
            in_acceptance_criteria = False

        if current_heading and in_acceptance_criteria:
            criterion_match = LEGACY_ACCEPTANCE_CRITERION.match(line)
            if criterion_match:
                identifier = f"{current_heading}.{criterion_match.group(1)}"
                if identifier in criteria:
                    errors.append(
                        f"requirements.md:{line_number}: duplicate criterion {identifier}"
                    )
                criteria.add(identifier)
                criteria_by_heading[current_heading].add(identifier)

    if not headings:
        errors.append("requirements.md: no '### Requirement N' headings found")
    for heading in sorted(headings, key=int):
        if not criteria_by_heading[heading]:
            errors.append(
                f"requirements.md: requirement {heading} has no acceptance criteria"
            )
    return criteria


def split_table_row(line):
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def is_separator_row(cells):
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def legacy_validate_coverage(source, references, requirements, errors):
    for requirement in sorted(requirements - references):
        errors.append(f"{source}: criterion {requirement} has no coverage")


def legacy_parse_design(text, requirements, errors):
    lines = structural_lines(text)
    decisions = set()
    properties = set()
    property_validation = {}
    current_property = None
    in_traceability = False
    saw_traceability = False
    saw_header = False
    coverage = set()

    for line_number, line in enumerate(lines, 1):
        decision_match = LEGACY_DECISION_HEADING.match(line)
        if decision_match:
            identifier = decision_match.group(1)
            if identifier in decisions:
                errors.append(f"design.md:{line_number}: duplicate decision {identifier}")
            decisions.add(identifier)

        property_match = LEGACY_PROPERTY_HEADING.match(line)
        if property_match:
            current_property = property_match.group(1)
            if current_property in properties:
                errors.append(
                    f"design.md:{line_number}: duplicate property {current_property}"
                )
            properties.add(current_property)
            property_validation[current_property] = False
        elif line.startswith("### ") or line.startswith("## "):
            current_property = None

        validation_match = LEGACY_PROPERTY_VALIDATES.search(line)
        if validation_match and current_property:
            criterion = validation_match.group(1)
            property_validation[current_property] = True
            if criterion not in requirements:
                errors.append(
                    f"design.md:{line_number}: property references missing criterion {criterion}"
                )

        if line.casefold() == "## traceability":
            in_traceability = True
            saw_traceability = True
            saw_header = False
            continue
        if in_traceability and line.startswith("## "):
            in_traceability = False
        if not in_traceability:
            continue

        cells = split_table_row(line)
        if cells is None or is_separator_row(cells):
            continue
        if not saw_header:
            expected = [
                "criterion",
                "design coverage",
                "verification type",
                "planned check / expected signal",
            ]
            if [cell.casefold() for cell in cells] != expected:
                errors.append(
                    f"design.md:{line_number}: Traceability table has unexpected columns"
                )
            saw_header = True
            continue
        if len(cells) != 4:
            errors.append(
                f"design.md:{line_number}: Traceability row must contain four columns"
            )
            continue
        criterion, design_mapping, verification_type, planned_check = cells
        if not LEGACY_CRITERION_ID.fullmatch(criterion):
            errors.append(
                f"design.md:{line_number}: invalid criterion reference {criterion!r}"
            )
            continue
        coverage.add(criterion)
        if criterion not in requirements:
            errors.append(f"design.md:{line_number}: references missing criterion {criterion}")
        mapped_decisions = set(LEGACY_DECISION_IN_TEXT.findall(design_mapping))
        if not mapped_decisions:
            errors.append(
                f"design.md:{line_number}: criterion {criterion} lacks a design decision reference"
            )
        for decision in sorted(mapped_decisions - decisions):
            errors.append(f"design.md:{line_number}: references missing decision {decision}")
        if not verification_type or verification_type.casefold() in LEGACY_PLACEHOLDERS:
            errors.append(
                f"design.md:{line_number}: criterion {criterion} lacks a verification type"
            )
        if not planned_check or planned_check.casefold() in LEGACY_PLACEHOLDERS:
            errors.append(
                f"design.md:{line_number}: criterion {criterion} lacks a planned check"
            )

    if not saw_traceability:
        errors.append("design.md: no '## Traceability' section found")
    elif not saw_header:
        errors.append("design.md: Traceability table is missing")
    for identifier, validated in property_validation.items():
        if not validated:
            errors.append(f"design.md: property {identifier} has no Validates reference")
    legacy_validate_coverage("design.md", coverage, requirements, errors)
    return LegacyDesignCoverage(coverage, decisions)


def legacy_metadata_from_body(body, line, errors):
    metadata = {}
    pattern = re.compile(r"^\s*-\s+(_?[A-Za-z][A-Za-z_ ]*):\s*(.*?)\s*$")
    for offset, source in enumerate(body, 1):
        match = pattern.match(source)
        if not match:
            continue
        key = match.group(1)
        canonical = key.casefold()
        value = match.group(2).strip().strip("_").strip()
        if canonical in metadata:
            errors.append(f"tasks.md:{line + offset}: duplicate metadata {key}")
        metadata[canonical] = value
    return metadata


def legacy_parse_tasks(text, errors):
    lines = structural_lines(text)
    starts = []
    exemptions = set()
    for index, line in enumerate(lines):
        if LEGACY_NESTED_TASK.match(line):
            errors.append(
                f"tasks.md:{index + 1}: legacy executable packets must be top-level checkboxes"
            )
        match = LEGACY_TASK_HEADING.match(line)
        if match:
            starts.append((index, match))
        exemption = LEGACY_NO_TASK.match(line)
        if exemption:
            exemptions.add(exemption.group(1))

    tasks = []
    for position, (start, match) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        for candidate in range(start + 1, end):
            if lines[candidate].strip() and not lines[candidate].startswith((" ", "\t")):
                end = candidate
                break
        body = lines[start + 1 : end]
        metadata = legacy_metadata_from_body(body, start + 1, errors)
        tasks.append(
            LegacyTask(match.group(3).strip(), match.group(1), start + 1, body, metadata)
        )
    return tasks, exemptions


def comma_values(value):
    if not value:
        return []
    return [item.strip().strip("`") for item in value.split(",") if item.strip()]


def legacy_exact_criterion_references(value, source, errors):
    references = set()
    for item in comma_values(value):
        if not LEGACY_CRITERION_ID.fullmatch(item):
            errors.append(
                f"{source}: invalid criterion reference {item!r}; use exact IDs such as 1.2"
            )
        else:
            references.add(item)
    return references


def legacy_normalized_paths(value, source, errors):
    paths = []
    for item in comma_values(value):
        normalized = posixpath.normpath(item)
        if item.startswith("/") or normalized == ".." or normalized.startswith("../"):
            errors.append(f"{source}: path must be repository-relative: {item!r}")
            continue
        paths.append(normalized.removeprefix("./"))
    return paths


def legacy_paths_overlap(left, right):
    left = left.rstrip("/")
    right = right.rstrip("/")
    return left == right or left.startswith(f"{right}/") or right.startswith(f"{left}/")


def legacy_validate_task_graph(tasks, errors):
    tasks_by_id = {task.task_id: task for task in tasks if task.task_id}
    dependencies = {}
    for task in tasks:
        if not task.task_id:
            continue
        task_dependencies = comma_values(task.metadata.get("_blocked_by"))
        dependencies[task.task_id] = task_dependencies
        for dependency in task_dependencies:
            if dependency not in tasks_by_id:
                errors.append(
                    f"tasks.md:{task.line}: task {task.task_id} depends on missing task {dependency}"
                )
            if dependency == task.task_id:
                errors.append(
                    f"tasks.md:{task.line}: task {task.task_id} depends on itself"
                )
            dependency_task = tasks_by_id.get(dependency)
            task_wave = task.metadata.get("_wave")
            dependency_wave = dependency_task.metadata.get("_wave") if dependency_task else None
            if (
                task_wave
                and dependency_wave
                and task_wave.isdigit()
                and dependency_wave.isdigit()
                and int(dependency_wave) >= int(task_wave)
            ):
                errors.append(
                    f"tasks.md:{task.line}: dependency {dependency} must be in an earlier wave than {task.task_id}"
                )

    visiting = set()
    visited = set()

    def visit(identifier, path):
        if identifier in visited:
            return
        if identifier in visiting:
            cycle = " -> ".join(path[path.index(identifier) :] + [identifier])
            errors.append(f"tasks.md: dependency cycle: {cycle}")
            return
        visiting.add(identifier)
        for dependency in dependencies.get(identifier, []):
            if dependency in dependencies:
                visit(dependency, path + [dependency])
        visiting.remove(identifier)
        visited.add(identifier)

    for identifier in dependencies:
        visit(identifier, [identifier])


def legacy_validate_task_conflicts(tasks, errors):
    for index, left in enumerate(tasks):
        left_wave = left.metadata.get("_wave")
        if not left_wave:
            continue
        left_reads = legacy_normalized_paths(
            left.metadata.get("_reads"), f"tasks.md:{left.line}", errors
        )
        left_writes = legacy_normalized_paths(
            left.metadata.get("_writes"), f"tasks.md:{left.line}", errors
        )
        for right in tasks[index + 1 :]:
            if right.metadata.get("_wave") != left_wave:
                continue
            right_reads = legacy_normalized_paths(
                right.metadata.get("_reads"), f"tasks.md:{right.line}", errors
            )
            right_writes = legacy_normalized_paths(
                right.metadata.get("_writes"), f"tasks.md:{right.line}", errors
            )
            conflict = any(
                legacy_paths_overlap(a, b)
                for a in left_writes
                for b in right_writes + right_reads
            )
            conflict = conflict or any(
                legacy_paths_overlap(a, b) for a in right_writes for b in left_reads
            )
            if conflict:
                errors.append(
                    f"tasks.md:{right.line}: tasks {left.task_id or left.title!r} and "
                    f"{right.task_id or right.title!r} have conflicting paths in wave {left_wave}"
                )


def legacy_validate_tasks(text, requirements, decisions, errors, warnings):
    tasks, exemptions = legacy_parse_tasks(text, errors)
    if not tasks:
        errors.append("tasks.md: no top-level legacy checkbox packets found")
        return exemptions

    task_ids = set()
    task_references = set(exemptions)
    required_fields = (
        "_id",
        "_validation",
        "_requirements",
        "outcome",
        "design",
        "done when",
    )

    for task in tasks:
        for field in required_fields:
            value = task.metadata.get(field)
            if not value:
                errors.append(f"tasks.md:{task.line}: task {task.title!r} lacks {field}")
            elif value.casefold() in LEGACY_PLACEHOLDERS:
                errors.append(
                    f"tasks.md:{task.line}: task {task.title!r} has placeholder {field}"
                )

        task_id = task.task_id
        if task_id:
            if not LEGACY_TASK_ID.fullmatch(task_id):
                errors.append(
                    f"tasks.md:{task.line}: invalid durable task ID {task_id!r}"
                )
            if task_id in task_ids:
                errors.append(f"tasks.md:{task.line}: duplicate durable task ID {task_id}")
            task_ids.add(task_id)

        references = legacy_exact_criterion_references(
            task.metadata.get("_requirements") or "", f"tasks.md:{task.line}", errors
        )
        task_references.update(references)
        for reference in sorted(references - requirements):
            errors.append(f"tasks.md:{task.line}: references missing criterion {reference}")

        design_value = task.metadata.get("design") or ""
        design_references = set(LEGACY_DECISION_IN_TEXT.findall(design_value))
        if design_value and not design_references:
            errors.append(
                f"tasks.md:{task.line}: task {task.title!r} lacks a design decision reference"
            )
        for decision in sorted(design_references - decisions):
            errors.append(f"tasks.md:{task.line}: references missing decision {decision}")

        wave = task.metadata.get("_wave")
        if wave and (not wave.isdigit() or int(wave) < 1):
            errors.append(f"tasks.md:{task.line}: _wave must be a positive integer")
        legacy_normalized_paths(
            task.metadata.get("_reads"), f"tasks.md:{task.line}", errors
        )
        legacy_normalized_paths(
            task.metadata.get("_writes"), f"tasks.md:{task.line}", errors
        )
        if task.marker in {"x", "X", "-"} and not task.metadata.get("_evidence"):
            warnings.append(
                f"legacy task {task_id or task.title!r} is complete or superseded "
                "without _Evidence; preserved for compatibility"
            )

    for exemption in sorted(exemptions - requirements):
        errors.append(f"tasks.md: no-task entry references missing criterion {exemption}")
    legacy_validate_coverage("tasks.md", task_references, requirements, errors)
    legacy_validate_task_graph(tasks, errors)
    legacy_validate_task_conflicts(tasks, errors)
    return task_references


def validate_legacy_contents(feature_name, requirements, design, tasks):
    errors = []
    warnings = [
        "legacy coding task dialect detected; compatibility validation only; "
        "use canonical mode for new or materially rewritten plans"
    ]
    if not KEBAB_CASE.fullmatch(feature_name):
        errors.append(
            "feature directory must use kebab-case, with optional semantic-version segments: "
            f"{feature_name}"
        )
    requirement_ids = legacy_parse_requirements(requirements, errors)
    design_coverage = legacy_parse_design(design, requirement_ids, errors)
    legacy_validate_tasks(
        tasks, requirement_ids, design_coverage.decisions, errors, warnings
    )
    return errors, warnings


def validate_partial_contents(feature_name, requirements, design, dialect):
    errors = []
    warnings = []
    if not KEBAB_CASE.fullmatch(feature_name):
        errors.append(
            "feature directory must use kebab-case, with optional semantic-version segments: "
            f"{feature_name}"
        )

    if dialect == "legacy":
        warnings.append(
            "legacy coding specification dialect detected; compatibility validation only"
        )
        requirement_ids = legacy_parse_requirements(requirements, errors)
        if design is not None:
            legacy_parse_design(design, requirement_ids, errors)
        return errors, warnings

    requirement_list = REQUIREMENT_ID.findall(requirements)
    requirement_counts = Counter(requirement_list)
    requirement_ids = set(requirement_list)
    if not requirement_ids:
        errors.append("requirements.md contains no explicit acceptance criterion IDs")
    duplicates = sorted(
        item for item, count in requirement_counts.items() if count > 1
    )
    if duplicates:
        errors.append(f"duplicate acceptance criterion IDs: {', '.join(duplicates)}")
    if design is not None:
        design_ids = design_references(design)
        unknown_design = sorted(design_ids - requirement_ids)
        if unknown_design:
            errors.append(
                f"design.md references unknown requirements: {', '.join(unknown_design)}"
            )
        missing_design = sorted(requirement_ids - design_ids)
        if missing_design:
            errors.append(
                f"design.md does not trace requirements: {', '.join(missing_design)}"
            )
    return errors, warnings


def run_self_test():
    requirements = """\
# Requirements: Example
### Requirement 1: Store values
1. **1.1** WHEN a value is submitted THEN THE system SHALL store it.
2. **1.2** IF the value is invalid, THEN THE system SHALL reject it.
"""
    design = """\
# Design: Example
## Requirements traceability
| Requirement | Design element | Verification |
| --- | --- | --- |
| 1.1 | Store | Integration test |
| 1.2 | Validate | Error-path test |
"""
    tasks = """\
# Implementation Plan: Example
## Tasks
### Milestone 1: Values are stored safely
- [ ] 1. Store values
  - [ ] 1.1. Store validated values in the existing repository
    - _Requirements: 1.1, 1.2_
    - _Depends on: none_
    - _Reads: src/lib.rs_
    - _Writes: src/lib.rs_
    - _Validation: focused test_
"""
    errors, warnings = validate_contents("example-feature", requirements, design, tasks)
    if errors or warnings:
        raise AssertionError(f"valid fixture failed: errors={errors}, warnings={warnings}")

    broken_tasks = tasks.replace("_Requirements: 1.1, 1.2_", "_Requirements: 9.9_")
    errors, _ = validate_contents("example-feature", requirements, design, broken_tasks)
    if not any("unknown requirements" in error for error in errors):
        raise AssertionError("invalid fixture did not report unknown task requirements")

    broken_tasks = tasks.replace("_Depends on: none_", "_Depends on: 9.1_")
    errors, _ = validate_contents("example-feature", requirements, design, broken_tasks)
    if not any("unknown tasks" in error for error in errors):
        raise AssertionError("invalid fixture did not report unknown dependencies")

    parent_dependency_tasks = tasks.replace("_Depends on: none_", "_Depends on: 1_")
    errors, _ = validate_contents(
        "example-feature", requirements, design, parent_dependency_tasks
    )
    if not any("parent epics instead of leaves" in error for error in errors):
        raise AssertionError("invalid fixture allowed a dependency on an epic")

    duplicate_metadata_tasks = tasks.replace(
        "    - _Validation: focused test_",
        "    - _Validation: focused test_\n    - _Validation: another test_",
    )
    errors, _ = validate_contents(
        "example-feature", requirements, design, duplicate_metadata_tasks
    )
    if not any("duplicate metadata" in error for error in errors):
        raise AssertionError("invalid fixture allowed duplicate leaf metadata")

    epic_leaf = tasks.replace(
        "  - [ ] 1.1. Store validated values in the existing repository\n", ""
    ).replace("    - _", "  - _")
    errors, _ = validate_contents("example-feature", requirements, design, epic_leaf)
    if not any("has no implementation leaves" in error for error in errors):
        raise AssertionError("epic-as-leaf fixture did not require decomposition")
    if not any("metadata must belong to implementation leaves" in error for error in errors):
        raise AssertionError("epic metadata fixture did not reject metadata on an epic")
    if not any("does not cover requirements" in error for error in errors):
        raise AssertionError("epic metadata fixture incorrectly traced requirements to an epic")

    broad_tasks = tasks.replace(
        "_Writes: src/lib.rs_",
        "_Writes: backend/**, web/view.tsx, clients/mobile.kt, migrations/1.sql, "
        "deploy/app.yaml, docs/sharing.md_",
    )
    errors, warnings = validate_contents("example-feature", requirements, design, broad_tasks)
    if errors:
        raise AssertionError(f"warning fixture failed: errors={errors}")
    expected_warnings = ("broad write globs", "6 write targets", "top-level subsystems")
    for expected in expected_warnings:
        if not any(expected in warning for warning in warnings):
            raise AssertionError(f"warning fixture did not report {expected!r}")

    overlapping_tasks = tasks.replace(
        "    - _Validation: focused test_\n",
        "    - _Validation: focused test_\n"
        "  - [ ] 1.2. Reject invalid values at the repository boundary\n"
        "    - _Requirements: 1.2_\n"
        "    - _Depends on: none_\n"
        "    - _Reads: src/lib.rs_\n"
        "    - _Writes: src/lib.rs_\n"
        "    - _Validation: focused rejection test_\n",
    )
    errors, warnings = validate_contents(
        "example-feature", requirements, design, overlapping_tasks
    )
    if errors:
        raise AssertionError(f"overlap fixture failed: errors={errors}")
    if not any("sequence overlapping write" in warning for warning in warnings):
        raise AssertionError("overlap fixture did not report unsequenced writes")

    sequenced_tasks = overlapping_tasks.replace(
        "    - _Depends on: none_\n    - _Reads: src/lib.rs_\n"
        "    - _Writes: src/lib.rs_\n    - _Validation: focused rejection test_",
        "    - _Depends on: 1.1_\n    - _Reads: src/lib.rs_\n"
        "    - _Writes: src/lib.rs_\n    - _Validation: focused rejection test_",
    )
    errors, warnings = validate_contents(
        "example-feature", requirements, design, sequenced_tasks
    )
    if errors or any("sequence overlapping write" in warning for warning in warnings):
        raise AssertionError(
            f"sequenced overlap fixture failed: errors={errors}, warnings={warnings}"
        )

    print("validate_spec self-test passed")


def main():
    parser = argparse.ArgumentParser(description="Validate a feature specification pack")
    parser.add_argument("feature_dir", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="require requirements.md, design.md, and tasks.md",
    )
    parser.add_argument(
        "--dialect",
        choices=("auto", "canonical", "legacy"),
        default="auto",
        help="validate canonical plans, legacy coding packets, or auto-detect existing packs",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0
    if args.feature_dir is None:
        parser.error("feature_dir is required unless --self-test is used")

    required_files = ("requirements.md", "design.md", "tasks.md")
    existing_files = {
        name: args.feature_dir / name
        for name in required_files
        if (args.feature_dir / name).is_file()
    }
    if not existing_files:
        print("ERROR: no specification documents found", file=sys.stderr)
        return 1

    missing_files = [name for name in required_files if name not in existing_files]
    if args.require_complete and missing_files:
        print(f"ERROR: missing files: {', '.join(missing_files)}", file=sys.stderr)
        return 1
    if "tasks.md" in existing_files and missing_files:
        print(
            "ERROR: tasks.md requires a complete requirements.md, design.md, and tasks.md pack",
            file=sys.stderr,
        )
        return 1
    if "design.md" in existing_files and "requirements.md" not in existing_files:
        print("ERROR: design.md requires requirements.md", file=sys.stderr)
        return 1

    try:
        contents = {
            name: path.read_text(encoding="utf-8")
            for name, path in existing_files.items()
        }
    except (OSError, UnicodeError) as error:
        print(f"ERROR: cannot read specification: {error}", file=sys.stderr)
        return 1

    tasks = contents.get("tasks.md")
    detected_dialect = args.dialect
    if detected_dialect == "auto":
        detected_dialect = (
            "legacy" if tasks and LEGACY_DIALECT_MARKER.search(tasks) else "canonical"
        )

    if tasks is None:
        errors, warnings = validate_partial_contents(
            args.feature_dir.name,
            contents["requirements.md"],
            contents.get("design.md"),
            detected_dialect,
        )
    elif detected_dialect == "legacy":
        errors, warnings = validate_legacy_contents(
            args.feature_dir.name,
            contents["requirements.md"],
            contents["design.md"],
            tasks,
        )
    else:
        errors, warnings = validate_contents(
            args.feature_dir.name,
            contents["requirements.md"],
            contents["design.md"],
            tasks,
        )
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1

    if tasks is not None and detected_dialect == "canonical":
        print(
            "MANUAL REVIEW REQUIRED: confirm every implementation leaf is one focused, "
            "independently reviewable unit"
        )
    print(f"Validated {args.feature_dir} ({detected_dialect} dialect)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
