#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


CONSTANT_PATTERN = re.compile(
    r"^pub const ([A-Z][A-Z0-9_]*): u32 = (\d+);$", re.MULTILINE
)
RANGE_NAMES = {
    "PARAM_REPLACEABLE_KIND_MIN",
    "PARAM_REPLACEABLE_KIND_MAX",
    "EPHEMERAL_KIND_MIN",
    "EPHEMERAL_KIND_MAX",
}
PRIVACY_ARRAYS = {
    "AUTHOR_ONLY_KINDS": "AUTHOR_ONLY",
    "RESULT_GATED_KINDS": "RESULT_GATED",
    "P_GATED_KINDS": "RECIPIENT_GATED",
    "SHARED_GATED_KINDS": "AUTHOR_OR_SHARED",
}


@dataclass(frozen=True)
class CatalogKind:
    name: str
    value: int
    protocols: tuple[str, ...]
    status: str


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_catalog(path: Path) -> tuple[dict[str, CatalogKind], dict[str, int]]:
    event_kinds: dict[str, CatalogKind] = {}
    ranges: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["record_type"] not in {"event_kind", "kind_range"}:
                continue
            name = row["name"]
            try:
                value = int(row["numeric_value"])
            except ValueError as error:
                raise ValueError(f"{name}: invalid numeric value") from error
            if row["record_type"] == "kind_range":
                if name in ranges:
                    raise ValueError(f"duplicate kind range {name}")
                ranges[name] = value
                continue
            if name in event_kinds:
                raise ValueError(f"duplicate event kind {name}")
            protocols = tuple(filter(None, row["protocol_references"].split(";")))
            event_kinds[name] = CatalogKind(name, value, protocols, row["status"])
    return event_kinds, ranges


def load_source(path: Path) -> tuple[dict[str, int], dict[str, set[str]]]:
    source = path.read_text(encoding="utf-8")
    constants = {name: int(value) for name, value in CONSTANT_PATTERN.findall(source)}
    privacy: dict[str, set[str]] = {}
    for array_name in PRIVACY_ARRAYS:
        pattern = re.compile(
            rf"pub const {array_name}: &\[u32\] = &\[(.*?)\];", re.DOTALL
        )
        match = pattern.search(source)
        if match is None:
            raise ValueError(f"missing source privacy array {array_name}")
        body = re.sub(r"//.*", "", match.group(1))
        privacy[array_name] = set(re.findall(r"\b[A-Z][A-Z0-9_]+\b", body))
    return constants, privacy


def validate_inputs(
    event_kinds: dict[str, CatalogKind],
    ranges: dict[str, int],
    source_constants: dict[str, int],
    privacy: dict[str, set[str]],
) -> None:
    catalog_constants = {name: kind.value for name, kind in event_kinds.items()}
    catalog_constants.update(ranges)
    missing = sorted(set(source_constants) - set(catalog_constants))
    extra = sorted(set(catalog_constants) - set(source_constants))
    changed = sorted(
        name
        for name in set(source_constants) & set(catalog_constants)
        if source_constants[name] != catalog_constants[name]
    )
    if missing or extra or changed:
        raise ValueError(
            "unclassified or divergent Buzz kind constants: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    if set(ranges) != RANGE_NAMES:
        raise ValueError(f"kind ranges must be exactly {sorted(RANGE_NAMES)}")
    if len(event_kinds) != 133:
        raise ValueError(f"expected 133 event kinds, found {len(event_kinds)}")
    values: dict[int, str] = {}
    for kind in event_kinds.values():
        previous = values.get(kind.value)
        if previous is not None:
            raise ValueError(
                f"duplicate event kind value {kind.value}: {previous}, {kind.name}"
            )
        values[kind.value] = kind.name
        if kind.status not in {
            "registered",
            "defined-unused",
            "internal-not-relay-event",
        }:
            raise ValueError(f"{kind.name}: unsupported catalog status {kind.status!r}")
    for array_name, names in privacy.items():
        unknown = sorted(names - set(event_kinds))
        if unknown:
            raise ValueError(f"{array_name}: unknown event kinds {unknown}")


def rust_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def privacy_expression(name: str, privacy: dict[str, set[str]]) -> str:
    gates = [
        rust_gate
        for array_name, rust_gate in PRIVACY_ARRAYS.items()
        if name in privacy[array_name]
    ]
    if not gates:
        return "PrivacyGates::COMMUNITY"
    expression = f"PrivacyGates::{gates[0]}"
    for gate in gates[1:]:
        expression += f".union(PrivacyGates::{gate})"
    return expression


def persistence_expression(value: int) -> str:
    if 20000 <= value <= 29999:
        return "PersistenceClass::Ephemeral"
    if value in {0, 3, 41} or 10000 <= value <= 19999:
        return "PersistenceClass::Replaceable"
    if 30000 <= value <= 39999:
        return "PersistenceClass::ParameterizedReplaceable"
    return "PersistenceClass::Regular"


def replacement_expression(value: int) -> str:
    if 20000 <= value <= 29999:
        return "ReplacementBehavior::DiscardAfterDelivery"
    if value in {0, 3, 41} or 10000 <= value <= 19999:
        return "ReplacementBehavior::AuthorAndKind"
    if 30000 <= value <= 39999:
        return "ReplacementBehavior::AuthorKindAndDiscriminator"
    return "ReplacementBehavior::RetainAll"


def render_registry(
    event_kinds: dict[str, CatalogKind],
    ranges: dict[str, int],
    privacy: dict[str, set[str]],
) -> str:
    kinds_by_name = sorted(event_kinds.values(), key=lambda kind: kind.name)
    kinds_by_value = sorted(event_kinds.values(), key=lambda kind: kind.value)
    lines = [
        "// Generated by script/generate-kind-registry.py.",
        "// Do not edit by hand.",
        "",
        "use crate::head::PersistenceClass;",
        "",
        "#[derive(Clone, Copy, Debug, Eq, PartialEq)]",
        "pub enum ReplacementBehavior {",
        "    RetainAll,",
        "    AuthorAndKind,",
        "    AuthorKindAndDiscriminator,",
        "    DiscardAfterDelivery,",
        "}",
        "",
        "#[derive(Clone, Copy, Debug, Eq, PartialEq)]",
        "pub struct PrivacyGates(u8);",
        "",
        "impl PrivacyGates {",
        "    pub const COMMUNITY: Self = Self(0);",
        "    pub const AUTHOR_ONLY: Self = Self(1 << 0);",
        "    pub const RESULT_GATED: Self = Self(1 << 1);",
        "    pub const RECIPIENT_GATED: Self = Self(1 << 2);",
        "    pub const AUTHOR_OR_SHARED: Self = Self(1 << 3);",
        "",
        "    pub const fn union(self, other: Self) -> Self {",
        "        Self(self.0 | other.0)",
        "    }",
        "",
        "    pub const fn contains(self, gate: Self) -> bool {",
        "        gate.0 != 0 && self.0 & gate.0 == gate.0",
        "    }",
        "",
        "    pub const fn is_community_visible(self) -> bool {",
        "        self.0 == 0",
        "    }",
        "}",
        "",
        "#[derive(Clone, Copy, Debug, Eq, PartialEq)]",
        "pub enum KindStatus {",
        "    Registered,",
        "    DefinedUnused,",
        "    InternalNotRelayEvent,",
        "}",
        "",
        "#[derive(Clone, Copy, Debug, Eq, PartialEq)]",
        "pub struct KindMetadata {",
        "    pub name: &'static str,",
        "    pub value: u32,",
        "    pub protocols: &'static [&'static str],",
        "    pub persistence: PersistenceClass,",
        "    pub replacement: ReplacementBehavior,",
        "    pub privacy: PrivacyGates,",
        "    pub status: KindStatus,",
        "}",
        "",
    ]
    for name in sorted(ranges):
        lines.append(f"pub const {name}: u32 = {ranges[name]};")
    lines.append("")
    for kind in kinds_by_name:
        lines.append(f"pub const {kind.name}: u32 = {kind.value};")
    lines.extend(["", f"pub const EVENT_KIND_COUNT: usize = {len(kinds_by_value)};", ""])
    lines.append("pub static EVENT_KINDS: [KindMetadata; EVENT_KIND_COUNT] = [")
    for kind in kinds_by_value:
        protocols = ", ".join(rust_string(protocol) for protocol in kind.protocols)
        status = {
            "defined-unused": "DefinedUnused",
            "internal-not-relay-event": "InternalNotRelayEvent",
            "registered": "Registered",
        }[kind.status]
        lines.extend(
            [
                "    KindMetadata {",
                f"        name: {rust_string(kind.name)},",
                f"        value: {kind.value},",
                f"        protocols: &[{protocols}],",
                f"        persistence: {persistence_expression(kind.value)},",
                f"        replacement: {replacement_expression(kind.value)},",
                f"        privacy: {privacy_expression(kind.name, privacy)},",
                f"        status: KindStatus::{status},",
                "    },",
            ]
        )
    lines.extend(
        [
            "];",
            "",
            "pub fn metadata_for_kind(value: u32) -> Option<&'static KindMetadata> {",
            "    EVENT_KINDS",
            "        .binary_search_by_key(&value, |metadata| metadata.value)",
            "        .ok()",
            "        .and_then(|index| EVENT_KINDS.get(index))",
            "}",
            "",
            "#[cfg(test)]",
            "mod tests {",
            "    use super::*;",
            "",
            "    #[test]",
            "    fn registry_is_sorted_unique_and_complete() {",
            "        assert_eq!(EVENT_KINDS.len(), 133);",
            "        assert!(EVENT_KINDS.windows(2).all(|pair| pair[0].value < pair[1].value));",
            "        assert_eq!(metadata_for_kind(KIND_PROFILE).map(|kind| kind.name), Some(\"KIND_PROFILE\"));",
            "        assert!(metadata_for_kind(u32::MAX).is_none());",
            "    }",
            "",
            "    #[test]",
            "    fn boundary_kinds_have_expected_storage_and_replacement_rules() {",
            "        assert_eq!(crate::head::persistence_class(EPHEMERAL_KIND_MIN as u16), PersistenceClass::Ephemeral);",
            "        assert_eq!(crate::head::persistence_class(EPHEMERAL_KIND_MAX as u16), PersistenceClass::Ephemeral);",
            "        assert_eq!(crate::head::persistence_class(PARAM_REPLACEABLE_KIND_MIN as u16), PersistenceClass::ParameterizedReplaceable);",
            "        assert_eq!(crate::head::persistence_class(PARAM_REPLACEABLE_KIND_MAX as u16), PersistenceClass::ParameterizedReplaceable);",
            "        assert_eq!(metadata_for_kind(KIND_MUTE_LIST).map(|kind| kind.replacement), Some(ReplacementBehavior::AuthorAndKind));",
            "    }",
            "",
            "    #[test]",
            "    fn privacy_gates_preserve_buzz_overlap() {",
            "        let metric = metadata_for_kind(KIND_AGENT_TURN_METRIC).expect(\"metric kind\");",
            "        assert!(metric.privacy.contains(PrivacyGates::RESULT_GATED));",
            "        assert!(metric.privacy.contains(PrivacyGates::RECIPIENT_GATED));",
            "        let persona = metadata_for_kind(KIND_PERSONA).expect(\"persona kind\");",
            "        assert!(persona.privacy.contains(PrivacyGates::AUTHOR_OR_SHARED));",
            "        assert!(metadata_for_kind(KIND_TEXT_NOTE).expect(\"text kind\").privacy.is_community_visible());",
            "    }",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def format_rust(source: str) -> str:
    result = subprocess.run(
        ["rustfmt", "--edition", "2024", "--emit", "stdout"],
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"rustfmt failed: {result.stderr.strip()}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verify-unclassified-guard", action="store_true")
    arguments = parser.parse_args()

    root = repository_root()
    catalog_path = root / ".agents/local/collaboration/collaborative-workspace/catalogs/protocol.csv"
    source_path = root / "projects/buzz/crates/buzz-core/src/kind.rs"
    output_path = root / "crates/nostr_compat/src/generated_kinds.rs"
    try:
        event_kinds, ranges = load_catalog(catalog_path)
        source_constants, privacy = load_source(source_path)
        validate_inputs(event_kinds, ranges, source_constants, privacy)
        if arguments.verify_unclassified_guard:
            unclassified = dict(source_constants)
            unclassified["KIND_UNCLASSIFIED_TEST"] = 65534
            try:
                validate_inputs(event_kinds, ranges, unclassified, privacy)
            except ValueError as error:
                if "KIND_UNCLASSIFIED_TEST" not in str(error):
                    raise
            else:
                raise ValueError("unclassified-kind guard accepted a missing catalog row")
        rendered = format_rust(render_registry(event_kinds, ranges, privacy))
    except (OSError, ValueError) as error:
        print(f"kind registry error: {error}", file=sys.stderr)
        return 1

    if arguments.check:
        current = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        if current != rendered:
            print(f"generated kind registry is stale: {output_path}", file=sys.stderr)
            return 1
        print("Kind registry check passed: event_kinds=133 ranges=4")
        return 0

    output_path.write_text(rendered, encoding="utf-8")
    print(f"Generated {output_path}: event_kinds=133 ranges=4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
