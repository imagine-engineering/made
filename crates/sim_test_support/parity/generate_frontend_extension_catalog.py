#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
SOURCE = ROOT / "catalogs/frontend-extensions.csv"
POLICY = ROOT / "frontend-extension-policy.json"
CATALOG = ROOT / "catalogs/frontend-extension-dispositions.csv"
RUST = WORKSPACE / "crates/sim_ui/src/generated_frontend_extension_catalog.rs"

CLASSIFICATIONS = {
    "declarative_rust_wasm": "DeclarativeRustWasm",
    "legacy_identifier_mapping": "LegacyIdentifierMapping",
    "lossless_placeholder": "LosslessPlaceholder",
    "documented_only": "DocumentedOnly",
    "deliberate_defer": "DeliberateDefer",
}
DECLARATIVE_SURFACES = {
    "command",
    "keybinding",
    "menu",
    "setting",
    "bottom-panel",
    "node-panel",
    "about-badge",
    "topbar-badge",
    "action-bar-button",
    "node-widget",
    "selection-toolbox",
    "canvas-menu",
    "node-menu",
}
IDENTIFIER = re.compile(r"^[a-z][a-z0-9-]*(?:\.[a-z][a-z0-9-]*)*$")


def rust_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if policy.get("schema_version") != 1:
        raise RuntimeError("frontend-extension-policy.json must use schema_version 1")
    entries = policy.get("entries")
    if not isinstance(entries, dict):
        raise RuntimeError("frontend-extension-policy.json entries must be an object")
    source_ids = [row["feature_id"] for row in source_rows]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("frontend extension source catalog has duplicate feature IDs")
    missing = sorted(set(source_ids) - set(entries))
    extra = sorted(set(entries) - set(source_ids))
    if missing or extra:
        raise RuntimeError(f"frontend extension policy closure failed: missing={missing}, extra={extra}")

    output_rows: list[dict[str, str]] = []
    for row in source_rows:
        feature_id = row["feature_id"]
        entry = entries[feature_id]
        if not isinstance(entry, list) or len(entry) != 4:
            raise RuntimeError(f"{feature_id} policy entry must have four fields")
        classification, native_surface, replacement_owner, reason = entry
        if classification not in CLASSIFICATIONS:
            raise RuntimeError(f"{feature_id} has unknown classification {classification!r}")
        if native_surface is not None and (not isinstance(native_surface, str) or not native_surface):
            raise RuntimeError(f"{feature_id} has an invalid native surface")
        if classification == "declarative_rust_wasm" and native_surface not in DECLARATIVE_SURFACES:
            raise RuntimeError(f"{feature_id} has an undeclared Rust/WASM surface {native_surface!r}")
        if classification == "lossless_placeholder" and native_surface is not None:
            raise RuntimeError(f"{feature_id} placeholder cannot advertise a native surface")
        if not isinstance(replacement_owner, str) or not IDENTIFIER.fullmatch(replacement_owner):
            raise RuntimeError(f"{feature_id} has an invalid replacement owner")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1_024:
            raise RuntimeError(f"{feature_id} has an invalid reason")
        output_rows.append(
            {
                **row,
                "classification": classification,
                "native_surface": native_surface or "",
                "replacement_owner": replacement_owner,
                "reason": reason,
                "production_javascript": "prohibited",
            }
        )

    fieldnames = [
        *source_rows[0],
        "classification",
        "native_surface",
        "replacement_owner",
        "reason",
        "production_javascript",
    ]
    with CATALOG.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    lines = [
        "#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]",
        "pub enum GeneratedFrontendExtensionDispositionKind {",
        *[f"    {variant}," for variant in CLASSIFICATIONS.values()],
        "}",
        "",
        "#[derive(Clone, Copy, Debug, Eq, PartialEq)]",
        "pub struct GeneratedFrontendExtensionDisposition {",
        "    pub feature_id: &'static str,",
        "    pub entry_kind: &'static str,",
        "    pub name: &'static str,",
        "    pub behavior: &'static str,",
        "    pub availability: &'static str,",
        "    pub source_file: &'static str,",
        "    pub evidence_level: &'static str,",
        "    pub classification: GeneratedFrontendExtensionDispositionKind,",
        "    pub native_surface: Option<&'static str>,",
        "    pub replacement_owner: &'static str,",
        "    pub reason: &'static str,",
        "}",
        "",
        "pub const GENERATED_FRONTEND_EXTENSION_DISPOSITIONS: &[GeneratedFrontendExtensionDisposition] = &[",
    ]
    for row in output_rows:
        surface = (
            f"Some({rust_string(row['native_surface'])})"
            if row["native_surface"]
            else "None"
        )
        lines.extend(
            [
                "    GeneratedFrontendExtensionDisposition {",
                f"        feature_id: {rust_string(row['feature_id'])},",
                f"        entry_kind: {rust_string(row['entry_kind'])},",
                f"        name: {rust_string(row['name'])},",
                f"        behavior: {rust_string(row['behavior'])},",
                f"        availability: {rust_string(row['availability'])},",
                f"        source_file: {rust_string(row['source_file'])},",
                f"        evidence_level: {rust_string(row['evidence_level'])},",
                "        classification: "
                f"GeneratedFrontendExtensionDispositionKind::{CLASSIFICATIONS[row['classification']]},",
                f"        native_surface: {surface},",
                f"        replacement_owner: {rust_string(row['replacement_owner'])},",
                f"        reason: {rust_string(row['reason'])},",
                "    },",
            ]
        )
    lines.extend(["];"])
    RUST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated {len(output_rows)} frontend extension dispositions with zero JavaScript execution rows.")


if __name__ == "__main__":
    main()
