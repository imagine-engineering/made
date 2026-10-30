#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
import subprocess

from generate_shell_catalog import KEYMAP_CONTEXT


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
CATALOG = ROOT / "catalogs/zed-architecture.csv"
EXECUTION_DISPOSITIONS = ROOT / "catalogs/native-execution-dispositions.csv"
GENERATED_EXECUTION_CATALOG = (
    REPO_ROOT / "crates/sim_ui/src/generated_execution_catalog.rs"
)

EXECUTION_UI_OWNER = "comfy-parity-execution-ui"
NATIVE_API_OWNER = "comfy-parity-native-api-host"
NATIVE_IMAGE_OWNER = "comfy-parity-native-execution-e2e"
NATIVE_MEMORY_OWNER = "comfy-parity-native-memory-planner"
WORKFLOW_FORMATS_OWNER = "comfy-parity-workflow-formats"
NATIVE_GRAPH_OWNER = "comfy-parity-native-graph"
WORKFLOW_EXPERIENCE_OWNER = "comfy-parity-workflow-experience"
ASSET_VIEWERS_OWNER = "comfy-parity-assets-editors-viewers"
SETTINGS_OWNER = "comfy-parity-settings-localization-ui"
DIAGNOSTICS_OWNER = "comfy-parity-process-diagnostics"
PERFORMANCE_OWNER = "comfy-parity-performance"
EXTENSION_UI_OWNER = "comfy-parity-frontend-extension-compatibility"


def queue_ids(*numbers: int) -> tuple[str, ...]:
    return tuple(f"COMFY-QUEUE-{number:03d}" for number in numbers)


def read_catalog_rows(name: str) -> list[dict[str, str]]:
    with (ROOT / "catalogs" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


TASK_18_QUEUE_DISPOSITIONS: dict[str, tuple[str, str]] = {}


def assign_queue_disposition(kind: str, owner: str, *numbers: int) -> None:
    for feature_id in queue_ids(*numbers):
        if feature_id in TASK_18_QUEUE_DISPOSITIONS:
            raise RuntimeError(f"duplicate Task 18 queue disposition for {feature_id}")
        TASK_18_QUEUE_DISPOSITIONS[feature_id] = (kind, owner)


assign_queue_disposition("shared_closure", NATIVE_IMAGE_OWNER, 1, 2)
assign_queue_disposition("shared_closure", NATIVE_API_OWNER, 3)
assign_queue_disposition("shared_closure", NATIVE_IMAGE_OWNER, 4)
assign_queue_disposition("later_owned", NATIVE_API_OWNER, *range(5, 25))
assign_queue_disposition("later_owned", SETTINGS_OWNER, *range(25, 33))
assign_queue_disposition("shared_closure", NATIVE_IMAGE_OWNER, 33, 36, 37, 38)
assign_queue_disposition("native", EXECUTION_UI_OWNER, 35, 39)
assign_queue_disposition("later_owned", NATIVE_MEMORY_OWNER, 34)
assign_queue_disposition("later_owned", WORKFLOW_EXPERIENCE_OWNER, 40)
assign_queue_disposition("shared_closure", NATIVE_IMAGE_OWNER, 41)
assign_queue_disposition("shared_closure", WORKFLOW_EXPERIENCE_OWNER, 42, 43)
assign_queue_disposition("later_owned", DIAGNOSTICS_OWNER, 44, 45)
assign_queue_disposition("shared_closure", NATIVE_IMAGE_OWNER, 46, 47, 48)
assign_queue_disposition("later_owned", WORKFLOW_EXPERIENCE_OWNER, 49, 50)
assign_queue_disposition("native", EXECUTION_UI_OWNER, *range(51, 58))
assign_queue_disposition("shared_closure", NATIVE_IMAGE_OWNER, 58, 59, 60)
assign_queue_disposition("foundation", NATIVE_GRAPH_OWNER, 61, 62, 63)
assign_queue_disposition("native", EXECUTION_UI_OWNER, *range(64, 69))
assign_queue_disposition("foundation", WORKFLOW_FORMATS_OWNER, 69)
assign_queue_disposition("foundation", NATIVE_GRAPH_OWNER, 70)
assign_queue_disposition("later_owned", ASSET_VIEWERS_OWNER, 71)
assign_queue_disposition("native", EXECUTION_UI_OWNER, 72, 73, *range(75, 81))
assign_queue_disposition("shared_closure", NATIVE_IMAGE_OWNER, 74)
assign_queue_disposition("later_owned", PERFORMANCE_OWNER, 81)
assign_queue_disposition("foundation", WORKFLOW_FORMATS_OWNER, 82)
assign_queue_disposition("native", EXECUTION_UI_OWNER, *range(83, 92))
assign_queue_disposition("later_owned", NATIVE_API_OWNER, 92)
assign_queue_disposition("later_owned", SETTINGS_OWNER, 93)
assign_queue_disposition("native", EXECUTION_UI_OWNER, *range(94, 99), *range(100, 107))
assign_queue_disposition("shared_closure", NATIVE_IMAGE_OWNER, 99)
assign_queue_disposition("foundation", NATIVE_GRAPH_OWNER, 107)
assign_queue_disposition("native", EXECUTION_UI_OWNER, 108, 110, 111, 112)
assign_queue_disposition("shared_closure", NATIVE_IMAGE_OWNER, 109, 113)
assign_queue_disposition("shared_closure", WORKFLOW_EXPERIENCE_OWNER, 114)
assign_queue_disposition("native", EXECUTION_UI_OWNER, 115, 116)
assign_queue_disposition("later_owned", WORKFLOW_EXPERIENCE_OWNER, 117)
assign_queue_disposition("foundation", WORKFLOW_FORMATS_OWNER, 118)
assign_queue_disposition("shared_closure", NATIVE_API_OWNER, 119)

EXPECTED_QUEUE_IDS = set(queue_ids(*range(1, 120)))
if set(TASK_18_QUEUE_DISPOSITIONS) != EXPECTED_QUEUE_IDS:
    missing = sorted(EXPECTED_QUEUE_IDS - set(TASK_18_QUEUE_DISPOSITIONS))
    extra = sorted(set(TASK_18_QUEUE_DISPOSITIONS) - EXPECTED_QUEUE_IDS)
    raise RuntimeError(f"Task 18 queue disposition ledger is incomplete: missing={missing}, extra={extra}")

def task18_execution_commands() -> dict[str, tuple[str, str, bool]]:
    source_rows = {
        row["command_id"]: row for row in read_catalog_rows("frontend-commands.csv")
    }
    dispositions = read_catalog_rows("native-command-dispositions.csv")
    selected = {
        row["command_id"]: (
            source_rows[row["command_id"]]["feature_id"],
            row["owner_task_id"],
            row["disposition"] == "executable",
        )
        for row in dispositions
        if row["placement"] == "ExecutionDock"
        and row["owner_task_id"] in {EXECUTION_UI_OWNER, NATIVE_MEMORY_OWNER}
    }
    if len(selected) != 9:
        raise RuntimeError(f"expected 9 canonical Task 18/27 commands, found {len(selected)}")
    return selected


def task18_job_run_menu_owners() -> dict[str, str]:
    job_surfaces = {"job context", "job history actions", "run-mode menu"}
    source_ids = [
        row["feature_id"]
        for row in read_catalog_rows("frontend-menus.csv")
        if row["menu_surface"] in job_surfaces
    ]
    dispositions = {
        row["feature_id"]: row
        for row in read_catalog_rows("native-menu-dispositions.csv")
    }
    selected = {
        feature_id: dispositions[feature_id]["owner_task_id"]
        for feature_id in source_ids
    }
    if len(selected) != 17:
        raise RuntimeError(f"expected 17 canonical job/run menus, found {len(selected)}")
    return selected


def task18_component_owners() -> dict[str, str]:
    source_rows = {
        row["feature_id"]: row
        for row in read_catalog_rows("frontend-component-surfaces.csv")
    }
    disposition_rows = read_catalog_rows("native-component-dispositions.csv")
    selected = {
        disposition["feature_id"]: disposition["owner_task_id"]
        for disposition in disposition_rows
        if disposition["owner_task_id"] == EXECUTION_UI_OWNER
        or source_rows[disposition["feature_id"]]["domain"] == "queue-execution-ui"
    }
    if len(selected) != 25:
        raise RuntimeError(f"expected 25 canonical execution components, found {len(selected)}")
    return selected


TASK_18_EXECUTION_COMMANDS = task18_execution_commands()
TASK_18_JOB_RUN_MENU_OWNERS = task18_job_run_menu_owners()
TASK_18_COMPONENT_OWNERS = task18_component_owners()


def rust_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_execution_disposition_registry() -> None:
    rows = [
        {
            "feature_id": feature_id,
            "disposition": kind,
            "owner_task_id": owner,
        }
        for feature_id, (kind, owner) in sorted(TASK_18_QUEUE_DISPOSITIONS.items())
    ]
    with EXECUTION_DISPOSITIONS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    disposition_literals = {
        "native": lambda _owner: "ExecutionFeatureDisposition::Native",
        "foundation": lambda owner: (
            "ExecutionFeatureDisposition::Foundation { "
            f"owner: {rust_string(owner)} }}"
        ),
        "later_owned": lambda owner: (
            "ExecutionFeatureDisposition::LaterOwned { "
            f"owner: {rust_string(owner)} }}"
        ),
        "shared_closure": lambda owner: (
            "ExecutionFeatureDisposition::SharedClosure { "
            f"later_owner: {rust_string(owner)} }}"
        ),
    }
    generated_rows: list[str] = []
    for row in rows:
        try:
            disposition = disposition_literals[row["disposition"]](row["owner_task_id"])
        except KeyError as error:
            raise RuntimeError(
                f"unknown execution disposition {row['disposition']!r}"
            ) from error
        generated_rows.extend(
            [
                "    GeneratedExecutionCatalogRow {",
                f"        feature_id: {rust_string(row['feature_id'])},",
                f"        disposition: {disposition},",
                "    },",
            ]
        )
    rust = [
        "use crate::ExecutionFeatureDisposition;",
        "",
        "#[derive(Clone, Copy, Debug, Eq, PartialEq)]",
        "pub struct GeneratedExecutionCatalogRow {",
        "    pub feature_id: &'static str,",
        "    pub disposition: ExecutionFeatureDisposition,",
        "}",
        "",
        f"pub static GENERATED_EXECUTION_CATALOG: [GeneratedExecutionCatalogRow; {len(rows)}] = [",
        *generated_rows,
        "];",
        "",
    ]
    GENERATED_EXECUTION_CATALOG.write_text("\n".join(rust), encoding="utf-8")
    subprocess.run(
        ["rustfmt", "--edition", "2024", str(GENERATED_EXECUTION_CATALOG)],
        check=True,
    )


def validate_task18_disposition_ledger() -> None:
    frontend_rows = {
        row["feature_id"]: row for row in read_catalog_rows("frontend-features.csv")
        if row["feature_id"].startswith("COMFY-QUEUE-")
    }
    if set(frontend_rows) != EXPECTED_QUEUE_IDS:
        raise RuntimeError("frontend queue catalog no longer contains exactly COMFY-QUEUE-001..119")
    command_rows = {row["command_id"]: row for row in read_catalog_rows("frontend-commands.csv")}
    command_dispositions = {
        row["command_id"]: row
        for row in read_catalog_rows("native-command-dispositions.csv")
    }
    if len(command_dispositions) != 118:
        raise RuntimeError("native command disposition ledger no longer has 118 unique rows")
    for command_id, (feature_id, owner, native) in TASK_18_EXECUTION_COMMANDS.items():
        row = command_rows.get(command_id)
        if row is None or row["feature_id"] != feature_id:
            raise RuntimeError(f"Task 18 command catalog drift for {command_id}")
        disposition = command_dispositions.get(command_id)
        if (
            disposition is None
            or disposition["placement"] != "ExecutionDock"
            or disposition["owner_task_id"] != owner
        ):
            raise RuntimeError(f"Task 18 command ownership drift for {command_id}")
        has_native_action = disposition["disposition"] == "executable"
        if has_native_action != native:
            raise RuntimeError(f"Task 18 command native-action drift for {command_id}")

    menu_rows = {row["feature_id"]: row for row in read_catalog_rows("frontend-menus.csv")}
    if not set(TASK_18_JOB_RUN_MENU_OWNERS).issubset(menu_rows):
        raise RuntimeError("Task 18 job/run menu catalog lost a normative row")
    disposition_rows = {
        row["feature_id"]: row
        for row in read_catalog_rows("native-menu-dispositions.csv")
    }
    if len(disposition_rows) != 236:
        raise RuntimeError("native menu disposition ledger no longer has 236 unique rows")
    for feature_id, expected_owner in TASK_18_JOB_RUN_MENU_OWNERS.items():
        disposition = disposition_rows.get(feature_id)
        if disposition is None or disposition["owner_task_id"] != expected_owner:
            raise RuntimeError(f"Task 18 menu registry drift for {feature_id}")

    component_rows = {
        row["feature_id"]: row
        for row in read_catalog_rows("frontend-component-surfaces.csv")
    }
    if not set(TASK_18_COMPONENT_OWNERS).issubset(component_rows):
        raise RuntimeError("Task 18 component catalog lost a normative row")
    component_dispositions = {
        row["feature_id"]: row
        for row in read_catalog_rows("native-component-dispositions.csv")
    }
    if len(component_dispositions) != 805:
        raise RuntimeError("native component disposition ledger no longer has 805 unique rows")
    for feature_id, expected_owner in TASK_18_COMPONENT_OWNERS.items():
        disposition = component_dispositions.get(feature_id)
        if disposition is None or disposition["owner_task_id"] != expected_owner:
            raise RuntimeError(f"Task 18 component ownership drift for {feature_id}")
def accessible_comfy_bootstrap_present() -> bool:
    main_source = (REPO_ROOT / "crates/zed/src/main.rs").read_text(encoding="utf-8")
    graph_source = (REPO_ROOT / "crates/sim_ui/src/graph_render.rs").read_text(encoding="utf-8")
    return (
        "Application::with_platform(platform)" in main_source
        and "Application::new_inaccessible" not in main_source
        and "ZED_EXPERIMENTAL_A11Y" not in main_source
        and ".key_context(crate::COMFY_GRAPH_KEY_CONTEXT)" in graph_source
        and ".role(Role::Application)" in graph_source
    )


ACCESSIBLE_COMFY_BOOTSTRAP = accessible_comfy_bootstrap_present()
ACCESSIBILITY_STATUS = "partial" if ACCESSIBLE_COMFY_BOOTSTRAP else "conflicting"
ACCESSIBILITY_EVIDENCE = (
    "Production uses `Application::with_platform`; the native graph has an application role, "
    "a scoped `ComfyGraph` key context, semantic entity/control labels, focus, and live announcements. "
    "VAL-GPUI-012/013 own this bootstrap; later surface tasks retain the whole-application audit."
    if ACCESSIBLE_COMFY_BOOTSTRAP
    else "`build_application` defaults to `Application::new_inaccessible` unless "
    "`ZED_EXPERIMENTAL_A11Y=1`; production parity requires an accessible default and tests."
)

NATIVE_GRAPH_PRESENT = all(
    marker
    in (REPO_ROOT / path).read_text(encoding="utf-8")
    for path, marker in (
        ("crates/sim_ui/src/sim_ui.rs", "register_serializable_item::<GraphWorkspaceItem>"),
        ("crates/sim_ui/src/sim_ui.rs", "register_project_item::<GraphWorkspaceItem>"),
        (
            "crates/sim_ui/src/graph_render.rs",
            ".key_context(crate::COMFY_GRAPH_KEY_CONTEXT)",
        ),
        ("crates/sim_runtime/src/graph.rs", "pub enum GraphCommand"),
    )
)
actions_source = (REPO_ROOT / "crates/sim_ui/src/actions.rs").read_text(encoding="utf-8")
shell_source = (REPO_ROOT / "crates/sim_ui/src/shell.rs").read_text(encoding="utf-8")
zed_menus_source = (REPO_ROOT / "crates/zed/src/zed/app_menus.rs").read_text(
    encoding="utf-8"
)
keymap_sections = json.loads(
    (REPO_ROOT / "assets/keymaps/default-sim.json").read_text(encoding="utf-8")
)
GRAPH_SHELL_PRESENT = (
    "GENERATED_COMMAND_CATALOG" in actions_source
    and "pub fn command_registry()" in actions_source
    and "GENERATED_KEYBINDING_CATALOG" in shell_source
    and "pub fn menu_registry()" in shell_source
    and "pub fn comfy_menu() -> Menu" in shell_source
    and "sim_ui::comfy_menu()" in zed_menus_source
    and len(keymap_sections) == 1
    and keymap_sections[0].get("context") == KEYMAP_CONTEXT
    and len(keymap_sections[0].get("bindings", {})) == 34
)
EXECUTION_PRESENTATION_PRESENT = all(
    marker in (REPO_ROOT / path).read_text(encoding="utf-8")
    for path, marker in (
        (
            "crates/sim_runtime/src/execution_presentation.rs",
            "pub struct ExecutionPresentationService",
        ),
        ("crates/sim_ui/src/execution_model.rs", "pub struct ExecutionUiModel"),
        ("crates/sim_ui/src/execution_panel.rs", "impl Panel for ExecutionPanel"),
        ("crates/sim_ui/src/queue_panel.rs", "pub struct QueuePanelContent"),
        ("crates/sim_ui/src/history_panel.rs", "pub struct HistoryPanelContent"),
        ("crates/sim_ui/src/output_view.rs", "pub struct OutputView"),
        ("crates/zed/src/zed.rs", "sim_ui::ExecutionPanel::load"),
    )
)

CURRENT_IMPLEMENTATION_UPDATES: dict[str, dict[str, str]] = {}
if ACCESSIBLE_COMFY_BOOTSTRAP:
    CURRENT_IMPLEMENTATION_UPDATES["ZED-ARCH-001"] = {
        "current_comfy_status": "partial",
        "constraint_or_gap": "Production GPUI accessibility is enabled without an environment gate; later route, platform, screen-reader, localization, and visual audits remain",
        "recommended_mapping": "Retain the accessible bootstrap and complete VAL-GPUI-011 through the later surface/platform owners",
    }
if NATIVE_GRAPH_PRESENT:
    CURRENT_IMPLEMENTATION_UPDATES.update(
        {
            "ZED-ARCH-004": {
                "current_comfy_status": "partial",
                "constraint_or_gap": "GraphWorkspaceItem is a registered native project/workspace item; later panels and content items remain",
                "recommended_mapping": "Extend through the declared execution, asset, workflow-experience, content, settings, and diagnostics owners",
            },
            "ZED-ARCH-005": {
                "current_comfy_status": "partial",
                "constraint_or_gap": "GraphWorkspaceItem has registered versioned serialization/restoration and tested workflow authority; later cross-service persistence audit remains",
                "recommended_mapping": "Retain lossless schema/restoration and complete VAL-DOMAIN-001/006 in the persistence audit",
            },
            "ZED-ARCH-021": {
                "current_comfy_status": "partial",
                "constraint_or_gap": "The native GPUI graph implements typed ports/widgets/links, selection, groups, reroutes, subgraphs, viewport, minimap, commands, undo/redo, focus, and serialization; library/execution/content breadth remains",
                "recommended_mapping": "Keep GraphCommand as the mutation boundary and complete later node-library, execution, content, accessibility, and performance matrices",
            },
            "ZED-ARCH-055": {
                "current_comfy_status": "partial",
                "constraint_or_gap": "Validated native foundations now exist, but exact feature-level equivalence still requires every mapped executable task and final closure artifact",
                "recommended_mapping": "Promote only the exact rows whose implementation and validation evidence pass; retain missing, conflicting, deferred, and uncertain rows until final reconciliation",
            },
        }
    )
if GRAPH_SHELL_PRESENT:
    CURRENT_IMPLEMENTATION_UPDATES.update(
        {
            "ZED-ARCH-018": {
                "current_comfy_status": "partial",
                "constraint_or_gap": "A graph-scoped Comfy keymap and typed 118-command registry are present; later-owned command results remain with their executable tasks",
                "recommended_mapping": "Preserve ComfyGraph scoping and user override precedence while later owners activate their registered commands",
            },
            "ZED-ARCH-019": {
                "current_comfy_status": "partial",
                "constraint_or_gap": "The generated canonical Comfy menu targets registered native actions and all 236 menu rows retain placement/owner dispositions; later surfaces remain",
                "recommended_mapping": "Add later menu contributions only with real enablement, visible errors, and the existing registry identities",
            },
        }
    )
if ACCESSIBLE_COMFY_BOOTSTRAP and NATIVE_GRAPH_PRESENT:
    CURRENT_IMPLEMENTATION_UPDATES["ZED-ARCH-024"] = {
        "current_comfy_status": "partial",
        "constraint_or_gap": "The native graph exposes an application semantic root, entity/control labels and states, keyboard focus, errors, and live announcements; later whole-application and platform audits remain",
        "recommended_mapping": "Keep semantic state derived from the graph model and complete VAL-GPUI-011 across later surfaces and platforms",
    }
if EXECUTION_PRESENTATION_PRESENT:
    CURRENT_IMPLEMENTATION_UPDATES.update(
        {
            "ZED-ARCH-007": {
                "current_comfy_status": "partial",
                "constraint_or_gap": "The production-registered ExecutionPanel provides profile-scoped queue, history, output, and error tabs; later node-library, assets/models, operations, logs, and diagnostics panels remain",
                "recommended_mapping": "Retain one application-owned execution model and extend only through the declared later panel owners",
            },
            "ZED-ARCH-029": {
                "current_comfy_status": "partial",
                "constraint_or_gap": "OutputView preserves ordered output identity, media kind, metadata, unavailable state, and capability-gated view/download/recover/remove actions; specialized media viewers remain later-owned",
                "recommended_mapping": "Keep typed artifact references and register specialized viewers through comfy-parity-assets-editors-viewers",
            },
            "ZED-ARCH-042": {
                "current_comfy_status": "partial",
                "constraint_or_gap": "ExecutionPresentationService and ExecutionUiModel reduce monotonic profile/attempt events, reject stale or cross-profile updates, coalesce notifications, and preserve terminal ordering",
                "recommended_mapping": "Retain the canonical reducer as the sole GPUI execution projection and complete later worker/device recovery matrices",
            },
        }
    )


UPDATES = {
    "ZED-ARCH-003": ("No Comfy workflow/native-worker lifecycle participates", "Register workflow items early and coordinate Zed-owned Rust worker cancellation/shutdown after save decisions"),
    "ZED-ARCH-007": ("No Comfy panels", "Use for node library, queue, history, assets, models, native worker logs, and operations"),
    "ZED-ARCH-008": ("No Comfy dialogs", "Use for bounded runtime-profile, legacy-mapping, permission, and destructive decisions"),
    "ZED-ARCH-010": ("No Comfy workflow/runtime-profile/attempt records", "Create a dedicated Comfy DB domain with only necessary WorkspaceDb dependency"),
    "ZED-ARCH-012": ("Silent UI success after a parity-critical write failure would violate requirements", "Await and surface critical workflow, attempt, output, and operation-journal writes"),
    "ZED-ARCH-016": ("These are not native Comfy runtime profiles", "Persist native runtime profiles with stable ProfileId, device/memory/API/plugin/provider policy, and inactive legacy connection data"),
    "ZED-ARCH-028": ("No Comfy file layout, artifact index, metadata codec, or operation journal", "Use for workflow authority, artifact/model/plugin staging, transactional outputs, and failure injection"),
    "ZED-ARCH-029": ("No Comfy output association, metadata, native codec contract, or editing", "Adapt as a native output-view primitive after runtime/artifact integration"),
    "ZED-ARCH-034": ("No native Comfy handler schemas, limits, idempotency, or authentication", "Use only as a provider client primitive; implement Comfy HTTP as a native Rust host over runtime services"),
    "ZED-ARCH-035": ("Response timeout/cancellation/limits remain relevant only for approved provider clients", "Apply per-provider operation policies; GPUI does not call a Comfy server"),
    "ZED-ARCH-036": ("Product-specific client code is not a native Comfy WebSocket host and drops some send errors", "Implement Rust WebSocket compatibility projection from the native event bus with explicit errors and bounds"),
    "ZED-ARCH-037": ("RPC transport is not Comfy and production requires no external Comfy reconnection", "Reuse reducer/epoch testing patterns for worker supervision and native API client sessions only"),
    "ZED-ARCH-038": ("No native worker readiness, private IPC, device-fence cancellation, or recovery", "Use or extend for the Zed-owned Rust compute worker only, with bounded shutdown and verified ownership"),
    "ZED-ARCH-039": ("It does not itself establish cross-platform process-tree semantics", "Do not cite it as native worker process-tree ownership; use util::process::Child where an owned worker is required"),
    "ZED-ARCH-040": ("A visible terminal is not the authoritative native worker owner", "Use for bounded sanitized logs or approved environment tooling, never Python engine ownership"),
    "ZED-ARCH-041": ("Adding an application runtime here would inflate all workspace/test construction", "Prefer an application RuntimeSupervisor/global with explicit native profile entities"),
    "ZED-ARCH-042": ("No native Comfy runtime state reducer", "Store runtime-profile/worker tasks and reject stale profile, attempt, worker, and event-sequence completions"),
    "ZED-ARCH-043": ("No backend/model/plugin/codec byte progress, journal, snapshot, or rollback", "Build a durable native component/artifact operation manager; do not reproduce Python/custom-node update internals"),
    "ZED-ARCH-044": ("Existing host is Wasmtime/WASI precedent but lacks explicit Comfy tensor/node ports and has broader editor/Node integration", "Build a dedicated bounded comfy_plugin_host with versioned WIT, explicit ports, opaque handles, grants, legacy mappings, and no Python/JavaScript execution"),
    "ZED-ARCH-045": ("External URL launch is not an owned compatibility host and production browser handoff is prohibited for extension execution", "Use only for ordinary approved documentation/navigation; preserve unsupported web-extension data as native placeholders"),
    "ZED-ARCH-046": ("No native Comfy provider/plugin secret namespace; development storage has weaker guarantees", "Store secret references only and scope grants by runtime profile/provider/plugin"),
    "ZED-ARCH-047": ("Project trust is not native API, model parser, plugin, provider, codec, or vendor-FFI trust", "Implement distinct trust records and prompts for each native boundary"),
    "ZED-ARCH-048": ("Command sandboxing is not validated for the Rust compute worker, GPU drivers, vendor FFI, or codecs", "Treat it as a pattern; build and certify platform worker/plugin/FFI boundaries without launching Python"),
    "ZED-ARCH-049": ("Host specs cannot establish the exact operation/dtype/layout/memory matrix of native compute adapters", "Native worker probes plus checked-in certified backend matrices are authoritative"),
    "ZED-ARCH-050": ("No native runtime model/output/cache/plugin/snapshot roots", "Add typed derived paths and non-ASCII/platform/path-containment tests"),
    "ZED-ARCH-052": ("No stable per-workflow native runtime-profile binding", "Persist ProfileId on each workflow and isolate workers, handles, events, models, plugins, queues, outputs, and secrets"),
    "ZED-ARCH-054": ("No verified native model/plugin/backend/codec journal, resumability, rollback, cleanup, or worker recovery", "Introduce transactional staged native operations with restart reconciliation"),
    "ZED-ARCH-055": ("No target Comfy implementation was found; planned native architecture is not support evidence", "Treat rows as missing, conflicting, deferred, or uncertain until an exact native implementation and validation exists"),
    "ZED-ARCH-056": ("Every new native crate shares a root Cargo.toml write and can create wave conflicts", "Register all proposed native crates in one foundation task; family tasks write disjoint files and a later task generates central registries"),
}


ADDITIONS = [
    {
        "architecture_id": "ZED-ARCH-057", "domain": "tensor-runtime", "source_file": "Cargo.toml",
        "symbol_or_area": "workspace dependencies", "evidence_level": "code-inferred", "availability": "active",
        "reusable_primitive": "wgpu, Metal bindings, Wasmtime, image/audio support",
        "current_comfy_status": "missing",
        "constraint_or_gap": "No direct native tensor, autograd, safetensors, tokenizer, diffusion, sampler, or model-family runtime dependency exists",
        "recommended_mapping": "Create a Zed-owned comfy_tensor facade and certify native Rust backend adapters; do not expose a third-party framework in compatibility APIs",
    },
    {
        "architecture_id": "ZED-ARCH-058", "domain": "inference-runtime", "source_file": "crates/llama_cpp",
        "symbol_or_area": "HTTP client", "evidence_level": "code-inferred", "availability": "active",
        "reusable_primitive": "remote model service client patterns",
        "current_comfy_status": "missing",
        "constraint_or_gap": "llama_cpp is an HTTP client, not an embedded/native tensor or diffusion runtime",
        "recommended_mapping": "Do not count it as native inference; implement Comfy model execution in the Rust worker",
    },
    {
        "architecture_id": "ZED-ARCH-059", "domain": "gpu-device", "source_file": "crates/gpui_wgpu",
        "symbol_or_area": "WgpuContext Device Queue", "evidence_level": "code-inferred", "availability": "active",
        "reusable_primitive": "rendering GPU context",
        "current_comfy_status": "missing",
        "constraint_or_gap": "The rendering device has no inference scheduling, memory-planning, long-kernel cancellation, or device-loss isolation contract",
        "recommended_mapping": "Initially use a separate worker-owned compute device; share only after a separately validated scheduling and loss-recovery design",
    },
    {
        "architecture_id": "ZED-ARCH-060", "domain": "wasm-plugin", "source_file": "crates/extension_host/src/wasm_host.rs",
        "symbol_or_area": "Wasmtime Component Model async epoch interruption", "evidence_level": "code-inferred", "availability": "active",
        "reusable_primitive": "Wasmtime 36, WIT versioning, Component Model, async, epoch interruption",
        "current_comfy_status": "missing",
        "constraint_or_gap": "The generic host has no typed Comfy ports/tensor handles, uses broad editor APIs, and lacks the required per-invocation quotas and legacy mapping contract",
        "recommended_mapping": "Reuse narrow patterns only in a dedicated comfy_plugin_host with explicit WIT ports, grants, opaque handles, fuel/epoch/deadline/memory/table/channel/output limits",
    },
    {
        "architecture_id": "ZED-ARCH-061", "domain": "media", "source_file": "crates/media; crates/image_viewer; crates/audio",
        "symbol_or_area": "media primitives", "evidence_level": "code-inferred", "availability": "platform-specific",
        "reusable_primitive": "image rendering and some audio/platform media primitives",
        "current_comfy_status": "missing",
        "constraint_or_gap": "No cross-platform native codec registry covers Comfy image/HDR/metadata/audio/video/3D contracts; crates/media is not a general cross-platform video stack",
        "recommended_mapping": "Build bounded versioned Rust readers/writers and reviewed native FFI where needed; no required FFmpeg command subprocess",
    },
    {
        "architecture_id": "ZED-ARCH-062", "domain": "native-release-boundary", "source_file": "crates excluding projects/comfy and crates/sim_test_support/parity",
        "symbol_or_area": "production dependency and runtime paths", "evidence_level": "code-inferred", "availability": "active",
        "reusable_primitive": "Cargo metadata, package scripts, tests",
        "current_comfy_status": "missing",
        "constraint_or_gap": "No gate currently proves future production code cannot add Python/Comfy/Node-extension/browser/external-Comfy dependencies",
        "recommended_mapping": "Add reverse-dependency, package, binary/settings/menu/CLI, network-trace, and isolated native E2E release gates",
    },
]


def write_catalog() -> None:
    with CATALOG.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    by_id = {row["architecture_id"]: row for row in rows}
    for identifier, (gap, mapping) in UPDATES.items():
        by_id[identifier]["constraint_or_gap"] = gap
        by_id[identifier]["recommended_mapping"] = mapping
    for identifier, values in CURRENT_IMPLEMENTATION_UPDATES.items():
        by_id[identifier].update(values)
    for row in ADDITIONS:
        by_id[row["architecture_id"]] = row
    ordered = [by_id[key] for key in sorted(by_id, key=lambda value: int(value.rsplit("-", 1)[1]))]
    with CATALOG.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)


def markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def task18_queue_ledger_table() -> str:
    rows = {row["feature_id"]: row for row in read_catalog_rows("frontend-features.csv")}
    lines = [
        "| Feature | Source contract | Disposition | Current owner | Closure owner | Target status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for feature_id in sorted(TASK_18_QUEUE_DISPOSITIONS):
        kind, owner = TASK_18_QUEUE_DISPOSITIONS[feature_id]
        current_owner = owner if kind == "foundation" else EXECUTION_UI_OWNER
        if kind == "later_owned":
            current_owner = "none"
        closure_owner = EXECUTION_UI_OWNER if kind == "native" else owner
        status = "deferred" if kind == "later_owned" else "partial"
        lines.append(
            f"| `{feature_id}` | {markdown_cell(rows[feature_id]['name'])} | `{kind}` | "
            f"`{current_owner}` | `{closure_owner}` | `{status}` |"
        )
    return "\n".join(lines)


def task18_command_ledger_table() -> str:
    rows = {row["command_id"]: row for row in read_catalog_rows("frontend-commands.csv")}
    lines = [
        "| Command | Feature | Owner | Native action now | Target status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for command_id, (feature_id, owner, native) in TASK_18_EXECUTION_COMMANDS.items():
        lines.append(
            f"| `{command_id}` | `{rows[command_id]['feature_id']}` | `{owner}` | "
            f"`{'yes' if native else 'no'}` | `{'partial' if native else 'deferred'}` |"
        )
    return "\n".join(lines)


def task18_menu_ledger_table() -> str:
    rows = {row["feature_id"]: row for row in read_catalog_rows("frontend-menus.csv")}
    lines = [
        "| Feature | Menu action | Owner | Target status |",
        "| --- | --- | --- | --- |",
    ]
    for feature_id, owner in TASK_18_JOB_RUN_MENU_OWNERS.items():
        status = "partial" if owner == EXECUTION_UI_OWNER else "deferred"
        lines.append(
            f"| `{feature_id}` | {markdown_cell(rows[feature_id]['label_or_action'])} | "
            f"`{owner}` | `{status}` |"
        )
    return "\n".join(lines)


def task18_component_ledger_table() -> str:
    rows = {
        row["feature_id"]: row
        for row in read_catalog_rows("frontend-component-surfaces.csv")
    }
    lines = [
        "| Feature | Source surface | Owner | Target status |",
        "| --- | --- | --- | --- |",
    ]
    for feature_id, owner in TASK_18_COMPONENT_OWNERS.items():
        status = "partial" if owner == EXECUTION_UI_OWNER else "deferred"
        lines.append(
            f"| `{feature_id}` | {markdown_cell(rows[feature_id]['name'])} | "
            f"`{owner}` | `{status}` |"
        )
    return "\n".join(lines)


TASK_18_QUEUE_LEDGER_TABLE = task18_queue_ledger_table()
TASK_18_COMMAND_LEDGER_TABLE = task18_command_ledger_table()
TASK_18_MENU_LEDGER_TABLE = task18_menu_ledger_table()
TASK_18_COMPONENT_LEDGER_TABLE = task18_component_ledger_table()


REPORT = f"""# Zed native-runtime architecture evidence

## Outcome

The target has strong generic GPUI, workspace, persistence, task, Wasmtime, rendering, media, process, settings, and testing primitives plus validated native Comfy foundations for schemas, settings, trust, tensors/CPU execution, the Rust worker, safe formats, Rust/WASM plugins, registries, execution reducers, workflow/media adapters, file services, the GPUI graph shell, and a profile-scoped execution presentation service with a production-registered Execution dock panel. These foundations are partial until their later breadth and release tasks pass; planned work is never counted as current support.

Production must be a Zed-owned Rust control plane plus a Zed-owned Rust compute worker per selected device group. ComfyUI is a development-only conformance oracle. Production may not launch, manage, bundle, connect to, or depend on ComfyUI/Python, and may not execute Python or JavaScript compatibility extensions.

## Inspected target areas

- `crates/zed/src/main.rs`, `crates/zed/src/zed.rs`, generated canonical menus, and visual-test infrastructure.
- `crates/workspace`, `crates/gpui`, `crates/ui`, `crates/zed_actions`, `assets/keymaps`, `crates/settings`, `crates/settings_content`, `crates/settings_ui`, and `crates/db`.
- `crates/git_ui/src/git_graph.rs`, GPUI interaction tests, `crates/http_client`, `crates/remote`, `crates/util/src/process.rs`, `crates/terminal`, and `crates/fs`.
- `crates/image_viewer`, `crates/audio`, `crates/media`, `crates/auto_update`, `crates/extension_host`, `crates/sandbox`, `crates/system_specs`, `crates/paths`, and credentials providers.
- Root `Cargo.toml`, `crates/llama_cpp`, and `crates/gpui_wgpu` for native compute/runtime evidence.

The machine-readable companion is `catalogs/zed-architecture.csv`.

## Current support and constraints

| Capability | Status | Evidence-backed conclusion |
| --- | --- | --- |
| Native execution, tensors, autograd, RNG | partial | The native tensor facade, deterministic CPU contract, autograd/RNG foundations, prompt compiler, DAG/cache/queue/history reducers, and private execution events are implemented; the generated operator/device/model breadth remains. |
| Native models/formats/samplers/schedulers | partial | Safe bounded model formats and descriptor registries are implemented; full family, quantization, attention, sampler, scheduler, latent, and diffusion execution remains. |
| Native worker/device/memory planner | partial | Versioned private Rust IPC, worker supervision, cancellation, output transactions, recovery, and the CPU backend are implemented; vendor devices and the full memory planner remain. |
| Workflow/graph/UI | partial | Lossless workflow/prompt/media adapters, file authority, a registered serializable GPUI graph item, typed ports/widgets/links, commands, persistence, generated scoped keymaps and native menus, and the profile-scoped Execution dock panel with queue/history/output/error projections are implemented; later panels/editors/shell breadth remains. |
| Native HTTP/WebSocket/CLI host | missing | Generic clients exist; no Rust Comfy handlers/event projection or `zed comfy` contract exists. |
| Rust/WASM plugins | partial | The dedicated versioned Rust/WIT SDK and bounded Component Model host implement explicit typed ports, handles, grants, limits, cancellation, and deterministic legacy mapping; frontend/Python legacy breadth remains. |
| Media/output compatibility | partial | Bounded shared metadata carriers, native asset namespaces, indexing, and transactional outputs are implemented; the full image/audio/video/3D codec and editor matrix remains. |
| Accessibility | {ACCESSIBILITY_STATUS} | {ACCESSIBILITY_EVIDENCE} |
| Python/JavaScript extension behavior | conflicting | Source execution contracts are intentionally replaced by Rust/WASM and lossless placeholders; they are not delegated to Python or a browser. |
| Cloud/paid providers | missing/uncertain | Existing clients are not verified Comfy provider contracts; they remain disabled without approved APIs, grants, credentials, and tests. |

## Task 18 exact execution ownership ledger

This ledger is authoritative for Task 18. `regenerate_native_zed_evidence.py` rejects any mismatch between these 119 source feature IDs and `crates/sim_ui/src/execution_catalog.rs`; it also rejects command, menu, component-catalog, and VAL-GPUI-005 component-set drift. `partial` records a concrete native or consumed foundation implementation without claiming later closure. `deferred` retains the exact later executable owner without misclassifying an intentionally later-owned row as an unaccounted gap.

### Queue feature dispositions

{TASK_18_QUEUE_LEDGER_TABLE}

### Execution command dispositions

{TASK_18_COMMAND_LEDGER_TABLE}

### Job and run menu dispositions

{TASK_18_MENU_LEDGER_TABLE}

### Execution component dispositions

{TASK_18_COMPONENT_LEDGER_TABLE}

## Recommended production architecture

```text
GPUI workspace item / dock panels / settings
                |
                v
ComfyRuntime application service
  workflow + queue/history + cache + journals + native API projection
                |
       private versioned Rust IPC
                |
Zed-owned Rust compute worker per device group
  tensor/autograd/RNG + native backends + memory planner
  ArtifactIndex/ModelStore + model families/patches
  native DAG executor + nodes + samplers/schedulers
  bounded Rust/WASM plugin host + native media/output transactions
```

GPUI never talks to the public compatibility host internally. The worker isolates GPU faults and large model-memory lifetimes but remains part of Zed. Live tensors/device pointers do not cross IPC. Public HTTP/WebSocket and headless CLI project the same native services and never forward to ComfyUI.

## Compute and model implications

Comfy source evidence uses autograd for training nodes, custom operations, and gradient-dependent samplers, so inference-only tensor support is insufficient. A Zed-owned tensor facade must define shape, broadcasting, dtype promotion/accumulation, strides/layout, view/copy, empty/scalar, NaN/infinity/rounding, device/fallback, determinism, VJP, RNG, cancellation, and structured errors. A native Rust backend ecosystem may sit behind that facade, but its types cannot become workflow/plugin compatibility APIs.

The reference CPU backend anchors semantics. CUDA, ROCm, Metal, DirectML, XPU, NPU, MLU, and CoreX adapters require actual operation/dtype/layout/memory certification. WGPU is not described as MPS or DirectML merely because it uses Metal or D3D12. The GPUI rendering device has no current inference scheduling, long-kernel cancellation, memory isolation, or device-loss contract, so initial compute devices are worker-owned and separate.

Model loading requires bounded safetensors and GGUF readers plus a restricted weights-only PyTorch archive/pickle reader that never executes reducers. Every one of the 94 family rows needs a detector, descriptor, tiny fixture, mapping, forward checkpoints, dtype/device matrix, and exact failure/cancellation/OOM cases. LoRA/LoHa/LoKr/OFT, ControlNet, VAE, CLIP, merges, quantization, and patches form ordered copy-on-write graphs included in cache identity.

## GPUI ownership

- `ComfyRuntime` is an application service/global, not an expansion of every `workspace::AppState` constructor.
- Each workflow is a serializable workspace item bound to a stable native `ProfileId`. Node library, queue/history, assets/models, operations, logs, and diagnostics are dock panels. Sustained editors are workspace items; bounded choices use modals/popovers.
- Foreground entity work remains short and fallible. Expensive parsing, indexing, hashing, model/tensor/media work runs in background tasks or the Rust worker. Stored task handles/cancellation tokens match ownership; failures reach visible entities and durable attempt/operation states.
- New persistence domains register through existing migration patterns. Critical workflow/attempt/output/journal writes are awaited and surfaced. Settings integrate through the central settings schema/page/defaults, not a private unregistered file.
- Graph rendering needs an accessible semantic companion with stable node/port/widget relationships, actions, selection, and focus. Production cannot retain the inaccessible default.

## Rust/WASM plugin boundary

Curated plugins use a versioned Rust source trait and are linked/signed components; no stable Rust dylib ABI is promised. Third-party Rust authors compile a versioned WIT Component Model. Manifests declare plugin/API versions, digest/signature/provenance, node versions, explicit port IDs/types/cardinality/default/lazy/serialization, legacy Python/JS identifiers, grants, effects, cache/determinism, and declarative UI.

WASM stores use bounded memory/table/instances/channels/output, fuel/epoch/deadlines, capability revocation, opaque invocation-scoped tensor/model/asset handles, and deterministic legacy resolution. No raw GPU pointers, Python modules, JavaScript/DOM/LiteGraph hooks, arbitrary web directories, Node host, or browser fallback executes.

## Validation consequences

- Add operator/autograd/RNG catalogs and exact CPU/backend matrices.
- Compare all 44 sampler trajectories, all 9 sigma schedules, all 33 latent formats, and every model-family checkpoint, not only final images.
- Implement every one of 565 local and 224 API-node rows natively or through a native provider, with exact per-node schema and behavior evidence.
- Run the first native slice `LoadImage -> ImageScale -> ImageInvert -> PreviewImage -> SaveImage`, including cache, cancellation, worker kill/recovery, metadata/output transaction, GPUI inspection, and no-network/no-Python/no-source-tree gates.
- Run the shape-reduced diffusion slice through checkpoint loading, CLIP, latent, KSampler, VAE, and SaveImage with intermediate checkpoints, OOM, cancellation, and recovery.
- Inspect Cargo reverse dependencies, package manifests/binaries/settings/menus/CLI, and runtime network/process traces to prove no production Comfy/Python/JS/browser/external fallback.
- Use GPUI executor timers in GPUI tests and `./script/clippy` for repository lint validation.

## Open uncertainties

- Native backend ecosystem selection remains an implementation ADR after prototype conformance and licensing/distribution measurement; the Zed tensor facade and fixtures prevent vendor lock-in.
- Device rows cannot be promoted without actual hardware/driver certification. Unavailable labs remain conditional, not guessed.
- Native codec libraries, vendor SDKs, model licenses, package size, signing/notarization, and unsafe FFI require platform/security review.
- Cloud/paid service semantics remain unverified without approved contracts and non-mutating test accounts.
- This generator does not execute runtime tests; completed task evidence and VAL-GPUI-012/013 artifacts record executable validation, while later uncompleted rows retain no passing claim.
"""


def main() -> None:
    write_execution_disposition_registry()
    validate_task18_disposition_ledger()
    write_catalog()
    (ROOT / "evidence-zed.md").write_text(REPORT, encoding="utf-8")
    print("Regenerated native Zed evidence and 62 architecture rows.")


if __name__ == "__main__":
    main()
