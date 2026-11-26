#!/usr/bin/env python3

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
OUTPUT = ROOT / "catalogs/backend-conditioning-contracts.csv"

FIELDS = [
    "contract_id",
    "kind",
    "source_path",
    "source_symbol",
    "source_ordinal",
    "source_sha256",
    "symbol_sha256",
    "native_owner",
    "implementation_task",
    "validation_surface",
    "disposition",
    "current_zed_status",
    "zed_evidence",
    "validation_artifact_sha256",
    "closure_artifact",
]

VALIDATION_IDENTIFIER = re.compile(r"VAL-[A-Z0-9-]+-\d{3}")
SHA256 = re.compile(r"[0-9a-f]{64}")
MAX_VALIDATION_ARTIFACT_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class SourcePlan:
    path: str
    kind: str
    owner: str
    task: str
    validation: str
    class_pattern: str | None = None
    method_class_pattern: str | None = None
    functions: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    assignments: tuple[str, ...] = ()
    all_top_level: bool = False


PLANS = (
    SourcePlan(
        "projects/comfy/ComfyUI/comfy/conds.py",
        "conditioning_value",
        "sim_model::conditioning",
        "comfy-parity-conditioning-value-foundation",
        "sim_model::conditioning::tests",
        class_pattern=r"^COND.*",
    ),
    SourcePlan(
        "projects/comfy/ComfyUI/comfy/samplers.py",
        "guidance",
        "sim_sampler::guidance",
        "comfy-parity-conditioning-guidance-adapter",
        "sim_sampler::guidance::tests",
        functions=("get_area_and_mult", "calc_cond_batch", "sampling_function"),
    ),
    SourcePlan(
        "projects/comfy/ComfyUI/comfy/sampler_helpers.py",
        "guidance_hook",
        "sim_sampler::guidance",
        "comfy-parity-conditioning-guidance-adapter",
        "sim_sampler::guidance::tests",
        functions=(
            "prepare_mask",
            "convert_cond",
            "get_models_from_cond",
            "get_additional_models",
            "prepare_sampling",
            "cleanup_models",
        ),
    ),
    SourcePlan(
        "projects/comfy/ComfyUI/comfy/hooks.py",
        "guidance_hook",
        "sim_sampler::guidance",
        "comfy-parity-conditioning-guidance-adapter",
        "sim_sampler::guidance::tests",
        class_pattern=r"Hook|Hooks|WeightHook|ModelOptions",
    ),
    SourcePlan(
        "projects/comfy/ComfyUI/comfy/patcher_extension.py",
        "guidance_hook",
        "sim_sampler::guidance",
        "comfy-parity-conditioning-guidance-adapter",
        "sim_sampler::guidance::tests",
        class_pattern=r"Hook|Wrapper|Wrappers|Callbacks|PatcherInjection",
    ),
    SourcePlan(
        "projects/comfy/ComfyUI/comfy/controlnet.py",
        "controlnet",
        "sim_model::controlnet",
        "comfy-parity-controlnet-chain-foundation",
        "sim_model::controlnet::tests",
        class_pattern=r"^(StrengthType|Control.*|T2IAdapter|ControlLoraOps)$",
    ),
    SourcePlan(
        "projects/comfy/ComfyUI/comfy/sd.py",
        "model_execution",
        "sim_model::clip|sim_model::vae",
        "comfy-parity-clip-execution-foundation|comfy-parity-vae-execution-foundation",
        "sim_model::clip::tests|sim_model::vae::tests",
        class_pattern=r"^(CLIP.*|TEModel|VAE)$",
        functions=(
            "load_clip",
            "detect_te_model",
            "t5xxl_detect",
            "llama_detect",
            "load_text_encoder_state_dicts",
        ),
    ),
    SourcePlan(
        "projects/comfy/ComfyUI/comfy/clip_model.py",
        "clip_architecture",
        "sim_model::clip",
        "comfy-parity-clip-execution-foundation",
        "sim_model::clip::tests",
        class_pattern=r".*",
        functions=(
            "clip_preprocess",
            "siglip2_flex_calc_resolution",
            "siglip2_preprocess",
            "siglip2_pos_embed",
        ),
    ),
    SourcePlan(
        "projects/comfy/ComfyUI/comfy/sd1_clip.py",
        "clip_architecture",
        "sim_model::clip",
        "comfy-parity-clip-execution-foundation",
        "sim_model::clip::tests",
        class_pattern=r".*",
        functions=(
            "gen_empty_tokens",
            "parse_parentheses",
            "token_weights",
            "escape_important",
            "unescape_important",
            "safe_load_embed_zip",
            "expand_directory_list",
            "bundled_embed",
            "load_embed",
        ),
    ),
    SourcePlan(
        "projects/comfy/ComfyUI/comfy/model_patcher.py",
        "patch_mapping",
        "sim_model::patches",
        "comfy-parity-patch-loading-merge-quantized-adapter",
        "sim_model::patches::tests",
        method_class_pattern=r"^ModelPatcher$",
        methods=("add_patches", "get_key_patches", "patch_weight_to_device"),
    ),
    SourcePlan(
        "projects/comfy/ComfyUI/comfy/weight_adapter/__init__.py",
        "weight_adapter_registry",
        "sim_model::weight_adapter",
        "comfy-parity-weight-adapter-runtime-bypass",
        "sim_model::weight_adapter::tests",
        assignments=("adapters", "adapter_maps"),
    ),
    SourcePlan(
        "projects/comfy/ComfyUI/comfy/weight_adapter/base.py",
        "weight_adapter_runtime",
        "sim_model::weight_adapter",
        "comfy-parity-weight-adapter-runtime-bypass",
        "sim_model::weight_adapter::tests",
        class_pattern=r"^(?:WeightAdapterBase|WeightAdapterTrainBase)$",
    ),
    SourcePlan(
        "projects/comfy/ComfyUI/comfy/weight_adapter/base.py",
        "patch_payload",
        "sim_model::patch_graph",
        "comfy-parity-patch-graph-semantic-foundation",
        "sim_model::patch_graph::tests",
        functions=(
            "weight_decompose",
            "pad_tensor_to_shape",
            "tucker_weight_from_conv",
            "tucker_weight",
            "factorization",
        ),
    ),
    SourcePlan(
        "projects/comfy/ComfyUI/comfy/weight_adapter/bypass.py",
        "weight_adapter_runtime",
        "sim_model::weight_adapter",
        "comfy-parity-weight-adapter-runtime-bypass",
        "sim_model::weight_adapter::tests",
        all_top_level=True,
    ),
    SourcePlan(
        "projects/comfy/ComfyUI/comfy/lora.py",
        "patch_mapping",
        "sim_model::patches",
        "comfy-parity-patch-loading-merge-quantized-adapter",
        "sim_model::patches::tests",
        functions=(
            "load_lora",
            "model_lora_keys_clip",
            "model_lora_keys_unet",
            "prefetch_prepared_value",
        ),
    ),
    SourcePlan(
        "projects/comfy/ComfyUI/comfy/lora.py",
        "patch_semantics",
        "sim_model::patch_graph",
        "comfy-parity-patch-graph-semantic-foundation",
        "sim_model::patch_graph::tests",
        functions=(
            "pad_tensor_to_shape",
            "calculate_shape",
            "calculate_weight",
        ),
    ),
    SourcePlan(
        "projects/comfy/ComfyUI/comfy_extras/nodes_model_merging.py",
        "patch_mapping",
        "sim_model::patches",
        "comfy-parity-patch-loading-merge-quantized-adapter",
        "sim_model::patches::tests",
        class_pattern=r"^(?:Model(?:MergeSimple|MergeBlocks|Add|Subtract)|CLIP(?:MergeSimple|Add|Subtract))$",
    ),
)

VAE_TILING_PLANS = (
    SourcePlan(
        "projects/comfy/ComfyUI/comfy/sd.py",
        "vae_tiling",
        "sim_model::vae",
        "comfy-parity-vae-multidimensional-tiling",
        "VAL-VAE-001",
        method_class_pattern=r"^VAE$",
        methods=(
            "vae_encode_crop_pixels",
            "decode_tiled_",
            "decode_tiled_1d",
            "decode_tiled_3d",
            "encode_tiled_",
            "encode_tiled_1d",
            "encode_tiled_3d",
            "decode_tiled",
            "encode_tiled",
        ),
    ),
    SourcePlan(
        "projects/comfy/ComfyUI/comfy/utils.py",
        "vae_tiling",
        "sim_model::vae",
        "comfy-parity-vae-multidimensional-tiling",
        "VAL-VAE-001",
        functions=("get_tiled_scale_steps", "tiled_scale_multidim", "tiled_scale"),
    ),
)


def task_states(
    tasks_path: Path = ROOT / "tasks.md",
) -> dict[str, tuple[bool, str, frozenset[str]]]:
    if not tasks_path.exists():
        return {}
    encoded = tasks_path.read_text(encoding="utf-8")
    states = {}
    for match in re.finditer(
        r"^- \[([ xX~])\] \d+\..*?\n(?P<body>.*?)(?=^- \[[ xX~]\] \d+\.|\Z)",
        encoded,
        re.MULTILINE | re.DOTALL,
    ):
        body = match.group("body")
        identifier = re.search(r"^\s+- _id:\s*([^\s]+)\s*$", body, re.MULTILINE)
        if identifier is None:
            continue
        evidence_matches = re.findall(
            r"^\s+- _validation_evidence:\s*(.+?)\s*$", body, re.MULTILINE
        )
        evidence = evidence_matches[0].strip() if len(evidence_matches) == 1 else ""
        if re.match(r"(?i)^STALE(?:\b|_)", evidence):
            evidence = ""
        writes_match = re.search(
            r"^\s+- Writes:\s*(.+?)\s*$", body, re.MULTILINE
        )
        writes = frozenset(
            value.strip()
            for value in writes_match.group(1).split(",")
            if value.strip()
        ) if writes_match is not None else frozenset()
        states[identifier.group(1)] = (
            match.group(1).lower() == "x",
            evidence,
            writes,
        )
    return states


def has_symlink_component(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def safe_workspace_file(workspace: Path, value: object) -> Path | None:
    if not nonempty_string(value):
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        return None
    path = workspace / relative
    if has_symlink_component(path, workspace) or not path.is_file():
        return None
    return path


def artifact_covers_row(
    payload: object,
    row: dict[str, str],
    closure_artifact: str,
    workspace: Path,
    declared_writes_by_task: dict[str, frozenset[str]],
) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("schema_version") != 1:
        return False
    if payload.get("validation_id") != closure_artifact:
        return False
    if payload.get("overall_status") not in {"partial", "passed"}:
        return False

    environment = payload.get("environment")
    if not isinstance(environment, dict) or not all(
        nonempty_string(environment.get(field))
        for field in ("os", "arch", "backend", "device", "dtype")
    ):
        return False

    implementation = payload.get("implementation")
    if not isinstance(implementation, dict):
        return False
    producer_path = implementation.get("path")
    if (
        closure_artifact == "VAL-CLIP-001"
        and producer_path != "crates/sim_model/tests/support/clip_receipt.rs"
    ):
        return False
    implementation_path = safe_workspace_file(workspace, producer_path)
    producer_sha256 = implementation.get("sha256")
    if (
        implementation_path is None
        or not isinstance(producer_sha256, str)
        or SHA256.fullmatch(producer_sha256) is None
        or digest(implementation_path.read_bytes()) != producer_sha256
    ):
        return False

    task_results = payload.get("task_results")
    if not isinstance(task_results, dict) or not task_results:
        return False
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    implementation_paths_by_task: dict[str, frozenset[str]] = {}
    for task_id, result in task_results.items():
        if not nonempty_string(task_id) or not isinstance(result, dict):
            return False
        passed = result.get("passed")
        failed = result.get("failed")
        skipped = result.get("skipped")
        if not all(nonnegative_integer(value) for value in (passed, failed, skipped)):
            return False
        if result.get("status") != "passed" or passed == 0 or failed != 0 or skipped != 0:
            return False
        if closure_artifact == "VAL-CLIP-001":
            validation_source = result.get("validation_source")
            if not isinstance(validation_source, dict):
                return False
            test_source = safe_workspace_file(workspace, validation_source.get("path"))
            test_sha256 = validation_source.get("sha256")
            if (
                test_source is None
                or not isinstance(test_sha256, str)
                or SHA256.fullmatch(test_sha256) is None
                or digest(test_source.read_bytes()) != test_sha256
            ):
                return False
        task_case_ids = result.get("case_ids", [])
        if (
            not isinstance(task_case_ids, list)
            or any(not nonempty_string(case_id) for case_id in task_case_ids)
            or len(task_case_ids) != len(set(task_case_ids))
            or not TASK_REQUIRED_CASES.get(task_id, frozenset()).issubset(task_case_ids)
        ):
            return False
        task_implementations = result.get("implementations")
        if task_implementations is None:
            task_implementation = result.get("implementation")
            if not isinstance(task_implementation, dict):
                return False
            task_implementations = [task_implementation]
        if not isinstance(task_implementations, list) or not task_implementations:
            return False
        task_implementation_paths = set()
        for task_implementation in task_implementations:
            if not isinstance(task_implementation, dict):
                return False
            relative_path = task_implementation.get("path")
            task_implementation_path = safe_workspace_file(workspace, relative_path)
            task_implementation_sha256 = task_implementation.get("sha256")
            if (
                not isinstance(relative_path, str)
                or relative_path in task_implementation_paths
                or task_implementation_path is None
                or not isinstance(task_implementation_sha256, str)
                or SHA256.fullmatch(task_implementation_sha256) is None
                or digest(task_implementation_path.read_bytes())
                != task_implementation_sha256
            ):
                return False
            task_implementation_paths.add(relative_path)
        required_implementations = TASK_IMPLEMENTATION_CLOSURES.get(task_id, frozenset())
        if not required_implementations.issubset(task_implementation_paths):
            return False
        declared_writes = declared_writes_by_task.get(task_id, frozenset())
        if not task_implementation_paths.issubset(declared_writes):
            return False
        implementation_paths_by_task[task_id] = frozenset(task_implementation_paths)
        total_passed += passed
        total_failed += failed
        total_skipped += skipped

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return False
    if not all(
        nonnegative_integer(summary.get(field))
        for field in ("passed", "failed", "skipped")
    ):
        return False
    if (
        summary.get("passed") != total_passed
        or summary.get("failed") != total_failed
        or summary.get("skipped") != total_skipped
        or total_passed == 0
        or total_failed != 0
        or total_skipped != 0
    ):
        return False

    task_result = task_results.get(row["implementation_task"])
    if not isinstance(task_result, dict) or task_result.get("status") != "passed":
        return False
    if not implementation_paths_by_task.get(row["implementation_task"]):
        return False

    contracts = payload.get("contracts")
    if not isinstance(contracts, list) or not contracts:
        return False
    seen_contracts = set()
    seen_case_ids = set()
    matching_contract = None
    for contract in contracts:
        if not isinstance(contract, dict):
            return False
        contract_id = contract.get("contract_id")
        if not nonempty_string(contract_id) or contract_id in seen_contracts:
            return False
        seen_contracts.add(contract_id)
        case_ids = contract.get("case_ids")
        if (
            not isinstance(case_ids, list)
            or not case_ids
            or any(not nonempty_string(case_id) for case_id in case_ids)
            or len(case_ids) != len(set(case_ids))
        ):
            return False
        contract_case_ids = set(case_ids)
        if not seen_case_ids.isdisjoint(contract_case_ids):
            return False
        seen_case_ids.update(contract_case_ids)
        if contract_id == row["contract_id"]:
            matching_contract = contract
    if matching_contract is None:
        return False
    if closure_artifact == CONDITIONING_CLOSURE_ARTIFACT:
        expected_case_id = CONTRACT_CASES.get(row["contract_id"])
        if expected_case_id is None or matching_contract.get("case_ids") != [expected_case_id]:
            return False
    return (
        matching_contract.get("task_id") == row["implementation_task"]
        and matching_contract.get("source_sha256") == row["source_sha256"]
        and matching_contract.get("symbol_sha256") == row["symbol_sha256"]
        and matching_contract.get("status") == "passed"
    )


def validation_artifact(
    closure_artifact: str,
    row: dict[str, str],
    workspace: Path = WORKSPACE,
    declared_writes_by_task: dict[str, frozenset[str]] | None = None,
) -> tuple[str, str]:
    if VALIDATION_IDENTIFIER.fullmatch(closure_artifact) is None:
        return "", ""
    path = workspace / "target/comfy-parity" / f"{closure_artifact.lower()}.json"
    if (
        not path.is_file()
        or has_symlink_component(path, workspace)
        or path.stat().st_size > MAX_VALIDATION_ARTIFACT_BYTES
    ):
        return "", ""
    encoded = path.read_bytes()
    try:
        payload = json.loads(encoded.decode("utf-8"), object_pairs_hook=unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return "", ""
    if not artifact_covers_row(
        payload,
        row,
        closure_artifact,
        workspace,
        declared_writes_by_task or {},
    ):
        return "", ""
    return path.relative_to(workspace).as_posix(), digest(encoded)


def closure_fields(
    *,
    disposition: str,
    task_complete: bool,
    task_evidence: str,
    closure_artifact: str,
    artifact_path: str,
    artifact_digest: str,
) -> dict[str, str]:
    promoted = (
        disposition in {"native_rust", "native_fail_closed"}
        and task_complete
        and bool(task_evidence)
        and bool(closure_artifact)
        and bool(artifact_path)
        and re.fullmatch(r"[0-9a-f]{64}", artifact_digest) is not None
    )
    if promoted:
        return {
            "current_zed_status": "equivalent",
            "zed_evidence": (
                f"Owning task is complete with durable validation evidence for {disposition}; "
                f"{closure_artifact} artifact {artifact_path} has exact task and contract "
                f"coverage with SHA-256 {artifact_digest}."
            ),
            "validation_artifact_sha256": artifact_digest,
        }
    return {
        "current_zed_status": "missing",
        "zed_evidence": (
            "Executable closure remains pending; a checked task box without durable evidence and "
            "a schema-valid artifact covering this exact task and contract cannot promote this row."
        ),
        "validation_artifact_sha256": "",
    }


def validate_closure_rules() -> None:
    base = {
        "disposition": "native_rust",
        "task_complete": True,
        "task_evidence": "passed",
        "closure_artifact": "VAL-VAE-001",
        "artifact_path": "target/comfy-parity/val-vae-001.json",
        "artifact_digest": "a" * 64,
    }
    for changed in (
        {"task_complete": False},
        {"task_evidence": ""},
        {"artifact_digest": ""},
        {"closure_artifact": ""},
        {"disposition": "unsupported"},
    ):
        values = dict(base)
        values.update(changed)
        if closure_fields(**values)["current_zed_status"] != "missing":
            raise RuntimeError(f"conditioning closure self-test promoted {changed}")
    if closure_fields(**base)["current_zed_status"] != "equivalent":
        raise RuntimeError("conditioning closure self-test rejected valid completed row")
    fail_closed = dict(base)
    fail_closed["disposition"] = "native_fail_closed"
    if closure_fields(**fail_closed)["current_zed_status"] != "equivalent":
        raise RuntimeError("conditioning closure self-test rejected validated fail-closed row")


def validate_contract_cases(rows: list[dict[str, str]]) -> None:
    expected_task_counts = {
        CONDITIONING_TASK: 5,
        GUIDANCE_TASK: 12,
    }
    contract_rows = [
        row
        for row in rows
        if row["implementation_task"] in expected_task_counts
    ]
    observed_contract_ids = {row["contract_id"] for row in contract_rows}
    expected_contract_ids = set(CONTRACT_CASES)
    if observed_contract_ids != expected_contract_ids:
        raise RuntimeError(
            "VAL-CONDITIONING-001 contract case mapping differs from generated "
            "source rows: "
            f"missing={sorted(expected_contract_ids - observed_contract_ids)}, "
            f"unexpected={sorted(observed_contract_ids - expected_contract_ids)}"
        )
    observed_task_counts = {
        task_id: sum(
            row["implementation_task"] == task_id for row in contract_rows
        )
        for task_id in expected_task_counts
    }
    if observed_task_counts != expected_task_counts:
        raise RuntimeError(
            "VAL-CONDITIONING-001 generated task partition mismatch: "
            f"expected={expected_task_counts}, observed={observed_task_counts}"
        )
    if len(CONTRACT_CASES) != 17:
        raise RuntimeError(
            "VAL-CONDITIONING-001 contract case mapping must contain exactly 17 entries"
        )
    if len(set(CONTRACT_CASES.values())) != len(CONTRACT_CASES):
        raise RuntimeError(
            "VAL-CONDITIONING-001 contract case mapping contains duplicate case IDs"
        )
    for contract_id, case_id in CONTRACT_CASES.items():
        prefix = f"{contract_id}:"
        if not case_id.startswith(prefix) or not case_id.removeprefix(prefix):
            raise RuntimeError(
                f"VAL-CONDITIONING-001 case ID {case_id!r} is not bound to "
                f"contract {contract_id!r}"
            )


WEIGHT_ADAPTER_OWNER = "sim_model::weight_adapter"
WEIGHT_ADAPTER_TASK = "comfy-parity-weight-adapter-runtime-bypass"
WEIGHT_ADAPTER_VALIDATION = "sim_model::weight_adapter::tests"
WEIGHT_ADAPTER_CLOSURE_ARTIFACT = "VAL-WEIGHT-ADAPTER-001"
PATCH_ADAPTER_OWNER = "sim_model::patches"
PATCH_ADAPTER_TASK = "comfy-parity-patch-loading-merge-quantized-adapter"
PATCH_ADAPTER_VALIDATION = "sim_model::patches::tests"
PATCH_ADAPTER_CLOSURE_ARTIFACT = "VAL-PATCH-ADAPTER-001"
PATCH_GRAPH_OWNER = "sim_model::patch_graph"
PATCH_GRAPH_TASK = "comfy-parity-patch-graph-semantic-foundation"
PATCH_GRAPH_VALIDATION = "sim_model::patch_graph::tests"
PATCH_GRAPH_CLOSURE_ARTIFACT = "VAL-PATCH-001"
CONTROLNET_TASK = "comfy-parity-controlnet-chain-foundation"
CONTROLNET_VALIDATION = "sim_model::controlnet::tests"
CONTROLNET_CLOSURE_ARTIFACT = "VAL-CONTROLNET-001"
CONDITIONING_TASK = "comfy-parity-conditioning-value-foundation"
GUIDANCE_TASK = "comfy-parity-conditioning-guidance-adapter"
CONDITIONING_CLOSURE_ARTIFACT = "VAL-CONDITIONING-001"
CONTRACT_CASES = {
    "conditioning-conditioning-value-conds-condregular-505e5b9e": "conditioning-conditioning-value-conds-condregular-505e5b9e:regular-repeat-concat-size",
    "conditioning-conditioning-value-conds-condnoiseshape-7f11dbb1": "conditioning-conditioning-value-conds-condnoiseshape-7f11dbb1:noise-shape-region-repeat",
    "conditioning-conditioning-value-conds-condcrossattn-4d921d69": "conditioning-conditioning-value-conds-condcrossattn-4d921d69:cross-attention-lcm-concat",
    "conditioning-conditioning-value-conds-condconstant-0e559aad": "conditioning-conditioning-value-conds-condconstant-0e559aad:constant-equality-identity",
    "conditioning-conditioning-value-conds-condlist-21ce2116": "conditioning-conditioning-value-conds-condlist-21ce2116:list-itemwise-process-concat-size",
    "conditioning-guidance-samplers-get-area-and-mult-14d8dec2": "conditioning-guidance-samplers-get-area-and-mult-14d8dec2:resolved-area-mask-window-weight",
    "conditioning-guidance-samplers-calc-cond-batch-23aa4a02": "conditioning-guidance-samplers-calc-cond-batch-23aa4a02:compatible-batch-regional-accumulation",
    "conditioning-guidance-samplers-sampling-function-ef25ad1d": "conditioning-guidance-samplers-sampling-function-ef25ad1d:cfg-skip-and-hook-pipeline",
    "conditioning-guidance-hook-sampler-helpers-prepare-mask-048488c7": "conditioning-guidance-hook-sampler-helpers-prepare-mask-048488c7:mask-normalize-broadcast",
    "conditioning-guidance-hook-sampler-helpers-get-models-from-cond-1be91d68": "conditioning-guidance-hook-sampler-helpers-get-models-from-cond-1be91d68:typed-control-hook-reference-projection",
    "conditioning-guidance-hook-sampler-helpers-convert-cond-e8752d85": "conditioning-guidance-hook-sampler-helpers-convert-cond-e8752d85:typed-entry-set-conversion",
    "conditioning-guidance-hook-sampler-helpers-get-additional-models-7ba596bf": "conditioning-guidance-hook-sampler-helpers-get-additional-models-7ba596bf:prebound-additional-model-identity",
    "conditioning-guidance-hook-sampler-helpers-prepare-sampling-b141c606": "conditioning-guidance-hook-sampler-helpers-prepare-sampling-b141c606:prebound-bundle-load-and-execute",
    "conditioning-guidance-hook-sampler-helpers-cleanup-models-6f147c97": "conditioning-guidance-hook-sampler-helpers-cleanup-models-6f147c97:scope-drop-workspace-convergence",
    "conditioning-guidance-hook-hooks-hook-536ff505": "conditioning-guidance-hook-hooks-hook-536ff505:ordered-guidance-hook-phases",
    "conditioning-guidance-hook-hooks-weighthook-03327446": "conditioning-guidance-hook-hooks-weighthook-03327446:weight-hook-patchgraph-delegation",
    "conditioning-guidance-hook-patcher-extension-patcherinjection-116374da": "conditioning-guidance-hook-patcher-extension-patcherinjection-116374da:injection-hook-lifecycle-cancellation",
}
VAE_EXECUTION_TASK = "comfy-parity-vae-execution-foundation"
CLIP_EXECUTION_TASK = "comfy-parity-clip-execution-foundation"
TOKENIZER_TASK = "comfy-parity-clip-tokenizer-foundation"
TEXT_TASK = "comfy-parity-clip-text-transformer-foundation"
VISION_TASK = "comfy-parity-clip-vision-foundation"
TEXT_ENCODER_T5_TASK = "comfy-parity-clip-text-encoder-t5-foundation"
TEXT_ENCODER_DECODER_TASK = "comfy-parity-clip-text-encoder-decoder-foundation"
TEXT_ENCODER_MULTIMODAL_TASK = "comfy-parity-clip-text-encoder-multimodal-foundation"
TEXT_ENCODER_COMPOSITE_TASK = "comfy-parity-clip-text-encoder-composite-adapters"
TEXT_ENCODER_BREADTH_TASK = "comfy-parity-clip-text-encoder-breadth"
TEXT_ENCODER_SOURCE_GROUPS = {
    TEXT_ENCODER_T5_TASK: frozenset({"bert.py", "spiece_tokenizer.py", "t5.py"}),
    TEXT_ENCODER_DECODER_TASK: frozenset(
        {"gemma4.py", "gpt_oss.py", "llama.py", "qwen35.py"}
    ),
    TEXT_ENCODER_MULTIMODAL_TASK: frozenset(
        {
            "ideogram4.py",
            "jina_clip_2.py",
            "ovis.py",
            "qwen3vl.py",
            "qwen_vl.py",
            "sam3_clip.py",
        }
    ),
    TEXT_ENCODER_COMPOSITE_TASK: frozenset(
        {
            "ace.py",
            "ace15.py",
            "ace_text_cleaners.py",
            "anima.py",
            "aura_t5.py",
            "boogu.py",
            "cogvideo.py",
            "cosmos.py",
            "ernie.py",
            "flux.py",
            "genmo.py",
            "hidream.py",
            "hidream_o1.py",
            "hunyuan_image.py",
            "hunyuan_video.py",
            "hydit.py",
            "kandinsky5.py",
            "krea2.py",
            "long_clipl.py",
            "longcat_image.py",
            "lt.py",
            "lumina2.py",
            "newbie.py",
            "omnigen2.py",
            "pixart_t5.py",
            "pixeldit.py",
            "qwen_image.py",
            "sa3.py",
            "sa_t5.py",
            "sd2_clip.py",
            "sd3_clip.py",
            "wan.py",
            "z_image.py",
        }
    ),
}
TEXT_ENCODER_GROUP_OWNERS = {
    TEXT_ENCODER_T5_TASK: "sim_model::clip_text_encoder_t5",
    TEXT_ENCODER_DECODER_TASK: "sim_model::clip_text_encoder_decoder",
    TEXT_ENCODER_MULTIMODAL_TASK: "sim_model::clip_text_encoder_multimodal",
    TEXT_ENCODER_COMPOSITE_TASK: "sim_model::clip_text_encoder_composite",
}
TEXT_SYMBOLS = frozenset(
    {
        "CLIPAttention",
        "CLIPMLP",
        "CLIPLayer",
        "CLIPEncoder",
        "CLIPEmbeddings",
        "CLIPTextModel_",
        "CLIPTextModel",
        "SDClipModel",
        "SD1CheckpointClipModel",
        "SD1ClipModel",
    }
)
VISION_SYMBOLS = frozenset(
    {
        "clip_preprocess",
        "siglip2_flex_calc_resolution",
        "siglip2_preprocess",
        "siglip2_pos_embed",
        "Siglip2Embeddings",
        "CLIPVisionEmbeddings",
        "CLIPVision",
        "LlavaProjector",
        "CLIPVisionModelProjection",
    }
)
VAE_TILING_TASK = "comfy-parity-vae-multidimensional-tiling"
VAE_IMAGE_TASK = "comfy-parity-vae-image-architectures"
TASK_IMPLEMENTATION_CLOSURES = {
    CONTROLNET_TASK: frozenset(
        {
            "crates/sim_model/src/sim_model.rs",
            "crates/sim_model/src/controlnet.rs",
        }
    ),
    CONDITIONING_TASK: frozenset({"crates/sim_model/src/conditioning.rs"}),
    GUIDANCE_TASK: frozenset({"crates/sim_sampler/src/guidance.rs"}),
    VAE_EXECUTION_TASK: frozenset({"crates/sim_model/src/vae.rs"}),
    CLIP_EXECUTION_TASK: frozenset({"crates/sim_model/src/clip.rs"}),
    PATCH_ADAPTER_TASK: frozenset(
        {
            "crates/sim_model/src/patch_graph.rs",
            "crates/sim_model/src/patches.rs",
            "crates/sim_model/src/quantization.rs",
            "crates/sim_model/src/weight_adapter.rs",
            "crates/sim_model/tests/patch_adapters.rs",
        }
    ),
    WEIGHT_ADAPTER_TASK: frozenset(
        {
            "crates/sim_model/src/sim_model.rs",
            "crates/sim_model/src/weight_adapter.rs",
            "crates/sim_model/tests/weight_adapter_runtime.rs",
            "crates/sim_tensor/src/cpu_backend.rs",
            "crates/sim_tensor/src/operation.rs",
        }
    ),
    PATCH_GRAPH_TASK: frozenset(
        {
            "crates/sim_model/src/sim_model.rs",
            "crates/sim_model/src/clip.rs",
            "crates/sim_model/src/model_family.rs",
            "crates/sim_model/src/patch_graph.rs",
            "crates/sim_model/src/vae.rs",
            "crates/sim_model/tests/model_family_foundation.rs",
            "crates/sim_tensor/src/cpu_backend.rs",
            "crates/sim_tensor/src/operation.rs",
            "crates/sim_worker/src/memory_modes.rs",
            "crates/sim_worker/tests/memory_conformance.rs",
            "crates/sim_test_support/tests/patch_compute_boundary.rs",
        }
    ),
    TOKENIZER_TASK: frozenset(
        {
            "crates/sim_model/src/clip.rs",
            "crates/sim_model/src/clip_tokenizer.rs",
            "crates/sim_model/src/formats.rs",
            "crates/sim_model/src/model_store.rs",
        }
    ),
    TEXT_TASK: frozenset(
        {
            "crates/sim_model/src/clip_text.rs",
            "crates/sim_model/src/clip.rs",
            "crates/sim_model/src/sim_model.rs",
            "crates/sim_model/tests/clip_backend_admission.rs",
            "crates/sim_model/tests/clip_text.rs",
        }
    ),
    TEXT_ENCODER_T5_TASK: frozenset(
        {
            "crates/sim_model/src/clip_text_encoder_t5.rs",
            "crates/sim_model/src/sim_model.rs",
            "crates/sim_model/tests/clip_backend_admission.rs",
            "crates/sim_model/tests/clip_text_encoder_t5.rs",
        }
    ),
    TEXT_ENCODER_DECODER_TASK: frozenset(
        {
            "crates/sim_model/src/clip_text_encoder_decoder.rs",
            "crates/sim_model/src/sim_model.rs",
            "crates/sim_model/tests/clip_backend_admission.rs",
            "crates/sim_model/tests/clip_text_encoder_decoder.rs",
        }
    ),
    TEXT_ENCODER_MULTIMODAL_TASK: frozenset(
        {
            "crates/sim_model/src/clip_text_encoder_multimodal.rs",
            "crates/sim_model/src/sim_model.rs",
            "crates/sim_model/tests/clip_text_encoder_multimodal.rs",
        }
    ),
    TEXT_ENCODER_COMPOSITE_TASK: frozenset(
        {
            "crates/sim_model/src/clip_text_encoder_composite.rs",
            "crates/sim_model/src/sim_model.rs",
            "crates/sim_model/tests/clip_text_encoder_composite.rs",
        }
    ),
    VISION_TASK: frozenset(
        {
            "crates/sim_model/src/clip_vision.rs",
            "crates/sim_model/src/sim_model.rs",
            "crates/sim_model/tests/clip_vision.rs",
        }
    ),
    VAE_TILING_TASK: frozenset(
        {
            "crates/sim_model/src/vae.rs",
            "crates/sim_model/src/vae_tiling.rs",
            "crates/sim_model/tests/vae_architecture.rs",
            "crates/sim_model/tests/vae_tiling.rs",
        }
    ),
    VAE_IMAGE_TASK: frozenset(
        {
            "crates/sim_model/src/vae.rs",
            "crates/sim_model/src/vae_architecture.rs",
            "crates/sim_model/src/vae_image.rs",
            "crates/sim_model/src/native_ops.rs",
            "crates/sim_runtime/src/assets.rs",
            "crates/sim_model/tests/vae_architecture.rs",
            "crates/sim_model/tests/vae_image.rs",
            "crates/sim_test_support/src/bin/generate_vae_image_fixture.rs",
            "crates/sim_test_support/tests/ownership_consolidation.rs",
        }
    ),
}
TASK_REQUIRED_CASES = {
    CONTROLNET_TASK: frozenset(
        {
            "controlnet:all-eight-contracts",
            "controlnet:strength-and-slot-merge",
            "controlnet:hint-preprocessing-and-batching",
            "controlnet:vae-latent-and-chain-delegation",
            "controlnet:cancellation-oom-workspace-ownership",
        }
    ),
    CONDITIONING_TASK: frozenset(
        {
            "conditioning:all-contracts",
            "conditioning:values-regions-masks",
            "conditioning:cancellation-oom-workspace-ownership",
        }
    ),
    GUIDANCE_TASK: frozenset(
        {
            "guidance:all-contracts",
            "guidance:cfg-hooks-batching-regions",
            "guidance:cancellation-oom-workspace-ownership",
        }
    ),
    PATCH_ADAPTER_TASK: frozenset(
        {
            "task511:all-14-valid-invalid",
            "task511:key-discovery-load-diagnostics",
            "task511:merge-and-patch-plan-mapping",
            "task511:quantized-prefetch-cancellation",
            "task511:ownership-consolidation",
        }
    ),
    WEIGHT_ADAPTER_TASK: frozenset(
        {
            "task510:all-18-valid-invalid",
            "task510:canonical-autograd",
            "task510:caller-workspace-cancellation",
            "task510:typed-capability-rejection",
            "task510:ownership-consolidation",
        }
    ),
    TOKENIZER_TASK: frozenset(
        {
            "task348:resource-exhaustion-no-publication",
            "task348:canonical-workspace-free",
        }
    ),
    TEXT_TASK: frozenset(
        {
            "task339:source-provenance-and-ten-contracts",
            "task339:embedding-attention-activation-residual",
            "task339:causal-padding-and-pooling",
            "task339:hidden-list-all-final-capture-continuation",
            "task339:projection-and-unprojected-pooling",
            "task339:typed-target-shape-mask-layer-rejection",
            "task339:cancellation-oom-no-publication",
            "task339:ownership-consolidation",
        }
    ),
    TEXT_ENCODER_T5_TASK: frozenset(
        {
            "text-encoder-t5:source-provenance-and-exact-row-closure",
            "text-encoder-t5:sentencepiece-and-token-input-delegation",
            "text-encoder-t5:relative-attention-and-gated-feed-forward",
            "text-encoder-t5:typed-target-cancellation-oom-workspace",
        }
    ),
    TEXT_ENCODER_DECODER_TASK: frozenset(
        {
            "text-encoder-decoder:source-provenance-and-exact-row-closure",
            "text-encoder-decoder:rope-gqa-rmsnorm-and-gated-mlp",
            "text-encoder-decoder:causal-cache-and-generation-semantics",
            "text-encoder-decoder:typed-target-cancellation-oom-workspace",
        }
    ),
    TEXT_ENCODER_MULTIMODAL_TASK: frozenset(
        {
            "text-encoder-multimodal:source-provenance-and-exact-row-closure",
            "text-encoder-multimodal:canonical-text-and-vision-delegation",
            "text-encoder-multimodal:position-and-projection-semantics",
            "text-encoder-multimodal:typed-target-cancellation-oom-workspace",
        }
    ),
    TEXT_ENCODER_COMPOSITE_TASK: frozenset(
        {
            "text-encoder-composite:source-provenance-and-exact-row-closure",
            "text-encoder-composite:profile-and-wrapper-delegation",
            "text-encoder-composite:cleaner-tokenizer-and-output-semantics",
            "text-encoder-composite:typed-target-cancellation-oom-workspace",
        }
    ),
    VISION_TASK: frozenset(
        {
            "task340:source-provenance-and-nine-contracts",
            "task340:standard-and-flexible-preprocess",
            "task340:embeddings-and-position-resize",
            "task340:attention-pooling-and-projections",
            "task340:typed-target-and-shape-rejection",
            "task340:cancellation-oom-no-publication",
            "task340:ownership-consolidation",
        }
    ),
    VAE_TILING_TASK: frozenset(
        {
            "task353:source-digests-and-formulas",
            "task353:single-tile-direct-assignment",
            "task353:three-pass-feather-normalization",
            "task353:one-dimensional-reshape-and-channels",
            "task353:causal-three-dimensional-geometry",
            "task353:cancellation-oom-retry-atomicity",
            "task353:ownership-consolidation",
        }
    ),
    VAE_IMAGE_TASK: frozenset(
        {
            "task354:source-provenance-and-11-contracts",
            "task354:17-profile-manifests",
            "task354:encode-decode-equations",
            "task354:production-admission-dtypes-devices",
            "task354:cancellation-oom-retry-atomicity",
            "task354:ownership-consolidation",
        }
    ),
}
PATCH_GRAPH_SOURCE_MANIFEST = frozenset(
    {
        (
            "patch_payload",
            "projects/comfy/ComfyUI/comfy/weight_adapter/base.py",
            symbol,
        )
        for symbol in (
            "weight_decompose",
            "pad_tensor_to_shape",
            "tucker_weight_from_conv",
            "tucker_weight",
            "factorization",
        )
    }
    | {
        (
            "patch_semantics",
            "projects/comfy/ComfyUI/comfy/lora.py",
            symbol,
        )
        for symbol in (
            "pad_tensor_to_shape",
            "calculate_shape",
            "calculate_weight",
        )
    }
    | {
        (
            "patch_family_equation",
            f"projects/comfy/ComfyUI/comfy/weight_adapter/{family}.py",
            "calculate_weight",
        )
        for family in ("boft", "glora", "loha", "lokr", "lora", "oft")
    }
)
CONTROLNET_SOURCE_MANIFEST = frozenset(
    {
        (
            "controlnet",
            "projects/comfy/ComfyUI/comfy/controlnet.py",
            symbol,
        )
        for symbol in (
            "StrengthType",
            "ControlIsolation",
            "ControlBase",
            "ControlNet",
            "ControlLoraOps",
            "ControlLora",
            "ControlNetSD35",
            "T2IAdapter",
        )
    }
)
CLIP_EXECUTION_SOURCE_MANIFEST = frozenset(
    {
        (
            "model_execution",
            "projects/comfy/ComfyUI/comfy/sd.py",
            symbol,
        )
        for symbol in (
            "CLIP",
            "CLIPType",
            "load_clip",
            "TEModel",
            "detect_te_model",
            "t5xxl_detect",
            "llama_detect",
            "load_text_encoder_state_dicts",
        )
    }
)
VAE_EXECUTION_SOURCE_MANIFEST = frozenset(
    {
        (
            "model_execution",
            "projects/comfy/ComfyUI/comfy/sd.py",
            "VAE",
        )
    }
)
WEIGHT_ADAPTER_SOURCE_MANIFEST = frozenset(
    {
        (
            "weight_adapter_registry",
            "projects/comfy/ComfyUI/comfy/weight_adapter/__init__.py",
            symbol,
        )
        for symbol in ("adapters", "adapter_maps")
    }
    | {
        (
            "weight_adapter_runtime",
            "projects/comfy/ComfyUI/comfy/weight_adapter/base.py",
            symbol,
        )
        for symbol in ("WeightAdapterBase", "WeightAdapterTrainBase")
    }
    | {
        (
            "weight_adapter_runtime",
            "projects/comfy/ComfyUI/comfy/weight_adapter/bypass.py",
            symbol,
        )
        for symbol in (
            "get_module_type_info",
            "BypassForwardHook",
            "BypassInjectionManager",
            "create_bypass_injections_from_patches",
        )
    }
    | {
        (
            "weight_adapter_runtime",
            f"projects/comfy/ComfyUI/comfy/weight_adapter/{family}.py",
            symbol,
        )
        for family, symbols in (
            ("boft", ("BOFTAdapter",)),
            ("glora", ("GLoRAAdapter",)),
            ("loha", ("LohaDiff", "LoHaAdapter")),
            ("lokr", ("LokrDiff", "LoKrAdapter")),
            ("lora", ("LoraDiff", "LoRAAdapter")),
            ("oft", ("OFTDiff", "OFTAdapter")),
        )
        for symbol in symbols
    }
)
PATCH_ADAPTER_SOURCE_MANIFEST = frozenset(
    {
        ("patch_mapping", "projects/comfy/ComfyUI/comfy/model_patcher.py", symbol)
        for symbol in ("add_patches", "get_key_patches", "patch_weight_to_device")
    }
    | {
        ("patch_mapping", "projects/comfy/ComfyUI/comfy/lora.py", symbol)
        for symbol in (
            "load_lora",
            "model_lora_keys_clip",
            "model_lora_keys_unet",
            "prefetch_prepared_value",
        )
    }
    | {
        (
            "patch_mapping",
            "projects/comfy/ComfyUI/comfy_extras/nodes_model_merging.py",
            symbol,
        )
        for symbol in (
            "ModelMergeSimple",
            "ModelSubtract",
            "ModelAdd",
            "CLIPMergeSimple",
            "CLIPSubtract",
            "CLIPAdd",
            "ModelMergeBlocks",
        )
    }
)
EXPECTED_OWNERS = {
    "sim_model::clip",
    "sim_model::clip_text_encoder_composite",
    "sim_model::clip_text_encoder_decoder",
    "sim_model::clip_text_encoder_multimodal",
    "sim_model::clip_text_encoder_t5",
    "sim_model::conditioning",
    "sim_model::controlnet",
    "sim_model::patch_graph",
    "sim_model::patches",
    "sim_model::vae",
    "sim_model::weight_adapter",
    "sim_sampler::guidance",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def contract_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def closure_artifact_for(validation_surface: str, implementation_task: str) -> str:
    if implementation_task == CONTROLNET_TASK:
        return CONTROLNET_CLOSURE_ARTIFACT
    if implementation_task in {CONDITIONING_TASK, GUIDANCE_TASK}:
        return CONDITIONING_CLOSURE_ARTIFACT
    if implementation_task == VAE_EXECUTION_TASK:
        return "VAL-VAE-001"
    if implementation_task == CLIP_EXECUTION_TASK:
        return "VAL-CLIP-001"
    if implementation_task == PATCH_ADAPTER_TASK:
        return PATCH_ADAPTER_CLOSURE_ARTIFACT
    if implementation_task == WEIGHT_ADAPTER_TASK:
        return WEIGHT_ADAPTER_CLOSURE_ARTIFACT
    if implementation_task == PATCH_GRAPH_TASK:
        return PATCH_GRAPH_CLOSURE_ARTIFACT
    if VALIDATION_IDENTIFIER.fullmatch(validation_surface) is not None:
        return validation_surface
    return ""


def source_plans() -> tuple[SourcePlan, ...]:
    adapters = []
    adapter_root = WORKSPACE / "projects/comfy/ComfyUI/comfy/weight_adapter"
    for path in sorted(adapter_root.glob("*.py")):
        if path.name in {"__init__.py", "base.py", "bypass.py"}:
            continue
        adapters.append(
            SourcePlan(
                path.relative_to(WORKSPACE).as_posix(),
                "weight_adapter_runtime",
                WEIGHT_ADAPTER_OWNER,
                WEIGHT_ADAPTER_TASK,
                "sim_model::weight_adapter::tests",
                class_pattern=r".*(?:Adapter|Diff)$",
            )
        )
        adapters.append(
            SourcePlan(
                path.relative_to(WORKSPACE).as_posix(),
                "patch_family_equation",
                PATCH_GRAPH_OWNER,
                PATCH_GRAPH_TASK,
                PATCH_GRAPH_VALIDATION,
                method_class_pattern=r".*Adapter$",
                methods=("calculate_weight",),
            )
        )
    text_encoders = []
    text_encoder_root = WORKSPACE / "projects/comfy/ComfyUI/comfy/text_encoders"
    assigned_text_encoder_files: set[str] = set()
    for path in sorted(text_encoder_root.glob("*.py")):
        if path.name == "__init__.py":
            continue
        matching_tasks = [
            task_id
            for task_id, source_files in TEXT_ENCODER_SOURCE_GROUPS.items()
            if path.name in source_files
        ]
        if len(matching_tasks) != 1:
            raise RuntimeError(
                f"text encoder source {path.name} must have exactly one architecture owner; "
                f"found {matching_tasks}"
            )
        task_id = matching_tasks[0]
        assigned_text_encoder_files.add(path.name)
        text_encoders.append(
            SourcePlan(
                path.relative_to(WORKSPACE).as_posix(),
                "clip_text_encoder_architecture",
                TEXT_ENCODER_GROUP_OWNERS[task_id],
                task_id,
                "VAL-CLIP-001",
                all_top_level=True,
            )
        )
    declared_text_encoder_files = set().union(*TEXT_ENCODER_SOURCE_GROUPS.values())
    if assigned_text_encoder_files != declared_text_encoder_files:
        missing = sorted(declared_text_encoder_files - assigned_text_encoder_files)
        unexpected = sorted(assigned_text_encoder_files - declared_text_encoder_files)
        raise RuntimeError(
            "text encoder architecture partition does not match the pinned source tree: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return (*PLANS, *adapters, *text_encoders, *VAE_TILING_PLANS)


CatalogNode = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | ast.Assign | ast.AnnAssign


def symbol_name(node: CatalogNode) -> str | None:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    if len(targets) == 1 and isinstance(targets[0], ast.Name):
        return targets[0].id
    return None


def selected_nodes(plan: SourcePlan, tree: ast.Module) -> list[CatalogNode]:
    class_pattern = re.compile(plan.class_pattern) if plan.class_pattern else None
    method_class_pattern = (
        re.compile(plan.method_class_pattern) if plan.method_class_pattern else None
    )
    functions = set(plan.functions)
    methods = set(plan.methods)
    assignments = set(plan.assignments)
    selected = []
    for node in tree.body:
        if plan.all_top_level and isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and class_pattern and class_pattern.fullmatch(node.name):
            selected.append(node)
        elif (
            isinstance(node, ast.ClassDef)
            and method_class_pattern
            and method_class_pattern.fullmatch(node.name)
        ):
            selected.extend(
                child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name in methods
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in functions:
            selected.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and symbol_name(node) in assignments:
            selected.append(node)
    selected_names = {name for node in selected if (name := symbol_name(node)) is not None}
    missing_functions = functions.difference(selected_names)
    if missing_functions:
        raise RuntimeError(f"{plan.path} is missing cataloged functions: {sorted(missing_functions)}")
    missing_methods = methods.difference(selected_names)
    if missing_methods:
        raise RuntimeError(f"{plan.path} is missing cataloged methods: {sorted(missing_methods)}")
    missing_assignments = assignments.difference(selected_names)
    if missing_assignments:
        raise RuntimeError(
            f"{plan.path} is missing cataloged assignments: {sorted(missing_assignments)}"
        )
    if not selected:
        raise RuntimeError(f"{plan.path} produced no conditioning contract rows")
    return selected


def row_for(
    plan: SourcePlan,
    node: CatalogNode,
    source: bytes,
    source_digest: str,
    ordinal: int,
) -> dict[str, str]:
    lines = source.splitlines(keepends=True)
    name = symbol_name(node)
    if name is None:
        raise RuntimeError(f"{plan.path}:{node.lineno} has no catalogable symbol name")
    if node.end_lineno is None:
        raise RuntimeError(f"{plan.path}:{name} has no AST end position")
    symbol = b"".join(lines[node.lineno - 1 : node.end_lineno])
    owner = plan.owner
    implementation_task = plan.task
    validation_surface = plan.validation
    if plan.path.endswith("/sd.py"):
        if plan.kind == "vae_tiling":
            owner = "sim_model::vae"
            implementation_task = VAE_TILING_TASK
            validation_surface = "VAL-VAE-001"
        elif isinstance(node, ast.ClassDef) and name == "VAE":
            owner = "sim_model::vae"
            implementation_task = VAE_EXECUTION_TASK
            validation_surface = "sim_model::vae::tests"
        else:
            owner = "sim_model::clip"
            implementation_task = CLIP_EXECUTION_TASK
            validation_surface = "VAL-CLIP-001"
    elif plan.path.endswith("/clip_model.py"):
        if name in VISION_SYMBOLS:
            implementation_task = VISION_TASK
        elif name in TEXT_SYMBOLS:
            implementation_task = TEXT_TASK
        else:
            implementation_task = CLIP_EXECUTION_TASK
        validation_surface = "VAL-CLIP-001"
    elif plan.path.endswith("/sd1_clip.py"):
        if name in {
            "gen_empty_tokens",
            "ClipTokenWeightEncoder",
            "parse_parentheses",
            "token_weights",
            "escape_important",
            "unescape_important",
            "safe_load_embed_zip",
            "expand_directory_list",
            "bundled_embed",
            "load_embed",
            "SDTokenizer",
            "SD1Tokenizer",
        }:
            implementation_task = TOKENIZER_TASK
        elif name in TEXT_SYMBOLS:
            implementation_task = TEXT_TASK
        else:
            implementation_task = CLIP_EXECUTION_TASK
        validation_surface = "VAL-CLIP-001"
    return {
        "contract_id": (
            f"conditioning-{contract_slug(plan.kind)}-"
            f"{contract_slug(Path(plan.path).stem)}-{contract_slug(name)}-"
            f"{digest(name.encode('utf-8'))[:8]}"
        ),
        "kind": plan.kind,
        "source_path": plan.path,
        "source_symbol": name,
        "source_ordinal": str(ordinal),
        "source_sha256": source_digest,
        "symbol_sha256": digest(symbol),
        "native_owner": owner,
        "implementation_task": implementation_task,
        "validation_surface": validation_surface,
        "disposition": "native_rust",
        "closure_artifact": closure_artifact_for(
            validation_surface, implementation_task
        ),
    }


def vae_architecture_nodes(
    node: ast.ClassDef,
) -> list[ast.Assign | ast.AnnAssign]:
    constructor = next(
        (
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name == "__init__"
        ),
        None,
    )
    if constructor is None:
        raise RuntimeError("sd.py::VAE is missing __init__")
    assignments = []
    for child in ast.walk(constructor):
        if isinstance(child, ast.Assign):
            targets = child.targets
            value = child.value
        elif isinstance(child, ast.AnnAssign):
            targets = [child.target]
            value = child.value
        else:
            continue
        if value is None or not any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "first_stage_model"
            for target in targets
        ):
            continue
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Attribute)
            and isinstance(value.func.value.value, ast.Name)
            and value.func.value.value.id == "self"
            and value.func.value.attr == "first_stage_model"
            and value.func.attr == "eval"
        ):
            continue
        assignments.append(child)
    assignments.sort(key=lambda assignment: (assignment.lineno, assignment.col_offset))
    if len(assignments) != 31:
        raise RuntimeError(
            "sd.py::VAE architecture extraction expected exactly 31 pinned branches: "
            f"{len(assignments)}"
        )
    return assignments


def vae_state_dict_conversion_nodes(node: ast.ClassDef) -> list[ast.If]:
    constructor = next(
        (
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name == "__init__"
        ),
        None,
    )
    if constructor is None:
        raise RuntimeError("sd.py::VAE is missing __init__")
    conversions = [
        child
        for child in ast.walk(constructor)
        if isinstance(child, ast.If)
        and any(
            isinstance(descendant, ast.Call)
            and isinstance(descendant.func, ast.Attribute)
            and isinstance(descendant.func.value, ast.Name)
            and descendant.func.value.id == "diffusers_convert"
            and descendant.func.attr == "convert_vae_state_dict"
            for descendant in ast.walk(child)
        )
    ]
    conversions.sort(key=lambda conversion: (conversion.lineno, conversion.col_offset))
    if len(conversions) != 1:
        raise RuntimeError(
            "sd.py::VAE state-dict conversion extraction expected exactly one pinned branch: "
            f"{len(conversions)}"
        )
    return conversions


def vae_selection_nodes(node: ast.ClassDef) -> list[ast.If]:
    constructor = next(
        (
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name == "__init__"
        ),
        None,
    )
    if constructor is None:
        raise RuntimeError("sd.py::VAE is missing __init__")

    def assigns_architecture(candidate: ast.AST) -> bool:
        return any(
            isinstance(descendant, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "first_stage_model"
                for target in (
                    descendant.targets
                    if isinstance(descendant, ast.Assign)
                    else [descendant.target]
                )
            )
            for descendant in ast.walk(candidate)
        )

    branches = [
        child
        for child in ast.walk(constructor)
        if isinstance(child, ast.If)
        and child.lineno >= 503
        and child.lineno <= 917
        and assigns_architecture(child)
    ]
    branches.sort(key=lambda branch: (branch.lineno, branch.col_offset))
    if len(branches) != 30:
        raise RuntimeError(
            "sd.py::VAE selection extraction expected exactly 30 pinned branches: "
            f"{len(branches)}"
        )
    return branches


def row_for_vae_architecture(
    plan: SourcePlan,
    node: ast.Assign | ast.AnnAssign,
    source: bytes,
    source_digest: str,
    ordinal: int,
) -> dict[str, str]:
    if node.end_lineno is None:
        raise RuntimeError(f"{plan.path}:{node.lineno} has no AST end position")
    lines = source.splitlines(keepends=True)
    symbol = b"".join(lines[node.lineno - 1 : node.end_lineno])
    value = node.value
    if isinstance(value, ast.Call):
        architecture = ast.unparse(value.func)
    elif isinstance(value, ast.Constant) and value.value is None:
        architecture = "unbound"
    else:
        architecture = ast.unparse(value)
    source_symbol = f"VAE.__init__.{architecture}@L{node.lineno}"
    if node.lineno in {625, 785, 816, 858, 888}:
        implementation_task = "comfy-parity-vae-audio-architectures"
    elif node.lineno in {780, 903}:
        implementation_task = "comfy-parity-vae-structured-architectures"
    elif node.lineno in {569, 641, 663, 684, 697, 711, 725, 739, 755, 836, 841, 848}:
        implementation_task = "comfy-parity-vae-video-architectures"
    elif architecture == "unbound":
        implementation_task = "comfy-parity-vae-domain-loader-foundation"
    else:
        implementation_task = "comfy-parity-vae-image-architectures"
    return {
        "contract_id": (
            "conditioning-vae-architecture-sd-"
            f"{contract_slug(architecture)}-{digest(source_symbol.encode('utf-8'))[:8]}"
        ),
        "kind": "vae_architecture",
        "source_path": plan.path,
        "source_symbol": source_symbol,
        "source_ordinal": str(ordinal),
        "source_sha256": source_digest,
        "symbol_sha256": digest(symbol),
        "native_owner": "sim_model::vae",
        "implementation_task": implementation_task,
        "validation_surface": "VAL-VAE-001",
        "disposition": "native_rust" if architecture != "unbound" else "native_fail_closed",
        "closure_artifact": "VAL-VAE-001",
    }


def row_for_vae_state_dict_conversion(
    plan: SourcePlan,
    node: ast.If,
    source: bytes,
    source_digest: str,
    ordinal: int,
) -> dict[str, str]:
    if node.end_lineno is None:
        raise RuntimeError(f"{plan.path}:{node.lineno} has no AST end position")
    expression = ast.get_source_segment(source.decode("utf-8"), node.test)
    if expression is None:
        raise RuntimeError(f"{plan.path}:{node.lineno} has no source condition")
    lines = source.splitlines(keepends=True)
    symbol = b"".join(lines[node.lineno - 1 : node.end_lineno])
    source_symbol = f"VAE.__init__.state_dict_conversion@L{node.lineno}:{expression}"
    return {
        "contract_id": (
            "conditioning-vae-state-dict-conversion-sd-"
            f"l{node.lineno}-{digest(source_symbol.encode('utf-8'))[:8]}"
        ),
        "kind": "vae_state_dict_conversion",
        "source_path": plan.path,
        "source_symbol": source_symbol,
        "source_ordinal": str(ordinal),
        "source_sha256": source_digest,
        "symbol_sha256": digest(symbol),
        "native_owner": "sim_model::vae",
        "implementation_task": "comfy-parity-vae-domain-loader-foundation",
        "validation_surface": "VAL-VAE-001",
        "disposition": "native_rust",
        "closure_artifact": "VAL-VAE-001",
    }


def row_for_vae_selection(
    plan: SourcePlan,
    node: ast.If,
    source: bytes,
    source_digest: str,
    ordinal: int,
) -> dict[str, str]:
    expression = ast.get_source_segment(source.decode("utf-8"), node.test)
    if expression is None:
        raise RuntimeError(f"{plan.path}:{node.lineno} has no source condition")
    source_symbol = f"VAE.__init__.selection@L{node.lineno}:{expression}"
    return {
        "contract_id": (
            "conditioning-vae-selection-sd-"
            f"l{node.lineno}-{digest(source_symbol.encode('utf-8'))[:8]}"
        ),
        "kind": "vae_selection_branch",
        "source_path": plan.path,
        "source_symbol": source_symbol,
        "source_ordinal": str(ordinal),
        "source_sha256": source_digest,
        "symbol_sha256": digest(expression.encode("utf-8")),
        "native_owner": "sim_model::vae",
        "implementation_task": "comfy-parity-vae-domain-loader-foundation",
        "validation_surface": "VAL-VAE-001",
        "disposition": "native_rust",
        "closure_artifact": "VAL-VAE-001",
    }


def generate_rows() -> list[dict[str, str]]:
    rows = []
    identities = set()
    owners = set()
    ordinal = 0
    for plan in source_plans():
        path = WORKSPACE / plan.path
        source = path.read_bytes()
        tree = ast.parse(source, filename=plan.path)
        selected = selected_nodes(plan, tree)
        for node in selected:
            row = row_for(plan, node, source, digest(source), ordinal)
            identity = row["contract_id"]
            if identity in identities:
                raise RuntimeError(f"duplicate conditioning contract identity: {identity}")
            if "|" in row["native_owner"] or not row["native_owner"]:
                raise RuntimeError(f"conditioning contract has multiple or empty owners: {identity}")
            identities.add(identity)
            owners.add(row["native_owner"])
            rows.append(row)
            ordinal += 1
            if (
                plan.path.endswith("/sd.py")
                and isinstance(node, ast.ClassDef)
                and symbol_name(node) == "VAE"
            ):
                for conversion_node in vae_state_dict_conversion_nodes(node):
                    conversion_row = row_for_vae_state_dict_conversion(
                        plan,
                        conversion_node,
                        source,
                        digest(source),
                        ordinal,
                    )
                    conversion_identity = conversion_row["contract_id"]
                    if conversion_identity in identities:
                        raise RuntimeError(
                            "duplicate conditioning contract identity: "
                            f"{conversion_identity}"
                        )
                    identities.add(conversion_identity)
                    owners.add(conversion_row["native_owner"])
                    rows.append(conversion_row)
                    ordinal += 1
                for architecture_node in vae_architecture_nodes(node):
                    architecture_row = row_for_vae_architecture(
                        plan,
                        architecture_node,
                        source,
                        digest(source),
                        ordinal,
                    )
                    architecture_identity = architecture_row["contract_id"]
                    if architecture_identity in identities:
                        raise RuntimeError(
                            "duplicate conditioning contract identity: "
                            f"{architecture_identity}"
                        )
                    identities.add(architecture_identity)
                    owners.add(architecture_row["native_owner"])
                    rows.append(architecture_row)
                    ordinal += 1
                for selection_node in vae_selection_nodes(node):
                    selection_row = row_for_vae_selection(
                        plan,
                        selection_node,
                        source,
                        digest(source),
                        ordinal,
                    )
                    selection_identity = selection_row["contract_id"]
                    if selection_identity in identities:
                        raise RuntimeError(
                            "duplicate conditioning contract identity: "
                            f"{selection_identity}"
                        )
                    identities.add(selection_identity)
                    owners.add(selection_row["native_owner"])
                    rows.append(selection_row)
                    ordinal += 1
    if owners != EXPECTED_OWNERS:
        raise RuntimeError(
            f"conditioning owner closure mismatch: expected {sorted(EXPECTED_OWNERS)}, "
            f"observed {sorted(owners)}"
        )
    patch_graph_rows = [
        row for row in rows if row["implementation_task"] == PATCH_GRAPH_TASK
    ]
    observed_patch_graph_manifest = {
        (row["kind"], row["source_path"], row["source_symbol"])
        for row in patch_graph_rows
    }
    if observed_patch_graph_manifest != PATCH_GRAPH_SOURCE_MANIFEST:
        raise RuntimeError(
            "PatchGraph source manifest closure mismatch: expected "
            f"{sorted(PATCH_GRAPH_SOURCE_MANIFEST)}, observed "
            f"{sorted(observed_patch_graph_manifest)}"
        )
    if len(patch_graph_rows) != len(PATCH_GRAPH_SOURCE_MANIFEST):
        raise RuntimeError("PatchGraph source manifest contains duplicate rows")
    for row in patch_graph_rows:
        if row["native_owner"] != PATCH_GRAPH_OWNER:
            raise RuntimeError(
                f"PatchGraph row {row['contract_id']} has owner {row['native_owner']}"
            )
        if row["validation_surface"] != PATCH_GRAPH_VALIDATION:
            raise RuntimeError(
                "PatchGraph row "
                f"{row['contract_id']} has validation surface "
                f"{row['validation_surface']}"
            )
        if row["closure_artifact"] != PATCH_GRAPH_CLOSURE_ARTIFACT:
            raise RuntimeError(
                "PatchGraph row "
                f"{row['contract_id']} has closure artifact "
                f"{row['closure_artifact']}"
            )
        for field in ("source_sha256", "symbol_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", row[field]) is None:
                raise RuntimeError(
                    f"PatchGraph row {row['contract_id']} has invalid {field}"
                )
    weight_adapter_rows = [
        row for row in rows if row["implementation_task"] == WEIGHT_ADAPTER_TASK
    ]
    observed_weight_adapter_manifest = {
        (row["kind"], row["source_path"], row["source_symbol"])
        for row in weight_adapter_rows
    }
    if observed_weight_adapter_manifest != WEIGHT_ADAPTER_SOURCE_MANIFEST:
        raise RuntimeError(
            "weight-adapter source manifest closure mismatch: expected "
            f"{sorted(WEIGHT_ADAPTER_SOURCE_MANIFEST)}, observed "
            f"{sorted(observed_weight_adapter_manifest)}"
        )
    if len(weight_adapter_rows) != len(WEIGHT_ADAPTER_SOURCE_MANIFEST):
        raise RuntimeError("weight-adapter source manifest contains duplicate rows")
    for row in weight_adapter_rows:
        if row["native_owner"] != WEIGHT_ADAPTER_OWNER:
            raise RuntimeError(
                f"weight-adapter row {row['contract_id']} has owner {row['native_owner']}"
            )
        if row["validation_surface"] != WEIGHT_ADAPTER_VALIDATION:
            raise RuntimeError(
                "weight-adapter row "
                f"{row['contract_id']} has validation surface "
                f"{row['validation_surface']}"
            )
        if row["closure_artifact"] != WEIGHT_ADAPTER_CLOSURE_ARTIFACT:
            raise RuntimeError(
                "weight-adapter row "
                f"{row['contract_id']} has closure artifact "
                f"{row['closure_artifact']}"
            )
        for field in ("source_sha256", "symbol_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", row[field]) is None:
                raise RuntimeError(
                    f"weight-adapter row {row['contract_id']} has invalid {field}"
                )
    patch_adapter_rows = [
        row for row in rows if row["implementation_task"] == PATCH_ADAPTER_TASK
    ]
    observed_patch_adapter_manifest = {
        (row["kind"], row["source_path"], row["source_symbol"])
        for row in patch_adapter_rows
    }
    if observed_patch_adapter_manifest != PATCH_ADAPTER_SOURCE_MANIFEST:
        raise RuntimeError(
            "patch-adapter source manifest closure mismatch: expected "
            f"{sorted(PATCH_ADAPTER_SOURCE_MANIFEST)}, observed "
            f"{sorted(observed_patch_adapter_manifest)}"
        )
    if len(patch_adapter_rows) != len(PATCH_ADAPTER_SOURCE_MANIFEST):
        raise RuntimeError("patch-adapter source manifest contains duplicate rows")
    for row in patch_adapter_rows:
        if row["native_owner"] != PATCH_ADAPTER_OWNER:
            raise RuntimeError(
                f"patch-adapter row {row['contract_id']} has owner {row['native_owner']}"
            )
        if row["validation_surface"] != PATCH_ADAPTER_VALIDATION:
            raise RuntimeError(
                "patch-adapter row "
                f"{row['contract_id']} has validation surface "
                f"{row['validation_surface']}"
            )
        if row["closure_artifact"] != PATCH_ADAPTER_CLOSURE_ARTIFACT:
            raise RuntimeError(
                "patch-adapter row "
                f"{row['contract_id']} has closure artifact "
                f"{row['closure_artifact']}"
            )
        for field in ("source_sha256", "symbol_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", row[field]) is None:
                raise RuntimeError(
                    f"patch-adapter row {row['contract_id']} has invalid {field}"
                )
    controlnet_rows = [
        row for row in rows if row["implementation_task"] == CONTROLNET_TASK
    ]
    observed_controlnet_manifest = {
        (row["kind"], row["source_path"], row["source_symbol"])
        for row in controlnet_rows
    }
    if observed_controlnet_manifest != CONTROLNET_SOURCE_MANIFEST:
        raise RuntimeError(
            "ControlNet source manifest closure mismatch: expected "
            f"{sorted(CONTROLNET_SOURCE_MANIFEST)}, observed "
            f"{sorted(observed_controlnet_manifest)}"
        )
    if len(controlnet_rows) != len(CONTROLNET_SOURCE_MANIFEST):
        raise RuntimeError("ControlNet source manifest contains duplicate rows")
    for row in controlnet_rows:
        if row["native_owner"] != "sim_model::controlnet":
            raise RuntimeError(
                f"ControlNet row {row['contract_id']} has owner {row['native_owner']}"
            )
        if row["validation_surface"] != CONTROLNET_VALIDATION:
            raise RuntimeError(
                "ControlNet row "
                f"{row['contract_id']} has validation surface "
                f"{row['validation_surface']}"
            )
        if row["closure_artifact"] != CONTROLNET_CLOSURE_ARTIFACT:
            raise RuntimeError(
                "ControlNet row "
                f"{row['contract_id']} has closure artifact "
                f"{row['closure_artifact']}"
            )
    clip_execution_rows = [
        row for row in rows if row["implementation_task"] == CLIP_EXECUTION_TASK
    ]
    observed_clip_execution_manifest = {
        (row["kind"], row["source_path"], row["source_symbol"])
        for row in clip_execution_rows
    }
    if observed_clip_execution_manifest != CLIP_EXECUTION_SOURCE_MANIFEST:
        raise RuntimeError(
            "CLIP execution source manifest closure mismatch: expected "
            f"{sorted(CLIP_EXECUTION_SOURCE_MANIFEST)}, observed "
            f"{sorted(observed_clip_execution_manifest)}"
        )
    if len(clip_execution_rows) != len(CLIP_EXECUTION_SOURCE_MANIFEST):
        raise RuntimeError("CLIP execution source manifest contains duplicate rows")
    for row in clip_execution_rows:
        if row["native_owner"] != "sim_model::clip":
            raise RuntimeError(
                f"CLIP execution row {row['contract_id']} has owner "
                f"{row['native_owner']}"
            )
        if row["closure_artifact"] != "VAL-CLIP-001":
            raise RuntimeError(
                f"CLIP execution row {row['contract_id']} has closure artifact "
                f"{row['closure_artifact']}"
            )
    vae_execution_rows = [
        row for row in rows if row["implementation_task"] == VAE_EXECUTION_TASK
    ]
    observed_vae_execution_manifest = {
        (row["kind"], row["source_path"], row["source_symbol"])
        for row in vae_execution_rows
    }
    if observed_vae_execution_manifest != VAE_EXECUTION_SOURCE_MANIFEST:
        raise RuntimeError(
            "VAE execution source manifest closure mismatch: expected "
            f"{sorted(VAE_EXECUTION_SOURCE_MANIFEST)}, observed "
            f"{sorted(observed_vae_execution_manifest)}"
        )
    if len(vae_execution_rows) != len(VAE_EXECUTION_SOURCE_MANIFEST):
        raise RuntimeError("VAE execution source manifest contains duplicate rows")
    for row in vae_execution_rows:
        if row["native_owner"] != "sim_model::vae":
            raise RuntimeError(
                f"VAE execution row {row['contract_id']} has owner "
                f"{row['native_owner']}"
            )
        if row["closure_artifact"] != "VAL-VAE-001":
            raise RuntimeError(
                f"VAE execution row {row['contract_id']} has closure artifact "
                f"{row['closure_artifact']}"
            )
    return rows


def main() -> None:
    validate_closure_rules()
    rows = generate_rows()
    validate_contract_cases(rows)
    states = task_states()
    declared_writes_by_task = {
        task_id: state[2] for task_id, state in states.items()
    }
    for row in rows:
        task_complete, task_evidence, _declared_writes = states.get(
            row["implementation_task"], (False, "", frozenset())
        )
        closure_artifact = row["closure_artifact"]
        artifact_path, artifact_digest = validation_artifact(
            closure_artifact,
            row,
            declared_writes_by_task=declared_writes_by_task,
        )
        row.update(
            closure_fields(
                disposition=row["disposition"],
                task_complete=task_complete,
                task_evidence=task_evidence,
                closure_artifact=closure_artifact,
                artifact_path=artifact_path,
                artifact_digest=artifact_digest,
            )
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
