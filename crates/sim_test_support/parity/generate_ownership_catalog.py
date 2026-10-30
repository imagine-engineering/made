#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
POLICY = ROOT / "ownership-policy.json"
OUTPUT = ROOT / "catalogs/authoritative-ownership.csv"
TASKS = ROOT / "tasks.md"

FIELDS = [
    "concern",
    "canonical_owner",
    "owner_file",
    "owner_symbol",
    "consolidation_tasks",
    "allowed_adapters",
    "competing_symbols",
    "production_consumers",
    "requirements",
    "design",
    "validation",
    "current_status",
    "decision_reason",
    "definition_hits",
    "production_call_sites",
]

DECLARATION_PREFIX = (
    r"(?m)^[ \t]*(?:pub(?:[ \t]*\([^\r\n)]*\))?[ \t]+)?"
    r"(?:unsafe[ \t]+)?(?:struct|enum|trait|type|class|interface)[ \t]+"
)
GENERIC_DECLARATION_PATTERN = re.compile(DECLARATION_PREFIX + r"([A-Za-z_][A-Za-z0-9_]*)\b")
TASK_20_CONCERNS = {
    "asset_domain_adapter",
    "backend_capability",
    "cancellation",
    "execution_queue",
    "external_navigation_authorization",
    "permission_capability_domain",
    "plugin_signature_verification",
    "provider_request_authorization",
}
TASK_20_ID = "comfy-parity-authoritative-domain-ownership"
TASK_20_VALIDATION = "VAL-OWNERSHIP-DOMAIN-001"
TASK_39_ID = "comfy-parity-tensor-workspace-accounting-consolidation"
TASK_39_CONCERNS = {
    "native_attempt_memory_policy": {
        "task39-attempt-controller-exposes-planned-workspace-ceiling",
        "task39-worker-binds-planned-ceiling-to-backend-authorization",
        "task39-retry-test-preserves-the-planner-owned-ceiling",
    },
    "tensor_backend_allocation_and_cache": {
        "task39-backend-neutral-helper-charges-workspace-to-both-authoritative-counters",
        "task39-workspace-vector-retains-the-backend-lease",
        "task39-ownership-test-executes-the-workspace-chain",
    },
    "tensor_execution_context": {
        "task39-execution-context-binds-the-sealed-workspace-authorization",
        "task39-scratch-authorization-bounds-live-workspace-bytes",
    },
}
TASK_39_VALIDATIONS = {"VAL-MEMORY-001", "VAL-OWNERSHIP-001"}
TASK_338_ID = "comfy-parity-sampler-algorithm-family-ownership-consolidation"
TASK_338_CONCERNS = {
    "workspace_tensor_zz_sampler_ancestral_step_coefficients": {
        "task338-dpm2-ancestral-uses-standard-and-rectified-flow-owner",
        "task338-dpm-adaptive-uses-standard-owner",
        "task338-dpm-fast-uses-standard-owner",
        "task338-dpmpp-2s-uses-standard-and-rectified-flow-owner",
        "task338-dpmpp-2s-cfgpp-uses-standard-owner",
        "task338-dpmpp-sde-uses-standard-owner",
        "task338-euler-ancestral-uses-standard-and-rectified-flow-owner",
    },
    "workspace_tensor_zz_sampler_dpmpp_2m_sde_family": {
        "task338-2m-sde-gpu-selects-native-placement",
        "task338-2m-sde-heun-selects-heun-family-option",
        "task338-2m-sde-heun-gpu-composes-native-and-heun-adapters",
        "task338-2m-sde-family-keeps-output-placement-only",
    },
    "workspace_tensor_zz_sampler_dpmpp_3m_sde_family": {
        "task338-3m-sde-base-selects-cpu-seeded-transfer",
        "task338-3m-sde-gpu-selects-native-placement-and-capability",
    },
    "workspace_tensor_zz_sampler_dpmpp_sde_family": {
        "task338-dpmpp-sde-family-keeps-output-placement-only",
        "task338-dpmpp-sde-gpu-selects-native-placement-and-capability",
    },
    "workspace_tensor_zz_sampler_euler_family": {
        "task338-ddim-delegates-canonical-euler-owner",
        "task338-euler-row-delegates-canonical-owner",
        "task338-euler-ancestral-delegates-euler-helpers",
        "task338-euler-churn-validates-native-capability",
        "task338-euler-ancestral-validates-native-capability",
    },
    "workspace_tensor_zz_sampler_dpm_solver_family": {
        "task338-dpm-adaptive-delegates-dpm-solver-owner",
        "task338-dpm-fast-delegates-dpm-solver-owner",
        "task338-ownership-gate-rejects-duplicate-dpm-solver-equations",
    },
    "workspace_tensor_zz_sampler_exponential_heun_family": {
        "task338-exp-heun-deterministic-delegates-seeds-2-owner",
        "task338-exp-heun-sde-delegates-seeds-2-owner",
        "task338-seeds-2-reuses-phi-and-native-capability-owners",
    },
    "workspace_tensor_zz_sampler_family_uni_pc": {
        "task338-uni-pc-owner-retains-one-traversal-and-commit",
        "task328-uni-pc-bh2-selects-shared-bh2-variant",
        "task328-uni-pc-bh2-proves-no-family-duplication",
    },
}
TASK_338_VALIDATIONS = {
    "VAL-SAMPLER-001",
    "VAL-RNG-001",
    "VAL-SAMPLING-FOUNDATION-001",
    "VAL-OWNERSHIP-001",
}
TASK_307_ID = "comfy-parity-native-sampler-euler-ancestral-cfg-pp-comfy-model-0181"
TASK_307_CONCERNS = {
    "workspace_tensor_zz_sampler_ancestral_step_coefficients": {
        "task307-euler-cfgpp-uses-standard-ancestral-owner",
    },
    "workspace_tensor_zz_sampler_euler_cfg_pp_family": {
        "task307-euler-ancestral-cfgpp-delegates-family-owner",
    },
    "workspace_tensor_zz_sampler_euler_family": {
        "task307-euler-cfgpp-reuses-euler-callback-and-capability-owners",
    },
    "workspace_tensor_zz_sampler_scheduler_sampling_profile_and_noise_domain": {
        "task307-cfgpp-output-contract-has-one-sampler-owner",
        "task307-all-cfgpp-rows-map-the-canonical-output-contract",
    },
}
TASK_308_ID = "comfy-parity-native-sampler-euler-cfg-pp-comfy-model-0182"
TASK_308_CONCERNS = {
    "workspace_tensor_zz_sampler_euler_cfg_pp_family": {
        "task308-euler-cfgpp-selects-eta-zero-noise-zero-family",
        "task308-euler-cfgpp-adapter-proves-no-family-duplication",
    },
    "workspace_tensor_zz_sampler_scheduler_sampling_profile_and_noise_domain": {
        "task308-euler-cfgpp-output-is-canonical-alias",
    },
}
TASK_307_308_VALIDATIONS = {
    "VAL-SAMPLER-001",
    "VAL-RNG-001",
    "VAL-SAMPLING-FOUNDATION-001",
    "VAL-OWNERSHIP-001",
}
TASK_315_ID = "comfy-parity-workspace-final-ownership-audit"
TASK_315_CONCERNS = {
    "native_attempt_memory_policy": {
        "task315-worker-session-retains-the-paired-authority-through-compatibility-alias",
        "task315-native-attempt-cpu-authority-name-is-alias-only",
    },
    "tensor_execution_context": {
        "task315-authority-is-the-only-workspace-seal-issuer",
        "task315-authority-ownership-oracle-rejects-competing-issuers",
        "task315-cpu-authority-name-is-alias-only",
    },
    "tensor_backend_allocation_and_cache": {
        "task315-projection-backend-cannot-retain-or-mint-authority",
    },
}
TASK_315_VALIDATIONS = {
    "VAL-MEMORY-001",
    "VAL-CANCEL-001",
    "VAL-OWNERSHIP-001",
}
TASK_104_ID = "comfy-parity-native-device-amd-rocm-comfy-model-0014"
TASK_104_CONCERNS = {
    "backend_capability": {
        "task104-rocm-device-properties-map-every-abi-fact-into-canonical-validation",
        "task104-worker-protocol-6-schema-2-preserves-v1-decode-only-boundary",
        "task104-worker-legacy-bytes-and-predecode-version-rejection-are-pinned",
        "task104-linear-algebra-wire-and-v2-primitive-mappings-are-exhaustive",
        "task104-ownership-oracle-proves-rocm-property-and-wire-boundaries",
    },
    "tensor_backend_allocation_and_cache": {
        "task104-rocm-adapter-reuses-canonical-memory-and-workspace-owners",
        "task104-rocm-foreign-scratch-fails-before-runtime-allocation",
        "task104-ownership-oracle-proves-rocm-owner-reuse",
    },
}
TASK_104_VALIDATIONS = {"VAL-TENSOR-001", "VAL-MEMORY-001", "VAL-OWNERSHIP-001"}
TASK_101_ID = "comfy-parity-autograd-state-ownership-consolidation"
TASK_101_CONCERNS = {
    "autograd_checkpoint_execution": {
        "task101-operation-checkpoint-delegates-canonical-record",
        "task101-ownership-oracle-proves-state-owner-reuse",
    },
    "autograd_gradient_publication": {
        "task101-gradient-publication-and-zeroing-use-gradient-store",
        "task101-ownership-oracle-proves-state-owner-reuse",
    },
    "autograd_tape_and_reverse_traversal": {
        "task101-leaf-binding-uses-logical-tensor-identity",
        "task101-retained-backward-uses-canonical-tape",
        "task101-ownership-oracle-proves-state-owner-reuse",
    },
    "tensor_logical_identity": {
        "task101-tensor-owns-logical-identity",
        "task101-saved-tensor-validates-logical-identity-and-lineage",
        "task101-ownership-oracle-proves-state-owner-reuse",
    },
    "tensor_mutation_lineage": {
        "task101-tensor-write-bumps-shared-mutation-lineage",
        "task101-saved-tensor-validates-logical-identity-and-lineage",
        "task101-ownership-oracle-proves-state-owner-reuse",
    },
}
TASK_101_VALIDATIONS = {
    "VAL-AUTOGRAD-001",
    "VAL-TENSOR-001",
    "VAL-OWNERSHIP-001",
}
TASK_102_ID = "comfy-parity-quantized-autograd-adapter"
TASK_102_CONCERNS = {
    "model_quantization_contracts": {
        "task102-quantlinear-adapter-delegates-layout-and-scale-equations",
        "task102-native-module-maps-canonical-quantized-storage",
        "task102-quantized-content-identity-binds-source-and-encoding",
        "task102-materialization-retains-caller-workspace",
    },
}
TASK_102_VALIDATIONS = {
    "VAL-AUTOGRAD-001",
    "VAL-NUMERIC-FORMATS-001",
    "VAL-MODEL-FAMILY-001",
    "VAL-TENSOR-001",
    "VAL-OWNERSHIP-001",
}
TASK_511_ID = "comfy-parity-patch-loading-merge-quantized-adapter"
TASK_511_CONCERNS = {
    "model_quantization_contracts": {
        "task511-patch-adapter-delegates-quantized-materialization-and-codecs",
    },
    "model_weight_adapter_contracts": {
        "task511-patch-loader-delegates-source-family-selection-once",
        "task511-ownership-test-rejects-a-second-family-parser",
    },
    "workspace_tensor_z_h_patch_graph_domain": {
        "task511-patch-adapter-delegates-ordered-graph-lifecycle",
        "task511-single-tensor-application-filters-through-checked-graph",
    },
}
TASK_511_VALIDATIONS = {
    "VAL-PATCH-ADAPTER-001",
    "VAL-TENSOR-001",
    "VAL-CANCEL-001",
    "VAL-MEMORY-001",
    "VAL-OWNERSHIP-001",
}
TASK_512_ID = "comfy-parity-sd1-tokenizer-owner-consolidation"
TASK_512_CONCERNS = {
    "native_diffusion_language_tokenization_and_embedding_artifacts": {
        "generic-tokenizer-reuses-canonical-sd1-prompt-bounds",
        "clip-bpe-adapter-delegates-untruncated-content-and-decode",
        "sentencepiece-parser-retains-ordered-score-and-type-vocabulary",
        "sentencepiece-adapter-consumes-model-store-verified-vocabulary",
        "model-store-issues-inseparable-verified-tensor-payload",
        "model-store-issues-verified-sentencepiece-and-archive-payloads",
        "textual-inversion-consumes-only-store-scoped-payload",
        "native-diffusion-delegates-canonical-fixed-context-projection",
        "native-diffusion-runtime-binds-tokenizer-identity-to-cache-and-handles",
        "native-diffusion-preserves-canonical-tokenizer-cancellation",
    },
    "native_diffusion_model_slice": {
        "diffusion-tokenizer-adapter-calls-canonical-fixed-context-owner",
    },
}
TASK_512_VALIDATIONS = {
    "VAL-CLIP-001",
    "VAL-MODEL-FORMAT-001",
    "VAL-NATIVE-E2E-002",
    "VAL-CANCEL-001",
    "VAL-MEMORY-001",
    "VAL-OWNERSHIP-001",
}
TASK_103_ID = "comfy-parity-native-autograd-breadth"
TASK_103_CONCERNS = {
    "autograd_tape_and_reverse_traversal": {
        "task103-higher-order-context-delegates-recording-to-the-canonical-tape",
        "task103-analytical-custom-functions-compose-recorded-canonical-operations",
    },
    "autograd_checkpoint_execution": {
        "task103-breadth-checkpoints-delegate-canonical-execution",
        "task103-ownership-oracle-rejects-duplicate-autograd-foundations",
    },
    "autograd_custom_function_context": {
        "task103-custom-functions-reuse-canonical-function-context",
        "task103-ownership-oracle-rejects-duplicate-autograd-foundations",
    },
    "autograd_mode_scope": {
        "task103-tape-owns-scoped-mode-restoration",
        "task103-ownership-oracle-rejects-duplicate-autograd-foundations",
    },
    "model_quantization_contracts": {
        "task103-breadth-quant-row-maps-to-model-adapter",
    },
}
TASK_103_VALIDATIONS = {
    "VAL-AUTOGRAD-001",
    "VAL-NUMERIC-FORMATS-001",
    "VAL-MODEL-FAMILY-001",
    "VAL-TENSOR-001",
    "VAL-OWNERSHIP-001",
}
VENDOR_ABI_FOUNDATIONS = {
    "native_ffi_mlu_abi_and_package_foundation": {
        "task_id": "comfy-parity-device-foundation-cambricon-mlu-comfy-model-0017",
        "ownership_task_id": "comfy-parity-vendor-abi-wave39-ownership-consolidation",
        "mappings": {
            "task118-mlu-manifest-fixes-the-reviewed-abi",
            "task118-mlu-loader-requires-registry-certified-retained-images",
            "task118-ownership-oracle-proves-the-foundation-is-observation-only",
        },
    },
    "native_ffi_directml_abi_and_package_foundation": {
        "task_id": "comfy-parity-device-foundation-directml-comfy-model-0018",
        "ownership_task_id": "comfy-parity-vendor-abi-wave39-ownership-consolidation",
        "mappings": {
            "task121-directml-manifest-fixes-the-reviewed-abi",
            "task121-directml-loader-requires-registry-certified-retained-images",
            "task121-ownership-oracle-proves-the-foundation-is-observation-only",
        },
    },
    "native_ffi_npu_abi_and_package_foundation": {
        "task_id": "comfy-parity-device-foundation-huawei-ascend-npu-comfy-model-0019",
        "ownership_task_id": "comfy-parity-vendor-abi-wave39-ownership-consolidation",
        "mappings": {
            "task124-npu-manifest-fixes-the-reviewed-abi",
            "task124-npu-loader-requires-registry-certified-retained-images",
            "task124-ownership-oracle-proves-the-foundation-is-observation-only",
        },
    },
    "native_ffi_corex_provenance_and_structural_package_foundation": {
        "task_id": "comfy-parity-corex-provenance-blocked-structural-foundation",
        "ownership_task_id": "comfy-parity-vendor-abi-wave42-ownership-consolidation",
        "mappings": {
            "task318-corex-manifest-preserves-the-reviewed-provenance-blocker",
            "task318-corex-projection-rejects-every-certificate-before-loading",
            "task318-ownership-oracle-proves-runtime-loading-remains-prohibited",
        },
    },
    "native_ffi_xpu_abi_and_package_foundation": {
        "task_id": "comfy-parity-device-foundation-intel-xpu-comfy-model-0021",
        "ownership_task_id": "comfy-parity-vendor-abi-wave42-ownership-consolidation",
        "mappings": {
            "task130-xpu-manifest-fixes-the-reviewed-abi",
            "task130-xpu-loader-requires-registry-certified-retained-images",
            "task130-ownership-oracle-proves-the-foundation-is-observation-only",
        },
    },
    "native_ffi_cuda_abi_and_package_foundation": {
        "task_id": "comfy-parity-device-foundation-nvidia-cuda-comfy-model-0022",
        "ownership_task_id": "comfy-parity-vendor-abi-wave42-ownership-consolidation",
        "mappings": {
            "task133-cuda-manifest-fixes-the-reviewed-abi",
            "task133-cuda-loader-requires-registry-certified-retained-images",
            "task133-ownership-oracle-proves-the-foundation-is-observation-only",
        },
    },
}
VENDOR_ABI_FOUNDATION_VALIDATIONS = {
    "VAL-DEVICE-001",
    "VAL-NATIVE-BOUNDARY-001",
    "VAL-OWNERSHIP-001",
}
ALLOWED_DEFINITION_ROLES = {
    "abi_boundary",
    "adapter",
    "allowed_adapter",
    "boundary_dto",
    "boundary_input",
    "canonical",
    "canonical_configuration",
    "canonical_dispatch",
    "canonical_extension_point",
    "canonical_host_handle",
    "canonical_identity",
    "canonical_implementation",
    "canonical_interface",
    "canonical_policy",
    "canonical_registry",
    "canonical_saved_value",
    "canonical_service",
    "canonical_signed_abi_declaration",
    "canonical_state",
    "canonical_transaction",
    "canonical_transition",
    "certified_implementation",
    "checked_transition",
    "compatibility_alias",
    "checked_ui_boundary_dto",
    "configuration",
    "development_reference",
    "display_only",
    "focused_model_component",
    "generated_catalog_adapter",
    "input_adapter",
    "live_inventory_adapter",
    "lossless_presentation_adapter",
    "prohibited",
    "read_only_adapter_contract",
    "read_only_evidence",
    "related",
    "related_owner",
    "sealed_adapter",
    "sealed_authorization",
    "sealed_lifecycle_projection",
    "signed_sidecar_boundary",
    "source_adapter",
    "typed_adapter",
    "typed_ui_projection",
    "ui_projection",
    "wire_adapter",
}


def task_completion_states() -> dict[str, bool]:
    states: dict[str, bool] = {}
    pending_complete: bool | None = None
    for line in TASKS.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^- \[([ xX~-])\] [0-9]+\.", line)
        if heading is not None:
            pending_complete = heading.group(1).lower() == "x"
            continue
        task_id = re.match(r"^  - _id: ([a-z0-9-]+)$", line)
        if task_id is None:
            continue
        if pending_complete is None:
            raise RuntimeError(f"task {task_id.group(1)} has no completion heading")
        if task_id.group(1) in states:
            raise RuntimeError(f"duplicate task ID: {task_id.group(1)}")
        states[task_id.group(1)] = pending_complete
        pending_complete = None
    if not states:
        raise RuntimeError("tasks.md contains no executable task IDs")
    return states


def load_policy() -> dict[str, Any]:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise RuntimeError("ownership-policy.json must contain an object")
    if policy.get("schema_version") != 1:
        raise RuntimeError("ownership-policy.json must use schema_version 1")
    scan = policy.get("scan")
    if not isinstance(scan, dict):
        raise RuntimeError("ownership-policy.json must declare a scan object")
    for field in (
        "extensions",
        "excluded_directories",
        "excluded_files",
        "ownership_scope_prefixes",
    ):
        values = scan.get(field)
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value for value in values)
        ):
            raise RuntimeError(f"ownership-policy.json scan.{field} must be a non-empty string list")
    concerns = policy.get("concerns")
    if not isinstance(concerns, list) or not concerns:
        raise RuntimeError("ownership-policy.json must declare at least one concern")
    return policy


def repository_sources(policy: dict[str, Any]) -> list[Path]:
    extensions = set(policy["scan"]["extensions"])
    excluded_directories = set(policy["scan"]["excluded_directories"])
    excluded_files = set(policy["scan"]["excluded_files"])
    sources: list[Path] = []
    for directory, directory_names, file_names in os.walk(WORKSPACE):
        directory_names[:] = sorted(
            name for name in directory_names if name not in excluded_directories
        )
        directory_path = Path(directory)
        for file_name in sorted(file_names):
            if file_name.startswith("._"):
                continue
            path = directory_path / file_name
            if path.suffix not in extensions:
                continue
            relative = path.relative_to(WORKSPACE)
            if relative.as_posix() not in excluded_files:
                sources.append(path)
    return sorted(sources, key=lambda path: path.relative_to(WORKSPACE).as_posix())


def source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def position_line(text: str, position: int) -> int:
    return text.count("\n", 0, position) + 1


def definition_kind(path: str, symbol: str, concern: dict[str, Any]) -> str:
    for definition in concern["definitions"]:
        if definition["symbol"] != symbol:
            continue
        declared_path = definition.get("path")
        if declared_path is None or declared_path == path:
            return definition["role"]
    return "unclassified"


def repository_definition_hits(
    symbols: set[str],
    source_cache: dict[Path, str],
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for path, source in source_cache.items():
        relative = path.relative_to(WORKSPACE).as_posix()
        for match in GENERIC_DECLARATION_PATTERN.finditer(source):
            symbol = match.group(1)
            if symbol not in symbols:
                continue
            hits.append(
                {
                    "path": relative,
                    "line": position_line(source, match.start()),
                    "symbol": symbol,
                }
            )
    return sorted(
        hits,
        key=lambda hit: (hit["path"], hit["line"], hit["symbol"]),
    )


def configured_definition_hits(
    concerns: list[dict[str, Any]], source_cache: dict[Path, str]
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for concern in concerns:
        for definition in concern["definitions"]:
            pattern = definition.get("pattern")
            if pattern is None:
                continue
            declared_path = definition.get("path")
            candidates = (
                [(WORKSPACE / declared_path, source_cache.get(WORKSPACE / declared_path, ""))]
                if declared_path is not None
                else source_cache.items()
            )
            for path, source in candidates:
                if not source:
                    continue
                relative = path.relative_to(WORKSPACE).as_posix()
                for match in re.finditer(pattern, source, re.MULTILINE | re.DOTALL):
                    hits.append(
                        {
                            "path": relative,
                            "line": position_line(source, match.start()),
                            "symbol": definition["symbol"],
                        }
                    )
    unique = {
        (hit["path"], hit["line"], hit["symbol"]): hit for hit in hits
    }
    return sorted(
        unique.values(),
        key=lambda hit: (hit["path"], hit["line"], hit["symbol"]),
    )


def exact_definition_hits(
    concern: dict[str, Any],
    repository_hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    symbols = {definition["symbol"] for definition in concern["definitions"]}
    return [
        {
            **hit,
            "role": definition_kind(hit["path"], hit["symbol"], concern),
        }
        for hit in repository_hits
        if hit["symbol"] in symbols
    ]


def canonical_definition_issues(
    concern: dict[str, Any], hits: list[dict[str, Any]]
) -> list[str]:
    issues = []
    for definition in concern["definitions"]:
        if not definition["role"].startswith("canonical"):
            continue
        matching_hits = [
            hit
            for hit in hits
            if hit["symbol"] == definition["symbol"]
            and hit["path"] == definition["path"]
        ]
        if len(matching_hits) != 1:
            issues.append(
                f'{definition["path"]}::{definition["symbol"]} '
                f"expected exactly one definition, found {len(matching_hits)}"
            )
    return issues


def prohibited_definition_hits(
    concern: dict[str, Any], hits: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    prohibited = {
        (definition["path"], definition["symbol"])
        for definition in concern["definitions"]
        if definition["role"] == "prohibited"
    }
    return [hit for hit in hits if (hit["path"], hit["symbol"]) in prohibited]


def unexpected_definition_hits(
    hits: list[dict[str, Any]], ownership_scope_prefixes: list[str]
) -> list[dict[str, Any]]:
    return [
        hit
        for hit in hits
        if hit["role"] == "unclassified"
        and any(hit["path"].startswith(prefix) for prefix in ownership_scope_prefixes)
    ]


def mapping_obligations(
    concern: dict[str, Any],
    source_cache: dict[Path, str],
    task_states: dict[str, bool],
) -> tuple[list[str], list[str]]:
    missing = []
    deferred = []
    for mapping in concern.get("required_mappings", []):
        path = WORKSPACE / mapping["path"]
        source = source_cache.get(path)
        if source is None:
            source = source_text(path) if path.is_file() else ""
        if re.search(mapping["pattern"], source, re.MULTILINE | re.DOTALL) is not None:
            continue
        activation_task = mapping.get("activation_task")
        if activation_task is not None and not task_states[activation_task]:
            deferred.append(f'{mapping["name"]} until {activation_task}')
        else:
            missing.append(mapping["name"])
    return missing, deferred


def validate_task_activated_mapping_semantics() -> None:
    fixture_path = WORKSPACE / "crates/sim_mapping_fixture/src/adapter.rs"
    concern = {
        "required_mappings": [
            {
                "name": "checked-adapter",
                "path": "crates/sim_mapping_fixture/src/adapter.rs",
                "pattern": r"CanonicalGrant",
                "activation_task": "future-owner",
            }
        ]
    }
    missing, deferred = mapping_obligations(concern, {}, {"future-owner": False})
    if missing or deferred != ["checked-adapter until future-owner"]:
        raise RuntimeError("inactive task mapping was not represented as deferred")
    missing, deferred = mapping_obligations(concern, {}, {"future-owner": True})
    if missing != ["checked-adapter"] or deferred:
        raise RuntimeError("completed task did not activate its required mapping")
    missing, deferred = mapping_obligations(
        concern,
        {fixture_path: "fn consume(_: CanonicalGrant) {}"},
        {"future-owner": False},
    )
    if missing or deferred:
        raise RuntimeError("implemented task mapping was not detected before completion")

    extensionless_path = WORKSPACE / "script/clippy"
    extensionless_concern = {
        "required_mappings": [
            {
                "name": "explicit-extensionless-source",
                "path": "script/clippy",
                "pattern": r"cargo",
            }
        ]
    }
    missing, deferred = mapping_obligations(extensionless_concern, {}, {})
    if missing or deferred:
        raise RuntimeError(
            f"explicit extensionless mapping source was not inspected: {extensionless_path}"
        )


def rust_test_lines(source: str) -> set[int]:
    lines = source.splitlines()
    test_lines: set[int] = set()
    pending_test_item = False
    pending_semicolon_item = False
    test_depth: int | None = None
    brace_depth = 0
    for line_number, line in enumerate(lines, start=1):
        code = line.split("//", 1)[0]
        stripped = code.strip()
        if "cfg(test)" in stripped or "cfg(any(test" in stripped or stripped.startswith("#[test]"):
            pending_test_item = True
            test_lines.add(line_number)
        if test_depth is not None:
            test_lines.add(line_number)
        opens = code.count("{")
        closes = code.count("}")
        if pending_test_item:
            test_lines.add(line_number)
            if pending_semicolon_item:
                if ";" in code:
                    pending_test_item = False
                    pending_semicolon_item = False
            elif stripped and not stripped.startswith("#["):
                semicolon_item = re.match(
                    r"^(?:pub(?:\s*\([^)]*\))?\s+)?(?:use|type|const|static)\b",
                    stripped,
                ) is not None or re.match(
                    r"^(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+[A-Za-z_][A-Za-z0-9_]*\s*;",
                    stripped,
                ) is not None
                if semicolon_item:
                    pending_semicolon_item = True
                    if ";" in code:
                        pending_test_item = False
                        pending_semicolon_item = False
                elif opens > closes:
                    test_depth = brace_depth + 1
                    pending_test_item = False
                elif opens and opens == closes:
                    pending_test_item = False
        brace_depth += opens - closes
        if test_depth is not None and brace_depth < test_depth:
            test_depth = None
    return test_lines


def validate_rust_test_line_parser() -> None:
    fixture = """#[cfg(test)]
use crate::{
    TestOnly,
};
pub fn production_after_test_use() {}
#[test]
fn single_line_test() {}
pub fn production_after_single_line_test() {}
#[cfg(test)]
mod tests {
    fn nested_test_helper() {}
}
pub fn production_after_test_module() {}
"""
    test_lines = rust_test_lines(fixture)
    line_numbers = {
        line.strip(): index
        for index, line in enumerate(fixture.splitlines(), start=1)
        if line.strip()
    }
    for production_line in (
        "pub fn production_after_test_use() {}",
        "pub fn production_after_single_line_test() {}",
        "pub fn production_after_test_module() {}",
    ):
        if line_numbers[production_line] in test_lines:
            raise RuntimeError(
                f"Rust test-line parser misclassified production line: {production_line}"
            )
    for test_line in (
        "use crate::{",
        "TestOnly,",
        "fn single_line_test() {}",
        "fn nested_test_helper() {}",
    ):
        if line_numbers[test_line] not in test_lines:
            raise RuntimeError(f"Rust test-line parser omitted test line: {test_line}")


def validate_configured_pattern_scanners() -> None:
    path = WORKSPACE / "crates/sim_pattern_fixture/src/fixture.rs"
    source = """fn canonical_commit() {}
fn use_commit() { canonical_commit(); }
#[cfg(test)]
mod tests {
    fn test_only() { canonical_commit(); }
}
"""
    concern = {
        "definitions": [
            {
                "path": "crates/sim_pattern_fixture/src/fixture.rs",
                "symbol": "canonical_commit",
                "pattern": r"^fn canonical_commit\(",
            }
        ],
        "call_symbols": [],
        "call_patterns": [
            {
                "symbol": "canonical commit call",
                "pattern": r"\bcanonical_commit\(",
                "path_prefixes": ["crates/sim_pattern_fixture/"],
            }
        ],
    }
    source_cache = {path: source}
    definition_hits = configured_definition_hits([concern], source_cache)
    if definition_hits != [
        {
            "path": "crates/sim_pattern_fixture/src/fixture.rs",
            "line": 1,
            "symbol": "canonical_commit",
        }
    ]:
        raise RuntimeError("Configured ownership-definition scanner is not exact")
    call_sites = production_call_sites(concern, [], source_cache, definition_hits)
    if call_sites != [
        {
            "path": "crates/sim_pattern_fixture/src/fixture.rs",
            "line": 2,
            "symbol": "canonical commit call",
        },
    ]:
        raise RuntimeError("Configured ownership call-pattern scanner is not deterministic")
    canonical_concern = {
        "definitions": [
            {
                "path": "crates/sim_pattern_fixture/src/fixture.rs",
                "symbol": "canonical_commit",
                "role": "canonical_service",
            }
        ]
    }
    if canonical_definition_issues(canonical_concern, definition_hits):
        raise RuntimeError("canonical* role definition was not counted")
    if not canonical_definition_issues(canonical_concern, []):
        raise RuntimeError("missing canonical* role definition was not rejected")


def is_production_source(path: Path) -> bool:
    relative = path.relative_to(WORKSPACE)
    if path.suffix != ".rs" or not relative.parts or relative.parts[0] != "crates":
        return False
    if any(part in {"tests", "benches", "examples", "test_data"} for part in relative.parts):
        return False
    if len(relative.parts) > 1 and "test" in relative.parts[1].lower():
        return False
    stem = path.stem.lower()
    return not (stem == "tests" or stem.endswith("_test") or stem.endswith("_tests"))


def repository_production_call_sites(
    call_symbols: set[str],
    repository_hits: list[dict[str, Any]],
    source_cache: dict[Path, str],
) -> list[dict[str, Any]]:
    declaration_lines = {(hit["path"], hit["line"]) for hit in repository_hits}
    token_pattern = re.compile(
        r"\b(" + "|".join(re.escape(symbol) for symbol in sorted(call_symbols, key=lambda value: (-len(value), value))) + r")\b"
    )
    sites: list[dict[str, Any]] = []
    for path, source in source_cache.items():
        if not is_production_source(path):
            continue
        relative = path.relative_to(WORKSPACE).as_posix()
        test_lines = rust_test_lines(source)
        for line_number, line in enumerate(source.splitlines(), start=1):
            if line_number in test_lines or (relative, line_number) in declaration_lines:
                continue
            code = line.split("//", 1)[0]
            if not code.strip():
                continue
            for match in token_pattern.finditer(code):
                sites.append(
                    {"path": relative, "line": line_number, "symbol": match.group(1)}
                )
    return sorted(
        sites,
        key=lambda site: (site["path"], site["line"], site["symbol"]),
    )


def production_call_sites(
    concern: dict[str, Any],
    repository_sites: list[dict[str, Any]],
    source_cache: dict[Path, str],
    repository_hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    symbols = set(concern.get("call_symbols", []))
    prefixes = concern.get("call_path_prefixes")
    sites = [
        site
        for site in repository_sites
        if site["symbol"] in symbols
        and (prefixes is None or any(site["path"].startswith(prefix) for prefix in prefixes))
    ]
    declaration_lines = {(hit["path"], hit["line"]) for hit in repository_hits}
    for call_pattern in concern.get("call_patterns", []):
        path_prefixes = call_pattern.get("path_prefixes", prefixes)
        for path, source in source_cache.items():
            if not is_production_source(path):
                continue
            relative = path.relative_to(WORKSPACE).as_posix()
            if path_prefixes is not None and not any(
                relative.startswith(prefix) for prefix in path_prefixes
            ):
                continue
            test_lines = rust_test_lines(source)
            for match in re.finditer(call_pattern["pattern"], source, re.MULTILINE):
                line_number = position_line(source, match.start())
                if line_number in test_lines or (relative, line_number) in declaration_lines:
                    continue
                sites.append(
                    {
                        "path": relative,
                        "line": line_number,
                        "symbol": call_pattern["symbol"],
                    }
                )
    unique = {
        (site["path"], site["line"], site["symbol"]): site for site in sites
    }
    return sorted(
        unique.values(),
        key=lambda site: (site["path"], site["line"], site["symbol"]),
    )


def join(values: Iterable[str]) -> str:
    return " | ".join(values)


def format_definition_hits(hits: list[dict[str, Any]]) -> str:
    return join(
        f'{hit["role"]}@{hit["path"]}:{hit["line"]}:{hit["symbol"]}' for hit in hits
    )


def format_call_sites(sites: list[dict[str, Any]]) -> str:
    return join(f'{site["path"]}:{site["line"]}:{site["symbol"]}' for site in sites)


def current_status(
    concern: dict[str, Any],
    canonical_issues: list[str],
    prohibited: list[dict[str, Any]],
    unexpected: list[dict[str, Any]],
    missing_adapter_mappings: list[str],
) -> str:
    reasons = []
    if canonical_issues:
        reasons.append("canonical_definition_count_invalid")
    if prohibited:
        reasons.append("competing_definitions_present")
    if unexpected:
        reasons.append("unclassified_definitions_present")
    if missing_adapter_mappings:
        reasons.append("adapter_mapping_missing")
    if concern.get("known_open_reasons"):
        reasons.append("known_integration_gap")
    if reasons:
        return "consolidation_required[" + ",".join(reasons) + "]"
    return "authoritative_owner_confirmed"


def row_for(
    concern: dict[str, Any],
    repository_hits: list[dict[str, Any]],
    repository_sites: list[dict[str, Any]],
    source_cache: dict[Path, str],
    ownership_scope_prefixes: list[str],
    task_states: dict[str, bool],
) -> dict[str, str]:
    hits = exact_definition_hits(concern, repository_hits)
    canonical_issues = canonical_definition_issues(concern, hits)
    prohibited = prohibited_definition_hits(concern, hits)
    unexpected = unexpected_definition_hits(hits, ownership_scope_prefixes)
    missing_adapter_mappings, deferred_adapter_mappings = mapping_obligations(
        concern, source_cache, task_states
    )
    sites = production_call_sites(
        concern, repository_sites, source_cache, repository_hits
    )
    definitions_by_key = {
        (definition.get("path"), definition["symbol"]): definition
        for definition in concern["definitions"]
    }
    competing = []
    for hit in [*prohibited, *unexpected]:
        definition = definitions_by_key.get((hit["path"], hit["symbol"]))
        qualified = (
            hit["symbol"]
            if definition is None
            else definition.get("qualified", hit["symbol"])
        )
        competing.append(f'{hit["role"]}:{qualified}@{hit["path"]}:{hit["line"]}')
    structural_reason = []
    if canonical_issues:
        structural_reason.append(
            "canonical definition count violations: " + ", ".join(canonical_issues)
        )
    if prohibited:
        structural_reason.append(
            "prohibited definitions remain: "
            + ", ".join(
                f'{hit["path"]}:{hit["line"]}:{hit["symbol"]}' for hit in prohibited
            )
        )
    if unexpected:
        structural_reason.append(
            "unclassified in-scope definitions require a policy decision: "
            + ", ".join(
                f'{hit["path"]}:{hit["line"]}:{hit["symbol"]}' for hit in unexpected
            )
        )
    if missing_adapter_mappings:
        structural_reason.append(
            "required mappings not detected: " + ", ".join(missing_adapter_mappings)
        )
    activated_mapping_policies = [
        f'{mapping["name"]} with {mapping["activation_task"]}'
        for mapping in concern.get("required_mappings", [])
        if mapping.get("activation_task") is not None
    ]
    if activated_mapping_policies:
        structural_reason.append(
            "mapping obligations activate with their owning tasks: "
            + ", ".join(activated_mapping_policies)
        )
    structural_reason.extend(concern.get("known_open_reasons", []))
    consolidation_tasks = concern.get("consolidation_tasks", [])
    if structural_reason and consolidation_tasks:
        structural_reason.append(
            "owning tasks: " + ", ".join(consolidation_tasks)
        )
    decision_reason = concern["decision_reason"]
    if structural_reason:
        decision_reason += " Current evidence: " + "; ".join(structural_reason) + "."
    return {
        "concern": concern["concern"],
        "canonical_owner": concern["canonical_owner"],
        "owner_file": join(concern["owner_files"]),
        "owner_symbol": join(concern["owner_symbols"]),
        "consolidation_tasks": join(consolidation_tasks),
        "allowed_adapters": join(concern["allowed_adapters"]),
        "competing_symbols": join(competing),
        "production_consumers": join(concern["production_consumers"]),
        "requirements": join(concern["requirements"]),
        "design": join(concern["design"]),
        "validation": join(concern["validation"]),
        "current_status": current_status(
            concern, canonical_issues, prohibited, unexpected, missing_adapter_mappings
        ),
        "decision_reason": decision_reason,
        "definition_hits": format_definition_hits(hits),
        "production_call_sites": format_call_sites(sites),
    }


def validate_pattern(pattern: Any, label: str, flags: re.RegexFlag) -> None:
    if not isinstance(pattern, str) or not pattern:
        raise RuntimeError(f"{label} pattern must be a non-empty string")
    try:
        re.compile(pattern, flags)
    except re.error as error:
        raise RuntimeError(f"{label} pattern is invalid: {error}") from error


def validate_policy_path(value: Any, label: str, must_exist: bool) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} path must be a non-empty string")
    if "\\" in value:
        raise RuntimeError(f"{label} path must use repository-relative POSIX separators: {value}")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"{label} path must remain within the repository: {value}")
    resolved = (WORKSPACE / path).resolve()
    try:
        resolved.relative_to(WORKSPACE.resolve())
    except ValueError as error:
        raise RuntimeError(f"{label} path escapes the repository: {value}") from error
    if must_exist and not resolved.is_file():
        raise RuntimeError(f"{label} path does not name an existing file: {value}")
    return path.as_posix()


def validate_policy_prefix(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty string")
    if "\\" in value or value.startswith("/"):
        raise RuntimeError(f"{label} must be repository-relative: {value}")
    parts = Path(value).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise RuntimeError(f"{label} escapes the repository: {value}")


def validate_concerns(
    concerns: list[dict[str, Any]], task_states: dict[str, bool]
) -> None:
    if not all(isinstance(concern, dict) for concern in concerns):
        raise RuntimeError("every ownership concern must be an object")
    names = [concern.get("concern") for concern in concerns]
    if not all(isinstance(name, str) and name for name in names):
        raise RuntimeError("every ownership concern must have a non-empty string name")
    if names != sorted(names):
        raise RuntimeError("ownership concerns must be sorted by concern")
    if len(names) != len(set(names)):
        raise RuntimeError("ownership concerns must be unique")
    required = {
        "concern",
        "canonical_owner",
        "owner_files",
        "owner_symbols",
        "allowed_adapters",
        "definitions",
        "production_consumers",
        "requirements",
        "design",
        "validation",
        "decision_reason",
    }
    task_ids = set(task_states)
    for concern in concerns:
        missing = sorted(required - set(concern))
        if missing:
            raise RuntimeError(f'{concern.get("concern", "<unknown>")} missing fields: {missing}')
        concern_name = concern["concern"]
        for field in ("canonical_owner", "decision_reason"):
            if not isinstance(concern[field], str) or not concern[field]:
                raise RuntimeError(f"{concern_name} {field} must be a non-empty string")
        for field in (
            "owner_files",
            "owner_symbols",
            "allowed_adapters",
            "definitions",
            "production_consumers",
            "requirements",
            "design",
            "validation",
        ):
            values = concern[field]
            if not isinstance(values, list) or not values:
                raise RuntimeError(f"{concern_name} {field} must be a non-empty list")
        for field in (
            "owner_symbols",
            "allowed_adapters",
            "production_consumers",
            "requirements",
            "design",
            "validation",
        ):
            if not all(isinstance(value, str) and value for value in concern[field]):
                raise RuntimeError(f"{concern_name} {field} must contain non-empty strings")
        for owner_file in concern["owner_files"]:
            validate_policy_path(
                owner_file,
                f"{concern_name} owner file",
                must_exist=True,
            )
        definition_keys: set[tuple[str, str]] = set()
        for definition in concern["definitions"]:
            if not isinstance(definition, dict):
                raise RuntimeError(f"{concern_name} definition must be an object")
            missing_definition_fields = sorted(
                {"path", "qualified", "role", "symbol"} - set(definition)
            )
            if missing_definition_fields:
                raise RuntimeError(
                    f"{concern_name} definition is missing fields: {missing_definition_fields}"
                )
            role = definition["role"]
            if not isinstance(role, str):
                raise RuntimeError(f"{concern_name} definition role must be a string")
            if role not in ALLOWED_DEFINITION_ROLES:
                raise RuntimeError(f"{concern_name} uses unsupported definition role: {role!r}")
            for field in ("qualified", "symbol"):
                if not isinstance(definition[field], str) or not definition[field]:
                    raise RuntimeError(
                        f"{concern_name} definition {field} must be a non-empty string"
                    )
            definition_path = validate_policy_path(
                definition["path"],
                f"{concern_name} definition {definition['symbol']}",
                must_exist=role != "prohibited",
            )
            definition_key = (definition_path, definition["symbol"])
            if definition_key in definition_keys:
                raise RuntimeError(
                    f"{concern_name} repeats definition {definition_path}::{definition['symbol']}"
                )
            definition_keys.add(definition_key)
            pattern = definition.get("pattern")
            if pattern is not None:
                validate_pattern(
                    pattern,
                    f"{concern_name} definition {definition['symbol']}",
                    re.MULTILINE | re.DOTALL,
                )
        call_symbols = concern.get("call_symbols", [])
        if not isinstance(call_symbols, list):
            raise RuntimeError(f"{concern_name} call_symbols must be a list")
        for call_symbol in call_symbols:
            if not isinstance(call_symbol, str) or not call_symbol:
                raise RuntimeError(f"{concern_name} call_symbols must contain non-empty strings")
        call_patterns = concern.get("call_patterns", [])
        if not isinstance(call_patterns, list):
            raise RuntimeError(f"{concern_name} call_patterns must be a list")
        for call_pattern in call_patterns:
            if not isinstance(call_pattern, dict):
                raise RuntimeError(f"{concern_name} call pattern must be an object")
            missing_call_fields = sorted({"symbol", "pattern"} - set(call_pattern))
            if missing_call_fields:
                raise RuntimeError(
                    f"{concern_name} call pattern is missing fields: {missing_call_fields}"
                )
            validate_pattern(
                call_pattern["pattern"],
                f"{concern_name} call pattern {call_pattern['symbol']}",
                re.MULTILINE,
            )
            path_prefixes = call_pattern.get("path_prefixes", [])
            if not isinstance(path_prefixes, list):
                raise RuntimeError(
                    f"{concern_name} call-pattern path_prefixes must be a list"
                )
            for prefix in path_prefixes:
                validate_policy_prefix(prefix, f"{concern_name} call-pattern path prefix")
        call_path_prefixes = concern.get("call_path_prefixes", [])
        if not isinstance(call_path_prefixes, list):
            raise RuntimeError(f"{concern_name} call_path_prefixes must be a list")
        for prefix in call_path_prefixes:
            validate_policy_prefix(prefix, f"{concern_name} call path prefix")
        if "D41" not in concern["design"] or "VAL-OWNERSHIP-001" not in concern["validation"]:
            raise RuntimeError(f'{concern["concern"]} must map to D41 and VAL-OWNERSHIP-001')
        known_open_reasons = concern.get("known_open_reasons", [])
        if not isinstance(known_open_reasons, list) or not all(
            isinstance(reason, str) and reason for reason in known_open_reasons
        ):
            raise RuntimeError(f"{concern_name} known_open_reasons must be a string list")
        consolidation_tasks = concern.get("consolidation_tasks", [])
        if not isinstance(consolidation_tasks, list) or not all(
            isinstance(task_id, str) and task_id for task_id in consolidation_tasks
        ):
            raise RuntimeError(f"{concern_name} consolidation_tasks must be a string list")
        if known_open_reasons and not consolidation_tasks:
            raise RuntimeError(
                f'{concern["concern"]} has open ownership gaps without owning task IDs'
            )
        unknown_tasks = sorted(set(consolidation_tasks) - task_ids)
        if unknown_tasks:
            raise RuntimeError(
                f'{concern["concern"]} names unknown owning task IDs: {unknown_tasks}'
            )
        required_mappings = concern.get("required_mappings", [])
        if not isinstance(required_mappings, list):
            raise RuntimeError(f"{concern_name} required_mappings must be a list")
        mapping_names: set[str] = set()
        for mapping in required_mappings:
            if not isinstance(mapping, dict):
                raise RuntimeError(f"{concern_name} required mapping must be an object")
            missing_mapping_fields = sorted(
                {"name", "path", "pattern"} - set(mapping)
            )
            if missing_mapping_fields:
                raise RuntimeError(
                    f'{concern["concern"]} required mapping is missing fields: '
                    f"{missing_mapping_fields}"
                )
            mapping_name = mapping["name"]
            if not isinstance(mapping_name, str) or not mapping_name:
                raise RuntimeError(f"{concern_name} required mapping has an invalid name")
            if mapping_name in mapping_names:
                raise RuntimeError(f"{concern_name} repeats required mapping {mapping_name}")
            mapping_names.add(mapping_name)
            activation_task = mapping.get("activation_task")
            if activation_task is not None and (
                not isinstance(activation_task, str) or not activation_task
            ):
                raise RuntimeError(
                    f"{concern_name} mapping {mapping_name} has an invalid activation task"
                )
            validate_policy_path(
                mapping["path"],
                f"{concern_name} mapping {mapping_name}",
                must_exist=activation_task is None
                or task_states.get(activation_task, False),
            )
            validate_pattern(
                mapping["pattern"],
                f"{concern_name} mapping {mapping_name}",
                re.MULTILINE | re.DOTALL,
            )
            if activation_task is None:
                continue
            if activation_task not in task_ids:
                raise RuntimeError(
                    f'{concern["concern"]} mapping {mapping["name"]} names '
                    f"unknown activation task: {activation_task}"
                )
            if activation_task not in concern.get("consolidation_tasks", []):
                raise RuntimeError(
                    f'{concern["concern"]} mapping {mapping["name"]} activation '
                    f"task must also be a consolidation task: {activation_task}"
                )
            if not mapping.get("activation_reason"):
                raise RuntimeError(
                    f'{concern["concern"]} mapping {mapping["name"]} requires '
                    "an activation_reason"
                )
        if concern["concern"] in TASK_20_CONCERNS:
            if TASK_20_ID not in concern.get("consolidation_tasks", []):
                raise RuntimeError(
                    f'{concern["concern"]} must trace to {TASK_20_ID}'
                )
            if TASK_20_VALIDATION not in concern["validation"]:
                raise RuntimeError(
                    f'{concern["concern"]} must trace to {TASK_20_VALIDATION}'
                )
        required_task_39_mappings = TASK_39_CONCERNS.get(concern_name)
        if required_task_39_mappings is not None:
            if TASK_39_ID not in concern.get("consolidation_tasks", []):
                raise RuntimeError(f"{concern_name} must trace to {TASK_39_ID}")
            missing_validations = sorted(TASK_39_VALIDATIONS - set(concern["validation"]))
            if missing_validations:
                raise RuntimeError(
                    f"{concern_name} lacks Task 39 validations: {missing_validations}"
                )
            missing_task_39_mappings = sorted(required_task_39_mappings - mapping_names)
            if missing_task_39_mappings:
                raise RuntimeError(
                    f"{concern_name} lacks Task 39 mappings: {missing_task_39_mappings}"
                )
        required_task_315_mappings = TASK_315_CONCERNS.get(concern_name)
        if required_task_315_mappings is not None:
            if TASK_315_ID not in concern.get("consolidation_tasks", []):
                raise RuntimeError(f"{concern_name} must trace to {TASK_315_ID}")
            missing_validations = sorted(
                TASK_315_VALIDATIONS - set(concern["validation"])
            )
            if missing_validations:
                raise RuntimeError(
                    f"{concern_name} lacks Task 315 validations: {missing_validations}"
                )
            missing_task_315_mappings = sorted(
                required_task_315_mappings - mapping_names
            )
            if missing_task_315_mappings:
                raise RuntimeError(
                    f"{concern_name} lacks Task 315 mappings: {missing_task_315_mappings}"
                )
        required_task_338_mappings = TASK_338_CONCERNS.get(concern_name)
        if required_task_338_mappings is not None:
            if TASK_338_ID not in concern.get("consolidation_tasks", []):
                raise RuntimeError(f"{concern_name} must trace to {TASK_338_ID}")
            missing_validations = sorted(
                TASK_338_VALIDATIONS - set(concern["validation"])
            )
            if missing_validations:
                raise RuntimeError(
                    f"{concern_name} lacks Task 338 validations: {missing_validations}"
                )
            missing_task_338_mappings = sorted(
                required_task_338_mappings - mapping_names
            )
            if missing_task_338_mappings:
                raise RuntimeError(
                    f"{concern_name} lacks Task 338 mappings: {missing_task_338_mappings}"
                )
        for task_id, task_number, required_mappings in [
            (TASK_307_ID, 307, TASK_307_CONCERNS.get(concern_name)),
            (TASK_308_ID, 308, TASK_308_CONCERNS.get(concern_name)),
        ]:
            if required_mappings is None:
                continue
            if task_id not in concern.get("consolidation_tasks", []):
                raise RuntimeError(f"{concern_name} must trace to {task_id}")
            missing_validations = sorted(
                TASK_307_308_VALIDATIONS - set(concern["validation"])
            )
            if missing_validations:
                raise RuntimeError(
                    f"{concern_name} lacks Task {task_number} validations: "
                    f"{missing_validations}"
                )
            missing_task_mappings = sorted(required_mappings - mapping_names)
            if missing_task_mappings:
                raise RuntimeError(
                    f"{concern_name} lacks Task {task_number} mappings: "
                    f"{missing_task_mappings}"
                )
        required_task_104_mappings = TASK_104_CONCERNS.get(concern_name)
        if required_task_104_mappings is not None:
            if TASK_104_ID not in concern.get("consolidation_tasks", []):
                raise RuntimeError(f"{concern_name} must trace to {TASK_104_ID}")
            missing_validations = sorted(
                TASK_104_VALIDATIONS - set(concern["validation"])
            )
            if missing_validations:
                raise RuntimeError(
                    f"{concern_name} lacks Task 104 validations: {missing_validations}"
                )
            missing_task_104_mappings = sorted(
                required_task_104_mappings - mapping_names
            )
            if missing_task_104_mappings:
                raise RuntimeError(
                    f"{concern_name} lacks Task 104 mappings: {missing_task_104_mappings}"
                )
        required_task_101_mappings = TASK_101_CONCERNS.get(concern_name)
        if required_task_101_mappings is not None:
            if TASK_101_ID not in concern.get("consolidation_tasks", []):
                raise RuntimeError(f"{concern_name} must trace to {TASK_101_ID}")
            missing_validations = sorted(
                TASK_101_VALIDATIONS - set(concern["validation"])
            )
            if missing_validations:
                raise RuntimeError(
                    f"{concern_name} lacks Task 101 validations: {missing_validations}"
                )
            missing_task_101_mappings = sorted(
                required_task_101_mappings - mapping_names
            )
            if missing_task_101_mappings:
                raise RuntimeError(
                    f"{concern_name} lacks Task 101 mappings: {missing_task_101_mappings}"
                )
        required_task_102_mappings = TASK_102_CONCERNS.get(concern_name)
        if required_task_102_mappings is not None:
            if TASK_102_ID not in concern.get("consolidation_tasks", []):
                raise RuntimeError(f"{concern_name} must trace to {TASK_102_ID}")
            missing_validations = sorted(
                TASK_102_VALIDATIONS - set(concern["validation"])
            )
            if missing_validations:
                raise RuntimeError(
                    f"{concern_name} lacks Task 102 validations: {missing_validations}"
                )
            missing_task_102_mappings = sorted(
                required_task_102_mappings - mapping_names
            )
            if missing_task_102_mappings:
                raise RuntimeError(
                    f"{concern_name} lacks Task 102 mappings: {missing_task_102_mappings}"
                )
        required_task_511_mappings = TASK_511_CONCERNS.get(concern_name)
        if required_task_511_mappings is not None:
            if TASK_511_ID not in concern.get("consolidation_tasks", []):
                raise RuntimeError(f"{concern_name} must trace to {TASK_511_ID}")
            missing_validations = sorted(
                TASK_511_VALIDATIONS - set(concern["validation"])
            )
            if missing_validations:
                raise RuntimeError(
                    f"{concern_name} lacks Task 511 validations: {missing_validations}"
                )
            missing_task_511_mappings = sorted(
                required_task_511_mappings - mapping_names
            )
            if missing_task_511_mappings:
                raise RuntimeError(
                    f"{concern_name} lacks Task 511 mappings: {missing_task_511_mappings}"
                )
        required_task_512_mappings = TASK_512_CONCERNS.get(concern_name)
        if required_task_512_mappings is not None:
            if TASK_512_ID not in concern.get("consolidation_tasks", []):
                raise RuntimeError(f"{concern_name} must trace to {TASK_512_ID}")
            missing_validations = sorted(
                TASK_512_VALIDATIONS - set(concern["validation"])
            )
            if missing_validations:
                raise RuntimeError(
                    f"{concern_name} lacks Task 512 validations: {missing_validations}"
                )
            missing_task_512_mappings = sorted(
                required_task_512_mappings - mapping_names
            )
            if missing_task_512_mappings:
                raise RuntimeError(
                    f"{concern_name} lacks Task 512 mappings: {missing_task_512_mappings}"
                )
        required_task_103_mappings = TASK_103_CONCERNS.get(concern_name)
        if required_task_103_mappings is not None:
            if TASK_103_ID not in concern.get("consolidation_tasks", []):
                raise RuntimeError(f"{concern_name} must trace to {TASK_103_ID}")
            missing_validations = sorted(
                TASK_103_VALIDATIONS - set(concern["validation"])
            )
            if missing_validations:
                raise RuntimeError(
                    f"{concern_name} lacks Task 103 validations: {missing_validations}"
                )
            missing_task_103_mappings = sorted(
                required_task_103_mappings - mapping_names
            )
            if missing_task_103_mappings:
                raise RuntimeError(
                    f"{concern_name} lacks Task 103 mappings: {missing_task_103_mappings}"
                )
        vendor_foundation = VENDOR_ABI_FOUNDATIONS.get(concern_name)
        if vendor_foundation is not None:
            task_id = vendor_foundation["task_id"]
            if task_id not in concern.get("consolidation_tasks", []):
                raise RuntimeError(f"{concern_name} must trace to {task_id}")
            ownership_task_id = vendor_foundation["ownership_task_id"]
            if ownership_task_id not in concern.get("consolidation_tasks", []):
                raise RuntimeError(
                    f"{concern_name} must trace to {ownership_task_id}"
                )
            missing_validations = sorted(
                VENDOR_ABI_FOUNDATION_VALIDATIONS - set(concern["validation"])
            )
            if missing_validations:
                raise RuntimeError(
                    f"{concern_name} lacks vendor ABI validations: {missing_validations}"
                )
            missing_mappings = sorted(
                vendor_foundation["mappings"] - mapping_names
            )
            if missing_mappings:
                raise RuntimeError(
                    f"{concern_name} lacks vendor ABI mappings: {missing_mappings}"
                )


def main() -> None:
    validate_rust_test_line_parser()
    validate_configured_pattern_scanners()
    validate_task_activated_mapping_semantics()
    policy = load_policy()
    task_states = task_completion_states()
    validate_concerns(policy["concerns"], task_states)
    concerns = sorted(policy["concerns"], key=lambda concern: concern["concern"])
    sources = repository_sources(policy)
    source_cache = {path: source_text(path) for path in sources}
    definition_symbols = {
        definition["symbol"]
        for concern in concerns
        for definition in concern["definitions"]
    }
    call_symbols = {
        symbol for concern in concerns for symbol in concern.get("call_symbols", [])
    }
    repository_hits = repository_definition_hits(definition_symbols, source_cache)
    configured_hits = configured_definition_hits(concerns, source_cache)
    repository_hits = sorted(
        {
            (hit["path"], hit["line"], hit["symbol"]): hit
            for hit in [*repository_hits, *configured_hits]
        }.values(),
        key=lambda hit: (hit["path"], hit["line"], hit["symbol"]),
    )
    repository_sites = repository_production_call_sites(
        call_symbols, repository_hits, source_cache
    )
    rows = [
        row_for(
            concern,
            repository_hits,
            repository_sites,
            source_cache,
            policy["scan"]["ownership_scope_prefixes"],
            task_states,
        )
        for concern in concerns
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Wrote {len(rows)} authoritative ownership rows from "
        f"{len(sources)} repository source files to {OUTPUT.relative_to(WORKSPACE)}"
    )


if __name__ == "__main__":
    main()
