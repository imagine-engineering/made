#!/usr/bin/env python3

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "crates/sim_test_support/fixtures"
FIXTURE_NAMES = (
    "native_image/input.json",
    "native_image/expected.json",
    "native_image/workflow.json",
    "video/component-h264-mp4-10bit-backing/manifest.json",
    "video/general-demux-decode/manifest.json",
)
H264_OWNERSHIP = "val_ownership_native_video_component_h264_mp4_10bit_backing_001"
SUITES = (
    (
        "media_nodes_and_runtime",
        ("-p", "sim_media", "-p", "sim_nodes", "-p", "sim_runtime", "--all-targets"),
        (
            "metadata::tests::val_metadata_001",
            "metadata::tests::deterministic_hostile_truncation_matrix_never_panics_or_changes_source",
            "png::tests::native_png_encoding_preserves_rgba_preview_alpha",
            "native_node_payload::tests::canonical_audio_admits_zero_samples_without_padding",
            "native_node_payload::tests::encoded_h264_mp4_video_binds_component_identity_and_portable_storage",
            "native_node_payload::tests::canonical_splat_mesh_voxel_and_camera_payloads_bind_topology_and_residency",
            "video::tests::codec_plan_rejects_unsupported_profiles_limits_and_cancellation",
            "execution::tests::component_h264_mp4_backing_service_requires_checked_video_and_portable_contract",
            "execution::tests::image_preview_prepares_batched_pngs_and_rolls_back_partial_failure",
            "native_execution_controller::tests::committed_video_output_projects_typed_preview_and_output_events",
            "native_video_codec_service::tests::owned_encoded_output_materialization_is_accounted_atomic_and_retryable",
            "native_video_codec_service::tests::retained_video_codec_thread_returns_owned_codec_bytes_and_preserves_ltxv",
        ),
    ),
    (
        "packaged_image_and_codec_fixtures",
        (
            "-p", "sim_test_support", "--test", "native_image_e2e", "--test", "native_image_recovery",
            "--test", "video_codec_package_bootstrap", "--test", "video_general_demux_decode",
        ),
        (
            "val_native_e2e_001",
            "val_recovery_003",
            "signed_general_video_package_certifies_and_relocates_stably",
            "signed_general_video_package_rejects_tree_and_payload_mutations",
            "general_video_worker_bundle_is_ready_or_typed_unsupported",
            "general_demux_decode_fixture_is_exact_and_explicitly_uncertified",
            "general_demux_decode_fixture_pins_source_selection_and_numeric_boundaries",
        ),
    ),
    (
        "h264_ownership",
        ("-p", "sim_test_support", "--test", "ownership_consolidation", H264_OWNERSHIP, "--", "--exact"),
        (H264_OWNERSHIP,),
    ),
)
TEST_SUMMARY = re.compile(
    r"^test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;", re.MULTILINE
)
PASSED_TEST = re.compile(r"^test (\S+) \.\.\. ok$", re.MULTILINE)


def suite_result(exit_status, output, required):
    counts = [tuple(map(int, match)) for match in TEST_SUMMARY.findall(output)]
    passed, failed, ignored = (sum(count[index] for count in counts) for index in range(3))
    observed = set(PASSED_TEST.findall(output))
    cases = {case: case in observed for case in required}
    return {
        "exit_status": exit_status,
        "passed": passed,
        "failed": failed,
        "ignored": ignored,
        "cases": cases,
        "passed_required_checks": (
            exit_status == 0 and passed > 0 and failed == 0 and ignored == 0 and all(cases.values())
        ),
    }


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        try:
            json.dump(report, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.close()
            os.replace(temporary.name, path)
        finally:
            Path(temporary.name).unlink(missing_ok=True)


def main():
    target = ROOT / os.environ.get("CARGO_TARGET_DIR", "target")
    path = target / "comfy-parity/val-media-001.json"
    cargo = os.environ.get("CARGO", "cargo")
    report = {
        "schema_version": 1,
        "validation": "VAL-MEDIA-001",
        "scope": "executed native media, node, runtime, and packaged-fixture suites",
        "environment": {"platform": sys.platform, "architecture": platform.machine(), "backend": "native-rust"},
        "fixture_digests": {
            name: hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() for name in FIXTURE_NAMES
        },
        "suites": {},
        "unverified_acceptance": [
            "Source-linked coverage of every cataloged media codec, carrier, output, and preview.",
            "Visual verification of the cataloged application output and preview paths.",
            "Installed and signed native codec qualification on the target release host.",
        ],
    }
    for name, arguments, required in SUITES:
        command = [cargo, "test", "--locked", "--offline", "--color", "never", *arguments]
        print(f"Running {name}", flush=True)
        try:
            completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            output = completed.stdout
            exit_status = completed.returncode
        except OSError as error:
            output = str(error) + "\n"
            exit_status = 1
        print(output, end="", flush=True)
        report["suites"][name] = suite_result(exit_status, output, required)

    passed = all(suite["passed_required_checks"] for suite in report["suites"].values())
    # The test runs carry synthetic and scoped native evidence. Acceptance still needs full catalog and target receipts.
    report["status"] = "partial" if passed else "failed"
    write_report(path, report)
    print(f"VAL-MEDIA-001 {report['status']}: {path}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
