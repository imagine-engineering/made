#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: bash crates/sim_backend_rocm/abi/verify-completion-evidence.sh <flat-pinned-header-directory> [artifact-path]" >&2
    exit 2
fi

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd "${script_directory}/../../.." && pwd)
header_directory=$(cd "$1" && pwd)
artifact=${2:-"${repository_root}/target/sim-rocm-6.1.2-abi-proof.json"}
artifact_directory=$(dirname "${artifact}")
mkdir -p "${artifact_directory}"
artifact_directory=$(cd "${artifact_directory}" && pwd)
artifact="${artifact_directory}/$(basename "${artifact}")"
run_id="rocm-6.1.2-$(date -u +%Y%m%dT%H%M%SZ)-$$"
pending_artifact="${artifact}.pending-${run_id}"
trap 'rm -f "${pending_artifact}"' EXIT

export SIM_ROCM_REQUIRE_COMPLETION_EVIDENCE=1
export SIM_ROCM_REVIEWED_HEADER_DIR="${header_directory}"
export SIM_ROCM_COMPLETION_EVIDENCE_OUT="${pending_artifact}"
export SIM_ROCM_COMPLETION_EVIDENCE_RUN_ID="${run_id}"

cd "${repository_root}"

rustc --edition=2024 --test crates/sim_backend_rocm/build.rs \
    -o target/sim_backend_rocm_build_evidence_tests
target/sim_backend_rocm_build_evidence_tests
cargo test -p sim_backend_rocm --all-targets

python3 -c '
import json, pathlib, sys
artifact = pathlib.Path(sys.argv[1])
run_id = sys.argv[2]
document = json.loads(artifact.read_text())
assert document["status"] == "verified"
assert document["run_id"] == run_id
assert len(document["headers"]) == 9
assert document["symbol_count"] == 52
assert document["binding_count"] == 74
assert document["c_probe"]["static_assertions"] == "passed"
assert document["c_probe"]["measurements"] == document["rust_probe"]["measurements"]
assert len(document["layout_declarations"]) == 3
assert len(document["constant_declarations"]) == 19
' "${pending_artifact}" "${run_id}"

mv "${pending_artifact}" "${artifact}"
shasum -a 256 "${artifact}"
