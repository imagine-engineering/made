#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../../.." && pwd)"
review="$root/crates/sim_backend_npu/abi/reviewed-bindings-v1.txt"
manifest="$root/crates/sim_backend_npu/abi/symbols-v1.json"

expected_review_sha256="0b4481f131bfa8b311ee6e1f7a926eb3fdcfffc0e0165fb64ed4fd8e4036cb81"
actual_review_sha256="$(shasum -a 256 "$review" | awk '{print $1}')"
if [[ "$actual_review_sha256" != "$expected_review_sha256" ]]; then
    echo "reviewed AscendCL execution declarations changed: $actual_review_sha256" >&2
    exit 1
fi

expected_manifest_sha256="2df75a090079b923cdeea2f5464b29a9c78ef35223b6a16884ed07a778466b2d"
actual_manifest_sha256="$(shasum -a 256 "$manifest" | awk '{print $1}')"
if [[ "$actual_manifest_sha256" != "$expected_manifest_sha256" ]]; then
    echo "reviewed AscendCL ABI manifest changed: $actual_manifest_sha256" >&2
    exit 1
fi

while IFS= read -r symbol; do
    rg --fixed-strings --quiet "$symbol" "$review"
    rg --fixed-strings --quiet "\"name\":\"$symbol\"" "$manifest"
done <<'SYMBOLS'
aclCreateDataBuffer
aclCreateTensorDesc
aclDestroyDataBuffer
aclDestroyTensorDesc
aclopExecuteV2
aclrtCreateEvent
aclrtDestroyEvent
aclrtGetMemInfo
aclrtGetSocName
aclrtRecordEvent
aclrtSetCurrentContext
aclrtSynchronizeEvent
SYMBOLS

rg --fixed-strings --quiet 'source_sha256=91d8bd8a346bda371c8175066ac5155fb27ccfe4ba63091730ec29dcd96dd091' "$review"
rg --fixed-strings --quiet 'ACL_FLOAT16:1' "$review"
rg --fixed-strings --quiet 'ACL_FLOAT:0' "$review"
rg --fixed-strings --quiet 'ACL_FORMAT_ND:2' "$review"
rg --fixed-strings --quiet 'ACL_HBM_MEM:1' "$review"

for digest in \
    1cab4286a330cfb10337e6fde6ffcc7390a4855ea143db63e84d497d185391bc \
    be4f6d8ec73bf30cdf1ba42ceea8aeb3f1aebd8900192537ce6a5f80f317e765 \
    2b2eeaf361f9e26fee8b0d30c4cede7c586e08090487b60d7e97fd42445e9c3c \
    68d31a009dc580773ce7f568ec1082a0120864e15401f17ed3a9207bef3eabb1 \
    8d745b66e690ac8d10aabb952e10ffdbc212de835e014b0ada0afbf17718c898 \
    2ad93a44e0a829abab940bd471b078414edab9cd3b1789fd1ecc595d376408ba \
    4dafca0edb13d12d5cdf8b2a61e852ec189b3fd34ed705851b8d0839ee9ca91f \
    ac404bc76e96a7b091350c51fe40a8f75ee7032d0d0c29e5a472dc62969ea8f0
do
    rg --fixed-strings --quiet "$digest" "$manifest"
done
