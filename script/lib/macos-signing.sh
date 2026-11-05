# Sourced only by the macOS product packager; keep tracing off before secrets are used.
set +x
signing_keychain_arguments=()
signing_temporary_directory=""
signing_keychain=""
signing_search_list_snapshot=""

cleanup_signing() {
    local status=$?
    trap - EXIT
    if [[ -n "$signing_search_list_snapshot" && -f "$signing_search_list_snapshot" ]]; then
        if ! python3 - "$signing_search_list_snapshot" <<'RESTORE_KEYCHAINS'
import json
from pathlib import Path
import subprocess
import sys
keychains = json.loads(Path(sys.argv[1]).read_text())
subprocess.run(["security", "list-keychains", "-d", "user", "-s", *keychains], check=True)
RESTORE_KEYCHAINS
        then
            echo "Failed to restore the previous keychain search list" >&2
            status=1
        fi
    fi
    if [[ -n "$signing_keychain" && -f "$signing_keychain" ]]; then
        if ! security delete-keychain "$signing_keychain"; then
            echo "Failed to delete the temporary signing keychain" >&2
            status=1
        fi
    fi
    if [[ -n "$signing_temporary_directory" ]]; then
        if ! rm -rf "$signing_temporary_directory"; then
            echo "Failed to delete temporary signing files" >&2
            status=1
        fi
    fi
    exit "$status"
}

setup_production_signing() {
    umask 077
    signing_temporary_directory=$(mktemp -d "${TMPDIR:-/tmp}/copper-signing.XXXXXXXX")
    trap cleanup_signing EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    signing_keychain="$signing_temporary_directory/signing.keychain-db"
    local keychain_password
    keychain_password=$(openssl rand -hex 32)
    security create-keychain -p "$keychain_password" "$signing_keychain"
    security unlock-keychain -p "$keychain_password" "$signing_keychain"
    security set-keychain-settings -lut 21600 "$signing_keychain"
    python3 - "$signing_temporary_directory" <<'PY'
import base64
import os
from pathlib import Path
import sys
root = Path(sys.argv[1])
(root / "certificate.p12").write_bytes(base64.b64decode("".join(os.environ["MACOS_CERTIFICATE"].split()), validate=True))
(root / "notary.p8").write_text(os.environ["APPLE_NOTARIZATION_KEY"])
PY
    security import "$signing_temporary_directory/certificate.p12" -k "$signing_keychain" -P "$MACOS_CERTIFICATE_PASSWORD" -T /usr/bin/codesign >/dev/null
    rm "$signing_temporary_directory/certificate.p12"
    security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$keychain_password" "$signing_keychain" >/dev/null
    # codesign still requires search-list membership for identity/chain resolution
    # when --keychain restricts identity selection (see codesign(1)).
    signing_search_list_snapshot="$signing_temporary_directory/search-list.json"
    python3 - "$signing_search_list_snapshot" "$signing_keychain" <<'REGISTER_KEYCHAIN'
import json
from pathlib import Path
import shlex
import subprocess
import sys
snapshot, signing_keychain = sys.argv[1:]
result = subprocess.run(["security", "list-keychains", "-d", "user"], capture_output=True, text=True, check=True)
keychains = shlex.split(result.stdout)
Path(snapshot).write_text(json.dumps(keychains))
subprocess.run(["security", "list-keychains", "-d", "user", "-s", *keychains, signing_keychain], check=True)
REGISTER_KEYCHAIN
    signing_keychain_arguments=(--keychain "$signing_keychain")
    security find-identity -v -p codesigning "$signing_keychain" | grep -F -- "$IDENTITY" >/dev/null || {
        echo "The imported keychain does not contain MACOS_SIGNING_IDENTITY" >&2
        return 1
    }
}

notarize_and_staple() {
    local submission=$1
    local ticket_target=$2
    local label=$3
    local evidence_directory="${cargo_target_dir}/signing-evidence"
    mkdir -p "$evidence_directory"
    local result="$evidence_directory/${label}.json"
    xcrun notarytool submit "$submission" --wait --output-format json \
        --key "$signing_temporary_directory/notary.p8" \
        --key-id "$APPLE_NOTARIZATION_KEY_ID" --issuer "$APPLE_NOTARIZATION_ISSUER_ID" > "$result"
    python3 - "$result" <<'PY'
import json
from pathlib import Path
import sys
result = json.loads(Path(sys.argv[1]).read_text())
if result.get("status") != "Accepted":
    raise SystemExit("Apple notarization was not Accepted; inspect the submission ID in signing-evidence")
PY
    xcrun stapler staple "$ticket_target"
    xcrun stapler validate "$ticket_target"
}
