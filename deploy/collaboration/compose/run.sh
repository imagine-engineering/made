#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_directory}"

environment_file="${COLLABORATION_ENV_FILE:-.env}"
compose_files=(-f compose.yaml)
if [[ "${COLLABORATION_BUILD_LOCAL:-false}" == "true" ]]; then
  compose_files+=(-f compose.local.yaml)
fi

compose() {
  docker compose --env-file "${environment_file}" "${compose_files[@]}" "$@"
}

require_environment() {
  if [[ ! -f "${environment_file}" ]]; then
    echo "Missing ${script_directory}/${environment_file}; copy .env.example and replace its placeholders." >&2
    exit 1
  fi
  if grep -Eq '^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*=.*CHANGE_ME' "${environment_file}"; then
    echo "${environment_file} still contains CHANGE_ME placeholders." >&2
    exit 1
  fi
}

require_immutable_image() {
  local image="$1"
  if [[ ! "${image}" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]]; then
    echo "Rollback requires COLLABORATION_PREVIOUS_IMAGE=<repository>@sha256:<64 lowercase hex characters>." >&2
    exit 1
  fi
}

environment_value() {
  local key="$1"
  local line
  line="$(grep -E "^${key}=" "${environment_file}" | tail -n 1)"
  printf '%s' "${line#*=}"
}

verify_readiness() {
  compose --profile validation run --rm --no-deps collaboration-readiness
}

smoke() {
  local smoke_project="zed-collaboration-smoke-${PPID}"
  local smoke_files=(-f compose.yaml -f compose.smoke.yaml)
  cleanup_smoke() {
    docker compose -p "${smoke_project}" --env-file .env.smoke "${smoke_files[@]}" down --volumes --remove-orphans
  }
  trap cleanup_smoke EXIT
  docker compose -p "${smoke_project}" --env-file .env.smoke "${smoke_files[@]}" up --detach --wait
  docker compose -p "${smoke_project}" --env-file .env.smoke "${smoke_files[@]}" --profile validation run --rm --no-deps collaboration-readiness
  cleanup_smoke
  trap - EXIT
}

case "${1:-help}" in
  config)
    require_environment
    compose config
    ;;
  start|up)
    require_environment
    compose up --detach --wait
    verify_readiness
    ;;
  stop|down)
    require_environment
    compose down
    ;;
  status|ps)
    require_environment
    compose ps
    ;;
  logs)
    require_environment
    shift
    compose logs --follow "${@:-collaboration}"
    ;;
  clients)
    require_environment
    shift
    repository_root="$(cd "${script_directory}/../../.." && pwd)"
    ZED_PRODUCT_ID=rust \
      ZED_LOCAL_CARGO_FEATURES=multiplayer-tools,rust-tools \
      ZED_ADMIN_API_TOKEN="$(environment_value ZED_CLOUD_INTERNAL_API_KEY)" \
      "${repository_root}/script/zed-local" -2 --stateful "$@"
    ;;
  rollback)
    require_environment
    previous_image="${COLLABORATION_PREVIOUS_IMAGE:-}"
    if [[ -z "${previous_image}" ]]; then
      previous_image="$(environment_value COLLABORATION_PREVIOUS_IMAGE)"
    fi
    require_immutable_image "${previous_image}"
    COLLABORATION_IMAGE="${previous_image}" compose pull collaboration
    COLLABORATION_IMAGE="${previous_image}" compose up --detach --wait --force-recreate --no-deps collaboration
    COLLABORATION_IMAGE="${previous_image}" verify_readiness
    ;;
  smoke)
    smoke
    ;;
  help|-h|--help)
    cat <<'MSG'
Usage: ./run.sh <command>

Commands:
  config    Render the effective Compose configuration
  start     Start the stack, wait for health, then require /healthz readiness
  stop      Stop containers without deleting canonical data volumes
  status    Show Compose service status
  logs      Follow service logs (default: collaboration)
  clients   Build and launch two authenticated Rust-product clients
  rollback  Recreate only Collab from COLLABORATION_PREVIOUS_IMAGE
  smoke     Exercise health ordering with an isolated validation-only stack

Set COLLABORATION_BUILD_LOCAL=true to add compose.local.yaml and build
Dockerfile-collab. Rollback accepts only an immutable sha256 image reference and
does not change dependencies, volumes, or schema.
MSG
    ;;
  *)
    echo "Unknown command: $1" >&2
    exit 1
    ;;
esac
