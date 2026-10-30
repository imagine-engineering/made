#!/usr/bin/env bash
set -euo pipefail

repository=$(git rev-parse --show-toplevel)
drift_directory=$(mktemp -d "${TMPDIR:-/tmp}/zed-collaboration-migrations-XXXXXXXX")
run_id=$(basename "$drift_directory")
image="zed-collaboration-migrations:${run_id}"
network="$run_id"
postgres="${network}-postgres"
interrupted_migrator="${network}-interrupted"
ownership_label="dev.simtropolis.migration-test=${run_id}"
network_created=false
postgres_created=false
image_created=false
interrupted_created=false

cleanup() {
  local status=$?
  trap - EXIT
  if [[ "$interrupted_created" == true ]] && ! docker rm --force --volumes "$interrupted_migrator"; then
    echo "could not remove owned interrupted migrator: $interrupted_migrator" >&2
    status=1
  fi
  if [[ "$postgres_created" == true ]] && ! docker rm --force --volumes "$postgres"; then
    echo "could not remove owned PostgreSQL container: $postgres" >&2
    status=1
  fi
  if [[ "$network_created" == true ]] && ! docker network rm "$network"; then
    echo "could not remove owned migration network: $network" >&2
    status=1
  fi
  if [[ "$image_created" == true ]] && ! docker image rm "$image"; then
    echo "could not remove owned migration image: $image" >&2
    status=1
  fi
  if ! rm -rf -- "$drift_directory"; then
    status=1
  fi
  exit "$status"
}
trap cleanup EXIT

# Agent commands can share a parent process; PPID cannot establish resource ownership.
echo "migration_test_run_id=$run_id ownership_label=$ownership_label"
docker network create --label "$ownership_label" "$network" >/dev/null
network_created=true
docker run --detach --name "$postgres" --network "$network" \
  --label "$ownership_label" \
  --env POSTGRES_PASSWORD=migration-canary \
  --env POSTGRES_DB=collaboration \
  postgres:17-alpine >/dev/null
postgres_created=true
docker inspect --type container --format '{{range .Mounts}}{{if eq .Type "volume"}}migration_test_volume={{.Name}} {{end}}{{end}}' "$postgres"

# The initialization server accepts local sockets before the final TCP server starts.
for attempt in $(seq 1 60); do
  if docker exec "$postgres" pg_isready --host 127.0.0.1 --username postgres --dbname collaboration >/dev/null 2>&1; then
    break
  fi
  if [[ "$attempt" == "60" ]]; then
    echo "PostgreSQL did not become ready" >&2
    exit 1
  fi
  sleep 1
done

docker exec --interactive "$postgres" psql --username postgres --dbname collaboration \
  --set ON_ERROR_STOP=1 <crates/collab/migrations/20251208000000_test_schema.sql >/dev/null

docker build --file deploy/collaboration/migrations/Dockerfile --tag "$image" --label "$ownership_label" "$repository" >/dev/null
image_created=true

run_migration() {
  local database="$1"
  shift
  docker run --rm --label "$ownership_label" --network "$network" \
    --env "DATABASE_URL=postgres://postgres:migration-canary@${postgres}:5432/${database}" \
    --env COLLABORATION_REQUIRED_SCHEMA_VERSION=20261026000100 \
    "$image" "$@"
}

if docker run --rm --label "$ownership_label" \
  --env COLLABORATION_REQUIRED_SCHEMA_VERSION=20260824000400 \
  "$image" validate >/dev/null 2>&1; then
  echo "migration artifact accepted a mismatched required schema version" >&2
  exit 1
fi

run_migration collaboration up --target-version 20260820000500
run_migration collaboration verify | grep -q 'current=20260820000500.*applied=6'
run_migration collaboration up
run_migration collaboration verify | grep -q 'current=20261026000100.*applied=31'
run_migration collaboration up | grep -q 'migration_up_applied=0'

run_migration collaboration down --target-version 20260824000500
table_after_down=$(docker exec "$postgres" psql --username postgres --dbname collaboration --tuples-only --no-align \
  --command "SELECT to_regclass('public.collaboration_workflow_ready_queue_index') IS NULL")
[[ "$table_after_down" == "t" ]]
run_migration collaboration up
run_migration collaboration seal --expected-version 20261026000100
if run_migration collaboration down --target-version 20260824000500 >/dev/null 2>&1; then
  echo "rollback crossed the sealed compatibility boundary" >&2
  exit 1
fi
run_migration collaboration status | grep -q 'rollback_floor=20261026000100'

docker exec "$postgres" createdb --username postgres interrupted
docker exec --interactive "$postgres" psql --username postgres --dbname interrupted \
  --set ON_ERROR_STOP=1 <crates/collab/migrations/20251208000000_test_schema.sql >/dev/null
run_migration interrupted up --target-version 20260825000100
constraint_before=$(docker exec "$postgres" psql --username postgres --dbname interrupted --tuples-only --no-align \
  --command "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'collaboration_command_receipts_originating_adapter_check'")

# Block after the transport migration changes a constraint but before it commits.
# Killing this disposable database tests WAL recovery of partially executed DDL.
docker exec --detach --env PGAPPNAME=migration-interruption-lock "$postgres" \
  psql --username postgres --dbname interrupted --set ON_ERROR_STOP=1 \
  --command 'BEGIN; LOCK TABLE public.channels IN ACCESS EXCLUSIVE MODE; SELECT pg_sleep(120); COMMIT;'
for attempt in $(seq 1 60); do
  locked=$(docker exec "$postgres" psql --username postgres --dbname interrupted --tuples-only --no-align \
    --command "SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE application_name = 'migration-interruption-lock' AND wait_event = 'PgSleep')")
  [[ "$locked" == "t" ]] && break
  [[ "$attempt" != "60" ]] || { echo "interruption lock did not start" >&2; exit 1; }
  sleep 0.1
done
docker run --detach --name "$interrupted_migrator" --network "$network" \
  --label "$ownership_label" \
  --env "DATABASE_URL=postgres://postgres:migration-canary@${postgres}:5432/interrupted" \
  --env COLLABORATION_REQUIRED_SCHEMA_VERSION=20261026000100 "$image" up >/dev/null
interrupted_created=true
docker inspect --type container --format '{{range .Mounts}}{{if eq .Type "volume"}}migration_test_volume={{.Name}} {{end}}{{end}}' "$interrupted_migrator"
for attempt in $(seq 1 60); do
  interrupted=$(docker exec "$postgres" psql --username postgres --dbname interrupted --tuples-only --no-align \
    --command "SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE datname = 'interrupted' AND wait_event_type = 'Lock' AND query LIKE 'CREATE TABLE public.collaboration_zed_community_bindings%')")
  [[ "$interrupted" == "t" ]] && break
  [[ "$attempt" != "60" ]] || { echo "migration did not reach in-flight DDL" >&2; docker logs "$interrupted_migrator"; exit 1; }
  sleep 0.1
done
docker kill --signal KILL "$postgres" >/dev/null
interrupted_result=$(docker wait "$interrupted_migrator")
[[ "$interrupted_result" != "0" ]]
docker logs "$interrupted_migrator"
docker start "$postgres" >/dev/null
for attempt in $(seq 1 60); do
  if docker exec "$postgres" pg_isready --host 127.0.0.1 --username postgres --dbname interrupted >/dev/null 2>&1; then
    break
  fi
  [[ "$attempt" != "60" ]] || { echo "PostgreSQL did not recover" >&2; exit 1; }
  sleep 1
done
constraint_after=$(docker exec "$postgres" psql --username postgres --dbname interrupted --tuples-only --no-align \
  --command "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'collaboration_command_receipts_originating_adapter_check'")
[[ "$constraint_before" == "$constraint_after" ]]
table_after_crash=$(docker exec "$postgres" psql --username postgres --dbname interrupted --tuples-only --no-align \
  --command "SELECT to_regclass('public.collaboration_zed_community_bindings') IS NULL")
[[ "$table_after_crash" == "t" ]]
run_migration interrupted verify | grep -q 'current=20260825000100.*applied=21'
run_migration interrupted up
run_migration interrupted verify | grep -q 'current=20261026000100.*applied=31'
echo "in-flight migration crash, atomic rollback, and resume passed"

docker exec "$postgres" createdb --username postgres drift
run_migration drift up --target-version 20260815000100
cp crates/collab/migrations/*.up.sql crates/collab/migrations/*.down.sql "$drift_directory"
printf '\n-- checksum drift canary\n' >>"$drift_directory/20260815000100_collaboration_identity_bindings.up.sql"
if docker run --rm --label "$ownership_label" --network "$network" \
  --env "DATABASE_URL=postgres://postgres:migration-canary@${postgres}:5432/drift" \
  --env COLLABORATION_MIGRATION_DIRECTORY=/drift \
  --volume "$drift_directory:/drift:ro" \
  "$image" verify >/dev/null 2>&1; then
  echo "checksum drift did not halt the migration" >&2
  exit 1
fi
run_migration drift status | grep -q 'migration_status=halted.*halt_reason=checksum_drift'
if run_migration drift up >/dev/null 2>&1; then
  echo "a halted database resumed without operator recovery" >&2
  exit 1
fi

echo "collaboration migration smoke passed"
