#!/usr/bin/env python3

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import NoReturn
from urllib.parse import parse_qsl, unquote, urlsplit


FORMAT_VERSION = 1
LOCK_ID = 885_426_044_004
CONTROL_TABLE = "public.collaboration_schema_migration_control"
HISTORY_TABLE = "public.collaboration_schema_migration_history"
VERSION_PATTERN = re.compile(r"^[0-9]{14}$")
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class MigrationError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclasses.dataclass(frozen=True)
class Direction:
    file: str
    sha256: str


@dataclasses.dataclass(frozen=True)
class Migration:
    version: str
    name: str
    up: Direction
    down: Direction


@dataclasses.dataclass(frozen=True)
class Manifest:
    schema_floor: str
    schema_ceiling: str
    rollback_boundary: str
    migrations: tuple[Migration, ...]


@dataclasses.dataclass(frozen=True)
class AppliedMigration:
    version: str
    name: str
    up_sha256: str
    down_sha256: str


@dataclasses.dataclass(frozen=True)
class DatabaseState:
    status: str
    current_version: str
    rollback_floor: str
    control_version: int
    halt_reason: str | None
    applied: tuple[AppliedMigration, ...]


def fail(code: str) -> NoReturn:
    raise MigrationError(code)


def require_exact_keys(value: dict, expected: set[str], code: str) -> None:
    if set(value) != expected:
        fail(code)


def load_direction(value: object, suffix: str) -> Direction:
    if not isinstance(value, dict):
        fail("manifest_invalid")
    require_exact_keys(value, {"file", "sha256"}, "manifest_invalid")
    file = value["file"]
    sha256 = value["sha256"]
    if (
        not isinstance(file, str)
        or pathlib.PurePath(file).name != file
        or not file.endswith(suffix)
        or not isinstance(sha256, str)
        or SHA256_PATTERN.fullmatch(sha256) is None
    ):
        fail("manifest_invalid")
    return Direction(file=file, sha256=sha256)


def load_manifest(manifest_path: pathlib.Path, migration_directory: pathlib.Path) -> Manifest:
    try:
        raw = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        fail("manifest_invalid")
    if not isinstance(raw, dict):
        fail("manifest_invalid")
    require_exact_keys(
        raw,
        {"formatVersion", "schemaFloor", "schemaCeiling", "rollbackBoundary", "migrations"},
        "manifest_invalid",
    )
    if raw["formatVersion"] != FORMAT_VERSION or raw["schemaFloor"] != "0":
        fail("manifest_invalid")
    if (
        not isinstance(raw["schemaCeiling"], str)
        or VERSION_PATTERN.fullmatch(raw["schemaCeiling"]) is None
        or raw["rollbackBoundary"] != "service-activation"
        or not isinstance(raw["migrations"], list)
        or not raw["migrations"]
    ):
        fail("manifest_invalid")

    migrations = []
    for value in raw["migrations"]:
        if not isinstance(value, dict):
            fail("manifest_invalid")
        require_exact_keys(value, {"version", "name", "up", "down"}, "manifest_invalid")
        version = value["version"]
        name = value["name"]
        if (
            not isinstance(version, str)
            or VERSION_PATTERN.fullmatch(version) is None
            or not isinstance(name, str)
            or NAME_PATTERN.fullmatch(name) is None
        ):
            fail("manifest_invalid")
        up = load_direction(value["up"], ".up.sql")
        down = load_direction(value["down"], ".down.sql")
        stem = f"{version}_{name}"
        if up.file != f"{stem}.up.sql" or down.file != f"{stem}.down.sql":
            fail("manifest_invalid")
        migrations.append(Migration(version=version, name=name, up=up, down=down))

    versions = [migration.version for migration in migrations]
    if versions != sorted(set(versions)) or versions[-1] != raw["schemaCeiling"]:
        fail("manifest_invalid")

    expected_files = {
        direction.file
        for migration in migrations
        for direction in (migration.up, migration.down)
    }
    actual_files = {
        path.name
        for pattern in ("*.up.sql", "*.down.sql")
        for path in migration_directory.glob(pattern)
    }
    if actual_files != expected_files:
        fail("migration_inventory_drift")
    for migration in migrations:
        for direction in (migration.up, migration.down):
            try:
                digest = hashlib.sha256((migration_directory / direction.file).read_bytes()).hexdigest()
            except OSError:
                fail("migration_inventory_drift")
            if digest != direction.sha256:
                fail("migration_checksum_drift")

    return Manifest(
        schema_floor=raw["schemaFloor"],
        schema_ceiling=raw["schemaCeiling"],
        rollback_boundary=raw["rollbackBoundary"],
        migrations=tuple(migrations),
    )


def database_environment(database_url: str) -> dict[str, str]:
    try:
        parsed = urlsplit(database_url)
        port = parsed.port
        query = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ValueError:
        fail("database_url_invalid")
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or parsed.hostname is None
        or not parsed.path.startswith("/")
        or parsed.path == "/"
        or parsed.fragment
    ):
        fail("database_url_invalid")

    option_environment_names = {
        "application_name": "PGAPPNAME",
        "channel_binding": "PGCHANNELBINDING",
        "connect_timeout": "PGCONNECT_TIMEOUT",
        "gssencmode": "PGGSSENCMODE",
        "krbsrvname": "PGKRBSRVNAME",
        "options": "PGOPTIONS",
        "requirepeer": "PGREQUIREPEER",
        "sslcert": "PGSSLCERT",
        "sslcrl": "PGSSLCRL",
        "sslcrldir": "PGSSLCRLDIR",
        "sslkey": "PGSSLKEY",
        "ssl_max_protocol_version": "PGSSLMAXPROTOCOLVERSION",
        "ssl_min_protocol_version": "PGSSLMINPROTOCOLVERSION",
        "sslmode": "PGSSLMODE",
        "sslrootcert": "PGSSLROOTCERT",
        "sslsni": "PGSSLSNI",
        "target_session_attrs": "PGTARGETSESSIONATTRS",
    }
    if len(query) != len(dict(query)) or any(key not in option_environment_names for key, _ in query):
        fail("database_url_invalid")

    environment = os.environ.copy()
    for name in {
        "PGDATABASE",
        "PGHOST",
        "PGPASSWORD",
        "PGPORT",
        "PGUSER",
        *option_environment_names.values(),
    }:
        environment.pop(name, None)
    environment["PGHOST"] = parsed.hostname
    environment["PGDATABASE"] = unquote(parsed.path[1:])
    if parsed.username is not None:
        environment["PGUSER"] = unquote(parsed.username)
    if parsed.password is not None:
        environment["PGPASSWORD"] = unquote(parsed.password)
    if port is not None:
        environment["PGPORT"] = str(port)
    for key, value in query:
        environment[option_environment_names[key]] = value
    environment.setdefault("PGCONNECT_TIMEOUT", "10")
    environment.pop("DATABASE_URL", None)
    return environment


def run_psql(database_url: str, sql: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [
                "psql",
                "--no-psqlrc",
                "--quiet",
                "--tuples-only",
                "--no-align",
                "--field-separator=\t",
                "--set=ON_ERROR_STOP=1",
            ],
            input=sql,
            text=True,
            capture_output=True,
            env=database_environment(database_url),
            check=False,
        )
    except OSError:
        fail("psql_unavailable")


def require_psql(database_url: str, sql: str, code: str) -> str:
    result = run_psql(database_url, sql)
    if result.returncode != 0:
        fail(code)
    return result.stdout


def initialize_database(database_url: str) -> None:
    require_psql(
        database_url,
        f"""
CREATE TABLE IF NOT EXISTS {CONTROL_TABLE} (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    format_version integer NOT NULL CHECK (format_version = {FORMAT_VERSION}),
    status text NOT NULL CHECK (status IN ('ready', 'halted')),
    current_version numeric(20, 0) NOT NULL CHECK (current_version >= 0),
    rollback_floor numeric(20, 0) NOT NULL CHECK (
        rollback_floor >= 0 AND rollback_floor <= current_version
    ),
    control_version bigint NOT NULL CHECK (control_version > 0),
    halt_reason text CHECK (
        halt_reason IS NULL OR halt_reason IN (
            'checksum_drift', 'history_drift', 'execution_failure'
        )
    ),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CHECK ((status = 'ready') = (halt_reason IS NULL))
);
CREATE TABLE IF NOT EXISTS {HISTORY_TABLE} (
    version numeric(20, 0) PRIMARY KEY CHECK (version > 0),
    name text NOT NULL CHECK (
        octet_length(name) BETWEEN 1 AND 128
        AND name = lower(name)
    ),
    up_sha256 text NOT NULL CHECK (up_sha256 ~ '^[0-9a-f]{{64}}$'),
    down_sha256 text NOT NULL CHECK (down_sha256 ~ '^[0-9a-f]{{64}}$'),
    applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
REVOKE ALL ON {CONTROL_TABLE} FROM PUBLIC;
REVOKE ALL ON {HISTORY_TABLE} FROM PUBLIC;
INSERT INTO {CONTROL_TABLE} (
    singleton, format_version, status, current_version, rollback_floor,
    control_version, halt_reason
) VALUES (true, {FORMAT_VERSION}, 'ready', 0, 0, 1, NULL)
ON CONFLICT (singleton) DO NOTHING;
""",
        "metadata_initialization_failed",
    )


def read_database_state(database_url: str) -> DatabaseState:
    output = require_psql(
        database_url,
        f"""
SELECT 'CONTROL', status, current_version::text, rollback_floor::text,
       control_version::text, COALESCE(halt_reason, '')
FROM {CONTROL_TABLE} WHERE singleton;
SELECT 'HISTORY', version::text, name, up_sha256, down_sha256
FROM {HISTORY_TABLE} ORDER BY version;
""",
        "metadata_read_failed",
    )
    control = None
    history = []
    for line in output.splitlines():
        fields = line.split("\t")
        if fields[0] == "CONTROL" and len(fields) == 6:
            if control is not None:
                fail("history_drift")
            control = fields
        elif fields[0] == "HISTORY" and len(fields) == 5:
            history.append(
                AppliedMigration(
                    version=fields[1],
                    name=fields[2],
                    up_sha256=fields[3],
                    down_sha256=fields[4],
                )
            )
        elif line:
            fail("history_drift")
    if control is None:
        fail("history_drift")
    try:
        control_version = int(control[4])
    except ValueError:
        fail("history_drift")
    return DatabaseState(
        status=control[1],
        current_version=control[2],
        rollback_floor=control[3],
        control_version=control_version,
        halt_reason=control[5] or None,
        applied=tuple(history),
    )


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def halt(database_url: str, reason: str) -> None:
    if reason not in {"checksum_drift", "history_drift", "execution_failure"}:
        reason = "execution_failure"
    run_psql(
        database_url,
        f"""
UPDATE {CONTROL_TABLE}
SET status = 'halted', halt_reason = {sql_string(reason)},
    control_version = control_version + 1, updated_at = clock_timestamp()
WHERE singleton AND status = 'ready';
""",
    )


def verify_history(manifest: Manifest, state: DatabaseState) -> None:
    if state.status == "halted":
        fail("migration_halted")
    if state.status != "ready" or state.control_version <= 0:
        fail("history_drift")
    expected = manifest.migrations[: len(state.applied)]
    if len(state.applied) > len(manifest.migrations):
        fail("history_drift")
    for applied, migration in zip(state.applied, expected, strict=True):
        if (
            applied.version != migration.version
            or applied.name != migration.name
            or applied.up_sha256 != migration.up.sha256
            or applied.down_sha256 != migration.down.sha256
        ):
            fail("history_drift")
    expected_current = expected[-1].version if expected else manifest.schema_floor
    if state.current_version != expected_current:
        fail("history_drift")
    if int(state.rollback_floor) > int(state.current_version):
        fail("history_drift")
    if state.rollback_floor != manifest.schema_floor and state.rollback_floor not in {
        migration.version for migration in expected
    }:
        fail("history_drift")


def validated_state(database_url: str, manifest: Manifest) -> DatabaseState:
    state = read_database_state(database_url)
    try:
        verify_history(manifest, state)
    except MigrationError as error:
        if error.code == "history_drift":
            halt(database_url, "history_drift")
        raise
    return state


def target_index(manifest: Manifest, target: str) -> int:
    if target == manifest.schema_floor:
        return 0
    for index, migration in enumerate(manifest.migrations, start=1):
        if migration.version == target:
            return index
    fail("target_version_invalid")


def state_guard(state: DatabaseState) -> str:
    return f"""
DO $state_guard$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM {CONTROL_TABLE}
        WHERE singleton
          AND status = 'ready'
          AND current_version = {state.current_version}
          AND rollback_floor = {state.rollback_floor}
          AND control_version = {state.control_version}
    ) THEN
        RAISE EXCEPTION 'migration state changed';
    END IF;
END
$state_guard$;
"""


def execute_plan(database_url: str, sql: str) -> None:
    result = run_psql(database_url, sql)
    if result.returncode == 0:
        return
    if "migration lock unavailable" in result.stderr or "migration state changed" in result.stderr:
        fail("concurrent_migration")
    halt(database_url, "execution_failure")
    fail("migration_execution_failed")


def lock_sql() -> str:
    return f"""
DO $migration_lock$
BEGIN
    IF NOT pg_try_advisory_lock({LOCK_ID}) THEN
        RAISE EXCEPTION 'migration lock unavailable';
    END IF;
END
$migration_lock$;
"""


def up(database_url: str, manifest: Manifest, migration_directory: pathlib.Path, target: str) -> None:
    state = validated_state(database_url, manifest)
    current_index = len(state.applied)
    requested_index = target_index(manifest, target)
    if requested_index < current_index:
        fail("target_requires_down")
    pending = manifest.migrations[current_index:requested_index]
    if not pending:
        print(f"migration_up_applied=0 current={state.current_version}")
        return

    statements = [lock_sql(), state_guard(state)]
    for migration in pending:
        statements.extend(
            [
                "BEGIN;\nSET LOCAL lock_timeout = '10s';\nSET LOCAL statement_timeout = '240s';",
                (migration_directory / migration.up.file).read_text(),
                f"""
INSERT INTO {HISTORY_TABLE} (version, name, up_sha256, down_sha256)
VALUES (
    {migration.version}, {sql_string(migration.name)},
    {sql_string(migration.up.sha256)}, {sql_string(migration.down.sha256)}
);
UPDATE {CONTROL_TABLE}
SET current_version = {migration.version}, control_version = control_version + 1,
    updated_at = clock_timestamp()
WHERE singleton;
COMMIT;
""",
            ]
        )
    statements.append(f"SELECT pg_advisory_unlock({LOCK_ID});")
    execute_plan(database_url, "\n".join(statements))
    final_state = validated_state(database_url, manifest)
    if final_state.current_version != target:
        fail("history_drift")
    print(f"migration_up_applied={len(pending)} current={target}")


def down(database_url: str, manifest: Manifest, migration_directory: pathlib.Path, target: str) -> None:
    state = validated_state(database_url, manifest)
    current_index = len(state.applied)
    requested_index = target_index(manifest, target)
    if requested_index > current_index:
        fail("target_requires_up")
    if int(target) < int(state.rollback_floor):
        fail("rollback_boundary_crossed")
    pending = list(reversed(manifest.migrations[requested_index:current_index]))
    if not pending:
        print(f"migration_down_applied=0 current={state.current_version}")
        return

    statements = [lock_sql(), state_guard(state)]
    versions = [migration.version for migration in manifest.migrations]
    for migration in pending:
        previous_index = versions.index(migration.version)
        previous_version = manifest.schema_floor if previous_index == 0 else versions[previous_index - 1]
        statements.extend(
            [
                "BEGIN;\nSET LOCAL lock_timeout = '10s';\nSET LOCAL statement_timeout = '240s';",
                (migration_directory / migration.down.file).read_text(),
                f"""
DELETE FROM {HISTORY_TABLE} WHERE version = {migration.version};
UPDATE {CONTROL_TABLE}
SET current_version = {previous_version}, control_version = control_version + 1,
    updated_at = clock_timestamp()
WHERE singleton;
COMMIT;
""",
            ]
        )
    statements.append(f"SELECT pg_advisory_unlock({LOCK_ID});")
    execute_plan(database_url, "\n".join(statements))
    final_state = validated_state(database_url, manifest)
    if final_state.current_version != target:
        fail("history_drift")
    print(f"migration_down_applied={len(pending)} current={target}")


def seal(database_url: str, manifest: Manifest, expected_version: str) -> None:
    state = validated_state(database_url, manifest)
    if state.current_version != expected_version:
        fail("seal_version_mismatch")
    result = run_psql(
        database_url,
        f"""
{lock_sql()}
{state_guard(state)}
UPDATE {CONTROL_TABLE}
SET rollback_floor = current_version, control_version = control_version + 1,
    updated_at = clock_timestamp()
WHERE singleton;
SELECT pg_advisory_unlock({LOCK_ID});
""",
    )
    if result.returncode != 0:
        fail("concurrent_migration")
    final_state = validated_state(database_url, manifest)
    if final_state.rollback_floor != expected_version:
        fail("history_drift")
    print(f"migration_rollback_floor={expected_version}")


def print_status(state: DatabaseState) -> None:
    reason = state.halt_reason or "none"
    print(
        f"migration_status={state.status} current={state.current_version} "
        f"rollback_floor={state.rollback_floor} applied={len(state.applied)} "
        f"halt_reason={reason}"
    )


def parse_arguments() -> argparse.Namespace:
    script_directory = pathlib.Path(__file__).resolve().parent
    packaged_migrations = pathlib.Path("/migrations")
    if packaged_migrations.is_dir():
        default_migrations = packaged_migrations
    else:
        default_migrations = script_directory.parents[2] / "crates/collab/migrations"
    default_migrations = pathlib.Path(
        os.environ.get(
            "COLLABORATION_MIGRATION_DIRECTORY",
            default_migrations,
        )
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=script_directory / "manifest.json")
    parser.add_argument("--migration-directory", type=pathlib.Path, default=default_migrations)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("plan")
    up_parser = subparsers.add_parser("up")
    up_parser.add_argument("--target-version")
    down_parser = subparsers.add_parser("down")
    down_parser.add_argument("--target-version", required=True)
    subparsers.add_parser("verify")
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--expected-version", required=True)
    subparsers.add_parser("status")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    database_url = os.environ.get("DATABASE_URL", "")
    try:
        manifest = load_manifest(arguments.manifest, arguments.migration_directory)
    except MigrationError as error:
        if arguments.command != "validate" and database_url:
            try:
                initialize_database(database_url)
                halt(database_url, "checksum_drift")
            except MigrationError:
                pass
        print(f"migration_error={error.code}", file=sys.stderr)
        return 1

    required_schema_version = os.environ.get("COLLABORATION_REQUIRED_SCHEMA_VERSION")
    if required_schema_version and required_schema_version != manifest.schema_ceiling:
        print("migration_error=required_schema_version_mismatch", file=sys.stderr)
        return 1
    if arguments.command == "validate":
        print(
            f"migration_manifest_valid count={len(manifest.migrations)} "
            f"ceiling={manifest.schema_ceiling}"
        )
        return 0
    if arguments.command == "plan":
        for migration in manifest.migrations:
            print(f"{migration.version}\t{migration.name}")
        return 0
    if not database_url:
        print("migration_error=database_url_missing", file=sys.stderr)
        return 1

    try:
        initialize_database(database_url)
        if arguments.command == "up":
            up(
                database_url,
                manifest,
                arguments.migration_directory,
                arguments.target_version or manifest.schema_ceiling,
            )
        elif arguments.command == "down":
            down(database_url, manifest, arguments.migration_directory, arguments.target_version)
        elif arguments.command == "verify":
            state = validated_state(database_url, manifest)
            print_status(state)
        elif arguments.command == "seal":
            target_index(manifest, arguments.expected_version)
            seal(database_url, manifest, arguments.expected_version)
        elif arguments.command == "status":
            print_status(read_database_state(database_url))
        else:
            fail("command_invalid")
    except MigrationError as error:
        print(f"migration_error={error.code}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
