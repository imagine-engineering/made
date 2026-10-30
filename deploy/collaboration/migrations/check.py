#!/usr/bin/env python3

import hashlib
import json
import os
import pathlib
import re
import subprocess
import tempfile


DIRECTORY = pathlib.Path(__file__).resolve().parent
REPOSITORY = DIRECTORY.parents[2]
MIGRATIONS = REPOSITORY / "crates/collab/migrations"
MANIFEST = DIRECTORY / "manifest.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    document = json.loads(MANIFEST.read_text())
    require(document["formatVersion"] == 1, "migration format changed")
    require(document["schemaFloor"] == "0", "schema floor changed")
    require(document["schemaCeiling"] == "20261026000100", "schema ceiling changed")
    require(document["rollbackBoundary"] == "service-activation", "rollback boundary changed")
    migrations = document["migrations"]
    require(len(migrations) == 31, "migration inventory changed")
    versions = [migration["version"] for migration in migrations]
    require(versions == sorted(set(versions)), "migration order is not strict")
    require(versions[-1] == document["schemaCeiling"], "ceiling is not the final migration")

    expected_files = set()
    for migration in migrations:
        version = migration["version"]
        name = migration["name"]
        require(re.fullmatch(r"[0-9]{14}", version) is not None, "invalid migration version")
        require(re.fullmatch(r"[a-z][a-z0-9_]{0,127}", name) is not None, "invalid migration name")
        for direction in ("up", "down"):
            entry = migration[direction]
            expected_name = f"{version}_{name}.{direction}.sql"
            require(entry["file"] == expected_name, "migration filename drifted")
            expected_files.add(expected_name)
            digest = hashlib.sha256((MIGRATIONS / expected_name).read_bytes()).hexdigest()
            require(entry["sha256"] == digest, f"checksum drifted for {expected_name}")

    actual_files = {
        path.name
        for pattern in ("*.up.sql", "*.down.sql")
        for path in MIGRATIONS.glob(pattern)
    }
    require(actual_files == expected_files, "manifest does not cover every SQL pair")

    validation = subprocess.run(
        [
            "python3",
            str(DIRECTORY / "migrate.py"),
            "--manifest",
            str(MANIFEST),
            "--migration-directory",
            str(MIGRATIONS),
            "validate",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    require("count=31 ceiling=20261026000100" in validation.stdout, "runner rejected manifest")

    mismatched_environment = os.environ.copy()
    mismatched_environment["COLLABORATION_REQUIRED_SCHEMA_VERSION"] = "20260824000400"
    mismatch = subprocess.run(
        [
            "python3",
            str(DIRECTORY / "migrate.py"),
            "--manifest",
            str(MANIFEST),
            "--migration-directory",
            str(MIGRATIONS),
            "validate",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=mismatched_environment,
    )
    require(mismatch.returncode == 1, "required schema mismatch was accepted")
    require(
        "migration_error=required_schema_version_mismatch" in mismatch.stderr,
        "required schema mismatch class changed",
    )

    invalid_database_environment = os.environ.copy()
    invalid_database_environment["DATABASE_URL"] = "sqlite:///collaboration"
    invalid_database_url = subprocess.run(
        [
            "python3",
            str(DIRECTORY / "migrate.py"),
            "--manifest",
            str(MANIFEST),
            "--migration-directory",
            str(MIGRATIONS),
            "status",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=invalid_database_environment,
    )
    require(invalid_database_url.returncode == 1, "invalid database URL was accepted")
    require(
        "migration_error=database_url_invalid" in invalid_database_url.stderr,
        "database URL error class changed",
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        copied_migrations = pathlib.Path(temporary_directory)
        for file in expected_files:
            (copied_migrations / file).write_bytes((MIGRATIONS / file).read_bytes())
        changed = copied_migrations / migrations[0]["up"]["file"]
        changed.write_bytes(changed.read_bytes() + b"\n-- checksum drift canary\n")
        drift = subprocess.run(
            [
                "python3",
                str(DIRECTORY / "migrate.py"),
                "--manifest",
                str(MANIFEST),
                "--migration-directory",
                str(copied_migrations),
                "validate",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        require(drift.returncode == 1, "changed SQL was accepted")
        require("migration_error=migration_checksum_drift" in drift.stderr, "drift class changed")

    chart_directory = REPOSITORY / "deploy/collaboration/charts/collaboration"
    chart_text = "\n".join(
        path.read_text()
        for path in (
            chart_directory / "values.yaml",
            chart_directory / "values-production.yaml",
            chart_directory / "values.schema.json",
            chart_directory / "templates/_helpers.tpl",
            chart_directory / "templates/migration-job.yaml",
        )
    )
    for contract in (
        "collaboration-migrations",
        "migration.image.digest",
        "collaboration.migrationImage",
        "args: [\"up\"]",
        "COLLABORATION_REQUIRED_SCHEMA_VERSION",
    ):
        require(contract in chart_text, f"chart migration contract missing {contract}")
    print("collaboration migration package checks passed")


if __name__ == "__main__":
    main()
