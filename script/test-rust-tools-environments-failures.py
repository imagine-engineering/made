import os
import copy
import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent


class MatrixFailureTests(unittest.TestCase):
    def run_matrix(self, failure, physical=False, evidence_data=None, hung_target=None):
        with tempfile.TemporaryDirectory() as directory:
            cargo = Path(directory) / "cargo"
            # Stub Cargo outcomes while exercising the coordinator process-group cleanup.
            cargo.write_text('''#!/usr/bin/env python3
import os
import sys
import time
arguments = " ".join(sys.argv[1:])
if os.environ.get("AUDIT_FAIL") and os.environ["AUDIT_FAIL"] in arguments:
    sys.exit(17)
if os.environ.get("ZED_RUST_TOOLS_LONG_TEST") == "1":
    marker = os.environ.get("ZED_RUST_TOOLS_STARTED_MARKER")
    if marker:
        with open(marker, "w") as output:
            output.write("started")
    time.sleep(30)
if "rust_test_provider_isolates" in arguments:
    count = 0 if os.environ.get("AUDIT_FAIL") == "zero-tests" else 1
    print(f"test result: ok. {count} passed; 0 failed")
sys.exit(0)
''')
            cargo.chmod(0o755)
            environment = {
                key: value for key, value in os.environ.items()
                if not key.startswith("ZED_RUST_TOOLS_")
            }
            environment.update(
                PATH=directory + os.pathsep + environment["PATH"],
                AUDIT_FAIL=failure,
            )
            if hung_target is not None:
                target = Path(directory) / hung_target
                target.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n")
                target.chmod(0o755)
                environment["ZED_RUST_TOOLS_TARGET_TIMEOUT_SECONDS"] = "1"
                if hung_target == "ssh":
                    environment["ZED_RUST_TOOLS_SSH_HOST"] = "hung-host"
                    environment["ZED_RUST_TOOLS_SSH_REPO"] = "/fixture"
                elif hung_target == "docker":
                    environment["ZED_RUST_TOOLS_CONTAINER"] = "hung-container"
                    environment["ZED_RUST_TOOLS_CONTAINER_REPO"] = "/fixture"
            if evidence_data is not None:
                evidence_path = Path(directory) / "evidence.json"
                evidence_path.write_text(json.dumps(evidence_data))
                environment["ZED_RUST_TOOLS_PHYSICAL_EVIDENCE"] = str(evidence_path)
            if physical:
                environment["ZED_RUST_TOOLS_REQUIRE_PHYSICAL"] = "1"
            return subprocess.run(
                ["bash", str(ROOT / "script/test-rust-tools-environments"),
                 "--matrix", "--offline"],
                env=environment, capture_output=True, text=True,
                timeout=8 if hung_target is not None else 20,
            )

    def test_each_failure_is_reported(self):
        failures = ["--no-run", "tests::unit_passes", "--test integration",
                    "-- --ignored", "rust_test_provider_isolates", "zero-tests"]
        if os.name == "posix":
            failures.append("tests::cancellable")
        for failure in failures:
            with self.subTest(failure=failure):
                result = self.run_matrix(failure)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("local\tfailed\t", result.stdout)

    def test_automated_success_is_not_physical(self):
        result = self.run_matrix("")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("local\tautomated\t", result.stdout)
        self.assertIn("ssh\tunavailable\t", result.stdout)

    def test_required_physical_fails_closed(self):
        result = self.run_matrix("", physical=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("remain uncertified", result.stderr)
        self.assertNotIn("\tcertified\t", result.stdout)

    def test_unresponsive_transport_fails_its_cell(self):
        for command, cell in (("ssh", "ssh"), ("docker", "dev-container")):
            with self.subTest(command=command):
                result = self.run_matrix("", hung_target=command)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(f"{cell}\tfailed\t", result.stdout)
                self.assertIn("timed out", result.stderr)


class PhysicalEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        specification = importlib.util.spec_from_file_location(
            "rust_tools_evidence", ROOT / "script/test-rust-tools-evidence.py")
        cls.evidence = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(cls.evidence)

    def fixture(self):
        records = []
        for cell in sorted(self.evidence.REQUIRED | {"dev-container"}):
            records.append({
                "cell": cell, "capability": "supported", "tester": "synthetic tester", "reviewer": "synthetic reviewer",
                "review_approved": True, "date": "2026-09-12",
                "client_os": cell.removeprefix("local-") if cell.startswith("local-") else "macos",
                "host_os": cell.removeprefix("local-") if cell.startswith("local-") else "linux", "architecture": "arm64", "rust_version": "test-only", "dap_version": "test-only",
                "zed_build": "synthetic", "observation": "Synthetic fixture, not physical evidence",
                "read_only_denied": True, "peer_scoped_cancel": True,
                "checks": {name: {"status": "pass", "observation": "synthetic observation"}
                           for name in self.evidence.CHECKS},
            })
        return {"schema_version": 1, "policy": "rust-tools-physical-v1",
                "revision": "a" * 40, "worktree_diff_sha256": "b" * 64, "records": records}

    def validate(self, fixture):
        return self.evidence.validate(fixture, "a" * 40, "b" * 64, datetime.date(2026, 9, 12))

    def test_complete_synthetic_record_validates_mechanism_only(self):
        self.assertEqual(len(self.validate(self.fixture())), 6)

    def test_rejects_bad_documents(self):
        for value in (None, [], {}, {"schema_version": 2}, {"policy": "waive-all"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(value)

    def test_rejects_missing_stale_unreviewed_or_mismatched_records(self):
        changes = [
            lambda data: data.update(revision="b" * 40),
            lambda data: data.update(worktree_diff_sha256="c" * 64),
            lambda data: data["records"].pop(),
            lambda data: data["records"].append(copy.deepcopy(data["records"][0])),
            lambda data: data["records"][0].update(date="2026-01-01"),
            lambda data: data["records"][0].update(date="2026-10-01"),
            lambda data: data["records"][0].update(review_approved=False),
            lambda data: data["records"][0].update(reviewer=""),
            lambda data: data["records"][0]["checks"].pop("run"),
            lambda data: data["records"][0]["checks"]["run"].update(status="failed"),
            lambda data: data["records"][0]["checks"]["run"].update(status="unsupported"),
            lambda data: data["records"][0]["checks"]["run"].update(observation="x" * 2049),
            lambda data: data["records"].remove(next(record for record in data["records"]
                                                        if record["cell"] == "dev-container")),
            lambda data: next(record for record in data["records"]
                              if record["cell"] == "multiplayer").update(peer_scoped_cancel=False),
            lambda data: next(record for record in data["records"]
                              if record["cell"] == "ssh-linux").update(host_os="macos"),
        ]
        for index, change in enumerate(changes):
            data = self.fixture()
            change(data)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.validate(data)

    def test_optional_unsupported_is_explicit_and_cannot_waive_required_cells(self):
        data = self.fixture()
        unsupported = copy.deepcopy(data["records"][0])
        unsupported.update(cell="wsl", capability="unsupported", capability_reason="observed unsupported mode")
        data["records"].append(unsupported)
        self.assertEqual(self.validate(data)["wsl"][0], "unsupported")
        data["records"][0].update(capability="unsupported", capability_reason="not available")
        with self.assertRaises(ValueError):
            self.validate(data)

    def test_unsupported_debug_requires_capability_reason(self):
        data = self.fixture()
        outcome = data["records"][0]["checks"]["debug"]
        outcome["status"] = "unsupported"
        with self.assertRaises(ValueError):
            self.validate(data)
        outcome["capability_reason"] = "tested harness does not support debugging"
        self.validate(data)

    def test_cli_accepts_complete_synthetic_records_and_rejects_changed_snapshot(self):
        data = self.fixture()
        revision, digest = self.evidence.checkout_identity()
        data.update(revision=revision, worktree_diff_sha256=digest)
        for record in data["records"]:
            record["date"] = datetime.date.today().isoformat()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            for valid in (True, False):
                if not valid:
                    data["worktree_diff_sha256"] = "incorrect"
                path.write_text(json.dumps(data))
                result = subprocess.run([os.sys.executable,
                    str(ROOT / "script/test-rust-tools-evidence.py"), str(path),
                    "--revision", revision], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0 if valid else 1, result.stderr)
                self.assertEqual(result.stdout.count("\tcertified\t"), 6 if valid else 0)

    def test_coordinator_requires_valid_records_and_still_rejects_local_failure(self):
        data = self.fixture()
        revision, digest = self.evidence.checkout_identity()
        data.update(revision=revision, worktree_diff_sha256=digest)
        for record in data["records"]:
            record["date"] = datetime.date.today().isoformat()
        matrix = MatrixFailureTests()
        result = matrix.run_matrix("", physical=True, evidence_data=data)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("local\tautomated\t", result.stdout)
        self.assertEqual(result.stdout.count("\tcertified\t"), 6)
        result = matrix.run_matrix("--no-run", physical=True, evidence_data=data)
        self.assertEqual(result.returncode, 1)
        data["revision"] = "invalid"
        result = matrix.run_matrix("", physical=True, evidence_data=data)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("\tcertified\t", result.stdout)

    def test_cli_rejects_malformed_and_oversized_evidence(self):
        for data in (b"{", b" " * (self.evidence.MAX_BYTES + 1)):
            with tempfile.TemporaryDirectory() as directory:
                evidence = Path(directory) / "evidence.json"
                evidence.write_bytes(data)
                result = subprocess.run([os.sys.executable,
                    str(ROOT / "script/test-rust-tools-evidence.py"), str(evidence),
                    "--revision", self.evidence.checkout_identity()[0]], capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                self.assertNotIn("certified", result.stdout)


if __name__ == "__main__":
    unittest.main()
