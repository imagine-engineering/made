"""Check disposable-harness ownership; Docker is a command-recording fixture here."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
SMOKE = Path(os.environ.get("MIGRATION_SMOKE_UNDER_TEST", Path(__file__).with_name("smoke.sh")))


class MigrationOwnershipTests(unittest.TestCase):
    def run_harness(self, stage):
        with tempfile.TemporaryDirectory(prefix="migration-ownership-") as temporary:
            temporary = Path(temporary)
            calls = temporary / "calls.bin"
            docker = temporary / "docker"
            docker.write_text(
                '#!/bin/bash\n'
                'printf "%s\\0" "$@" >> "$DOCKER_TEST_CALLS"\n'
                'printf "\\n" >> "$DOCKER_TEST_CALLS"\n'
                'if [[ "$1 $2" == "network create" ]]; then\n'
                '  [[ "$DOCKER_TEST_STAGE" != network ]] || exit 42\n'
                'elif [[ "$1" == run ]]; then\n'
                '  exit 43\n'
                'elif [[ "$1 $2" == "network rm" && "$DOCKER_TEST_STAGE" == cleanup ]]; then\n'
                '  exit 44\n'
                'fi\n'
                'exit 0\n'
            )
            docker.chmod(0o700)
            environment = os.environ.copy()
            environment.update(
                PATH=str(temporary) + os.pathsep + environment["PATH"],
                TMPDIR=str(temporary),
                DOCKER_TEST_CALLS=str(calls),
                DOCKER_TEST_STAGE=stage,
            )
            result = subprocess.run(
                ["bash", str(SMOKE)], cwd=ROOT, env=environment,
                capture_output=True, text=True, timeout=10,
            )
            recorded = [record.decode().split("\0") for record in calls.read_bytes().split(b"\0\n") if record]
            self.assertFalse(list(temporary.glob("zed-collaboration-migrations-*")))
            return result, recorded

    def test_cleanup_failure_is_reported(self):
        result, calls = self.run_harness("cleanup")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("could not remove owned migration network", result.stderr)
        self.assertEqual(calls[-1], ["network", "rm", calls[0][-1]])

    def test_failed_network_creation_never_removes_unowned_resources(self):
        result, calls = self.run_harness("network")
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertEqual(len(calls), 1, calls)
        self.assertEqual(calls[0][:2], ["network", "create"])
        self.assertIn("--label", calls[0])

    def test_failed_container_creation_removes_only_its_unique_network(self):
        networks = []
        for _ in range(2):
            result, calls = self.run_harness("container")
            self.assertEqual(result.returncode, 43, result.stderr)
            self.assertEqual(len(calls), 3, calls)
            created_network = calls[0][-1]
            networks.append(created_network)
            self.assertEqual(calls[2], ["network", "rm", created_network])
            for call in calls[:2]:
                label = call[call.index("--label") + 1]
                self.assertEqual(label, "dev.simtropolis.migration-test=" + created_network)
            self.assertIn("migration_test_run_id=" + created_network, result.stdout)
        self.assertNotEqual(*networks, "runs from the same parent must have distinct ownership")


if __name__ == "__main__":
    unittest.main()
