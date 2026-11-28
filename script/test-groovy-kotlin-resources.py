#!/usr/bin/env python3

import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


REPOSITORY = Path(__file__).resolve().parent.parent


def main():
    if sys.platform != "darwin" and not sys.platform.startswith("linux"):
        raise RuntimeError("The owned-process resource check requires macOS or Linux ps")
    module_spec = importlib.util.spec_from_file_location(
        "groovy_kotlin_provider", REPOSITORY / "script/test-groovy-kotlin-provider.py"
    )
    probe = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(probe)
    processes = []

    def start(source):
        process = subprocess.Popen(
            [sys.executable, "-c", source],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        processes.append(process)
        if process.stdout.readline().strip() != "ready":
            raise AssertionError("The owned resource fixture did not start")
        return process

    gradle_fixture_ids = set()
    real_check_output = subprocess.check_output

    def fixture_inventory(*arguments, **keywords):
        inventory = real_check_output(*arguments, **keywords)
        return "\n".join(
            f"{' '.join(fields[:9])} /fixture/bin/java -Xmx512m {probe.GRADLE_DAEMON_ENTRYPOINT}"
            if int(fields[0]) in gradle_fixture_ids else line
            for line in inventory.splitlines()
            if (fields := line.split(maxsplit=9))
        )

    def register_daemon(directory, process, started):
        stamp = started.isoformat(timespec="milliseconds")
        registry = directory.parent.resolve()
        (directory / f"daemon-{process.pid}.out.log").write_text(
            f"{stamp} [INFO] [org.gradle.launcher.daemon.server.Daemon] "
            "start() called on daemon - DefaultDaemonContext[uid=fixture,"
            f"daemonRegistryDir={registry},pid={process.pid},idleTimeout=10800000]\n"
        )

    def peak_processes(resources, included, excluded):
        processes = resources["processesAtSampledCombinedPeak"]
        identifiers = [process["processId"] for process in processes]
        assert identifiers == sorted(set(identifiers)), processes
        assert included <= set(identifiers), processes
        assert excluded.isdisjoint(identifiers), processes
        assert sum(process["residentBytes"] for process in processes) == resources[
            "sampledPeakCombinedResidentBytes"
        ], resources
        return {process["processId"]: process for process in processes}

    try:
        # Keep the real process table and RSS, independent of whether Python rewrites argv[0].
        with tempfile.TemporaryDirectory(prefix="groovy-resource-control-") as directory, mock.patch.object(
            probe.subprocess, "check_output", side_effect=fixture_inventory
        ):
            home = Path(directory)
            daemon_directory = home / "daemon/8.13"
            daemon_directory.mkdir(parents=True)
            child_source = 'import time; allocation=b"x"*(30*1024*1024); time.sleep(60)'
            server = start(
                'import subprocess,sys,time; allocation=b"x"*(20*1024*1024); '
                f'subprocess.Popen([sys.executable,"-c",{child_source!r}]); '
                'print("ready",flush=True); time.sleep(60)'
            )
            daemon = start(
                'import time; allocation=b"x"*(40*1024*1024); '
                'print("ready",flush=True); time.sleep(60)'
            )
            register_daemon(daemon_directory, daemon, datetime.now(timezone.utc))
            stale_daemon = start(
                'import time; allocation=b"x"*(40*1024*1024); '
                'print("ready",flush=True); time.sleep(60)'
            )
            register_daemon(daemon_directory, stale_daemon, datetime.now(timezone.utc) - timedelta(days=1))
            gradle_fixture_ids.update((daemon.pid, stale_daemon.pid))
            unrelated = start(
                'import time; allocation=b"x"*(256*1024*1024); '
                'print("ready",flush=True); time.sleep(60)'
            )
            (daemon_directory / f"daemon-{unrelated.pid}.out.log").write_text("stale registry PID")
            sampler = probe.KotlinResourceSampler(server, home, interval=0.1)
            time.sleep(0.8)
            resources = sampler.finish()
            assert resources["successfulSamples"] >= 2, resources
            assert resources["observedIsolatedGradleDaemonPids"] == [daemon.pid], resources
            assert resources["unverifiedIsolatedGradleRegistryPids"] == sorted([unrelated.pid, stale_daemon.pid]), resources
            assert resources["sampledPeakOwnedProcessCount"] == 3, resources
            owned = peak_processes(resources, {server.pid, daemon.pid}, {unrelated.pid, stale_daemon.pid})
            assert owned[daemon.pid]["ownership"] == ["gradle"], owned
            assert owned[server.pid]["ownership"] == ["server"], owned
            children = set(owned) - {server.pid, daemon.pid}
            assert len(children) == 1 and owned[children.pop()]["parentProcessId"] == server.pid, owned
            assert 65 * 1024 * 1024 < resources["sampledPeakCombinedResidentBytes"] < 200 * 1024 * 1024, resources
            assert resources["sampledPeakIsolatedGradleResidentBytes"] > 25 * 1024 * 1024, resources
            assert resources["sampledPeakServerGroupResidentBytes"] > 35 * 1024 * 1024, resources

            editor = start(
                'import time; allocation=b"x"*(20*1024*1024); '
                'print("ready",flush=True); time.sleep(60)'
            )
            sampler = probe.KotlinResourceSampler(
                server, home, interval=0.1, editor_process_id=editor.pid
            )
            time.sleep(0.4)
            editor_resources = sampler.finish()
            assert editor_resources["explicitEditorProcessId"] == editor.pid, editor_resources
            assert editor_resources["sampledPeakOwnedProcessCount"] == 4, editor_resources
            editor_peak = peak_processes(editor_resources, {server.pid, daemon.pid, editor.pid}, {unrelated.pid, stale_daemon.pid})
            assert editor_peak[editor.pid]["ownership"] == ["editor"], editor_peak
            assert editor_resources["sampledPeakEditorTreeResidentBytes"] > 15 * 1024 * 1024, editor_resources
            assert editor_resources["sampledPeakCombinedResidentBytes"] < 230 * 1024 * 1024, editor_resources

            sampler = probe.KotlinResourceSampler(
                server, home, interval=0.1, editor_process_id=server.pid
            )
            time.sleep(0.4)
            overlapping_resources = sampler.finish()
            assert overlapping_resources["sampledPeakOwnedProcessCount"] == 3, overlapping_resources
            overlap_peak = peak_processes(overlapping_resources, {server.pid, daemon.pid}, {unrelated.pid, stale_daemon.pid, editor.pid})
            assert overlap_peak[server.pid]["ownership"] == ["server", "editor"], overlap_peak
            assert overlapping_resources["sampledPeakCombinedResidentBytes"] < 200 * 1024 * 1024, overlapping_resources
            assert overlapping_resources["sampledPeakEditorTreeResidentBytes"] == overlapping_resources["sampledPeakServerGroupResidentBytes"], overlapping_resources

            sampler = probe.KotlinResourceSampler(
                server, home, interval=0.1, editor_process_id=editor.pid
            )
            editor.terminate()
            editor.wait(timeout=5)
            time.sleep(0.3)
            try:
                sampler.finish()
            except RuntimeError as error:
                assert isinstance(error.__cause__, RuntimeError), error
                assert "editor process" in str(error.__cause__), error
            else:
                raise AssertionError("An editor that exited during sampling was silently excluded")
            try:
                probe.KotlinResourceSampler(server, home, editor_process_id=editor.pid)
            except RuntimeError as error:
                assert "editor process" in str(error), error
            else:
                raise AssertionError("An exited editor process was silently excluded")

            sampler = probe.KotlinResourceSampler(server, home, interval=0.1)
            inventory_attempted = threading.Event()

            def deny_inventory(*arguments, **keywords):
                inventory_attempted.set()
                raise PermissionError("denied numeric inventory")

            with mock.patch.object(probe.subprocess, "check_output", side_effect=deny_inventory):
                assert inventory_attempted.wait(timeout=5), "Sampler did not attempt numeric inventory"
                try:
                    sampler.finish()
                except RuntimeError as error:
                    assert isinstance(error.__cause__, PermissionError), error
                else:
                    raise AssertionError("A denied resource inventory was silently accepted")

            (daemon_directory / "daemon-invalid.out.log").write_text("invalid fixture PID")
            try:
                probe.isolated_gradle_daemon_process_ids(home)
            except AssertionError:
                pass
            else:
                raise AssertionError("Malformed isolated daemon PID was accepted")
            print(json.dumps({
                "ownedServerAndChild": server.pid,
                "ownedDetachedDaemon": daemon.pid,
                "unrelatedExcluded": unrelated.pid,
                "staleGradlePidExcluded": stale_daemon.pid,
                "resources": resources,
                "editorResources": editor_resources,
                "overlappingResources": overlapping_resources,
            }, indent=2))
    finally:
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            process.stdout.close()
    print("PASS owned server/daemon/editor aggregation and overlapping PID deduplication; unrelated and reused registry processes excluded; exited editor, denied inventory and malformed registry are actionable errors")


if __name__ == "__main__":
    main()
