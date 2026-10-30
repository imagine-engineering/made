#!/usr/bin/env python3

import argparse
import ipaddress
import json
import os
import queue
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parent.parent
FIXTURES = REPOSITORY / "crates/project/test_data"
PROBE_SOURCE = "package evidence\n\nfun useGreeting(): String = Greeting().message()\n"


@contextmanager
def probe_directory(retain_on_failure):
    directory = Path(tempfile.mkdtemp(prefix="groovy-kotlin-provider-"))
    failed = False
    try:
        yield directory
    except Exception:
        failed = True
        if retain_on_failure:
            print(f"Retained failed Kotlin probe state: {directory}", file=sys.stderr)
        raise
    finally:
        if not (failed and retain_on_failure):
            shutil.rmtree(directory)


class KotlinClient:
    def __init__(self, server, root, state, java_home, deny_network=False, stdio_argument=True, gradle_user_home=None, maven_repository=None):
        environment = os.environ.copy()
        if java_home is not None:
            environment["JAVA_HOME"] = str(java_home)
        if gradle_user_home is not None:
            environment["GRADLE_USER_HOME"] = str(gradle_user_home)
        state_paths = {
            "idea.system.path": state / "system",
            "idea.config.path": state / "config",
            "idea.log.path": state / "log",
            "user.home": state / "home",
        }
        if any(" " in str(path) for path in state_paths.values()):
            raise ValueError("The isolated state path cannot contain spaces")
        environment["JAVA_TOOL_OPTIONS"] = " ".join(
            [environment.get("JAVA_TOOL_OPTIONS", "")]
            + [f"-D{key}={value}" for key, value in state_paths.items()]
        ).strip()
        if maven_repository is not None:
            environment["MAVEN_REPOSITORY"] = str(maven_repository)
            environment["JAVA_TOOL_OPTIONS"] += f" -Dmaven.repo.local={maven_repository}"
        if deny_network:
            # Java's mapped IPv4 sockets do not match the macOS loopback filter.
            environment["JAVA_TOOL_OPTIONS"] += " -Djava.net.preferIPv4Stack=true"
        command = [str(server)]
        if stdio_argument:
            command.append("--stdio")
        if deny_network:
            command = [
                "/usr/bin/sandbox-exec",
                "-p",
                "(version 1)(allow default)(deny network*)"
                '(allow network-outbound (remote tcp "localhost:*"))'
                '(allow network-outbound (remote udp "localhost:*"))'
                '(allow network-bind (local ip "localhost:*"))'
                # A Gradle loopback listener needs inbound permission as well as bind.
                '(allow network-inbound (local ip "localhost:*"))',
                *command,
            ]
        self.stderr = tempfile.TemporaryFile()
        self.process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            start_new_session=(os.name != "nt"),
        )
        self.root = root
        self.messages = queue.Queue()
        self.notifications = []
        self.next_id = 1
        self.reader = threading.Thread(target=self.read_messages, daemon=True)
        self.reader.start()

    def read_messages(self):
        try:
            while True:
                headers = {}
                while True:
                    line = self.process.stdout.readline()
                    if not line:
                        raise EOFError("Kotlin server closed its output")
                    if line in (b"\r\n", b"\n"):
                        break
                    name, value = line.decode().split(":", 1)
                    headers[name.lower()] = value.strip()
                length = int(headers["content-length"])
                body = self.process.stdout.read(length)
                if len(body) != length:
                    raise EOFError("Kotlin server returned a partial message")
                self.messages.put(json.loads(body))
        except Exception as error:
            self.messages.put(error)

    def send(self, message):
        body = json.dumps(message, separators=(",", ":")).encode()
        self.process.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        self.process.stdin.flush()

    def receive(self, deadline):
        try:
            message = self.messages.get(timeout=max(0.01, deadline - time.monotonic()))
        except queue.Empty as error:
            raise TimeoutError("Kotlin server timed out") from error
        if isinstance(message, Exception):
            raise message
        if "method" in message and "id" in message:
            result = None
            if message["method"] == "workspace/configuration":
                result = [{} for _ in message.get("params", {}).get("items", [])]
            elif message["method"] == "workspace/workspaceFolders":
                result = [{"uri": self.root.as_uri(), "name": self.root.name}]
            elif message["method"] == "workspace/applyEdit":
                result = {"applied": False}
            self.send({"jsonrpc": "2.0", "id": message["id"], "result": result})
        elif "method" in message:
            self.notifications.append(message)
        return message

    def request(self, method, params, timeout=60):
        request_id = self.next_id
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            message = self.receive(deadline)
            if message.get("id") == request_id and "method" not in message:
                if "error" in message:
                    raise RuntimeError(f"Kotlin {method} failed: {message['error']}")
                return message.get("result")
        raise TimeoutError(f"Kotlin {method} timed out")

    def notify(self, method, params):
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def wait_for_import(self, tool, timeout, expected_status="SUCCESS"):
        deadline = time.monotonic() + timeout
        checked = 0
        while time.monotonic() < deadline:
            while checked < len(self.notifications):
                message = self.notifications[checked]
                checked += 1
                if message.get("method") != "intellij/workspaceImportState":
                    continue
                params = message.get("params", {})
                if params.get("phase") != "FINISHED":
                    continue
                folders = params.get("folders", [])
                if not folders or not any(folder.get("tool") == tool for folder in folders):
                    raise AssertionError(f"Kotlin import did not identify {tool}: {folders}")
                if any(folder.get("status") != expected_status for folder in folders):
                    raise AssertionError(f"Kotlin {tool} import did not reach {expected_status}: {folders}")
                return folders
            self.receive(deadline)
        raise TimeoutError(f"Kotlin {tool} import did not reach a terminal state")

    def close(self):
        if self.process.poll() is None:
            try:
                self.request("shutdown", None, timeout=5)
                self.notify("exit", None)
                self.process.wait(timeout=5)
            except (EOFError, TimeoutError, BrokenPipeError, RuntimeError, subprocess.TimeoutExpired):
                pass
        if self.process.poll() is None:
            try:
                if os.name == "nt":
                    self.process.terminate()
                else:
                    os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    if os.name == "nt":
                        self.process.kill()
                    else:
                        os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=5)
        self.stderr.close()


def isolated_gradle_daemon_process_ids(home):
    process_ids = set()
    for log in (home / "daemon/8.13").glob("daemon-*.out.log"):
        process_id = log.name.removeprefix("daemon-").removesuffix(".out.log")
        if not process_id.isdigit() or int(process_id) <= 0:
            raise AssertionError("The isolated Gradle registry has an invalid daemon log name")
        process_ids.add(int(process_id))
    return process_ids


class KotlinResourceSampler:
    def __init__(self, process, gradle_home, interval=0.5, editor_process_id=None):
        self.process_id = process.pid
        if editor_process_id is not None and editor_process_id <= 0:
            raise ValueError("The explicit editor PID must be positive")
        self.editor_process_id = editor_process_id
        self.editor_peak = 0
        self.gradle_home = gradle_home
        self.interval = interval
        self.started = time.monotonic()
        self.samples = 0
        self.server_peak = 0
        self.gradle_peak = 0
        self.combined_peak = 0
        self.process_count_peak = 0
        self.daemon_ids = set()
        self.error = None
        self.stopped = threading.Event()
        self.sample()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def sample(self):
        output = subprocess.check_output(
            ["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss="], text=True, timeout=10
        )
        rows = {}
        for line in output.splitlines():
            values = line.split()
            if len(values) != 4:
                raise AssertionError("Process resource inventory has an unexpected shape")
            process_id, parent_id, group_id, resident_kib = map(int, values)
            if process_id <= 0 or min(parent_id, group_id, resident_kib) < 0:
                raise AssertionError("Process resource inventory has invalid numeric values")
            rows[process_id] = (parent_id, group_id, resident_kib * 1024)

        def descendants(roots):
            owned = set(roots) & rows.keys()
            while True:
                added = {process_id for process_id, row in rows.items() if row[0] in owned} - owned
                if not added:
                    return owned
                owned.update(added)

        server_ids = descendants(
            {process_id for process_id, row in rows.items() if row[1] == self.process_id}
            | {self.process_id}
        )
        daemon_ids = set()
        if self.gradle_home is not None:
            # The probe creates a fresh isolated daemon registry; shared host caches
            # cannot safely establish ownership of a reparented Gradle daemon.
            daemon_ids = isolated_gradle_daemon_process_ids(self.gradle_home) & rows.keys()
        self.daemon_ids.update(daemon_ids)
        gradle_ids = descendants(daemon_ids)
        editor_ids = set()
        if self.editor_process_id is not None:
            if self.editor_process_id not in rows:
                raise RuntimeError("The explicit editor process exited or is unavailable during sampling")
            editor_ids = descendants({self.editor_process_id})
        combined_ids = server_ids | gradle_ids | editor_ids
        self.editor_peak = max(self.editor_peak, sum(rows[pid][2] for pid in editor_ids))
        self.samples += 1
        self.server_peak = max(self.server_peak, sum(rows[pid][2] for pid in server_ids))
        self.gradle_peak = max(self.gradle_peak, sum(rows[pid][2] for pid in gradle_ids))
        self.combined_peak = max(self.combined_peak, sum(rows[pid][2] for pid in combined_ids))
        self.process_count_peak = max(self.process_count_peak, len(combined_ids))

    def run(self):
        try:
            while not self.stopped.wait(self.interval):
                self.sample()
        except Exception as error:
            self.error = error

    def finish(self):
        self.stopped.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise TimeoutError("Kotlin process resource sampler did not stop")
        if self.error is not None:
            raise RuntimeError("Kotlin process resource sampling failed") from self.error
        if self.samples < 1 or self.server_peak <= 0:
            raise AssertionError("Kotlin process resource sampling observed no server process")
        return {
            "intervalSeconds": self.interval,
            "elapsedSeconds": round(time.monotonic() - self.started, 2),
            "successfulSamples": self.samples,
            "explicitEditorProcessId": self.editor_process_id,
            "sampledPeakEditorTreeResidentBytes": self.editor_peak,
            "sampledPeakServerGroupResidentBytes": self.server_peak,
            "sampledPeakIsolatedGradleResidentBytes": self.gradle_peak,
            "sampledPeakCombinedResidentBytes": self.combined_peak,
            "sampledPeakOwnedProcessCount": self.process_count_peak,
            "observedIsolatedGradleDaemonPids": sorted(self.daemon_ids),
            "scope": "server process group/current descendants; isolated Gradle daemon trees; explicit editor/current descendants when supplied",
            "limitations": "sampled sums of RSS can count shared pages more than once and miss between-sample peaks, detached untracked workers or PID reuse; unrelated/shared daemons are excluded; editor is excluded unless an explicit PID is supplied; caller must bind editor identity and every owned worker to the fixture; not whole-project physical memory or alpha acceptance",
        }


def location_uri(location):
    if isinstance(location, list):
        return location_uri(location[0]) if location else None
    if isinstance(location, dict):
        return location.get("uri") or location.get("targetUri")
    return None


def completion_labels(result):
    items = result if isinstance(result, list) else (result or {}).get("items", [])
    return {item.get("label") for item in items if isinstance(item, dict)}


def child_peak_resident_bytes():
    if os.name != "posix":
        return None
    import resource

    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform == "darwin":
        return peak
    if sys.platform.startswith("linux"):
        return peak * 1024
    return None


def wait_for_diagnostics(client, uri, expect_error):
    deadline = time.monotonic() + 30
    latest = None
    while time.monotonic() < deadline:
        latest = client.request(
            "textDocument/diagnostic",
            {"textDocument": {"uri": uri}},
            timeout=15,
        )
        if latest.get("kind") == "full":
            errors = [item for item in latest.get("items", []) if item.get("severity") == 1]
            if bool(errors) == expect_error and (expect_error or not latest.get("items")):
                return len(errors)
        time.sleep(0.5)
    raise AssertionError(f"Kotlin diagnostic transition did not complete: {latest}")


def wait_for_published_diagnostics(client, uri, expect_error, start_index=0, timeout=40):
    deadline = time.monotonic() + timeout
    checked = start_index
    while time.monotonic() < deadline:
        while checked < len(client.notifications):
            message = client.notifications[checked]
            checked += 1
            if message.get("method") != "textDocument/publishDiagnostics":
                continue
            params = message.get("params", {})
            if params.get("uri") != uri:
                continue
            diagnostics = params.get("diagnostics", [])
            errors = [item for item in diagnostics if item.get("severity") == 1]
            if expect_error and errors:
                return len(errors)
            if not expect_error and not diagnostics:
                return 0
        client.receive(deadline)
    raise AssertionError("Kotlin fallback did not publish the expected diagnostics")


def replace_probe_document(client, uri, version, text):
    client.notify(
        "textDocument/didChange",
        {
            "textDocument": {"uri": uri, "version": version},
            "contentChanges": [
                {
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 3, "character": 0},
                    },
                    "text": text,
                }
            ],
        },
    )


def check_semantics(client, source, probe, extra_source=None):
    source_uri = source.as_uri()
    probe_uri = probe.as_uri()
    client.notify(
        "textDocument/didOpen",
        {
            "textDocument": {
                "uri": source_uri,
                "languageId": "kotlin",
                "version": 1,
                "text": source.read_text(),
            }
        },
    )
    client.notify(
        "textDocument/didOpen",
        {
            "textDocument": {
                "uri": probe_uri,
                "languageId": "kotlin",
                "version": 1,
                "text": PROBE_SOURCE,
            }
        },
    )
    symbols = client.request("textDocument/documentSymbol", {"textDocument": {"uri": source_uri}})
    if not any(symbol.get("name") == "Greeting" for symbol in symbols or []):
        raise AssertionError(f"Kotlin document symbols omitted Greeting: {symbols}")

    line = PROBE_SOURCE.splitlines()[2]
    definition_position = {"line": 2, "character": line.rindex("Greeting") + 1}
    definition = client.request(
        "textDocument/definition",
        {"textDocument": {"uri": probe_uri}, "position": definition_position},
    )
    if location_uri(definition) != source_uri:
        raise AssertionError(f"Kotlin definition did not reach Greeting.kt: {definition}")

    completion_position = {"line": 2, "character": line.index(".message") + 1}
    completion = client.request(
        "textDocument/completion",
        {"textDocument": {"uri": probe_uri}, "position": completion_position},
    )
    labels = completion_labels(completion)
    if "message" not in labels:
        raise AssertionError(f"Kotlin completion omitted message: {sorted(labels)}")

    if extra_source is not None:
        extra_uri = extra_source.as_uri()
        client.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": extra_uri,
                    "languageId": "kotlin",
                    "version": 1,
                    "text": extra_source.read_text(),
                }
            },
        )
        extra_symbols = client.request(
            "textDocument/documentSymbol", {"textDocument": {"uri": extra_uri}}
        )
        if not any(
            symbol.get("name") == extra_source.stem for symbol in extra_symbols or []
        ):
            raise AssertionError(
                f"Kotlin symbols omitted the added source {extra_source.name}: {extra_symbols}"
            )
        wait_for_diagnostics(client, extra_uri, expect_error=False)

    wait_for_diagnostics(client, probe_uri, expect_error=False)
    invalid_source = PROBE_SOURCE.replace("Greeting().message()", "MissingGreeting().message()")
    replace_probe_document(client, probe_uri, 2, invalid_source)
    errors = wait_for_diagnostics(client, probe_uri, expect_error=True)
    replace_probe_document(client, probe_uri, 3, PROBE_SOURCE)
    wait_for_diagnostics(client, probe_uri, expect_error=False)
    return {
        "symbols": len(symbols),
        "definition": "Greeting.kt",
        "completion": "message",
        "invalidSourceErrors": errors,
        "restoredDiagnostics": 0,
        "indexedExtraSource": extra_source.name if extra_source is not None else None,
    }


def check_java_to_kotlin_navigation(client, root, kotlin_source):
    java_source = root / "application/src/main/java/evidence/Main.java"
    text = java_source.read_text()
    row, line = next(
        (row, line) for row, line in enumerate(text.splitlines()) if "Greeting" in line
    )
    java_uri = java_source.as_uri()
    client.notify(
        "textDocument/didOpen",
        {
            "textDocument": {
                "uri": java_uri,
                "languageId": "java",
                "version": 1,
                "text": text,
            }
        },
    )
    definition = client.request(
        "textDocument/definition",
        {
            "textDocument": {"uri": java_uri},
            "position": {"line": row, "character": line.index("Greeting") + 2},
        },
    )
    locations = definition if isinstance(definition, list) else [definition]
    if not any(location_uri(location) == kotlin_source.as_uri() for location in locations):
        raise AssertionError(
            f"Kotlin server did not resolve Java Greeting to Kotlin source: {definition}"
        )


def check_workspace_export(client, root, export_directory, fixture_name):
    export_directory.mkdir()
    response = client.request(
        "workspace/executeCommand",
        {"command": "exportWorkspace", "arguments": [str(export_directory)]},
        timeout=90,
    )
    if response is not None:
        raise AssertionError("Kotlin workspace export returned an unexpected result")
    if (root / "workspace.json").exists():
        raise AssertionError("Kotlin workspace export wrote into the fixture project")
    export = export_directory / "workspace.json"
    metadata = export.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 8 * 1024 * 1024:
        raise AssertionError("Kotlin workspace export is not a bounded regular file")
    workspace = json.loads(export.read_bytes())
    if workspace.get("externalSystem") != fixture_name.upper():
        raise AssertionError("Kotlin workspace export omitted the current build system")
    lists = {}
    for name, limit in [
        ("modules", 128),
        ("libraries", 4096),
        ("sdks", 128),
        ("javaSettings", 128),
    ]:
        values = workspace.get(name, [])
        if (
            not isinstance(values, list)
            or len(values) > limit
            or any(not isinstance(item, dict) for item in values)
        ):
            raise AssertionError(f"Kotlin workspace export has an invalid {name} list")
        lists[name] = values
    module_directories = set()
    for module in lists["modules"]:
        path = module.get("externalProjectPath")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise AssertionError("Kotlin workspace export omitted a fixture module path")
        relative = Path(path).resolve().relative_to(root.resolve())
        module_directories.add(relative.as_posix())
    if not {"application", "library"}.issubset(module_directories):
        raise AssertionError("Kotlin workspace export omitted a fixture module")
    java_sdks = [sdk for sdk in lists["sdks"] if sdk.get("type") in ("JavaSDK", "jdk")]
    if any(
        sdk.get("version") is not None and not isinstance(sdk["version"], str)
        for sdk in java_sdks
    ):
        raise AssertionError("Kotlin workspace export has an invalid Java SDK version field")
    levels = {
        value
        for settings in lists["javaSettings"]
        if isinstance(value := settings.get("languageLevelId"), str)
        and value.startswith("JDK_")
        and len(value) <= 32
        and all(
            character.isascii() and (character.isalnum() or character == "_")
            for character in value
        )
    }
    return {
        "modules": len(lists["modules"]),
        "libraries": len(lists["libraries"]),
        "javaSdks": len(java_sdks),
        "javaSdksWithoutVersion": sum(sdk.get("version") is None for sdk in java_sdks),
        "javaLanguageLevelIds": sorted(levels),
    }


def check_alternate_semantics(
    client, source, probe, extra_source, require_import_logs, diagnostic_timeout=40, build_system="gradle"
):
    source_uri = source.as_uri()
    probe_uri = probe.as_uri()
    for path in (source, probe):
        client.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": path.as_uri(),
                    "languageId": "kotlin",
                    "version": 1,
                    "text": path.read_text(),
                }
            },
        )
    line = PROBE_SOURCE.splitlines()[2]
    deadline = time.monotonic() + 180
    definition = None
    while time.monotonic() < deadline:
        definition = client.request(
            "textDocument/definition",
            {
                "textDocument": {"uri": probe_uri},
                "position": {"line": 2, "character": line.rindex("Greeting") + 1},
            },
            timeout=45,
        )
        if location_uri(definition) == source_uri:
            break
        time.sleep(2)
    if location_uri(definition) != source_uri:
        raise AssertionError(f"Kotlin fallback did not resolve Greeting.kt: {definition}")

    symbols = client.request("textDocument/documentSymbol", {"textDocument": {"uri": source_uri}})
    if not any(symbol.get("name") == "Greeting" for symbol in symbols or []):
        raise AssertionError(f"Kotlin fallback omitted Greeting symbol: {symbols}")
    completion = client.request(
        "textDocument/completion",
        {
            "textDocument": {"uri": probe_uri},
            "position": {"line": 2, "character": line.index(".message") + 1},
        },
    )
    if "message()" not in completion_labels(completion):
        raise AssertionError("Kotlin fallback omitted message() completion")

    if extra_source is not None:
        extra_uri = extra_source.as_uri()
        extra_text = extra_source.read_text()
        client.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": extra_uri,
                    "languageId": "kotlin",
                    "version": 1,
                    "text": extra_text,
                }
            },
        )
        extra_symbols = client.request(
            "textDocument/documentSymbol", {"textDocument": {"uri": extra_uri}}
        )
        if not any(
            symbol.get("name") == extra_source.stem for symbol in extra_symbols or []
        ):
            raise AssertionError(
                f"Kotlin fallback omitted added source {extra_source.name}: {extra_symbols}"
            )
        extra_line = extra_text.splitlines()[2]
        extra_definition = client.request(
            "textDocument/definition",
            {
                "textDocument": {"uri": extra_uri},
                "position": {
                    "line": 2,
                    "character": extra_line.index("Greeting") + 1,
                },
            },
        )
        if location_uri(extra_definition) != source_uri:
            raise AssertionError(
                f"Kotlin fallback added source did not resolve Greeting.kt: {extra_definition}"
            )
        wait_for_published_diagnostics(
            client, extra_uri, expect_error=False, timeout=diagnostic_timeout
        )

    wait_for_published_diagnostics(
        client, probe_uri, expect_error=False, timeout=diagnostic_timeout
    )
    invalid_source = PROBE_SOURCE.replace("Greeting().message()", "MissingGreeting().message()")
    before_invalid = len(client.notifications)
    replace_probe_document(client, probe_uri, 2, invalid_source)
    errors = wait_for_published_diagnostics(
        client,
        probe_uri,
        expect_error=True,
        start_index=before_invalid,
        timeout=diagnostic_timeout,
    )
    before_restored = len(client.notifications)
    replace_probe_document(client, probe_uri, 3, PROBE_SOURCE)
    wait_for_published_diagnostics(
        client,
        probe_uri,
        expect_error=False,
        start_index=before_restored,
        timeout=diagnostic_timeout,
    )

    logs = [
        message.get("params", {}).get("message", "")
        for message in client.notifications
        if message.get("method") == "window/logMessage"
    ]
    result = {
        "definition": "Greeting.kt",
        "completion": "message()",
        "invalidSourceErrors": errors,
        "restoredDiagnostics": 0,
        "indexedExtraSource": extra_source.name if extra_source is not None else None,
    }
    if build_system == "gradle":
        dependencies_resolved = any(
            "Successfully resolved dependencies for 'library' using Gradle" in message
            for message in logs
        )
        build_scripts_resolved = any(
            "Successfully resolved build script dependencies for 'library' using Gradle" in message
            for message in logs
        )
        if require_import_logs and not (dependencies_resolved and build_scripts_resolved):
            raise AssertionError("Kotlin fallback did not log Gradle dependency resolution")
        result.update(
            dependencyResolutionLogged=dependencies_resolved,
            buildScriptResolutionLogged=build_scripts_resolved,
        )
    elif build_system == "maven":
        resolutions = {
            name: any(
                "Found " in message and f" in {client.root / name}" in message
                for message in logs
            )
            for name in ("pom.xml", "library/pom.xml", "application/pom.xml")
        }
        if require_import_logs and not all(resolutions.values()):
            raise AssertionError(f"Kotlin fallback did not log cached Maven artifacts for each fixture POM: {resolutions}")
        result["mavenPomResolutionLogged"] = resolutions
    else:
        raise ValueError("Unsupported fallback fixture build system")
    return result


def make_gradle_wrapper_offline(root):
    wrapper = root / "gradlew"
    original = root / "gradlew.groovy-probe-original"
    wrapper.rename(original)
    wrapper.write_text(
        '#!/bin/sh\nprintf "run\\n" >> "$(dirname "$0")/.groovy-offline-wrapper-runs"\n'
        'exec "$(dirname "$0")/gradlew.groovy-probe-original" --offline "$@"\n'
    )
    wrapper.chmod(original.stat().st_mode)


def prepare_cached_gradle_home(home, cache_source, copy_dependencies=True):
    home.mkdir()
    for name in (("wrapper", "caches") if copy_dependencies else ("wrapper",)):
        source = cache_source / name
        if not source.is_dir():
            raise AssertionError(f"Cached Gradle probe needs a populated {name} directory")
        subprocess.run(["/bin/cp", "-cR", str(source), str(home / name)], check=True)
    if not copy_dependencies:
        (home / "caches").mkdir()
    for lock in (home / "caches").rglob("*.lock"):
        if lock.is_file():
            lock.unlink()
    init_directory = home / "init.d"
    init_directory.mkdir()
    (init_directory / "groovy-download-denial.gradle").write_text(
        """import groovy.json.JsonOutput
import java.net.InetSocketAddress
import java.net.Socket
import java.net.SocketException

def denied = false
try {
    new Socket().withCloseable { connection ->
        connection.connect(new InetSocketAddress("1.1.1.1", 443), 2000)
    }
} catch (SocketException error) {
    denied = error.message.contains("Operation not permitted")
}
new File(gradle.gradleUserHomeDir, "groovy-download-denial.json").text = JsonOutput.toJson([
    externalTcpDenied: denied,
    processId: ProcessHandle.current().pid(),
    runtimeVersion: System.getProperty("java.runtime.version")
])
if (!denied) {
    throw new GradleException("The isolated Gradle import JVM did not deny external TCP")
}
"""
    )


def check_cached_gradle_daemon(home):
    evidence = json.loads((home / "groovy-download-denial.json").read_text())
    process_id = evidence.get("processId")
    version = evidence.get("runtimeVersion")
    if (
        evidence.get("externalTcpDenied") is not True
        or not isinstance(process_id, int)
        or process_id <= 0
        or not (home / "daemon/8.13" / f"daemon-{process_id}.out.log").is_file()
        or not isinstance(version, str)
        or not 1 <= len(version) <= 128
        or any(not character.isascii() or not (character.isalnum() or character in ".+-_") for character in version)
    ):
        raise AssertionError("The cached Gradle import has no valid isolated denial evidence")
    output = subprocess.check_output(
        ["/usr/sbin/lsof", "-nP", "-a", "-p", str(process_id), "-iTCP", "-sTCP:LISTEN", "-F", "n"],
        text=True,
    )
    listeners = [line[1:].rsplit(":", 1)[0].strip("[]") for line in output.splitlines() if line.startswith("n")]
    if not listeners or not all(ipaddress.ip_address(address).is_loopback for address in listeners):
        raise AssertionError("The isolated Gradle daemon has no exclusively loopback listeners")
    return {
        "externalTcpDenied": True,
        "processId": process_id,
        "runtimeVersion": version,
        "loopbackListeners": len(listeners),
    }


def stop_cached_gradle_daemon(home, java_home):
    executables = list(home.glob("wrapper/dists/gradle-8.13-bin/*/gradle-8.13/bin/gradle"))
    if len(executables) != 1:
        raise AssertionError("The isolated Gradle fixture needs exactly one cached 8.13 launcher")
    environment = os.environ.copy()
    environment["JAVA_HOME"] = str(java_home)
    subprocess.run(
        [str(executables[0]), "--stop", "--gradle-user-home", str(home)],
        env=environment,
        cwd=home,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    process_ids = isolated_gradle_daemon_process_ids(home)
    deadline = time.monotonic() + 10
    while process_ids:
        for process_id in process_ids.copy():
            try:
                os.kill(process_id, 0)
            except ProcessLookupError:
                process_ids.remove(process_id)
        if process_ids:
            if time.monotonic() >= deadline:
                raise AssertionError("The isolated Gradle daemon did not exit after --stop")
            time.sleep(0.1)


def probe(
    server,
    java_home,
    fixture_name,
    expected_version,
    timeout,
    restarts,
    expect_denied_network,
    extra_kotlin_sources,
    require_java_to_kotlin_navigation,
    server_kind,
    alternate_offline_restart,
    alternate_denied_network_restart,
    retain_failed_state,
    require_workspace_export=False,
    cached_denied_network=False,
    expect_gradle_denied_network=False,
    sample_process_resources=False,
    resource_editor_process_id=None,
    alternate_maven_repository=None,
    alternate_maven_denied_network=False,
):
    with probe_directory(retain_failed_state or cached_denied_network or expect_gradle_denied_network) as directory:
        root = directory / fixture_name
        shutil.copytree(FIXTURES / f"groovy_jvm_{fixture_name}", root)
        if expect_denied_network:
            empty_repository = directory / "empty-maven"
            empty_repository.mkdir()
            (root / ".mvn/maven.config").write_text(
                f"-Dmaven.repo.local={empty_repository}\n"
            )
        source = root / "library/src/main/kotlin/evidence/Greeting.kt"
        extra_source = None
        for index in range(extra_kotlin_sources):
            extra_source = source.with_name(f"GroovyScale{index:03d}.kt")
            extra_source.write_text(
                "package evidence\n\n"
                f"class {extra_source.stem} {{ fun value(): String = Greeting().message() }}\n"
            )
        probe_source = source.with_name("UseGreeting.kt")
        probe_source.write_text(PROBE_SOURCE)
        results = []
        for launch in range(restarts):
            gradle_home = None
            if cached_denied_network or expect_gradle_denied_network:
                gradle_home = directory / f"cached-gradle-{launch}"
                cache_source = Path(os.environ.get("GRADLE_USER_HOME", Path.home() / ".gradle"))
                prepare_cached_gradle_home(gradle_home, cache_source, copy_dependencies=not expect_gradle_denied_network)
            if launch == 1 and alternate_offline_restart:
                make_gradle_wrapper_offline(root)
            denied_network = expect_denied_network or cached_denied_network or expect_gradle_denied_network or alternate_maven_denied_network or (
                alternate_denied_network_restart and launch > 0
            )
            client = KotlinClient(
                server,
                root,
                directory / "state",
                java_home,
                denied_network,
                stdio_argument=server_kind == "official",
                gradle_user_home=gradle_home,
                maven_repository=alternate_maven_repository,
            )
            started = time.monotonic()
            sampler = None
            try:
                if sample_process_resources:
                    sampler = KotlinResourceSampler(
                        client.process, gradle_home, editor_process_id=resource_editor_process_id
                    )
                initialized = client.request(
                    "initialize",
                    {
                        "processId": os.getpid(),
                        "clientInfo": {"name": "groovy-kotlin-provider-probe"},
                        "rootUri": root.as_uri(),
                        "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
                        "capabilities": {
                            "workspace": {"workspaceFolders": True, "configuration": True},
                            "textDocument": (
                                {"diagnostic": {}}
                                if server_kind == "official"
                                else {"publishDiagnostics": {}}
                            ),
                        },
                    },
                    timeout=timeout,
                )
                server_info = initialized.get("serverInfo", {})
                if server_info.get("version") != expected_version:
                    raise AssertionError(f"Expected {expected_version}, got {server_info}")
                client.notify("initialized", {})
                if server_kind == "alternate":
                    semantics = check_alternate_semantics(
                        client,
                        source,
                        probe_source,
                        extra_source,
                        require_import_logs=launch == 0,
                        diagnostic_timeout=120 if denied_network else 40,
                        build_system=fixture_name,
                    )
                    imported = [{"tool": fixture_name, "status": "semantic-ready"}]
                else:
                    folders = client.wait_for_import(
                        fixture_name,
                        timeout,
                        expected_status="FAILED" if expect_denied_network or expect_gradle_denied_network else "SUCCESS",
                    )
                    imported = [
                        {"tool": item.get("tool"), "status": item.get("status")}
                        for item in folders
                    ]
                if expect_denied_network:
                    fetch_failed = any(
                        message.get("method") == "intellij/importLog"
                        and "Could not transfer artifact" in message.get("params", {}).get("message", "")
                        for message in client.notifications
                    )
                    if not fetch_failed:
                        raise AssertionError("Maven failed without an observed artifact-transfer denial")
                    client.notify(
                        "textDocument/didOpen",
                        {
                            "textDocument": {
                                "uri": source.as_uri(),
                                "languageId": "kotlin",
                                "version": 1,
                                "text": source.read_text(),
                            }
                        },
                    )
                    wait_for_diagnostics(client, source.as_uri(), expect_error=False)
                    semantics = {"artifactFetchFailed": True, "validSourceDiagnostics": 0}
                elif expect_gradle_denied_network:
                    import_logs = [
                        message.get("params", {}).get("message", "")
                        for message in client.notifications
                        if message.get("method") == "intellij/importLog"
                    ]
                    plugin_failed = (
                        any("org.gradle.api.plugins.UnknownPluginException" in message for message in import_logs)
                        and any("could not resolve plugin artifact" in message for message in import_logs)
                    )
                    if not plugin_failed:
                        raise AssertionError("Gradle failed without the observed plugin-resolution error")
                    dependency_jars = list((gradle_home / "caches/modules-2/files-2.1").rglob("*.jar"))
                    if dependency_jars:
                        raise AssertionError("The empty-cache Gradle denial fetched dependency JARs")
                    client.notify(
                        "textDocument/didOpen",
                        {"textDocument": {"uri": source.as_uri(), "languageId": "kotlin", "version": 1, "text": source.read_text()}},
                    )
                    symbols = client.request("textDocument/documentSymbol", {"textDocument": {"uri": source.as_uri()}})
                    if not any(symbol.get("name") == "Greeting" for symbol in symbols or []):
                        raise AssertionError("Failed Gradle import lost source symbols")
                    wait_for_diagnostics(client, source.as_uri(), expect_error=False)
                    semantics = {"pluginResolutionFailed": True, "symbols": 1, "validSourceDiagnostics": 0, "cachedDependencyJars": 0}
                elif server_kind == "official":
                    semantics = check_semantics(client, source, probe_source, extra_source)
                    if require_java_to_kotlin_navigation:
                        check_java_to_kotlin_navigation(client, root, source)
                        semantics["javaToKotlinDefinition"] = "Greeting.kt"
                result = {
                    "launch": launch + 1,
                    "server": server_info,
                    "import": imported,
                    "semantics": semantics,
                    "offlineWrapper": alternate_offline_restart and launch > 0,
                    "deniedNetwork": denied_network,
                    "elapsedSeconds": round(time.monotonic() - started, 1),
                }
                if require_workspace_export:
                    commands = (
                        initialized.get("capabilities", {})
                        .get("executeCommandProvider", {})
                        .get("commands", [])
                    )
                    if "exportWorkspace" not in commands:
                        raise AssertionError("Kotlin server did not advertise workspace export")
                    result["workspaceExport"] = check_workspace_export(
                        client, root, directory / f"workspace-export-{launch}", fixture_name
                    )
                    result["elapsedSeconds"] = round(time.monotonic() - started, 1)
                if gradle_home is not None:
                    result["gradleImportJvm"] = check_cached_gradle_daemon(gradle_home)
                if server_kind == "alternate":
                    result["stdlibResolutionFailures"] = sum(
                        1
                        for message in client.notifications
                        if message.get("method") == "window/logMessage"
                        and "Could not resolve kotlin-stdlib using Gradle"
                        in message.get("params", {}).get("message", "")
                    )
                if alternate_offline_restart and launch > 0:
                    marker = root / ".groovy-offline-wrapper-runs"
                    if not marker.is_file():
                        raise AssertionError("Kotlin fallback did not invoke the offline Gradle wrapper")
                    result["offlineWrapperInvocations"] = len(marker.read_text().splitlines())
                results.append(result)
            except Exception:
                if retain_failed_state or cached_denied_network or expect_gradle_denied_network:
                    client.stderr.seek(0, os.SEEK_END)
                    stderr_size = client.stderr.tell()
                    client.stderr.seek(max(0, stderr_size - 1_000_000))
                    (directory / "server-stderr.txt").write_bytes(client.stderr.read())
                    (directory / "server-notifications.json").write_text(
                        json.dumps(client.notifications[-200:], indent=2)
                    )
                raise
            finally:
                try:
                    try:
                        client.close()
                        if results and results[-1]["launch"] == launch + 1:
                            results[-1]["serverExitCode"] = client.process.returncode
                    finally:
                        if gradle_home is not None:
                            stop_cached_gradle_daemon(gradle_home, java_home)
                            if results and results[-1]["launch"] == launch + 1:
                                results[-1]["gradleImportJvm"]["daemonStopped"] = True
                finally:
                    if sampler is not None:
                        resources = sampler.finish()
                        if results and results[-1]["launch"] == launch + 1:
                            results[-1]["processResources"] = resources
                        elif retain_failed_state or cached_denied_network or expect_gradle_denied_network:
                            (directory / "failed-process-resources.json").write_text(json.dumps(resources, indent=2) + "\n")
            if results:
                peak = child_peak_resident_bytes()
                if peak is not None:
                    results[-1]["childPeakResidentBytesSoFar"] = peak
        return results


def main():
    parser = argparse.ArgumentParser(description="Probe a pinned Kotlin server against Groovy fixtures")
    parser.add_argument("--server", required=True, type=Path, help="Path to the standalone server executable")
    parser.add_argument("--expected-version", required=True, help="Exact LSP serverInfo.version, such as ILS-263.4702.0")
    parser.add_argument("--server-kind", choices=("official", "alternate"), default="official")
    parser.add_argument(
        "--alternate-offline-restart",
        action="store_true",
        help="After a cached alternate Gradle import, require its wrapper to run offline",
    )
    parser.add_argument(
        "--alternate-denied-network-restart",
        action="store_true",
        help="After an online cache warmup, deny non-local network during fallback restart",
    )
    parser.add_argument(
        "--retain-failed-state",
        action="store_true",
        help="Keep a failed disposable fixture and server logs for local diagnosis",
    )
    parser.add_argument(
        "--alternate-maven-repository",
        type=Path,
        help="Existing disposable Maven cache shared by fallback artifact lookup and Maven subprocesses",
    )
    parser.add_argument(
        "--alternate-maven-denied-network",
        action="store_true",
        help="Deny external network for both cached Maven fallback launches (macOS)",
    )
    parser.add_argument("--java-home", type=Path, help="Host JDK for Maven or Gradle subprocesses")
    parser.add_argument("--fixture", choices=("maven", "gradle"), required=True)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument(
        "--extra-kotlin-sources",
        type=int,
        default=0,
        help="Add up to 500 deterministic Kotlin source files to the fixture's library module",
    )
    parser.add_argument(
        "--require-java-to-kotlin-navigation",
        action="store_true",
        help="Fail unless a Java Greeting reference resolves to Kotlin source",
    )
    parser.add_argument(
        "--require-workspace-export",
        action="store_true",
        help="Verify bounded official-server module/SDK export without inferring build/test JDK roles",
    )
    parser.add_argument(
        "--cached-denied-network",
        action="store_true",
        help="Require successful official Gradle import with isolated cloned caches and denied external downloads (macOS)",
    )
    parser.add_argument(
        "--expect-gradle-denied-network",
        action="store_true",
        help="Require failed official Gradle import with an empty dependency cache and denied external downloads (macOS)",
    )
    parser.add_argument(
        "--expect-denied-network",
        action="store_true",
        help="Require failed Maven import with an empty repository under macOS network denial",
    )
    parser.add_argument(
        "--sample-process-resources",
        action="store_true",
        help="Sample owned server/descendant RSS and isolated Gradle daemons; not whole-project memory (macOS/Linux)",
    )
    parser.add_argument(
        "--resource-editor-pid",
        type=int,
        help="Include this explicitly identified editor and descendants in resource sums; does not certify a complete app workflow",
    )
    arguments = parser.parse_args()
    if arguments.resource_editor_pid is not None and (
        not arguments.sample_process_resources or arguments.resource_editor_pid <= 0
    ):
        parser.error("an explicit positive editor PID requires --sample-process-resources")
    if arguments.sample_process_resources and sys.platform != "darwin" and not sys.platform.startswith("linux"):
        parser.error("process resource sampling requires macOS or Linux ps")
    if (arguments.cached_denied_network or arguments.expect_gradle_denied_network) and (
        sys.platform != "darwin"
        or arguments.fixture != "gradle"
        or arguments.server_kind != "official"
        or arguments.java_home is None
        or arguments.expect_denied_network
        or (arguments.cached_denied_network and arguments.expect_gradle_denied_network)
    ):
        parser.error("Gradle denial requires macOS, official Gradle, an explicit build Java home and one cache mode")
    if arguments.require_workspace_export and (
        arguments.server_kind != "official" or arguments.expect_denied_network or arguments.expect_gradle_denied_network
    ):
        parser.error("workspace export requires a successful official-server import")
    if arguments.timeout < 1 or arguments.restarts < 1:
        parser.error("timeout and restarts must be positive")
    if not 0 <= arguments.extra_kotlin_sources <= 500:
        parser.error("extra Kotlin sources must be between 0 and 500")
    if arguments.expect_gradle_denied_network and (arguments.extra_kotlin_sources or arguments.require_java_to_kotlin_navigation):
        parser.error("failed Gradle import does not support extra sources or Java-to-Kotlin navigation")
    if arguments.expect_denied_network and arguments.extra_kotlin_sources:
        parser.error("denied-network mode does not support extra Kotlin sources")
    if arguments.expect_denied_network and arguments.require_java_to_kotlin_navigation:
        parser.error("denied-network mode does not support Java-to-Kotlin navigation")
    if arguments.server_kind == "alternate" and (
        arguments.expect_denied_network or arguments.require_java_to_kotlin_navigation
    ):
        parser.error("alternate-server probe does not support official-server import failure or Java navigation modes")
    if arguments.alternate_maven_repository is not None:
        if arguments.server_kind != "alternate" or arguments.fixture != "maven":
            parser.error("alternate Maven cache requires the Maven fallback fixture")
        try:
            arguments.alternate_maven_repository = arguments.alternate_maven_repository.resolve(strict=True)
        except OSError:
            parser.error("alternate Maven cache must be an existing disposable directory")
        if not arguments.alternate_maven_repository.is_dir() or any(
            character.isspace() or character in "\"'" for character in str(arguments.alternate_maven_repository)
        ):
            parser.error("alternate Maven cache must be a directory without whitespace or quotes")
    if arguments.server_kind == "alternate" and arguments.fixture == "maven" and arguments.alternate_maven_repository is None:
        parser.error("Maven fallback requires an explicit disposable --alternate-maven-repository")
    if arguments.alternate_maven_denied_network and (
        arguments.server_kind != "alternate" or arguments.fixture != "maven"
        or arguments.alternate_maven_repository is None or sys.platform != "darwin"
    ):
        parser.error("alternate Maven network denial requires macOS and an explicit Maven fallback cache")
    if arguments.alternate_offline_restart and (
        arguments.server_kind != "alternate" or arguments.fixture != "gradle" or arguments.restarts < 2 or os.name != "posix"
    ):
        parser.error("alternate offline restart requires a POSIX Gradle fallback probe with two launches")
    if arguments.alternate_denied_network_restart and (
        arguments.server_kind != "alternate"
        or arguments.fixture != "gradle"
        or arguments.restarts < 2
        or not arguments.alternate_offline_restart
        or os.uname().sysname != "Darwin"
    ):
        parser.error(
            "alternate denied-network restart requires macOS, Gradle fallback, "
            "and offline wrapper restart"
        )
    if arguments.expect_denied_network and (
        arguments.fixture != "maven" or arguments.restarts != 1 or os.uname().sysname != "Darwin"
    ):
        parser.error("denied-network mode requires macOS, Maven, and --restarts 1")
    if not arguments.server.is_file():
        parser.error(f"Kotlin server is missing: {arguments.server}")
    print(
        json.dumps(
            {
                "fixture": arguments.fixture,
                "serverKind": arguments.server_kind,
                "deniedNetworkOnRestart": arguments.alternate_denied_network_restart,
                "mavenRepositoryAligned": arguments.alternate_maven_repository is not None,
                "mavenDeniedNetwork": arguments.alternate_maven_denied_network,
                "cachedDeniedNetwork": arguments.cached_denied_network,
                "emptyGradleCacheDeniedNetwork": arguments.expect_gradle_denied_network,
                "extraKotlinSources": arguments.extra_kotlin_sources,
                "sampleProcessResources": arguments.sample_process_resources,
                "explicitEditorProcessId": arguments.resource_editor_pid,
                "results": probe(
                    arguments.server.resolve(),
                    arguments.java_home.resolve() if arguments.java_home else None,
                    arguments.fixture,
                    arguments.expected_version,
                    arguments.timeout,
                    arguments.restarts,
                    arguments.expect_denied_network,
                    arguments.extra_kotlin_sources,
                    arguments.require_java_to_kotlin_navigation,
                    arguments.server_kind,
                    arguments.alternate_offline_restart,
                    arguments.alternate_denied_network_restart,
                    arguments.retain_failed_state,
                    arguments.require_workspace_export,
                    arguments.cached_denied_network,
                    arguments.expect_gradle_denied_network,
                    arguments.sample_process_resources,
                    arguments.resource_editor_pid,
                    arguments.alternate_maven_repository,
                    arguments.alternate_maven_denied_network,
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
