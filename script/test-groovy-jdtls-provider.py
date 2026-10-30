#!/usr/bin/env python3

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import unquote, urlparse


REPOSITORY = Path(__file__).resolve().parent.parent
FIXTURES = REPOSITORY / "crates/project/test_data"
SETTINGS_KEYS = [
    "org.eclipse.jdt.ls.core.sourcePaths",
    "org.eclipse.jdt.ls.core.classpathEntries",
    "org.eclipse.jdt.ls.core.vm.location",
    "org.eclipse.jdt.core.compiler.source",
    "org.eclipse.jdt.core.compiler.codegen.targetPlatform",
]
GRADLE_DENIED_NETWORK_PROFILE = """(version 1)
(allow default)
(deny network*)
(allow network-bind)
(allow network-inbound)
(allow network-outbound (remote ip "localhost:*"))
"""


def file_path(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise AssertionError(f"JDTLS returned a nonlocal project URI: {uri}")
    return Path(unquote(parsed.path)).resolve()


def confined(path, root):
    resolved = Path(path).resolve()
    root = root.resolve()
    if not resolved.is_relative_to(root):
        raise AssertionError(f"JDTLS returned a source outside the fixture: {resolved}")
    return resolved.relative_to(root).as_posix()


class JdtlsClient:
    def __init__(
        self,
        launcher,
        java_home,
        root,
        data,
        deny_external_network=False,
        gradle_user_home=None,
    ):
        environment = os.environ.copy()
        environment["JAVA_HOME"] = str(java_home)
        if gradle_user_home is not None:
            environment["GRADLE_USER_HOME"] = str(gradle_user_home)
        command = [
            str(launcher),
            f"--java-executable={java_home / 'bin' / ('java.exe' if os.name == 'nt' else 'java')}",
            "-data",
            str(data),
        ]
        if deny_external_network:
            environment["JAVA_TOOL_OPTIONS"] = (
                environment.get("JAVA_TOOL_OPTIONS", "") + " -Djava.net.preferIPv4Stack=true"
            ).strip()
            command = ["/usr/bin/sandbox-exec", "-p", GRADLE_DENIED_NETWORK_PROFILE, *command]
        self.stderr = tempfile.TemporaryFile()
        self.process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
        )
        self.root = root
        self.messages = queue.Queue()
        self.notifications = deque(maxlen=1000)
        self.notifications_lock = threading.Lock()
        self.next_id = 1
        self.reader = threading.Thread(target=self._read_messages, daemon=True)
        self.reader.start()

    def _read_messages(self):
        try:
            while True:
                headers = {}
                while True:
                    line = self.process.stdout.readline()
                    if not line:
                        raise EOFError("JDTLS closed its output")
                    if line in (b"\r\n", b"\n"):
                        break
                    name, value = line.decode().split(":", 1)
                    headers[name.lower()] = value.strip()
                length = int(headers["content-length"])
                body = self.process.stdout.read(length)
                if len(body) != length:
                    raise EOFError("JDTLS returned a partial message")
                self.messages.put(json.loads(body))
        except Exception as error:
            self.messages.put(error)

    def send(self, message):
        body = json.dumps(message, separators=(",", ":")).encode()
        self.process.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        self.process.stdin.flush()

    def request(self, method, params, timeout=45):
        request_id = self.next_id
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            try:
                message = self.messages.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty as error:
                raise TimeoutError(f"JDTLS timed out on {method}") from error
            if isinstance(message, Exception):
                raise message
            if message.get("id") == request_id and "method" not in message:
                if "error" in message:
                    raise RuntimeError(f"JDTLS {method} failed: {message['error']}")
                return message.get("result")
            if "id" in message and "method" in message:
                result = None
                if message["method"] == "workspace/configuration":
                    result = [{} for _ in message.get("params", {}).get("items", [])]
                elif message["method"] == "workspace/workspaceFolders":
                    result = [{"uri": self.root.as_uri(), "name": self.root.name}]
                elif message["method"] == "workspace/applyEdit":
                    result = {"applied": False}
                self.send({"jsonrpc": "2.0", "id": message["id"], "result": result})
            elif "method" in message:
                with self.notifications_lock:
                    self.notifications.append(message)

    def notify(self, method, params):
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def clear_document_notifications(self, uri):
        with self.notifications_lock:
            self.notifications = deque(
                (message for message in self.notifications
                 if message.get("params", {}).get("uri") != uri),
                maxlen=1000,
            )

    def close(self):
        if self.process.poll() is None:
            try:
                self.request("shutdown", None, timeout=10)
                self.notify("exit", None)
                self.process.wait(timeout=5)
            except (EOFError, TimeoutError, BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.stderr.close()


def discover(client, root, expected_sources):
    deadline = time.monotonic() + 180
    last_error = None
    while time.monotonic() < deadline:
        try:
            projects = client.request(
                "workspace/executeCommand",
                {"command": "java.project.getAll", "arguments": []},
            )
            modules = {confined(file_path(uri), root): uri for uri in projects or []}
            if set(modules) != {"application", "library"}:
                raise AssertionError(f"Expected two fixture modules, got {sorted(modules)}")
            settings = {}
            for module, uri in modules.items():
                response = client.request(
                    "workspace/executeCommand",
                    {"command": "java.project.getSettings", "arguments": [uri, SETTINGS_KEYS]},
                )
                paths = {
                    confined(path, root)
                    for path in response["org.eclipse.jdt.ls.core.sourcePaths"]
                }
                if not expected_sources[module].issubset(paths):
                    raise AssertionError(
                        f"{module} source roots missing: {sorted(expected_sources[module] - paths)}"
                    )
                if not response.get("org.eclipse.jdt.ls.core.vm.location"):
                    raise AssertionError(f"{module} has no project VM")
                if not isinstance(response.get("org.eclipse.jdt.ls.core.classpathEntries"), list):
                    raise AssertionError(f"{module} has no classpath entries")
                settings[module] = response
            return settings
        except (AssertionError, KeyError, TypeError, RuntimeError) as error:
            last_error = error
            time.sleep(2)
    raise AssertionError(f"JDTLS import did not reach the expected state: {last_error}")


def verify_generated_navigation(client, root):
    source = root / "application/src/main/java/evidence/Main.java"
    lines = source.read_text().splitlines()
    row, line = next((row, line) for row, line in enumerate(lines) if "BuildMessage" in line)
    client.notify(
        "textDocument/didOpen",
        {
            "textDocument": {
                "uri": source.as_uri(),
                "languageId": "java",
                "version": 1,
                "text": source.read_text(),
            }
        },
    )
    result = client.request(
        "textDocument/definition",
        {
            "textDocument": {"uri": source.as_uri()},
            "position": {"line": row, "character": line.index("BuildMessage") + 2},
        },
    )
    locations = result if isinstance(result, list) else [result]
    expected = "application/build/generated/sources/groovyFixture/java/evidence/BuildMessage.java"
    if not any(
        location and confined(file_path(location["uri"]), root) == expected
        for location in locations
    ):
        raise AssertionError(f"JDTLS did not navigate to generated {expected}: {result}")


def wait_for_java_diagnostics(client, uri, version, has_error):
    deadline = time.monotonic() + 45
    last_diagnostics = None
    while time.monotonic() < deadline:
        client.request("textDocument/documentSymbol", {"textDocument": {"uri": uri}})
        with client.notifications_lock:
            notifications = list(client.notifications)
        for notification in reversed(notifications):
            if notification.get("method") != "textDocument/publishDiagnostics":
                continue
            params = notification.get("params", {})
            if params.get("uri") != uri or params.get("version") not in (None, version):
                continue
            diagnostics = params.get("diagnostics", [])
            last_diagnostics = diagnostics
            if any(diagnostic.get("severity") == 1 for diagnostic in diagnostics) == has_error:
                return diagnostics, params.get("version") == version
            break
        time.sleep(0.1)
    raise AssertionError(f"Java diagnostics did not reach the requested state: {last_diagnostics}")


def apply_java_probe_quick_fix(source, uri, actions):
    if not source.isascii():
        raise AssertionError("The bounded Java quick-fix probe requires ASCII source")
    lines = source.splitlines(keepends=True)
    for action in actions:
        changes = (action.get("edit") or {}).get("changes", {})
        if set(changes) != {uri}:
            continue
        edits = changes[uri]
        if not any("private static String missingMessage()" in edit.get("newText", "")
                   for edit in edits):
            continue
        if not 1 <= len(edits) <= 32:
            raise AssertionError("The Java quick fix exceeded the probe edit limit")
        replacements = []
        for edit in edits:
            offsets = []
            for boundary in ("start", "end"):
                position = edit["range"][boundary]
                row, character = position["line"], position["character"]
                if not 0 <= row < len(lines) or not 0 <= character <= len(lines[row].rstrip("\r\n")):
                    raise AssertionError("The Java quick fix has an invalid source range")
                offsets.append(sum(len(line) for line in lines[:row]) + character)
            start, end = offsets
            if start > end or not isinstance(edit.get("newText"), str):
                raise AssertionError("The Java quick fix has an invalid replacement")
            replacements.append((start, end, edit["newText"]))
        previous_start = len(source) + 1
        repaired = source
        for start, end, replacement in sorted(replacements, reverse=True):
            if end > previous_start:
                raise AssertionError("The Java quick fix contains overlapping edits")
            repaired = repaired[:start] + replacement + repaired[end:]
            previous_start = start
        return repaired
    raise AssertionError("JDTLS offered no confined missing-method creation edit")


def verify_java_language_features(client, root):
    source = root / "application/src/main/java/evidence/LanguageProbe.java"
    uri = source.as_uri()
    original = '''package evidence;
public final class LanguageProbe {
    public static String message() { return "ready"; }
    public static void main(String[] arguments) {
        System.out.println(message());
    }
}
'''
    source.write_text(original)
    client.notify("textDocument/didOpen", {
        "textDocument": {"uri": uri, "languageId": "java", "version": 1, "text": original}
    })
    try:
        _, initial_versioned = wait_for_java_diagnostics(client, uri, 1, False)
        lines = original.splitlines()
        row, line = next((row, line) for row, line in enumerate(lines) if "println" in line)
        completion = client.request("textDocument/completion", {
            "textDocument": {"uri": uri},
            "position": {"line": row, "character": line.index("println") + 2},
            "context": {"triggerKind": 1},
        })
        items = completion if isinstance(completion, list) else (completion or {}).get("items", [])
        if not any(item.get("label", "").startswith("println") for item in items):
            raise AssertionError("JDTLS did not complete the Java println method")
        definitions = client.request("textDocument/definition", {
            "textDocument": {"uri": uri},
            "position": {"line": row, "character": line.index("message()") + 2},
        })
        locations = definitions if isinstance(definitions, list) else [definitions]
        if not any(
            location
            and confined(file_path(location.get("uri", location.get("targetUri", ""))), root)
                == "application/src/main/java/evidence/LanguageProbe.java"
            and location.get("range", location.get("targetSelectionRange", {}))
                .get("start", {}).get("line") == 2
            for location in locations
        ):
            raise AssertionError(f"JDTLS did not resolve the current Java method: {definitions}")
        invalid = original.replace('return "ready";', "return missingMessage();")
        client.clear_document_notifications(uri)
        client.notify("textDocument/didChange", {
            "textDocument": {"uri": uri, "version": 2},
            "contentChanges": [{"text": invalid}],
        })
        diagnostics, invalid_versioned = wait_for_java_diagnostics(client, uri, 2, True)
        errors = [diagnostic for diagnostic in diagnostics if diagnostic.get("severity") == 1]
        if not any(error.get("range", {}).get("start", {}).get("line") == 2 for error in errors):
            raise AssertionError(f"JDTLS error did not point at the edited method: {errors}")
        actions = client.request("textDocument/codeAction", {
            "textDocument": {"uri": uri},
            "range": next(error["range"] for error in errors if error["range"]["start"]["line"] == 2),
            "context": {"diagnostics": errors, "only": ["quickfix"]},
        })
        quick_fixes = [action for action in actions or []
                      if action.get("kind", "").startswith("quickfix")
                      and not action.get("disabled")
                      and (action.get("edit") or action.get("command") or action.get("data"))]
        if not quick_fixes:
            raise AssertionError(f"JDTLS offered no enabled quick fix for the current error: {actions}")
        repaired = apply_java_probe_quick_fix(invalid, uri, quick_fixes)
        if repaired == invalid or "private static String missingMessage()" not in repaired:
            raise AssertionError("The Java quick fix did not create the missing method")
        client.clear_document_notifications(uri)
        client.notify("textDocument/didChange", {
            "textDocument": {"uri": uri, "version": 3},
            "contentChanges": [{"text": repaired}],
        })
        _, fixed_versioned = wait_for_java_diagnostics(client, uri, 3, False)
        repaired_lines = repaired.splitlines()
        method_row = next(row for row, line in enumerate(repaired_lines)
                          if "private static String missingMessage()" in line)
        call_row, call_line = next((row, line) for row, line in enumerate(repaired_lines)
                                  if "return missingMessage();" in line)
        definitions = client.request("textDocument/definition", {
            "textDocument": {"uri": uri},
            "position": {"line": call_row, "character": call_line.index("missingMessage()") + 2},
        })
        locations = definitions if isinstance(definitions, list) else [definitions]
        if not any(
            location
            and location.get("uri", location.get("targetUri")) == uri
            and location.get("range", location.get("targetSelectionRange", {}))
                .get("start", {}).get("line") == method_row
            for location in locations
        ):
            raise AssertionError("JDTLS did not resolve the method created by its quick fix")
        client.clear_document_notifications(uri)
        client.notify("textDocument/didChange", {
            "textDocument": {"uri": uri, "version": 4},
            "contentChanges": [{"text": original}],
        })
        _, recovered_versioned = wait_for_java_diagnostics(client, uri, 4, False)
        return {
            "completion": True,
            "method_definition": True,
            "invalid_edit_diagnostics": True,
            "offered_quick_fixes": len(quick_fixes),
            "recovery_diagnostics": True,
            "quick_fix_applied": True,
            "quick_fix_diagnostics": True,
            "quick_fix_definition": True,
            "diagnostics_versioned": (initial_versioned and invalid_versioned
                                      and fixed_versioned and recovered_versioned),
        }
    finally:
        client.notify("textDocument/didClose", {"textDocument": {"uri": uri}})


def verify_maven_dependency_attributes(settings):
    dependencies = settings["application"]["org.eclipse.jdt.ls.core.classpathEntries"]
    if not any(
        entry.get("kind") == 1
        and isinstance(entry.get("attributes"), dict)
        and entry["attributes"].get("maven.groupId") == "org.junit.jupiter"
        and entry["attributes"].get("maven.artifactId") == "junit-jupiter"
        and entry["attributes"].get("maven.version") == "5.11.4"
        and entry["attributes"].get("maven.scope") == "test"
        for entry in dependencies
    ):
        raise AssertionError("JDTLS did not report the fixture's Maven artifact and scope")


def verify_gradle_dependency_source_sets(dependencies):
    if not any(
        entry.get("kind") == 1
        and isinstance(entry.get("attributes"), dict)
        and entry["attributes"].get("gradle_used_by_scope") == "main,test"
        for entry in dependencies
    ):
        raise AssertionError("JDTLS did not report Gradle classpath source-set use")


def verify_mixed_language_navigation(client, root, name):
    kotlin_class = root / (
        "library/target/classes/evidence/Greeting.class"
        if name == "maven"
        else "library/build/classes/kotlin/main/evidence/Greeting.class"
    )
    source = root / "application/src/main/java/evidence/Main.java"
    lines = source.read_text().splitlines()
    row, line = next((row, line) for row, line in enumerate(lines) if "Greeting" in line)
    client.notify(
        "textDocument/didOpen",
        {
            "textDocument": {
                "uri": source.as_uri(),
                "languageId": "java",
                "version": 1,
                "text": source.read_text(),
            }
        },
    )
    expected = "library/src/main/kotlin/evidence/Greeting.kt"
    deadline = time.monotonic() + 45
    result = None
    definition_found = False
    while time.monotonic() < deadline:
        result = client.request(
            "textDocument/definition",
            {
                "textDocument": {"uri": source.as_uri()},
                "position": {"line": row, "character": line.index("Greeting") + 2},
            },
        )
        locations = result if isinstance(result, list) else [result]
        definition_found = any(
            location
            and (uri := location.get("uri", location.get("targetUri")))
            and confined(file_path(uri), root) == expected
            for location in locations
        )
        if definition_found:
            break
        time.sleep(2)
    class_preserved = kotlin_class.is_file()
    print(
        json.dumps(
            {
                "fixture": f"{name}_mixed_language",
                "kotlin_class_after_import": class_preserved,
                "java_kotlin_definition": definition_found,
            }
        ),
        flush=True,
    )
    if not class_preserved or not definition_found:
        raise AssertionError(f"JDTLS did not retain and resolve the {name}-built Kotlin class from Java: {result}")


def run_offline_maven_failure(launcher, java_home):
    with tempfile.TemporaryDirectory(prefix="groovy-jdtls-maven-offline-") as directory:
        directory = Path(directory)
        root = directory / "maven"
        shutil.copytree(FIXTURES / "groovy_jvm_maven", root)
        repository = directory / "empty-maven-repository"
        repository.mkdir()
        settings_path = directory / "settings.xml"
        settings_path.write_text(
            '<settings xmlns="http://maven.apache.org/SETTINGS/1.2.0">'
            f"<localRepository>{repository}</localRepository></settings>"
        )
        (root / ".mvn/maven.config").write_text(
            f"-o\n-Dmaven.repo.local={repository}\n"
        )
        java_settings = {
            "import": {
                "maven": {"enabled": True, "offline": {"enabled": True}},
                "gradle": {"enabled": True},
            },
            "configuration": {"maven": {"userSettings": str(settings_path)}},
            "jdt": {"ls": {"kotlinSupport": {"enabled": True}}},
        }
        client = JdtlsClient(launcher, java_home, root, directory / "jdtls-data")
        started = time.monotonic()
        try:
            client.request(
                "initialize",
                {
                    "processId": os.getpid(),
                    "rootUri": root.as_uri(),
                    "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
                    "capabilities": {
                        "workspace": {"workspaceFolders": True, "configuration": True},
                        "window": {"workDoneProgress": True},
                    },
                    "initializationOptions": {"settings": {"java": java_settings}},
                },
                timeout=90,
            )
            client.notify("initialized", {})
            client.notify(
                "workspace/didChangeConfiguration",
                {"settings": {"java": java_settings}},
            )
            deadline = time.monotonic() + 120
            projects = None
            while time.monotonic() < deadline:
                projects = client.request(
                    "workspace/executeCommand",
                    {"command": "java.project.getAll", "arguments": []},
                    timeout=15,
                )
                with client.notifications_lock:
                    notifications = list(client.notifications)
                statuses = [
                    item.get("params", {})
                    for item in notifications
                    if item.get("method") == "language/status"
                ]
                logs = [
                    item.get("params", {}).get("message", "")
                    for item in notifications
                    if item.get("method") == "window/logMessage"
                ]
                warning = any(
                    status.get("type") == "ProjectStatus"
                    and status.get("message") == "WARNING"
                    for status in statuses
                )
                ready = any(status.get("type") == "ServiceReady" for status in statuses)
                missing_offline_dependency = any(
                    "in offline mode" in log and "has not been downloaded" in log
                    for log in logs
                )
                if warning and ready and missing_offline_dependency:
                    for _ in range(3):
                        time.sleep(2)
                        projects = client.request(
                            "workspace/executeCommand",
                            {"command": "java.project.getAll", "arguments": []},
                            timeout=15,
                        )
                        if projects:
                            raise AssertionError(
                                "JDTLS exposed Maven modules after the failed offline import"
                            )
                    break
                time.sleep(2)
            else:
                raise AssertionError(
                    "JDTLS did not report a terminal offline Maven dependency failure"
                )
            if projects:
                raise AssertionError("JDTLS exposed Maven modules despite failed offline import")
            if any(repository.rglob("*.jar")):
                raise AssertionError("Offline JDTLS import populated the empty Maven repository")
            print(
                json.dumps(
                    {
                        "fixture": "maven_offline_empty_cache",
                        "project_status": "WARNING",
                        "offline_dependency_log": True,
                        "modules": 0,
                        "downloaded_jars": 0,
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                    }
                )
            )
        finally:
            client.close()


def verify_gradle_network_boundary():
    probe = """import socket
listener = socket.socket()
listener.bind(('127.0.0.1', 0))
listener.listen(1)
with socket.create_connection(listener.getsockname(), timeout=2) as connection:
    accepted, _ = listener.accept()
    accepted.close()
listener.close()
try:
    socket.create_connection(('1.1.1.1', 443), timeout=2)
except PermissionError:
    pass
else:
    raise AssertionError('External TCP was not denied')
"""
    subprocess.run(
        [
            "/usr/bin/sandbox-exec", "-p", GRADLE_DENIED_NETWORK_PROFILE,
            sys.executable, "-c", probe,
        ],
        check=True,
        timeout=15,
    )


def run_offline_gradle_failure(launcher, java_home, gradle_home):
    if sys.platform != "darwin":
        raise ValueError("The denied-network Gradle probe requires macOS Seatbelt")
    if not (gradle_home / "lib").is_dir():
        raise ValueError("Gradle home must contain a lib directory")
    verify_gradle_network_boundary()
    with tempfile.TemporaryDirectory(prefix="groovy-jdtls-gradle-offline-") as directory:
        directory = Path(directory)
        root = directory / "gradle"
        shutil.copytree(FIXTURES / "groovy_jvm_gradle", root)
        user_home = directory / "empty-gradle-home"
        user_home.mkdir()
        java_settings = {
            "import": {
                "maven": {"enabled": False},
                "gradle": {
                    "enabled": True,
                    "offline": {"enabled": True},
                    "wrapper": {"enabled": False},
                    "home": str(gradle_home),
                    "user": {"home": str(user_home)},
                    "arguments": "--offline",
                },
            },
            "jdt": {"ls": {"kotlinSupport": {"enabled": True}}},
        }
        client = JdtlsClient(
            launcher, java_home, root, directory / "jdtls-data", deny_external_network=True
        )
        started = time.monotonic()
        try:
            client.request(
                "initialize",
                {
                    "processId": os.getpid(),
                    "rootUri": root.as_uri(),
                    "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
                    "capabilities": {
                        "workspace": {"workspaceFolders": True, "configuration": True},
                        "window": {"workDoneProgress": True},
                    },
                    "initializationOptions": {"settings": {"java": java_settings}},
                },
                timeout=90,
            )
            client.notify("initialized", {})
            client.notify("workspace/didChangeConfiguration", {"settings": {"java": java_settings}})
            deadline = time.monotonic() + 180
            projects = None
            statuses = []
            logs = []
            while time.monotonic() < deadline:
                try:
                    projects = client.request(
                        "workspace/executeCommand",
                        {"command": "java.project.getAll", "arguments": []},
                        timeout=15,
                    )
                except TimeoutError:
                    projects = None
                with client.notifications_lock:
                    notifications = list(client.notifications)
                statuses = [
                    item.get("params", {})
                    for item in notifications
                    if item.get("method") == "language/status"
                ]
                logs = [
                    item.get("params", {}).get("message", "")
                    for item in notifications
                    if item.get("method") == "window/logMessage"
                ]
                warning = any(
                    status.get("type") == "ProjectStatus"
                    and status.get("message") == "WARNING"
                    for status in statuses
                )
                ready = any(status.get("type") == "ServiceReady" for status in statuses)
                missing_plugin = any(
                    "UnknownPluginException" in log
                    and "org.jetbrains.kotlin.jvm" in log
                    and "2.0.21" in log
                    for log in logs
                )
                if warning and ready and missing_plugin:
                    for _ in range(3):
                        time.sleep(2)
                        projects = client.request(
                            "workspace/executeCommand",
                            {"command": "java.project.getAll", "arguments": []},
                            timeout=15,
                        )
                        if projects:
                            raise AssertionError(
                                "JDTLS exposed Gradle modules after the missing-plugin failure"
                            )
                    break
                time.sleep(2)
            else:
                raise AssertionError(
                    f"JDTLS did not report a missing Gradle plugin with project warning: "
                    f"status={statuses[-4:]}, logs={[log[-300:] for log in logs[-3:]]}"
                )
            print(
                json.dumps(
                    {
                        "fixture": "gradle_offline_empty_cache_external_denied",
                        "project_status": "WARNING",
                        "missing_plugin": "org.jetbrains.kotlin.jvm:2.0.21",
                        "modules": 0,
                        "local_daemon_socket_allowed": True,
                        "external_tcp_denied": True,
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                    }
                )
            )
        finally:
            client.close()


def run_cached_gradle_recovery(launcher, java_home, gradle_home):
    if sys.platform != "darwin":
        raise ValueError("The denied-network Gradle probe requires macOS Seatbelt")
    if not (gradle_home / "lib").is_dir():
        raise ValueError("Gradle home must contain a lib directory")
    verify_gradle_network_boundary()
    with tempfile.TemporaryDirectory(prefix="groovy-jdtls-gradle-recovery-") as directory:
        directory = Path(directory)
        root = directory / "gradle"
        shutil.copytree(FIXTURES / "groovy_jvm_gradle", root)
        user_home = directory / "gradle-home"
        user_home.mkdir()
        expected_sources = {
            "application": {
                "application/src/main/java",
                "application/build/generated/sources/groovyFixture/java",
            },
            "library": {"library/src/main/kotlin"},
        }
        stages = []
        try:
            for stage, offline in (("warm", False), ("cached_denied", True)):
                gradle_settings = {
                    "enabled": True,
                    "offline": {"enabled": offline},
                    "wrapper": {"enabled": False},
                    "home": str(gradle_home),
                    "user": {"home": str(user_home)},
                }
                if offline:
                    gradle_settings["arguments"] = "--offline"
                java_settings = {
                    "import": {"maven": {"enabled": False}, "gradle": gradle_settings},
                    "jdt": {"ls": {"kotlinSupport": {"enabled": True}}},
                }
                client = JdtlsClient(
                    launcher,
                    java_home,
                    root,
                    directory / f"jdtls-{stage}",
                    deny_external_network=offline,
                    gradle_user_home=user_home,
                )
                started = time.monotonic()
                try:
                    client.request(
                        "initialize",
                        {
                            "processId": os.getpid(),
                            "rootUri": root.as_uri(),
                            "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
                            "capabilities": {
                                "workspace": {"workspaceFolders": True, "configuration": True},
                                "window": {"workDoneProgress": True},
                            },
                            "initializationOptions": {"settings": {"java": java_settings}},
                        },
                        timeout=90,
                    )
                    client.notify("initialized", {})
                    client.notify(
                        "workspace/didChangeConfiguration",
                        {"settings": {"java": java_settings}},
                    )
                    settings = discover(client, root, expected_sources)
                    cached_jars = list(user_home.rglob("*.jar"))
                    if not any("kotlin-gradle-plugin" in path.name for path in cached_jars):
                        raise AssertionError(
                            "The isolated Gradle home did not cache the Kotlin build plugin"
                        )
                    verify_gradle_dependency_source_sets(
                        settings["application"]["org.eclipse.jdt.ls.core.classpathEntries"]
                    )
                    verify_generated_navigation(client, root)
                    stages.append(
                        {
                            "stage": stage,
                            "modules": sorted(settings),
                            "cached_jars": len(cached_jars),
                            "generated_navigation": True,
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    )
                finally:
                    client.close()
        finally:
            environment = os.environ.copy()
            environment["JAVA_HOME"] = str(java_home)
            environment["GRADLE_USER_HOME"] = str(user_home)
            subprocess.run(
                [str(gradle_home / "bin" / "gradle"), "--stop"],
                env=environment,
                check=True,
                timeout=30,
            )
        print(
            json.dumps(
                {
                    "fixture": "gradle_cached_recovery_external_denied",
                    "stages": stages,
                    "local_daemon_socket_allowed": True,
                    "external_tcp_denied_on_restart": True,
                }
            )
        )


def run_fixture(
    name,
    launcher,
    java_home,
    require_mixed_language_navigation=False,
    require_gradle_kotlin_navigation=False,
    require_java_language_features=False,
):
    with tempfile.TemporaryDirectory(prefix=f"groovy-jdtls-{name}-") as directory:
        root = (Path(directory) / name).resolve()
        shutil.copytree(FIXTURES / f"groovy_jvm_{name}", root)
        if name == "maven" and require_mixed_language_navigation:
            subprocess.run(
                [str(root / "mvnw"), "-o", "-q", "-DskipTests", "package"],
                cwd=root,
                check=True,
                timeout=180,
            )
            if not (root / "library/target/classes/evidence/Greeting.class").is_file():
                raise AssertionError("Maven did not compile the fixture's Kotlin class")
        if name == "gradle" and require_gradle_kotlin_navigation:
            subprocess.run(
                [str(root / "gradlew"), "--offline", ":application:build"],
                cwd=root,
                check=True,
                timeout=180,
            )
            if not (root / "library/build/classes/kotlin/main/evidence/Greeting.class").is_file():
                raise AssertionError("Gradle did not compile the fixture's Kotlin class")
        import_settings = {"maven": {"enabled": True}, "gradle": {"enabled": True}}
        java_settings = {
            "import": import_settings,
            "jdt": {"ls": {"kotlinSupport": {"enabled": True}}},
        }
        client = JdtlsClient(launcher, java_home, root, Path(directory) / "jdtls-data")
        try:
            initialized = client.request(
                "initialize",
                {
                    "processId": os.getpid(),
                    "rootUri": root.as_uri(),
                    "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
                    "capabilities": {
                        "workspace": {"workspaceFolders": True, "configuration": True},
                        "window": {"workDoneProgress": True},
                        "textDocument": {
                            "codeAction": {
                                "codeActionLiteralSupport": {
                                    "codeActionKind": {"valueSet": ["quickfix"]}
                                }
                            }
                        },
                    },
                    "initializationOptions": {
                        "settings": {"java": java_settings}
                    },
                },
                timeout=90,
            )
            commands = initialized["capabilities"]["executeCommandProvider"]["commands"]
            if not {"java.project.getAll", "java.project.getSettings"}.issubset(commands):
                raise AssertionError("JDTLS did not advertise the required project commands")
            client.notify("initialized", {})
            client.notify(
                "workspace/didChangeConfiguration",
                {"settings": {"java": java_settings}},
            )
            if name == "maven":
                expected = {
                    "application": {"application/src/main/java", "application/src/test/java"},
                    "library": {"library/src/main/kotlin"},
                }
            else:
                expected = {
                    "application": {
                        "application/src/main/java",
                        "application/build/generated/sources/groovyFixture/java",
                    },
                    "library": {"library/src/main/kotlin"},
                }
            settings = discover(client, root, expected)
            if name == "maven":
                verify_maven_dependency_attributes(settings)
                if require_mixed_language_navigation:
                    verify_mixed_language_navigation(client, root, name)
            else:
                dependencies = settings["application"]["org.eclipse.jdt.ls.core.classpathEntries"]
                if not any(
                    entry.get("kind") == 2 and entry.get("path") == "/library"
                    for entry in dependencies
                ):
                    raise AssertionError("Gradle application has no library project dependency")
                verify_gradle_dependency_source_sets(dependencies)
                verify_generated_navigation(client, root)
                if require_gradle_kotlin_navigation:
                    verify_mixed_language_navigation(client, root, name)
            java_features = (
                verify_java_language_features(client, root)
                if require_java_language_features else None
            )
            project_vms = {
                module: settings[module]["org.eclipse.jdt.ls.core.vm.location"]
                for module in sorted(settings)
            }
            source_levels = {
                module: settings[module].get("org.eclipse.jdt.core.compiler.source")
                for module in sorted(settings)
            }
            for module, level in source_levels.items():
                major = level.removeprefix("1.") if isinstance(level, str) else ""
                if not major.isdecimal() or not 1 <= int(major) <= 99:
                    raise AssertionError(f"{module} has no bounded Java source level: {level!r}")
            target_levels = {
                module: settings[module].get("org.eclipse.jdt.core.compiler.codegen.targetPlatform")
                for module in sorted(settings)
            }
            for module, level in target_levels.items():
                major = level.removeprefix("1.") if isinstance(level, str) else ""
                if not major.isdecimal() or not 1 <= int(major) <= 99:
                    raise AssertionError(f"{module} has no bounded Java target level: {level!r}")
            print(
                json.dumps(
                    {
                        "fixture": name,
                        "modules": sorted(settings),
                        "project_vms": project_vms,
                        "source_levels": source_levels,
                        "target_levels": target_levels,
                        "java_language_features": java_features,
                        "generated_navigation": name == "gradle",
                        "maven_dependency_attributes": name == "maven",
                        "gradle_source_set_attributes": name == "gradle",
                    }
                )
            )
        finally:
            client.close()


def main():
    parser = argparse.ArgumentParser(description="Probe JDTLS project facts and optional Java language features")
    parser.add_argument("--jdtls", required=True, type=Path, help="Path to the official JDTLS launcher")
    parser.add_argument("--java-home", required=True, type=Path, help="JDK 21+ used to run JDTLS")
    parser.add_argument(
        "--require-mixed-language-navigation",
        action="store_true",
        help="Fail unless a Java source resolves the Maven-built Kotlin class",
    )
    parser.add_argument(
        "--require-gradle-kotlin-navigation",
        action="store_true",
        help="Fail unless a Java source resolves the cached Gradle-built Kotlin class",
    )
    parser.add_argument(
        "--require-java-language-features",
        action="store_true",
        help="Require Java completion, current-method navigation, diagnostics/recovery and applied quick-fix navigation",
    )
    failure_mode = parser.add_mutually_exclusive_group()
    failure_mode.add_argument(
        "--expect-maven-offline-failure",
        action="store_true",
        help="Require JDTLS to report a missing dependency from an empty offline Maven repository",
    )
    failure_mode.add_argument(
        "--expect-gradle-offline-failure",
        action="store_true",
        help="Require a Gradle missing-plugin failure with an empty cache and external TCP denied",
    )
    failure_mode.add_argument(
        "--require-gradle-cached-offline-recovery",
        action="store_true",
        help="Warm an isolated Gradle cache, then require import with external TCP denied",
    )
    parser.add_argument(
        "--gradle-home", type=Path, help="Installed Gradle home for offline Gradle probes"
    )
    arguments = parser.parse_args()
    if not arguments.jdtls.is_file():
        parser.error("JDTLS launcher does not exist")
    java = arguments.java_home / "bin" / ("java.exe" if os.name == "nt" else "java")
    if not java.is_file():
        parser.error("JDK executable does not exist")
    if arguments.expect_maven_offline_failure:
        run_offline_maven_failure(arguments.jdtls.resolve(), arguments.java_home.resolve())
        return
    if arguments.expect_gradle_offline_failure:
        if arguments.gradle_home is None:
            parser.error("--gradle-home is required for the Gradle offline failure probe")
        run_offline_gradle_failure(
            arguments.jdtls.resolve(),
            arguments.java_home.resolve(),
            arguments.gradle_home.resolve(),
        )
        return
    if arguments.require_gradle_cached_offline_recovery:
        if arguments.gradle_home is None:
            parser.error("--gradle-home is required for the Gradle cached recovery probe")
        run_cached_gradle_recovery(
            arguments.jdtls.resolve(),
            arguments.java_home.resolve(),
            arguments.gradle_home.resolve(),
        )
        return
    for name in ("maven", "gradle"):
        run_fixture(
            name,
            arguments.jdtls.resolve(),
            arguments.java_home.resolve(),
            arguments.require_mixed_language_navigation,
            arguments.require_gradle_kotlin_navigation,
            arguments.require_java_language_features,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"JDTLS provider probe failed: {error}", file=sys.stderr)
        sys.exit(1)
