#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time


REPOSITORY = Path(__file__).resolve().parent.parent
FIXTURE = REPOSITORY / "crates/project/test_data/groovy_jvm_maven"
JDTLS_PROBE = Path(__file__).with_name("test-groovy-jdtls-provider.py")
BREAKPOINT_LINE = 7


def jdtls_client_class():
    specification = importlib.util.spec_from_file_location("groovy_jdtls_probe", JDTLS_PROBE)
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot load the existing JDTLS fixture client")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.JdtlsClient


class DapProbe:
    def __init__(self, port):
        self.connection = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.connection.settimeout(15)
        self.reader = self.connection.makefile("rb")
        self.sequence = 0
        self.process_id = None

    def request(self, command, arguments):
        self.sequence += 1
        body = json.dumps(
            {"seq": self.sequence, "type": "request", "command": command, "arguments": arguments}
        ).encode()
        self.connection.sendall(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        return self.sequence

    def receive(self):
        headers = {}
        while True:
            line = self.reader.readline()
            if not line:
                raise EOFError("Java Debug closed its DAP connection")
            if line == b"\r\n":
                break
            name, value = line.decode().split(":", 1)
            headers[name.lower()] = value.strip()
        message = json.loads(self.reader.read(int(headers["content-length"])))
        if message.get("event") == "processid":
            self.process_id = message.get("body", {}).get("processId")
        return message

    def until(self, predicate, timeout=45):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            message = self.receive()
            if message.get("type") == "response" and not message.get("success"):
                raise AssertionError(f"Java Debug {message.get('command')} failed: {message.get('message')}")
            if predicate(message):
                return message
        raise TimeoutError("Java Debug did not reach the expected DAP event")

    def response(self, sequence):
        return self.until(
            lambda message: message.get("type") == "response"
            and message.get("request_seq") == sequence
        )

    def close(self):
        self.reader.close()
        self.connection.close()


def probe(launcher, java_home, debug_jar, expected_sha256):
    digest = hashlib.sha256(debug_jar.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise AssertionError(f"Java Debug jar SHA-256 differs: {digest}")
    with tempfile.TemporaryDirectory(prefix="groovy-java-debug-") as directory:
        root = Path(directory) / "fixture"
        shutil.copytree(FIXTURE, root)
        source = root / "application/src/main/java/evidence/DebugMain.java"
        subprocess.run(
            [
                str(java_home / "bin/javac"),
                "-g",
                "-d",
                str(root / "application/target/classes"),
                str(source),
            ],
            check=True,
        )
        client = jdtls_client_class()(launcher, java_home, root, Path(directory) / "jdtls-data")
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
                    },
                    "initializationOptions": {
                        "bundles": [str(debug_jar)],
                        "settings": {"java": {"import": {"maven": {"enabled": True}}}},
                    },
                },
                timeout=90,
            )
            commands = initialized["capabilities"]["executeCommandProvider"]["commands"]
            if "vscode.java.startDebugSession" not in commands:
                raise AssertionError("JDTLS did not load the Java Debug bundle")
            client.notify("initialized", {})
            client.notify(
                "workspace/didChangeConfiguration",
                {"settings": {"java": {"import": {"maven": {"enabled": True}}}}},
            )
            main_classes = client.request(
                "workspace/executeCommand",
                {"command": "vscode.java.resolveMainClass", "arguments": ["evidence.DebugMain"]},
                timeout=60,
            )
            if not any(
                entry.get("mainClass") == "evidence.DebugMain"
                and entry.get("projectName") == "application"
                for entry in main_classes or []
            ):
                raise AssertionError(f"JDTLS did not resolve the fixture main: {main_classes}")
            classpath_groups = client.request(
                "workspace/executeCommand",
                {
                    "command": "vscode.java.resolveClasspath",
                    "arguments": ["evidence.DebugMain", "application", None],
                },
                timeout=60,
            )
            if not isinstance(classpath_groups, list) or not all(
                isinstance(group, list) for group in classpath_groups
            ):
                raise AssertionError(f"JDTLS returned invalid classpath groups: {classpath_groups}")
            classpaths = [entry for group in classpath_groups for entry in group]
            if not any(
                Path(entry).resolve() == (root / "application/target/classes").resolve()
                for entry in classpaths
            ):
                raise AssertionError("JDTLS omitted the fixture application classpath")
            port = client.request(
                "workspace/executeCommand",
                {"command": "vscode.java.startDebugSession"},
                timeout=60,
            )
            if not isinstance(port, int) or port <= 0:
                raise AssertionError(f"Java Debug did not open a DAP port: {port}")
            dap = DapProbe(port)
            try:
                dap.response(
                    dap.request(
                        "initialize",
                        {
                            "clientID": "groovy-provider-probe",
                            "adapterID": "java",
                            "linesStartAt1": True,
                            "columnsStartAt1": True,
                            "pathFormat": "path",
                        },
                    )
                )
                dap.request(
                    "launch",
                    {
                        "request": "launch",
                        "mainClass": "evidence.DebugMain",
                        "projectName": "application",
                        "classPaths": classpaths,
                        "cwd": str(root),
                        "stopOnEntry": False,
                        "console": "internalConsole",
                    },
                )
                dap.until(
                    lambda message: message.get("type") == "event"
                    and message.get("event") == "initialized"
                )
                breakpoint = dap.response(
                    dap.request(
                        "setBreakpoints",
                        {
                            "source": {"path": str(source)},
                            "breakpoints": [{"line": BREAKPOINT_LINE}],
                            "lines": [BREAKPOINT_LINE],
                            "sourceModified": False,
                        },
                    )
                )["body"]["breakpoints"]
                if len(breakpoint) != 1 or not breakpoint[0].get("verified"):
                    raise AssertionError(f"Java Debug did not verify the breakpoint: {breakpoint}")
                dap.response(dap.request("configurationDone", {}))
                stopped = dap.until(
                    lambda message: message.get("type") == "event"
                    and message.get("event") == "stopped",
                )
                if stopped.get("body", {}).get("reason") != "breakpoint":
                    raise AssertionError(f"Java Debug stopped for another reason: {stopped}")
                stack = dap.response(
                    dap.request("stackTrace", {"threadId": stopped["body"]["threadId"]})
                )["body"]["stackFrames"]
                if not stack:
                    raise AssertionError("Java Debug returned no stopped stack frame")
                frame = stack[0]
                # macOS may report the same temporary source under /var or /private/var.
                if Path(frame.get("source", {}).get("path", "")).resolve() != source.resolve():
                    raise AssertionError(f"Java Debug mapped the frame to another source: {frame}")
                if frame.get("line") != BREAKPOINT_LINE or dap.process_id is None:
                    raise AssertionError(f"Java Debug returned an unexpected frame or process: {frame}")
                result = {
                    "adapter_sha256": digest,
                    "process_id": dap.process_id,
                    "breakpoint_line": BREAKPOINT_LINE,
                    "frame": "application/src/main/java/evidence/DebugMain.java",
                    "frame_line": frame["line"],
                    "frame_name": frame["name"],
                }
                dap.response(dap.request("disconnect", {"terminateDebuggee": True}))
                return result
            finally:
                dap.close()
        finally:
            client.close()


def main():
    parser = argparse.ArgumentParser(description="Probe a pinned Java Debug adapter with the Groovy Maven fixture")
    parser.add_argument("--jdtls-launcher", required=True, type=Path)
    parser.add_argument("--java-home", required=True, type=Path)
    parser.add_argument("--debug-jar", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    arguments = parser.parse_args()
    result = probe(
        arguments.jdtls_launcher.resolve(),
        arguments.java_home.resolve(),
        arguments.debug_jar.resolve(),
        arguments.expected_sha256.lower(),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
