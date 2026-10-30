#!/usr/bin/env python3

import argparse
import hashlib
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path


PROTOCOL_REQUESTS = (
    ("GetCargoWorkspace", "get_cargo_workspace { project_id: 0 request_id: 101 }"),
    ("CancelCargoWorkspace", "cancel_cargo_workspace { project_id: 0 request_id: 101 }"),
    (
        "GetRustTestDiscovery",
        "get_rust_test_discovery { project_id: 0 request_id: 102 protocol_version: 1 }",
    ),
    (
        "CancelRustTestDiscovery",
        "cancel_rust_test_discovery { project_id: 0 request_id: 102 }",
    ),
    (
        "ResolveRustTestAction",
        'resolve_rust_test_action { project_id: 0 protocol_version: 1 discovery_generation: 1 node_id: "case" action: RUST_TEST_ACTION_KIND_RUN }',
    ),
    (
        "UpdateRustTestRun",
        'update_rust_test_run { project_id: 0 protocol_version: 1 discovery_generation: 1 run_id: "run" task_id: "task" }',
    ),
    (
        "GetStructuredExecutionSnapshot",
        'get_structured_execution_snapshot { project_id: 0 provider_id: "rust-tests" discovery_generation: 1 page_start: 0 page_size: 1 }',
    ),
    (
        "GetStructuredExecutionEvents",
        'get_structured_execution_events { project_id: 0 provider_id: "rust-tests" run_id: "run" chunk_size: 1 }',
    ),
    (
        "InterpretRustCoverageArtifact",
        "interpret_rust_coverage_artifact { project_id: 0 max_bytes: 1 }",
    ),
)


def run_protoc(proto_path: Path, operation: str, content: bytes) -> bytes:
    return subprocess.run(
        [
            "protoc",
            operation,
            "-I",
            str(proto_path.parent),
            str(proto_path),
        ],
        input=content,
        capture_output=True,
        check=True,
    ).stdout


def receive_exact(stream: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = stream.recv(size - len(result))
        if not chunk:
            raise RuntimeError("host closed the protocol stream before responding")
        result.extend(chunk)
    return bytes(result)


def receive_envelope(stream: socket.socket, proto_path: Path) -> str:
    length = struct.unpack("<I", receive_exact(stream, 4))[0]
    if length > 1024 * 1024:
        raise RuntimeError(f"host response exceeds probe limit: {length} bytes")
    return run_protoc(
        proto_path,
        "--decode=zed.messages.Envelope",
        receive_exact(stream, length),
    ).decode()


def wait_for_sockets(process: subprocess.Popen, directory: Path) -> None:
    socket_paths = [directory / f"{name}.sock" for name in ("stdin", "stdout", "stderr")]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"host exited during startup with code {process.returncode}")
        if all(path.exists() for path in socket_paths):
            return
        time.sleep(0.05)
    raise TimeoutError("host did not bind all three protocol sockets within 30 seconds")


def probe(directory: Path, proto_path: Path) -> None:
    with (
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as incoming,
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as outgoing,
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as logs,
    ):
        for stream, name in (
            (incoming, "stdin"),
            (outgoing, "stdout"),
            (logs, "stderr"),
        ):
            stream.settimeout(15)
            stream.connect(str(directory / f"{name}.sock"))

        for envelope_id, (request_name, payload) in enumerate(PROTOCOL_REQUESTS, 1):
            request = run_protoc(
                proto_path,
                "--encode=zed.messages.Envelope",
                f"id: {envelope_id}\n{payload}\n".encode(),
            )
            incoming.sendall(struct.pack("<I", len(request)) + request)
            for _ in range(20):
                response = receive_envelope(outgoing, proto_path)
                if f"responding_to: {envelope_id}" not in response:
                    continue
                if (
                    "error {" not in response
                    or f"no handler registered for {request_name}" not in response
                ):
                    raise AssertionError(f"{request_name}: unexpected response: {response}")
                print(f"{request_name}: no handler registered; response to {envelope_id}")
                break
            else:
                raise AssertionError(f"{request_name}: no matching response")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe a separately built feature-disabled remote_server process"
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    arguments = parser.parse_args()

    if shutil.which("protoc") is None:
        parser.error("protoc is required")
    binary = arguments.binary.resolve(strict=True)
    with binary.open("rb") as binary_file:
        actual_sha256 = hashlib.file_digest(binary_file, "sha256").hexdigest()
    if actual_sha256 != arguments.expected_sha256.lower():
        parser.error(f"binary SHA-256 mismatch: {actual_sha256}")

    proto_path = Path(__file__).resolve().parent.parent / "crates/proto/proto/zed.proto"
    temporary_root = "/private/tmp" if sys.platform == "darwin" else None
    with tempfile.TemporaryDirectory(prefix="rust-host-probe-", dir=temporary_root) as temporary:
        directory = Path(temporary)
        with (directory / "stdout.log").open("wb") as stdout_log, (
            directory / "stderr.log"
        ).open("wb") as stderr_log:
            process = subprocess.Popen(
                [
                    str(binary),
                    "run",
                    "--log-file",
                    str(directory / "server.log"),
                    "--pid-file",
                    str(directory / "server.pid"),
                    "--stdin-socket",
                    str(directory / "stdin.sock"),
                    "--stdout-socket",
                    str(directory / "stdout.sock"),
                    "--stderr-socket",
                    str(directory / "stderr.sock"),
                ],
                stdout=stdout_log,
                stderr=stderr_log,
            )
            try:
                wait_for_sockets(process, directory)
                probe(directory, proto_path)
                print(f"{len(PROTOCOL_REQUESTS)} disabled Rust handlers rejected by a separate host process")
                print(f"binary SHA-256: {actual_sha256}")
            except Exception:
                stderr_log.flush()
                print((directory / "stderr.log").read_text(errors="replace"), file=sys.stderr)
                raise
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
