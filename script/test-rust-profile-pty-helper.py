#!/usr/bin/env python3

import argparse
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import tempfile


def run_case(helper: Path, directory: Path, name: str, shell_command: str, expected_status: int) -> None:
    case_directory = directory / name
    case_directory.mkdir()
    socket_path = case_directory / "completion"
    transcript = case_directory / "terminal.txt"
    marker = case_directory / "ran"
    token = os.urandom(32)
    environment = os.environ.copy()
    environment["ZED_PROFILE_TASK_TOKEN"] = token.hex()

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        listener.listen(1)
        listener.settimeout(15)
        command = [
            "script",
            "-q",
            "-e",
            str(transcript),
            str(helper),
            "observe-task",
            "--socket",
            str(socket_path),
            "--",
            "/bin/sh",
            "-c",
            shell_command,
            "probe",
            str(marker),
        ]
        process = subprocess.Popen(
            command,
            cwd=case_directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(15)
                received = bytearray()
                while len(received) < len(token):
                    part = connection.recv(len(token) - len(received))
                    if not part:
                        raise AssertionError(f"{name}: helper closed before registration")
                    received.extend(part)
                if bytes(received) != token:
                    raise AssertionError(f"{name}: helper sent a different registration token")
                connection.sendall(b"\x01")
            output, _ = process.communicate(timeout=15)
        except (OSError, subprocess.TimeoutExpired, AssertionError):
            process.kill()
            process.communicate(timeout=5)
            raise

    if token.hex() in output or token.hex() in transcript.read_text():
        raise AssertionError(f"{name}: registration token appeared in terminal output")
    if process.returncode != expected_status:
        raise AssertionError(
            f"{name}: PTY command returned {process.returncode}, expected {expected_status}: {output}"
        )
    if name == "success":
        if marker.read_text() != "pty-ok":
            raise AssertionError("success: helper did not execute the command inside the PTY")
    elif marker.exists():
        raise AssertionError(f"{name}: a failed command wrote an unexpected marker")
    print(f"{name}: registered, token hidden, PTY exit {process.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("helper", type=Path, help="built remote_server executable")
    arguments = parser.parse_args()
    helper = arguments.helper.resolve(strict=True)
    if not os.access(helper, os.X_OK):
        parser.error(f"helper is not executable: {helper}")
    digest = hashlib.sha256(helper.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="rust-profile-pty-", dir="/tmp") as temporary:
        directory = Path(temporary)
        run_case(
            helper,
            directory,
            "success",
            'test -t 1 && test -z "$ZED_PROFILE_TASK_TOKEN" && printf pty-ok > "$1"',
            0,
        )
        run_case(helper, directory, "failure", 'test -t 1 && exit 7', 7)
    print(f"remote_server sha256 {digest}")


if __name__ == "__main__":
    main()
