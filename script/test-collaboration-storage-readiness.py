"""Offline tests of exact canonical readiness functions; never run the storage driver."""
import ast
import os
from pathlib import Path
import struct
import types
import unittest


SOURCE = Path(os.environ.get("STORAGE_READINESS_SOURCE", Path(__file__).resolve().with_name("validate-collaboration-storage")))
FUNCTIONS = {"remaining_probe_time", "receive_exact", "verify_host_protocol", "wait_host_ready"}
tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
if "wait_host_ready" not in {node.name for node in definitions}:
    raise RuntimeError("Canonical readiness owner absent")


class FragmentedSocket:
    def __init__(self, reply, fragment, fail_receive=False):
        self.reply = reply
        self.fragment = fragment
        self.fail_receive = fail_receive
        self.sent = []
        self.timeouts = []
        self.requests = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_arguments):
        self.closed = True

    def settimeout(self, seconds):
        if seconds <= 0:
            raise AssertionError("nonpositive socket timeout")
        self.timeouts.append(seconds)

    def sendall(self, message):
        self.sent.append(message)

    def recv(self, requested):
        self.requests.append(requested)
        if self.fail_receive:
            raise TimeoutError("fixture receive timed out")
        length = min(requested, self.fragment, len(self.reply))
        part, self.reply = self.reply[:length], self.reply[length:]
        return part


def authentication_frame(method=10, body=b"SCRAM-SHA-256\x00\x00", kind=b"R", declared=None):
    payload = struct.pack("!I", method) + body
    return kind + struct.pack("!I", len(payload) + 4 if declared is None else declared) + payload


class ReadinessFrames(unittest.TestCase):
    def prepare(self, reply, postgres=True, fragment=1, clock_step=0, fail_receive=False):
        connections = []
        sleeps = []
        clock = [-clock_step]

        def monotonic():
            clock[0] += clock_step
            return clock[0]

        def connect(address, timeout):
            self.assertEqual(address[0], "127.0.0.1")
            self.assertGreater(timeout, 0)
            connection = FragmentedSocket(reply, fragment, fail_receive)
            connections.append(connection)
            return connection

        namespace = {
            "socket": types.SimpleNamespace(create_connection=connect),
            "time": types.SimpleNamespace(monotonic=monotonic, sleep=sleeps.append),
            "struct": struct,
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), "exec"), namespace)
        return namespace["wait_host_ready"], connections, sleeps, postgres

    def run_probe(self, prepared):
        probe, _connections, _sleeps, postgres = prepared
        return probe("127.0.0.1:15432" if postgres else "127.0.0.1:16379", postgres)

    def reject(self, reply, postgres=True, fragment=1, **options):
        prepared = self.prepare(reply, postgres, fragment, **options)
        with self.assertRaises(RuntimeError):
            self.run_probe(prepared)
        self.assertEqual(len(prepared[1]), 30)
        self.assertTrue(all(connection.closed for connection in prepared[1]))
        self.assertEqual(prepared[2], [1] * 30)
        return prepared

    def test_postgres_fragmented_header_and_payload_accepted(self):
        prepared = self.prepare(authentication_frame())
        self.assertTrue(self.run_probe(prepared)["protocol_response_verified"])
        self.assertEqual(len(prepared[1]), 1)
        connection = prepared[1][0]
        self.assertEqual(connection.reply, b"")
        self.assertEqual(connection.sent[0][4:8], struct.pack("!I", 196608))
        self.assertEqual(struct.unpack("!I", connection.sent[0][:4])[0], len(connection.sent[0]))
        self.assertTrue(connection.closed)

    def test_postgres_multiple_scram_mechanisms_accepted(self):
        prepared = self.prepare(authentication_frame(body=b"SCRAM-SHA-256-PLUS\x00SCRAM-SHA-256\x00\x00"))
        self.assertTrue(self.run_probe(prepared)["protocol_response_verified"])

    def test_postgres_partial_header_denied(self):
        self.reject(b"R\x00")

    def test_postgres_header_without_payload_denied(self):
        self.reject(authentication_frame()[:5])

    def test_postgres_payload_truncated_denied(self):
        self.reject(authentication_frame()[:-1])

    def test_postgres_wrong_message_type_denied(self):
        self.reject(authentication_frame(kind=b"E"))

    def test_postgres_short_declared_frame_denied(self):
        self.reject(authentication_frame(declared=7))

    def test_postgres_overlong_declared_frame_denied_before_body(self):
        prepared = self.reject(authentication_frame(declared=1025))
        self.assertEqual(prepared[1][0].requests, [5, 4, 3, 2, 1])

    def test_postgres_wrong_authentication_method_denied(self):
        self.reject(authentication_frame(method=0))

    def test_postgres_unterminated_sasl_list_denied(self):
        self.reject(authentication_frame(body=b"SCRAM-SHA-256\x00"))

    def test_postgres_empty_sasl_list_denied(self):
        self.reject(authentication_frame(body=b"\x00\x00"))

    def test_postgres_duplicate_sasl_mechanism_denied(self):
        self.reject(authentication_frame(body=b"SCRAM-SHA-256\x00SCRAM-SHA-256\x00\x00"))

    def test_postgres_non_ascii_sasl_mechanism_denied(self):
        self.reject(authentication_frame(body=b"\xff\x00\x00"))

    def test_postgres_missing_expected_scram_mechanism_denied(self):
        self.reject(authentication_frame(body=b"OTHER\x00\x00"))

    def test_redis_fragmented_frame_accepted(self):
        prepared = self.prepare(b"+PONG\r\n", postgres=False)
        try:
            result = self.run_probe(prepared)
        except RuntimeError as error:
            self.fail(f"valid fragmented Redis PONG rejected: {error}")
        self.assertTrue(result["protocol_response_verified"])
        self.assertEqual(len(prepared[1]), 1)
        self.assertEqual(prepared[1][0].reply, b"")
        self.assertEqual(prepared[1][0].sent, [b"*1\r\n$4\r\nPING\r\n"])

    def test_redis_partial_crlf_denied(self):
        self.reject(b"+PONG\r", postgres=False)

    def test_redis_wrong_resp_denied(self):
        self.reject(b"-ERROR\r\n", postgres=False)

    def test_redis_oversized_resp_denied_with_bounded_read(self):
        prepared = self.reject(b"+" + b"X" * 4096, postgres=False, fragment=4096)
        self.assertEqual(prepared[1][0].requests, [7])

    def test_connection_receive_timeout_denied(self):
        self.reject(b"", fail_receive=True)

    def test_trickle_response_cannot_reset_whole_attempt_deadline(self):
        self.reject(authentication_frame(), clock_step=0.5)

    def test_non_loopback_refused_before_connection(self):
        prepared = self.prepare(authentication_frame())
        with self.assertRaisesRegex(RuntimeError, "only to loopback"):
            prepared[0]("192.0.2.1:5432", True)
        self.assertEqual(prepared[1], [])


if __name__ == "__main__":
    unittest.main()
