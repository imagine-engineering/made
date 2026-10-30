import contextlib
import socket
import ssl
import subprocess
import tempfile
import threading
import unittest
import warnings
from pathlib import Path

import urllib3
from urllib3.util.ssl_match_hostname import CertificateError


class ProxyTlsPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="made-proxy-tls-")
        cls.addClassCleanup(cls.directory.cleanup)
        fixture = (
            Path(__file__).resolve().parents[2] / "crates/http_client_tls/test_data"
        )
        cls.certificate = Path(cls.directory.name) / "certificate.pem"
        cls.key = Path(cls.directory.name) / "key.pem"
        for command, source, destination in [
            ("x509", fixture / "server.der", cls.certificate),
            ("pkey", fixture / "server-key.der", cls.key),
        ]:
            subprocess.run(
                [
                    "openssl",
                    command,
                    "-inform",
                    "DER",
                    "-in",
                    str(source),
                    "-out",
                    str(destination),
                ],
                check=True,
                capture_output=True,
            )
        cls.server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        cls.server_context.load_cert_chain(cls.certificate, cls.key)

    @contextlib.contextmanager
    def proxy_listener(self, *, certificate_refusal_expected=False):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(0.1)
        address = listener.getsockname()
        stopped = threading.Event()
        errors = []

        def serve():
            while not stopped.is_set():
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                except OSError as error:
                    if not stopped.is_set():
                        errors.append(error)
                    return
                try:
                    connection.settimeout(2)
                    with self.server_context.wrap_socket(connection, server_side=True):
                        pass
                except ssl.SSLError as error:
                    if not certificate_refusal_expected:
                        errors.append(error)
                except OSError as error:
                    errors.append(error)
                finally:
                    connection.close()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        try:
            yield f"https://localhost:{address[1]}"
        finally:
            stopped.set()
            listener.close()
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive(), "owned TLS listener did not stop")
            self.assertEqual(errors, [], "unexpected local proxy TLS failure")

    def proxy_context(self):
        return ssl.create_default_context(cafile=str(self.certificate))

    def relaxed_target_context(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    def test_forwarding_preserves_explicit_proxy_context_and_reused_policy(self):
        proxy_context = self.proxy_context()
        target_context = self.relaxed_target_context()
        certificate_store = proxy_context.cert_store_stats()
        with self.proxy_listener() as proxy_url, warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with urllib3.ProxyManager(
                proxy_url,
                proxy_ssl_context=proxy_context,
                ssl_context=target_context,
                cert_reqs=ssl.CERT_NONE,
                use_forwarding_for_https=True,
                timeout=2,
            ) as manager:
                for target in ["https://first.invalid/", "https://second.invalid/"]:
                    connection = manager.connection_from_url(target)._new_conn()
                    try:
                        connection.connect()
                        self.assertIs(connection.sock.context, proxy_context)
                        self.assertTrue(connection.proxy_is_verified)
                        self.assertFalse(connection.is_verified)
                    finally:
                        connection.close()
                    self.assertEqual(proxy_context.verify_mode, ssl.CERT_REQUIRED)
                    self.assertEqual(
                        proxy_context.cert_store_stats(), certificate_store
                    )
        self.assertEqual(target_context.verify_mode, ssl.CERT_NONE)

    def test_forwarding_does_not_apply_target_identity_or_client_certificate(self):
        target_settings = [
            {"server_hostname": "unrelated-target.invalid"},
            {"assert_hostname": "unrelated-target.invalid"},
            {"assert_fingerprint": "00" * 32},
            {
                "cert_file": str(
                    Path(self.directory.name) / "missing-target-certificate.pem"
                )
            },
        ]
        for settings in target_settings:
            with self.subTest(settings=settings), self.proxy_listener() as proxy_url:
                proxy_context = self.proxy_context()
                with urllib3.ProxyManager(
                    proxy_url,
                    proxy_ssl_context=proxy_context,
                    cert_reqs=ssl.CERT_NONE,
                    use_forwarding_for_https=True,
                    timeout=2,
                    **settings,
                ) as manager:
                    connection = manager.connection_from_url(
                        "https://target.invalid/"
                    )._new_conn()
                    try:
                        connection.connect()
                        self.assertIs(connection.sock.context, proxy_context)
                        self.assertTrue(connection.proxy_is_verified)
                    finally:
                        connection.close()

    def test_untrusted_proxy_is_refused_with_relaxed_target_policy(self):
        proxy_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        with self.proxy_listener(certificate_refusal_expected=True) as proxy_url:
            with urllib3.ProxyManager(
                proxy_url,
                proxy_ssl_context=proxy_context,
                cert_reqs=ssl.CERT_NONE,
                use_forwarding_for_https=True,
                timeout=2,
            ) as manager:
                connection = manager.connection_from_url(
                    "https://target.invalid/"
                )._new_conn()
                try:
                    with self.assertRaises(
                        (ssl.SSLError, urllib3.exceptions.SSLError)
                    ) as failure:
                        connection.connect()
                    self.assertIn("CERTIFICATE_VERIFY_FAILED", str(failure.exception))
                finally:
                    connection.close()
        self.assertEqual(proxy_context.verify_mode, ssl.CERT_REQUIRED)

    def test_proxy_hostname_assertion_is_not_overridden_by_target_policy(self):
        with self.proxy_listener() as proxy_url:
            with urllib3.ProxyManager(
                proxy_url,
                proxy_ssl_context=self.proxy_context(),
                proxy_assert_hostname="wrong-proxy.invalid",
                assert_hostname="localhost",
                cert_reqs=ssl.CERT_NONE,
                use_forwarding_for_https=True,
                timeout=2,
            ) as manager:
                connection = manager.connection_from_url(
                    "https://target.invalid/"
                )._new_conn()
                try:
                    with self.assertRaises(CertificateError):
                        connection.connect()
                finally:
                    connection.close()

    def test_tunnel_proxy_tls_preserves_required_proxy_policy(self):
        proxy_context = self.proxy_context()
        proxy_context.check_hostname = False
        with self.proxy_listener() as proxy_url:
            with urllib3.ProxyManager(
                proxy_url,
                proxy_ssl_context=proxy_context,
                cert_reqs=ssl.CERT_NONE,
                timeout=2,
            ) as manager:
                connection = manager.connection_from_url(
                    "https://target.invalid/"
                )._new_conn()
                address = ("localhost", int(proxy_url.rsplit(":", 1)[1]))
                with socket.create_connection(address, timeout=2) as connection_socket:
                    with connection._connect_tls_proxy(
                        "localhost", connection_socket
                    ) as wrapped:
                        self.assertIs(wrapped.context, proxy_context)
                        self.assertTrue(connection.proxy_is_verified)
                        self.assertEqual(proxy_context.verify_mode, ssl.CERT_REQUIRED)


if __name__ == "__main__":
    unittest.main()
