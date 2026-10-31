import contextlib
import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

import refresh_website as refresh


TAG = "rust-v1.16.6"
WEBHOOK = "https://webhooks.amplify.eu-west-2.amazonaws.com/prod/webhooks?id=example&token=private-token&operation=startbuild"
URL = "https://github.com/imagine-engineering/made/releases/download/rust-v1.16.6/actual-asset.dmg"
RELEASE = {"tag_name": TAG, "draft": False, "published_at": "2026-09-05T12:00:00Z", "assets": [{"state": "uploaded", "browser_download_url": URL}]}
HTML = '<article id="rust-v1.16.6"><span>Copper 1.16.6</span><time datetime="2026-09-05T12:00:00.000Z"></time><a href="' + URL + '">Download</a></article>'
API = (200, {}, json.dumps(RELEASE).encode())
CURRENT = (200, {}, HTML.encode())
STALE = (200, {}, b'<meta name="simtropolis-build" content="unchanged-commit">')


class RefreshTests(unittest.TestCase):
    def run_refresh(self, responses, **options):
        elapsed = [0]
        waits = []

        def sleep(seconds):
            elapsed[0] += seconds
            waits.append(seconds)

        output = io.StringIO()
        with patch.object(refresh, "request", side_effect=responses) as request, contextlib.redirect_stdout(output):
            try:
                result = refresh.refresh(TAG, WEBHOOK, opener=object(), sleep=sleep, monotonic=lambda: elapsed[0], deployment_timeout=15, **options)
            except refresh.RefreshError as error:
                result = error
        posts = [call for call in request.call_args_list if call.args[0].get_method() == "POST"]
        self.assertNotIn("private-token", output.getvalue())
        self.assertNotIn(WEBHOOK, output.getvalue())
        return result, posts, waits, output.getvalue()

    def test_invalid_configuration_fails_before_any_network(self):
        for webhook in ["", "http://example.com", WEBHOOK + "&token=second", WEBHOOK.replace("startbuild", "delete"), WEBHOOK.replace("amazonaws.com", "amazonaws.com.evil.test"), WEBHOOK + "#fragment", WEBHOOK.replace("example", ""), WEBHOOK.replace("private-token", "")]:
            with self.subTest(webhook=webhook), patch.object(refresh, "request") as request:
                with self.assertRaises(refresh.RefreshError):
                    refresh.refresh(TAG, webhook)
                request.assert_not_called()

    def test_already_current_content_avoids_duplicate_build(self):
        result, posts, _, log = self.run_refresh([API, CURRENT])
        self.assertEqual(result, "already-current")
        self.assertEqual(len(posts), 0)
        self.assertIn("already deployed", log)

    def test_acceptance_then_verified_content(self):
        result, posts, _, log = self.run_refresh([API, STALE, (202, {}, b""), CURRENT])
        self.assertEqual(result, "deployed")
        self.assertEqual(len(posts), 1)
        self.assertIn("Rebuild accepted", log)
        self.assertIn("exact asset URLs", log)

    def test_accepted_post_does_not_make_stale_deployment_successful(self):
        result, posts, _, _ = self.run_refresh([API, STALE, (200, {}, b""), STALE, STALE])
        self.assertIsInstance(result, refresh.RefreshError)
        self.assertEqual(len(posts), 1)
        self.assertIn("deployment was not verified", str(result))

    def test_transient_retry_respects_headers_and_verifies_content(self):
        for status in [429, 503]:
            result, posts, waits, _ = self.run_refresh([API, STALE, (status, {"retry-after": "12"}, b""), STALE, (200, {}, b""), CURRENT])
            self.assertEqual(result, "deployed")
            self.assertEqual(len(posts), 2)
            self.assertEqual(waits, [12])

    def test_content_appearing_before_retry_prevents_duplicate(self):
        result, posts, _, _ = self.run_refresh([API, STALE, (503, {}, b""), CURRENT])
        self.assertEqual(result, "deployed")
        self.assertEqual(len(posts), 1)

    def test_retry_attempts_and_wait_budget_are_bounded(self):
        result, posts, waits, _ = self.run_refresh([API, STALE, (503, {}, b""), STALE, (503, {}, b""), STALE, (503, {}, b"")])
        self.assertIsInstance(result, refresh.RefreshError)
        self.assertEqual(len(posts), 3)
        self.assertEqual(waits, [2, 4])
        result, posts, waits, _ = self.run_refresh([API, STALE, (429, {"retry-after": "121"}, b"")])
        self.assertIsInstance(result, refresh.RefreshError)
        self.assertEqual(len(posts), 1)
        self.assertEqual(waits, [])

    def test_retry_after_http_date_and_invalid_value(self):
        self.assertEqual(refresh.retry_delay({"retry-after": "Thu, 01 Jan 1970 00:00:20 GMT"}, 0, now=lambda: 10), 10)
        with self.assertRaises(refresh.RefreshError):
            refresh.retry_delay({"retry-after": WEBHOOK}, 0)

    def test_authentication_configuration_and_redirects_never_retry(self):
        for status in [301, 302, 307, 400, 401, 403, 404, 422]:
            result, posts, waits, _ = self.run_refresh([API, STALE, (status, {"retry-after": "1"}, WEBHOOK.encode())])
            self.assertIsInstance(result, refresh.RefreshError)
            self.assertEqual(len(posts), 1)
            self.assertEqual(waits, [])

    def test_lost_response_or_ambiguous_server_failure_is_not_reposted(self):
        for status in [None, 408, 500, 502, 504]:
            result, posts, _, log = self.run_refresh([API, STALE, (status, {}, b""), STALE, CURRENT])
            self.assertEqual(result, "deployed")
            self.assertEqual(len(posts), 1)
            self.assertIn("Acceptance unknown", log)
        result, posts, _, _ = self.run_refresh([API, STALE, (None, {}, b""), STALE, STALE])
        self.assertIsInstance(result, refresh.RefreshError)
        self.assertEqual(len(posts), 1)

    def test_verify_only_has_no_webhook_side_effect(self):
        result, posts, _, _ = self.run_refresh([API, CURRENT], verify_only=True)
        self.assertEqual(result, "already-current")
        self.assertEqual(len(posts), 0)
        result, posts, _, _ = self.run_refresh([API, STALE], verify_only=True)
        self.assertIsInstance(result, refresh.RefreshError)
        self.assertEqual(len(posts), 0)

    def test_version_date_and_actual_links_must_match_inside_release_article(self):
        expected = {"tag": TAG, "version": "1.16.6", "published": "2026-09-05T12:00:00.000Z", "urls": {URL}}
        for html in [HTML.replace(URL, URL + "-invented"), HTML.replace(TAG, "rust-v1.16.5", 1), HTML.replace("Copper 1.16.6", "Copper 1.16.5"), HTML.replace("2026-09-05", "2026-09-04"), HTML.replace("href=", "data-href="), HTML.replace("</article>", '<a href="' + URL + '-removed">Extra</a></article>'), HTML + HTML, '<script>' + HTML + '</script>']:
            with self.subTest(html=html), patch.object(refresh, "request", return_value=(200, {}, html.encode())):
                self.assertFalse(refresh.page_is_current(expected, object()))
        with patch.object(refresh, "request", return_value=CURRENT):
            self.assertTrue(refresh.page_is_current(expected, object()))

    def test_invalid_or_failed_release_metadata_never_sends_webhook(self):
        for body in [b"[]", b"not JSON", json.dumps({**RELEASE, "draft": True}).encode(), json.dumps({**RELEASE, "assets": []}).encode(), json.dumps({**RELEASE, "assets": [{"state": "uploaded", "browser_download_url": "https://evil.test/asset"}]}).encode()]:
            result, posts, _, _ = self.run_refresh([(200, {}, body)])
            self.assertIsInstance(result, refresh.RefreshError)
            self.assertEqual(len(posts), 0)
        for status in [401, 403, 404]:
            result, posts, waits, _ = self.run_refresh([(status, {}, b"")])
            self.assertIsInstance(result, refresh.RefreshError)
            self.assertEqual(posts, [])
            self.assertEqual(waits, [])

    def test_network_adapter_does_not_expose_secret_exception_or_body(self):
        class BrokenOpener:
            def open(self, *args, **kwargs):
                raise URLError(WEBHOOK)
        self.assertEqual(refresh.request(Request(WEBHOOK, data=b"{}"), BrokenOpener()), (None, {}, b""))
        class DeniedOpener:
            def open(self, *args, **kwargs):
                raise HTTPError(WEBHOOK, 403, WEBHOOK, {}, io.BytesIO(WEBHOOK.encode()))
        self.assertEqual(refresh.request(Request(WEBHOOK, data=b"{}"), DeniedOpener()), (403, {}, b""))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            refresh.diagnostic("rebuild", 403, "Rejected", {"x-amzn-requestid": "private-token"})
        self.assertNotIn("private-token", output.getvalue())

    def test_successful_adapter_reads_get_but_never_reads_post_body(self):
        class Response:
            status = 200
            headers = {}
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self, limit):
                return b"public metadata"
        class Opener:
            def open(self, *args, **kwargs):
                return Response()
        self.assertEqual(refresh.request(Request("https://github.com"), Opener()), (200, {}, b"public metadata"))
        self.assertEqual(refresh.request(Request(WEBHOOK, data=b"{}", method="POST"), Opener()), (200, {}, b""))

    def test_github_get_retries_without_ever_posting_on_failure(self):
        result, posts, waits, _ = self.run_refresh([(503, {}, b""), API, CURRENT])
        self.assertEqual(result, "already-current")
        self.assertEqual(posts, [])
        self.assertEqual(waits, [2])
        result, posts, waits, _ = self.run_refresh([(503, {}, b"")] * 3)
        self.assertIsInstance(result, refresh.RefreshError)
        self.assertEqual(posts, [])
        self.assertEqual(waits, [2, 4])

    def test_github_token_is_only_sent_to_github(self):
        with patch.object(refresh, "request", side_effect=[API, STALE, (200, {}, b""), CURRENT]) as request, contextlib.redirect_stdout(io.StringIO()):
            refresh.refresh(TAG, WEBHOOK, opener=object(), token="private-github-token")
        for call in request.call_args_list:
            sent = call.args[0]
            if sent.full_url.startswith("https://api.github.com/"):
                self.assertEqual(sent.get_header("Authorization"), "Bearer private-github-token")
            else:
                self.assertIsNone(sent.get_header("Authorization"))


if __name__ == "__main__":
    unittest.main()
