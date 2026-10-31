#!/usr/bin/env python3
"""Refresh Amplify and verify the published Copper release on the public page."""

import argparse
from pathlib import Path
import tomllib
from product_release import semver_key
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class RefreshError(Exception):
    pass


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def diagnostic(stage, status=None, message="", headers=None):
    record = {"stage": stage, "status": status, "message": message}
    print(json.dumps(record), flush=True)


def validate_webhook(url):
    try:
        parts = urlsplit(url)
        query = parse_qs(parts.query, keep_blank_values=True, strict_parsing=True)
        valid = (
            parts.scheme == "https"
            and re.fullmatch(r"webhooks\.amplify\.[a-z0-9-]+\.amazonaws\.com", parts.netloc)
            and parts.path == "/prod/webhooks"
            and not parts.fragment
            and set(query) == {"id", "token", "operation"}
            and all(len(values) == 1 and values[0].strip() for values in query.values())
            and query["operation"] == ["startbuild"]
            and url.isascii()
            and not any(character.isspace() for character in url)
        )
    except ValueError:
        valid = False
    if not valid:
        raise RefreshError("Configure AMPLIFY_REBUILD_WEBHOOK for the existing website app's main branch.")
    # The opaque URL does not encode the app or branch. Verify those in the console.
    return url


def retry_delay(headers, attempt, now=time.time):
    value = headers.get("retry-after")
    if value is None:
        return 2 ** (attempt + 1)
    try:
        seconds = int(value) if re.fullmatch(r"\d+", value) else parsedate_to_datetime(value).timestamp() - now()
        return max(1, seconds)
    except (TypeError, ValueError, OverflowError):
        raise RefreshError("Invalid Retry-After header; inspect the hosting service before recovery.") from None


def request(request, opener, limit=4_000_000):
    try:
        with opener.open(request, timeout=30) as response:
            body = b"" if request.get_method() == "POST" else response.read(limit + 1)
            if len(body) > limit:
                raise RefreshError("Response exceeded the allowed size.")
            return response.status, response.headers, body
    except HTTPError as error:
        status, headers = error.code, error.headers
        error.close()
        return status, headers, b""
    except (URLError, TimeoutError, OSError, HTTPException):
        # A lost response can follow a successful POST. Never print its secret URL.
        return None, {}, b""


def published_release(tag, opener, token="", sleep=time.sleep):
    catalog = tomllib.loads((Path(__file__).resolve().parent.parent / "products/flavors.toml").read_text())
    product = next(product for product in catalog["products"] if product["id"] == "rust")
    prefix = product["release_tag_prefix"]
    try:
        if not tag.startswith(prefix):
            raise ValueError("different product")
        semver_key(tag[len(prefix):])
    except ValueError:
        raise RefreshError("Expected an existing Copper release tag from the product catalog.") from None
    if token and (not token.isascii() or any(character.isspace() for character in token)):
        raise RefreshError("Invalid GitHub authentication configuration; no webhook was sent.")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "made-website-refresh"}
    if token:
        headers["Authorization"] = "Bearer " + token
    waited = 0
    for attempt in range(3):
        status, response_headers, body = request(Request("https://api.github.com/repos/imagine-engineering/made/releases/tags/" + quote(tag, safe=""), headers=headers), opener)
        if status == 200:
            break
        diagnostic("publication", status, "Unable to read published GitHub release metadata.")
        if status not in (None, 429, 500, 502, 503, 504) or attempt == 2:
            raise RefreshError("GitHub release verification failed; no webhook was sent.")
        wait = retry_delay(response_headers, attempt)
        if waited + wait > 120:
            raise RefreshError("GitHub retry wait exceeds budget; no webhook was sent.")
        waited += wait
        sleep(wait)
    try:
        release = json.loads(body)
        published = datetime.fromisoformat(release["published_at"].replace("Z", "+00:00"))
        assets = release["assets"]
        urls = {asset["browser_download_url"] for asset in assets if asset["state"] == "uploaded"}
        valid = release["tag_name"] == tag and not release["draft"] and bool(urls) and len(urls) == len(assets)
        for url in urls:
            parts = urlsplit(url)
            valid = valid and parts.scheme == "https" and parts.netloc == "github.com" and parts.path.startswith("/imagine-engineering/made/releases/download/" + tag + "/") and not parts.query and not parts.fragment
        if not valid:
            raise ValueError()
    except (ValueError, TypeError, KeyError, AttributeError):
        raise RefreshError("Invalid, draft, or incomplete Copper release metadata; no webhook was sent.") from None
    extensions = {"linux": ".tar.gz", "macos": ".dmg", "windows": ".exe"}
    package_suffixes = tuple({extensions[target.split("-", 1)[0]] for target in product["targets"]})
    all_urls = urls
    urls = {url for url in all_urls if urlsplit(url).path.endswith(package_suffixes)}
    if not urls:
        raise RefreshError("Published release has no supported desktop packages; no webhook was sent.")
    diagnostic("publication", 200, "Published release metadata verified: " + tag + ".")
    return {"tag": tag, "version": tag[len(prefix):], "urls": urls, "all_urls": all_urls, "published": published.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")}


class ReleasePage(HTMLParser):
    def __init__(self, tag):
        super().__init__(convert_charrefs=True)
        self.tag = tag
        self.active = False
        self.found = 0
        self.urls = set()
        self.dates = set()
        self.text = []

    def handle_starttag(self, name, attributes):
        attributes = dict(attributes)
        if name == "article" and attributes.get("id") == self.tag:
            self.active = True
            self.found += 1
        if self.active:
            if name == "a" and attributes.get("href", "").startswith("https://github.com/imagine-engineering/made/releases/download/"):
                self.urls.add(attributes["href"])
            if name == "time":
                self.dates.add(attributes.get("datetime"))

    def handle_endtag(self, name):
        if name == "article":
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.text.append(data)


def page_is_current(release, opener):
    status, headers, body = request(Request("https://simtropolis.ai/releases?release=" + quote(release["tag"], safe=""), headers={"Cache-Control": "no-cache", "User-Agent": "made-website-refresh"}), opener)
    if status != 200:
        return False
    parser = ReleasePage(release["tag"])
    try:
        parser.feed(body.decode("utf-8"))
    except (UnicodeError, ValueError):
        return False
    return parser.found == 1 and release["urls"] <= parser.urls <= release["all_urls"] and release["published"] in parser.dates and "Copper " + release["version"] in "".join(parser.text)


def wait_for_deployment(release, opener, sleep=time.sleep, monotonic=time.monotonic, timeout=900):
    deadline = monotonic() + timeout
    while True:
        if page_is_current(release, opener):
            diagnostic("deployment", 200, "Verified triggering version, publication date, and exact asset URLs on /releases: " + release["tag"] + ".")
            return True
        remaining = deadline - monotonic()
        if remaining <= 0:
            return False
        sleep(min(15, remaining))


def refresh(tag, webhook, *, verify_only=False, opener=None, sleep=time.sleep, monotonic=time.monotonic, token="", deployment_timeout=900):
    if not verify_only:
        validate_webhook(webhook)
    opener = opener or build_opener(NoRedirects)
    release = published_release(tag, opener, token, sleep)
    if page_is_current(release, opener):
        diagnostic("deployment", 200, "Triggering release and exact assets are already deployed; no duplicate rebuild requested.")
        return "already-current"
    if verify_only:
        raise RefreshError("The triggering release's version, date, and exact assets are not deployed.")
    waited = 0
    for attempt in range(3):
        status, headers, _ = request(Request(webhook, data=b"{}", headers={"Content-Type": "application/json"}, method="POST"), opener)
        if status is not None and 200 <= status < 300:
            diagnostic("rebuild", status, "Rebuild accepted; deployment is not yet verified.", headers)
            break
        if status in (429, 503):
            diagnostic("rebuild", status, "Transient webhook response; retries are bounded and can enqueue duplicate website builds if the service accepted this request.", headers)
            if attempt == 2:
                raise RefreshError("Webhook retry attempts exhausted; Copper remains published. Use refresh-only recovery after checking Amplify's build queue.")
            wait = retry_delay(headers, attempt)
            if waited + wait > 120:
                raise RefreshError("Required webhook retry wait exceeds budget; retry later using refresh-only recovery.")
            waited += wait
            diagnostic("retry", status, f"Waiting {wait:g} seconds before webhook attempt {attempt + 2} of 3.")
            sleep(wait)
            if page_is_current(release, opener):
                diagnostic("deployment", 200, "Triggering release and exact assets appeared before retry; no duplicate rebuild requested.")
                return "deployed"
            continue
        if status is None or status in (408, 500, 502, 504):
            diagnostic("rebuild", status, "Acceptance unknown. Observing deployment without resending the non-idempotent POST; a build may already be running.", headers)
            break
        diagnostic("rebuild", status, "Webhook configuration or authentication rejected; no retry.", headers)
        raise RefreshError("Invalid or unauthorized webhook; Copper remains published. Fix configuration before refresh-only recovery.")
    if wait_for_deployment(release, opener, sleep, monotonic, deployment_timeout):
        return "deployed"
    raise RefreshError("Website deployment was not verified within 15 minutes. Copper remains published; inspect existing Amplify builds before refresh-only recovery to avoid duplicates.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    try:
        refresh(os.environ.get("COPPER_RELEASE_TAG", ""), os.environ.get("AMPLIFY_REBUILD_WEBHOOK", ""), verify_only=arguments.verify_only, token=os.environ.get("GH_TOKEN", ""))
    except RefreshError as error:
        diagnostic("failed", message=str(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
