"""Non-publishing release policy tests. No GitHub writes or live signing."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import product_release as release


DECISION = {
    "product_id": "rust", "display_name": "Copper", "update_namespace": "rust",
    "tag": "rust-v0.1.0-rc.1", "version": "0.1.0-rc.1", "channel": "preview", "prerelease": True,
    "commit_sha": "a" * 40, "release_gate_variable": "COPPER_RELEASE_ENABLED", "required_ci_workflows": ["run_tests.yml"],
}


def run(**changes):
    result = {"id": 10, "head_sha": DECISION["commit_sha"], "head_branch": "main", "event": "push",
              "path": ".github/workflows/run_tests.yml", "repository": {"full_name": "owner/repo"},
              "head_repository": {"full_name": "owner/repo"}, "status": "completed", "conclusion": "success",
              "html_url": "https://github.com/owner/repo/actions/runs/10"}
    result.update(changes)
    return result


def draft(**changes):
    result = {"id": 1, "draft": True, "tag_name": DECISION["tag"], "target_commitish": DECISION["commit_sha"],
              "body": release.marker(DECISION), "prerelease": True, "published_at": None, "assets": []}
    result.update(changes)
    return result


class Eligibility(unittest.TestCase):
    def test_exact_commit_success_and_latest_attempt(self):
        self.assertEqual(release.eligible_ci([run()], DECISION, "owner/repo", "run_tests.yml")["id"], 10)
        for change in ({"head_sha": "b" * 40}, {"head_branch": "release"}, {"event": "pull_request"},
                       {"path": ".github/workflows/other.yml"}, {"head_repository": {"full_name": "fork/repo"}},
                       {"repository": {"full_name": "fork/repo"}}, {"conclusion": "failure"},
                       {"status": "in_progress"}, {"conclusion": "cancelled"}, {"conclusion": "skipped"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                release.eligible_ci([run(**change)], DECISION, "owner/repo", "run_tests.yml")
        with self.assertRaises(ValueError):
            release.eligible_ci([run(), run(id=11, conclusion="failure")], DECISION, "owner/repo", "run_tests.yml")
        with self.assertRaises(ValueError):
            release.eligible_ci([run(run_attempt=1), run(run_attempt=2, conclusion="failure")], DECISION, "owner/repo", "run_tests.yml")

    def test_remote_annotated_tag_peels_to_exact_commit(self):
        with patch.object(release, "api", side_effect=[{"object": {"type": "tag", "sha": "tag-object"}},
                                                       {"object": {"type": "commit", "sha": DECISION["commit_sha"]}}]):
            self.assertEqual(release.remote_commit("owner/repo", DECISION["tag"]), DECISION["commit_sha"])
        with patch.object(release, "api", return_value={"object": {"type": "tree", "sha": "invalid"}}):
            with self.assertRaises(ValueError):
                release.remote_commit("owner/repo", DECISION["tag"])

    def test_guard_requires_gate_tag_main_and_exact_ci(self):
        with patch.object(release, "repository", return_value="owner/repo"), \
                patch.dict(os.environ, RELEASE_GATES='{"COPPER_RELEASE_ENABLED":"true"}'), \
                patch.object(release, "remote_commit", return_value=DECISION["commit_sha"]) as remote, \
                patch.object(release, "api", side_effect=[{"sha": "main"}, {"status": "ahead"}]), \
                patch.object(release, "pages", return_value=[run()]), \
                patch.object(release, "existing_release", return_value=None):
            self.assertEqual(release.guard(DECISION)[0], "owner/repo")
            remote.assert_called_once()
        with patch.object(release, "repository", return_value="owner/repo"), patch.dict(os.environ, RELEASE_GATES='{}'):
            with self.assertRaisesRegex(ValueError, "COPPER_RELEASE_ENABLED"):
                release.guard(DECISION)
        with patch.object(release, "repository", return_value="owner/repo"), \
                patch.dict(os.environ, RELEASE_GATES='{"COPPER_RELEASE_ENABLED":"true"}'), \
                patch.object(release, "remote_commit", return_value="b" * 40):
            with self.assertRaisesRegex(ValueError, "tag changed"):
                release.guard(DECISION)
        with patch.object(release, "repository", return_value="owner/repo"), \
                patch.dict(os.environ, RELEASE_GATES='{"COPPER_RELEASE_ENABLED":"true"}'), \
                patch.object(release, "remote_commit", return_value=DECISION["commit_sha"]), \
                patch.object(release, "api", side_effect=[{"sha": "main"}, {"status": "diverged"}]):
            with self.assertRaisesRegex(ValueError, "main history"):
                release.guard(DECISION)

    def test_api_failures_do_not_expose_credentials(self):
        from types import SimpleNamespace
        with patch.object(release.subprocess, "run", return_value=SimpleNamespace(returncode=1, stderr=b"HTTP 403 PRIVATE_TOKEN", stdout=b"")):
            with self.assertRaises(RuntimeError) as error:
                release.api("repos/owner/repo/releases")
            self.assertNotIn("PRIVATE_TOKEN", str(error.exception))
        with patch.object(release.subprocess, "run", return_value=SimpleNamespace(returncode=1, stderr=b"HTTP 404", stdout=b"")):
            self.assertIsNone(release.api("repos/owner/repo/releases/tags/absent", optional=True))

    def test_wrong_repository_fails(self):
        with patch.dict(os.environ, GITHUB_REPOSITORY="owner/repo"), patch.object(release, "command", return_value="https://github.com/fork/repo.git"):
            with self.assertRaisesRegex(ValueError, "workflow repository"):
                release.repository()


class DraftSafety(unittest.TestCase):
    def test_only_owned_unpublished_matching_draft_can_resume(self):
        release.check_draft(None, DECISION)
        release.check_draft(draft(), DECISION)
        for change in ({"draft": False}, {"published_at": "date"}, {"target_commitish": "b" * 40},
                       {"tag_name": "jvm-v0.1.0"}, {"body": "unowned draft"}, {"prerelease": False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                release.check_draft(draft(**change), DECISION)

    def test_partial_resume_requires_exact_asset_digests(self):
        hashes = {"rust-0.1.0.dmg": "1234", "verification.json": "4567"}
        asset = {"name": "rust-0.1.0.dmg", "state": "uploaded", "digest": "sha256:1234"}
        self.assertEqual(release.verify_remote_assets([asset], hashes), {asset["name"]})
        for changes in ({"digest": "sha256:different"}, {"digest": None}, {"name": "jvm-0.1.0.dmg"}, {"state": "starter"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                release.verify_remote_assets([dict(asset, **changes)], hashes)
        with self.assertRaises(ValueError):
            release.verify_remote_assets([asset, asset], hashes)

    def test_packaged_evidence_binds_commit_version_channel_product_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"platform": "macos", "target": "aarch64-apple-darwin", "artifact": "rust-macos-aarch64",
                   "artifact_suffix": "macos-aarch64.dmg", "plan_file": "rust-macos-aarch64.json"}
            decision = dict(DECISION, matrix={"include": [row]})
            filename = f"rust-{decision['version']}-macos-aarch64.dmg"
            row["artifact_name"] = filename
            plan = {**release.identity(decision), "platform": "macos", "target": row["target"], "signing": "required", "artifact_name": filename}
            receipt = {**release.identity(decision), "platform": "macos", "signing": "required", "artifact_name": filename, "signing_identity": "fixture identity"}
            (root / filename).write_bytes(b"package fixture, not signed")
            receipt["sha256"] = release.checksum(root / filename)
            receipt_path = root / "verification-rust-macos-aarch64.json"
            receipt_path.write_text(json.dumps(receipt))
            with patch.object(release.Path, "read_text", autospec=True) as read:
                def text(path, *args, **kwargs):
                    return json.dumps(plan if path.name == row["plan_file"] else receipt)
                read.side_effect = text
                hashes, _ = release.package_inventory(decision, root)
                self.assertIn(filename, hashes)
                for key, value in (("commit_sha", "b" * 40), ("product_id", "jvm"), ("version", "0.2.0"), ("channel", "stable"), ("signing", "off"), ("sha256", "bad")):
                    bad = dict(receipt, **{key: value})
                    read.side_effect = lambda path, *args, **kwargs: json.dumps(plan if path.name == row["plan_file"] else bad)
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        release.package_inventory(decision, root)
                read.side_effect = text
                (root / "unrelated.exe").write_bytes(b"extra")
                with self.assertRaisesRegex(ValueError, "inventory"):
                    release.package_inventory(decision, root)

    def test_draft_operation_never_publishes_and_rechecks_before_upload(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(release, "load_decision", return_value=DECISION), \
                patch.dict(os.environ, EXPECTED_COMMIT=DECISION["commit_sha"], EXPECTED_PRODUCT="rust", EXPECTED_TAG=DECISION["tag"], GITHUB_RUN_ID="123"), \
                patch.object(release, "guard", return_value=("owner/repo", [run()])) as guard, \
                patch.object(release, "package_inventory", return_value=({"package": "digest"}, [])), \
                patch.object(release, "release_notes", return_value="notes"), \
                patch.object(release, "existing_release", return_value=None), \
                patch.object(release, "checksum", return_value="digest"), \
                patch.object(release, "command") as command, \
                patch.object(release, "api") as api:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                Path("release-artifacts").mkdir()
                assets = [{"name": name, "state": "uploaded", "digest": "sha256:digest"} for name in ("package", "rust-0.1.0-rc.1-verification.json", "rust-0.1.0-rc.1-SHA256SUMS")]
                api.side_effect = [draft(), draft(), draft(), draft(), draft(assets=assets), draft(assets=assets)]
                with contextlib.redirect_stdout(io.StringIO()):
                    release.draft()
                self.assertGreaterEqual(guard.call_count, 5)
                payloads = [call.args[2] for call in api.call_args_list if len(call.args) == 3]
                self.assertTrue(payloads[0]["draft"])
                self.assertTrue(payloads[0]["prerelease"])
                self.assertEqual(payloads[0]["make_latest"], "false")
                self.assertTrue(all(payload.get("draft", True) for payload in payloads))
                self.assertFalse(any("tag" in call.args[:2] or "push" in call.args for call in command.call_args_list))
            finally:
                os.chdir(previous)

    def test_published_release_race_stops_before_upload(self):
        # The same check executes before every upload, not just before builds start.
        with self.assertRaisesRegex(ValueError, "already published"):
            release.check_draft(draft(draft=False), DECISION)


class ProductIsolation(unittest.TestCase):
    def test_latest_is_semantic_product_and_channel_specific_and_excludes_drafts(self):
        def published(tag, prerelease=False, **changes):
            return {"tag_name": tag, "draft": False, "prerelease": prerelease, "published_at": "date", **changes}
        releases = [published("jvm-v99.0.0"), published("rust-v0.9.0"), published("rust-v0.10.0"),
                    published("rust-v0.11.0-rc.10", True), published("rust-v0.11.0-rc.2", True),
                    published("rust-v9.0.0", draft=True), published("rust-v2.0.0", published_at=None),
                    published("rust-v3.0.0-rc.1", False), published("rust-v01.1.0")]
        copper = {"release_tag_prefix": "rust-v"}
        groovy = {"release_tag_prefix": "jvm-v"}
        self.assertEqual(release.latest_release(releases, copper, "stable")["tag_name"], "rust-v0.10.0")
        self.assertEqual(release.latest_release(releases, copper, "preview")["tag_name"], "rust-v0.11.0-rc.10")
        self.assertEqual(release.latest_release(releases, groovy, "stable")["tag_name"], "jvm-v99.0.0")
        self.assertIsNone(release.latest_release(releases, groovy, "preview"))

    def test_catalog_rejects_ambiguous_product_prefixes_and_namespaces(self):
        copper = {"id": "rust", "update_namespace": "rust", "release_tag_prefix": "rust-v"}
        groovy = {"id": "jvm", "update_namespace": "jvm", "release_tag_prefix": "jvm-v"}
        release.validate_catalog([copper, groovy])
        for changes in ({"release_tag_prefix": "rust-v"}, {"release_tag_prefix": "rust-v-extra-v"},
                        {"update_namespace": "rust"}, {"id": "rust"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                release.validate_catalog([copper, dict(groovy, **changes)])

    def test_semver_validation_and_precedence(self):
        ordered = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"]
        self.assertEqual(sorted(ordered[::-1], key=release.semver_key), ordered)
        self.assertEqual(release.semver_key("1.0.0+build.1"), release.semver_key("1.0.0+build.2"))
        for value in ("01.0.0", "1.2", "1.0.0-01", "1.0.0-", "1.0.0\n"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                release.semver_key(value)

    def test_notes_use_product_baseline_without_copying_other_product_commit_messages(self):
        releases = [
            {"tag_name": "jvm-v99.0.0", "draft": False, "published_at": "date", "prerelease": False},
            {"tag_name": "rust-v0.0.9", "draft": False, "published_at": "date", "prerelease": False},
            {"tag_name": "rust-v0.1.0-rc.1", "draft": False, "published_at": "date", "prerelease": True},
        ]
        with patch.object(release, "pages", return_value=releases), patch.object(release, "api") as api:
            notes = release.release_notes(dict(DECISION, tag="rust-v0.1.0", version="0.1.0", channel="stable", prerelease=False), "owner/repo")
            self.assertIn("rust-v0.0.9...rust-v0.1.0", notes)
            self.assertNotIn("jvm", notes)
            self.assertNotIn("rc.1", notes)
            api.assert_not_called()

    def test_notes_baseline_cannot_use_other_products(self):
        releases = [{"tag_name": "jvm-v9.0.0", "draft": False, "published_at": "date", "prerelease": False}]
        with patch.object(release, "pages", return_value=releases), patch.object(release, "api") as api:
            notes = release.release_notes(dict(DECISION, version="0.1.0", prerelease=False, channel="stable"), "owner/repo")
            self.assertIn("Copper 0.1.0", notes)
            api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
