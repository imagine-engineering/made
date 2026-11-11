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
    "preparation_scope": "full",
    "product_id": "rust", "display_name": "Copper", "update_namespace": "rust",
    "tag": "copper-v0.1.0-rc.1", "version": "0.1.0-rc.1", "channel": "preview", "prerelease": True,
    "commit_sha": "a" * 40, "required_ci_workflows": ["run_tests.yml"],
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


def published_release(version, scope="full"):
    decision = dict(DECISION, version=version, tag="copper-v" + version,
                    channel="preview" if "-" in version else "stable", prerelease="-" in version,
                    preparation_scope=scope)
    names = list(release.verification_asset_names("rust", version))
    names += [f"rust-{version}-macos-aarch64.dmg", "verification-rust-macos-aarch64.json"]
    if scope == "full":
        linux = f"rust-{version}-linux-x86_64.tar.gz"
        names += [linux, linux + ".asc", linux + ".pub", "verification-rust-linux-x86_64.json",
                  f"rust-{version}-windows-x86_64.exe", "verification-rust-windows-x86_64.json"]
    return dict(draft(), tag_name=decision["tag"], draft=False, published_at="date", prerelease=decision["prerelease"],
                body=release.marker(decision) + "\n" + release.ready_marker(decision),
                assets=[{"name": name, "state": "uploaded", "digest": "sha256:" + "a" * 64} for name in names])


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
        with patch.object(release, "api", return_value=None):
            with self.assertRaisesRegex(ValueError, "tag is missing"):
                release.remote_commit("owner/repo", DECISION["tag"])

    def test_copper_requires_tag_main_and_exact_ci_without_enablement(self):
        with patch.object(release, "repository", return_value="owner/repo"), \
                patch.dict(os.environ, RELEASE_GATES="{}"), \
                patch.object(release, "remote_commit", return_value=DECISION["commit_sha"]) as remote, \
                patch.object(release, "api", side_effect=[{"sha": "main"}, {"status": "ahead"}]), \
                patch.object(release, "pages", return_value=[run()]), \
                patch.object(release, "existing_release", return_value=None):
            self.assertEqual(release.guard(DECISION)[0], "owner/repo")
            remote.assert_called_once()
        with patch.object(release, "repository", return_value="owner/repo"), patch.dict(os.environ, RELEASE_GATES="{}"):
            with self.assertRaisesRegex(ValueError, "GROOVY_RELEASE_ENABLED"):
                release.guard(dict(DECISION, product_id="jvm", release_gate_variable="GROOVY_RELEASE_ENABLED"))
        with patch.object(release, "repository", return_value="owner/repo"), \
                patch.dict(os.environ, RELEASE_GATES="{}"), \
                patch.object(release, "remote_commit", return_value="b" * 40):
            with self.assertRaisesRegex(ValueError, "tag changed"):
                release.guard(DECISION)
        with patch.object(release, "repository", return_value="owner/repo"), \
                patch.dict(os.environ, RELEASE_GATES="{}"), \
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
    def test_draft_lookup_does_not_depend_on_published_tag_endpoint(self):
        with patch.object(release, "api", return_value=None), patch.object(release, "pages", return_value=[draft()]):
            self.assertEqual(release.existing_release("owner/repo", DECISION["tag"])["id"], 1)
        with patch.object(release, "api", return_value=None), patch.object(release, "pages", return_value=[draft(), draft(id=2)]):
            with self.assertRaisesRegex(ValueError, "multiple releases"):
                release.existing_release("owner/repo", DECISION["tag"])

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
            receipt["macos_verification"] = {
                kind: {"stapled_ticket_validated": True, "gatekeeper_assessed": True,
                       "signature": {"timestamp": "fixture"}, "nested_code": ["fixture"], "entitlements": {"fixture": True}}
                for kind in ("application", "dmg")}
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
                for verification in ({}, {"application": receipt["macos_verification"]["application"]},
                                     {**receipt["macos_verification"], "application": {"stapled_ticket_validated": False}}):
                    bad = dict(receipt, macos_verification=verification)
                    read.side_effect = lambda path, *args, **kwargs: json.dumps(plan if path.name == row["plan_file"] else bad)
                    with self.subTest(verification=verification), self.assertRaises(ValueError):
                        release.package_inventory(decision, root)
                read.side_effect = text
                (root / "unrelated.exe").write_bytes(b"extra")
                with self.assertRaisesRegex(ValueError, "inventory"):
                    release.package_inventory(decision, root)

    def test_draft_operation_never_publishes_and_rechecks_before_upload(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(release, "load_decision", return_value=DECISION), \
                patch.dict(os.environ, EXPECTED_COMMIT=DECISION["commit_sha"], EXPECTED_PRODUCT="rust", EXPECTED_TAG=DECISION["tag"], EXPECTED_SCOPE="full", GITHUB_RUN_ID="123"), \
                patch.object(release, "guard", return_value=("owner/repo", [run()])) as guard, \
                patch.object(release, "package_inventory", return_value=({"package": "digest"}, [])), \
                patch.object(release, "release_notes", return_value="notes"), \
                patch.object(release, "existing_release", return_value=None), \
                patch.object(release, "checksum", return_value="digest"), \
                patch.object(release, "command") as command, \
                patch.object(release, "upload_asset"), \
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


class TagPushAndDraft(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.directory)
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(self.root)
        self.row = {
            "platform": "macos", "target": "aarch64-apple-darwin",
            "artifact": "rust-macos-aarch64", "plan_file": "rust-macos-aarch64.json",
            "artifact_name": "rust-0.1.0-rc.1-macos-aarch64.dmg",
        }
        self.decision = dict(DECISION, preparation_scope="macos-only", signing="required",
                             matrix={"include": [self.row]}, repository="owner/repo")
        self.live = None
        self.event = {"ref": f"refs/tags/{DECISION['tag']}", "after": DECISION["commit_sha"],
                      "deleted": False, "forced": False, "repository": {"full_name": "owner/repo"}}
        self.event_path = self.root / "event.json"
        self.event_path.write_text(json.dumps(self.event))
        self.enterContext(patch.dict(
            os.environ, GITHUB_EVENT_NAME="push", GITHUB_REF_TYPE="tag", GITHUB_REF_NAME=DECISION["tag"],
            GITHUB_EVENT_PATH=str(self.event_path), GITHUB_SHA=DECISION["commit_sha"],
            GITHUB_REF=f"refs/tags/{DECISION['tag']}", RELEASE_GATES="{}", GITHUB_RUN_ID="123",
            EXPECTED_COMMIT=DECISION["commit_sha"], EXPECTED_PRODUCT="rust", EXPECTED_TAG=DECISION["tag"],
            EXPECTED_SCOPE="macos-only",
        ))
        self.enterContext(patch.object(release, "repository", return_value="owner/repo"))
        self.api = self.enterContext(patch.object(release, "api", side_effect=self.github))
        self.ci = self.enterContext(patch.object(release, "eligible_ci", wraps=release.eligible_ci))
        self.runs = [run()]
        self.enterContext(patch.object(release, "pages", side_effect=self.page))
        self.tag_sha = DECISION["commit_sha"]
        self.uploaded = []
        self.real_upload = release.upload_asset
        self.upload = self.enterContext(patch.object(release, "upload_asset", side_effect=self.store_asset))
        self.enterContext(patch.object(release, "release_notes", return_value="Product notes"))

    def page(self, path):
        if path.endswith("/releases"):
            return [self.live.copy()] if self.live else []
        return self.runs

    def github(self, path, method="GET", payload=None, **kwargs):
        if path.endswith("/releases") and method == "POST":
            self.assertTrue(payload["draft"])
            self.live = dict(payload, id=42, assets=[], published_at=None,
                             upload_url="https://uploads.github.com/repos/owner/repo/releases/42/assets{?name,label}")
            return self.live.copy()
        if "/releases/tags/" in path:
            return self.live.copy() if self.live and not self.live["draft"] else None
        if path.endswith("/releases/42"):
            if method == "PATCH":
                self.assertNotIn("draft", payload)
                self.live.update(payload)
            return self.live.copy()
        if "/git/ref/tags/" in path:
            return {"object": {"type": "commit", "sha": self.tag_sha}} if self.tag_sha else None
        if path.endswith("/commits/main"):
            return {"sha": self.tag_sha}
        if "/compare/" in path:
            return {"status": "identical"}
        self.fail(f"unexpected API request: {path}")

    def store_asset(self, name, live, path):
        self.assertEqual(name, "owner/repo")
        self.assertEqual(live["id"], 42)
        self.uploaded.append(path.name)
        self.live["assets"].append({"name": path.name, "state": "uploaded",
                                    "digest": "sha256:" + release.checksum(path)})

    def prepare_files(self):
        # Draft upload validates the same source plan on manual recovery.
        os.environ["GITHUB_EVENT_NAME"] = "workflow_dispatch"
        plans = Path("release-bundle-plans")
        plans.mkdir(exist_ok=True)
        (plans / "decision.json").write_text(json.dumps(self.decision))
        plan = {**release.identity(self.decision), "platform": "macos", "target": self.row["target"],
                "artifact_name": self.row["artifact_name"], "signing": "required"}
        (plans / self.row["plan_file"]).write_text(json.dumps(plan))
        packages = Path("release-artifacts")
        packages.mkdir(exist_ok=True)
        for file in packages.iterdir():
            file.unlink()
        package = packages / self.row["artifact_name"]
        package.write_bytes(b"test fixture, not a signed artifact")
        receipt = dict(plan, sha256=release.checksum(package), signing_identity="fixture",
                       macos_verification={kind: {"stapled_ticket_validated": True, "gatekeeper_assessed": True,
                                                  "signature": {"timestamp": "fixture"}, "nested_code": ["fixture"],
                                                  "entitlements": {"fixture": True}}
                                           for kind in ("application", "dmg")})
        self.receipt_path = packages / "verification-rust-macos-aarch64.json"
        self.receipt_path.write_text(json.dumps(receipt))

    def test_push_stable_and_prerelease_bind_event_sha_and_select_only_apple_silicon(self):
        for version, channel, prerelease in (("0.1.0", "stable", False), ("0.1.0-rc.1", "preview", True)):
            tag = "copper-v" + version
            self.event["ref"] = f"refs/tags/{tag}"
            self.event_path.write_text(json.dumps(self.event))
            decision = dict(self.decision, tag=tag, version=version, channel=channel, prerelease=prerelease)
            with self.subTest(version=version), patch.dict(os.environ, GITHUB_REF_NAME=tag, GITHUB_REF=self.event["ref"]), \
                    patch.object(release, "command", side_effect=[json.dumps(decision), self.tag_sha, "controller"]) as command, \
                    patch.object(release, "write_outputs") as output:
                release.resolve_tag_push()
                result = output.call_args.args[0]
                self.assertEqual(result["tag_object_sha"], self.event["after"])
                self.assertEqual(result["matrix"]["include"], [self.row])
                self.assertEqual(result["ci"][0]["head_sha"], DECISION["commit_sha"])
                self.assertEqual(result["prerelease"], prerelease)
                self.assertIn("macos-only", command.call_args_list[0].args)

    def test_invalid_push_context_is_rejected_before_planning(self):
        for changes in ({"deleted": True}, {"forced": True}, {"ref": "refs/heads/main"},
                        {"after": "invalid"}, {"repository": {"full_name": "fork/repo"}}):
            self.event_path.write_text(json.dumps(dict(self.event, **changes)))
            with self.subTest(changes=changes), patch.object(release, "command") as command:
                with self.assertRaises(ValueError):
                    release.resolve_tag_push()
                command.assert_not_called()
        with patch.dict(os.environ, GITHUB_EVENT_NAME="release"), self.assertRaisesRegex(ValueError, "tag push"):
            release.resolve_tag_push()

    def test_unknown_product_or_non_apple_silicon_matrix_cannot_access_signing(self):
        for changes in ({"product_id": "jvm"}, {"matrix": {"include": [dict(self.row, target="x86_64-apple-darwin")]}}):
            with self.subTest(changes=changes), patch.object(release, "command", side_effect=[json.dumps(dict(self.decision, **changes)), self.tag_sha]), \
                    patch.object(release, "write_outputs") as output:
                with self.assertRaises(ValueError):
                    release.resolve_tag_push()
                output.assert_not_called()
        self.ci.assert_not_called()

    def test_missing_moved_tags_failed_pending_and_absent_ci_cannot_plan(self):
        for error, sha, runs in (("removed or moved", None, [run()]),
                                ("removed or moved", "b" * 40, [run()]),
                                ("CI is not successful", DECISION["commit_sha"], [run(conclusion="failure")]),
                                ("CI is not successful", DECISION["commit_sha"], [run(status="in_progress")]),
                                ("missing exact-commit", DECISION["commit_sha"], [])):
            self.tag_sha, self.runs = sha, runs
            with self.subTest(error=error), patch.object(release, "command", side_effect=[json.dumps(self.decision), DECISION["commit_sha"]]), \
                    patch.object(release, "write_outputs") as output:
                with self.assertRaisesRegex(ValueError, error):
                    release.resolve_tag_push()
                output.assert_not_called()
        self.upload.assert_not_called()

    def test_moved_annotated_tag_object_is_rejected_even_with_same_commit(self):
        decision = dict(self.decision, tag_object_sha="b" * 40)
        with self.assertRaisesRegex(ValueError, "removed or moved"):
            release.guard(decision)

    def test_full_preparation_uses_the_same_ci_policy_without_enablement(self):
        full = dict(self.decision, preparation_scope="full", matrix={"include": [
            self.row, dict(self.row, platform="windows"), dict(self.row, platform="linux")]})
        release.guard(full)
        for runs in ([], [run(conclusion="failure")], [run(status="queued")]):
            self.runs = runs
            with self.subTest(runs=runs), self.assertRaises(ValueError):
                release.guard(full)

    def test_stable_draft_is_not_a_prerelease_or_publication(self):
        self.decision.update(tag="copper-v0.1.0", version="0.1.0", channel="stable", prerelease=False)
        os.environ["EXPECTED_TAG"] = self.decision["tag"]
        self.row["artifact_name"] = "rust-0.1.0-macos-aarch64.dmg"
        self.prepare_files()
        release.draft()
        self.assertTrue(self.live["draft"])
        self.assertFalse(self.live["prerelease"])
        self.assertEqual(self.live["make_latest"], "false")
        self.assertIn(release.ready_marker(self.decision), self.live["body"])

    def test_downloaded_decision_cannot_change_tag_push_identity(self):
        for change in ({"commit_sha": "b" * 40}, {"preparation_scope": "full"},
                       {"tag_object_sha": "b" * 40}):
            with self.subTest(change=change), patch.object(release, "load_decision", return_value=dict(self.decision, **change)):
                with self.assertRaisesRegex(ValueError, "decision differs"):
                    release.checked_decision()

    def test_failed_signature_or_unsigned_artifacts_cannot_upload(self):
        self.prepare_files()
        receipt = json.loads(self.receipt_path.read_text())
        receipt["macos_verification"]["dmg"]["stapled_ticket_validated"] = False
        self.receipt_path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "ticket"):
            release.draft()
        self.prepare_files()
        with patch.object(release, "load_decision", return_value=dict(self.decision, signing="off")):
            with self.assertRaisesRegex(ValueError, "development artifacts"):
                release.draft()
        self.upload.assert_not_called()

    def test_interrupted_upload_reuses_identical_assets_preserves_notes_and_completes(self):
        self.prepare_files()
        def interrupt(name, live, path):
            if self.uploaded:
                raise RuntimeError("simulated transport failure")
            self.store_asset(name, live, path)
        self.upload.side_effect = interrupt
        with self.assertRaisesRegex(RuntimeError, "transport failure"):
            release.draft()
        self.assertEqual(len(self.uploaded), 1)
        reviewed = release.marker(self.decision) + "\nMy reviewed release notes"
        self.live["body"] = reviewed
        self.prepare_files()
        self.upload.side_effect = self.store_asset
        release.draft()
        self.assertEqual(len(self.uploaded), 4)
        self.assertEqual(len(set(self.uploaded)), 4)
        self.assertEqual(self.live["body"], reviewed + "\n\n" + release.ready_marker(self.decision))
        self.assertTrue(self.live["draft"])
        self.assertIn("macOS only", self.live["name"])
        self.prepare_files()
        release.draft()
        self.assertEqual(len(self.uploaded), 4)

    def test_moved_tag_stops_mid_upload(self):
        self.prepare_files()
        def move_tag(name, live, path):
            self.store_asset(name, live, path)
            self.tag_sha = "b" * 40
        self.upload.side_effect = move_tag
        with self.assertRaisesRegex(ValueError, "tag changed"):
            release.draft()
        self.assertEqual(len(self.uploaded), 1)

    def test_retagging_the_draft_stops_mid_upload(self):
        self.prepare_files()
        def retag_draft(name, live, path):
            self.store_asset(name, live, path)
            self.live["tag_name"] = "copper-v0.2.0-rc.1"
        self.upload.side_effect = retag_draft
        with self.assertRaisesRegex(ValueError, "different product, commit, version"):
            release.draft()
        self.assertEqual(len(self.uploaded), 1)

    def test_published_immutable_and_conflicting_assets_stop_upload(self):
        self.prepare_files()
        self.live = dict(draft(), body=release.marker(self.decision))
        for changes in ({"draft": False}, {"immutable": True}, {"prerelease": False},
                        {"assets": [{"name": self.row["artifact_name"], "state": "uploaded", "digest": "sha256:wrong"}]}):
            with self.subTest(changes=changes), patch.dict(self.live, changes):
                with self.assertRaises(ValueError):
                    release.draft()
            self.prepare_files()
        self.upload.assert_not_called()
        with self.assertRaisesRegex(ValueError, "release ID changed"):
            release.check_draft(self.live, self.decision, 99)

    def test_upload_transport_is_bound_to_the_release_id(self):
        self.prepare_files()
        path = Path("release-artifacts", self.row["artifact_name"])
        live = {"id": 42, "upload_url": "https://uploads.github.com/repos/owner/repo/releases/42/assets{?name,label}"}
        with patch.object(release, "HTTPSConnection") as connection, patch.dict(os.environ, GH_TOKEN="test-token"):
            connection.return_value.getresponse.return_value.status = 201
            self.real_upload("owner/repo", live, path)
            args = connection.return_value.request.call_args.args
            self.assertEqual(args[:2], ("POST", f"/repos/owner/repo/releases/42/assets?name={path.name}"))
            self.assertEqual(args[3]["Content-Length"], str(path.stat().st_size))
            connection.return_value.getresponse.return_value.status = 403
            with self.assertRaisesRegex(RuntimeError, "immutability"):
                self.real_upload("owner/repo", live, path)
            with self.assertRaisesRegex(ValueError, "endpoint does not match"):
                self.real_upload("owner/repo", dict(live, upload_url="https://example.com/assets"), path)


class ProductIsolation(unittest.TestCase):
    def test_macos_only_publication_never_becomes_a_windows_linux_or_generic_update(self):
        partial = dict(DECISION, tag="copper-v0.2.0", version="0.2.0", channel="stable", prerelease=False,
                       preparation_scope="macos-only", matrix={"include": [{"platform": "macos"}]})
        published = published_release("0.2.0", "macos-only")
        product = next(product for product in release.catalog_products() if product["id"] == "rust")
        for platform in (None, "linux", "windows"):
            self.assertIsNone(release.latest_release([published], product, "stable", platform))
        self.assertEqual(release.latest_release([published], product, "stable", "macos"), published)
        self.assertIsNone(release.latest_release([dict(published, draft=True)], product, "stable", "macos"))
        self.assertIsNone(release.latest_release([dict(published, assets=[])], product, "stable", "macos"))
        full_marker = release.marker(dict(partial, preparation_scope="full"))
        self.assertIsNone(release.latest_release([dict(published, body=full_marker)], product, "stable"))
        self.assertIsNone(release.latest_release([dict(published, body="<!-- made-product-release invalid -->")], product, "stable", "macos"))

    def test_scope_is_bound_to_draft_ownership_and_cannot_expand_partial_matrix(self):
        partial = dict(DECISION, preparation_scope="macos-only", matrix={"include": [{"platform": "windows"}]})
        with self.assertRaisesRegex(ValueError, "only Copper macOS"):
            release.guard(partial)
        with self.assertRaisesRegex(ValueError, "different product, commit, version"):
            release.check_draft(draft(), partial)

    def test_latest_is_semantic_product_and_channel_specific_and_excludes_drafts(self):
        def published(tag, prerelease=False, **changes):
            return {"tag_name": tag, "draft": False, "prerelease": prerelease, "published_at": "date", **changes}
        releases = [published("jvm-v99.0.0"), published("copper-v0.9.0"), published("copper-v0.10.0"),
                    published("copper-v0.11.0-rc.10", True), published("copper-v0.11.0-rc.2", True),
                    published("copper-v9.0.0", draft=True), published("copper-v2.0.0", published_at=None),
                    published("copper-v3.0.0-rc.1", False), published("copper-v01.1.0")]
        copper = {"release_tag_prefix": "copper-v"}
        groovy = {"release_tag_prefix": "jvm-v"}
        self.assertEqual(release.latest_release(releases, copper, "stable")["tag_name"], "copper-v0.10.0")
        self.assertEqual(release.latest_release(releases, groovy, "stable")["tag_name"], "jvm-v99.0.0")
        self.assertIsNone(release.latest_release(releases, groovy, "preview"))

    def test_stable_and_preview_feeds_wait_for_verified_assets_and_isolate_platforms(self):
        copper = next(product for product in release.catalog_products() if product["id"] == "rust")
        for version, channel in (("0.2.0", "stable"), ("0.2.0-rc.10", "preview")):
            complete = published_release(version, "macos-only")
            fallback = published_release("0.1.0" if channel == "stable" else "0.1.0-rc.1")
            for missing in complete["assets"]:
                partial = dict(complete, assets=[asset for asset in complete["assets"] if asset != missing])
                with self.subTest(version=version, missing=missing["name"]):
                    self.assertEqual(release.latest_release([fallback, partial], copper, channel, "macos"), fallback)
            for change in ({"draft": True}, {"body": "Operator notes"},
                           {"body": complete["body"].split("\n")[0]},
                           {"assets": [dict(complete["assets"][0], state="starter"), *complete["assets"][1:]]},
                           {"assets": [dict(complete["assets"][0], digest=None), *complete["assets"][1:]]},
                           {"assets": complete["assets"] + [complete["assets"][0]]}):
                with self.subTest(change=change):
                    self.assertIsNone(release.latest_release([dict(complete, **change)], copper, channel, "macos"))
            self.assertEqual(release.latest_release([fallback, complete], copper, channel, "macos"), complete)
            for platform in (None, "windows", "linux"):
                self.assertEqual(release.latest_release([fallback, complete], copper, channel, platform), fallback)
            self.assertIsNone(release.latest_release([complete], copper, "preview" if channel == "stable" else "stable", "macos"))


    def test_catalog_rejects_ambiguous_product_prefixes_and_namespaces(self):
        copper = {"id": "rust", "update_namespace": "rust", "release_tag_prefix": "copper-v"}
        groovy = {"id": "jvm", "update_namespace": "jvm", "release_tag_prefix": "jvm-v"}
        release.validate_catalog([copper, groovy])
        for changes in ({"release_tag_prefix": "copper-v"}, {"release_tag_prefix": "copper-v-extra-v"},
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
            {"tag_name": "copper-v0.0.9", "draft": False, "published_at": "date", "prerelease": False},
            {"tag_name": "copper-v0.1.0-rc.1", "draft": False, "published_at": "date", "prerelease": True},
        ]
        releases[1] = published_release("0.0.9")
        with patch.object(release, "pages", return_value=releases), patch.object(release, "api") as api:
            notes = release.release_notes(dict(DECISION, tag="copper-v0.1.0", version="0.1.0", channel="stable", prerelease=False), "owner/repo")
            self.assertIn("copper-v0.0.9...copper-v0.1.0", notes)
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
