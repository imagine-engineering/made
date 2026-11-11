#!/usr/bin/env python3
"""Prepare verified signed product packages as an owned draft for human publication."""
import argparse
import hashlib
from http.client import HTTPSConnection, HTTPException
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from urllib.parse import quote, urlencode


APPLE_SILICON_TARGET = "aarch64-apple-darwin"


def validate_catalog(products):
    for key in ("id", "update_namespace", "release_tag_prefix"):
        values = [product[key] for product in products]
        if len(set(values)) != len(values):
            raise ValueError(f"ambiguous catalog {key}")
    prefixes = [product["release_tag_prefix"] for product in products]
    for index, prefix in enumerate(prefixes):
        if not re.fullmatch(r"[a-z0-9-]+-v", prefix):
            raise ValueError("invalid catalog release prefix")
        if any(prefix.startswith(other) or other.startswith(prefix) for other in prefixes[index + 1:]):
            raise ValueError("ambiguous catalog release prefixes")
    return products


def catalog_products():
    return validate_catalog(tomllib.loads(Path("products/flavors.toml").read_text())["products"])


def command(*arguments, **options):
    result = subprocess.run(arguments, capture_output=True, **options)
    if result.returncode:
        if Path(arguments[0]).name == "xtask":
            raise RuntimeError("Release planner rejected the request: " + result.stderr.decode().strip())
        raise RuntimeError(f"{arguments[0]} failed (exit {result.returncode}); inspect the preceding build/CI logs")
    return result.stdout.decode().strip()


def api(path, method="GET", payload=None, optional=False):
    arguments = ["gh", "api", "--method", method, path]
    data = None
    if payload is not None:
        arguments += ["--input", "-"]
        data = json.dumps(payload).encode()
    result = subprocess.run(arguments, input=data, capture_output=True)
    if optional and result.returncode and b"HTTP 404" in result.stderr:
        return None
    if result.returncode:
        raise RuntimeError(f"GitHub {method} {path} failed; check token permissions and service availability")
    return json.loads(result.stdout)


def pages(path):
    items = []
    for page in range(1, 1001):
        response = api(path + ("&" if "?" in path else "?") + f"per_page=100&page={page}")
        entries = response if isinstance(response, list) else response["workflow_runs"]
        items.extend(entries)
        if len(entries) < 100:
            return items
    raise RuntimeError("GitHub pagination exceeded its safety limit")


def repository():
    name = os.environ.get("GITHUB_REPOSITORY") or json.loads(command("gh", "repo", "view", "--json", "nameWithOwner"))["nameWithOwner"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name):
        raise ValueError("invalid repository identity")
    remote = command("git", "remote", "get-url", "origin")
    if remote not in (f"https://github.com/{name}", f"https://github.com/{name}.git", f"git@github.com:{name}.git"):
        raise ValueError("origin is not the workflow repository")
    return name


def remote_commit(name, tag):
    response = api(f"repos/{name}/git/ref/tags/{quote(tag, safe='')}", optional=True)
    if response is None:
        raise ValueError("existing release tag is missing")
    reference = response["object"]
    for _ in range(10):
        if reference["type"] == "commit":
            return reference["sha"]
        if reference["type"] != "tag":
            break
        reference = api(f"repos/{name}/git/tags/{reference['sha']}")["object"]
    raise ValueError("release tag does not resolve to a commit")


def eligible_ci(runs, decision, name, workflow):
    candidates = [run for run in runs if run.get("head_sha") == decision["commit_sha"]
                  and run.get("head_branch") == "main" and run.get("event") == "push"
                  and run.get("path") == f".github/workflows/{workflow}"
                  and (run.get("head_repository") or {}).get("full_name") == name
                  and (run.get("repository") or {}).get("full_name") == name]
    if not candidates:
        raise ValueError(f"missing exact-commit main push CI: {workflow}; run required CI on main, then recover with release_tag on main")
    latest = max(candidates, key=lambda run: (run["id"], run.get("run_attempt", 1)))
    if latest.get("status") != "completed" or latest.get("conclusion") != "success":
        raise ValueError(f"latest exact-commit required CI is not successful: {workflow}; wait for or fix CI, then recover with release_tag on main")
    return {key: latest[key] for key in ("id", "head_sha", "html_url", "conclusion")}


def identity(decision):
    return {key: decision[key] for key in ("product_id", "tag", "version", "commit_sha", "channel", "update_namespace", "preparation_scope")}


def ready_marker(decision):
    return "<!-- made-product-release-verified " + json.dumps(identity(decision), sort_keys=True, separators=(",", ":")) + " -->"


def marker(decision):
    return "<!-- made-product-release " + json.dumps(identity(decision), sort_keys=True, separators=(",", ":")) + " -->"


def existing_release(name, tag):
    published = api(f"repos/{name}/releases/tags/{quote(tag, safe='')}", optional=True)
    # The tag endpoint promises published releases; list releases also exposes owned drafts.
    matches = [release for release in pages(f"repos/{name}/releases") if release.get("tag_name") == tag]
    if published and not any(release["id"] == published["id"] for release in matches):
        matches.append(published)
    if len(matches) > 1:
        raise ValueError("multiple releases use this tag; review conflicting drafts before retrying")
    return matches[0] if matches else None


def check_draft(release, decision, release_id=None):
    if release is None:
        return
    if release.get("draft") is not True or release.get("published_at"):
        raise ValueError("release is already published; it cannot be overwritten")
    if release.get("immutable"):
        raise ValueError("release is immutable; asset preparation requires a mutable draft")
    if release_id is not None and release["id"] != release_id:
        raise ValueError("release ID changed while preparing")
    if (release["tag_name"] != decision["tag"] or release["target_commitish"] != decision["commit_sha"]
            or marker(decision) not in (release.get("body") or "")
            or release["prerelease"] != decision["prerelease"]):
        raise ValueError("existing draft belongs to a different product, commit, version, scope, or preparation owner")


def check_tag(name, decision):
    if remote_commit(name, decision["tag"]) != decision["commit_sha"]:
        raise ValueError("tag changed while preparing; refusing stale packages")


def guard(decision):
    if decision["preparation_scope"] not in ("full", "macos-only"):
        raise ValueError("unknown release preparation scope")
    if decision["preparation_scope"] == "macos-only" and (
            decision["product_id"] != "rust" or not decision["matrix"]["include"]
            or any(row["platform"] != "macos" for row in decision["matrix"]["include"])):
        raise ValueError("macos-only preparation must contain only Copper macOS targets")
    name = repository()
    if decision["product_id"] != "rust":
        variable = decision.get("release_gate_variable")
        gates = json.loads(os.environ.get("RELEASE_GATES", "{}"))
        if not variable or gates.get(variable) != "true":
            raise ValueError(f"{variable or 'product release gate'} must be configured after release qualification")
    if decision.get("tag_object_sha"):
        reference = api(f"repos/{name}/git/ref/tags/{quote(decision['tag'], safe='')}", optional=True)
        if reference is None or reference["object"]["sha"] != decision["tag_object_sha"]:
            raise ValueError("pushed tag was removed or moved; refusing stale preparation")
    check_tag(name, decision)
    # GitHub's compare endpoint also proves both commits exist in this repository.
    main = api(f"repos/{name}/commits/main")["sha"]
    comparison = api(f"repos/{name}/compare/{decision['commit_sha']}...{main}")
    if comparison["status"] not in ("ahead", "identical"):
        raise ValueError("tag commit is not on main history")
    evidence = []
    for workflow in decision["required_ci_workflows"]:
        runs = pages(f"repos/{name}/actions/workflows/{workflow}/runs?event=push&branch=main&head_sha={decision['commit_sha']}")
        evidence.append(eligible_ci(runs, decision, name, workflow))
    check_draft(existing_release(name, decision["tag"]), decision)
    return name, evidence


def write_outputs(decision):
    decision.setdefault("signing", "required")
    Path("release-bundle-plans").mkdir(exist_ok=True)
    Path("release-bundle-plans/decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        for key in ("commit_sha", "product_id", "version", "tag", "channel", "matrix", "preparation_scope", "controller_sha", "signing"):
            value = decision.get(key, "")
            output.write(f"{key}={json.dumps(value, separators=(',', ':')) if isinstance(value, dict) else value}\n")


def resolve(tag, preparation_scope):
    # Validation and planning are read-only. The Rust owner uses the catalog and exact local tag.
    decision = json.loads(command("target/debug/xtask", "release-version", "--tag", tag, "--preparation-scope", preparation_scope))
    reference = api(f"repos/{repository()}/git/ref/tags/{quote(tag, safe='')}", optional=True)
    if reference is None:
        raise ValueError("existing release tag is missing")
    decision["tag_object_sha"] = reference["object"]["sha"]
    name, evidence = guard(decision)
    decision["repository"] = name
    decision["ci"] = evidence
    decision["controller_sha"] = command("git", "rev-parse", "HEAD")
    write_outputs(decision)


def resolve_tag_push():
    if os.environ.get("GITHUB_EVENT_NAME") != "push" or os.environ.get("GITHUB_REF_TYPE") != "tag":
        raise ValueError("automatic draft preparation requires a tag push")
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    name = repository()
    tag = os.environ.get("GITHUB_REF_NAME", "")
    if (event.get("ref") != f"refs/tags/{tag}" or os.environ.get("GITHUB_REF") != event["ref"]
            or event.get("deleted") is not False or event.get("forced") is not False
            or event.get("repository", {}).get("full_name") != name):
        raise ValueError("tag push repository/ref is invalid, deleted or force-updated")
    tag_object = event.get("after", "")
    if not re.fullmatch(r"[0-9a-f]{40}", tag_object):
        raise ValueError("tag push does not identify a valid Git object")
    decision = json.loads(command("target/debug/xtask", "release-version", "--tag", tag, "--preparation-scope", "macos-only"))
    if decision["product_id"] != "rust":
        raise ValueError("automatic draft preparation is supported only for Copper")
    if decision["commit_sha"] != command("git", "rev-parse", "--verify", f"{tag_object}^{{commit}}"):
        raise ValueError("tag commit changed since the push")
    rows = decision["matrix"]["include"]
    if len(rows) != 1 or rows[0]["platform"] != "macos" or rows[0]["target"] != APPLE_SILICON_TARGET:
        raise ValueError("automatic Copper preparation requires the catalog Apple Silicon target")
    decision["tag_object_sha"] = tag_object
    name, evidence = guard(decision)
    decision.update(repository=name, ci=evidence, controller_sha=command("git", "rev-parse", "HEAD"))
    write_outputs(decision)


def load_decision():
    return json.loads(Path("release-bundle-plans/decision.json").read_text())


def bundle_plans():
    decision = load_decision()
    for row in decision["matrix"]["include"]:
        environment = dict(os.environ, RELEASE_VERSION=decision["version"])
        catalog_arguments = ["--catalog-commit", decision["commit_sha"]] if decision["tag"] else []
        plan = json.loads(command("target/debug/xtask", "bundle", "--product", decision["product_id"],
                                  "--platform", row["platform"], "--target", row["target"], "--channel", decision["channel"],
                                  "--signing", decision.get("signing", "required"), "--dry-run", *catalog_arguments, env=environment))
        plan["commit_sha"] = decision["commit_sha"]
        plan["preparation_scope"] = decision["preparation_scope"]
        Path("release-bundle-plans", row["plan_file"]).write_text(json.dumps(plan, indent=2) + "\n")


def checksum(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def package_inventory(decision, directory):
    expected = set()
    records = []
    for row in decision["matrix"]["include"]:
        plan = json.loads(Path("release-bundle-plans", row["plan_file"]).read_text())
        for key, value in {"product_id": decision["product_id"], "commit_sha": decision["commit_sha"],
                           "channel": decision["channel"], "version": decision["version"], "update_namespace": decision["update_namespace"],
                           "platform": row["platform"], "target": row["target"], "signing": "required",
                           "preparation_scope": decision["preparation_scope"]}.items():
            if plan.get(key) != value:
                raise ValueError(f"bundle plan {key} does not match approved release")
        filename = plan["artifact_name"]
        if Path(filename).name != filename or filename != row["artifact_name"]:
            raise ValueError("package filename does not match product/version")
        expected.add(filename)
        receipt_name = f"verification-{row['artifact']}.json"
        expected.add(receipt_name)
        receipt = json.loads((directory / receipt_name).read_text())
        for key, value in dict(identity(decision), platform=row["platform"], artifact_name=filename, signing="required").items():
            if receipt.get(key) != value:
                raise ValueError(f"verification evidence {key} does not match approved release")
        if not receipt.get("signing_identity"):
            raise ValueError("verified signing identity is missing")
        if receipt.get("sha256") != checksum(directory / filename):
            raise ValueError("verified artifact hash changed")
        if row["platform"] == "macos":
            verification = receipt.get("macos_verification", {})
            for kind in ("application", "dmg"):
                code = verification.get(kind, {})
                if (not code.get("stapled_ticket_validated") or not code.get("gatekeeper_assessed")
                        or not code.get("signature", {}).get("timestamp")):
                    raise ValueError("macOS notarization, ticket, timestamp or Gatekeeper evidence missing")
            if not verification["application"].get("nested_code") or not verification["application"].get("entitlements"):
                raise ValueError("macOS nested code or entitlement evidence missing")
        if row["platform"] == "linux":
            expected.update((filename + ".asc", filename + ".pub"))
            command("python3", "script/sign-linux-archive", "--verify-only", str(directory / filename))
        records.append(receipt)
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != expected or any(path.is_dir() or path.is_symlink() for path in directory.iterdir()):
        raise ValueError("downloaded files do not exactly match selected product/platform inventory")
    return {name: checksum(directory / name) for name in sorted(expected)}, records


def verify_remote_assets(assets, hashes):
    names = set()
    for asset in assets:
        name = asset["name"]
        if name in names or name not in hashes or asset.get("state") != "uploaded" or asset.get("digest") != "sha256:" + hashes[name]:
            raise ValueError(f"existing release asset conflicts with this verified build: {name}; review the release before retrying")
        names.add(name)
    return names


def download_packages():
    decision = load_decision()
    destination = Path("release-artifacts")
    destination.mkdir(exist_ok=False)
    for row in decision["matrix"]["include"]:
        command("gh", "run", "download", os.environ["GITHUB_RUN_ID"], "--repo", repository(), "--name", row["artifact"], "--dir", str(destination))


def semver_key(value):
    match = re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?", value)
    if not match:
        raise ValueError("invalid semantic version")
    pre = match[4]
    identifiers = []
    if pre:
        for part in pre.split("."):
            if part.isdecimal():
                if part != str(int(part)):
                    raise ValueError("invalid numeric prerelease")
                identifiers.append((0, int(part)))
            else:
                identifiers.append((1, part))
    return tuple(map(int, match.group(1, 2, 3))) + (not bool(pre), tuple(identifiers))


def published_scope(release):
    body = release.get("body") or ""
    marker_match = re.search(r"<!-- made-product-release (\{[^\n]*\}) -->", body)
    if marker_match:
        return json.loads(marker_match[1]).get("preparation_scope", "full")
    if "made-product-release" in body:
        raise ValueError("malformed product release ownership marker")
    return "full"


def verification_asset_names(product_id, version):
    prefix = f"{product_id}-{version}"
    return f"{prefix}-verification.json", f"{prefix}-SHA256SUMS"


def verified_release(release, product, version, scope):
    body = release.get("body") or ""
    ownership = re.search(r"<!-- made-product-release (\{[^\n]*\}) -->", body)
    if not ownership:
        return False
    decision = json.loads(ownership[1])
    if (decision.get("product_id") != product["id"] or decision.get("version") != version
            or decision.get("tag") != release["tag_name"]
            or decision.get("update_namespace") != product["update_namespace"]
            or decision.get("preparation_scope") != scope
            or decision.get("channel") != ("preview" if release["prerelease"] else "stable")
            or decision.get("commit_sha") != release.get("target_commitish")
            or not re.fullmatch(r"[0-9a-f]{40}", decision.get("commit_sha", ""))
            or ready_marker(decision) not in body):
        return False
    targets = [target for target in product.get("targets", [])
               if scope == "full" or target.startswith("macos-")]
    if not targets:
        return False
    required = set(verification_asset_names(product["id"], version))
    for target in targets:
        system, architecture = target.split("-", 1)
        extension = {"macos": "dmg", "linux": "tar.gz", "windows": "exe"}[system]
        package = product["artifact_name"].format(product=product["id"], version=version,
                                                  platform=system, arch=architecture, extension=extension)
        required.update((package, f"verification-{product['id']}-{target}.json"))
        if system == "linux":
            required.update((package + ".asc", package + ".pub"))
    assets = release.get("assets", [])
    names = {asset["name"] for asset in assets
             if asset.get("state") == "uploaded" and re.fullmatch(r"sha256:[0-9a-f]{64}", asset.get("digest") or "")}
    return required == names and len(assets) == len(required)


def latest_release(releases, product, channel, platform=None):
    if channel not in ("stable", "preview"):
        raise ValueError("unsupported product release channel")
    candidates = []
    for release in releases:
        tag = release.get("tag_name", "")
        prefix = product["release_tag_prefix"]
        if release.get("draft") is not False or not release.get("published_at") or not tag.startswith(prefix):
            continue
        try:
            scope = published_scope(release)
            if scope not in ("full", "macos-only") or (scope == "macos-only" and platform != "macos"):
                continue
            version = tag[len(prefix):]
            key = semver_key(version)
        except ValueError:
            continue
        prerelease = not key[3]
        if prerelease != release.get("prerelease") or prerelease != (channel == "preview"):
            continue
        if product.get("id") == "rust" or prerelease:
            if not verified_release(release, product, version, scope):
                continue
        elif "targets" in product:
            names = {asset.get("name") for asset in release.get("assets", []) if asset.get("state") == "uploaded"}
            expected = set()
            for target in product["targets"]:
                target_platform, architecture = target.split("-", 1)
                if platform and platform != target_platform:
                    continue
                extension = {"macos": "dmg", "windows": "exe", "linux": "tar.gz"}[target_platform]
                expected.add(product["artifact_name"].format(product=product["id"], version=version,
                                                            platform=target_platform, arch=architecture, extension=extension))
            if not expected or not expected <= names:
                continue
        elif platform:
            continue
        candidates.append((key, release))
    return max(candidates, key=lambda entry: (entry[0], entry[1]["published_at"], entry[1]["tag_name"]))[1] if candidates else None


def release_notes(decision, name):
    products = catalog_products()
    product = next(product for product in products if product["id"] == decision["product_id"])
    releases = pages(f"repos/{name}/releases")
    # Only earlier releases of this product/channel delimit these notes.
    previous = latest_release([release for release in releases if release.get("tag_name") != decision["tag"]
                               and release.get("tag_name", "").startswith(product["release_tag_prefix"])
                               and _older(release["tag_name"][len(product["release_tag_prefix"]):], decision["version"])], product, decision["channel"])
    if previous:
        comparison = f"https://github.com/{name}/compare/{quote(previous['tag_name'], safe='')}...{quote(decision['tag'], safe='')}"
        return (f"{decision['display_name']} {decision['version']} ({decision['channel']}).\n\n"
                f"[Changes from the preceding published {decision['display_name']} release in this channel]({comparison}).")
    return (f"{decision['display_name']} {decision['version']} from `{decision['commit_sha']}`. "
            "No earlier published release of this product/channel is available as a notes baseline.")


def _older(version, target):
    try:
        return semver_key(version) < semver_key(target)
    except ValueError:
        return False


def checked_decision():
    decision = load_decision()
    for key, environment_key in (("commit_sha", "EXPECTED_COMMIT"), ("product_id", "EXPECTED_PRODUCT"),
                                 ("tag", "EXPECTED_TAG"), ("preparation_scope", "EXPECTED_SCOPE")):
        if os.environ.get(environment_key) != decision[key]:
            raise ValueError("downloaded decision differs from approved job outputs")
    if decision.get("signing", "required") != "required" or not decision["tag"] or decision["channel"] not in ("stable", "preview"):
        raise ValueError("development artifacts cannot be attached to a release")
    if os.environ.get("GITHUB_EVENT_NAME") == "push" and os.environ.get("GITHUB_REF_TYPE") == "tag":
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        if decision.get("tag_object_sha") != event.get("after") or event.get("ref") != f"refs/tags/{decision['tag']}":
            raise ValueError("downloaded decision differs from the approved tag push")
    return decision


def draft():
    decision = checked_decision()
    name, evidence = guard(decision)
    directory = Path("release-artifacts")
    hashes, records = package_inventory(decision, directory)
    notes = release_notes(decision, name)
    manifest_name, checksum_name = verification_asset_names(decision["product_id"], decision["version"])
    (directory / manifest_name).write_text(json.dumps({**identity(decision), "packages": records, "sha256": hashes}, indent=2, sort_keys=True) + "\n")
    hashes[manifest_name] = checksum(directory / manifest_name)
    (directory / checksum_name).write_text("".join(f"{digest}  {filename}\n" for filename, digest in sorted(hashes.items())))
    hashes[checksum_name] = checksum(directory / checksum_name)
    run_url = f"https://github.com/{name}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    body = (f"{marker(decision)}\n\n{notes}\n\n### Verification\n\n"
            f"Product: `{decision['product_id']}` · Version: `{decision['version']}` · Channel: `{decision['channel']}`\n\n"
            f"Commit: `{decision['commit_sha']}`\n\nPackages passed required signing and platform smoke verification. "
            f"See `{manifest_name}` and `{checksum_name}`.\n\nPreparation: {run_url}\n\n"
            + "\n".join(f"Required CI: {run['html_url']}" for run in evidence)
            + "\n\nReview the packages and verification evidence before clicking **Publish release**.\n")
    label = " (macOS only)" if decision["preparation_scope"] == "macos-only" else ""
    if label:
        body += "\n**macOS-only preparation.** Windows and Linux packages are not included or qualified by this release. This release is excluded from their updater selection and from the default full-platform release feed.\n"
    release = existing_release(name, decision["tag"])
    check_draft(release, decision)
    verify_remote_assets(release["assets"] if release else [], hashes)
    if release is None:
        # guard() verified the existing tag. target_commitish cannot cause a new tag in this path.
        guard(decision)
        release = api(f"repos/{name}/releases", "POST", {
            "tag_name": decision["tag"], "target_commitish": decision["commit_sha"],
            "name": f"{decision['display_name']} {decision['version']}{label} (preparing)",
            "body": marker(decision) + "\n\nPreparation is incomplete. Do not publish.",
            "draft": True, "prerelease": decision["prerelease"], "make_latest": "false"})
    release_id = release["id"]
    for filename in sorted(hashes):
        guard(decision)
        release = api(f"repos/{name}/releases/{release_id}")
        check_draft(release, decision, release_id)
        present = verify_remote_assets(release["assets"], hashes)
        if filename not in present:
            upload_asset(name, release, directory / filename)
    guard(decision)
    release = api(f"repos/{name}/releases/{release_id}")
    check_draft(release, decision, release_id)
    if verify_remote_assets(release["assets"], hashes) != set(hashes):
        raise ValueError("draft asset inventory is incomplete")
    placeholder = marker(decision) + "\n\nPreparation is incomplete. Do not publish."
    if release.get("body") != placeholder:
        body = release["body"]
    if ready_marker(decision) not in body:
        body += "\n\n" + ready_marker(decision)
    # Never send draft=false, even when updating a fully verified draft.
    api(f"repos/{name}/releases/{release_id}", "PATCH", {
        "name": f"{decision['display_name']} {decision['version']}{label}", "body": body, "make_latest": "false"})
    print(f"Verified draft ready for human review: https://github.com/{name}/releases")


def upload_asset(name, release, path):
    # Keep the write token on GitHub's uploads host and stream the package.
    endpoint = f"/repos/{name}/releases/{release['id']}/assets"
    if release.get("upload_url", "").split("{", 1)[0] != "https://uploads.github.com" + endpoint:
        raise ValueError("release asset endpoint does not match the verified release ID")
    connection = HTTPSConnection("uploads.github.com", timeout=120)
    try:
        with path.open("rb") as source:
            connection.request("POST", endpoint + "?" + urlencode({"name": path.name}), source, {
                "Authorization": "Bearer " + os.environ["GH_TOKEN"],
                "Content-Type": "application/octet-stream",
                "Content-Length": str(path.stat().st_size),
                "Accept": "application/vnd.github+json",
            })
            status = connection.getresponse().status
    except (OSError, HTTPException):
        raise RuntimeError(f"upload interrupted for {path.name}; retry the upload job with the same build artifacts") from None
    finally:
        connection.close()
    if status != 201:
        raise RuntimeError(f"upload failed for {path.name} (HTTP {status}); check release immutability, asset conflicts and token permissions")


def contract(version, commit, preparation_scope, development=False):
    # Non-publishing signing validation reuses the same catalog/matrix and bundlers.
    semver_key(version)
    product = next(product for product in catalog_products() if product["id"] == "rust")
    # The resolver remains authoritative; contract validation never resolves or creates a Git tag.
    result = json.loads(command("target/debug/xtask", "release-version", "--tag", product["release_tag_prefix"] + version, "--preparation-scope", preparation_scope, "--plan-only"))
    result["commit_sha"] = commit
    result["tag"] = ""
    result["signing"] = "off" if development else "required"
    write_outputs(result)


def latest(product_id, channel, platform):
    products = [product for product in catalog_products() if product["id"] == product_id]
    if len(products) != 1 or products[0]["status"] != "enabled":
        raise ValueError("unknown or unqualified product")
    product = products[0]
    release = latest_release(pages(f"repos/{repository()}/releases"), product, channel, platform)
    if release is None:
        raise ValueError("no published release for this product/channel")
    assets = release["assets"]
    if platform:
        suffix = {"macos": ".dmg", "windows": ".exe", "linux": ".tar.gz"}[platform]
        assets = [asset for asset in assets if asset["name"].endswith(suffix)]
    print(json.dumps({"product_id": product_id, "update_namespace": product["update_namespace"],
                      "channel": channel, "version": release["tag_name"][len(product["release_tag_prefix"]):],
                      "tag": release["tag_name"], "assets": assets}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    resolve_parser = commands.add_parser("resolve")
    resolve_parser.add_argument("--tag", required=True)
    resolve_parser.add_argument("--preparation-scope", choices=("full", "macos-only"), default="full")
    contract_parser = commands.add_parser("contract")
    contract_parser.add_argument("--development", action="store_true")
    contract_parser.add_argument("--version", required=True)
    contract_parser.add_argument("--commit", required=True)
    contract_parser.add_argument("--preparation-scope", choices=("full", "macos-only"), default="full")
    latest_parser = commands.add_parser("latest")
    latest_parser.add_argument("--product", required=True)
    latest_parser.add_argument("--channel", choices=("stable", "preview"), required=True)
    latest_parser.add_argument("--platform", choices=("macos", "linux", "windows"))
    for name in ("bundle-plans", "download-packages", "draft", "resolve-tag-push"):
        commands.add_parser(name)
    arguments = parser.parse_args()
    try:
        if arguments.operation == "resolve":
            resolve(arguments.tag, arguments.preparation_scope)
        elif arguments.operation == "contract":
            contract(arguments.version, arguments.commit, arguments.preparation_scope, arguments.development)
        elif arguments.operation == "latest":
            latest(arguments.product, arguments.channel, arguments.platform)
        elif arguments.operation == "bundle-plans":
            bundle_plans()
        elif arguments.operation == "download-packages":
            download_packages()
        elif arguments.operation == "resolve-tag-push":
            resolve_tag_push()
        elif arguments.operation == "draft":
            draft()
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        print(f"Release preparation blocked: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
