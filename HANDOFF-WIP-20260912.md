# WIP handoff — 2026-09-12

This transfer checkpoint preserves unfinished work. It does not authorize or perform
more implementation, deployment, extraction, source retirement or migrations.
Stop after pushing. The older full-remediation goal is blocked pending a new user
instruction; this handoff does not resume it.

## What is complete and what is unfinished

- UI remediation commit `237e0eb308cb70e34fb43a3d50986ec7557dc2b3`, integrated in
  `3f39fddc16a0586380c603435c1de4cad030e8c2`, was pushed and verified earlier.
  Native review tabs/footer, compact gutter, startup review/draft fixes, restored
  item reconciliation and F6 region routing are implemented. Historical validation:
  1,057 owner tests, both build profiles, profile gates, `./script/clippy`, and
  128 final production-renderer captures passed. This handoff does not rerun them.
- `.agents/specs/sim-private-extraction/` is already committed in
  `732682edcb3e365151b23107acf0c17e792a6008` and was present on `origin/dev` at the
  handoff fetch. Preserve that commit; there is no remaining untracked plan to
  recreate. It is planning only: no extraction implementation, private repository
  creation or migration has happened. Its 38 implementation leaves remain pending.
- `deploy/collaboration/migrations/tests/smoke.sh` adds a disposable PostgreSQL
  in-flight DDL interruption/crash/recovery scenario. It is **WIP and unvalidated**.
  It holds a table lock, waits for the final migration, kills only its disposable
  PostgreSQL container, then checks atomic rollback and resume. The earlier ordinary
  SQL smoke passed, but that does not validate this addition. Prior Docker probes
  timed out. No new SQL/Docker run was performed for this preservation request.
- Remaining unique intermediate UI captures are committed as historical WIP evidence;
  duplicate content is mapped to its existing Git copy in the handoff inventory,
  including failed/incomplete runs. One file is zero bytes:
  `test-results/collaborative-workspace/remediation-20260912/native-footer-width-light-retry/menu-non-active-collapsed.png`.
  It is preserved, not passing evidence. Current final captures remain under
  `native-region-focus-light/` and `native-region-focus-dark/`.

Read `test-results/collaborative-workspace/remediation-20260912/checkpoint.md`,
`task-checklist.json`, `validation.json`, `native-checklist.json` and
`native-interaction-unlocked.md` for exact audit/task/criteria mappings and gaps.
The new `handoff-inventory.json` records every nonignored untracked file inspected,
its hash, transfer path and incomplete-artifact status. Historical notes saying
intermediate directories are local-only are superseded by this WIP transfer.

## Validation performed for this handoff

- `git diff --check` and staged diff whitespace checks: pass.
- `bash -n deploy/collaboration/migrations/tests/smoke.sh`: pass (syntax only).
- `python3 .agents/skills/feature-spec/scripts/validate_spec.py .agents/specs/sim-private-extraction --require-complete --dialect canonical`:
  pass with documented cross-subsystem/manual-review warnings; not implementation acceptance.
- All 143 untracked raw log/diff copies were verified byte-identical to existing
  tracked gzip archives. An additional 265 files exactly duplicate already tracked
  or newly included evidence, leaving 165 unique new evidence files (about 93 MB). Remaining text was checked for private keys and common
  token/access-key patterns; none found. The SQL `migration-canary` value is an
  existing disposable test constant, not an operational credential.
- 406 nonempty intermediate PNGs have valid chunk structure and no text metadata;
  the zero-byte file above is explicitly incomplete. Structural inspection and
  representative visual review do not establish parity or rule out every possible
  sensitive pixel. These are the previously generated controlled audit captures.
- No production functionality, build or integration test is newly claimed to pass.
  No Rust implementation changed in this transfer, so no new Rust lint/build run
  was necessary. Earlier actual `./script/clippy` results remain in the audit logs.

## Next commands on the other machine

First obtain the pushed branch without overwriting any local work:

```sh
git fetch origin dev
git switch dev
git merge --ff-only origin/dev
git status --short
```

If the other machine has divergent commits or local edits, stop and preserve them
before integration; never reset or force-push to get a clean checkout.

Provision the repository's pinned Rust toolchain, platform SDK/linker, native
libraries, Git and Docker. Native Mac validation needs an unlocked desktop and
Computer Use/accessibility permissions. Fetch dependencies while online before
using the historical `--offline` commands. Ensure substantial free build disk space;
several prior runs failed from disk exhaustion. Do not delete source/evidence to
recover space. Rebuild rather than copying host-specific executables.

To validate the saved SQL experiment, using disposable resources only:

```sh
docker info
bash -n deploy/collaboration/migrations/tests/smoke.sh
bash deploy/collaboration/migrations/tests/smoke.sh > /tmp/sql-interruption-handoff.log 2>&1
```

Require a responsive daemon and access to required build/container images first.
Retain the exit code and complete output. Investigate wait-query matching, lock
reachability, migrator termination and transaction rollback if it fails; do not
label it passed from syntax checking or the older smoke run.

After explicit authorization to resume UI validation:

```sh
cargo build --locked -p zed --bin zed --no-default-features --features zed/multiplayer-tools,zed/rust-tools,gpui_platform/runtime_shaders
cargo test --locked -p workspace -p git_ui -p sidebar -p agent_ui -p gpui_platform --lib --features workspace/multiplayer-tools,workspace/test-support,git_ui/multiplayer-tools,sidebar/multiplayer-tools,agent_ui/multiplayer-tools,gpui_platform/runtime_shaders
./script/clippy --locked --no-all-features -p workspace -p agent_ui -p zed -p gpui_platform --no-default-features --features workspace/multiplayer-tools,workspace/test-support,agent_ui/multiplayer-tools,zed/multiplayer-tools,zed/rust-tools,zed/visual-tests,gpui_platform/runtime_shaders
script/check-multiplayer-tools --quick
cargo build --locked -p zed --bin zed --no-default-features --features zed/agentic-tools,zed/rust-tools,gpui_platform/runtime_shaders
```

Building Standard last replaces `target/debug/zed`; rebuild/copy the intended
Multiplayer candidate before native acceptance. Made is the product, Copper its
Rust profile; `zed` is its inherited binary name. Verify executable path/features
and source hash before acceptance, never use an unrelated installed Zed build.
Use a fresh isolated `--user-data-dir` and disposable Git project. Recheck native
F6/Shift-F6, physical divider drag, minimize/Dock behavior and screen-reader output.
Remaining density/truncated labels, exact unavailable-item native restart, live
populated conversations and authenticated two-client workflows are still open.
Authentication remains deferred; obtain explicit access and two seeded accounts
before those checks. Do not replace service evidence with fixture ACP results.

For the extraction plan, read its README/design/tasks/validation files first.
Its proposed preparation scripts do not exist yet. Do not execute speculative
commands as though implementation exists. Several parity generators write even
with `--check`; use a disposable copy only after a separate implementation request.

## What Git does not transfer

- **408 redundant evidence files** were removed after explicit cleanup authorization
  and byte-for-byte verification against pushed Git content: 143 raw `.log`/`.diff`
  copies already preserved as tracked `.gz`, plus 265 exact content duplicates.
  Every path/hash and its transferred source is recorded in `handoff-inventory.json`.
  Restore original paths if needed with:

  ```sh
  python3 - <<'PY'
  import gzip, hashlib, json, pathlib
  data = json.loads(pathlib.Path('test-results/collaborative-workspace/remediation-20260912/handoff-inventory.json').read_text())
  for item in data['files']:
      destination = pathlib.Path(item['path'])
      if not destination.exists():
          data = pathlib.Path(item['git_path']).read_bytes()
          if item['transfer'].startswith('already tracked'):
              data = gzip.decompress(data)
          assert hashlib.sha256(data).hexdigest() == item['sha256']
          destination.parent.mkdir(parents=True, exist_ok=True)
          destination.write_bytes(data)
  PY
  ```

- **Build caches/dependencies:** `target/`, nested `target/`, `node_modules/`,
  Python caches/environments and generated tooling lint lockfiles. Rebuild/fetch.
- **Machine state:** `.idea/`, `.DS_Store`, ignored audit runtime profiles/databases,
  local credentials/keychains, user settings and desktop permissions. Excluded;
  configure fresh. Never put operational tokens or profile databases into Git.
- **Native audit environment:** `/private/tmp/made-remediation-20260912-8teo9wou/`
  contains the local launcher, copied candidate binaries, isolated profiles and
  disposable real Git project. It is not transferred. Rebuild candidates and
  recreate a disposable project/profile. To reproduce the exact prior persistence
  state, separately and securely transfer a reviewed profile/project copy from
  this location; do not publish it. Paths in existing provenance describe the old
  host and are not expected to exist on the new machine.
- **Ignored reference/oracle sources:** `projects/sim/{ComfyUI,ComfyUI-Frontend,Comfy-Desktop,sim-cli,docs,embedded-docs}`,
  and ignored `projects/{buzz,godot,goose}` reference trees. These are not newly
  tracked code changes and were not bulk-imported. Provision exact snapshots from
  the recorded manifests/verified archives; preserve local source modifications
  through a separately reviewed secure archive if needed. The extraction design
  explains why enclosing-repository HEAD is not their source revision. The full
  collaboration inventory may require the Buzz reference checkout.
- Older ignored `test-results/collaborative-workspace/audit-20260910/native-*`
  captures and `baseline.diff` remain local historical material; current committed
  acceptance captures are sufficient for this UI checkpoint, but copy those older
  artifacts separately if reproducing that historical audit. Its `runtime-profile*`
  and `runtime-project` are machine-specific and must be reviewed before transfer.

The 408 verified redundant evidence copies were removed in the subsequent authorized
cleanup; their exact pushed equivalents and SHA-256 values are recorded in
`test-results/collaborative-workspace/remediation-20260912/redundant-evidence-cleanup.json`.
All unique evidence and excluded machine/reference resources remain untouched.
No unique unfinished work was deleted.
The handoff authorizes preservation and publication only; stop after verification.
