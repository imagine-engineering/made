# Multiplayer build profiles

Zed supports two release configurations. `multiplayer-tools` is the only public Cargo feature for the Collaborative Workspace and Buzz-derived platform. It is opt-in and is not part of Zed's default feature set.

The profiles are:

- `standard`: Editor Zed with `multiplayer-tools` disabled. Multiplayer-only application composition, Buzz services, compatibility transports, migrations, jobs and assets must not be compiled, registered or packaged.
- `multiplayer`: Zed with `multiplayer-tools` enabled explicitly. Shared Editor, project, worktree, Git, ACP, credentials, settings and existing collaboration owners remain the same implementations used by Standard Zed.

Use the profile helper to obtain deterministic build and artifact metadata:

```sh
script/multiplayer-build-profile standard --dry-run
script/multiplayer-build-profile multiplayer --dry-run
```

The corresponding desktop commands are:

```sh
cargo build -p zed --no-default-features
cargo build -p zed --no-default-features --features multiplayer-tools

cargo run -p zed --no-default-features
cargo run -p zed --no-default-features --features multiplayer-tools
```

Release automation must record the helper output with the artifact metadata. It must not infer the profile from a directory name, package name or runtime setting. Multiplayer packages must use the `multiplayer` profile and therefore record `artifact_capability=multiplayer-tools:true`.

## Artifact inspection

Inspect an assembled artifact directory before signing or publishing it:

```sh
script/multiplayer-build-profile standard --inspect path/to/assembled-artifact
script/multiplayer-build-profile multiplayer --inspect path/to/assembled-artifact
```

The Standard profile fails when it finds known Buzz-owned service, protocol, migration or deployment payload names. The Multiplayer profile reports those entries without rejecting them. This inspection complements the Cargo dependency-tree check; hiding a UI surface is not sufficient isolation.

The existing Zed collaboration server and clients are shared infrastructure. The profile helper deliberately reports `shared_collab_deployment=retained` and does not reject generic `collab` artifacts. Only functionality classified as Collaborative Workspace-exclusive by the capability audit belongs behind this release capability.

Disabling the profile never deletes collaborative data, credentials, signing material or a saved Collaborative Workspace preference. Rollback publishes a Standard artifact, stops multiplayer-exclusive services and migrations according to the migration plan, and preserves those stores for a later compatible Multiplayer build.

## Verification

Run the fast local matrix while developing feature-boundary changes:

```sh
script/check-multiplayer-tools --quick
```

Run the release-equivalent build, tests, warning-denied Clippy, package smoke and dependency audit for both profiles before landing a boundary change:

```sh
script/check-multiplayer-tools --full
```

One profile can be diagnosed independently with `--profile standard` or `--profile multiplayer`. To inspect a captured Cargo tree without compiling, use:

```sh
script/check-multiplayer-tools --tree-only --tree-file path/to/cargo-tree.txt
```

Both modes run unfiltered library suites, not name-filtered subsets. `--quick` runs the whole `workspace` library suite for each profile in addition to the capability-admission tests, and `--full` also runs the `git_ui`, `sidebar` and `agent_ui` library suites for each profile. A name-filtered run cannot prove a profile is healthy: a test that asserts behavior gated behind a feature the profile does not enable still compiles, so only an unfiltered run of its owning crate reports the failure.

The Standard dependency audit must remain free of every package owned exclusively by the Collaborative Workspace capability audit. When a leaf introduces another exclusive crate or packaged payload, update `script/check-multiplayer-tools` or `script/multiplayer-build-profile` in that same change.

## Classifying future changes

Classify by semantic ownership, not by repository path:

- Existing Editor, project, worktree, Git, ACP, credentials, settings, media, audio, remote-development and collaboration behavior stays shared and always compiled.
- A Collaborative Workspace-only adapter, extension, action, view, asset, migration, job or registration is enabled only through `multiplayer-tools`, even when it lives in a shared crate.
- Only dependency-light version, setting and deep-link representations needed to preserve state or reject an unsupported operation remain in Standard Zed. Rejection must occur before tenant or resource lookup.
- Server and deployment artifacts exclusive to the Collaborative Workspace use the same explicit release capability. Generic Zed collaboration infrastructure remains shared.

If a proposed task writes both shared behavior and an independently reviewable multiplayer-only implementation, split and sequence it before coding. Both configurations remain supported after every boundary leaf; hiding UI while still compiling or shipping exclusive dependencies is a failure.
