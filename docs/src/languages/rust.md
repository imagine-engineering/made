---
title: Rust
description: "Configure Rust language support in Zed, including language servers, formatting, and debugging."
---

# Rust

Rust support is available natively in Zed.

- Tree-sitter: [tree-sitter/tree-sitter-rust](https://github.com/tree-sitter/tree-sitter-rust)
- Language Server: [rust-lang/rust-analyzer](https://github.com/rust-lang/rust-analyzer)
- Debug Adapter: [CodeLLDB](https://github.com/vadimcn/codelldb) (primary), [GDB](https://sourceware.org/gdb/) (secondary, not available on Apple silicon)

## Cargo workspace tools

Zed distributions built with the `rust-tools` capability include a dockable **Cargo** panel and Rust discovery and actions in the shared **Tests** panel. Disabling this capability removes the Cargo panel and its Rust-specific workflows; the shared Tests panel can remain available through other enabled capabilities. Existing Rust language support, grammars, rust-analyzer integration, and language tasks remain available independently.

The Cargo panel discovers every visible `Cargo.toml` through the authoritative project host. It supports virtual workspaces, standalone packages, multiple Cargo roots, and local, SSH/remote-server, multiplayer, WSL, and development-container projects where those project modes are otherwise supported. The tree shows:

- workspace members and their manifests;
- library, binary, example, test, benchmark, and build-script targets;
- package-defined features separately from features enabled in the resolved workspace;
- direct normal, development, build, optional, renamed, target-specific, path, registry, Git, and workspace-inherited dependencies; and
- declared profiles and toolchain files, the host compiler, Cargo's unresolved default target, and the active Cargo preset.

Dependencies are intentionally direct-only. The panel does not recursively render Cargo's resolved transitive graph, so dependency cycles cannot create an unbounded tree. Open a package to navigate to its `Cargo.toml`, or a target to navigate to its source file. Refresh, Expand All, and Collapse All are always explicit panel operations; relevant manifest, lockfile, toolchain, and Rust-source changes also trigger debounced background refreshes.

Cargo workspace performance checks include a hermetic 1,000-package model gate and a 250 ms foreground budget for at least 10,000 visible rows. The automated GPUI test drives production refresh, notice and preset paths through the real panel with a fixture metadata runner, including reconciliation, layout and painting of a bounded viewport. It reports background projection and total elapsed time separately, and uses virtual executor timers for debounce waits. This verifies production UI code; it does not measure physical GPU presentation or certify transport and screen-reader behavior.

### Cargo presets and actions

Cargo presets are versioned settings under `cargo.presets`. User settings provide defaults, trusted project settings may override entries by ID, and an invalid entry is isolated without disabling other presets. A preset can select a Cargo subcommand, package or workspace scope, target, profile, features, target triple, structured arguments and environment, working directory, and ordinary task presentation. The selected preset and safe package/target selection are restored per workspace; environment values and result history are never stored in workspace panel state.

Build, Check, Run, Test, Bench, Doc, Clippy, Fmt, Clean, and the offline locked dependency Tree action compile the selected Cargo node and preset into an ordinary Zed task. Clean requires a second explicit invocation because it removes build artifacts. Debug compiles into an ordinary DAP scenario using the Cargo locator and CodeLLDB. Cargo metadata discovery never becomes a second build or test process runner. Existing task terminals, history, cancellation, rerun behavior, `tasks.json`, `debug.json`, and DAP remain authoritative.

Run with Coverage uses an existing `cargo-llvm-cov` installation on the project host. It runs an ordinary Cargo task, declares a bounded project-relative JSON artifact, and publishes language-neutral gutter and summary facts only after the task succeeds and the authoritative host validates the report. Zed does not install the collector, fetch dependencies, parse terminal output, or fall back to a client-local report. If the collector is missing, install it deliberately on the project host and invoke the action again.

External profiling remains task-first. A configured external profiler may wrap a compiled Cargo preset and declare one bounded SVG, HTML, or file artifact for Zed to open after success. An experimental macOS native profile model and viewer are implemented behind the explicit local `rust-native-profiling-experimental` opt-in. The macOS collector/privacy ADR is approved; selected-workload capture, runtime-helper shipping and remaining certification are incomplete. Zed does not install the profiler.

The Tests panel discovers Rust unit, integration, binary-harness, example-harness, benchmark, ignored, and doctest cases using bounded structured output on the project host. Run, cancel, rerun-failed, terminal reveal, source navigation, and supported debug actions route through the same Tasks and DAP systems. Doctest debugging is currently unavailable and is reported as an action-specific reason rather than silently running another command.

Cargo metadata and test discovery require a trusted project and an available Cargo toolchain on the authoritative host. Metadata follows that host’s Cargo network settings and can download missing dependencies during normal operation. Explicit `CARGO_NET_OFFLINE` and Cargo `net.offline` settings remain effective. This is independent of local-model inference network isolation. Toolchain auto-installation remains blocked, and passive compiler/test collectors retain their offline policy. These operations never fall back to a client-local filesystem. Partial failures retain safe stale results where possible. A disconnected host or feature/protocol mismatch produces a stable actionable state until the connection or build capability changes.

Failed metadata rows show the affected manifest and Cargo’s full bounded diagnostic, including a missing dependency and offline restriction when Cargo reports them. Double-click a row to open its manifest, review the workspace/host Cargo configuration or environment, and fetch dependencies explicitly in a network-enabled terminal if intended. Use **Refresh** to retry. Copper never silently overrides an intentional offline setting.

### Scope and roadmap

The Rust-specific Cargo dashboard and test discovery described here require the existing `rust-tools` build capability. Shared Tests infrastructure remains available independently. Named Cargo presets and the other unfinished workflows below require separate explicit local opt-ins.

- **Next:** refine active-configuration ergonomics and broaden fixture-backed test protocol compatibility when stable Rust tooling exposes it safely.
- **Implemented bounded paths:** language-neutral call hierarchy, direct dependency insight and Task-declared external profiling artifacts. Call hierarchy remains available independently of the experimental Rust workflow flags.
- **Experimental, local opt-in:** coverage (`rust-coverage-experimental`), named Cargo presets (`rust-named-cargo-presets-experimental`), exact doctest actions (`rust-exact-doctest-actions-experimental`) and macOS native profiling (`rust-native-profiling-experimental`). These flags default Off, including for staff.
- **Incomplete:** full macOS native capture/viewer and helper shipping qualification, physical project-mode and screen-reader certification, and loaded production-performance evidence. Linux and Windows native profiling are deferred; other platform requirements remain active.
- **External:** Cargo, rustc, rust-analyzer, and debug-adapter behavior remains owned by those tools and protocols.
- **Rejected for this workspace:** terminal-output scraping, automatic `cargo-nextest` installation, a second Rust semantic index, a universal build-system model, and a public provider API.
- **Out of scope:** manifest or dependency mutation, automatic network activity, vulnerability/license auditing, and making every existing Rust language component conditional on `rust-tools`.

### Environment certification harness

Maintainers can run the hermetic matrix coordinator from a Zed checkout:

```sh
./script/test-rust-tools-environments --matrix --offline
```

The local cell runs offline Cargo CLI build/test smoke checks and a captured-fixture stale-state test. Its bounded POSIX process-group signal probe checks CLI cancellation and cleanup; it does not prove cancellation through Zed Tasks or remote transports. That signal probe is explicitly unavailable on non-POSIX hosts. Run `python3 script/test-rust-tools-environments-failures.py` to check coordinator failure propagation separately. SSH, WSL, development-container, and multiplayer rows are reported as unavailable until their documented environment variables or evidence file are supplied. An `observed` row proves that the fixture ran in that environment; it is not a substitute for the production Zed transport checklist. Release certification must run the Cargo and Tests panels through the actual project mode, verify that execution occurs on the authoritative host, disconnect/reconnect during discovery, cancel a running test, and debug a supported case. `ZED_RUST_TOOLS_PHYSICAL_EVIDENCE` supplies dated, operator-reviewed physical records to `script/test-rust-tools-evidence.py`, which checks their revision, implementation digest, freshness, checklist and required cells before the coordinator emits certified rows. `ZED_RUST_TOOLS_REQUIRE_PHYSICAL=1` fails closed without accepted required records; automated smoke, `observed` checkout probes and provisioning alone do not certify a physical cell.

Manual screen-reader certification is tracked in `.agents/specs/rust-workspace/rust-tools-platform/accessibility-evidence.md`. Automated role/name/state and keyboard-navigation tests do not by themselves certify macOS VoiceOver or Windows NVDA.

<!--
TBD: Polish Rust docs. Zed has strong Rust support, and the docs should reflect that clearly.
TBD: Users may not know what inlayHints, don't start there.
TBD: Provide explicit examples not just `....`
-->

## Inlay Hints

The following configuration can be used to change the inlay hint settings for `rust-analyzer` in Rust:

```json [settings]
{
  "lsp": {
    "rust-analyzer": {
      "initialization_options": {
        "inlayHints": {
          "maxLength": null,
          "lifetimeElisionHints": {
            "enable": "skip_trivial",
            "useParameterNames": true
          },
          "closureReturnTypeHints": {
            "enable": "always"
          }
        }
      }
    }
  }
}
```

See [Inlay Hints](https://rust-analyzer.github.io/book/features.html#inlay-hints) in the Rust Analyzer Manual for more information.

## Target directory

The `rust-analyzer` target directory can be set in `initialization_options`:

```json [settings]
{
  "lsp": {
    "rust-analyzer": {
      "initialization_options": {
        "rust": {
          "analyzerTargetDir": true
        }
      }
    }
  }
}
```

A `true` setting will set the target directory to `target/rust-analyzer`. You can set a custom directory with a string like `"target/analyzer"` instead of `true`.

## Binary

You can configure which `rust-analyzer` binary Zed should use.

By default, Zed will try to find a `rust-analyzer` in your `$PATH` and try to use that. If that binary successfully executes `rust-analyzer --help`, it's used. Otherwise, Zed will fall back to installing its own stable `rust-analyzer` version and use that.

If you want to install a pre-release `rust-analyzer` version instead, you can instruct Zed to do so by setting `pre_release` to `true` in your `settings.json`:

```json [settings]
{
  "lsp": {
    "rust-analyzer": {
      "fetch": {
        "pre_release": true
      }
    }
  }
}
```

If you want to disable Zed looking for a `rust-analyzer` binary, you can set `ignore_system_version` to `true` in your `settings.json`:

```json [settings]
{
  "lsp": {
    "rust-analyzer": {
      "binary": {
        "ignore_system_version": true
      }
    }
  }
}
```

If you want to use a binary in a custom location, you can specify a `path` and optional `arguments`:

```json [settings]
{
  "lsp": {
    "rust-analyzer": {
      "binary": {
        "path": "/Users/example/bin/rust-analyzer",
        "arguments": []
      }
    }
  }
}
```

This `"path"` has to be an absolute path.

## Alternate Targets

If you want rust-analyzer to provide diagnostics for a target other than your current platform (e.g. for windows when running on macOS) you can use the following Zed lsp settings:

```json [settings]
{
  "lsp": {
    "rust-analyzer": {
      "initialization_options": {
        "cargo": {
          "target": "x86_64-pc-windows-msvc"
        }
      }
    }
  }
}
```

If you are using `rustup`, you can find a list of available target triples (`aarch64-apple-darwin`, `x86_64-unknown-linux-gnu`, etc) by running:

```sh
rustup target list --installed
```

## LSP tasks

Zed provides tasks using tree-sitter, but rust-analyzer has an LSP extension method for querying file-related tasks via LSP.
This is enabled by default and can be configured as

```json [settings]
"lsp": {
  "rust-analyzer": {
    "enable_lsp_tasks": true,
  }
}
```

## Manual Cargo Diagnostics fetch

By default, rust-analyzer has `checkOnSave: true` enabled, which causes every buffer save to trigger a `cargo check --workspace --all-targets` command.
If disabled with `checkOnSave: false` (see the example of the server configuration json above), it's still possible to fetch the diagnostics manually, with the `editor: run/clear/cancel flycheck` commands in Rust files to refresh cargo diagnostics; the project diagnostics editor will also refresh cargo diagnostics with {#action editor::RunFlycheck} command when the setting is enabled.

## More server configuration

<!--
TBD: Is it possible to specify RUSTFLAGS? https://github.com/imagine-engineering/made/issues/14334
-->

The Rust-analyzer [manual](https://rust-analyzer.github.io/book/) describes various features and configuration options for the rust-analyzer language server.
Rust-analyzer in Zed runs with the default parameters.

### Large projects and performance

One of the main caveats that might cause extensive resource usage on large projects, is the combination of the following features:

```
rust-analyzer.checkOnSave (default: true)
    Run the check command for diagnostics on save.
```

```
rust-analyzer.check.workspace (default: true)
    Whether --workspace should be passed to cargo check. If false, -p <package> will be passed instead.
```

```
rust-analyzer.cargo.allTargets (default: true)
    Pass --all-targets to cargo invocation
```

Which would mean that every time Zed saves, a `cargo check --workspace --all-targets` command is run, checking the entire project (workspace), lib, doc, test, bin, bench and [other targets](https://doc.rust-lang.org/cargo/reference/cargo-targets.html).

While that works fine on small projects, it does not scale well.

The alternatives would be to use [tasks](../tasks.md), as Zed already provides a `cargo check --workspace --all-targets` task and the ability to cmd/ctrl-click on the terminal output to navigate to the error, and limit or turn off the check on save feature entirely.

Check on save feature is responsible for returning part of the diagnostics based on cargo check output, so turning it off will limit rust-analyzer with its own [diagnostics](https://rust-analyzer.github.io/book/diagnostics.html).

Consider more `rust-analyzer.cargo.` and `rust-analyzer.check.` and `rust-analyzer.diagnostics.` settings from the manual for more fine-grained configuration.
Here's a snippet for Zed settings.json (the language server will restart automatically after the `lsp.rust-analyzer` section is edited and saved):

```json [settings]
{
  "lsp": {
    "rust-analyzer": {
      "initialization_options": {
        // get more cargo-less diagnostics from rust-analyzer,
        // which might include false-positives (those can be turned off by their names)
        "diagnostics": {
          "experimental": {
            "enable": true
          }
        },
        // To disable the checking entirely
        // (ignores all cargo and check settings below)
        "checkOnSave": false,
        // To check the `lib` target only.
        "cargo": {
          "allTargets": false
        },
        // Use `-p` instead of `--workspace` for cargo check
        "check": {
          "workspace": false
        }
      }
    }
  }
}
```

### Multi-project workspaces

If you want rust-analyzer to analyze multiple Rust projects in the same folder that are not listed in `[members]` in the Cargo workspace,
you can list them in `linkedProjects` in the local project settings:

```json [settings]
{
  "lsp": {
    "rust-analyzer": {
      "initialization_options": {
        "linkedProjects": ["./path/to/a/Cargo.toml", "./path/to/b/Cargo.toml"]
      }
    }
  }
}
```

### Snippets

There's a way to get custom completion items from rust-analyzer, that will transform the code according to the snippet body:

```json [settings]
{
  "lsp": {
    "rust-analyzer": {
      "initialization_options": {
        "completion": {
          "snippets": {
            "custom": {
              "Arc::new": {
                "postfix": "arc",
                "body": ["Arc::new(${receiver})"],
                "requires": "std::sync::Arc",
                "scope": "expr"
              },
              "Some": {
                "postfix": "some",
                "body": ["Some(${receiver})"],
                "scope": "expr"
              },
              "Ok": {
                "postfix": "ok",
                "body": ["Ok(${receiver})"],
                "scope": "expr"
              },
              "Rc::new": {
                "postfix": "rc",
                "body": ["Rc::new(${receiver})"],
                "requires": "std::rc::Rc",
                "scope": "expr"
              },
              "Box::pin": {
                "postfix": "boxpin",
                "body": ["Box::pin(${receiver})"],
                "requires": "std::boxed::Box",
                "scope": "expr"
              },
              "vec!": {
                "postfix": "vec",
                "body": ["vec![${receiver}]"],
                "description": "vec![]",
                "scope": "expr"
              }
            }
          }
        }
      }
    }
  }
}
```

## Debugging

Zed supports debugging Rust binaries and tests out of the box with `CodeLLDB` and `GDB`. Run {#action debugger::Start} ({#kb debugger::Start}) to launch one of these preconfigured debug tasks.

For more control, you can add debug configurations to `.zed/debug.json`. See the examples below.

- [CodeLLDB configuration documentation](https://github.com/vadimcn/codelldb/blob/master/MANUAL.md#starting-a-new-debug-session)
- [GDB configuration documentation](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Debugger-Adapter-Protocol.html)

### Build binary then debug

```json [debug]
[
  {
    "label": "Build & Debug native binary",
    "build": {
      "command": "cargo",
      "args": ["build"]
    },
    "program": "$ZED_WORKTREE_ROOT/target/debug/binary",
    // sourceLanguages is required for CodeLLDB (not GDB) when using Rust
    "sourceLanguages": ["rust"],
    "request": "launch",
    "adapter": "CodeLLDB"
  }
]
```

### Automatically locate a debug target based on build command

When you use `cargo build` or `cargo test` as the build command, Zed can infer the path to the output binary.

```json [debug]
[
  {
    "label": "Build & Debug native binary",
    "adapter": "CodeLLDB",
    "build": {
      "command": "cargo",
      "args": ["build"]
    },
    // sourceLanguages is required for CodeLLDB (not GDB) when using Rust
    "sourceLanguages": ["rust"]
  }
]
```
