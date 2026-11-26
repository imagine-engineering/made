# Groovy JVM product candidate

Groovy is the planned Java and Kotlin focused `jvm` product. It is not an
enabled release. The name does not imply support for the Groovy programming
language. The [provider evidence](./groovy-provider-matrix.md) records what is
verified and which platform and package checks remain open.

## Prepare a development host {#prepare-a-development-host}

Install a JDK 21 or newer on the machine that runs the Java language server.
The JDK used to run JDTLS can differ from the JDK used by Maven or Gradle and
from your project's target runtime. Set `JAVA_HOME` for the project host or
select a JDTLS JDK explicitly:

```json [settings]
{
  "lsp": {
    "jdtls": {
      "settings": {
        "java_home": "/absolute/path/to/jdk-21-or-newer"
      }
    }
  }
}
```

The local onboarding check reports an invalid `JAVA_HOME` when it does not
contain an executable `bin/java`, even if a different JDK is available on
`PATH`. Correct the setting and select **Check local tools again** to refresh
the result. If JDTLS starts with an older Java runtime, its language-server
status asks for a complete JDK 21+ at `lsp.jdtls.settings.java_home` on the
project host and a server restart. The local check does not certify a remote
host or an overridden JDTLS path.

The Java extension also supplies `gradle-language-server` for Gradle build
scripts. Its `lsp.gradle-language-server.settings.java_home` is a separate
host override; setting `lsp.jdtls.settings.java_home` does not change it.
Neither override selects the JDK for Maven/Gradle Tasks or the project's
compiler/test toolchains. Those remain controlled by the host environment,
wrapper, and provider-specific project configuration. Groovy does not yet
identify every effective build/test JDK automatically.

Open the folder containing `pom.xml`, `settings.gradle`, `settings.gradle.kts`,
`build.gradle`, or `build.gradle.kts`. A Maven or Gradle wrapper at that root is
needed for Groovy's detected build Task. The task runs from that folder on the
project host. Trust the worktree before running a build. Setup does not install
a system JDK or run a build script. A remote SSH, WSL, or container project
needs its tools on that host; the local onboarding check cannot certify them.
The Groovy onboarding page links to a Temurin JDK setup guide and shows Maven
and Gradle guides when those local tools are missing. **Check local tools
again** refreshes the local probe after you change your environment; it does
not install anything.

## Java and Kotlin providers {#java-and-kotlin-providers}

The Groovy catalog selects the [`java`](https://github.com/zed-extensions/java)
and [`kotlin`](https://github.com/zed-extensions/kotlin) extensions as defaults.
The Java extension supplies Eclipse JDTLS. The Kotlin extension offers
JetBrains' official `kotlin-lsp` by default and
`kotlin-language-server` as an explicit alternate. An extension or server
download requires network access unless it is already cached. No server binary
is bundled or certified for release by this development candidate.

The approved macOS ARM64 Kotlin LSP build for qualification is `271.2085.0`.
If an older installed EAP build has expired, Kotlin support cannot start.
In **Language Servers**,
choose **Check for Kotlin LSP Update** to compare the installed version with
the latest official standalone build for the project host. A newer compatible
archive provides an **Open Kotlin LSP release** action. The check reads release
metadata. You choose whether to download and install a server.

For SSH, WSL, and container projects, the check is available while connected
directly to the project host. Install the standalone server there, set or
update `lsp.kotlin-lsp.binary.path` to that executable, and restart the
language server. The check preserves your current provider and binary override.

For a standalone qualification run, obtain the macOS ARM64 archive from the
official release. `script/test-groovy-kotlin-provider.py --approved-archive
<archive> --fixture maven --java-home <jdk>` checks the approved SHA-256,
extracts a disposable copy, and verifies the language server's reported build.
Use `--fixture gradle` for the Gradle run. The pin applies to the qualification
probe. The installed extension still uses its own upstream resolver.
[Official Kotlin LSP 271.2085.0 release](https://github.com/Kotlin/kotlin-lsp/releases/tag/kotlin-lsp/v271.2085.0)

To choose the alternate Kotlin server, use the Settings Editor or add this to
your project or user settings:

```json [settings]
{
  "languages": {
    "Kotlin": {
      "language_servers": ["kotlin-language-server", "!kotlin-lsp", "..."]
    }
  }
}
```

You can also point either server to a binary you installed yourself. See the
[Java](../languages/java.md) and [Kotlin](../languages/kotlin.md) settings
guides. A project setting can override your user default. Keep only one Kotlin
server enabled for a buffer.

For offline Maven projects using the alternate server, make its host
`MAVEN_REPOSITORY` and Maven's `maven.repo.local` refer to the same populated
repository. Cache the Maven Dependency Plugin used by `dependency:list` and
install any required reactor artifacts before going offline. The alternate
resolves dependencies for each module POM separately.

If the host extension policy denies a server setup download, review extension
permissions on the project host. After changing `granted_extension_capabilities`,
restart the application or remote server on that host before retrying. The
extension host reads these grants at startup. A language-server restart alone
does not refresh them. For remote projects, restart the server on the
authoritative host.

You can also configure a compatible installed binary and restart the language
server. Preferred Kotlin uses `lsp.kotlin-lsp.binary.path`; the alternate uses
`lsp.kotlin-language-server.binary.path`. Choosing the alternate requires its
own download permission or installed binary. The Java extension's Gradle
build-script server uses the same host permission recovery rule or a compatible
installed-command override. This is separate from JDTLS and your Tasks
build/test toolchain. Groovy keeps your server selection until you change it.
Source and host-delegate checks cover the recovery status; current
installed-product presentation and successful recovery remain unverified.

If cached Java setup files still fail because the host extension policy denies
its `java -version` probe, review extension process permissions on the project
host and restart that host's application or remote server after changing the
grant. A server cache does not grant process permission. Installed-WASM checks
cover cached command recovery through a fresh host; native error presentation
and successful product permission recovery remain unverified.

Groovy enables JDTLS's Gradle Kotlin JVM support option by default for Java
navigation to Kotlin source. You can turn it off in user or project settings:

```json [settings]
{
  "lsp": {
    "jdtls": {
      "initialization_options": {
        "settings": {
          "java": { "jdt": { "ls": { "kotlinSupport": { "enabled": false } } } }
        }
      }
    }
  }
}
```

A cached standalone Gradle fixture and a later installed-extension Groovy
development run on macOS ARM64 resolved Java to Kotlin with this option. The
corresponding Maven fixture still fails after JDTLS import. Packaged Groovy and
other platforms remain unverified, so this default is not a release claim for
mixed-language navigation.

The alternate server is a selectable recovery path, not a verified replacement
for every project. Its pinned `1.3.13` candidate failed the checked-in
multi-module Maven fixture even when Maven was available: it reported no
artifacts and an internal Kotlin analysis error. A standalone Gradle fixture
did return navigation, completion, and diagnostics, but its installed-extension
and other-platform paths remain unverified. Groovy shows bounded
language-server health messages for the observed Maven failures and a Gradle
standard-library resolution failure. For the latter, restore the host Gradle
cache or allow dependency downloads, then restart the selected server. See the
[provider evidence](./groovy-provider-matrix.md) before relying on either
server for a release target.

If JDTLS reports a missing Maven artifact in offline mode, populate the Maven
cache on the project host or turn off Maven offline mode, then reimport. The
shared language-server status shows a bounded recovery message and keeps the
provider log available for the specific artifact.

If JDTLS reports that Gradle cannot resolve a build plugin, check the plugin
repositories and cache on the project host, then reimport. The shared status
shows a bounded action while the full provider log remains available. This
message does not establish that Gradle import ran offline.

If a default extension install fails, open Extensions to read its reason and
select **Retry install** after resolving the problem. A removed or disabled
default stays removed or disabled until you change that choice. A missing JDK,
denied network request, incomplete offline cache, or server startup failure
must be resolved on the project host. Editing can continue while setup is
blocked, but a provider-backed import or navigation result should not be
assumed current.

## Dependencies panel {#dependencies-panel}

Use **Maximize Groovy panel** to give the Dependencies, Gradle, or Maven
view the workspace's available space. **Restore Groovy panel size** returns
to the previous dock size and preserves the current tree selection. The
workspace's **Toggle Zoom** action also works while the panel has focus.
The panel is pinned by default. **Unpin Groovy panel** hides it when you
focus the editor or another tool window. Reopen it using the normal
Groovy view action, or choose **Pin Groovy panel** to keep it visible.
The workspace remembers the pin setting.

Right-click the Groovy status bar icon and choose **Dock Top** to put the
panel above the editor. Drag the panel's lower edge to change its height.
The workspace remembers the dock edge.

Choose **Undock** to open Groovy over the workspace at its current dock edge
and size. The editor keeps its layout while the overlay is open. It hides
when focus moves to another work area. **Dock** returns it to unpinned docked
mode, and **Pin Groovy panel** keeps it docked and visible.

Choose **Float** to keep Groovy above its project window. Float uses the
project window’s display to choose its initial bounds. Choose **Open in
Window** to use Groovy in an independent window. The normal Groovy view
action reopens the panel in its saved mode. Choose **Return to Dock** to pin
the panel at its saved edge.

The `jvm-tools` development build has a Dependencies panel for trusted
Maven and Gradle projects. Open it from **View → Dependencies**. It shows
provider-labeled rows from both JDTLS and Kotlin LSP when both snapshots are
current; one provider's library facts no longer hide the other's. If only one
provider is current, the panel omits the other's stale rows and reports partial
coverage. Kotlin LSP can report module references, named Maven or Gradle
artifacts, and classpath scopes. JDTLS reports module references and external
library counts. For Maven entries with provider-supplied group, artifact,
version, and scope attributes, it also shows validated coordinates and scope;
Gradle entries can show validated source-set use such as `main, test` beside a
safe `.jar` name. Other entries show a safe file name or an unnamed count.
JDTLS Gradle source-set use does not establish an artifact coordinate or a
Maven scope. Neither provider establishes whether an artifact is a direct or
transitive declaration. Unknown facts are labeled as such. JDTLS module rows also distinguish Java
source compatibility, class-file target compatibility, and the provider's
project VM version. A missing target is explicitly unreported; it is never
inferred from source level or VM version. **JDTLS launch Java** separately
shows the major from the host installation selected by the current managed
Java proxy command (or a recognized direct Java command). A configured JDTLS
launcher, unrecognized override, or unreadable installation metadata leaves
that fact unknown. This is command metadata, not an independent running-JVM
observation or proof of a compiler. The module row explicitly labels build JDK
and test runtime as unknown; neither is inferred from these provider facts.
Build/test JDK identity and an installed target runtime remain unverified.
Only numeric Java facts cross the remote connection; host toolchain paths are
not dependency labels. Ordinary panel rendering and **Refresh Dependencies**
do not launch a Java probe or configure a build. Explicit import,
synchronization and graph collection controls can execute trusted build
configuration scripts; see
[the development views](#gradle-and-maven-development-views).
**Refresh Dependencies** requests a new provider snapshot; the Kotlin adapter
uses a temporary host directory and removes its workspace export after reading
it.

Opening a module uses its visible `pom.xml`, `build.gradle.kts`, or
`build.gradle` after checking the current provider generation and worktree
permissions. Current public Gradle source-directory rows can reveal their
folder in the Project Panel. Declared annotation-processor output is
distinguished from unknown or absent output; this does not identify every
generator or prove generated files exist. Missing, private or stale paths
remain unavailable. Library and dependency graph rows do not open or download
external artifact files. A remote project
queries JDTLS or Kotlin LSP on its host and receives only bounded,
project-relative paths and validated dependency facts. Connected fake-host
tests cover both providers, trust revocation, stale replies, and disconnect;
an earlier installed macOS ARM64 Groovy development run displayed Kotlin
provider modules and artifacts with Partial status. Its installed JDTLS
process did not populate Java project rows in that run. A later development
binary did display JDTLS `application` and `library` modules, a project
reference, and ten external libraries with Maven coordinates and scopes under
Partial status. A later same-machine macOS ARM64 SSH-loopback development
journey displayed both installed providers' module rows and recovered after a
host JDTLS override was repaired. This is bounded loopback evidence; a separate
host, complete dependency coverage, packages, and screen-reader behavior remain
open. An empty or stale panel is not evidence that dependency resolution
succeeded. A late-starting provider now refreshes the active panel without a
manual Refresh; an import failure leaves only current provider rows visible
under incomplete coverage.

## Gradle and Maven development views {#gradle-and-maven-development-views}

The development build also has Gradle and Maven views in the shared panel.
Retained software checks are recorded in the
[provider evidence](./groovy-provider-matrix.md). Installed-product, physical
remote and platform qualification remain separate from those checks.

Use the provider-specific project and module tree to inspect admitted Gradle
tasks and configurations or Maven lifecycle phases and plugin goals. Unknown
or incomplete provider facts stay labelled; a directory name alone does not
establish module membership. **Options** changes the view's presentation.
Ordinary refresh requests current metadata; it does not certify a complete
reimport.

In the Maven view, expand **Profiles → Definitions** and press Enter on a
definition with **Open profile POM** available. The editor opens the visible
POM that declares the profile, including a parent POM when the provider reports
it. Definitions from the cache can still show their source coordinates when
the POM is unavailable to the project.

The Maven project row menu also offers **Open Maven user settings** and
**Open Maven global settings** for configured files in a visible, trusted
worktree of the local workspace. The commands use the same JDTLS settings paths
as Maven Tasks.
If a configured file is missing and its parent directory exists, the menu
offers **Create Maven user settings** or **Create Maven global settings**.
The command creates a Maven settings template at that path and opens it in the
editor. An existing file is preserved if it appears before creation.
For a settings file outside the build workspace, configure its absolute path
in your user settings under `lsp.jdtls.settings`. When the effective path
matches that user setting, **Open or Create Maven user settings** or
**Open or Create Maven global settings** opens the file in a separate local
workspace.

Use a POM or `settings.xml` to define profiles for Maven 3.9.16. This version
ignores standalone `profiles.xml` files beside a POM.
[Maven 3.9.16 project builder](https://maven.apache.org/ref/3.9.16/maven-core/xref/org/apache/maven/project/DefaultProjectBuilder.html#L698)

Build import and Gradle metadata synchronization require the relevant provider
capabilities and a trusted project. These actions can execute build configuration
scripts and write caches. **Cancel selected import** applies to the current
local import started from this panel.

On a trusted direct connection, **Synchronize metadata** reports when cached
Gradle topology is unavailable on the project host. Use **Reimport Gradle
projects** to apply the host's effective Gradle selection in offline mode, then
use **Synchronize metadata** again to load tasks and configurations. Both
actions rely on the project host's cached Gradle installation and dependencies.

For **Reload Maven projects** and **Apply saved Maven selection**, set
`java.import.maven.offline.enabled` to `true` under the project host's
`lsp.jdtls.settings` and use a populated Maven cache. This configures JDTLS's
Maven importer. The Maven wrapper's `-o` flag configures command-line builds.

The scoped Gradle provider uses a cached Gradle installation for **Synchronize
metadata**. In your host's `lsp.jdtls.settings`, set
`java.import.gradle.home` to the absolute installation directory, set
`java.import.gradle.wrapper.enabled` to `false`, and set
`java.import.gradle.offline.enabled` to `true`. Clear any explicit
`java.import.gradle.version` so JDTLS uses that installation. Use an
installation compatible with the build and a Gradle user home with the
dependencies already cached. Refresh the imported project after changing
these settings.

To inspect dependency relationships, select a Maven project or a resolvable
Gradle configuration and use **Collect offline graph**. This local operation
executes trusted configuration and resolves cached metadata; offline does not
mean read-only. Missing cache entries and unresolved relationships remain
explicit. **Cancel graph collection** cancels only that operation. Default
library summaries do not establish directness; separately collected graphs
retain the provider's component or occurrence identities and available edges.

Graph search and fact filters read admitted data without resolving it again.
**Tree groups** switches the fact list between grouped and flat rows. Maven's
scope filter uses the reported edge scopes. Gradle diagram scopes come from
resolvable configurations reported by the provider.

Use Tab and Shift-Tab to move between the tree and controls in the build-tool
panel and Dependency Analyzer. Use the arrow keys to navigate tree rows.

### Dependency diagram and analyzer {#dependency-diagram-and-analyzer}

Open **Show Dependency Diagram** from the module or configuration menu. Use
the diagram toolbar to fit, zoom, change scope, search for and reveal a
dependency, or open the Structure overview. Drag blank space to pan. Use
Command-click or Control-click to toggle nodes in the selection. Shift-drag
from blank space to extend the selection with a rectangle. Use **Layout** in
the blank-canvas menu to switch between left-to-right and top-to-bottom ranks.
Use Tab and Shift-Tab to move between diagram controls, the canvas, and nodes.
On the focused canvas, use {#kb groovy_dependencies_panel::ShowDependencyDiagramContextMenu}
for view and layout commands while keeping the current selection. Use the
same shortcut on a focused node to open its dependency menu. The canvas pans
to show the focused node when it is clipped. The details overlay shows the
focused dependency's coordinates and version.
In the Structure overview, tab to a selected dependency and press Enter or
Space to center the view on it while keeping the current selection and zoom.
Use {#kb groovy_dependencies_panel::ShowDependencyDiagramContextMenu} to open
the menu for the focused overview dependency. Secondary-click any dependency
block to open its menu while keeping the current selection. Secondary-click
empty overview space for view and layout commands.
Search includes dependencies beyond the initial canvas view, including
Gradle requests that failed to resolve. Choosing a result brings its immediate
connections into the bounded view even when the dependency is already drawn.
Search ranks exact dependency coordinates ahead of partial matches. In the
analyzer you can paste a full coordinate even when group IDs are hidden.
The temporary diagram preview offers the same search. Escape closes the
search first, then closes the preview and returns to the workspace.
On a larger graph, **Copy Diagram Source** copies the current bounded canvas
as Mermaid source and identifies the limited coverage in the copied text.

| Earlier panel control                                                                             | Where to find it                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cancel selected import                                                                            | Use **Cancel selected import** on the active import row.                                                                                                                                                                                      |
| Graph facts: Tree groups                                                                          | Switch between grouped and flat fact rows in the Dependencies panel.                                                                                                                                                                          |
| Show diagram, Close diagram                                                                       | Use **Show Dependency Diagram** from the module or configuration menu and close the tab with the editor's close action. The same menu offers **Show Dependency Diagram Popup** and **Analyze Dependencies**. Use Escape to dismiss the popup. |
| Neighbours, Shortest root paths                                                                   | Use the diagram toolbar or dependency menu. The fact tree also offers both actions when a dependency row is selected.                                                                                                                         |
| All root paths, Conflict origins                                                                  | Use **More inspections** in the dependency menu. **Inspect conflict origins** appears when the provider records mediation. A bounded path result reports truncation.                                                                          |
| All usages                                                                                        | **All usages** in the diagram opens the analyzer on the same dependency and scope. **Usages in the tree** shows reverse relationships in the Dependencies panel.                                                                              |
| Pin dependency, Paths to pinned                                                                   | Use **Pin dependency** and **Inspect paths between** in the dependency menu's **Compare** submenu. **Inspect paths between** follows paths from the pin to the selected dependency. Pin the same dependency again to clear its pin.           |
| Add to set, Set relations, Clear set                                                              | Use **Add to selection set**, **Inspect selection set**, and **Clear selection set** in **Compare**. The selection set supports comparison across several dependencies.                                                                       |
| Clear inspection                                                                                  | Use **Clear inspection** to remove the active result and keep the pin and selection set.                                                                                                                                                      |
| Copy diagram source                                                                               | **Copy Diagram Source** copies Mermaid. **Copy Diagram Image** copies SVG. **Export Image to File** saves a new SVG.                                                                                                                          |
| Add selected operation                                                                            | Use **Add to Queue** from an eligible operation's menu.                                                                                                                                                                                       |
| Run configuration, Remove last operation, Clear ordered selection, Run ordered offline operations | Expand **Ordered operations** for **Remove last**, **Clear**, and **Run offline**. The queue row's menu offers saved configurations when available.                                                                                           |
| Expand, collapse, and refresh icons                                                               | Use **Expand All**, **Collapse All**, and **Refresh Dependencies** in the Dependencies tree toolbar.                                                                                                                                          |

Open the diagram context menu with {#kb groovy_dependencies_panel::ShowDependencyDiagramContextMenu}.
In the analyzer and Dependencies tree, use {#kb groovy_dependencies_panel::ShowDependencyRowMenu}
on the selected tree row or focused usage. Tab and Shift-Tab scroll the
focused usage into view. If that usage is replaced, keyboard focus returns
to the analyzer tree. The analyzer supports searchable tree and flat lists,
scope and conflict filters, group IDs, and a Usages pane. **Show in Dependency
Diagram** reveals the selected occurrence in the diagram. Declaration navigation
requires a current provider-reported location. Stale data, trust revocation,
and changed provider ownership invalidate the derived presentation.

**All Usages** shows the exact linked dependency in the Usages pane while
keeping the current list filter. If the dependency is outside the list,
the pane offers **Show in Diagram** to return to it.
For large Maven graphs, search covers the entire admitted scope under the
current filter, including occurrences beyond the displayed rows. Choosing
one of those results opens its Usages pane.

### Ordered offline operations {#ordered-offline-operations}

For a current task, phase or goal, **Run selected offline operation**
uses the existing Tasks and terminal path. On an eligible operation, use
**Add to Queue**. Open **Ordered operations** in Dependencies, or use the
ordered-operations control in the diagram or analyzer toolbar. Expand the
section to review the order, remove the last operation, clear the queue, or
run offline. The queue stays available when you switch between Dependencies,
Maven, and Gradle. Groovy clears it when its project data becomes stale.
Saved run configurations appear in the queue row's context menu when
available. Cancel a running queue from its Task terminal.

On a direct remote connection, refresh JVM metadata with the matching
Groovy-enabled host before selecting operations. In the Gradle view, use
**Synchronize metadata** to populate tasks on a trusted project host. This
explicit action executes the host's Gradle build configuration. The host prepares the offline
command using its own build tools and settings, and checks the plan again before
the terminal starts. If the connection, selected module, or provider metadata
changes, refresh and select the operation again.

For Gradle, put invocation-wide options in `java.import.gradle.arguments`
under `lsp.jdtls.settings`. For example, use
`["--no-parallel", "--max-workers=2"]` to disable parallel project execution
and limit concurrent workers. The setting also accepts `["--max-workers", "2"]`
and literal `-Pname=value` project properties. Gradle uses the last of
`--parallel` and `--no-parallel` if both appear. Set one positive worker
count. Discovered Tasks remain offline. Use **Task options** for arguments
declared by the selected task.

**Saved configurations** lists project and user Tasks with their source
locations. The selected row, quick-run button, and queue menu all target the
same saved entry. When labels match, choose by source location. Removing a
configuration invalidates its old quick-run and queue targets.

For a Maven phase or goal, open **Task options** and enter a JSON array of the
options you want. Use `-Pci` to activate a declared profile, `-P!legacy` to
deactivate one, or `-DskipTests` to skip test execution. Use
`--non-recursive` to run against the selected POM without its submodules.
Choose `--fail-fast` to stop on a reactor failure or `--fail-at-end` to finish
unaffected modules and report failure at the end. Set a fixed thread count
with `--threads=2` or a multiplier with `--threads=0.5C`. Maven calculates the
multiplier against the host JVM's available processors.

These options apply to the entire ordered Maven run. Conflicting options in
queued entries prevent dispatch. Maven still applies its own precedence for
`.mvn/maven.config`. For example, a configured fail-fast flag takes
precedence over fail-at-end.

To inspect an effective POM, expand **Plugins** under the Maven project and
choose the `effective-pom` goal under `maven-help-plugin` when the provider
reports its cached descriptor. Use the selected operation’s profile options
if you need to change activation, then choose **Run selected offline
operation**. Maven writes the effective XML to the Task terminal using the
host Maven settings and the selected profiles.

When the provider reports goals for a declared `maven-dependency-plugin`,
run `resolve-sources` to check cached source archives or `go-offline` to
check repository readiness for the project’s dependencies, plugins, and
reports. To check Javadoc archives, select `resolve` and use **Task options**
to activate a project profile that sets the `javadoc` classifier and `jar`
type. Review the resolved and unresolved artifacts in the Task terminal. Maven can
exit successfully when a source archive is absent.

Complete arbitrary-argument handling, before-launch hooks, and remote queue
execution remain open. Selected test results appear in the Tests view.
Supported debugging uses the Debug panel.

The complete required scope and remaining gaps are in the
[tool-window parity inventory](./groovy-tool-window-parity.md#capability-matrix).
These bounded development capabilities do not establish full IntelliJ IDEA
Gradle/Maven parity or a release-ready Groovy package.

## Build, tests, and Debug scope {#build-tests-and-debug-scope}

The current candidate adds detected Maven `package` and Gradle `build` Tasks
when the matching wrapper is present. The terminal shows process output and
errors. Cached Maven and Gradle builds passed through the product terminal on
macOS ARM64. Selected JUnit Jupiter results in the Tests view have local
fixture evidence for cached Maven and Gradle passes, plus a Maven fail and
rerun. An installed-extension Groovy development run on macOS ARM64 found and
ran a selected JUnit method in the shared Tests view, displaying one fresh
pass. The same run launched a supported Java main class through the installed
Java Debug adapter and shared Debug panel. It stopped at entry and at an
editor breakpoint, showed the current source frame, and terminated. This
required a test-only JDTLS JDK override; Groovy did not provision that JDK.
In a later development binary, Refresh Tests followed by Run Selected Tests
again produced a fresh passed Maven JUnit result in the shared Tests view.
That binary also opened a trusted, cached two-module Gradle project: Go to
Definition from Java `Greeting` opened its Kotlin source, and a selected Java
JUnit method passed through the shared Gradle Task terminal and Tests view.
A later CLI GPUI fixture joins the real Java/Kotlin providers, Editor menus,
Dependencies navigation, build Task, and selected-test failure/rerun/recovery
in one cached Gradle Project. It also verifies two real Java main launches:
toolbar Continue ends the first process; restricting trust while the second
is stopped at an Editor breakpoint terminates that session and actual
process and clears its active source marker. Both normal and
external-network-denied variants pass on macOS ARM64 with explicit cached
providers and JDKs. These fixture checks do not verify installed defaults or
a physical desktop journey.

Packaged-app, other-platform, remote, and spoken-accessibility validation
remain release gates. A listed Task alone does not prove that its host tools
or dependencies are available.

Java discovery currently verifies method selectors against saved source and
existing JUnit reports. A newly added test may need a build/test Task to write
its report before **Refresh Tests** can discover it. Cached report names do
not certify a current-session pass; a selected run needs both terminal Task
completion and a fresh matching JUnit report. Save edited test source before
refreshing. Display-only, parameterized, ambiguous, or unsupported dynamic
cases do not gain a runnable selector merely from their display names.

For SSH actions, trust must be available on the project host at authorization
and lifecycle updates. If the host reports that trust is unavailable, reconnect
and refresh Tests; restoring host trust does not revive an expired action
grant. A denied remote terminal stops the Task before its wrapper runs. The
connection must stay current while the Task prepares its environment. If it
changes during preparation, run the Task again once connected. The
same-machine PTY-enabled fixture verifies Tests cancellation with a matching
Task summary and no late report, while separate-host and disconnect cases
remain open. Java run Debug also needs the host's adapter and TCP forwarding;
its bounded loopback breakpoint evidence does not establish every topology.

The installed Java extension exposes a `Java` debug adapter. In a trusted
local project, select a verified Java case in Tests and choose
**Debug Selected Test**. A freshly verified source selector is enough to start
Debug while Tests still shows the previous JUnit result. The debugger waits for
the selected test JVM to suspend, then attaches through the Java adapter.
Gradle uses its test Debug
port 5005. Maven uses a separate loopback port for each launch. Stopping the
Debug session or restricting project trust cancels its selected test Task,
including during terminal preparation.

The debugger supplies the selected module's Java test source directory to the
adapter. When the pinned adapter fails a whole stack request on a frame
without source, the debugger retries in bounded pages and logs the number of
frames it recovered.

Selected-test breakpoint journeys in the installed desktop product and on
separate hosts remain qualification gates. The extension's built-in run and
test task scripts require `/bin/sh`; Windows needs a
separately validated route. Structured Kotlin tests, mixed-language refactors,
coroutine debugging, HotSwap, Spring/Jakarta/JPA depth, and database tooling
are separate product decisions.

## Optional provider resource qualification {#optional-provider-resource-qualification}

The standalone Kotlin probe accepts `--sample-process-resources` on macOS
and Linux. It samples numeric PID, parent, group and RSS data for its own
server group/current descendants. In `--cached-denied-network` or
`--expect-gradle-denied-network` mode, its fresh isolated Gradle registry
also identifies detached import daemons. The probe requires those daemon
PIDs to disappear after `--stop`; it does not stop shared host daemons.
Run `python3 script/test-groovy-kotlin-resources.py` to check this ownership
boundary and explicit measurement failures with disposable processes.

Reported maxima are sampled sums of RSS, which can count shared pages
more than once and miss peaks between samples or untracked detached workers.
The combined peak unions owned PID sets at each sample; independent
server/daemon peaks may overlap and must not be added. Supply `--resource-editor-pid <pid>` alongside resource sampling to include an
explicitly identified editor and its current descendants; an unavailable editor
fails the sample. Without it, the editor is excluded. The caller must bind that
PID to its exact candidate/profile and account for every owned worker. PID reuse
and detached untracked workers remain limits. These measurements do not establish
whole-project physical memory or Kotlin alpha acceptance. Linux execution and the remaining
platforms still need target-specific evidence.

Groovy remains planned until the [provider and release evidence](./groovy-provider-matrix.md)
covers the claimed Java and Kotlin workflow, platform packages, remote hosts,
and accessibility.

## Accepted first-release qualification scope {#accepted-first-release-qualification-scope}

Provider delivery uses extension-managed downloads. Governing download terms,
required notices, exact approved provider pins, permission/offline/failure behavior
and tested platform cells remain required; this is no bundled-server distribution
claim. Kotlin qualification uses a macOS ARM64 reference with 32 GiB RAM and an
8 GiB combined resident-memory ceiling. This is an accepted target, not a measured
pass. Complete preferred/fallback Maven and Gradle journeys must include the editor
and every owned JVM/build worker in the resource evidence.

Agent structured-result sharing is unadvertised in the first release; actual
agent acceptance remains open. Production identity remains provisional and
production packaging remains blocked. VoiceOver verification is deferred at the
user's request.

The November 20, 2026 dev test run passed all 16 jobs at
`b6c7f505ad008efaba12a8ca8eee2aa5c3744ca3`, including `windows_jvm_validation`.
The workflow runs on pull requests, pushes to `main`, and manual dispatch.
A direct push to `dev` leaves current-source CI qualification open until a run
covers that revision. [GitHub Actions run 44824772699](https://github.com/imagine-engineering/made/actions/runs/44824772699)
