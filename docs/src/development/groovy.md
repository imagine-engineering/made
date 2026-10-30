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

Explicit selection import, saved Maven selection and Gradle metadata
synchronization require the relevant provider capabilities and a trusted local
project. These actions can execute build configuration scripts and write
caches. Missing capabilities or ambiguous ownership require repair and a new
refresh. They do not grant trust, prove full synchronization or provide a
remote fallback. **Cancel selected import** applies only to this panel's
current import.

To inspect dependency relationships, select a Maven project or a resolvable
Gradle configuration and use **Collect offline graph**. This local operation
executes trusted configuration and resolves cached metadata; offline does not
mean read-only. Missing cache entries and unresolved relationships remain
explicit. **Cancel graph collection** cancels only that operation. Default
library summaries do not establish directness; separately collected graphs
retain the provider's component or occurrence identities and available edges.

Graph search and fact filters inspect admitted data without resolving it
again. Maven's scope filter preserves reported scopes; it does not reinterpret
Gradle configurations. **Neighbours** and **Shortest root paths** inspect the
selected current dependency, including equal shortest paths and explicit
coverage gaps. These views do not prove source usages or declaration origins.
**Show diagram** renders a bounded diagram from the same current facts. Image
interaction, full diagram exports and reference window modes remain gaps.
Stale data, trust revocation and changed provider ownership disable actions or
remove their derived presentation.

For an admitted current local task, phase or goal, **Run selected offline
operation** uses the existing Tasks and terminal path. **Add selected
operation** and **Run ordered offline operations** compose bounded operations
for the same current module. These controls do not provide named run
configurations, complete arbitrary-argument handling, before-launch hooks or
remote execution parity. Selected test results still belong to the Tests
view, and supported debugging uses the Debug panel.

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
same-machine PTY-enabled fixture verifies Tests cancellation with a matching
Task summary and no late report, while separate-host and disconnect cases
remain open. Java run Debug also needs the host's adapter and TCP forwarding;
its bounded loopback breakpoint evidence does not establish every topology.

The installed Java extension exposes a `Java` debug adapter. Selected-test
Debug remains unsupported in this candidate and is required by the parity
plan. Java run debugging evidence does not qualify selected-test debugging.
Its built-in run and test task scripts require `/bin/sh`; Windows needs a
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
production packaging remains blocked. CI is unavailable and its gates remain
open. VoiceOver verification is deferred at the user's request.
