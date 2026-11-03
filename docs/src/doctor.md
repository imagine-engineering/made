# Made Doctor

Open **Doctor** from the command palette, or enter `/doctor` in the native agent. Doctor shows the project, selected worktree and path, execution host, language check family and last collection time. A sole open worktree is selected automatically; choose explicitly when several are open. **Change execution host…** opens the existing remote-project controls. Doctor uses the selected project's host and never substitutes your local computer for an unavailable remote host.

Choose a section in the persistent left navigation. It uses the same navigation controls as Settings; the selected section scrolls independently. Switching sections returns their content to the top and preserves collected evidence. Choose the section that matches your question:

| Section              | What it examines                                                                                                   | Next action                                                                                                                                        |
| -------------------- | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Configuration health | Copper/Made's configured model, extensions, context servers, project authority and automation owners               | **Collect checks**, then inspect problems and their next steps. A configuration observation does not prove a model response or network connection. |
| Development health   | Your project's toolchain declaration, cached Cargo/JVM metadata, language servers and recorded build/test evidence | **Collect checks** to inspect existing evidence. Set up a separately reviewed investigation to run a relevant build or test.                       |
| Application behavior | Your application's correctness, crashes or performance                                                             | Select a symptom, validation target and reproduction in the investigation form. Review the exact execution and effects before approving it.        |

**Collect checks** reads configuration and cached development evidence; it does not run a build or test, fetch dependencies, execute the application, install tools or apply repairs. While collecting, use **Cancel collection** to stop Doctor waiting. A failed or cancelled collection offers **Retry checks**. Disabled collection controls explain the missing selection or active operation.

Copper exposes Rust development checks. Groovy exposes JVM development checks when its existing JVM feature is enabled; this does not establish Groovy release readiness. Missing integrations or host capabilities remain explicitly unsupported. Generic checks and local diagnosis remain available without an agent or GitHub account.

Opening Doctor reveals and focuses its editor tab, including when the agent panel was zoomed. Invoking it again reuses that project's Doctor interface and keeps its selections. In the native agent, `/doctor` and its opening result become conversation context. Subsequent collection updates add a bounded, redacted, problem-first excerpt to that originating conversation. The latest collection identifies itself as replacing earlier snapshots and includes zero counts so resolved blocks remain explicit. Separately approved operations also report their reviewed identity, progress and observed outcome; an earlier command exit cannot describe a new running action. The full report remains in Doctor. Repeated adjacent statuses are deduplicated and normal conversation restoration retains them. Updating context sends no model request. A live follow-up still needs an available configured model.

Results show problems first, followed by unverified and successful observations. Each includes its evidence, owner/source, observation time, limitations and next step. **Passed** means only the stated observation passed. It does not establish an unperformed build, test, provider response or application check. **Not run**, **Running**, **Warning**, **Failed**, **Blocked**, **Unsupported**, **Cancelled** and **Unverified** remain distinct. **Stale** identifies retained evidence or inputs changed since collection; collect again before relying on it.

Redacted exports retain observation times and label stale inputs. If freshness changes while choosing an export destination, Doctor asks you to review the report again instead of exporting a current-looking result.

For a local llama.cpp model, provider model discovery is distinct from a successful inference request. If unavailable, inspect loading and failure state in **Local models** within the llama.cpp provider settings. Doctor does not infer that a local model needs an API key or silently switch to cloud inference.

**Advanced actions** in the left navigation opens check-family selection, history, provider and dependency probes, security evidence, installations, restarts and native repair proposals. Opening these forms executes nothing. Each actual operation retains its own review and approval. Keep your application's source and settings unchanged while collecting evidence; test intentional failures and repairs in a disposable project.

## Diagnose and repair

1. Collect product checks. For an investigation, choose product health, development health, or application correctness/performance, and specify the symptom and relevant validation target.
2. Inspect timestamped checks and evidence. Observations, hypotheses, historical results, blocked prerequisites and unknown coverage are distinct.
3. Review a concrete action. The preview identifies the host, owner operation or exact command, expected writes/subprocesses, downloads, network, credentials, permissions and interruption limits.
4. Explicitly approve that action. Changed authority invalidates the proposal. Progress and cancellation refer to the existing owner operation; cancellation may leave reached development changes.
5. Inspect the relevant recheck. Success of a command or an agent assertion does not verify a fix. Application verification requires the reproduced failure/baseline, selected change and relevant functional tests or comparable measurements.

Product checks include project trust, provider/model availability, MCP and plugin status, recipes/schedules and Docker prerequisites through their current owners. Passive checks do not authenticate, switch providers, execute recipes, run containers or enable unattended schedules.

Rust checks include effective toolchain, declared components/targets, Cargo workspace evidence, rust-analyzer and reached build/test/debug failures. JVM checks distinguish the language-server JDK, build/test JDK and target compatibility, JAVA_HOME, Java/Kotlin providers, wrappers, project import and observed test/debug outcomes. Observed metadata is not proof of application correctness.

Installation uses only supported, already installed user-scoped rustup, mise Java or SDKMAN managers with exact reviewed packages, sources and destinations. Unknown manager versions/configuration remain blocked. SDKMAN official hook execution is disclosed, and first-default creation needs a separate approval. Doctor does not bootstrap managers, elevate privileges, change project pins or silently change defaults/JAVA_HOME. Application restart uses the normal save/shutdown flow; owned remote-server restart requires exclusive idle authority. OS, Docker-engine and arbitrary service restarts are excluded.

## Investigate a developed application

Select an existing Task, test or supported debugger configuration and disclose development effects. Doctor may execute project code only after review; this is not a sandbox or authorization for production, deployments, migrations, destructive operations, load testing or external services. Existing project trust, tool permissions and host authority still apply.

Collection defaults to 60 seconds, permits a reviewed maximum of five minutes and limits collected evidence to 10 MiB. Cancellation and cleanup use the current operation owners. Only selected bounded diagnostics and aggregate counters are collected. Heap/core dumps, environment/request payloads and sensitive runtime data are excluded. Debug collection excludes variables, memory, evaluation, raw output, child/reverse-terminal launches and remote runtime attachment.

Profiling uses the existing project and profile-results surfaces where supported. Selected confined local macOS Samply 0.13.1 profile/symbol artifacts can be imported; import is not a new capture or a comparable CPU/memory baseline. Unsupported JVM JFR, remote artifacts/attachment or unavailable collectors stay incomplete. Comparable performance verification needs the same host, workload, configuration and cache conditions; a smaller sample count alone does not establish an improvement.

Source, configuration and build-file repair proposals use NativeAgent, Thread/AcpThread, ThreadStore and normal change review. Selecting investigation observations opens their Advanced repair controls and collapses the reproduction form without discarding its inputs or results. Restricted repair proposals require a provider that implements reviewed endpoint confinement. The current llama.cpp provider does not implement it: ordinary local conversations work separately, but Doctor blocks that repair handoff and never substitutes a cloud provider. A separately reviewed editor change and Doctor verification remain available. Proposal sessions expose source inspection/edit tools with once-only approval. Their slash-command and skill suggestions are suppressed, and command dispatch cannot expand the proposal into terminal, MCP, network, child-agent or recipe execution. Doctor never uses a proposal to grant tool permissions or scheduling consent.

## Dependencies and security evidence

Reviewed dependency collection uses existing Cargo workspace and Maven/Gradle owners. Declared constraints and resolved versions, direct/transitive/workspace/multiple-version instances and introducing paths remain distinct. Unsupported resolution, incomplete provenance and unknown repositories remain explicit.

RustSec and OSV's Maven advisory data are evaluated locally against the observed resolved graph. Findings show the source, identifier/aliases, affected range, available severity/fix versions, dependency path, freshness and limitations. An affected dependency is not proof that the application is exploitable. Withdrawn notices, unsupported semantics and incomplete scans are labelled; no findings is never proof of security.

Dataset refresh is manual and requires an exact download/effects preview. Data older than 24 hours is stale. The deletable offline host cache expires after 30 days and is capped at 256 MiB. Failed refreshes preserve the previous valid dataset without making it fresh. This delivery supports advisory-cache operations only for a local project; a selected remote host reports unsupported cache access rather than using a local substitute.

Optional version lookup sends only the explicitly selected and confirmed public crate name or Maven coordinates to crates.io or Maven Central after request review. Private/path/Git/workspace dependencies remain blocked. Unknown-origin Maven catalog lookup requires separate public-coordinate confirmation and returns catalog candidates, preserving unknown artifact origin. Diagnosis never fetches dependencies, rewrites lockfiles, changes repositories or installs audit tools.

Select a concrete compatible, breaking or replacement candidate for a NativeAgent repair proposal. Major/replacement choices require explicit selection and normal review. Verification requires fresh advisory matching against the changed resolved graph and relevant reviewed build/tests; an update command or unrelated tests cannot prove a security fix.

## Optional GitHub evidence and setup

Explicitly select an existing `GITHUB_TOKEN` credential, expected account, GitHub.com repository and sources, then review the exact read-only requests. Existing Dependabot alerts, bot update PR candidates and code-scanning alerts/analyses retain source, timestamps, scanner/rule/version, reference/commit, coverage and relevance to the current checkout. Package/advisory or rule/location correlation preserves all occurrences and does not invent resolved versions, analysis coverage or exploitability.

Denied/disabled/unavailable services, partial pagination, stale/future timestamps, branch/commit mismatch and API failures remain distinguishable from complete zero-finding results. Existing PR links open for review; Doctor never merges or dismisses findings. Local/offline diagnosis does not depend on GitHub. Enterprise substitution, new OAuth/grants, SARIF/source download, secret-scanning values, dependency uploads and workflow triggers are excluded.

Security setup is a separate unpublished NativeAgent proposal for an exact repository-root Dependabot file or an existing scanning workflow. Review instructions, scanner/service version, triggers/jobs, runner, network/download destinations, symbolic credential/permission references, costs and the validation target. Repository private status does not establish eligibility, enabled services, billing or runner authority: missing evidence stays unknown. Settings links are manual handoffs.

Source drafting never publishes, enables services, runs CI or changes credentials/permissions. Publication can trigger workflows and requires a separate explicit decision for the final files and all effects. A draft or model assertion does not prove that security services work.

## History and limitations

Doctor persists redacted summaries, approved action metadata, outcomes and session links in the existing product application database for 30 days, capped at 100 records and 5 MiB. Delete, clear and explicit export are available. Raw logs, full graphs, credentials and source content are excluded. Clearing Doctor history does not delete agent sessions. Interrupted actions are recorded and never replayed automatically.

Feature availability and unsupported cases are reported at the selected owner/host. Deterministic fixture validation is distinct from real toolchain, platform, GitHub account, CI and production-runtime evidence. Doctor reports remaining uncertainty instead of certifying a project secure or fixed.

See [Tasks](./tasks.md), [Debugger](./debugger.md), [Agent Panel](./ai/agent-panel.md) and [Project Settings](./configuring-zed.md) for the existing surfaces Doctor uses.
