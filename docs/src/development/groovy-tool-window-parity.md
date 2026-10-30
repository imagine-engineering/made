# Groovy Gradle/Maven tool-window parity inventory

Reference baseline: official IntelliJ IDEA **2026.2 Help**, inspected 2026-10-03.
The screenshot is a visual reference, not the acceptance boundary. No screenshot
attachment was available in this turn; functionality is inventoried from official
pages. Requirements 8–12 and canonical tasks in the existing Groovy pack govern
Made behavior. Full parity is **not delivered or qualified**.

States: **implemented** means source plus focused evidence, not platform release;
**incomplete** has required work/evidence outstanding; **blocked** names a gate;
**product decision** records an explicitly accepted equivalent, with acceptance
still required. There are no silent exclusions. Reference changes require inventory
review rather than claiming parity against an unbounded future IDEA release.

## Official sources

- [G: official reference](https://www.jetbrains.com/help/idea/jetgradle-tool-window.html)
- [M: official reference](https://www.jetbrains.com/help/idea/maven-projects-tool-window.html)
- [GT: official reference](https://www.jetbrains.com/help/idea/work-with-gradle-tasks.html)
- [MG: official reference](https://www.jetbrains.com/help/idea/work-with-maven-goals.html)
- [GP: official reference](https://www.jetbrains.com/help/idea/work-with-gradle-projects.html)
- [GD: official reference](https://www.jetbrains.com/help/idea/work-with-gradle-dependency-diagram.html)
- [MD: official reference](https://www.jetbrains.com/help/idea/work-with-maven-dependencies.html)
- [GC: official reference](https://www.jetbrains.com/help/idea/run-debug-gradle.html)
- [MC: official reference](https://www.jetbrains.com/help/idea/run-debug-configuration-maven.html)
- [GS: official reference](https://www.jetbrains.com/help/idea/gradle-settings.html)
- [MS: official reference](https://www.jetbrains.com/help/idea/maven.html)
- [W: official reference](https://www.jetbrains.com/help/idea/tool-windows.html)

## Capability matrix

Task IDs are in `.agents/specs/groovy-jvm-product/tasks.md`; requirement IDs are
in that same pack. Existing owners are mapped in its design. A row may depend on
more than one source outcome; qualification remains 11.3/11.4 and flavor 7.4.

| ID  | Source                                                                             | Reference capability                             | State            | Requirements | Task owner                                                                                                                                                                                                                                                     | Evidence/gap                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| --- | ---------------------------------------------------------------------------------- | ------------------------------------------------ | ---------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| G01 | [G](https://www.jetbrains.com/help/idea/jetgradle-tool-window.html)                | Linked project/module tree                       | incomplete       | 8.1          | 7.1, 7.11, 8.1                                                                                                                                                                                                                                                 | Cached metadata projects into the same read-only Gradle view; typed host facts pass framed remote admission/revocation/reconnect checks. Linking, installed/default providers, native and real-host qualification remain missing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| G02 | [G](https://www.jetbrains.com/help/idea/jetgradle-tool-window.html)                | Grouped tasks/modules                            | incomplete       | 8.2          | 8.2                                                                                                                                                                                                                                                            | Provider task and logical module grouping, composite-root separation and bounded workspace preferences pass mounted checks; Kotlin logical Gradle facts and native/remote qualification remain missing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| G03 | [G](https://www.jetbrains.com/help/idea/jetgradle-tool-window.html)                | Ignored/toolbar visibility                       | incomplete       | 8.2          | 8.2                                                                                                                                                                                                                                                            | Per-tool toolbar visibility persists through shared workspace KVP; provider ignore/filter semantics and native/remote qualification remain missing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| G04 | [G](https://www.jetbrains.com/help/idea/jetgradle-tool-window.html)                | Recent run configurations                        | incomplete       | 10.2         | 9.2, 9.3                                                                                                                                                                                                                                                       | Task history exists, no provider-scoped configuration tree.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| G05 | [G](https://www.jetbrains.com/help/idea/jetgradle-tool-window.html)                | Task discovery/quick-run                         | incomplete       | 9.1, 10.1    | 7.1, 7.11, 7.60, 7.61, 7.67, 7.68, 7.69, 9.1, 9.6, 9.7, 9.8, 9.9, 9.10                                                                                                                                                                                         | Current typed ordered offline plans and named single-operation controls pass local software checks through existing Tasks (9.6–9.9). Ordered UI composition passes local software checks (9.10). Full options/configurations/effective settings/default delivery, actual native builds/selected results and physical remote qualification remain open.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| G06 | [G](https://www.jetbrains.com/help/idea/jetgradle-tool-window.html)                | Selected/all reimport                            | incomplete       | 9.3          | 7.18, 7.19, 7.4, 8.11                                                                                                                                                                                                                                          | Ordinary local/framed reads now skip configuring commands and retain current host generations (7.18). Explicit local metadata synchronization passes bounded checks (7.19), but it is not full build reimport. Remote synchronization, policy and causal import completion remain missing. Explicit saved local Maven/Gradle selection controls and owned cancellation pass mounted software checks (8.11); full reimport, policy editing and physical/native/remote qualification remain open.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| G07 | [G](https://www.jetbrains.com/help/idea/jetgradle-tool-window.html)                | Link/unlink/ignore                               | incomplete       | 9.2          | 7.22, 7.25, 7.26, 7.27, 7.28, 7.29, 7.30, 7.31, 7.32, 7.33, 7.34, 7.35, 7.36, 7.37, 7.38, 7.39, 7.42, 7.52, 7.53, 7.55, 7.56, 7.57, 7.3, 8.7, 8.11, 7.86, 7.87, 7.88, 7.89, 7.90, 7.91, 7.92, 7.93, 7.94, 7.95, 7.96, 7.97, 7.98, 7.99, 7.100, 7.101, 7.102    | Guarded import/unlink, typed settings and provider identities have bounded local/framed evidence. Mounted saved-selection controls and cancellation pass8.11. Explicit local module policy7.93 and retained-root caller7.98–7.99 now sequence typed reads, child/root phases, SDK orphan preflight, restore/import and final root verification through existing LSP ownership. Fake-provider software orchestration and actual cached SDK phases7.95/7.97 are separate evidence, not a paired installed workflow. SDK7.100 observes that an explicit empty configuration list suppresses fresh metadata imports but does not close retained resources. Full production startup/default replay, nested/link/empty selections, complete required/orphan policy, policy editing8.7, causal synchronization7.4, native/remote/platform/default delivery and release gates remain open. Joint controlled-provider initialization7.101 now has bounded software parameter/lifecycle evidence; its remaining full-policy and paired/native gates stay open. Automatic saved selected-root/module policy7.102 has bounded software evidence through the existing explicit policy path; full defaults/nested/empty-selection/causal/native qualification remains open.                                                                                                                                                                                                                                                |
| G08 | [G](https://www.jetbrains.com/help/idea/jetgradle-tool-window.html)                | Source acquisition                               | incomplete       | 11.2         | 10.3                                                                                                                                                                                                                                                           | Providers can acquire artifacts; explicit verified control/policy not qualified.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| G09 | [G](https://www.jetbrains.com/help/idea/jetgradle-tool-window.html)                | Import module selection                          | incomplete       | 9.2          | 7.22, 7.52, 7.53, 7.55, 7.56, 7.57, 7.3, 8.7, 8.11, 7.86, 7.87, 7.88, 7.89, 7.90, 7.91, 7.92, 7.93, 7.98, 7.99, 7.100, 7.101, 7.102                                                                                                                            | Configuration-file root selection, SDK fresh/retained module states and explicit resource close/restore have bounded headless evidence; local typed module/root policy orchestration7.93/7.99 has separate software evidence. Earlier exclusion-flag-only probes retained a required library; explicit resource phases do not settle complete required-library/orphan policy or full synchronization. Mounted saved import controls8.11 retain their bounded scope. Full production startup/default replay, nested/link/empty selections, persisted automatic reconciliation, policy editing, paired installed/native/physical remote/platform qualification and complete parity remain open. SDK7.100 distinguishes fresh suppression from retained-resource closing; joint controlled-provider initialization7.101 has bounded software evidence, while one-provider default replay and full policy remain open. Automatic saved selected-root/module policy7.102 has bounded software evidence through the existing explicit policy path; full defaults/nested/empty-selection/causal/native qualification remains open.                                                                                                                                                                                                                                                                                                                                                                                  |
| G10 | [G](https://www.jetbrains.com/help/idea/jetgradle-tool-window.html)                | Offline/settings entry                           | incomplete       | 9.4          | 7.5                                                                                                                                                                                                                                                            | Existing JDTLS overrides and cached wrapper evidence; unified controls absent.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| G11 | [G](https://www.jetbrains.com/help/idea/jetgradle-tool-window.html)                | Expand/collapse                                  | implemented      | 8.2          | 2.5, 8.5                                                                                                                                                                                                                                                       | Existing shared actions plus named toolbar controls; 7 mounted UI checks pass; native whole Gradle view unqualified.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| G12 | [G](https://www.jetbrains.com/help/idea/jetgradle-tool-window.html)                | Open build configuration                         | incomplete       | 8.3          | 8.3                                                                                                                                                                                                                                                            | Existing confined module manifest navigation; complete task/config origin mapping missing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| G13 | [G](https://www.jetbrains.com/help/idea/jetgradle-tool-window.html)                | Refresh dependencies                             | incomplete       | 9.3          | 7.4, 10.13                                                                                                                                                                                                                                                     | Ordinary snapshot refresh does not resolve dependencies; selected explicit offline graph collection exists at the local project API. Explicit local offline Collect/Cancel controls pass mounted keyboard/failure/recovery checks (10.18). Refresh/reimport policy, native and remote provider execution remain open.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| G14 | [GP](https://www.jetbrains.com/help/idea/work-with-gradle-projects.html)           | Composite/source-set/buildSrc import             | incomplete       | 8.1, 9.1     | 7.1, 7.78, 7.79, 7.80, 7.81                                                                                                                                                                                                                                    | Composite identities and declared roots pass cached provider/local/framed checks. Configured JavaCompile annotation-processor output now distinguishes unknown/absent/declared through provider, protocol8 and the shared tree, with current directory reveal and refusal/recovery software checks. Actual generation, other generators, toolchain/buildSrc discovery, native/real-host qualification and full import remain incomplete.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| G15 | [GP](https://www.jetbrains.com/help/idea/work-with-gradle-projects.html)           | External-change/automatic/manual reload          | incomplete       | 9.3          | 7.4                                                                                                                                                                                                                                                            | Snapshot invalidation exists; policy-controlled provider reimport incomplete.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| G16 | [GT](https://www.jetbrains.com/help/idea/work-with-gradle-tasks.html)              | Custom/multiple task arguments                   | incomplete       | 10.1         | 9.1, 9.6, 9.7, 9.8, 9.9, 9.10                                                                                                                                                                                                                                  | Current typed ordered offline plans and named single-operation controls pass local software checks through existing Tasks (9.6–9.9). Ordered UI composition passes local software checks (9.10). Full options/configurations/effective settings/default delivery, actual native builds/selected results and physical remote qualification remain open.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| G17 | [GT](https://www.jetbrains.com/help/idea/work-with-gradle-tasks.html)              | Task activation/shortcuts/quick lists            | incomplete       | 10.2, 10.3   | 9.3, 9.5                                                                                                                                                                                                                                                       | No task activation model or provider shortcut labels.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| G18 | [GT](https://www.jetbrains.com/help/idea/work-with-gradle-tasks.html)              | Script/task/fork Debug                           | blocked          | 11.3         | 10.5                                                                                                                                                                                                                                                           | No qualified adapter/script/fork breakpoint journeys; inspect-only audit.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| G19 | [GC](https://www.jetbrains.com/help/idea/run-debug-gradle.html)                    | Named configs/targets/storage                    | incomplete       | 10.2         | 9.2                                                                                                                                                                                                                                                            | Existing settings/Tasks targets; provider configuration projection missing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| G20 | [GC](https://www.jetbrains.com/help/idea/run-debug-gradle.html)                    | Environment/JVM/parallel policy                  | incomplete       | 10.1, 10.2   | 7.5, 9.2                                                                                                                                                                                                                                                       | No discovery-execution effective settings contract.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| G21 | [GC](https://www.jetbrains.com/help/idea/run-debug-gradle.html)                    | Ordered before-launch/log/reveal                 | incomplete       | 10.2, 10.3   | 9.4                                                                                                                                                                                                                                                            | Current terminals exist; ordered hooks/log aliases/file-output settings incomplete.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| G22 | [GC](https://www.jetbrains.com/help/idea/run-debug-gradle.html)                    | Test forcing/task graph Debug                    | blocked          | 10.4, 11.3   | 3.4, 10.5                                                                                                                                                                                                                                                      | Selected Java tests force fresh reports; forked graph Debug not qualified.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| G23 | [GS](https://www.jetbrains.com/help/idea/gradle-settings.html)                     | Wrapper/system/JVM/home selection                | incomplete       | 9.4          | 7.5                                                                                                                                                                                                                                                            | Existing wrapper Tasks/JDTLS settings; consistent provider import settings unverified.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| G24 | [GS](https://www.jetbrains.com/help/idea/gradle-settings.html)                     | Parallel model/toolchains/annotations            | incomplete       | 9.4          | 7.5                                                                                                                                                                                                                                                            | No bounded typed control/provider qualification.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| G25 | [GS](https://www.jetbrains.com/help/idea/gradle-settings.html)                     | IDEA compiler/.iml                               | product decision | 9.4          | 7.5                                                                                                                                                                                                                                                            | Accepted equivalent: delegated builds and Made settings; same outputs/source sets still need proof.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| G26 | [GD](https://www.jetbrains.com/help/idea/work-with-gradle-dependency-diagram.html) | Configuration/transitive/cyclic graph            | incomplete       | 11.1         | 7.16, 7.17, 8.6, 10.1, 10.7, 10.12, 10.13, 10.14, 10.15, 10.16, 10.17, 10.18                                                                                                                                                                                   | Cached declarations and explicit pinned resolution facts pass provider, current local snapshot and framed wire checks. Structural/resolution coverage remains distinct. Literal configuration-name correction, fixed-depth shared-tree facts and explicit local Collect/Cancel controls pass focused checks (10.15–10.18). Full graph analysis, exact origins/artifacts and native/physical remote qualification remain open.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| G27 | [GD](https://www.jetbrains.com/help/idea/work-with-gradle-dependency-diagram.html) | Scope/conflict/duplicate/usage analysis          | incomplete       | 11.1         | 10.1, 10.2, 10.7, 10.12, 10.13, 10.14, 10.19, 10.20, 10.22                                                                                                                                                                                                     | Pinned conflicts/constraints/variants/causes are admitted to current local snapshots and bounded wire facts. Bounded direct/conflict/unavailable fact filters pass current shared-tree/keyboard/preference checks (10.19). Bounded literal graph-fact search passes mounted shared-editor and zero-query checks (10.20). Full origin/usage/multiselection analysis and qualified workflows remain pending. Bounded selected neighbours/shortest root paths pass software checks (10.22); broader analysis and physical qualification remain open.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| G28 | [GD](https://www.jetbrains.com/help/idea/work-with-gradle-dependency-diagram.html) | Graph paths/neighbours/search                    | incomplete       | 11.1         | 10.2, 10.17, 10.20, 10.22                                                                                                                                                                                                                                      | Shared facts, literal search, single-component neighbours and every equal-shortest root path pass captured/mounted software checks. Constraints do not introduce inclusion paths. Multiselection/arbitrary paths, exact origins and native/physical remote qualification remain open.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| G29 | [GD](https://www.jetbrains.com/help/idea/work-with-gradle-dependency-diagram.html) | Diagram/popup/layout/export/overview             | incomplete       | 11.2         | 10.4, 10.23                                                                                                                                                                                                                                                    | No qualified shared graph presentation; approval retains spatial/export gaps. Bounded current diagrams use the shared Markdown/Mermaid owner and pass SVG/mounted software checks (10.23); large/spatial/export/full analysis and physical qualification remain open.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| G30 | [GD](https://www.jetbrains.com/help/idea/work-with-gradle-dependency-diagram.html) | Build-origin dependency navigation               | incomplete       | 11.1         | 10.2                                                                                                                                                                                                                                                           | Manifest navigation exists, exact dependency declaration origins absent.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| M01 | [M](https://www.jetbrains.com/help/idea/maven-projects-tool-window.html)           | Reactor/module tree                              | incomplete       | 8.1          | 7.2, 8.1, 7.75                                                                                                                                                                                                                                                 | Cached provider/snapshot project module links, reactor positions and profile origins into a read-only Maven view; linking/reimport, installed/default, native and remote qualification remain open. Existing cached Maven Reader retains active descendant models beneath an unavailable aggregate/root (7.75); pinned 0.29 controls lose one/five models and 0.30 cached fresh/restart fixtures retain them. Unavailable metadata stays explicit; full policy, default delivery, native/remote and complete parity remain open.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| M02 | [M](https://www.jetbrains.com/help/idea/maven-projects-tool-window.html)           | Lifecycle/plugins/profiles/repositories          | incomplete       | 9.1          | 7.2, 7.12, 7.13, 8.1                                                                                                                                                                                                                                           | Cached phases/plugins/profile facts appear in the mounted tree; v3 declarations pass denied-network provider and typed local/framed admission checks. Static parameter declarations now project read-only with stable identities and explicit unknown effective values/CLI mapping. Goal/profile actions, reimport, acquisition, repositories and native/default/physical remote qualification remain incomplete.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| M03 | [M](https://www.jetbrains.com/help/idea/maven-projects-tool-window.html)           | Group/ignored/basic/coordinate/version options   | incomplete       | 8.2          | 8.2                                                                                                                                                                                                                                                            | Provider grouping, labels, toolbar and exact basic-phase filter persist with strict legacy migration and mounted metadata proof; basic membership/default pinned to official source, preserving provider order/IDs. Ignored projects and native/visual/remote restart qualification remain missing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| M04 | [M](https://www.jetbrains.com/help/idea/maven-projects-tool-window.html)           | Reimport/source generation                       | incomplete       | 9.3          | 7.4, 10.3, 8.11                                                                                                                                                                                                                                                | Snapshot queries exist; explicit effective source generation/reimport incomplete. Explicit saved local Maven/Gradle selection controls and owned cancellation pass mounted software checks (8.11); full reimport, policy editing and physical/native/remote qualification remain open.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| M05 | [M](https://www.jetbrains.com/help/idea/maven-projects-tool-window.html)           | Link/unlink/ignore/patterns                      | incomplete       | 9.2          | 7.14, 7.15, 7.20, 7.21, 7.23, 7.27, 7.29, 7.30, 7.31, 7.32, 7.33, 7.34, 7.35, 7.36, 7.37, 7.38, 7.39, 7.40, 7.41, 7.42, 7.43, 7.44, 7.45, 7.46, 7.47, 7.48, 7.49, 7.50, 7.51, 7.55, 7.3, 8.7, 8.11, 7.70, 7.71, 7.72, 7.73, 7.75, 7.76, 7.82, 7.83, 7.84, 7.85 | Bounded local saved-root/exact-ignore and advertised v2 POM-mask phases pass model and real FakeLSP lifecycle checks (7.73), preserving unrelated roots and sources. Current imported/ignored identities stay on the existing snapshot/query generation across refresh and graph collection; obsolete classifications are discarded with partial coverage, and settings/trust/cancellation/server/input changes reject late publication. Legacy commands and Gradle/Copper regressions are preserved. Protocol7 carries current bounded relative phase facts through the existing host/client snapshot; real protobuf and held framed reply checks pass with exact generation, permission and path admission (7.76). This supplies no startup authority or physical remote qualification. Provider0.30 cached headless fresh/restart checks retain masked-parent/root descendants (7.75) and required-library substitution (7.72); these are phase/cached-model facts. Full default-delivery/native qualification and complete required/orphan/all-provider policy (7.3), policy editing (8.7), causal synchronization (7.4), installed/default delivery and physical native/remote/platform/resource/accessibility/terms qualification remain open. No full parity or release claim. Typed current default-candidate planning/freshness is software-qualified (7.82); bounded local runtime admission/application has software evidence (7.83–7.85); installed/native and full synchronization remain open. |
| M06 | [M](https://www.jetbrains.com/help/idea/maven-projects-tool-window.html)           | Sources/Javadoc/effective POM                    | incomplete       | 11.2         | 10.3                                                                                                                                                                                                                                                           | No explicit profile-aware actions with verified delivery/offline errors.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| M07 | [M](https://www.jetbrains.com/help/idea/maven-projects-tool-window.html)           | Settings/profile file navigation                 | incomplete       | 9.4          | 7.5                                                                                                                                                                                                                                                            | Editor can open files; provider-specific create/open controls absent.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| M08 | [M](https://www.jetbrains.com/help/idea/maven-projects-tool-window.html)           | Offline/skip-tests/settings                      | incomplete       | 9.4          | 7.5                                                                                                                                                                                                                                                            | Cached Tasks pass; unified import/execution controls absent.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| M09 | [M](https://www.jetbrains.com/help/idea/maven-projects-tool-window.html)           | Repository display/reindex                       | incomplete       | 11.2         | 10.3                                                                                                                                                                                                                                                           | No provider repository maintenance; no additional semantic index allowed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| M10 | [M](https://www.jetbrains.com/help/idea/maven-projects-tool-window.html)           | Collapse/grouped tab/view controls               | incomplete       | 8.2          | 8.2, 8.4                                                                                                                                                                                                                                                       | Named selectors, collapse and persisted selected-view/toolbar options have mounted evidence; remaining settings, native/remote restart and workspace modes remain incomplete.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| M11 | [MG](https://www.jetbrains.com/help/idea/work-with-maven-goals.html)               | Ordered phases/goals/module quick-run            | incomplete       | 10.1         | 9.1, 9.6, 9.7, 9.8, 9.9, 9.10                                                                                                                                                                                                                                  | Current typed ordered offline plans and named single-operation controls pass local software checks through existing Tasks (9.6–9.9). Ordered UI composition passes local software checks (9.10). Full options/configurations/effective settings/default delivery, actual native builds/selected results and physical remote qualification remain open.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| M12 | [MG](https://www.jetbrains.com/help/idea/work-with-maven-goals.html)               | Saved/recent goal configurations                 | incomplete       | 10.2         | 9.2, 9.3                                                                                                                                                                                                                                                       | No provider named configuration tree.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| M13 | [MG](https://www.jetbrains.com/help/idea/work-with-maven-goals.html)               | Build/rebuild/run activation/shortcuts           | incomplete       | 10.2, 10.3   | 9.3, 9.5                                                                                                                                                                                                                                                       | No ordered activation lifecycle integration.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| M14 | [MG](https://www.jetbrains.com/help/idea/work-with-maven-goals.html)               | Goal/multiple-goal Debug                         | blocked          | 11.3         | 10.5                                                                                                                                                                                                                                                           | No qualified goal adapter/configuration workflow.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| M15 | [MC](https://www.jetbrains.com/help/idea/run-debug-configuration-maven.html)       | Host/storage/environment/JRE/VM options          | incomplete       | 10.1, 10.2   | 7.5, 9.2                                                                                                                                                                                                                                                       | Existing host Tasks only; configuration equivalence unqualified.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| M16 | [MC](https://www.jetbrains.com/help/idea/run-debug-configuration-maven.html)       | Profiles/home/settings/repository/threads        | incomplete       | 9.4          | 7.5                                                                                                                                                                                                                                                            | Typed effective overrides/discovery contract missing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| M17 | [MC](https://www.jetbrains.com/help/idea/run-debug-configuration-maven.html)       | Recursive/workspace/skip/checksum/log/fail flags | incomplete       | 9.4, 10.1    | 7.5, 9.1                                                                                                                                                                                                                                                       | No complete typed flag-to-provider-plan projection.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| M18 | [MC](https://www.jetbrains.com/help/idea/run-debug-configuration-maven.html)       | Logs/ordered before-launch hooks                 | incomplete       | 10.2, 10.3   | 9.4                                                                                                                                                                                                                                                            | Same existing Task owner must sequence; ancillary reference hooks unqualified.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| M19 | [MS](https://www.jetbrains.com/help/idea/maven.html)                               | Wrapper/.mvn config/settings precedence          | incomplete       | 9.4          | 7.5                                                                                                                                                                                                                                                            | Corrected Maven wrapper digest passes bounded local bootstrap; host settings parity missing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| M20 | [MD](https://www.jetbrains.com/help/idea/work-with-maven-dependencies.html)        | Direct/transitive/scope/conflict graph           | incomplete       | 11.1         | 10.1, 10.8, 10.9, 10.10, 10.11, 10.14, 10.17, 10.18, 10.19, 10.20, 10.21, 10.22                                                                                                                                                                                | Actual Maven occurrence/mediation/scope/exclusion facts pass pinned collection, local admission and framed transport. Shared fixed-depth facts, explicit local collection/cancellation, direct/mediation/unavailable facets and literal Maven scope filters pass automated checks (10.17–10.21). Cut-cycle occurrence mapping, exact origins, multiselection/arbitrary-path/usage analysis and native/physical remote journeys remain open. Bounded selected neighbours/shortest root paths pass software checks (10.22); broader analysis and physical qualification remain open.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| M21 | [MD](https://www.jetbrains.com/help/idea/work-with-maven-dependencies.html)        | Origin/exclusion/usage/root/neighbour analysis   | incomplete       | 11.1         | 10.2                                                                                                                                                                                                                                                           | Editor/POM remains source of truth; exact graph/declaration actions absent.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| M22 | [MD](https://www.jetbrains.com/help/idea/work-with-maven-dependencies.html)        | Diagram/filter/search/export                     | incomplete       | 11.2         | 10.2, 10.4, 10.17, 10.19, 10.20, 10.21, 10.23                                                                                                                                                                                                                  | Admitted fixed-depth facts, bounded metadata facets and literal search pass shared-tree/editor software checks. Spatial diagram/layout/export and native/physical remote presentation remain unimplemented or unqualified; no waiver is inferred. Bounded current diagrams use the shared Markdown/Mermaid owner and pass SVG/mounted software checks (10.23); large/spatial/export/full analysis and physical qualification remain open.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| S01 | [W](https://www.jetbrains.com/help/idea/tool-windows.html)                         | Dock/hide/move/resize/tab modes                  | incomplete       | 8.2          | 8.4, 8.8, 8.9                                                                                                                                                                                                                                                  | Existing shared Dock now reconciles Groovy entity notifications and Copper settings moves with bounded mounted focus/size proof. Versioned position persistence, latest-write ordering and entity reload/release pass automated checks; top/float/window/undock/pin equivalents and native/restart evidence remain incomplete.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| S02 | [GC](https://www.jetbrains.com/help/idea/run-debug-gradle.html)                    | Nonlocal targets/ancillary coverage/hooks        | incomplete       | 10.2         | 9.2, 9.4                                                                                                                                                                                                                                                       | SSH owners exist; every Docker/WSL/log/coverage/hook equivalence needs explicit evidence.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| S03 | [GP](https://www.jetbrains.com/help/idea/work-with-gradle-projects.html)           | Cancel/recover/stale sync                        | incomplete       | 9.3, 10.4    | 7.4, 7.59, 9.1, 8.11                                                                                                                                                                                                                                           | Existing Tasks/test cancellation and snapshots; selected local import/unlink supports exact phase receipts and recovery checks. Full reimport lifecycle absent. Explicit saved local Maven/Gradle selection controls and owned cancellation pass mounted software checks (8.11); full reimport, policy editing and physical/native/remote qualification remain open.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| S04 | [M](https://www.jetbrains.com/help/idea/maven-projects-tool-window.html)           | Dependency analyzer entry                        | incomplete       | 11.1         | 10.2, 10.17, 10.18, 10.19, 10.20, 10.21, 10.22                                                                                                                                                                                                                 | Existing Dependencies/Maven/Gradle view exposes explicit local offline graph collection, current fact rows, metadata facets and literal search. A qualified analyzer journey with multiselection/arbitrary-path/usage/origin actions remains open. Bounded selected neighbours/shortest root paths pass software checks (10.22); broader analysis and physical qualification remain open.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| A01 | [GT](https://www.jetbrains.com/help/idea/work-with-gradle-tasks.html)              | Selected tests/Debug                             | blocked          | 10.4, 11.3   | 3.4, 10.6                                                                                                                                                                                                                                                      | Cached local Java Tests passed; Kotlin discovery and selected DAP breakpoints not qualified.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| A02 | [GP](https://www.jetbrains.com/help/idea/work-with-gradle-projects.html)           | Remote trust/offline/restart journeys            | blocked          | 12.2         | 11.3, 11.4                                                                                                                                                                                                                                                     | Old source native cached observations are bounded; expanded candidate and real remote not qualified.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| A03 | [W](https://www.jetbrains.com/help/idea/tool-windows.html)                         | Keyboard/focus/screen reader/platform            | blocked          | 12.2         | 6.5, 11.3                                                                                                                                                                                                                                                      | 0e33 file Picker omitted AX rows; corrective source test passes; changed-source native, VoiceOver and platforms open.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| A04 | [GS](https://www.jetbrains.com/help/idea/gradle-settings.html)                     | Provider terms and release cells                 | blocked          | 12.3         | 1.1, 11.5                                                                                                                                                                                                                                                      | Internal adapter delivery direction approved; exact payload terms/pins/platform proof still open.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| A05 | [W](https://www.jetbrains.com/help/idea/tool-windows.html)                         | Actions infrastructure                           | blocked          | 12.3         | 11.5                                                                                                                                                                                                                                                           | User reports unavailable; no polling/dispatch/enabling; local tests are not CI success.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

## Accepted decisions and remaining gates

On 2026-10-03 the owner approved all four presented recommendations: delegated
Maven/Gradle builds plus Made settings (no IDEA compiler/.iml writer); shared
presentation/graph analysis while spatial/export/window-mode gaps remain open;
an internal extension-managed discovery adapter using official Gradle Tooling API
and Maven model/plugin metadata (no public SDK); and existing DAP with a pinned,
terms-approved Java adapter and separate script/fork/goal/test qualification.
This approval selects architecture, not exact artifact hashes or governing terms.
The original extension-managed delivery, ARM64 32 GiB/8 GiB ceiling, unadvertised
agent sharing and provisional production identity decisions remain intact.

Unresolved delivery gates: exact helper/runtime/adapter versions, immutable
hashes, applicable terms/notices and platform coverage. Do not infer a binary
redistribution grant from a source license. Production identity/update service and
platform cells remain under the flavor pack. No generic replacement compiler,
panel framework, Cargo model, project index, execution engine or result store is
approved. Existing DAP/UI owners must justify missing infrastructure concretely.

## Evidence classes and fixture acceptance

- Automated: mounted tree/Picker selection and focus; provider bounds and schema;
  stale/revoked action denial; host-only plan projection and platform quoting;
  ordered/cancelled/failing tasks, fresh stable JUnit IDs and Copper regressions.
- Cached local integration: existing two-module Maven/Gradle Java/Kotlin fixtures;
  add a composite Gradle build, profile-dependent Maven module/generated sources,
  custom task options, duplicate basenames, transitive diamond/conflicts/cycle,
  cancellation and deliberate missing tool/plugin/dependency failures. Compare
  effective provider facts, actual commands and fresh report IDs with expected
  fixture behavior. A warmed cache pass does not prove denied network.
- Native: exact approved source/binary/profile plus disposable fixtures; all
  discovery, configurations, dependency navigation, link/unlink, reload, settings,
  selected tests, restart/fallback/recovery and owned-worker sampling; new bytes
  need a new exact candidate approval. Current 0e33 observations are separate.
- Debug: approved adapter, source-mapped Java/Kotlin and build-script/fork/goal/test
  breakpoints, cancellation/recovery, no false session for unsupported targets.
- Remote: real connected separate host, effective host JDK/cache/tools, trust,
  reconnect, cancellation and navigation permissions; loopback RPC tests are local
  software evidence only.
- Visual: reference correspondence separately from workflow/AX evidence; graphical
  similarity alone is no functional pass. Physical platform: each advertised
  OS/architecture package plus keyboard/focus and real assistive technology.

GitHub Actions remains stopped and infrastructure-blocked. VoiceOver stays off,
its physical gate deferred. `jvm` remains planned; no release is published.

Remote metadata checkpoint (2026-10-04): canonical 7.11 carries provider-specific
Gradle/Maven project, source/task/option and reactor/profile/plugin/lifecycle
facts through the existing JDTLS snapshot. Framed in-process host/client,
privacy, revocation and reconnect checks pass alongside Copper regressions.
This is distinct from actual installed remote providers, tool execution and
physical/native/platform qualification. G01/G05/G14 and corresponding Maven
rows remain incomplete; no full reference capability or release gate is waived.

### Project-selection ownership reconciliation — 2026-10-04

G07/G09/M05 remain incomplete. The existing provider selected-import contract is
consumed by 7.20; current LSP invocation is 7.21. Gradle and Maven mutation/
exclusion qualifications are separate 7.22/7.23; saved intent stays 7.3 and
controls stay with the existing panel under 8.7. Full import/reload completion
remains 7.4. Partial cached Maven discovery no longer imposes a cyclic full
reimport prerequisite on source linking work. No completed evidence or reference
capability is rewritten. Request/receipt checks will not establish product
linking, native restart, remote execution or release readiness.

### Selected Maven invocation lifecycle checkpoint (task 7.21)

The existing LSP owner now guards explicit local selected-POM imports with current
trust/files/server/generation, preserves the phase across expected import progress,
provider events and ordinary reads, and cancels on real input/trust/server changes.
Mounted fake-provider checks confirm bounded failures and read recovery. Receipt
completion does not establish causal synchronization, provider selection policy,
persistence, product controls or native/remote journeys. G07/G09/M05 remain
incomplete; 7.3, 7.4, 7.22, 7.23 and 8.7 retain their canonical ownership. No full
parity or release gate is marked passed by this checkpoint.

### Maven exclusion and orphan provider qualification (task 7.23)

The pinned provider passes 72 headless checks across inherited and aggregate-only
fixtures. Whole-root scanning exclusion survives restart; required child modules
still import, explicit selected POMs bypass root scan exclusion, and changed
selection or removed module declarations retain old imported projects in an
existing profile. Fresh-profile membership differs. These facts require explicit
provider membership reconciliation under 7.3/8.7; they do not implement Ignore,
Unlink, automatic orphan removal or full reload in either tool window. M05/G07/G09
remain incomplete, with the original full parity requirements and release gates.

### Gradle mutation qualification follow-up — 2026-10-04

G07/G09 remain **incomplete**. Task 7.22 passes 53 headless provider checks/ten
journeys, including unsupported upstream facts: invalid selections acknowledge
null, malformed arguments expose raw Java errors, and late cancellation of an
acknowledged request does not stop its workspace job. Missing-required-project
metadata failure/repair passes sanitized; cached undeclared-module membership
persists versus fresh-profile omission. Sources/no test reports/owned worker
shutdown pass. This is provider evidence, not rendered/visual/native/remote parity.

7.24–7.26 now explicitly own a guarded Buildship import phase, its isolated runtime
qualification and current local LSP invocation. Existing 7.3/8.7/7.4 keep persisted
intent, controls and full synchronization; full reference capabilities remain
required. Evidence: target/groovy-gradle-model/gradle-mutation-failures and
gradle-mutation-bindings.json. VoiceOver, CI infrastructure, platform/package/
production identity/provider delivery and native approval remain open; `jvm`
stays planned.

### Guarded Gradle provider phase — 2026-10-04

G07/G09 remain **incomplete**. Internal development provider 0.8.0 now offers a
strict request-bound Buildship import-phase receipt rather than an early null
acknowledgement. Final artifact passes 29 real Gradle checks/two sessions and
46 existing Maven metadata/import regressions under denied external network.
Malformed/foreign/tool/offline/bootstrap inputs, missing required project/cache,
repair, held-request rejection/cancellation recovery and restart are covered;
source preservation/no test reports/owned worker stop pass. Cancellation is
cooperative and may be partial, with no hard worker termination/rollback claim.

7.24/7.25 are bounded source/contract completions; current client/trust/generation
invocation stays 7.26, intent/controls 7.3/8.7 and full sync 7.4. No native/visual/
remote/platform/default-download or full parity pass. Provider terms, production
identity/packages, Kotlin budget/platform, CI infrastructure and deferred VoiceOver
remain open. Exact pending native approval is not transferred to this new source.
Evidence: target/groovy-gradle-model/gradle-import-phase and
gradle-import-phase-bindings.json; `jvm` stays planned.

## Guarded local Gradle client checkpoint — 2026-10-04

Task 7.26 now supplies a strict Gradle phase request/receipt and explicit local
LSP invocation. It shares the existing Maven phase token/task, current trust and
root/server guards, cancellation, teardown and sanitized transport; Maven API
and wording remain unchanged. No implicit import, upstream-null fallback, UI
control or saved intent is added. Every present recognized Gradle root input
must be safe; a valid input cannot hide an unsafe sibling. Expected import
progress/events/read refresh cannot replace the phase; success schedules only a
separate model read. Remote invocation stays unavailable for this source leaf.

The final-source unified Rust 1.97.1 build passes, followed by 106 active focused
checks (24 JVM model, 14 UI/model, 8 LSP, 16 Copper provider, 19 structured store,
1 sanitizer, 23 shared Tests, 1 framed JDTLS); 11 existing/opt-in tests ignored.
Process suites run serially. Repository warnings-denied Clippy and workspace
formatting pass. Initial mounted cleanup incorrectly used FakeFs remove_file
on a symlink; moving only that isolated link fixes the fixture and the complete
14-check UI suite passes. Retained logs include that failure, linker compact-
unwind size warning and existing block future-compatibility notice. Evidence:
`target/groovy-gradle-model/gradle-client-*.log`, `gradle-client-bindings.json`.

This is bounded automated source evidence, not native/visual/platform/remote
qualification or full synchronization. Linking/ignore/saved intent remain 7.3,
controls 8.7, causal reload 7.4; guarded unlink correction remains 7.27–7.30. Full
parity is incomplete. Provider terms/default delivery, native/physical remote,
Kotlin resource qualification and production packaging remain open; CI stopped,
VoiceOver deferred and jvm planned.

## Guarded source-preserving unlink checkpoint — 2026-10-04

Completed bounded source leaves 7.27–7.29 add separate internal
`groovy.maven.unlinkSelected` / `groovy.gradle.unlinkSelected` commands through
the existing JDTLS Eclipse workspace owner. Exact initialized root and up to
128 unique canonical imported project identities are validated before mutation,
including provider nature, symlink/foreign/missing/duplicate/bound rejection.
Root/module identities are rechecked after waiting for the workspace rule.
Resource deletion has content deletion disabled; diagnostics and deleted-project
events stay with JDTLS. Successful receipt identifies only the selected removal
and explicitly leaves full synchronization unknown. No Task or build executor,
project index, result store, public SDK or generic panel framework is introduced.

Final development provider 0.9.0 SHA-256:
`d76b19a34194ea5e5afd3e04ce731e384e3c1bf97ff963fe98d5bdbf06e5efc5`.
Warnings-denied pinned compile targets Java 17 bytecode. Actual runtime binding
now includes existing JDTLS payload `org.eclipse.lsp4j_1.0.0.v20260209-1721.jar`
SHA `ccd78893facc6bfcc359d56cba05d3d5b85eb41e4c40d4b4215ca45db5f416d9`.
Its about.html SHA `9c7157f55d3d48ddecd668cb0c462565f1757071fde71b17d14e821ba2e71828`
declares EPL 1.0 plus redistributor conditions. Exact governing terms/complete
notices remain 1.1; the bundle uses that host payload and does not embed it or
claim bundled-server redistribution rights.

Real denied-network fixtures pass 60 Maven checks/four journeys and 34 Gradle
checks/two journeys. A temporary test-only delegate seeds a resource marker and
holds the existing workspace rule; stack evidence confirms the real unlink
handler is waiting before cancellation. Cancellation rejects success and keeps
membership intact; replacing the root while held is rejected before mutation.
Repaired retry clears marker diagnostics, emits exact deleted identities and
preserves sources/generated sentinel hashes and unrelated roots. Relink and
same-profile restart pass. Only fresh owned Gradle workers stop (exit 0).
This proves cooperative waiting-phase cancellation, not hard-stop or rollback.
Existing 11 provider classes are byte-identical and default metadata helper ASTs
unchanged; prior 7.22/7.23 required/orphan scanner facts are retained, with no new
automatic ignore/orphan reconciliation claim. Initial missing LSP4J binding and
test-only OSGI singleton failures are retained separately from passing runs.

Evidence: `target/groovy-gradle-model/unlink-current-*`,
`unlink-provider-bindings.json`, and durable guarded-unlink-20261004 archive.
Rust remains delivered 23c9b3f58a/integrated 790d60bc92; 106 focused JVM/shared/
Copper checks and repository Clippy/formatting pass after the safe dev merge.
Client source remains 7.30, intent 7.3, controls 8.7, full sync 7.4 and native/
physical remote 11.3/11.4. Full parity incomplete; distribution/default acquisition,
platform/resource/production identity remain open. CI stopped, VoiceOver deferred,
jvm planned and no release published.

## Guarded local unlink client checkpoint — 2026-10-04

Task 7.30 adds explicit local Maven/Gradle unlink APIs through the same LSP-owned
phase, request and cancellation plumbing as selected imports. Selection requires
exact current provider module identities on the trusted worktree, visible safe
directories and the advertised guarded command. Only the active phase token may
revalidate its retained pre-phase snapshot. Root, server, trust and input bindings
are rechecked before dispatch and receipt admission. Strict bounded phase receipts
cannot certify full synchronization. Provider kind stays authoritative in JDTLS;
no Cargo model, second index/store/executor or UI framework is added.

Final source checks pass: 25 model, 17 UI (including mounted import/unlink
lifecycles), eight LSP, 16 active Copper provider, 19 structured-store, one
sanitizer, 23 shared Tests and one framed transport check: 110 active, 11 existing
or opt-in ignores. Mounted fixtures cover missing guarded commands, unobserved
and stale identities, deleted-project events, held progress, explicit cancel,
trust/input/server changes, teardown, sanitized failures and repaired recovery.
Existing import behavior remains covered. These are automated source checks;
no native/visual or physical remote workflow is inferred. Build retains the
macOS compact-unwind warning and existing block future-compatibility notice.

Evidence: `target/groovy-gradle-model/unlink-client-final-*.log` and
`unlink-client-bindings.json`. Saved intent remains 7.3, controls 8.7, full causal
synchronization 7.4 and physical remote 11.4. Full parity, default delivery/terms,
Kotlin resource/platform and production packaging gates remain open. Actions
stopped, VoiceOver deferred, jvm planned; no release published.

## Saved provider intent contract checkpoint — 2026-10-04

Task 7.31 adds passive typed `jvm_builds` intent to the existing settings content
and a JVM-only project consumer. Maven selected POMs, Gradle selected build files
and exact ignored project directories stay distinct. Missing versus explicitly
empty selection, vector replacement and independent provider overrides are
preserved. Bounded unique host-relative paths reject escapes, absolute/foreign
identities, control/glob text and wrong build-file names with fixed recovery text.
Admission also reuses SettingsStore's existing per-file error records: malformed
user values cannot fall back to all-root selection, applicable project errors
block admission, repair/removal recovers and other worktrees stay independent.
No error store/parser, Cargo project model, index, executor or panel framework.
Copper parsing needs no JVM/JDK/process initialization. The existing VS Code
constructor keeps JVM intent absent.

Final-source unified builds pass. 26 model, one shared-store intent, 37 settings,
17 UI, eight LSP, 16 active Copper provider, 19 structured-store, one sanitizer,
23 shared Tests and one framed transport checks pass: 149 active, 11 existing or
opt-in ignores. Store tests cover precedence, explicit emptiness, unrelated LSP
preservation, replay/removal and malformed-field admission. This is settings
replay/source evidence, not a native restart or applied provider reconciliation.
Repository warnings-denied Clippy, workspace formatting and both canonical/manual
audits pass. Build retains macOS compact-unwind and existing block future-
compatibility notices.

The ownership audit confirms a further gap from pinned [JDTLS source](https://raw.githubusercontent.com/eclipse-jdtls/eclipse.jdt.ls/v1.61.0/org.eclipse.jdt.ls.core/src/org/eclipse/jdt/ls/core/internal/JDTDelegateCommandHandler.java): default
java.project.getAll enumerates Java projects; includeNonJava true enumerates
resources, while getSettings requires Java semantics. Corrective 7.32 must qualify
that existing API for Maven POM/aggregate and Gradle non-Java roots; 7.33 keeps
bounded identities in the current snapshot/host protocol. No second project index
or invented classpath. Actual runtime qualification remains open. Saved intent
application/required-orphan/pattern policy stays 7.3, controls 8.7 and full sync 7.4.

Evidence: `target/groovy-gradle-model/intent-final-*.log`,
`intent-check-commands.json` and `intent-bindings.json`. Full tool-window parity,
native/physical remote/default delivery/terms/platform/resource/production identity
and release gates remain open. Actions stopped, VoiceOver deferred, jvm planned;
no release published.

### Safe dev integration recheck — 2026-10-04

Source `a7fc1ecbbed5509280a8ebaa2d029d8ed575797b` safely merges dev changes
through `f48807859f`. Fresh unified build passes in 5m29; all 149 active checks
pass, with 11 existing/opt-in ignores. Repository Clippy passes in 2m37 and
workspace format check passes. Shared agent/UI changes were rebuilt through the
Groovy consumers. `integration-evidence.tar.gz` and `integration-manifest.json`
bind exact commands and source. Rebuilt binary hashes could not be retained:
`target/debug/deps` was empty at archival; no cause or reconstructed hash is claimed. Original source evidence retains its original hashes; original evidence remains
immutable. Compact-unwind and block future-compatibility notices remain. This
adds local source/regression evidence only; physical/native/remote workflows,
applied selection, non-Java membership and full parity remain open. Actions
stopped, VoiceOver deferred, jvm planned.

### Imported non-Java resource qualification (7.32) — 2026-10-04

Pinned provider 0.9.0 (`d76b19a34194ea5e5afd3e04ce731e384e3c1bf97ff963fe98d5bdbf06e5efc5`),
JDTLS 1.61.0-SNAPSHOT and existing cached M2E/Buildship inputs pass **50 checks in
six real headless journeys**: inherited Maven, aggregate-only Maven, and Gradle
composite/base roots, each with same-profile restart. External network is denied;
Maven/Gradle use offline copied caches and a copied JDTLS install. No native app is
launched. Sources and generated sentinels remain unchanged; Java children and an
unrelated root survive exact non-Java guarded unlink and offline import repair.
Gradle's isolated owned-home stop succeeds (one daemon stopped); no other worker
is selected. Default metadata probe/helper ASTs remain identical; explicit mode
arguments reject conflicting modes.

Actual wire evidence corrects the earlier source-only assumption: the existing
`java.project.getAll` includeNonJava option must be a **JSON string**, because
[pinned JSONUtility](https://raw.githubusercontent.com/eclipse-jdtls/eclipse.jdt.ls/v1.61.0/org.eclipse.jdt.ls.core/src/org/eclipse/jdt/ls/core/internal/JSONUtility.java)
converts JsonElement/string/model instances, not the deserialized map. The bare
Java query omits the reactor POM and Gradle base/composite roots. Enumeration
alone reports no provider kind. Contrary to the initial failure hypothesis,
`getSettings` with only `org.eclipse.jdt.ls.core.natureIds` succeeds on these
resources: Maven reports only its M2E nature; Gradle reports only Buildship nature,
with no Java nature. This supports bounded optional nature facts, not fabricated
Java classpaths/source/VM semantics. Missing/unsupported facts must remain unknown;
guarded provider mutation still validates authoritative current ownership.

Failed nested-sandbox, map-option and nature-assumption attempts are retained,
followed by final source-bound passing receipts. Durable evidence is
`.agents/evidence-archives/groovy-jvm/imported-resources-20261004/`; local working
receipts are `target/groovy-gradle-model/resource-membership-20261004/`.
This completes only the opt-in provider qualification boundary. Existing product
snapshot still misses these resources; 7.33 owns admission/framed generation and
permissions, 7.3 applied saved intent, 8.7 controls and 7.4 full causal synchronization.
Commands explicitly keep fullSynchronizationKnown false. No full parity, visual,
native/physical remote, platform/resource, default delivery/terms or production
package claim. GitHub Actions stopped; VoiceOver deferred; jvm planned.

Imported membership admission (7.33) is in progress: existing JDTLS project
entries and framed protocol v3 carry imported Java/non-Java, metadata-only or
unknown provenance. Legacy v1/v2 projections retain unknown membership. Unlink
requires an observed imported identity; shared-tree accessible metadata exposes
provenance. Directory/privacy/bounds/duplicate/generation admission owners remain
shared, and Java semantics cannot be fabricated for non-Java/metadata-only entries.
No generic snapshot framework, project index, execution engine or result store
was added. All-target repository Clippy and fmt pass after scoped fixes; clean
compact-debug test build and runtime regressions remain pending. This is no native,
physical remote, full-sync or full-parity qualification. Actions stopped; VoiceOver
deferred; jvm planned.

### Imported membership admission completion (7.33) — 2026-10-04

The existing JDTLS project collection now retains imported Java/non-Java,
metadata-only or unknown provenance. Java semantic requests use only the qualified
Java list; JSON-string includeNonJava enumeration adds bounded resource identities
without invented Java source/classpath/VM facts. Metadata-only provider entries
and legacy v1/v2 projections are not import evidence. Guarded local unlink requires
an observed imported entry; the existing server retains build-nature authority.
Protocol v3 carries provenance through the existing framed snapshot and ownership,
privacy, generation and permission admission. Duplicate/foreign/private/file/non-
Java-semantic/unknown-enum payloads are rejected or withheld. Unsupported resource
enumeration keeps qualified Java entries with partial coverage and a visible
refresh/setup recovery hint. Shared language_tool_tree rows expose provenance
using existing accessible metadata; no second collection/index/framework/store or
execution engine was added.

A clean compact-debug unified build passes in **33m39s**. **153 active checks pass**:
27 JVM model, one shared-store intent, 37 settings, 20 UI/lifecycle, eight LSP,
16 Copper provider, 19 structured-store, one sanitizer, 23 shared Tests and one
framed host lifecycle. **11 existing/opt-in ignores are not passes.** New cases
exercise non-Java Maven/Gradle held unlink, cancellation, changed inputs/trust,
server replacement and recovery; unavailable-resource coverage, legacy projections,
private/foreign/malformed bounds, unknown kind and framed revocation/reconnect.
Exact source/binary hashes and commands are retained. The all-target repository
Clippy, final workspace fmt and both complete canonical validators pass. Earlier
Prost conversion/test-helper compile failures are retained; scoped fixes pass.
Expected negative-path RPC/server teardown logs remain in passing test receipts.
Only finished owned incremental caches were reclaimed under the exact compiler
guard, with receipts retained. Compact-unwind and block future-compatibility
notices remain.

Durable source evidence is `.agents/evidence-archives/groovy-jvm/imported-admission-20261004/`;
working logs are `target/groovy-gradle-model/resource-membership-20261004/admission-*`.
This completes the bounded existing-snapshot admission correction. Applied saved
selection/required-orphan/ignore/pattern policy (7.3), causal synchronization (7.4),
effective host settings (7.5), controls (8.7), execution/configuration/graph/Debug
parity and physical/native/remote/platform/resource/default-delivery/terms/package
qualification remain open. Framed tests are automated source evidence, not physical
remote or native certification. Actions stopped; VoiceOver deferred; jvm planned.

### Independent startup controls qualification (7.34) — 2026-10-04

The opt-in mixed Maven/Gradle startup probe passes **22 checks across six real
headless pinned JDTLS journeys**, plus ten syntax/default-helper/CLI checks.
On pristine fixtures, disabling only Gradle preserves default Maven import and
vice versa; both disabled yields no imported build projects. Existing guarded
offline Maven/Gradle phases restore the selected reactor/build without dropping
other roots. Same-profile restart retains Java and non-Java imported resources.
A fresh profile over already-imported source files still admits Java projects via
existing Eclipse metadata even with both build importers disabled; its observed
non-Java roots are absent. Startup suppression is therefore not exclusion or
unlinking acceptance. The initial shared-fixture isolation failure is retained,
followed by separated pristine/retained fixtures; its script hash was not captured
and is not reconstructed. The final script and pinned provider inputs are hashed.

Every successful journey preserves source/build/generated sentinel hashes, emits
no test reports, uses copied offline Maven cache and externally denied networking,
and stops only the isolated Gradle home's worker (exit 0). Existing default and
import-selection helpers are AST unchanged. Buildship's denied version-list request
is retained in provider logs; offline configuration does not prevent that attempt.
The probe is not installed/default delivery, native, physical remote, Tasks/Tests/
Debug, full synchronization, required/orphan/ignore-pattern or full-parity proof.
Startup projection (7.35), full persisted reconciliation (7.3), causal sync (7.4),
host settings (7.5) and controls (8.7) retain their owners. The global
projectConfigurations list is deliberately absent. No generic infrastructure,
index/store/engine/Cargo model or public SDK was added.

Durable evidence: `.agents/evidence-archives/groovy-jvm/startup-controls-20261004/`;
working receipts: `target/groovy-gradle-model/startup-controls-20261004/`.
GitHub Actions remains stopped, VoiceOver deferred and jvm planned.

### Nested selected root qualification (7.36) — 2026-10-04

The existing opt-in mixed pinned provider fixture passes **19 checks across two
real headless journeys**, plus ten syntax/default-helper/CLI checks. With only
the container initialized, guarded nested Gradle import fails with a sanitized
error; offline Maven import under that container succeeds. Adding the selected
nested Gradle folder through workspace/didChangeWorkspaceFolders permits guarded
offline import, preserving Maven resources. Removal drops its Java/non-Java Gradle
resources and preserves Maven; the removed root is rejected by guarded import.
Re-addition alone recovers Java resources through existing Eclipse metadata;
explicit guarded repair restores the non-Java Gradle base root. Same-profile
restart with container plus chosen folder retains all qualified identities.
Source/build/generated sentinel hashes remain unchanged and no tests are executed.
External network is denied; only the isolated Gradle worker is stopped (exit 0).

Retain exact initialized-folder URIs/events, phase acknowledgements with
fullSynchronizationKnown false, stable five-second membership observations,
sanitized negative responses and provider logs. The unchanged fixture client's
workspace-folder request reply still names its container, so this qualifies the
explicit initialize/notification contract, not a complete product folder owner.
Source/provider hashes match before and after; the default metadata/import helpers
are AST unchanged. The six default startup journeys were not rerun after this
opt-in mode extension; 7.34's original receipts retain their committed binding.
No installed/default delivery, trust UI, native/physical remote/full causal sync,
required/orphan/ignore-pattern, platform/resource or full-parity qualification.

The existing production LSP folder, identity and trust/permission owner must apply
these facts in 7.3; no generic root model, index, panel framework, execution engine
or result store is added. Startup projection 7.35 remains in progress pending the
fresh Rust build and tests. All parity rows stay incomplete. Durable evidence:
`.agents/evidence-archives/groovy-jvm/linked-roots-20261004/`; working logs:
`target/groovy-gradle-model/linked-roots-20261004/`. Actions stopped, VoiceOver
deferred, jvm planned; no release.

### Controlled startup import projection (7.35) — 2026-10-04

The existing LSP launch-disposition, seed, initialization/configuration merge and
restart owners now project independent Maven/Gradle startup controls from validated
passive `jvm_builds` intent. Explicit selected paths (including an empty list) or
nonempty ignored-directory intent suppress only that provider's automatic imports.
Absent/empty-ignore intent preserves defaults; the other provider, extension bundles
and unrelated settings remain intact. Global projectConfigurations is not generated.
Conflicting advanced overrides and malformed saved settings fail through existing
sanitized setup status before binary acquisition; repair recovers the existing owner.
Controlled false values survive initial, requested and dynamic configuration.
Removal/switching controls uses existing server restart; ordinary dynamic settings
preserve the current server. False boolean/string overrides remain compatible.

**165 active checks pass**, including the actual Project integration target, trust
restrictions, shared LSP settings, current JVM views, Copper providers, structured
results, shared Tests and framed host reduction. Eleven existing/opt-in ignores are
not passes. A separate Copper integration check with JVM features disabled passes
**1/1**, including passive JVM settings. Source/binary hashes match across the
main run. Warnings-denied repository Clippy and workspace formatting pass. Retain
initial ambiguous AsRef, integration ConnectionResult and Copper feature-selector
failures, corrected reruns, compact-unwind and existing block future-compat notices.
A library-only command does not qualify integration behavior, and zero-active-test
filters are rejected.

This closes only startup projection. Applied selected/ignored/required-library/
pattern/orphan reconciliation remains 7.3, controls 8.7, causal full synchronization
7.4. Physical native/restart/remote, installed/default provider delivery, Kotlin
resource/platform/accessibility, terms and release gates remain open. No second
index, executor, result store or panel is added. Actions stopped, VoiceOver deferred
and jvm planned. Durable evidence: `.agents/evidence-archives/groovy-jvm/startup-projection-20261004/`.

<!-- impl: crates/project/src/jvm_project.rs#JvmStartupImportControls -->
<!-- impl: crates/project/src/manifest_tree/server_tree.rs#InnerTreeNode -->
<!-- impl: crates/project/src/lsp_store.rs#LanguageServerSeedSettings -->

### Imported build-kind provider capability (7.37) — 2026-10-04

Development provider **0.10.0**, SHA-256
`b1b75efe40b92707af2e2b42c69d0eb6417a9d7bfb4462a02f64424f245e857c`,
adds separate cached-only `groovy.maven.importedProjects` and
`groovy.gradle.importedProjects` reads at the existing Eclipse resource owner.
Version-1 responses contain canonical confined project identities and actual Java
nature, bounded enumeration/records, explicit partial coverage and unknown full
synchronization. Ambiguous Maven/Gradle natures are omitted with partial coverage;
invalid/uninitialized/foreign requests return fixed actionable sanitized errors.
These reads run no build logic and mutate no provider/source state.

**39 checks across two real pinned headless journeys pass** with external network
denied: known-empty controlled startup, guarded cached imports, exact Maven POM
aggregate/Java and Gradle base/Java identities, separation of unrelated roots,
actual ambiguous nature omission and restoration, sanitized malformed/foreign
requests, retained-profile restart and source/build/generated-file preservation.
Only isolated owned Gradle workers stop (exit 0). The ambiguous nature is created
and restored by an opt-in test-only delegate compiled by the existing fixture
owner; it is excluded from the product provider. Final source/artifact hashes
match across the run. The earlier 34-check preliminary run is separately bound
to its earlier probe source, not counted as final-source evidence.

Warnings-denied compilation uses the cached JBR compiler with Java-17 target
bytecode; the initial Java-17 compiler cannot read pinned Java-21 dependency
classes and its failed log is retained. All eleven other provider classes are
byte-identical to 0.9.0. The existing guarded unlink body/helpers and default
generated unlink fixture source remain unchanged; existing default probe helpers
and CLI guards pass 22 static checks. Default legacy mutation journeys were not
rerun after this additive read capability; their historical bindings remain intact.

Production snapshot admission/transport and applied saved selection are not wired
by this provider-only leaf; 7.3 retains reconciliation and 7.4 full synchronization.
No second project index, resolver, executor, result store, generic panel or public
SDK is added. This is development capability evidence, not installed/default
delivery, physical native/remote, platform/resource, terms, full parity or release
qualification. Actions stopped, VoiceOver deferred and jvm planned. Durable
evidence: `.agents/evidence-archives/groovy-jvm/imported-build-kinds-20261004/`.

<!-- impl: tooling/jvm/gradle-model/ProjectUnlinkHandler.java#importedProjects -->

### Authoritative imported build-kind admission (7.38) — 2026-10-04

The existing JDTLS snapshot now carries a separate typed imported build kind
(unknown/Maven/Gradle), independent of Java/non-Java membership and optional
cached build metadata. Trusted local and current remote-host refreshes read only
advertised cached-only identity commands from bounded initialized LSP folders
confined to the worktree. Canonical/foreign/duplicate/oversized/conflicting and
Java-nature-invalid replies reject the generation before mutation. Explicit
partial/changed enumeration remains incomplete. Absent capability or known-empty
reads preserve unknown ownership without incorrectly downgrading otherwise
complete dependency data. No build file/model inference or new index/store occurs.

Snapshot protocol v4 preserves known Java/non-Java ownership through the existing
host privacy reduction and client admission; v1-v3 explicitly clear the new fact.
Invalid wire enums or known ownership without imported membership are rejected.
Existing generation/server/trust/revocation checks remain the owner, including
trust registry removal after dispatch. Tree rendering and execution are unchanged.
Selected unlink's stronger affirmative-ownership preflight is separately planned
in 7.39; applied selection/ignore/pattern/required-library/orphan policy stays 7.3.

**169 active checks pass; 11 ignores are not passes.** The corrected unified
library/Project integration build, 35 model, 20 UI, eight LSP, 37 settings,
startup/settings/trust integrations, 16 Copper provider, 19 shared store, one
sanitizer, 23 shared Tests and one framed-host checks pass with unchanged final
source/binary hashes. Repository warnings-denied Clippy, whole-workspace fmt,
complete canonical validators and manual ownership/decomposition audit pass.
Retain the temporary-capability borrow compile failure, old v4-rejection test
failure and real unknown-coverage mounted regression, plus their corrections,
initial missing failed-source-hash limit and existing compact-unwind/block notices.
The aborted text replacement wrote no Rust change; its unchanged-source build
is not final qualification. Exact commands/bindings and failure history are in
`.agents/evidence-archives/groovy-jvm/imported-kind-admission-20261004/`.

This is automated/source and mounted/framed fixture evidence, not native or
physical remote mutation, installed/default acquisition, provider terms,
resource/platform/accessibility, full synchronization/full parity or release
qualification. No editor/LSP lifecycle, extension setup, Tasks/terminals, Tests
store, DAP, remote permissions, catalog or generic panel owner is duplicated.
Actions remain stopped, VoiceOver deferred and jvm planned.

<!-- impl: crates/project/src/jvm_project.rs#JdtlsImportedBuildKind -->
<!-- impl: crates/project/src/jvm_project.rs#query_imported_build_kinds -->
<!-- impl: crates/project/src/jvm_project.rs#admit_imported_build_kinds -->
<!-- impl: crates/project/src/jvm_project.rs#project_to_proto_for_version -->

### Required Maven library replacement (7.40) — 2026-10-04

The pinned development provider 0.10.0 passes 21 strict headless checkpoints
across cached/missing required-artifact unlink and same-profile restart, plus
nine static/CLI checks. Actual M2E/JDT classpaths remove the required project
reference and expose a repository library entry; cached fixture bytes exist
and match, while missing bytes produce source errors. Both states survive
restart with startup imports disabled. Source/POM hashes and unrelated projects
remain intact. Entry presence is separate from artifact availability. All ten
existing probe definitions are unchanged; no product interpretation/parser was
added. The first ambiguous label and subsequent pre-oracle observations are
retained separately; the final source/provider bindings and strict receipts are
in `.agents/evidence-archives/groovy-jvm/required-library-20261004/`. The unique
fixture namespace remains absent from the cache seed, but no comprehensive seed
hash was captured. Fixture packaging is not a qualified Tasks install action.

This resolves one provider prerequisite for required-library ignore behavior.
Applied saved selection/ignore/POM-pattern/orphan policy, visible ignored entities
and actionable install/error controls remain 7.3/8.7; full causal sync remains
7.4. No mounted/native/physical remote/default delivery/terms/resource/platform/
accessibility/full-parity or release pass is inferred. GitHub Actions stopped,
VoiceOver deferred, jvm planned.

### Cached Maven container reader (7.41) — 2026-10-04

Development provider 0.11.0 reads cached M2E models beneath initialized container
folders without requiring a root POM. A regular root POM keeps rooted-reactor
traversal; an escaping root POM fails with a sanitized error. Container reads seed
only existing confined cached-facade POMs, never scan/import/resolve new files.
Bounded current cached models remain the sole source of modules/profiles/goals/order.

Pinned warnings-denied Java-17 compilation, **87 container/rooted v1/v2/v3
checkpoints**, **21 original-mode replay checkpoints** and **ten static/CLI checks**
pass. Eleven other compiled classes are unchanged from 0.10.0. Exact warm model
identities/order, explicit unavailable declared modules, foreign/escaping rejection,
read-only membership, source preservation and cached/missing library behavior are
retained with before/after source/provider hashes. On suppressed-startup restart,
Eclipse resources and classpaths survive but Maven models are cold/unavailable.
The read reports that limit explicitly; it does not invent metadata or qualify
full restart parity. The initial incorrect warm-cache expectations and corrected
cold facts are retained in the container-models-20261004 evidence archive; the first
failed reply was not captured.

This provider leaf does not wire automatic client container projection, scoped
hydration, saved selection/ignore/POM-pattern/orphan policy or full synchronization.
Those remain 7.3/7.4/8.7. Corrective 7.42 also retains server-side dual-nature unlink
rejection. Native/physical remote/default delivery/terms/resource/platform/accessibility/
full-parity and release gates remain open; Actions stopped, VoiceOver deferred,
jvm planned. No release or bundled redistribution claim.

### Guarded unique-provider unlink (7.42) — 2026-10-04

Development provider 0.12.0 requires the chosen Maven/Gradle nature and absence
of the other both before dispatch and under the existing Eclipse workspace rule.
Unknown or dual ownership cannot remove resources. No alternate index, resolver,
execution engine or result store is added. Only ProjectUnlinkHandler.class and the
manifest change; eleven other classes remain identical. Warnings-denied Java-17
compilation passes; SHA-256 8b237cec1c31b252cd36db2b207fc9c3548877cd2d9e96e14a312b30ae354577.

**79 mixed-provider checkpoints**, **60 existing Maven regression checkpoints**,
**34 existing Gradle regression checkpoints** and **11 static/CLI checks** pass.
Fresh/restart, Java/non-Java actual dual natures, all eight immediate opposite/
chosen-provider rejections, four held-rule ownership changes, sanitized errors,
exact source-preserving unlink/repair, unrelated identities, diagnostics, user
cancellation, replaced roots and successful owned-Gradle-worker cleanup are
retained. Existing default/identity fixture payloads remain byte-identical;
before/after source/provider bindings match.

The wrong initial Equinox-jar launcher invocation and later mixed-existing-Maven
repair failure are retained in the dual-unlink-20261004 immutable archive (69
entries; SHA-256 8b07083aa0f88b9e9e0297669c401dd6945e3939abf49c78857678b879c9bc8d).
The final recovery selects only the removed Maven POM; repeated/current/mixed
hydration remains corrective 7.43, not a passed parity journey. The first launch
lacks full command/source bindings. This does not qualify client saved selection/
ignore/pattern/orphan policy, visible mutation controls, causal synchronization,
native/physical remote/default delivery/terms/resource/platform/accessibility or
full parity. Shared client 7.39 remains unfinished. Actions stopped, VoiceOver
deferred, jvm planned; no release or redistribution claim.

### Scoped selected Maven hydration (7.43) — 2026-10-04

Development provider 0.13.0 imports only missing selected resources and updates
exact current selected Maven projects through M2E's supported per-project
configuration API under its existing workspace rule. The existing unique-provider
unlink predicate is reused within the internal package; there is no new public SDK,
resolver, project cache/index, execution engine or result store. Selected POM bytes,
root, offline setting and ownership are rechecked around the held operation. Pinned
bytecode verifies host offline mode and no forced dependency updates. Events describe
new membership only; exact selected POM receipts retain full synchronization unknown.

Warnings-denied Java-17 compilation, **183 hydration/model/library checkpoints**,
**60 existing Maven regression checkpoints** and **14 static/CLI checks** pass.
Artifact SHA-256 4801c728270c2dc6b319d59eaf6bacc14584a71de7349888872c62f0813e98d1.
Only MavenImportHandler.class, ownership-predicate package visibility and manifest
change; other provider class payloads match 0.12.0. Repeated and mixed selections,
cold remaining-model hydration/restart, held cancellation/root/nature/POM changes,
unavailable parent/model failure and recovery preserve unrelated identities and
sources. Success hydration is qualified in the cached-required-artifact cell; the
separate missing-artifact cell retains read/classpath/source-error observations.

The deprecated request-based API build, immediate post-cleanup cached-model read
failure and incorrect cancellation-only ownership oracle are retained. A recorded
two-second settling interval precedes the later successful cleanup read; it does
not qualify immediate synchronization. Final source hashes match before/after and
all three existing fixture generator payloads remain byte-identical. Earlier
prototype bindings and failed-reply capture retain their limits. Immutable archive:
maven-hydration-20261004, 86 entries, SHA-256
f98d6f7bea06db94e3dd7a00713efbf03b18df2350ae3b3383c8011fab9af0a2.

Root-reactor automatic expansion, saved selection/ignore/POM-pattern/orphan policy,
causal synchronization, visible mutation controls, native/physical remote, default
provider delivery/terms/resource/platform/accessibility/full-parity/release gates
remain open under existing owners. This phase never broadens exact selection or
claims complete dependency resolution from a populated model. Actions stopped,
VoiceOver deferred, jvm planned; no release or server redistribution claim.

### Current ownership and explicit read refresh (7.39) — 2026-10-04

Selected local Maven/Gradle unlink now requires affirmative current chosen-provider
ownership for Java and non-Java resources, at existing preflight and held-phase
boundaries. Unknown or conflicting kinds reject without a mutation command and give
a sanitized setup/refresh action. Existing guarded server validation stays mandatory.
Explicit refresh starts a new Java/non-Java read even when the old snapshot was
current, exposes pending/stale state immediately and leaves held mutation/cancellation
ownership intact. Only already-current same-server Gradle configuration facts may
survive for freshly observed unchanged module membership/kind; the existing privacy/
source filter is reapplied and no Gradle configuration scripts run from read refresh.

**96 active checks pass; 13 ignored checks are not passes**: 20 mounted Groovy,
eight LSP, sixteen Copper provider, nineteen shared structured-store, twenty-three
Tests, one shared initialization-options and nine remote-owner checks. Unified
test compilation, repository Clippy, whole formatting and source/binary hash checks
pass after normal integration of origin/dev through 5bed0cc758. The macOS test
linker emits compact-unwind size warnings; performance/platform readiness is not
inferred. Remote fixtures prove logical host trust/permissions/framed data and a
local TCP stub, not physical remote or a downloaded production Debug adapter.

The initial metadata-loss regression, wrong held-query oracle, invented test API,
read counter error and sandbox-denied TCP fixture remain retained. The final normal-
host TCP retry passes with unchanged artifacts; Java/non-Java refresh actually sends
two enumeration commands. Earlier Cargo artifacts disappeared and unchanged
packages rebuilt; no cleanup source is attributed or unrelated files deleted.
Immutable archive unlink-ownership-20261004: 69 entries, SHA-256
420d3cf2854c4e60a30a1ddd4d24365a27025f3c101cfdffc064eb279a9ab242.
Final Rust bindings are merge 3fa2c1b57493d0e8fb7933419ad0ac1558d36ab4 plus the
retained source patch, with matching hashes through all final checks.

This closes one existing local authorization/read-refresh correction. Saved policy,
visible link/unlink controls, root expansion and causal synchronization remain
7.3/8.7/7.4. Native/physical remote/default delivery/terms/resource/platform/
accessibility/full-parity/release gates remain open. No second framework, Cargo
mapping, cache, index, execution engine or result store is added. Actions stopped,
VoiceOver deferred, jvm planned; no desktop candidate launched or release published.

M05 root-link clarification (2026-10-04): exact selected-POM hydration 7.43 is
complete as that phase; explicit effective reactor expansion is qualified in 7.44,
with automatic client saved-policy application still 7.3. It reuses the same M2E import/configuration owner
and does not introduce a parallel provider SDK/parser/index/engine. M05 remains
incomplete, with full parity still required and all release gates unchanged.

### Scoped effective Maven reactor import (7.44) — 2026-10-04

Development provider 0.14.0 adds distinct internal `groovy.maven.importReactor`:
initialized root URI, one root POM URI, exact existing excluded directory URIs and
explicit offline=true. It preserves `importSelected`'s exact 32-POM contract. M2E's
supported execution context reads effective modules with host settings/active
profiles, offline enabled and snapshot updates disabled. Iterative traversal
follows only declared modules, with 128 POMs, 1 MiB per input and 8 MiB aggregate
input bounds; no filesystem-wide scan/shared XML parser/cache/index/SDK. Duplicate,
cyclic, escaping, missing and oversized inputs fail before import. Exact excluded
subtrees are skipped before model descent; existing imported resources are not
implicitly unlinked, and missing exclusion directories currently reject.

The resulting POMs reuse the same existing guarded workspace import/configuration
phase. Current root, POM bytes, exclusions and unique Maven ownership are checked
around its workspace-rule wait; cancellation and sanitized partial failures are
retained. Receipts add the reactor and exclusion identities to exact admitted POMs,
report phase completion and keep full synchronization unknown. External parent/
settings generations remain a 7.4 causal synchronization boundary. Model reads may
load project extensions/resolve offline models and require existing client trust/
permission authorization as an explicit phase; they are not passive cached reads.

Warnings-denied Java-17 compilation, **40 reactor checkpoints**, **183 exact selected-
POM regression checkpoints** and **18 static/CLI/pinned-input checks** pass with
external network denied. Fresh/repeated/mixed-current/missing/cold-restart journeys
cover nested and parentless modules, host-active versus inactive profiles, ignored
subtrees, unrelated roots, required-library cached artifact replacement, original/
generated source hashes, count/byte/path/symlink/missing-model failures, held-rule
cancellation/root/ownership/POM changes and recovery. Replacement classpath delay is
recorded, not immediate synchronization. Only the Maven import class, manifest and
command registration differ from 0.13.0; eleven other provider classes match, so
shared Rust/Copper source and the prior 96 active regression results are unchanged.
The first deprecated execution API build failure and initial smaller successful
fixture remain retained, not counted as additional final checks. Source/binary
hashes match before/after final qualification. Immutable archive maven-reactor-
20261004: 40 entries, SHA-256 84745a453e70afa8adcdb0a9924cc24ed438fe540bd22d9d4c710a2143e68591.
Provider SHA-256 7b6c7a61054b131bcb19c03fcfb6ed8e901f4fbe3f1ba15b4ed20ad543bb4976.

This closes one provider reactor-root phase. Applied saved selection/ignore/pattern/
orphan policy, visible controls, causal reimport, native/physical remote, default
provider delivery/terms, Kotlin resource/platform/accessibility, production packaging,
full parity and release remain open under existing owners. Actions stopped;
VoiceOver deferred; jvm planned. No new desktop candidate launched or release published.

### Guarded local reactor client — 2026-10-04

Tasks 7.45/7.46 admit the distinct reactor phase through the existing local JVM
model/LSP lifecycle. Strict bounded root/exclusion/POM receipts and current visible
regular worktree identities, trust, server/root/generation, held progress and
cancellation are verified; private/missing/foreign/excluded expanded inputs reject
success with sanitized partial errors and retained stale facts. Exact Maven/Gradle
import/unlink APIs remain compatible. No shared project model, second index/result
store/executor/permission framework or remote mutation RPC is added.

Final source-bound no-run build, repository Clippy and formatting pass. **135 active
checks pass**, with **13 opt-in checks ignored**: 22 mounted JVM UI/lifecycle, 37
model/transport, eight LSP, sixteen Copper provider, nineteen structured-result,
23 Tests, one shared initialization-options and nine logical remote checks.
Initial E0308 lookup failure was fixed; the sandbox TCP startup failure is retained
and the same hashed remote binary passes on authorized host retry. Apple compact-
unwind linker and pre-existing block future-compatibility warnings are retained;
these checks do not qualify performance/platform behavior. Source and executable
hashes match before/after. Immutable reactor-client-20261004 archive: 28 entries,
SHA-256 be3c9bd5f3af599da5c04135be1fe1062d644067d379655529cc2f0b2a628d3a.

Saved selection/ignore/pattern/orphan application, visible controls, causal sync,
native/physical remote/default delivery/terms/Kotlin resources/platform/accessibility,
production packaging/full parity/release remain open. Actions stopped; VoiceOver
deferred; jvm planned. No desktop candidate launched or release published.

### Saved Maven policy source evidence — 2026-10-04

Tasks 7.47/7.48 now project explicit saved Maven reactor roots and ignored
subtrees into one existing local LSP phase task/token. Current complete ownership
is required for exclusions; only affected Maven resources are unlinked. Gradle,
unmentioned/orphan resources and sources remain intact. Effective user/project
settings, trust, current server/root/input and expanded POM visibility are checked
between commands and after held replies. Explicit empty selection imports nothing;
absent defaults remain provider-owned; ignored-only defaults reject until bounded
provider discovery is qualified. Exact selected-POM APIs remain unchanged.

The final source-bound build, repository Clippy and formatting pass. **139 active
checks pass; thirteen ignored checks are not passes**: 23 JVM UI/model/lifecycle
checks (the new policy fixture mounts Project/LSP/settings entities, not a native
panel/window), 39 JVM model/transport, eight LSP, sixteen Copper provider, nineteen
structured-result, 23 Tests, one shared initialization-options, one shared settings
and nine logical remote checks. The new policy test also passes twenty GPUI
scheduler replays, counted separately. The loopback adapter initially fails under
the sandbox; the identical hashed binary passes the authorized host retry. This
is logical transport evidence, not physical remote qualification.

Retained failures include disk-full dependency compilation, missing ReadGlobal
trait import, a mis-scoped test-build feature argument and the unregistered Java
fixture causing forbidden parking. These were corrected before final validation.
Review also found expanded-input visibility had to be checked before the next
reactor command; the final negative fixture proves this guard and recovery.
Final source/executable hashes match before/after all final checks. Apple compact-
unwind and pre-existing block future-compatibility warnings remain; no performance
or platform inference. No new index/store/executor/permission framework or Cargo
model is introduced. All-provider startup/restart/default/pattern/orphan policy,
visible controls, full causal synchronization, native/physical remote/default
provider delivery/terms, Kotlin resource/platform/accessibility, production
packaging/full parity/release remain open. Actions stopped; VoiceOver deferred;
jvm planned. No desktop candidate launched or release published.

Immutable maven-saved-policy-20261004 archive: 51 entries, SHA-256 8742125948a6d9a0d2030404a980076b5e2424413f40d87f386d54787aeaa16a.

### Explicit Maven startup replay and shared recovery evidence — 2026-10-04

Task 7.49 captures bounded effective Maven intent on the existing local Running
JDTLS server. The first admitted metadata snapshot consumes it once through the
existing saved phase owner. Startup/restart use current settings; changed captured
intent, private inputs and partial ownership reject before mutation. Empty intent
imports nothing. An admitted manual saved application supersedes pending startup;
ordinary reads never repeat completed or failed imports. Teardown discards the
marker. Provider defaults, fresh known-empty ownership, Gradle replay and complete
settings-change/default/pattern/orphan policy remain open under 7.50/7.51/7.3/7.4.

Final source-bound build, repository Clippy and Rust formatting pass. **140 active
checks pass; thirteen opt-in checks remain ignored, not passed**: 24 JVM UI/model/
lifecycle, 39 JVM model/transport, eight LSP, sixteen Copper provider, nineteen
structured-result, 23 Tests, one shared initialization-options, one shared settings
and nine logical remote checks. Both Maven policy tests also pass twenty scheduler
seeds each (forty replays counted separately). Final source and six executable
hashes match before/after all final checks. Checker invocation timestamps follow
the frozen changed source. These are GPUI entity/protocol checks, not new native
window or physical remote qualification.

The first focused run failed both tests: direct cancellation dropped the task
before its completion callback could record import failure. The common JDTLS
Maven/Gradle phase owner now records failed_import before cancellation, retains it
after command failure, discards pending Maven startup intent, and prevents ordinary
provider progress from clearing the error. The existing Dependencies status gives
sanitized JVM build settings/server recovery guidance. Explicit metadata refresh
may recover readable facts but does not certify the failed import. Exact phase
command/receipt contracts and separate Kotlin status behavior remain intact.
Intermediate positive checks are retained as diagnostics; final counts use only
the frozen source run. Loopback TCP startup failed in the sandbox; the identical
hashed remote executable passed all nine on the authorized host retry. Fixture
error/denial canaries are expected, not product credentials. Apple compact-unwind
and pre-existing block future-compatibility warnings remain, with no performance
or platform inference. Build space was reclaimed only from five unused generated
old test executables matching retained hashes; cleanup receipts are retained and
the fresh build regenerated them. Source, archives and approved candidates were
preserved. No JVM workers, native candidate or release were launched.

No new executor/index/result store/permission framework/Cargo model/remote mutation
RPC is added. Provider delivery/terms, full synchronization and visible controls,
native/physical remote, Kotlin resource/platform/accessibility, production identity/
packaging, full tool-window parity and release remain open. Actions stopped;
VoiceOver deferred; jvm planned.

Immutable maven-startup-replay-20261004 archive: 37 entries, SHA-256 d23ac6bf8d02761b9e151f446fe341fe2d7c6c755dd793587db17cfaf7c536c1.

### Current Maven ownership coverage evidence — 2026-10-04

Tasks 7.50/7.51 distinguish known-empty imported ownership from unavailable or
partial enumeration. Existing JDTLS queries return an ephemeral bounded scope,
count and coverage receipt; the local Running server retains its query token and
header. Saved Maven exclusions require matching live server/worktree/root,
initialized-folder scope, snapshot generation and permission-admitted imported
count and live permission visibility. Both cached ownership providers must answer completely. Missing commands,
unknown ownership, malformed/duplicate scope and hidden imported resources reject
before mutation. Semantic metadata remains partial when unknown; a complete
ownership inventory does not certify source/dependency freshness or full sync.
Configuring Gradle metadata reads invalidate the earlier ownership receipt.

The existing GPUI lifecycle fixtures prove fresh known-empty startup and manual
application, once-only execution, unavailable/missing-provider/private-inventory
rejection and live revocation, incomplete semantic metadata with complete ownership, restart and
explicit recovery. Existing settings/phase/trust/cancellation and provider
contracts are reused. No project facts are duplicated, and no index/store,
executor, generic panel/Cargo model, remote mutation RPC or public provider SDK
is added. The opaque coverage type is public only to fit the existing public
Running enum; its fields and scope API remain internal.

Final-source builds, warnings-denied repository Clippy and formatting pass.
**144 active checks pass; thirteen opt-in checks remain ignored, not passed**:
26 JVM UI/lifecycle/model, 41 JVM model/transport, eight LSP, sixteen Copper,
nineteen structured-result, 23 Tests, one shared initialization-options, one
shared settings and nine logical remote. Scheduler replays are recorded separately
in the manifest. Final source and six executable hashes match before/after checks.
These are automated entity/protocol and existing mounted tests, not fresh native
window or physical remote qualification.

The first build failed because the explicit configuring Gradle consumer still
expected the old query result; the repaired consumer also invalidates earlier
coverage. The default-target build passed but did not select disabled-by-default
project unit tests; explicit library selection supplies those checks; the final combined library/test
build includes the live-permission correction and the current integration executable. Initial loopback TCP
sandbox denial is retained; the identical hashed remote binary passes the
explicitly authorized host retry. Apple compact-unwind and pre-existing block
future-compatibility notices remain, with no performance or platform inference.
Only four unused generated old test executables matching retained hashes were
removed for build space; the newer integration output, source, archives, native
candidates and profiles were preserved. No JVM worker or native app was launched.

All-provider/default/pattern/required/orphan policy (7.3), causal sync (7.4), visible
controls, provider delivery/terms, native/physical remote, Kotlin resource/platform/
accessibility, production identity/packaging and full parity/release gates stay
open. Next Gradle projection 7.52 is planned, not implemented. Actions stopped;
VoiceOver deferred; jvm planned. The official provider-source recheck corroborates
free direct/extension Kotlin LSP evaluation but does not settle exact standalone
terms/notices or redistribution.

Immutable maven-ownership-coverage-20261004 archive: 42 entries, SHA-256 28fd84e196de456b0c8c13f56c132b22b35473b43d307dc4ea299693ca8a63ee.

### Local Dock and Gradle admission evidence — 2026-10-05

G07/G09 remain incomplete: bounded current-root saved Gradle policy and unsupported-selection refusal pass, while full defaults/nested/exclusion/required/orphan/pattern/startup policy and controls remain open. S01 remains incomplete: existing Left/Bottom/Right live movement passes shared/Groovy/Copper mounted focus/size checks; position persistence is planned 8.9, and top/float/window/undock/pin modes plus native/restart evidence remain open.

[Immutable evidence](../../../.agents/evidence-archives/groovy-jvm/shared-dock-position-20261004/README.md) retains 163 active checks, thirteen ignored, thirty separate scheduler replays, source/compiler/eight executable hashes and failed attempts/repairs. Build/Clippy/format/canonical/manual decomposition checks pass. Automated proof does not qualify visual/native/physical remote/platform/VoiceOver, full parity or release. GitHub Actions is stopped, VoiceOver deferred and jvm planned.

### Dock persistence and shared LSP removal — 2026-10-05

Existing KVP schema 4 preserves legacy presentation settings and restores Left/Bottom/Right. Rapid/idempotent movement, focus, entity reload, held panel-release save, missing identity and sanitized write-failure recovery pass. The existing Drop owner is reused. Shared LSP removal now admits enabled Boolean/registration-ID notifications, preserves disabled/absent membership and unrelated folders, and suppresses repeat events. Two pre-fix failures are retained; no folder lease/index/queue/provider engine is added.

[Immutable evidence](../../../.agents/evidence-archives/groovy-jvm/dock-position-preferences-20261005/README.md): 195 active checks pass, thirteen opt-ins remain ignored, and thirty scheduler replays are separate. Final source/eleven executable hashes, build, warnings-denied Clippy, formatting and canonical validation pass. Two initial Copper process deadline failures remain recorded; unchanged fixtures pass on the normal host without cause attribution. The fresh generated check cache/clone is not qualification.

S01/G07/G09/M05 remain incomplete. Saved Gradle startup is planned 7.55; full link/ignore/default/nested/pattern/orphan policy, causal synchronization, actions/configurations/graphs/modes and Debug parity remain open. Native process restart/physical remote, VoiceOver, delivery/terms/pins, alpha memory/platforms, stopped CI, provisional identity/packaging and release gates are unchanged. No native app/JVM worker/unverified adapter launched; jvm remains planned.

### Saved Gradle startup — 2026-10-05

The existing Running JDTLS/read/phase owner now consumes captured controlled Gradle root intent once after a current admitted read. Mixed Maven/Gradle intent serializes through the same stored phase and follow-up read; empty Maven intent falls through. Failure/cancellation/read failure or revoked trust discards pending replay, and ordinary recovery reads do not retry mutation. Current settings, missing input, effective removal/restart and explicit recovery pass initialized fake-provider fixtures. No new queue/index/store/executor or Cargo JVM model is introduced.

[Immutable evidence](../../../.agents/evidence-archives/groovy-jvm/gradle-startup-replay-20261005/README.md): 158 active selected checks pass, eleven opt-ins remain ignored and forty scheduler replays are separate. Source/executable hashes, current test builds, Clippy, formatting and canonical/manual audits pass. The corrected feature-selector rejection, retained linker/future-compatibility notes and generated-cache observations remain explicit.

G07/G09/M05 remain incomplete. Root-only saved Gradle startup is automated evidence, while full defaults/nested/exclusion/pattern/required/orphan policy, controls, causal synchronization, actions/configurations/graphs/modes and Debug parity remain open. Planned 7.56 obtains actual Gradle required-library classpath evidence; Maven replacement cannot establish it. Native editor restart, physical remote/default delivery/terms/notices/pins, Kotlin alpha memory/platform, VoiceOver, stopped CI, production identity/packaging and release gates remain open. No native app/JVM worker/unverified adapter launched; jvm remains planned.

### Startup post-merge checks — 2026-10-05

Safe conflict-free integration of origin/dev ce5b565d8f produced d996b95c472fc2b90aa113860ece806767998955; the Groovy startup sources and used test executables remain unchanged. All 158 affected active checks pass again with eleven opt-ins ignored. Warnings-denied affected repository Clippy, whole formatting, both canonical validators and manual ownership review pass. The remote collaborative UI/workflow-definition changes are preserved; no Actions poll/dispatch/enable operation occurred, and CI gates remain infrastructure-blocked, not passed. No native candidate or JVM worker launched; VoiceOver deferred and jvm planned. Archive 16 entries, SHA-256 8d0a112551a644c1fc211cd5071da6761954851bcf04eceea6785fc96a33c2b6.

[Immutable post-merge evidence](../../../.agents/evidence-archives/groovy-jvm/startup-postmerge-ce5-20261005/README.md). Local checks do not qualify native restart, physical remote, full parity or any release gate.

### Gradle required-library qualification — 2026-10-05

2026-10-05: 18 guard/observation checks across four real pinned JDTLS/Buildship cached/missing-output unlink and suppressed-startup restart cells complete, but required-library replacement fails in every cell: the exact application project reference remains, no prepared binary replacement appears, source errors remain and the removed library stays absent. The prepared Jar contains byte-identical compiled Required.class. All fifteen pinned provider/tool/JDK inputs and final source bindings match; source/generated hashes and no-test-report assertions pass. Existing guarded-unlink mode passes all 34 compatibility checks. Owned Gradle homes stop successfully; no unrelated worker is stopped. Initial preparation loopback failure and corrected IPv4 attempt are retained separately. Immutable gradle-required-library-20261005 archive: 61 entries, SHA-256 3aa700a92bcc2ea88ff8febdf037d7aad3a5d87625f01cf43aa8f7044ebff64c. This closes provider qualification only; 9.2/7.3 required-library parity remains unmet. Rust/Copper sources are unchanged from bfb17f6b09801aa110e5d4d129b88585b47c95dd and its retained 158 selected active post-merge checks, eleven ignores and Clippy evidence. Native/remote/default delivery/terms/resource/platform/full parity/release remain open; Actions stopped, VoiceOver deferred, jvm planned.

[Negative provider evidence](../../../.agents/evidence-archives/groovy-jvm/gradle-required-library-20261005/README.md). G07/G09 remain incomplete: source-preserving unlink succeeds, but ignored required modules have no binary replacement, including with verified cached outputs. A separate provider correction remains required. The four observation cells and old-mode compatibility do not qualify native workflows, Tasks execution, remote ownership or release readiness.

### Authoritative Gradle output declarations — 2026-10-05

2026-10-05: development provider 0.15.0, SHA-256 070023d5149700c0abd758291bd0a8098d0e6d2a434c24e7c0d0179efd6cb7ad, compiles with warnings denied to Java-17 bytecode using the existing Java21-capable JBR compiler required by pinned Eclipse dependencies. 33 denied-network real JDTLS/Gradle checkpoints across two same-profile journeys pass: custom/standard/missing/present output declarations, non-Java known-empty sets, stable identities, strict requests/confinement/symlink/count/text rejection, sanitized failure/repair, in-flight cancellation, cached-home refusal/recovery and temporary-script cleanup. All 34 existing guarded-unlink compatibility checkpoints pass. No output-producing task, configuration resolution or test reports are observed; fixture source hashes and all sixteen provider/tool/compiler inputs match. Six unrelated compiled provider classes remain byte-identical. Initial Java17 compiler mismatch and sandbox JVM/cleanup denial are retained separately from host success. Owned Gradle homes stop successfully. Immutable gradle-output-declarations-20261005 archive: 50 entries, SHA-256 e24156e13ee9ae84f5459f0954e73cf82eeb589d0ef4d9945ca7ca3d7617271a. Output provenance does not qualify artifact/variant selection, classpath replacement, saved ignore, full sync, UI/remote/native/default/terms/resources/platform/release. Shared Rust/Copper code remains unchanged and source-current bfb17f6b checks apply. Actions stopped, VoiceOver deferred, jvm planned.

[Provider evidence](../../../.agents/evidence-archives/groovy-jvm/gradle-output-declarations-20261005/README.md). G07/G09/G14 remain incomplete. Authoritative source-set output declarations are now available through an explicit internal provider query, with stable missing/present identities and per-build confinement. This does not repair the negative 7.56 required-library classpath result, choose variants, add a product UI reader or establish an ignore operation. Full reference scope and release gates remain unchanged.

### Explicit Gradle resolution graphs — 2026-10-05

2026-10-05: pinned development provider 0.16.0, SHA-256 1e174a2b37b4511d23b5ec6baeec5f6e8bac023baadc171c311890cd8f49eac6, compiles with warnings denied to Java-17 bytecode using the existing Java21-capable compiler. Final denied-network offline fixture passes 37 graph checkpoints across two same-profile journeys: real diamond 1.0/2.0 conflict selecting 2.0, rich preferred/rejected constraints with private reason descriptions omitted, cycles, main/included project identities, selected variants, sanitized unresolved edges, strict module/configuration/offline requests, oversized component/edge/text fixtures, failure/repair/cancellation, cached-home refusal/recovery and temporary-script cleanup. The response budget is enforced before large JSON serialization and may reject oversized fixtures before count ceilings. All 33 output-only and 34 guarded-unlink compatibility checkpoints pass; sources and sixteen pinned inputs match, seven unaffected class payloads remain identical. No fixture artifact Jar, classes or test reports are produced; only owned Gradle homes are stopped. Initial constraint-assertion failure, ENOTEMPTY cleanup failure and an unrecorded foreign-membership rejection during relink remain retained. A bounded logged retry repairs only ENOTEMPTY fixture cleanup; foreign-membership rejection stays strict, gains bounded URI evidence and is not reproduced by the final run. Its cause is not attributed, and full causal sync/native/remote remain open. Immutable gradle-resolution-graph-20261005 archive: 132 entries, SHA-256 2c93f3446d4de9c1747570a131148ec434d940fc3e0fa9e4d4f3a9d90e4fde72. This closes the internal Gradle provider contract only: artifact readiness/origin ranges, Maven graphs, current product admission/tree analysis, full parity/default/terms/platform/native/remote/release remain unqualified. Shared Rust/Copper code is unchanged. Actions stopped, VoiceOver deferred, jvm planned.

[Provider evidence](../../../.agents/evidence-archives/groovy-jvm/gradle-resolution-graph-20261005/README.md). G26/G27/G28/G30 remain incomplete. Real resolution metadata now has an explicit bounded internal Gradle contract (10.7), but product current-generation admission/shared-tree analysis, artifact readiness/source origins and corresponding Maven graph evidence remain missing. The earlier unrecorded foreign relink membership and its strict successful retry remain distinct evidence; causal synchronization is not certified. Required-library fallback, physical workflows, distribution and release gates remain open.

### Explicit offline Maven dependency graphs — 2026-10-05

2026-10-05: pinned development provider 0.17.0, SHA-256 8295bfd61ff4013ea183adb821d16293cb45f6d174c3fad6a84beb122b79fd20, builds with warnings denied to Java-17 bytecode. Final denied-network offline Maven fixture passes 34 graph checkpoints across initial and same-profile journeys; all 45 existing cached Maven metadata checkpoints pass (79 total). Real nearest/first diamond conflicts, actual winner references, dependency-management prevalues, direct scopes/optionality, exclusions, cycles and workspace-module metadata pass. Seven strict/offline/foreign/unimported/symlink request guards, synthetic edge/component/text/depth/deadline limits, pre-cancellation and in-flight projection/traversal monitor cancellation, stale facade rejection, explicit M2E import repair and missing-descriptor partial coverage pass. The existing public M2E registry execution owner supplies the configured workspace reader; the facade context alone did not. A clone-only strict descriptor policy repairs the observed default policy falsely complete missing-POM case. Source/generated/POM bytes are preserved and no fixture Jar/test report is acquired; sixteen unaffected compiled provider classes and all 12 existing probe definitions remain identical. Eleven compile-input pins and three runtime-tool hashes are retained; source bindings match. Restart explicitly hydrates selected cached POMs through existing import; an earlier cold-model timeout remains retained and automatic startup hydration is not certified. Initial fixture compilation/registration/evidence-cleanup failures, stale immediate-read failure and intermediate coverage failures are retained. Immutable maven-resolution-graph-20261005 archive: 136 entries, SHA-256 0cb54056b086a4ce54eb900ec034162913e4ce88b30f8a840739e2729cf2c45d. This closes the internal Maven provider contract only. Product generation admission/tree analysis, source origins/artifact readiness, physical/native/remote/default/terms/platform/full parity/release remain open; prompt cancellation of arbitrary blocking host work is not qualified by synthetic guards. Shared Rust/Copper sources are unchanged and prior affected-owner regression evidence remains source-current. Actions stopped, VoiceOver deferred, jvm planned.

[Provider evidence](../../../.agents/evidence-archives/groovy-jvm/maven-resolution-graph-20261005/README.md). M20/M21/S04 remain incomplete. Gradle and Maven now expose separately qualified internal graph contracts, while current product admission, shared-tree filters/paths/navigation, source origins, artifact readiness and remote/native workflows remain pending. No full parity gap is waived; stopped Actions, deferred VoiceOver, Kotlin alpha resource/platform qualification, production identity/packaging and provider distribution gates remain open.

### Maven dependency occurrence context — 2026-10-05

2026-10-05: pinned provider 0.18.0, SHA-256 f3a2579fb73c05454f5353e2acd83d5538d44b23709e3aa6295879528abe7ce5, builds with warnings denied. Maven graph response v2 retains actual DependencyNode occurrences, ordered children, deterministic path-derived IDs and exact selected-winner occurrence references alongside coordinate components. Real denied-network offline cached fixtures pass 45 graph checkpoints across two same-profile journeys plus all 45 existing Maven metadata checkpoints (90 total). The first unexcluded branch retains a leaf; the later same-coordinate excluded/omitted branch has a distinct occurrence with no children and points to the actual first winner. Repeat/restart occurrence identities and component/winner references match. Cut-cycle preceding/cyclic descriptors are retained separately with cycle occurrence mapping explicitly unknown; no closing node edge is invented. Unknown winner rejection and existing component/edge/text/depth/deadline/cancellation/stale/offline/partial/recovery guards pass. Sources/generated/POM bytes are preserved; no fixture Jar/test report is acquired. Sixteen unaffected provider classes, twelve existing probe definitions, eleven compile pins and three runtime-tool hashes match. Source hashes agree before/after the final run. The v1 missing-occurrence baseline is retained; the initial exclusion-order and cycle-edge assumptions failed and were corrected to the actual Resolver semantics. Immutable maven-occurrence-context-20261005 archive: 58 entries, SHA-256 4c3f7454c046c98a55765f705fe45b660f2045a37809c1ce85d9bd9b29682d86. Completed 10.8 and its v1 archive are preserved. This closes the provider projection correction only: current product admission, shared-tree paths/analysis, remote/native/default, exact source origins/artifact readiness, arbitrary blocking-work cancellation, terms/platform/full parity/release remain open. Shared Rust/Copper sources are unchanged. Actions stopped, VoiceOver deferred, jvm planned.

[Corrective provider evidence](../../../.agents/evidence-archives/groovy-jvm/maven-occurrence-context-20261005/README.md). M20/M21/S04 remain incomplete. Product admission must preserve coordinate versus occurrence identity and the explicitly unknown cut-cycle node mapping; shared-tree paths/filters, remote/native workflows and exact origins remain pending. No smaller first-release substitute or full-parity waiver is inferred.

### Maven graph decoder/query boundary — 2026-10-05

2026-10-05: Maven-specific v2 project decoder and exact offline/cachedOnly LSP query pass five focused checks and 91 unique active project-owner checks (48 JVM model including those five, eight LSP, sixteen Copper and nineteen structured-result); one existing Copper opt-in stays ignored. Twenty additional deterministic query scheduler replays pass separately. The captured real provider fixture retains 15 occurrences, 14 components, 14 edges and one cut cycle; only rootUri/pomUri are normalized, with the original receipt bound by hash. Binding/version/offline/coverage, uniqueness, coordinate/node/winner references, contiguous child positions, reachability, cut-cycle descriptors, fixed partial categories, text/count/depth/byte budgets and private-error sanitization pass. Advertised-command routing, missing result, provider exception and malformed result fail safely. Final source/fixture/executable hashes match. Repository all-targets/all-features project Clippy and formatting pass. The redundant test clone failure and interrupted redundant JVM-only build remain retained; linker unwind-size and upstream block future-compatibility warnings are recorded. Immutable maven-graph-admission-20261005 archive: 38 entries, SHA-256 0214e084ce33d604e19134f6afef4bdd0b37158d256625afdbad9e781738def6. Provider Java sources and 0.18.0 binary are unchanged; no GUI app, real build worker, output task, artifact download or CI run is started. This closes only response/query validation; selected local snapshot admission (10.11), Gradle admission, remote transport, shared-tree analysis, origins/artifacts, native/resource/platform/terms/packaging/full parity/release remain open. Actions stopped, VoiceOver deferred, jvm planned.

[Decoder evidence](../../../.agents/evidence-archives/groovy-jvm/maven-graph-admission-20261005/README.md). M20/M21/S04 remain incomplete; no current graph product/tree/remote/native or full-parity claim is made.

### Selected Maven graph lifecycle — 2026-10-05

2026-10-05: the selected local Maven graph query uses existing trusted imported ownership, JdtlsProjectState generations and stored refresh tasks. Immutable graph payloads share the existing Maven snapshot through Arc. Ten focused lifecycle cases and 135 unique owner checks pass: 43 JVM panel/lifecycle, 48 JVM model, eight LSP, sixteen Copper, nineteen structured-result and one shared JVM settings; one existing Copper opt-in remains ignored. Two hundred additional scheduler replays pass (ten cases, twenty iterations each). Current admission, advanced ownership receipt/repeat query, unavailable command, sanitized failure/coordinate mismatch, explicit recovery, owned cancellation, input change, replacement server, trust revocation and same-worktree folder invalidation pass. An unrelated worktree folder correctly preserves scope. Startup/read/restart never collect a graph automatically. Existing visibility filtering is reapplied at admission. Initial folder fixtures confused fake versus running provider IDs and same-worktree versus unrelated scope; failed runs are retained and corrected without weakening product ownership. Initial settings filter selected zero tests; the exact settings rerun passes. A harness substring assertion misread ten passed; numeric parsing confirms successful scheduler output. Final source/fixture/executable hashes match; combined JVM/Copper compilation, repository all-targets/all-features project/jvm_ui Clippy and formatting pass, with existing linker/upstream future-compatibility warnings retained. Immutable maven-graph-lifecycle-20261005 archive: 35 entries, SHA-256 422138651fe83fe2b26627c592118d5174f916dc0182a1e9b5108f47106e5806. Provider 0.18.0 is unchanged. No GUI app, real JVM/build worker, download or CI run starts. Remote graph transport, shared-tree analysis, actual origins/artifacts, physical cancellation/native/resource/platform/terms/packaging/full parity/release remain open. Actions stopped, VoiceOver deferred, jvm planned.

[Lifecycle evidence](../../../.agents/evidence-archives/groovy-jvm/maven-graph-lifecycle-20261005/README.md). M20/M21/S04 remain incomplete; local automated graph admission does not qualify tree/remote/native or full parity.

### Gradle graph response/query boundary — 2026-10-05

2026-10-05: the Gradle-specific selected configuration decoder/query passes five focused checks and 140 unique owner checks: 43 JVM panel/lifecycle, 53 JVM model, eight LSP, sixteen Copper, nineteen structured-result and one shared JVM settings; one Copper opt-in remains ignored. Twenty deterministic query scheduler replays pass separately. The captured real 10.7 provider fixture changes only outer absolute URI bindings; graphJson bytes retain eight components, eleven edges, conflict/constraint selection, rich version rejection/preference, composite substitution, selected variants, a real cycle and one unresolved edge. Structural coverage stays known while resolution is incomplete; origins/artifacts stay unknown. Version/selection/URI binding, coordinate versus project identities, references/reachability, pinned enumerated causes, unresolved fixed category and count/text/depth budgets reject malformed/private responses. Canonical and Java File URI forms are accepted exactly; aliases/foreign paths are rejected. Empty module versions are preserved when the coordinate identity agrees. Missing/unadvertised/exception/malformed LSP responses fail with fixed sanitized recovery messages. Final source/fixture/executable hashes match; combined JVM/Copper compilation, repository all-targets/all-features project Clippy and formatting pass. Initial and final checks plus existing linker/upstream future-compatibility warnings are retained. Immutable gradle-graph-admission-20261005 archive: 39 entries, SHA-256 674c3602b8b6234de313e2057fe6d77578306a00aa11628373a38b49ffb72944. Provider 0.18.0 is unchanged. No GUI app, actual JVM/build worker, download or CI run starts. This closes only response/query validation; selected Gradle lifecycle (10.13), remote transport, shared-tree analysis, origins/artifacts, physical/native/resource/platform/terms/packaging/full parity/release remain open. Actions stopped, VoiceOver deferred, jvm planned.

[Decoder evidence](../../../.agents/evidence-archives/groovy-jvm/gradle-graph-admission-20261005/README.md). G26 remains incomplete at the product/tree/remote/native layer; no full-parity claim is made.

### Selected Gradle graph lifecycle — 2026-10-05

2026-10-05: selected local Gradle configuration graph admission shares the existing Maven query/task lifecycle, retaining separate provider-specific guards and immutable typed payloads inside existing project snapshots. Ten focused Gradle cases and 150 unique owner checks pass: 53 JVM panel/lifecycle (including all ten Maven cases), 53 JVM model, eight LSP, sixteen Copper, nineteen structured-result and one shared JVM settings; one Copper opt-in remains ignored. Four hundred deterministic scheduler replays pass (twenty Maven/Gradle cases, seeds 0–19). Known imported/resolvable configuration selection, unsupported/unknown/nonresolvable denial, build/project/configuration identity, current ownership, same versus unrelated folder scope, sanitized failure, explicit retry, owned cancellation, input changes, restart and trust revocation pass. Switching configuration replaces the single selected graph. Unresolved resolution remains current partial data with origins/artifacts unknown. Configuration work deliberately invalidates ownership; a fresh read is required and tested. Ordinary read refresh preserves Gradle metadata but now clears the old selected graph before the new generation; no hidden recollection occurs. Initial fixture omitted explicit Gradle metadata synchronization, then omitted its required ownership read; both ten-case failures are retained and corrected without weakening guards. Two Copper real-child startup checks failed in the initial suite and isolated runs; unchanged seed-1837 full suite, traced seed-zero overflow, and final normal seed-zero full suite pass. The initial child-start failure cause remains unqualified; no Copper fix or latency qualification is claimed. Final source/fixture/executable hashes match; combined compilation, repository all-targets/all-features project/jvm_ui Clippy and formatting pass, with existing linker/upstream future-compatibility warnings retained. Immutable gradle-graph-lifecycle-20261005 archive: 39 entries, SHA-256 b088e22452e0e0f581a83bc6d7990c60be2267833026d80ef90aca2bb0988dfb. Provider 0.18.0 and Copper provider source are unchanged. No GUI app, real JVM/build worker, download or CI run starts. Remote graph transport, shared-tree analysis, non-ASCII configuration-name coverage, actual origins/artifacts, physical/native/resource/platform/terms/packaging/full parity/release remain open. Actions stopped, VoiceOver deferred, jvm planned.

[Lifecycle evidence](../../../.agents/evidence-archives/groovy-jvm/gradle-graph-lifecycle-20261005/README.md). G26 remains incomplete at the tree/remote/native and full configuration-name layer; no full-parity or release claim is made.

### Selected graph snapshot transport — 2026-10-05

2026-10-05: existing JDTLS project snapshot protocol 5 carries optional provider-specific Maven-v2 occurrence and Gradle-v1 resolution facts. Absolute root/POM/module/build URI bindings are omitted. Enclosing visible imported module, relative build/POM, coordinates/configuration, server/worktree/generation and host trust/permission owners remain authoritative. Local and remote admission reuse the same bounded semantic validators; malformed selections, ownership, references, flags, counts, sizes, unknown/private fields and private parser errors are rejected. Missing facts remain unknown; incomplete resolution/collection and unavailable origins/artifacts retain their flags. Host versions 1–4 omit graphs; receipt under old versions rejects new fields. Unsupported older hosts retain explicit protocol errors, with no client-local JVM fallback. Serialization errors propagate; aggregate eight-MiB snapshot and provider-specific limits remain enforced. Three new protocol checks are a subset of 155 unique active checks: 53 JVM panel/lifecycle, 56 JVM model, eight LSP, sixteen Copper, nineteen structured-result, one shared settings and two framed remote checks. One Copper opt-in remains ignored. Eighty deterministic scheduler replays pass separately (three protocol and one framed case, seeds 0–19). The existing framed snapshot fixture admits explicitly selected host Maven and Gradle graphs, verifies protocol-4 omission/protocol-5 receipt, and retains its input-change, stale/replacement/trust/permission/reconnect assertions. Its graph payloads are bounded fake-provider fixtures; richer occurrence/conflict/cycle facts are covered by separately captured real-provider decoder fixtures. Initial remote build fails for missing Metal compiler; the existing runtime-shader feature builds matching software client/remote test owners. Initial framed API arguments and a missing GPUI debounce-clock advance are corrected, with failures retained and ownership guards unchanged. All-target/all-feature warnings-denied repository Clippy for project/jvm_ui/remote_server, formatting, canonical validators and actual-write/decomposition audit pass; existing linker/upstream future-compatibility and expected blocked/fake remote logs are retained. Final source/fixture/catalog/three executable hashes match. Immutable graph-snapshot-transport-20261005 archive: 39 entries, SHA-256 02865423396312bccaa1e96bb604aa172daeb00321e3d6eca0722641476fcc9c. Provider 0.18.0 and Copper source are unchanged. No GUI app, actual JVM/build worker, download or CI run starts. Remote explicit graph execution controls, shared-tree graph analysis, valid non-ASCII names (10.15/10.16), exact origins/artifacts, physical remote/native/Metal/platform/resource/terms/identity/packaging/full parity/release gates remain open. Actions stopped, VoiceOver deferred, jvm planned.

[Transport evidence](../../../.agents/evidence-archives/groovy-jvm/graph-snapshot-transport-20261005/README.md). G26/G27/M20 remain incomplete: current snapshot facts and software framed transport do not implement or qualify graph tree analysis, exact origins, remote execution controls, native workflows or full parity.

### Literal configuration provider correction — 2026-10-05

2026-10-05: development provider 0.19.0 (97869c5a9a275ccd0b404cf4da9c4f0b049cd9f2d68cc0f0eacdb6e7bbca06cb) accepts bounded literal configuration names using the existing Tooling API ConfigurationContainer lookup. UTF-8 text is bounded to 4096 bytes; empty, control and unpaired-surrogate inputs fail with sanitized errors. Actual pinned Gradle 8.13 validates its name grammar. Final denied-network cached offline JDTLS/Gradle fixture passes 54 checks across two same-profile journeys: exact stable Unicode/space/emoji names, 129-byte and exact 4096-byte ASCII/Unicode names, absent/nonresolvable/control/oversized/forbidden selections, existing conflict/constraint/variant/composite/cycle/unresolved facts, restart/failure recovery, cancellation and read-only boundaries. Owned isolated workers stop with exit zero and source/input bindings match. Initial 46-check fixture passes; exploratory colon/separator declarations fail because pinned Gradle forbids those names, with failure retained and corrected to invalid-selection cases. Pinned bytecode inspection is evidence only, not an internal API dependency. Java-17 bytecode compilation with warnings denied passes using the existing qualified compiler and eleven cached build inputs. Jar entry comparison preserves unaffected Maven/unlink/DTO bytes; its initial harness package-prefix mistake is recorded. Immutable archive has 35 entries, SHA-256 b4d506d926180fccb46ffe2d07ce1731104cb0f45c2f6126cdd0072bcdb0cf42. Shared Rust/Copper owners are unchanged and source-current post-merge checks remain applicable. Rust selection admission remains 10.16; graph presentation/origins/artifacts, real remote/native, Tasks/Tests/Debug, default delivery/terms/platform/resources/full parity/release remain unqualified. No distribution or bundling claim. Actions stopped, VoiceOver deferred, jvm planned.

[Retained evidence](../../../.agents/evidence-archives/groovy-jvm/gradle-configuration-names-20261005/README.md). Full graph/window parity remains incomplete.

### Qualified literal selection — 2026-10-05

2026-10-05: existing selected Gradle project query and imported configuration metadata share one provider-specific literal-name guard qualified against pinned Gradle 8.13. Valid Unicode/space/emoji and 129-byte/exact 4096-byte ASCII/Unicode selections preserve exact JSON and graph identities. Empty/control/oversized names, Gradle-forbidden punctuation and leading/trailing dots fail before dispatch; no supplied secret appears in errors. Unknown and nonresolvable selections remain rejected against current imported metadata. The existing local lifecycle fixture switches its second selected configuration to runtime 📦 α, verifies admitted identity and read-refresh graph invalidation, while stale/input/restart/trust/folder/cancellation checks remain strict. Fake presentation-only colon name is corrected to a valid space name; it was never actual provider proof. 154 active checks pass: 53 JVM panel/lifecycle, 57 JVM model, eight LSP, sixteen Copper provider, nineteen structured-result and one shared settings; one existing Copper opt-in stays ignored. Twenty separate scheduler replays of literal switching pass (seeds 0–19). Source-matching test builds, all-target/all-feature warnings-denied repository Clippy, formatting and diff checks pass. Existing linker unwind-size and upstream block future-compatibility warnings are retained. Immutable archive: 18 entries, SHA-256 afbe2988c37559a12a8f2d84edc7e01d56eb6bf87928ae272709c261fa07e470; source and executable hashes match. No new executor, result store, index, Cargo model or provider API. Native, physical remote, full graph analysis/exact origins, Tasks/Tests/Debug, terms/default/platform/resource/packaging/full parity/release gates remain open. Actions stopped, VoiceOver deferred, jvm planned.

[Evidence](../../../.agents/evidence-archives/groovy-jvm/configuration-name-admission-20261005/README.md). G26 remains incomplete; literal-name correction does not qualify the full tool window.

### Admitted graph tree facts — 2026-10-05

2026-10-05: the existing Dependencies/Gradle/Maven panel projects admitted provider graphs through language_tool_tree, with fixed-depth components/occurrences and edge sections. Maven rows retain distinct occurrence identities, requested versus selected coordinates, directness, scope, optional/exclusion, mediation, unavailable descriptors and explicit unknown cycle-occurrence coverage. Gradle rows retain configuration identity, selected project/module components, requested/selected selectors, rich version constraints, selection causes, composite/conflict flags, variants, constraints and sanitized unresolved categories. Structural edges preserve cycles/diamonds without recursive path expansion. Unknown graphs are named not collected; source origins/artifact readiness stay unavailable and every graph row has no panel navigation or execution target. Stable length-prefixed provider IDs and deterministic duplicate occurrence suffixes survive ordering changes; shared selection/status/keyboard/accessibility owners are reused. Ordinary rendering issues no provider command. Actual current local lifecycle fixture snapshots reach the real panel projection for both providers. Captured immutable real-provider facts exercise projection semantics; these are distinct from new real-provider or native journeys. 156 active checks pass: 55 JVM panel/lifecycle, 57 JVM model, eight LSP, sixteen Copper provider, nineteen structured-result and one shared settings; one existing Copper opt-in remains ignored. Forty separate scheduler replays pass across both Maven and Gradle admission/projection tests (seeds 0–19). Initial two new assertions assumed zero-based depth and action eligibility in the shared host; inspection shows one-based depth and the existing panel navigation map owns target eligibility. Corrected tests retain that owner and assert missing-origin graph rows have no targets. Initial failed test logs/source/executable bindings and an unsuccessful edit-marker assertion are retained; no shared guard was weakened. Source-matching test builds, warnings-denied all-target/all-feature repository Clippy, formatting, diff and canonical/manual audits pass. Existing linker/upstream future-compatibility warnings are retained. Immutable archive: 26 entries, SHA-256 000b1e55f45ef58c276674e12e374ed743770c7bb8d4b6122b01fe115ff90308; source/executable hashes match. No new graph model/index/store/framework/Cargo mapping/provider command/engine. Full 10.1/10.2 source origins, search/filter/root/neighbour analysis, explicit resolution controls, remote execution, spatial/export features, native/physical remote/VoiceOver/platform/resources/terms/defaults/packaging/full parity/release remain open. Actions stopped, VoiceOver deferred, jvm planned.

[Evidence](../../../.agents/evidence-archives/groovy-jvm/graph-tree-facts-20261005/README.md). G26/G27/M20 remain incomplete; fixed-depth graph facts do not qualify full analysis or tool-window parity.

### Exact current query cancellation — 2026-10-05

2026-10-05: the existing LSP project-query owner exposes its actual live pending ProviderRefreshToken and supports exact-token cancellation through the same state/task removal path. The existing worktree-only cancellation entry point delegates without changing its current-query behavior. Token-qualified cancellation returns false for completed, repeated, wrong-worktree, wrong-server and superseded tokens; it cannot cancel a newer refresh/query. Existing current Maven/Gradle graph lifecycle fixtures cancel held provider requests, reject unrelated tokens without dispatch or ownership change, and preserve newer queries after ordinary completion, sanitized failure, input changes, trust changes and server replacement/recovery. The pending-query bool uses the same live token accessor. No operation queue/registry, store, executor, permission owner, remote fallback or public provider SDK is added. 156 active local checks pass: 55 JVM panel/lifecycle, 57 JVM model, eight LSP, sixteen Copper provider, nineteen structured-result and one shared settings; one existing Copper opt-in remains ignored. Forty scheduler replays of Maven/Gradle late-token safety pass (seeds 0–19). Source-matching builds, all-target/all-feature warnings-denied repository Clippy, formatting, diff and canonical/manual audits pass; existing linker/upstream future-compatibility warnings remain retained. A draft test edit assumed a preexisting second-task variable; actual source inspection corrected the binding before compilation, with no failed test/build claim. Immutable archive: 20 entries, SHA-256 a7048fb49a16caa3d70fd8e7078b8d7acc68c237734fd2df3e132f0c320386db; source/executable hashes match. Panel Collect/Cancel controls remain 10.18, not implemented here. Physical remote/native, full synchronization/analysis/Debug/terms/default/platform/resources/packaging/full parity/release gates remain open. Actions stopped, VoiceOver deferred, jvm planned.

[Evidence](../../../.agents/evidence-archives/groovy-jvm/query-token-cancellation-20261005/README.md). G26/G27/M20 remain incomplete; cancellation safety does not implement panel controls or qualify full parity.

### Explicit offline graph controls and focused keyboard ownership — 2026-10-05

2026-10-05: the existing Groovy Dependencies/Gradle/Maven panel now exposes persistent named Collect offline graph and Cancel graph collection buttons. Typed current Maven POM or declared resolvable Gradle configuration targets retain server/worktree/generation identity and delegate to the existing LSP provider query and exact-token cancellation owners. Rendering, selection and snapshot refresh do not resolve dependencies. Missing/nonresolvable/stale/revoked targets cannot dispatch. Remote collection is explicitly unavailable: collaborative-client ownership preflight disables the control and rejects forced invocation without client-side collection. This software fixture uses existing collaborative test state; it does not qualify SSH host execution or physical remote workflows. No Cargo model, generic panel framework, semantic index, execution engine or result store is added. Current graph facts remain in the shared language_tool_tree and source navigation retains its existing map. Sanitized failure and cancellation notices use existing GPUI status/live metadata plus workspace notifications; successful repair clears them.

Actual mounted MultiWorkspace checks traverse focus to named controls and send paired Enter/Space events. An initial key-down-only probe was insufficient; paired events exposed an actual inherited product keymap bug (Maven opened its selected manifest). Groovy-specific control contexts now exclude tree-only bindings while reusing shared Button/GPUI behavior. Failed probes/diagnostics are retained; the corrected controls cancel held requests, reject late targets, recover after private provider failures and project changes, and render named polite status across consecutive accessibility frames. The existing tree-focused and persistent control suite passes. Shared GPUI/Button/tree owners are unchanged.

159 active local checks pass: 58 JVM panel/lifecycle, 57 JVM model, eight LSP, sixteen Copper provider, nineteen structured-result and one shared settings; one existing Copper opt-in remains ignored. Forty separate scheduler replays of the mounted Maven/Gradle control workflows pass (seeds 0–19 each). Source-matching builds, all-target/all-feature warnings-denied repository Clippy, formatting and diff checks pass. Existing linker/upstream future-compatibility warnings remain retained. Immutable archive: 39 entries, SHA-256 43610408e8378a6b324850e717e1d7b1ad617411daef97623b59441f09643338; final source/executable hashes match. These results close 8.10 and 10.18 local boundaries only. Native launch/visual keyboard, physical remote, VoiceOver, full graph filters/origins/artifacts/diagrams, synchronization/execution/Debug, provider terms/defaults/platform/resource/identity/packaging and full parity/release gates remain open. GitHub Actions stopped, VoiceOver deferred, jvm planned.

[Evidence](../../../.agents/evidence-archives/groovy-jvm/graph-collection-controls-20261005/README.md). G13/G26/G27/M20 remain incomplete; local controls are not full synchronization, graph analysis, remote/native or tool-window parity qualification.

### Admitted graph fact filters — 2026-10-05

2026-10-05: the existing panel projects admitted dependency graph facts with All facts, Direct dependencies, Conflict resolution and Unavailable metadata filters. Maven uses provider direct edges, omitted mediation occurrences and unavailable descriptors; Gradle uses root edges, selected conflict-resolution causes and unresolved edges. Retained endpoints/occurrences preserve original row identities, including repeated conflict edges and reordered provider arrays. Empty filtered results and missing graphs retain explicit coverage; no source origin/artifact/cycle completeness is invented. Existing language_tool_tree selection removes filtered targets and clears revoked snapshots. Filter changes rebuild presentation only and issue no provider commands. No Cargo model, second graph/index/store, generic framework or execution owner is added.

The existing shared Button exposes the current filter name/value and next choice, with product control key context retained. Mounted Maven/Gradle workflows use actual focus traversal and paired Enter events, inspect name/value across two frames, cycle back to All, and assert zero additional queries. Existing ordered workspace KVP saves/reloads the filter with other view preferences, preserving held-write release and failure recovery. Bounded strict schema 5 migrates schemas 1–4 to All, preserving prior flags and schema-4 dock position; older versions reject newer/foreign fields and current invalid filters fail safely. Projection checks use captured real-provider graphs; Maven descriptor absence is additionally exercised by an explicit partial-metadata software fixture, not a new real-provider run.

161 active checks pass: 60 JVM panel/lifecycle, 57 JVM model, eight LSP, sixteen Copper provider, nineteen structured-result and one shared settings; one existing Copper opt-in remains ignored. Forty separate final-source mounted filter scheduler replays pass (seeds 0–19 per provider). Source-matching builds, all-target/all-feature warnings-denied repository Clippy, formatting and diff checks pass; existing linker/upstream future-compatibility warnings remain retained. Initial test compilation used a missing GPUI input flag and a nonexistent visible-row activation field; actual APIs were used before the passing build. The initial mounted Maven navigation assertion assumed expansion survived an empty filtered section; the test now explicitly expands restored All rows, preserving the shared tree owner. That failed log/source/executable receipt is retained. An edit-marker failure wrote no source. Immutable archive: 29 entries, SHA-256 4050eab894cb900a5c13137c5b79d49f72b109f0984718af83bed4cc08553b67; final source/executable hashes match.

This closes bounded 10.19 only. Full 10.1/10.2 search/scope/root/neighbour paths/origin navigation, diagrams/export, effective settings/synchronization/execution/Debug, physical native/remote/VoiceOver, provider terms/defaults/platform/resources/identity/packaging and full parity/release gates remain open. Actions stopped, VoiceOver deferred, jvm planned.

[Evidence](../../../.agents/evidence-archives/groovy-jvm/graph-fact-filters-20261005/README.md). G26/G27/M20 remain incomplete; fact filtering does not qualify full graph analysis or tool-window parity.

### Admitted graph fact search — 2026-10-05

2026-10-05: the existing Groovy panel now searches admitted graph labels and presented facts after provider-specific facets, using a trimmed case-insensitive lowercase literal query. The existing Editor single-line input owns editing; the shared search-role/name/value/focus pattern and SetValue handler mirror existing Extensions/settings input owners. Product control key context protects typing from tree actions, and shared Button clears the input. Query text is ephemeral, not logged or saved. UTF-8 queries over 512 bytes or containing controls suppress graph children and show a fixed recovery message; provider snapshot currentness/ownership is unchanged. The existing shared host may represent a nonempty error view as StaleError, which is retained rather than redefined. Uncollected/partial coverage and no-match roots remain named without echoing queries. Module/task/manifest rows remain available. Only graph ancestors are revealed through existing expand_nodes; identities, selection and current trust remain shared-owner behavior. No index, graph store, panel framework, execution engine or Cargo mapping is added. Editor moves from test-only to production dependency; the existing lockfile already contains that owner and is unchanged.

Captured real Maven/Gradle facts prove literal/case/Unicode matching, original ancestor/leaf IDs, no-match/uncollected coverage, preserved task rows and invalid input. Mounted Maven/Gradle workflows collapse the tree, focus the actual shared editor, type SHARED, inspect SearchInput name/value and focus across two frames, inspect matching facts, suppress no-match/oversized results, clear through the existing action and recover; provider command counts never increase. Existing trust, stale/restart, filter/preference and graph-control suites pass. Registered SetValue uses the established handler; physical assistive-action/VoiceOver behavior is not qualified by these software checks.

162 active local checks pass: 61 JVM panel/lifecycle, 57 JVM model, eight LSP, sixteen Copper provider, nineteen structured-result and one shared settings; one existing Copper opt-in remains ignored. Forty separate final-source mounted editor/search scheduler replays pass (seeds 0–19 per provider). Source-matching builds, all-target/all-feature warnings-denied repository Clippy, formatting and diff checks pass; existing linker/upstream future-compatibility warnings remain retained. Three initial test failures assumed a leaf coordinate in both captured providers and Error instead of the existing nonempty StaleError. Actual fixture/owner inspection corrected those assertions; failed source/executable/check receipts are retained. The corrected real shared coordinate and status behavior are not product fallbacks. Immutable archive: 28 entries, SHA-256 7f0d97289e017a7d67f04079125827df4ee7e0923d21b9d2ed107ce62169671a; final source/executable hashes match.

This closes bounded 10.20 only. Full 10.1/10.2 scope/root/neighbour paths/origin navigation, diagrams/export, host settings/synchronization/execution/Debug, physical native/remote/VoiceOver, provider terms/defaults/platform/resources/identity/packaging and full parity/release gates remain open. Actions stopped, VoiceOver deferred, jvm planned.

[Evidence](../../../.agents/evidence-archives/groovy-jvm/graph-fact-search-20261005/README.md). G26/G27/M20 remain incomplete; literal fact search does not qualify full graph analysis or tool-window parity.

### Maven scope facets implemented — 2026-10-05

2026-10-05: the existing Dependencies/Maven view now filters admitted Maven graph edges by All, compile, runtime, test, provided, system, import or other/unspecified scopes. Matching uses exact provider scope facts, including omitted occurrences; it does not infer effective/default scope or convert Gradle configurations. Scope composes with current direct/mediation/unavailable facets and subsequent literal search. The root, retained edge endpoints and selected occurrence identities remain stable; descriptor/structural/collection gaps stay explicit. Empty results do not become complete metadata. Scope changes use existing shared-tree status/selection and issue no provider queries. A persistent named shared Button supplies value, next choice, keyboard and focus behavior. Existing bounded workspace KVP preferences use strict schema 6, migrating schemas 1–5 to All scopes while preserving former flags, schema-4/5 dock position and schema-5 fact filter. Old schemas reject the new field; unknown scope values fail safely. No provider, graph model, semantic index, execution engine, result store, Cargo model or panel framework is added.

Captured real Maven scope facts project 12 compile, one provided and one test edge, with zero runtime edges. Explicit software variations over that captured topology exercise all eight facets, unknown/empty/case-distinct scopes, omitted mediation, partial descriptor gaps and fact/search composition. Retained rows preserve sorted occurrence identities and unknown origins/artifacts. Mounted Maven Enter/name/value/focus and zero-query checks pass across two accessibility frames; Gradle workflows remain unchanged. Ordered preference write/reload/release/recovery retains a nondefault scope. The initial projection test assumed unsorted rows and unsuffixed IDs; it is corrected to the actual shared graph-row owner, with failed source/log/executable receipts retained. No product workaround was introduced. The final formatter correction and early cache/build preflight refusals remain recorded; only inventoried inactive generated task-cache outputs were reclaimed, preserving source, archives, current test binaries and all native candidates/fixtures.

164 active checks pass: 63 JVM panel/lifecycle, 57 JVM model, eight LSP, sixteen Copper provider, nineteen structured-result and one shared settings; one existing Copper opt-in remains ignored. Forty separate final-source mounted Maven/Gradle scheduler replays pass (seeds 0–19 per provider). Source-bound test builds, warnings-denied all-target/all-feature repository Clippy for the changed jvm_ui crate, formatting, diff and both canonical validators pass. Existing linker/upstream future-compatibility warnings remain retained. Immutable archive: 97 entries, SHA-256 03ea8b6e933a79b85940471e808c2847317b58e5288466caa3619cfc13a4608a; final source/executable hashes match.

This closes bounded 10.21 only. Full 10.1/10.2 root/neighbour/path/usage/origin analysis and cycle mapping, diagrams/export, effective host settings/synchronization/execution/Debug, native/physical remote/VoiceOver, provider terms/defaults/platform/resources/identity/packaging and full parity/release gates remain open. The unlaunched c625e263ae native candidate predates this facet and cannot qualify it. Actions stopped, VoiceOver deferred, jvm planned.

[Evidence](../../../.agents/evidence-archives/groovy-jvm/maven-graph-scopes-20261005/README.md). M20/M21/M22 remain incomplete; scope facets do not qualify the full analyzer or tool-window parity.

### Selected phase receipt implementation — 2026-10-05

7.59 adds a live local import/unlink ProviderRefreshToken accessor and exact-receipt cancellation through existing LspStore/JdtlsProjectState phase ownership. A matching receipt delegates to the current cancellation path, preserving partial failure state, startup cleanup and owned-request cancellation. Completed, prior-generation, synthetic other-worktree and stopped-server receipts return false without cancelling replacement work. Four existing held Maven/Gradle import/unlink fixtures retain progress/read-refresh, sanitized errors and input/trust/provider/restart recovery assertions. The other-worktree receipt is a software guard fixture, not a second physical pending project. No phase registry, executor, RPC, result store, index, SDK, Cargo model or panel framework is added. All 210 active affected checks pass, including Copper provider, shared settings, Cargo preset model/editor/menu and shared Dock regressions; one existing Copper opt-in remains ignored. Eighty final-source scheduler replays pass (four phase fixtures, seeds 0–19). Source/executable hashes match, build and all-target/all-feature warnings-denied repository Clippy for project/jvm_ui pass, as do formatting, diff and both canonical validators. Existing linker/upstream future-compatibility warnings remain retained. Archive: 106 entries, SHA-256 f024ec1c788b8437626797a6d92239eefa3352166f94eca36068114e4acb3e22. This closes bounded 7.59 only; persistent project controls, whole synchronization, physical/native/remote parity and all release gates remain open. The c625e263ae native candidate remains unlaunched awaiting exact approval and predates this implementation. Actions stopped, VoiceOver deferred/off, jvm planned.

### Selected graph inspection implementation — 2026-10-05

10.22 adds explicit selected dependency Neighbours, Shortest root paths and Clear inspection controls within the existing Groovy panel. Shared Buttons/tree own names/values/focus/paired Enter and selection. Only current snapshot server/worktree/generation plus a provider node identity and mode are retained; no graph is copied into inspection state. Gradle components and Maven occurrence ancestry remain separate. Maven requested/selected mediation is not substituted for ancestry. Gradle constraints remain neighbour facts but never shorten dependency inclusion paths. Bounded forward/reverse distances retain every equal-shortest path; fixed-depth rows avoid cycle unfolding and preserve duplicate occurrences, diamonds, unknown coverage and unresolved outgoing neighbours. Root-self, unreachable, parallel edges and a 4096-node chain pass software checks. Captured Maven mediation and Gradle constraint facts pass. Literal search applies; ordinary fact/scope facets do not hide inspection ancestry, and the accessible description states that boundary. Invalid selection gives a fixed actionable error without erasing current inspection; successful selection clears that error. Mounted Maven/Gradle focus/name/value/Enter and duplicate-ID/zero-query assertions pass; partial graph flags remain explicit. External input, revoked trust, changed server/generation/worktree/node receipts and view changes remove inspection. Receipt mutation is software state evidence, not a physical restart/reconnect journey. All 212 active affected checks pass, including Copper provider/shared Settings/Cargo preset model/editor/menu/Dock regressions; one existing Copper opt-in remains ignored. Forty final-source mounted scheduler replays pass (two provider fixtures, seeds 0–19). Final source/executable hashes match; source-bound compilation, all-target/all-feature warnings-denied jvm_ui repository Clippy, formatting, diff, both canonical validators and manual audit pass. The initial successful suite had a source-binding rejection because formatting finished during compilation; that unqualified run is retained and a frozen-source rebuild/check succeeds. The prior passing pre-selection-error run is retained separately. Existing linker/upstream future-compatibility warnings remain. Archive: 144 entries, SHA-256 5f9ca2403fb3cba4fd30a4b88315ce789c8a6307af6333425c529761cdce8a15. This closes bounded 10.22 only: multiselection/arbitrary paths/full usages/origins, flat/tree parity, diagrams/layout/export, physical/native/remote execution, default provider delivery/terms/platform/resources/identity/packaging and every release gate remain open. No new Cargo model, framework, semantic index, provider API, executor or result store. No GUI, download, CI or release starts; c625e263ae remains unlaunched pending exact approval and predates this capability. Actions stopped, VoiceOver deferred/off, jvm planned.

### Saved local build-selection controls — 2026-10-05

8.11 adds named Apply saved Maven selection, Import Gradle selection and exact-owned Cancel selected import controls to the existing panel toolbar. Typed selected navigation or the sole visible local worktree supplies the host; ambiguous/missing/nonlocal hosts are refused. Existing saved Maven roots/exclusions and effective Gradle import settings delegate to the LSP phase owner. Only a receipt, waiter and UI status persist; no settings store, generic framework, Cargo project model, index, executor or result store is added. Mounted shared focus/paired Enter and two-frame polite status pass. Held provider requests cancel through the exact receipt; a retired UI receipt cannot cancel an actual replacement host phase. Success, private failure, fixed actionable sanitized notices, repaired retry and revoked trust pass. Remote controls expose explicit disabled/unavailable metadata; forced remote and absent-trust invocations start no host phase. All 214 active affected checks pass (67 JVM panel/lifecycle, 57 model, 8 LSP, 16 Copper, 19 structured results, 1 JVM settings, 37 shared Settings, 9 Cargo preset/editor/menu/Dock); one existing Copper opt-in remains ignored. Forty final-source mounted Maven/Gradle scheduler replays pass (seeds 0–19). Source-bound compilation, all-target/all-feature warnings-denied jvm_ui repository Clippy, formatting, diff, both canonical validators and manual actual-write/decomposition audit pass. Two test-context compilation failures and the earlier passing source are retained; scoped fixture corrections preserve the existing graph-trust fixture. Final source/executable hashes match. Archive: 100 entries, SHA-256 b066fc23e50efbc67c54e07b292dc98a4c9f6956eba79e01fe30e9328b4ab1a3. This completes this local panel consumer only. Full default linking/unlinking/ignore/orphans, persistent selection editing, all/automatic causal reimport, effective build settings, Tasks/configurations, native/remote/platform/provider terms/resources/identity/packaging and every parity/release gate remain open. No native candidate launched; c625e263ae approval is still pending and that binary predates these controls. GitHub Actions stopped, VoiceOver off/deferred, jvm planned.

### Current dependency diagram implementation — 2026-10-05

10.23 adds explicit Show diagram and Close diagram controls in the existing Groovy panel, using Made Markdown/Mermaid for layout, SVG/raster and renderer interaction/errors. Only the current graph receipt and shared Markdown entity persist. Maven occurrences/ancestry remain separate from Gradle components/constraints; mediated edges and constraints are dotted, unresolved Gradle targets explicit placeholders. No coordinate deduplication, recursive expansion, background collection, second semantic graph/index/store/framework or new executor. Numeric generated node IDs and decimal-entity labels keep provider text out of syntax/directives/links. Unicode survives the actual shared SVG renderer; quotes, backticks, directive/click-like labels, cycles, diamonds, parallel edges and duplicate-coordinate occurrences render without active links/scripts/foreignObject. Unsupported control/angle-bracket labels, missing/duplicate references and budgets fail explicitly: 64 total nodes, 128 edges, 256 characters per label and 128 KiB complete generated document, with no truncation. Existing tree facts remain available at their own limits. Captured Maven/Gradle mounted controls receive shared keyboard focus/paired Enter, with named group metadata across two frames and zero provider commands. Diagram receipt remains independent of path inspection; ordinary tree facets preserve the same renderer entity. Stale server/generation/worktree, external build changes, view change and restricted trust drop presentation. Partial coverage remains explicit. All 216 active affected checks pass (69 JVM panel/lifecycle, 57 model, 8 LSP, 16 Copper, 19 structured results, 1 JVM settings, 37 shared Settings, 9 Cargo preset/editor/menu/Dock); one existing Copper opt-in remains ignored. Forty final-source mounted provider scheduler replays pass (seeds 0–19 each). Source-bound compilation, all-target/all-feature warnings-denied jvm_ui repository Clippy, whole formatting/diff, canonical validators and manual actual-write audit pass. The initial render-context compilation failure is retained and corrected. A helper syntax error and a corrective preflight assertion failed before source writes. The first build's binding changed during formatting/lock generation and is not qualification evidence. An inspected path-inspection coupling error was corrected before workflow checks; the stronger independent-receipt regression now passes. The earlier successful suite is retained separately. Final source/executable hashes match; existing linker/upstream future-compatibility warnings remain. Archive: 94 entries, SHA-256 a50f65787150573fb68cd27673f6ee5b4701013ec7d06c4ded286199ea9156bd. This completes bounded current diagram presentation only. Large diagrams, node selection/navigation within the image, multiselection, arbitrary/full paths/usages/origins, popup/overview/layout/export/window modes, visual resemblance/native raster interaction, physical remote/platform, provider terms/default delivery, Kotlin memory/platform, identity/packaging and all full parity/release gates remain open. No native candidate launched; c625e263ae approval remains pending and that candidate predates this capability. Actions stopped, VoiceOver off/deferred, jvm planned.

### Selected local offline operations — 2026-10-05

9.6–9.9 add a bounded local offline operation path through existing owners. TaskTemplate's internal literal resolution preserves normal substitution/IDs while quoting provider argv as data. Typed current JVM receipts derive ordered known Maven phases/exact versioned non-managed plugin goals or exact Gradle task paths at the actual build root. Bounds are 1–32 operations, 4096-byte tokens/paths and 64 KiB argv; unsafe template suffix paths and unknown versions are refused. Inventory retains opaque receipts at existing history plus one pending admission, binds effective JVM settings, wrapper/manifests/root and resolved fields, and rejects stale generations, replacement roots, changed/missing files, parse errors and altered tasks. Workspace repeats live admission after save; remote/collaborative execution has no client fallback. Normal Copper error-chain logging is preserved; discovered operation spawn errors remain sanitized in notices and the existing structured task handle.

The existing Maven/Gradle panel uses typed row selections and named persistent shared Buttons. Mounted paired Enter/focus and accessibility metadata, exact terminal events, unknown-version refusal, stale refresh/recovery, wrapper repair, trust revocation and all three remote view controls pass. No provider execution command, second framework/index/executor/result store, Cargo model or public SDK is added. These controls submit one selected operation; arbitrary options/ordered UI composition/configurations and full effective settings remain incomplete.

All 291 active final-source checks pass across Task, shell/terminal quoting/cancellation/results, JVM panel/model/LSP/Inventory/Workspace, structured store, JVM/shared settings and Copper provider/presets/editor/menu/Dock. One existing Copper opt-in stays ignored. Eighty separate scheduler replays (seeds 0–19 for mounted Maven, Gradle, admission and Inventory) pass. Source-bound builds, warnings-denied all-target/all-feature repository Clippy, whole formatting/diff, both canonical validators and manual ownership/decomposition audit pass. Existing linker unwind-size and upstream block future-compatibility warnings are retained; optional cargo-machete is unavailable. Initial fixture/trait/lint/NUL-probe failures, zero-test preflight, interrupted owned builds, remote view assertions and the seed-2 initialization race are retained with corrections; final source/executable hashes match. Archive: 591 entries, SHA-256 63584ac9f47d5f08ce9120d64a0e5031d952bbc57b1ae383eff0a9804cffdbfd.

This qualifies these bounded software increments only. Native terminal builds and selected Tests reports, process restart, remote execution, Windows/other shells, full parity, Debug delivery/terms, preferred/fallback Kotlin and owned-resource/platform/package/release gates remain open. No new native candidate launched; the older c625e263ae confirmation is pending and cannot cover this source/scope. GitHub Actions is stopped and infrastructure-dependent gates are open, VoiceOver is off/deferred, jvm stays planned. No release is published.

[Retained evidence](../../../.agents/evidence-archives/groovy-jvm/selected-jvm-operation-controls-20261005/README.md).

### Gradle option declaration collisions — 2026-10-05

2026-10-05: development provider 0.20.0, SHA-256 6767e1b6530e09012b93b62417af44340595233743f48701484c4f517ba01eb2, passes pinned warnings-denied Java-17 bytecode compilation and 37 externally denied-network real JDTLS/Gradle checkpoints. Distinct setters and same-named private inherited fields with equal display metadata retain declarations but mark coverage incomplete; equal overridden signatures stay known, differing metadata and interface precedence stay incomplete. Repeated payloads are stable. Existing composite/source/configuration/budget/cancellation/offline/failure/recovery checks pass, with no requested output tasks or reports. Gradle itself invokes the getter canary during task creation before provider reflection; its failure is sanitized and repair recovers. Initial incompatible heap and incorrect getter-side-effect assumption attempts remain retained, as does a diagnostic offline help call (not product Tasks evidence). Only the builder class and manifest differ from provider 0.19.0; twenty other payloads match exactly. All pinned inputs and final source bindings match. Cleanup through the existing checked Gradle stop uses a new fixture-owned home, not another profile's worker registry. Archive 60 entries, SHA-256 cb0fab306c0221663d9af76732c502a2dd3bafb79d560d06505dac01264bb0d9. No Rust/client/remote schema/shared owner changes; incoming shared terminal postmerge Rust checks remain queued, not passed. Accepted values, interface/option/default/full parity, native/remote/platform/resources/terms/packaging/release remain open; Actions stopped, VoiceOver deferred and jvm planned.

[Retained provider evidence](../../../.agents/evidence-archives/groovy-jvm/gradle-option-collisions-20261005/README.md). G05/G16 remain incomplete: bounded option declarations are corrected, while accepted values, complete interface semantics, executable arguments/configurations and native/remote qualification remain open. The full reference is preserved.

### Hidden Gradle option declarations — 2026-10-05

2026-10-05: development provider 0.21.0, SHA-256 6fa6b8259debe20314f5f7cca6c7270ed5d620ac071060b8363890e35ac1bf37, compiles to Java-17 bytecode with warnings denied and passes 39 actual externally denied-network JDTLS/Gradle checkpoints. Same-signature private and static declarations retain bounded metadata with incomplete coverage; public equal overrides remain known. Prior setter/field/interface/conflict, stable composite/source/configuration metadata, budget limits, sanitized task-creation failure and repair, offline/bootstrap/recovery/cancellation checks pass. The unchanged prior 0.20.0 provider fails the new collision coverage fixture as expected, proving the regression control. The existing builder and manifest change. Five other class payloads have shifted source line tables and identical javap instruction/member listings; fifteen other payloads remain byte-identical. The rejected initial payload-equality assumption is retained. Source and pinned input hashes match. Existing checked cleanup stops only workers in the new owned home. Retained archive: 50 entries, SHA-256 4a169d9946f87e9ab08b48d70cd2073732a6ef1ff38a293a1a5ca153ef90b474. No Rust/client/remote schema or new store/index/framework/execution owner is introduced. Postmerge shared terminal Rust checks are queued behind the shared build window, not passed. Accepted values/full interface semantics, native/physical remote/platform/resources/terms/default/packaging/full parity/release remain open; Actions stopped, VoiceOver deferred and jvm planned.

[Provider evidence](../../../.agents/evidence-archives/groovy-jvm/gradle-hidden-option-methods-20261005/README.md). G05/G16 stay incomplete: declaration completeness is corrected; full values/arguments/configurations/default/native/remote parity remains open.

### Ordered local offline operations — 2026-10-05

2026-10-05: 9.10 adds one ephemeral ordered selection of up to 32 current typed operations to the existing Maven/Gradle panel. Shared tree rows retain chosen order and repeats; shared persistent Buttons own add/remove-last/clear/run names, values, focus and paired Enter. Only Run submits one combined receipt through existing Workspace/Inventory/Tasks admission and literal offline terminal argv. Wrong module/generation/build kind, over-limit/stale/trust/missing-wrapper and remote attempts cannot add terminal events. Refresh and wrapper repair allow explicit fresh retry. No provider execution command, queue, framework, Cargo model, index, SDK or result store is added.

All 297 final-source active checks pass, including 73 panel cases, existing Tasks/model/LSP/Inventory/Workspace/store/settings and Copper/preset/menu/Dock owners. One existing Copper opt-in remains ignored. All four incoming interactive-terminal stop shell/history cases pass, preserving Task cancellation. 120 separate scheduler replays pass (six fixtures, seeds 0–19). Source/executable bindings, frozen-source compilation, all-target/all-feature warnings-denied repository Clippy for task/project/workspace/jvm_ui/terminal, formatting, diff, both canonical validators and manual decomposition/ownership audit pass. The group-ID compilation failure and remote rejection notice replacement are retained with their corrections. An owned cold-profile lint attempt was interrupted; inspected profile-only fingerprint differences led to the successful warm release-debug-zero retry. Existing linker/upstream future-compatibility warnings remain; optional cargo-machete is unavailable. Archive 178 entries, SHA-256 2b865828bfe714c1114a1f7bf5dcc74328fa9ca6e0cd80a4e36102db1ac39915.

This closes bounded local 9.10 only. Recorded terminal events do not prove successful native builds or selected test reports. Full argument/settings/configuration/trigger/history/parity, remote execution, current native candidate/physical platforms, preferred/fallback Kotlin resource/default/terms qualification, Debug and packaging/release gates remain open. No app launches or release occurred. Actions stopped, VoiceOver deferred, jvm planned.

[Retained evidence](../../../.agents/evidence-archives/groovy-jvm/ordered-operation-controls-20261005/README.md).

### Gradle enum option declaration qualification — 2026-10-05

2026-10-05: development provider 0.22.0, SHA-256 c848a881cecd59723dd6703ad751d331274a7173f42cc6f68b2183145345d822, compiles to Java-17 bytecode with warnings denied. Thirteen reflection checks qualify exact scalar/Property/field enum names, unknown raw/wildcard/type-variable/collection/String shapes, known-empty, the 128/129-constant boundary and an effective initialization positive control. Forty-six real cached denied-network JDTLS/Gradle checks qualify stable additive declarations, untouched fixture-owned enum/OptionValues canaries, sanitized overflow and repair, existing collision/interface/source/configuration/budget/offline/cancellation/recovery behavior. The unchanged 0.21.0 provider fails the new declaration fixture as expected. Only the builder behavior and manifest differ: fifteen other payloads are byte-identical and five have identical instruction/member listings with shifted source line tables. Eleven compile-input pins match. The initial fixture package-shadowing failure, edit assertion failure, strengthened temporary-path canary and unavailable JBR javap lookup remain retained. Checked shutdown reports no workers in the isolated owned home; no requested output tasks or reports occur. Archive 81 entries, SHA-256 23b876a0df5a4ad987eebe28a20baf9dc882e90f00e1d2be7a120e731f64b244.

This closes bounded 7.62 only. Declared enum members are not custom setter acceptance; dynamic suggestions/defaults/requiredness/interfaces, product client/UI/options execution, native/physical remote/default delivery/platform/resources/terms/packaging/full parity/release remain open. No shared Rust owner changes in this increment; separate 1.8 LSP changes and runtime/Copper qualification remain uncommitted and open. No native app or release starts. Actions stopped, VoiceOver deferred, jvm planned.

[Retained evidence](../../../.agents/evidence-archives/groovy-jvm/gradle-enum-options-20261005/README.md).

### Local project-consumer and enum admission qualification — 2026-10-05

2026-10-05: task 1.8 retains bounded idempotent local project language intent through the existing LocalLspStore/LanguageServerTree, with real-buffer sharing and existing trust/settings/stop/restart/provider-replacement ownership. All seven real FakeLSP lifecycle cases pass; trust-wait/sharing, revocation/recovery and pending-startup withdrawal also pass twenty scheduler seeds each (60 replays). Task 7.63 admits bounded declared enum names through the existing local decoder/snapshot, distinguishes unknown from known-empty, rejects fifteen malformed-pair cases before mutation and preserves current identities. Three enum tests and the current host snapshot downgrade fixture pass. All 307 unique active affected checks pass: shared Tasks/shell/terminal cancellation/results, 73 JVM panel, 63 JVM model, 15 LSP, 5 inventory, 17 workspace Tasks, 16 Copper provider, 19 structured-store, JVM/shared Settings and mounted Cargo preset/Dock owners. One existing Copper opt-in remains ignored. Combined locked/offline executable compilation and all-target/all-feature warnings-denied repository Clippy pass. The final jvm_project source is byte-identical to the tested/linted source; final lsp_store differs only by an exact retained rustfmt replay wrapping the trust-fixture initialization call, and final whole-workspace formatting passes. Initial missing-trust fixtures, redundant-clone lint failure, failed formatting check and failed temporary-file formatter replay are preserved with their corrective evidence. Historical 20 GiB reserve interruptions remain; that agent guard was replaced by a measured 10 GiB threshold on the same warm graph, retaining all prior executable APFS clones. Three successful build checkpoints have frozen source; final build minimum free space is 11,568,467,968 bytes. No SDK/cache/source or verified executable deletion was performed by this work. Existing linker unwind-size and upstream future-compatibility warnings remain. Archive: 133 entries, SHA-256 5204ed4c57506249a2abaf7d96b2abe06385c61ae2995df6bfd8de4f4fe49385. This completes only local lifecycle 1.8 and local admission 7.63. Groovy default startup 1.9, authoritative remote registration 1.10, enum transport/presentation 7.64/7.65, full options/custom acceptance and tool-window parity, native/physical remote/provider delivery/terms/resources/platform/identity/packaging and release gates remain open. No native app, download, CI or release starts. Actions stopped, VoiceOver deferred/off, jvm planned.

[Retained evidence](../../../.agents/evidence-archives/groovy-jvm/project-consumer-and-enum-admission-20261005/README.md).

### Qualified enum snapshot transport — 2026-10-05

2026-10-05: task 7.64 carries bounded declared enum names and known-empty coverage through the existing typed JDTLS snapshot protocol 6. A supported older-version projection loses only the additive enum fact; new clients still require matching host protocol and do not silently retry or query locally. Malformed flag/name pairs, duplicated/private/control names and 129-member overflow fail before admission with sanitized errors. Current typed wire and framed host/client cases preserve exact 128-member and known-empty facts. All 314 active affected software checks pass: 307 local JVM/shared/Copper checks and seven framed JVM/Copper permission, disconnect, reconnect and cancellation checks. One existing Copper opt-in remains ignored. Locked/offline compilation, whole-workspace formatting and all-target/all-feature warnings-denied repository Clippy pass. All retained production source bindings match final source. The first native-shader build failed because Xcode's MetalToolchain is missing; the successful graph uses the existing gpui_platform/runtime_shaders feature, which qualifies software/headless fixtures only, not native rendering or an app. The first model run exposed a stale unsupported-version-6 test assumption (62 passed, one failed); the assertion now uses current protocol plus one and the final 63-model suite passes. The interrupted prior-source Clippy attempt, frozen-source 10 GiB reserve interruption and all failed checkpoints remain retained. The identical-source lint retry starts only after live free capacity recovers above 30 GiB; no cache or evidence deletion is performed by this work. No toolchain download, binary/evidence/source cleanup or other worker termination occurred. The final build minimum free space was 11922604032 bytes under a live 10 GiB guard. Prior executable hashes/APFS clone receipts remain; executables are not redistributed in this archive. Archive 113 entries, SHA-256 cbfe9b7d5a66df7c6a0c1693fda27c0b274c98f89cf2f99d3d3bc7e6f8aa8df7. This closes only software transport 7.64; enum presentation 7.65, default startup 1.11/1.12/1.9, authoritative remote registration 1.10, custom acceptance/full option parity, native/physical remote, provider terms/delivery, Kotlin resource/platform/identity/packaging and full tool-window/release qualification remain open. Actions stopped, VoiceOver deferred/off, jvm planned; no release.

### Qualified enum tree presentation — 2026-10-06

2026-10-06: task 7.65 renders bounded declared enum members as inert children of existing Gradle option rows through language_tool_tree. Existing option/module/task identities stay stable; visible and accessible labels distinguish unknown, known-empty and member counts while keeping custom setter acceptance, requiredness and accepted values unknown. Actual mounted provider/refresh fixture passes keyboard right/left/Enter, no navigation or projection query, current refresh, stable selected member/active descendant/accesskit identity and panel focus over consecutive frames. All 101 active focused checks pass: 73 JVM panel, eleven direct shared-tree, sixteen Copper provider and one mounted Copper Dock. One existing Copper opt-in stays ignored. Initial concurrent-run Copper EOF failure (15 pass, one fail) is retained; its Project binary is byte-identical to the qualified pre-UI executable, exact seed-zero reproduction passes, the affected case passes twenty scheduler seeds and the final whole Copper group/Dock pass without source changes. This resolves the observed regression check for this delivery, not a claim that an OS polling deadline or concurrent-load flake was fixed. Locked/offline compilation, whole-workspace formatting and all-target/all-feature warnings-denied repository Clippy pass; all source and retained executable bindings match. Archive 35 entries, SHA-256 56c7a80c00c0389442d7f00a18c044a7a1fdca5ed866674e63d7f42a11fee1ab. Only the existing JVM panel file changes product code; shared owners, provider queries, execution/result/index/navigation/startup ownership remain intact. Software metadata/focus fixtures are not native rendering, VoiceOver, physical remote or platform qualification. Full options/custom acceptance and reference tool-window parity, default startup 1.11/1.12/1.9 and remote 1.10, provider terms/delivery, Kotlin resources, platform/identity/packaging and release gates remain open. Actions stopped, VoiceOver deferred/off, jvm planned; no release.

### Gradle enum list declarations — task 7.66

2026-10-06: development provider 0.24.0, SHA-256 8cf932589c6f7762f57205d2c5d20319fc7a04c0ca6384aec873715913187372, compiles to Java-17 bytecode with warnings denied and passes 30 reflection checks plus 47 real cached externally denied-network JDTLS/Gradle checks. Existing enum declaration fields now expose bounded List/ListProperty enum names, including natural Kotlin covariant upper-bound signatures. Raw, variable, nested, lower-bound/non-enum and Set/SetProperty shapes remain unknown; scalar Property wildcard behavior is unchanged. Known-empty and 128/129-member bounds, untouched initialization/getter/OptionValues canaries, stable replies, collisions/interfaces, sanitized failure/repair, cancellation, offline bootstrap and recovery pass. Previous 0.22.0 fails the strengthened list budget oracle as expected. The first 0.23.0 Java reflection checkpoint passes but the natural Kotlin list fixture fails; its source/probe is retained, and the fixture is not narrowed using JvmSuppressWildcards. Both isolated owned homes report no running Gradle daemons after checked shutdown. All eleven compile-input pins match. Only builder behavior and manifest differ from 0.22.0: fifteen other payloads are byte-identical and five have identical instruction/member listings with shifted line tables. Existing Rust/shared/Copper source bindings remain unchanged from the separate current 343-check/80-replay checkpoint. Archive 99 entries, SHA-256 663c1f52f23bd1a0b655ad9061459e2a23e4bcf0735a0db3665ed8c2e8fa5313. This completes only bounded provider declaration 7.66. Dynamic suggestions, requiredness/defaults, interface precedence, custom argument acceptance and full tool-window parity stay incomplete; actual compiled Groovy workspace attachment 1.9, host authority 1.10, native/physical remote/provider delivery and terms/Kotlin resources/platform/identity/packaging and release gates stay open. No cache/SDK/evidence/executable is deleted, no native app/CI/VoiceOver/release runs, and jvm remains planned.

[Retained qualification](../../../.agents/evidence-archives/groovy-jvm/list-enum-options-20261006/README.md).

### Unambiguous interface option declarations — task7.67

2026-10-06: development provider0.25.0 SHA-256 fe2ad7f8341d4a45c08f6e1f4f516cecdd47af2fa3f688e36f911eda6d914bb1 passes the pinned warnings-denied Java-17 build, fifteen interface reflection checks, thirty existing enum regressions and fifty-one real cached externally denied-network JDTLS/Gradle checks. Unique direct/inherited/diamond interface declarations retain stable metadata without invocation; class/interface and sibling name collisions, including equal sibling declarations, stay incomplete. Static/private interface methods stay incomplete. Exact128-interface reflection admission,129-interface sanitized failure, actual Gradle overflow and repair, enum initialization positive control, prior collisions/budgets/composite/source/configuration/offline/cancellation/failure recovery all pass. Prior provider0.24.0 fails the unique-interface oracle as expected; the negative result is retained. Both fixture-owned homes report no running daemons after checked isolated shutdown. Eleven compile pins match; twenty other archive payloads are byte-identical to0.24.0 and only Builder/manifest change. Current shared/JVM/Copper source equality is separately retained against the actual Groovy software checkpoint, excluding the subsequently committed additive consumer protocol. This is not a new Rust whole-graph/current Copper/lint, installed/default/native/physical remote/CLI acceptance/resource/platform/distribution or release qualification. Archive83 entries SHA-256 71a1153c6fb942de2a7bff40f3fa4a52e97a38f7aa81fe4ae4ffb88ecf538396. Only7.67 completes; full conflicting-interface precedence, dynamic suggestions/defaults/requiredness/options acceptance and full tool-window parity remain7.1/9.1. Tasks1.9/1.13 remain incomplete. No native app, cache deletion, unverified download, CI or VoiceOver runs; jvm remains planned.

[Retained qualification](../../../.agents/evidence-archives/groovy-jvm/interface-options-20261006/README.md).

### Class option annotation precedence — task 7.68

2026-10-06: development provider 0.26.0 SHA-256 3efd39e1304dad382935d762260478fcd3fe62b7fcacb895af5bd90c3f88a52a passes the pinned warning-denied Java-17 build, 56 real cached externally denied-network JDTLS/Gradle checks, 30 existing enum regressions and 15 interface reflection checks. Matching closest-class declarations retain effective descriptions over superclass and interface annotations; equivalent same-signature/same-metadata interface declarations are complete. Distinct setters, private/static hiding and conflicting sibling metadata remain incomplete. Separate pinned Gradle 8.13 help reports confirm both changed-class and class/interface descriptions without executing the fixture argument tasks; help is never a product discovery parser. Prior provider 0.25.0 fails the new class/interface coverage oracle as expected. Exact interface/enum bounds, initialization and invocation canaries, stable facts, composite/source/configuration/offline/failure/cancellation/recovery pass. Checked shutdown reports no workers in the isolated owned home after the current and negative runs. All eleven compile pins match; twenty unrelated provider payloads are byte-identical and only Builder/manifest change. Current Rust consumers remain separately source-bound; Remote provenance 1.16 and compiled workspace 1.9 WIP remain uncompiled or incompletely qualified and are not certified by these checks. Archive 67 entries SHA-256 9224ea9aa878e389d232fd593bfefe920283e318ada048c7af59ff9cbce28106. Only 7.68 completes. Dynamic suggestions/defaults/requiredness, custom argument acceptance, conflicting interface precedence and full parity remain open, as do native/default/physical remote/resource/platform/terms/identity/package/release gates. No native app, CI, VoiceOver, SDK/cache/evidence/executable deletion or release occurs; jvm stays planned.

[Retained qualification](../../../.agents/evidence-archives/groovy-jvm/option-precedence-20261006/README.md).

### Distinct Gradle option-name quota — task7.69

2026-10-06: provider0.27.0 SHA-256 b7e3c349ceedab57076d8eab5824df2c5a8dd3909ebe03a9544fba558689be4b passes warning-denied Java-17 compilation,60 real cached externally denied-network JDTLS/Gradle checks,5 exact option-name helper checks,15 interface checks and30 enum regressions. Exactly128 distinct names admit repeated nearest-class/superclass/interface overrides without losing effective descriptions; a new129th name fails before mutation. Actual Gradle overflow returns a sanitized category and repaired discovery recovers. Provider0.26.0 fails the new exact-limit helper as expected; its failure remains retained beside completed7.68. Existing collision/precedence/canary/stable/composite/source/configuration/offline/cancellation/failure recovery pass. Checked isolated shutdown reports no owned Gradle daemons. Eleven compile pins and frozen source bindings match; twenty other payloads are byte-identical to0.26.0, with only Builder/manifest changing. Archive73 entries SHA-256 5547d116018124b3cfc34a0f9c8a3505fab6b82bbdcaf354f76ab3dda02c72c3. Only7.69 completes. Existing Rust consumers are unchanged by this quota correction; uncompiled Remote1.16 and incomplete workspace1.9/current Copper/lint remain separate open work. No native/default/physical remote/custom argument acceptance/Kotlin resident-resource/platform/distribution/identity/package/full parity/release qualification is claimed. No native app, CI, VoiceOver, unverified download or cache/SDK/evidence deletion occurs; jvm remains planned.

[Retained qualification](../../../.agents/evidence-archives/groovy-jvm/option-budget-20261006/README.md).

### Provider quota after upstream integration — 2026-10-06

2026-10-06 postmerge: incoming origin/dev0a4f9d98ee integrates without conflict at217999309744edf081d6f3c9c9cee0e9bb768498 after provider milestoneedb5e9b9c8. All60 pinned cached externally denied-network provider checks pass again with frozen Java/probe source; checked fixture-owned shutdown reports no Gradle daemons. All five existing Rust/Cargo WIP contents retain exact SHA. Received shared Editor/Proto/two Clippy-profile evidence matches current incoming source bindings and its archive hash; it is source-owned evidence, not newly executed current Groovy/Copper/WIP qualification. Both canonical validators, Python parsing, formatting and diff pass. Current1.16 compilation/runtime/Clippy and corrected current Copper1.9 remain blocked by insufficient capacity: minimum observed free7,824,773,120 bytes, below the unchanged 8 GiB floor plus 1 GiB focused initial margin or5GiB broader initial margin. No reserves lowered or files deleted. Archive20 entries SHA-256 158196ff182c6382bf5dbaad905f62ca8ec8b11d071582f734b29c70ce6d0f92. Full tool-window/native/physical remote/provider delivery/terms/Kotlin resources/platform/identity/package/CI/release gates remain open; jvm planned, Actions stopped, VoiceOver deferred.

[Postmerge qualification](../../../.agents/evidence-archives/groovy-jvm/option-budget-postmerge-20261006/README.md).

### Resumed shared-owner integration — 2026-10-06

2026-10-06 resumed integration: checkout fast-forwards from064c855084241a1bf6594e44385b79a001b55ead to incoming origin/dev360926e4d28761b497d13b1c7f71a5b091cecf61. Shared Editor caret/edit/inlay changes and Project InlayId::BlockBoundary arrive through their existing owners; Agent UI draft/context/reset changes preserve their own checkpoint qualification. All five Groovy Rust/Cargo WIP files retain exact SHA, and provider0.27.0 source bindings remain byte-identical. Seventeen received evidence archive containers and383 locally retained members match their recorded SHA/size;113 equivalent prior-evidence references are separately identified. Received draft IO/error-isolation/reset-input runtime checks are explicitly unexecuted, not green. Historical owner lints/static/runtime results retain their exact source scope. Current Groovy/Copper regression, Remote1.16/unit/lint and workspace1.9/current Copper/lint builds are not executed: resumed free capacity is about1.5GiB, below9GiB focused or13GiB broader initial requirements; bounded Java replay also remains below its5GiB initial margin. No native app or new candidate is launched. No code/cache/SDK/binary/evidence is removed and no reserve lowered. All task states and full parity/release gates remain unchanged: jvm planned, Actions stopped, VoiceOver deferred. Rebuild current graphs and request a new exact candidate approval only after a changed native artifact is concrete.

[Integration reconciliation](../../../.agents/evidence-archives/groovy-jvm/resumed-shared-integration-20261006/README.md).

## Current shared channel runtime checkpoint — 2026-10-09

2026-10-09: safely fast-forwarded origin/dev to eeb85f875d7e8f4666a2b18aec47157f3605d8c5, preserving the three JVM UI file hashes and the sole product_flavor lock delta. Incoming Doctor/restart APIs and Groovy scoped factory are retained together in the existing RemoteClient. Initial compilation exposed the missed reconnect stream conversion (E0308); the existing unscoped reconnect now stamps peer zero. Frozen-source corrected Remote compilation passes in 13.57 seconds. All 52 unit checks pass with zero ignores, and nine GPUI channel/Doctor/reconnect/stream fixtures pass 20 scheduler seeds (180 executions; six additional ordinary test functions pass once). Shared unscoped/Copper channel compatibility, held old-scope requests, separate original_sender_id, response routing, stream cleanup and idle restart sealing remain covered. The retained executable is c739737f0886074e1d2bfe5bcefb0020d02797c0f0a9b9ed02b9b9c98cdc190f. Formatting/diff and both canonical validators pass. Lint preflight refuses before any compiler starts: 9565118464 free bytes is below the unchanged 8 GiB floor plus 1 GiB initial margin (9663676416 bytes). No lint pass or whole current Copper/workspace/native/physical remote qualification is inferred. Task 1.16 remains partial;1.17/1.14/1.15/1.10 stay pending. Existing workspace1.9 WIP and every parity/release gate are preserved. Immutable archive 31 entries SHA-256 7e436851ad4cfe73073fb6abc559d29ab66619792630ced10ef2fcbd0a21594f. Actions stopped, VoiceOver deferred/off, jvm planned; no app or release starts. [Retained evidence](../../../.agents/evidence-archives/groovy-jvm/receiving-connection-scope-runtime-20261009/README.md).

## Maven ignore-mask reference audit — 2026-10-10

2026-10-10: current official IDEA 2026.2 documentation confirms Maven ignored POM patterns, retained ignored projects and Gradle orphan-module restoration. Five original source blobs from the earlier pinned reference 2f55a79617872c5774aa993087f353b8a610ec5e match their Git SHA; its exact LICENSE.txt and NOTICE.txt are retained. Original PatternUtil/Strings source compiles in an isolated JDK17 probe with minimal API stubs; all 27 bounded common-mask cases pass after correcting a missing Logger fixture overload. The initial fixture compile failure is retained. Wildcards cross separators; literal bracket/brace/pipe/plus/dot masks and whole-path case-sensitive matching differ from shared PathMatcher glob classes and literal prefix/suffix matching. This is an isolated pinned algorithm/JDK regex oracle, not an IDEA runtime, Groovy provider, import or shared Rust pass. Backslash masks, host-path normalization, physical platforms, bounded product admission and complete ignore/orphan behavior remain unqualified. No compiler-provider, app, Gradle worker, download payload, Actions or VoiceOver starts; jvm remains planned. [Retained source/oracle evidence](../../../.agents/evidence-archives/groovy-jvm/maven-mask-reference-20261010/README.md).

M05/G07/G09 and task 7.3 remain incomplete: current Gradle request/client support only the selected initialized root without requested exclusions; Maven settings have no pattern property. Complete retained-ignored rows, required-library/orphan handling, settings/restart and local/remote host application remain open under 7.3/8.7/7.4. Provider terms/default delivery, native/physical/platform/resource/package/full parity and release gates are unchanged.

## Maven mask desired-setting source checkpoint — 2026-10-10

The current source adds Maven `ignored_pom_patterns` through shared settings and refuses nonempty application until matching/provider/lifecycle qualification. The desired list persists for review; an empty override clears inherited values. Shared SettingsStore, JVM admission and real FakeLSP default-startup refusal/clearing fixtures are authored but uncompiled and unrun; no successful product interpretation or native journey is claimed. This supersedes the prior source finding that the property was absent, while M05 and task 7.3 remain incomplete. Existing exact-directory selection and Gradle behavior are preserved by intended compatibility checks that still require execution. Tasks 7.70–7.73 separate persistence, bounded matching, real M2E application and existing LSP replay. Current Rust/Copper compilation/runtime/lint remain held behind the shared build owner. Actions stopped, VoiceOver deferred/off, jvm planned; no release.

## Isolated Maven setting-code check — 2026-10-10

Three isolated exact-production schema/merge/fallible-parser/intent checks pass with warnings denied, including the authored mask refusal case. A cached Serde profile mismatch caused the retained initial compile failure; only harness library selection changed. Source and dependency hashes remain frozen. This provides no shared SettingsStore, FakeLSP, Copper owner, provider-application, native or platform verdict. Existing raw settings-error admission remains essential because fallible parsing may retain a value with Failed status. Task 7.70 stays partial; matching/application/replay 7.71–7.73 and full M05/G07/G09 parity remain open. The root's fixed-reserve failure retains the shared heavy hold. Actions stopped, VoiceOver deferred/off, jvm planned; no release.

### Cached Maven profile guard follow-up — 2026-10-10

Development provider0.28.0 SHA-256 53318dc76f8a0a5581b414280f5288d810b48ee2c14d6443bf681fc41c9f1fda compiles to Java-17 bytecode with warnings denied against eleven verified pinned inputs. Ten direct production Reader checks pass: exact128-ID repeated active override, pre-mutation overflow refusal, malformed/control text, absent original model and cancellation in both loops. Three exact old production-loop cancellation controls fail as expected; the corrected loops pass all three. Source bindings match; only Reader/manifest change from0.27.0, twenty other payloads remain byte-identical. No JDTLS/Gradle worker, native app, Rust compiler or download launches. Real externally denied-network cached Maven/Gradle compatibility runs remain held by the shared coordinator and required before completion; current Rust/Copper WIP is separately unqualified. Keep7.74/7.2 and full parity/release gates open. Actions stopped, VoiceOver deferred/off, jvm planned.

M02 remains incomplete: cached POM/parent profile origins do not qualify user/global settings definitions, explicit/default profile controls, profile-changing reimport or configuration-specific profiles. [Maven inheritance semantics](https://maven.apache.org/guides/introduction/introduction-to-profiles.html#Profile_Inheritance) and [IDEA profile controls](https://www.jetbrains.com/help/idea/work-with-maven-profiles.html) remain acceptance inputs for existing7.2/7.4/7.5/9.2 owners. No parity requirement is waived. [Retained partial evidence](../../../.agents/evidence-archives/groovy-jvm/maven-profile-budget-20261010/README.md).

## Resumed validation — 2026-10-10

Task **7.74 is complete** for bounded cached Maven profile projection: pinned
development provider **0.28.0** (SHA-256
`53318dc76f8a0a5581b414280f5288d810b48ee2c14d6443bf681fc41c9f1fda`)
passes the prior warnings-denied Java 17 build, ten direct checks, three retained
expected old-loop failures and **45 Maven / 60 Gradle** real cached compatibility
checks with external network denied. Source bindings are unchanged; owned workers
are absent after cleanup. Full profile control/reimport and task 7.2 remain open.

Normal Copper Remote lint passes; task 1.16 still awaits its current runtime
rebuild. Reader1.18 and current shared SettingsStore/FakeLSP/default-consumer
runtime/Copper gates remain open. Approved **0e33** native Picker typing preserves
the query after Down; its diagnostic tree records the selected named command as
the ComboBox active descendant and retains Tests/Dependencies/Debug control names.
This is narrow old-candidate evidence. Native provider/build/test/restart,
discard-state, worker-RSS, screen-reader, current-WIP, physical remote and package
qualification are not inferred. The app quit and fixture disk/settings stayed
unchanged; a preexisting unsaved Greeting buffer was observed.

Full reference parity and all release gates remain open. GitHub Actions stopped,
VoiceOver off, `jvm` planned; no release. Evidence:
`.agents/evidence-archives/groovy-jvm/resume-checks-20261010/`
(39 entries; SHA-256
`7b0db3fef3ac4689b4e884e91d9e471fd9da14ce76bafb5aa0cd1358e32b4509`).

## Current Remote completion — 2026-10-10

**Task 1.16 is complete** for receiving-connection provenance. Current
normal-Copper Remote source rebuild passes **52 unit tests, 0 ignored**;
the channel filter passes nine GPUI cases across 20 scheduler seeds
(180 executions), plus six ordinary cases once. Current warnings-denied
repository Remote Clippy and unchanged-source formatting are retained.
Source/executable/reserve and matching queue release guards pass.

Native-reader lifetime, JVM host authority, the current shared-owner graph,
native/physical remote, full reference parity and release gates remain open.
GitHub Actions stopped, VoiceOver off, `jvm` planned; no release. Evidence:
`.agents/evidence-archives/groovy-jvm/remote-current-runtime-20261010/`
(13 entries; SHA-256 `6caa011647cb33055b4598d213c072427823dc1f3df902551024c56071501c01`).

## Provider delivery-source refinement — 2026-10-10

Task1.1 has additional official publisher-usage and immutable installation-source
evidence in the provider matrix and
`.agents/evidence-archives/groovy-jvm/provider-delivery-source-20261010/`.
No provider binary, pin, default, native candidate or reference capability changes.
Artifact-specific terms/notices, default delivery and all native/remote/platform/
resource/Debug/parity/release qualification remain open. This source review does
not mark a parity row implemented. Actions stopped, VoiceOver off, `jvm` planned.

## Bounded uncommitted-task delivery — 2026-10-10

Canonical leaves **1.9, 1.18 and 7.70** complete their specified local software
boundaries. This supersedes their earlier dated incomplete checkpoints; all
negative controls and failed attempts remain retained. It does not complete a
milestone, native workflow, full tool-window parity or release gate.

- **1.9:** catalog-selected Groovy workspace initialization attaches the existing
  local Project/LspStore default lifecycle outside panel rendering. Both actual
  compiled development identities (`rust` and opt-in `jvm`) pass all **77 JVM UI
  tests and five catalog checks**, plus **ten GPUI fixtures at 20 iterations each**
  per identity. Fixtures prove Kotlin-only Maven/Gradle metadata without a Java
  buffer or panel, trust gating, idempotence, Copper zero-Java startup and
  collaborative refusal. These use FakeLSP, not downloaded default providers.
- **1.18:** the existing accepted-connection driver retains its input reader Task;
  six framing/lifetime/isolation fixtures pass **20 iterations each**. Existing
  remote editing, profile transport, command, log and Doctor regressions pass.
  The separately archived original detached-reader controls retain their five
  expected failures and malformed-frame pass. Physical sockets/SSH, producer
  provenance and a generic frame-allocation bound are not certified.
- **7.70:** optional Maven masks persist through the existing SettingsStore;
  absence inherits and explicit empty clears. Two model/selection checks pass,
  two settings fixtures pass **20 iterations each**, and twelve default-lifecycle
  fixtures pass **20 iterations each**, including sanitized refusal and recovery
  after clearing. Nonempty masks remain explicitly unsupported. Matching,
  M2E application and lifecycle replay **7.71–7.73** remain unstarted; full
  M05/G07/G09 and ignore/orphan parity remain incomplete.

Affected repository `./script/clippy` passes locked/offline, release,
all-targets/all-features and warnings denied for Project, JVM UI, settings-content,
diagnostics and RemoteServer. The separate non-agentic RemoteServer lint passes.
A minimal conditional Doctor loop-variable correction unblocks the pre-existing
non-agentic unused-variable error without changing controls, IDs or actions.
Panel test repairs retain editing buffers and await current provider facts;
restored Gradle wrappers use the existing explicit synchronization owner.
The actual Groovy replay additionally exposed seed16 asserting partial coverage
while metadata was still stale; a bounded GPUI-timer wait retains the same final
partial/current-row/navigation oracle. The original failure is retained alongside
the corrected twenty-iteration replay under both compiled identities.
No production panel/LSP retention workaround or new infrastructure is added.

Focused Copper qualification passes four existing panel fixtures at **20
iterations each**, all **45 language-tools** tests and all **38 settings** tests.
The broader Cargo UI run is **79 passed / nine failed** and remains open: eight
selected-profile native cases require the actual SDK/signature/architecture/
dependency-qualified installed App; `cargo_dashboard_foreground_budget` reports
off-thread activity outside GPUI's scheduler (not an exceeded foreground budget).
These unchanged Cargo/profile owners are outside the bounded repair scope.
No whole-Copper native or broad-suite pass is claimed.

Source/SDK/executable bindings, stage exit codes, monitored capacity and matching
queue closure/independent owned-process absence are retained. The approved
isolated **4 GiB disk reserve plus 1 GiB admission margin** is a build guard,
not the proposed Kotlin/editor combined resident-memory qualification. Initial
lint/CLI/fixture failures and one source-changing build remain explicitly
unqualified. Existing dependency future-compatibility and stable-rustfmt option
warnings are retained; unavailable optional tools are not passed checks.

Native default acquisition, physical remote/SSH, full reference parity,
provider delivery/terms, Kotlin resource/platform qualification, production
identity/packaging and release gates remain open. Actions-dependent evidence is
blocked by infrastructure; Actions stays stopped. VoiceOver stays off/deferred.
`jvm` remains planned with empty package targets; no release or app launch occurs.
Only this bounded delivery runs; the broader Groovy goal remains paused.

[Retained delivery evidence](../../../.agents/evidence-archives/groovy-jvm/uncommitted-task-delivery-20261010/README.md).

### Maven phase-fact snapshot transport — 2026-10-16

Canonical7.76 carries only bounded relative imported/ignored Maven phase POM identities through the existing JDTLS snapshot protocol7. Remote facts carry no host settings digest or local startup authority. Current host cached replies recheck generation/server/trust/settings/visibility/membership and the existing8MiB aggregate bound; current client admission shares the existing token/server/worktree/permission path. Older1–6 projections omit facts and mark lost coverage incomplete; absent facts remain unknown, while an empty imported set with nonempty ignored identities remains meaningful. This does not infer required/orphan/full-synchronization state or provide automatic remote startup (1.14/1.15).

Exact-source local software checks pass:78 JVM model,80 JVM UI,21 startup/settings/Copper-buffer lifecycle,15 protocol, two framed JVM and six framed Copper functions. JVM-disabled local Copper Rust passes25 active functions with one existing ignored test; the inventory contains Rust and excludes JVM models. GPUI fixtures use20 scheduler iterations. Remote Copper test dependencies enable Project JVM code through JVM UI, so that behavior cohort is not claimed as a completely JVM-free build. Release/all-target/all-feature warnings-denied Project/Proto/Remote/JVM UI repository Clippy and JVM-disabled Project/Rust Clippy pass. Official protoc preserves697 existing messages with one optional message addition. Real protobuf checks cover absence/known-empty, legacy/unsupported versions,128-ID bounds, malformed/private/foreign/contradictory membership and omitted host digest. The existing framed fixture holds actual facts across supersession and absent client trust authority, then requires a fresh current generation for recovery. Its first instant-channel-close assertion fails because executor cancellation is scheduled; the corrected state-based check passes, with frozen failed/initial/final cohorts retained. Existing stale host input, trust, server recovery and disconnect/reconnect cases remain covered.

These are software fixtures, not current native, physical SSH, default provider delivery, platform/accessibility/resource, full tool-window parity or release qualification. Full selection/required-orphan policy7.3, causal sync7.4, shared-tree policy editing8.7, remote consumer authority1.14/1.15 and all remaining parity/release gates stay open. Actions stopped/infrastructure-dependent gates open, VoiceOver off/deferred, jvm planned; no native candidate launched and no release. [Retained software evidence](../../../.agents/evidence-archives/groovy-jvm/maven-selection-wire-20261016/README.md).

## Startup intent on the current connection — planning, 2026-10-16

Host admission1.14 now has retained software qualification; current client
lifetime1.15 and product attachment1.10 remain pending. The client audit identifies
one missing prerequisite in the existing channel RPC owner: startup grants and
empty withdrawals must not use ordinary reconnect replay. Canonical1.20 retains
that bounded transmission/cancellation/closed-connection boundary, including
positive Copper replay/stream/Doctor regressions; it adds no transport, manager,
wire pair, semantic index, result store, generic panel or public provider SDK.
Requirement6.5 traces that freshness obligation to1.20/1.15. This is planning only,
not implemented or runtime-qualified. All52 reference and four shared matrix
states/owners remain intact. Full parity, native/physical remote/provider/default
acquisition/terms/resources/platform/package/release qualification remains open.
Actions stopped, VoiceOver off/deferred and jvm planned; no release.

## JVM request transmission source checkpoint — 2026-10-16

Canonical1.20's initial four request fixtures pass across20 scheduler seeds in
an isolated Rust1.97.1 cache with frozen source and a4GiB reserve. Review then
found the one-way entry point. One private replay policy now also guards that
path; a fifth fixture preserves ordinary one-way buffering. Final-source focused,
full Remote/Copper/reconnect/Doctor/stream and lint checks remain queued. The
initial four-case pass does not qualify that changed cohort. Client1.15/product1.10
and every full parity/release gate remain open, including the earlier incomplete
integrated graph. No new wire, transport, framework, semantic index, result store
or public SDK. Actions stopped, VoiceOver off/deferred and jvm planned.

## Current-connection transmission software qualification — 2026-10-16

Canonical1.20 completes only the existing ChannelClient transmission/bookkeeping
boundary. Five final fixtures pass20 scheduler seeds; the full all-feature Remote
suite passes82 active functions with one existing ignore. Ordinary send/request
replay, response routing, reconnect, stream cleanup, Doctor and SSH/Docker remain
covered. Release/all-target/all-feature repository Remote Clippy passes with
warnings denied, as do Rust formatting, feature-boundary checks, both canonical
validators, whitespace and manual ownership/decomposition review. Frozen source
and the4GiB reserve are retained. Client1.15 and product1.10 remain separate.
The earlier combined Project/JVM UI/framed integration gate is still incomplete
after its shared cache disappeared; these scoped Remote checks do not qualify
those unrun cells. Full parity and every native/physical/provider/default/
platform/accessibility/resource/packaging/release gate remain open. Actions
stopped, VoiceOver off/deferred, jvm planned; no app launch or release.

[Retained software evidence](../../../.agents/evidence-archives/groovy-jvm/connection-current-jvm-intent-20261016/README.md).

## Remote client current-intent source checkpoint — 2026-10-16

Canonical1.15 now has source and three authored existing-channel fixtures for
bounded current intent, stale/malformed acknowledgments, selection/trust loss,
disconnect/reconnect, timeout, sanitized old-host errors and explicit recovery.
The existing remote LSP owner and Project connection hook retain lifetime; no
client-local provider or new index/store/engine is added. This is uncompiled,
unrun source progress. Product1.10, the incomplete integrated graph and every
native/physical/provider/platform/resource/full-parity/package/release gate
remain open. Actions stopped, VoiceOver off/deferred and jvm planned.

## Remote client corrective checkpoint — 2026-10-16

Task1.15's corrected first runtime cohort passes4 functions and fails one teardown
fixture at seed0. Driving GPUI's app entity-retirement step preserves the exact
one-withdrawal assertion; that single seed replay passes. Six final authored
functions now include inactive isolation and malformed/foreign/oversized scope
refusal. The full final cohort, shared Copper/framed/lifecycle and strict lint
checks remain open after a foreign-build guard refusal before compilation.
The embedded existing RemoteLspStore owns teardown; no client copy, shutdown
worker or new runtime framework. Original compiler/runtime failures and separate
source bindings remain retained. No native/physical/provider/platform/resource/
full-parity/package/release gate passes; Actions stopped, VoiceOver off/deferred,
jvm planned.

## Remote project-only Restart prerequisite — planning, 2026-10-16

Source inspection finds the existing restart RPC dropping all-project scope;
canonical1.21 (host) and1.23 (client) record its correction before product1.10.
Reuse the existing field,
request/ack, host restart and permission owners; preserve Copper buffer/selector
behavior and require current direct authority for leased JVM consumers. Fresh
client intent follows only a matching successful host acknowledgment. There is
no new wire, retry engine, permission service or SDK. This planning entry passes
no runtime/native/provider/parity/release gate. Completed tasks and every matrix
state remain unchanged; Actions stopped, VoiceOver off/deferred, jvm planned.

## All-project Restart transmission prerequisite — planning, 2026-10-16

Canonical1.22 reuses1.20's private existing nonreplay policy for the all=true
restart path before1.21 makes that field effective. An obsolete global restart
must not cross reconnect and clear a newer stop state. Ordinary all=false Copper
buffer/selector replay remains required. This is a separate transport leaf,
with no new wire or runtime owner. All other task/matrix states and release gates
stay intact. Planning only; Actions stopped, VoiceOver off/deferred, jvm planned.

## Current focused client proof and independent restart source — 2026-10-16

Task1.15's six focused client functions pass20 seeds; its prior failed seed0
passes separately. Current full shared/Copper/framed/UI/lint checks remain open
after foreign-build preflight refusals. Task1.22 independently adds all=true
restart control to the existing private nonreplay predicate, with two authored,
uncompiled channel fixtures and positive ordinary all=false replay/routing
controls. No new infrastructure or runtime qualification follows from that
source.1.15/1.22 remain incomplete and1.21/product1.10 remain pending. Actions
stopped, VoiceOver off/deferred, jvm planned; all full parity/release gates open.

## Restart owner reconciliation — planning, 2026-10-16

The manual task audit splits the pending restart follow-up before implementation:
host1.21 owns admitted global restart, transport1.22 owns current-connection
transmission, client1.23 owns matching-acknowledgment intent refresh through1.15's
existing bounded owner, and product1.10 attaches the completed outcome. No new
framework, copied client or retry engine; ordinary Copper behavior remains
required. Completed task/matrix states and all release gates stay intact.
Planning only; Actions stopped, VoiceOver off/deferred, jvm planned.

## Available-provider client proof and framed correction — 2026-10-16

All seven current1.15 client functions pass20 seeds with Java/Kotlin lazy-language
and binary traps present but never invoked. A failed disabled withdrawal now
retires its fingerprint; unrelated settings cannot restart disabled tooling.
Earlier combined Remote84/one existing ignore, lifecycle30, JVM model78 and
host-admission framed2 pass. The next Java metadata fixture fails seed0 on the
historical immediate-refusal expectation. Its corrected oracle preserves all
metadata/permission checks and requires host refusal, no accepted scope/local
language load/host startup, plus actionable failure before disabling isolated
intent. Corrected framed replay and remaining shared/lint checks are open.
Source/failure cohorts stay separate;1.15/1.22 remain incomplete, Actions stopped,
VoiceOver off/deferred and jvm planned. No native/full-parity/release claim.

### Current host restart source — unqualified, 2026-10-16

Canonical 1.21 now has source for current-connection global restart through the
existing host LSP owner, including revalidation after shutdown and precedence
for newer stop/restart or trust changes. The existing all-stop latch blocks
acquisition during the wait; a weak operation identity retains no runtime owner.
No-buffer recovery and held-shutdown negative cases are authored alongside the
existing framed Java/Kotlin owner-sharing fixture. They are **uncompiled/unrun**
at the approved disk-capacity gate. Client 1.15's remaining shared checks and
host 1.21's runtime/Clippy checks remain open; 1.23 and product attachment 1.10
remain pending. No native app runs. Actions stay stopped, VoiceOver stays
off/deferred, full parity remains unverified and jvm remains planned.

### Initial-settings gap confirmed — 2026-10-16

The original no-wait saved-selection fixture sends an actual first JDTLS
Initialize with Maven enabled, then a replacement with Maven disabled. Its
seed0 diagnostic fails and restores the exact fixture. The prior fixture-only
explanation is superseded; task1.24 owns a barrier in the existing settings/LSP
owners. Both strict shared and JVM-disabled Copper lint configurations and319
active functions pass on their bound pre-correction cohorts, but they do not
qualify this initial-load failure.1.15/1.21 stay incomplete,1.23/1.10 pending.
No native or parity/release gate passes. Actions stopped, VoiceOver off/deferred,
jvm planned.

Current corrective1.24 replay: seven focused software functions pass exactseed0
and20 scheduler seeds with the fixture-side wait removed. The tests cover first
Java defaults/explicit selections/invalid/unreadable settings, explicit Kotlin
fallback, actual no-binary acquisition, ordered reads, removed roots and save
recovery. The first selected-Maven oracle is corrected to preserve7.73's disabled
automatic importer for explicit selections; an unrestricted default stays enabled.
The serial audit parser failure is retained separately; broader current shared
checks and strict lint remain in progress.1.15/1.21/1.24 remain incomplete. No
native, physical, alpha/resource, full-parity or release claim follows.

### Current lifecycle software qualification — 2026-10-16

Tasks1.15,1.21 and corrective1.24 qualify their software acceptance. Java/Kotlin
startup waits for the existing current folder settings reads before acquisition
and Initialize. Unreadable files retain last-good values and refuse startup;
readable saves recover and removed-folder work cannot restore settings. The
existing host restart owner rechecks connection, intent, provider/trust and
manual-action scope after shutdown. Existing Copper behavior is retained.

The regression matrix passes327 distinct functions with two existing ignores
before lint identifies one redundant production clone. Exact comparison verifies
that moving the value at its last use is the sole production correction;28 other
bindings match at that stage. Six focused cohorts pass55 distinct functions on
final production bytes. Lint then identifies two fixture-only cleanups: an unused
setup lookup and a no-op path conversion. Their exact delta preserves acceptance
oracles; all8 affected integration functions and both strict repository Clippy
configurations pass afterward. GPUI lifecycle sweeps use20 seeds; seven separate
exactseed0 replays precede the move. Formatting, feature boundaries, both canonical
validators and manual ownership/decomposition audit pass.
The original startup failure, mistaken selection oracle, both lint failures and two
audit-parser interruptions remain retained under their actual bindings. The
327-function matrix is not claimed as fully rerun on final bytes. Evidence
archive SHA-256 `2ce13cc5896c8c356614f18475897dec17c7f6bddcde52ddbfea01807d01f3c4`.

Client post-Ack1.23 and remote product attachment1.10 remain pending. No native
app launched; native/physical/provider/resource/platform/full-parity/package/
release gates stay open. Actions stopped, VoiceOver off/deferred, jvm planned.

### Current client restart acknowledgment qualification — 2026-10-16

Task1.23 qualifies current remote restart through the existing client request
task, generation and withdrawal owner. Live host leases survive while client
admission is cleared; current Ack forces fresh intent. Changed settings, trust,
roots, connection and manual actions, failure/timeout, cancellation and teardown
cannot revive old work. Selective completed admission preserves other-provider
ownership; off/empty JVM scope retains ordinary Copper buffer restart. No fields,
protocol, copied client, retry manager, store or execution engine are added.

All32 focused runtime functions pass20 seeds, and seven new client functions
pass exactseed0 separately. The existing scoped framed fixture now drives actual
client intent, bufferless host restart and fresh admission, with changed Java/
Kotlin host IDs, no client fake provider and verified teardown. Its prior owner/
quota/revocation assertions remain. Initial fixture assumptions about the normal
RemoteStarted handshake and empty reconnect withdrawal are corrected with failures
retained. Both strict repository Clippy configurations, whole formatting, feature
boundary, both canonical validators and manual ownership audit pass. The factory
used by both fixtures is test-support+jvm-tools only and delegates to existing
constructors. Evidence80 entries, SHA-256 `3ef1685f3cdb16fb4ef10a597b7aab255e7d3bd80897723e6d2af1ee2206cb70`.

Ack proves neither provider readiness nor that an older host honors all=true.
Product remote attachment1.10 and all native/physical/provider/default-delivery/
alpha-resource/platform/full-parity/identity/package/CI/release gates remain open.
No native app launched. Actions stopped, VoiceOver off/deferred, jvm planned.

### Scoped mounted mock qualification — 2026-10-17

Task1.25 supplies scoped test construction for product/workspace attachment1.10.
The existing MockRemoteConnection optionally binds a fixed nonzero receiving peer
for its proxy/input lifetime. Normal factories retain None/peer-zero behavior.
Both modes reuse the same registry, input loop, ChannelClient replacement task,
responses and Doctor activity. Supplied relay identity stays separate. Seven
established constructors and production channel fields remain byte-identical.

Four actual Mock/RemoteClient functions pass exactseed0 and20 seeds, including
held original scope, response routing, busy/idle seal and zero-scope refusal. Full
Remote passes88 active functions with one existing ignore; seven Copper/Doctor
headless functions pass20 seeds. Strict repository Remote Clippy, whole formatting,
feature boundary, both canonical validators and manual ownership audit pass.
All30 source bindings match. Archive66 entries, SHA-256 `22771a9530a61442b22fb7316b80c96df288b42d7809c9127ac2a1a96e8c62f2`.

This is software test transport evidence. Product1.10 and native/physical SSH,
provider/default-delivery/alpha-resource/platform/full-parity/identity/package/CI/
release gates remain open. No native app or real remote-server process launches.
Actions stopped, VoiceOver off/deferred, jvm planned; no release.

### Remote attachment and Maven origin software qualification — 2026-10-17

Task1.10 is complete at its existing software product-attachment boundary. The
compiled-jvm Workspace observer now accepts direct remote projects, excludes
collaborative guests and uses the same LspStore setter and current host scope,
trust/settings/start/stop/restart/status/retry owners. Other products retain
false defaults. The catalog dependency is test-only; one lock relation changes.
No runtime transport, parser, framework, index, execution engine, store or SDK.

Nine commands/24 distinct functions qualify cohort A. After a private new test
fixture URI/path portability correction, the three affected functions rerun20
GPUI seeds under actual rust and guarded jvm identities on final B; strict
affected-owner lint and static checks pass. Inverse source review proves all
product code, existing functions and39 other guarded inputs unchanged. The
whole24-function cohort was not rerun on B. Initial exactseed0 checks remain a
separate cohort. All artifacts bind their actual sources. Archive168 entries,
SHA-256 d8d80280deae895c408c3e77dc0ea6c3c5217c5615acc043ca0ac6d6201159a1.

Task7.2 remains partial. Existing Maven parent-origin metadata uses accurate
shared-row wording while retaining opaque IDs, source/child activity and action
owners. Existing metadata and mounted provider fixtures pass both identities;
whole-file inverse review checks the narrow wording/identity delta. Supplemental
archive4 entries, SHA-256 b9b36c666bbf02e2b35b4d743c20c9c1c0faee7b991ba1a99a138916e05577e6.

This is scoped mock/Project/Workspace/FakeLSP software evidence. Actual older
hosts/SSH/provider processes/default downloads and terms, native rendering,
editor-plus-owned-worker resources, physical platforms/accessibility, full
Gradle/Maven parity and packaging/CI/release gates remain open. No native app
launched. Actions stopped, VoiceOver off/deferred, jvm planned; no release.

### Gradle compiler-output provider qualification — 2026-10-17

Task7.78 completes only the additive configured annotation-processor output
provider boundary. Development0.31.0 SHA-256
44b72c7d103eaab88a6fa659f81401e24bb1336fab781f33b0153d3f3618a4f8
passes warning-denied Java17 compilation,16 direct fresh/restart SDK cases,
67 real cached externally denied-network model checks,35 source-output/same-
profile restart checks,45 Maven compatibility checks and five CLI guards.
16 old payloads are byte-identical; five others retain identical instruction/
member listings with shifted debug tables. Only Builder behavior and manifest
version change. SDK cleanup failures and their owned-thread diagnosis remain
retained; final fixture closes existing SDK project/global service owners and
exits naturally. Dedicated homes stop only their owned workers;4GiB floor holds.
Archive203 entries SHA-256 228320a45010db6ec459662826d15ce31b597034fa6b1bb97aa03cc72244de87.

The known/nullable URI pair describes configured JavaCompile annotation-
processor output, including absent files and known absent properties; other
tasks are unknown. Existing confined outputUri, JSON request, trusted Gradle
configuration, offline/cancellation and host lifecycle remain owners. No Task
action, IDEA plugin, OptionValues getter, index, SDK or execution engine is
added. No shared Rust/Copper input changed from separate4c8388a569 evidence.
Local admission7.79, wire7.80 and tree7.81 remain pending; root7.1 remains
partial. Successful generation, arbitrary Java/Kotlin generators, installed/
default delivery and terms, native/physical remote/resources/platform/AT, full
parity and packaging/CI/release gates remain open. No app or release launched.
Actions stopped, VoiceOver off/deferred, jvm planned.

### Gradle compiler-output shared tree qualification — 2026-10-17

Task7.81 completes the existing Dependencies consumer for declared annotation-
processor output. One stable role per source set distinguishes unknown, absent
and declared output; ordinary source identities and other-generator uncertainty
remain. Only a current provider-navigable public directory activates the existing
Project reveal event with its current entry ID. Missing/wrong-kind/stale/private/
external/FIFO paths stay inert; unsafe ancestry removes path-bearing labels.
Project entry changes invalidate through the same retained active-panel owner.
Manifest opening, language_tool_tree, ProjectPanel and all shared owners retain
their existing behavior; no new index/store/engine/framework/SDK or permission path.

<!-- impl: crates/jvm_ui/src/dependencies_panel.rs#project_rows_for_view -->
<!-- impl: crates/jvm_ui/src/dependencies_panel.rs#source_directory_status -->
<!-- impl: crates/jvm_ui/src/dependencies_panel.rs#activate_selected -->

The extended mounted provider fixture passes exactseed0. Final103 distinct
functions pass:80 JVM UI,11 shared tree,7 framed JVM/product and5 Copper remote;
GPUI20 seeds, ordinary once. Stable row/AccessKit IDs, selected-row focus metadata,
keyboard Enter/reveal events, explicit refresh, no provider query/editor item on
activation, external-symlink refusal, recovery and stale/trust refusal pass.
Existing Maven/Kotlin/Tasks/status controls and Copper-no-Java checks pass.
Strict JVM UI/RemoteServer release/all-target/all-feature warnings-denied lint,
formatting/boundary/both canonical validators/manual decomposition pass. All43
bindings match;41 other owners unchanged.4GiB reserve holds. Archive100 entries
SHA-256 c1ab48afac2290050bfee218f5cbb925964befa821dd0e689e2070305b334ab3. Initial API/type, incorrect host-predicate, preflight-count
and FakeFs symlink-cleanup failures remain retained; final test uses existing
in-memory rename, no real filesystem deletion.

This is software provider/tree/framed evidence, including the Project reveal-event
contract. Native rendering and physical ProjectPanel focus, real generation/editing,
SSH/platform/accessibility/resources/provider delivery/terms/full parity/package/
CI/release gates remain open. No app or real JVM/build worker launched. Actions
stopped, VoiceOver off/deferred and jvm planned; no release.

### Default Maven candidate model qualification — 2026-10-17

Task7.82 completes the typed model boundary. A current Worktree snapshot can
supply up to32 standard portable Maven POM input candidates through existing
JVM discovery, retaining its65,536-entry scan bound. Nested/independent candidates
remain distinct; private/ignored/hidden/aliased/unavailable entries are omitted.
Exact exclusions and explicit-empty/absent intent retain their meaning. Masks and
effective reactor/module interpretation remain with M2E. Settings are unchanged.

<!-- impl: crates/project/src/jvm_project.rs#default_maven_import_inputs -->
<!-- impl: crates/project/src/jvm_project.rs#SavedMavenSelection -->
<!-- impl: crates/project/src/jvm_project.rs#MavenSelectionFacts -->

Existing phase facts can carry an optional local candidate fingerprint. Added,
removed or unavailable public candidates retire obsolete classifications through
the same validation/admission owner. Existing wire payload omits this digest and
settings; input identities are not a full content or synchronization proof.
Runtime callers still supply no default context; mask-only/default exclusion
admission and application remain7.83. There is no new index/store/SDK/engine,
persistence migration, UI framework or Cargo model.

MatrixA passes93 distinct functions:85 JVM models,5 lifecycle/Copper and3 mounted
provider/UI, GPUI20 seeds and ordinary once. A final guard makes discovered names
use the existing explicit portable-path validator. After that sole helper/new
fixture update, both affected functions pass exactseed0 and20 seeds plus strict
Project release/all-target/all-feature lint. Whole-file inverse proof preserves
all other code and42 inputs. The whole93 did not rerun on final source. Whole
formatting/boundary/both canonical validators/manual decomposition pass. All43
final bindings match;4GiB floor holds. Archive113 entries SHA-256 7b8680be9d3aa345fdb039a05fb17720b2564aab9026752b8dceec4f99110366.
Initial lifetime compile and1pass/1fail control-name fixture remain failed: control
characters were already omitted by visibility; a visible bracketed name exercises
the final portable-path guard. No failed aggregate is relabeled passed.

This is typed software-model evidence. Full7.3 ignore/orphans and Gradle policy,
7.4 causal reload,8.7 controls, actual provider/default delivery/terms/native/SSH/
platform/accessibility/resources/full parity/package/CI/release remain open. No
app, real JVM/build worker or release launched. Actions stopped, VoiceOver off/
deferred, jvm planned.

### Provider-owned Maven default-root discovery — 2026-10-17

Task7.84 adds `groovy.maven.defaultImportRoots` to the existing development JDTLS
bundle0.32.0. It accepts at most32 current canonical POM candidate URIs under one
initialized host folder and returns default roots plus separately retained linked
Maven roots. It runs no importer, reads no POM contents and changes no preferences
or projects. M2E remains the reactor/module owner; discovery is not import or full
synchronization. The pending typed admission7.85 and lifecycle7.83 must use this
read before applying policy. Installed/default delivery remains unqualified.

<!-- impl: tooling/jvm/gradle-model/MavenImportHandler.java#defaultImportRoots -->

The pinned JDTLS1.61.0 SDK's actual BasicFileDetector predicate, default depth5
and Maven/non-Maven project ownership supply semantics. The bounded production
query does not recursively scan. Lazy access to its private predicate/depth must
succeed or discovery refuses actionably; there is no locally copied glob engine.
The immutable upstream source commit is08eafe6ff60c7159ef88571d47b6a9ef82fef94e;
SDK SHA-256 b382afc38a89e41959a56aee532ae567c7f804c3e4d4f13444114468789ec032.
Source/API provenance and EPL-2.0 headers are retained as research evidence;
this does not authorize bundled-server redistribution.

Real headless JDTLS fresh/restart baseline and negated-exclusion journeys compare
bounded responses with the actual SDK scanner on disposable fixtures.83 checks
pass: nested unlinked refusal, independent/invalid-XML roots, exact depth boundary,
built-in and user exclusions/negation, already-linked descendants, non-Maven
ownership and recovery, empty sets, malformed/foreign/alias/size/count cases,
sanitized failures and pre-cancellation. POM/settings bytes and project membership
are preserved by the read; automatic Maven import remains disabled. External
network is denied. The surprising SDK rule that negated exclusions permit walking
excluded ancestors is preserved and tested rather than reinterpreted.

All100 existing Maven reactor/mask/import/restart checks and45 legacy cached
Maven metadata checks also pass.19 other class
entries, including every Gradle owner, remain byte-identical; Maven legacy source
is identical after removing only the new dispatch/imports/methods. The matcher
class changes debug line metadata, while disassembled instructions are identical.
The initial overstrict byte-equality assertion remains a failed comparison in the
evidence. Both passed discovery cohorts and all failed/provisional lifecycle
checks are retained separately; none establishes completed7.83.

No GUI app or Gradle worker was launched for this discovery probe. Native/SSH,
physical/accessibility/resources, full ignore/orphan/reload/offline parity,
provider delivery/terms/platform/package/CI/release gates remain open. Actions
stopped, VoiceOver off/deferred and `jvm` planned; no release.

### Bounded default Maven policy through shared LSP ownership — 2026-10-17

Tasks7.85 and7.83 add typed admission and local application of saved Maven masks
and exact exclusions when `selected_poms` is absent. Explicit empty selection,
explicit roots, user/project override clearing and other providers retain their
meaning. Startup disables automatic Maven import through the existing settings
barrier before controlled phases. An older provider with mask support but no
`groovy.maven.defaultImportRoots` capability refuses this action; no fallback
imports every visible POM. Explicit selected roots remain available.

<!-- impl: crates/project/src/jvm_project.rs#MavenDefaultImportRootsQuery -->
<!-- impl: crates/project/src/lsp_store.rs#execute_selected_local_jdtls_phase -->

The existing Worktree discovery supplies at most32 current portable POM input
candidates. The typed query binds the original worktree ID even for an empty set,
root, exact version/flags and unique current candidate subsets. It distinguishes
provider roots/retained roots from M2E's actual imported/ignored module facts.
Byte/count/path/partial/scope contradictions, missing/private/noncandidate roots,
absent capability, missing data and provider errors refuse actionably without
exposing provider error causes. Current candidate validation reuses discovery.
No project index, generic matcher, store, SDK or execution owner is added.

The existing LSP operation/task/token owns reads, cancellation and mutations.
It queries current provider roots before remaining phases, then checks the same
server, worktree, trust, saved JVM intent, host JDTLS settings, candidate set and
operation token before acting. Default candidates execute containing directories
first; a nested phase is skipped only when a preceding M2E receipt already
reports that POM. Unrelated nested POMs are never implicitly linked. Current
MavenSelectionFacts retains only existing imported/ignored facts and a private
local candidate fingerprint; wire/settings serialization remains unchanged.

An initial1pass/2fail mounted check exposed unnecessary nested phases before the
containing reactor. Parent-first execution fixes it without inferring membership.
A later1pass/2fail held-response fixture reproduced a second bug: dynamic Java
LSP settings can change without replacing the server. Comparing effective host
JDTLS LspSettings through the existing settings owner now rejects the old reply
before mutation. Both failed cohorts and the earlier provisional82pass/1fail UI
cohort remain retained. No failed aggregate is relabeled passed.

MatrixA passes202 distinct functions on its retained source:87 JVM models,
20 lifecycle/Copper,83 JVM UI,7 framed JVM and5 Copper remote. Following the
settings/worktree-ID corrections, the final affected matrix covers118 functions:
3 typed query cases,20 lifecycle/Copper,83 JVM UI,7 framed JVM and5 Copper remote.
The same three mounted default-policy functions also pass under the actual
guarded jvm development product across20 seeds; catalog status stays planned.
GPUI functions run20 seeds; ordinary tests run once. The other85 model functions
did not rerun on final source. Whole-file component proof preserves their code,
all model source outside the query component, LSP source outside the selected
phase method and UI production code;48 other source bindings stay unchanged.
Strict release/all-target/all-feature owner lint, formatting/boundary/canonical
validators/manual decomposition and final archive verification are separately
recorded against the final receipts.

The unchanged0.32.0 provider has separate retained real pinned-SDK evidence:
83 discovery,100 reactor/mask/restart and45 cached metadata checks, external
network denied and original source cache unchanged. Those checks are not rerun
on the final Rust source. Provider SHA-256 is
f7ee0b6b1aee5e205845640183706236552aafeed67454079f6717b9aef17f67.
The fixtures do not prove installed/default delivery or native application use.

Full7.3 ignore/orphan/all-provider policy,7.4 causal synchronization/settings
acknowledgement,8.7 policy editing, native default setup, live SSH and physical
platform/accessibility/resource qualification remain open. This is not an atomic
cross-process import transaction or full tool-window parity. Distribution terms,
production identity/update packaging, infrastructure CI and release gates remain
open. `jvm` stays planned; Actions stopped, VoiceOver off/deferred; no release.

### Closed Gradle resource qualification — task7.86 final

2026-10-17: pinned Buildship3.1.12/JDTLS1.61.0 qualification passes54 checks across four cached/missing-output fresh/restart journeys using unchanged development provider0.32.0. Closed resources remain outside Java membership through repeated existing Buildship sync; required-library JAR/class bytes and known zero source errors qualify, and explicit restore recovers project references. Unlink instead reimports the absent module. Three failed cohorts show bare close, diagnostic close and explicit full save cannot survive JDTLS startup cleanup. The final minimal recipe recreates a closed SDK resource from its existing .project description before controlled sync; no full save is needed. Production saved-policy replay, fresh never-imported modules without descriptions, Ignore/orphan controls and complete7.3/8.7 remain open. Sixteen old test-only factory variants have byte-identical source/descriptors; no production Java/Rust/Copper owner changes. Exact immutable source/blob/API provenance, Java17 warnings-denied fixture compilation, Python/CLI/format/boundary/both canonical validators/manual decomposition and all cohort evidence retained in .agents/evidence-archives/groovy-jvm/gradle-closed-module-provider-20261017. Final4GiB reserve holds; minimum9772843008 bytes. This is test-only SDK contract qualification, not product Ignore, Tasks/selected-tests/DAP/native/SSH/accessibility/resources/platform/terms/delivery/package/full parity/Actions/release proof. External network is denied with IPv4 localhost permitted; Buildship denied version-metadata attempts remain logged. Only isolated owned Gradle homes are stopped. Actions stopped, VoiceOver off/deferred, jvm planned.

The negative restart control is deliberate: it requires this pinned JDTLS to remove bare closed registry metadata. A future SDK that changes cleanup must requalify recovery rather than silently bypass the control. Existing saved settings, LspStore phase/task/token, host trust and provider ownership must authorize any eventual production recreation. The test-only factory command is advertised solely for this opt-in probe. Previously imported .project availability is a prerequisite; inventing project names for fresh modules is not qualified.

Manual decomposition audit:7.86 has one reviewable SDK-contract outcome, one Gradle journey owner and one existing test-only fixture-factory dependency. Shared test fixture writes remain sequential. Failed cohorts are immutable evidence, not task-completion claims.7.3 remains the production policy owner,8.7 the controls owner,7.4 causal synchronization. Copper source and production owner bindings are unchanged from c7297867b3; its prior affected regressions remain that commit's evidence, not rerun results here.

### Gradle module-state phase — task7.87 delivered

<!-- impl: tooling/jvm/gradle-model/GradleImportHandler.java#applyModuleStates -->

2026-10-17: development provider0.33.0 SHA-256 a5163a00dd384bae30ecae785acf4922320dedd8ad96cdb24244dc910b4784ae adds groovy.gradle.applyModuleStates in the existing GradleImportHandler. A shared existing build-input confinement guard runs before model discovery; an escaped-script canary proves no execution. Exact bounded requests use existing offline Gradle model discovery for authoritative membership, SDK InputStream description parsing from bounded bytes, exclusive nature/name/current-resource checks and the existing workspace scheduling rule. Whole-target preflight and descriptor/configuration/root rechecks precede closure/recovery/restore; existing diagnostic clearing is shared with ProjectUnlinkHandler. Source content is preserved; sync stays separate and fullSynchronizationKnown:false. Final90 checks across cached/missing-output fresh/restart SDK journeys plus12 direct actual-handler admission/pre-cancel checks pass. Held cancellation, changed descriptor, actual dual-nature mutation and root replacement refuse before closure; alias root/module, foreign/nonmodule/duplicate/count/key/type/description/name/ownership cases refuse actionably with sanitized messages. Three failed held-test cohorts remain failed: gate diagnostics expose .project filesystem timestamps stale after external fixture edits; refreshing the exact .project resource repairs the fixture. A stricter oracle failure incorrectly required generic failure for root replacement; exact provider cancellation is the intended outcome and the corrected final oracle passes. Earlier75/87/88 passing cohorts retain their own source/artifact bindings and do not substitute for final90. Sixteen old factory variants remain source/descriptor-identical with the new mode absent. Old Gradle import body is source-identical after inlining its factored build-input guard, unlink class differs by exactly one private-access bit, other payload entries are byte-identical; no Rust/Copper source changes. Python/CLI/digest/format/boundary/both complete canonical validators/manual audit and verified archive recorded in .agents/evidence-archives/groovy-jvm/gradle-module-state-provider-20261017. Final4GiB reserve holds, minimum9733906432 bytes. Production caller/admission/saved-policy replay, fresh modules without descriptions, ignore/orphan controls and7.3/7.4/8.7/full parity remain open. No native app/SSH/physical/accessibility/memory/platform/provider distribution/packaging/Actions/release qualification; jvm planned, VoiceOver off/deferred.

The capability is an explicit internal mutation phase, requiring trusted caller
and host permissions before the existing model query can execute Gradle scripts.
The caller is not implemented by this task. Requests require one to128 unique
non-root modules across closed/restored arrays; roots and modules must be canonical
initialized/current identities. SDK project names are read from bounded existing
descriptions, never inferred from a Gradle path in Rust. Missing descriptions
refuse, including fresh never-imported modules. SDK metadata recovery remains
closed until explicit restore. Phase receipts describe only the requested SDK
state, not durable saved intent, causal synchronization or complete Ignore parity.

Manual decomposition audit: one provider module-state outcome and one existing
Gradle journey owner; the shared test-only Maven fixture factory supplies the
existing workspace gate, with observed outcomes and exact-file refresh confined
to the opt-in mode. Existing nonmodule factory variants and product handlers are
protected by source/payload comparison. No new generic panel, project index,
result store, execution engine, Cargo model or public SDK. Copper production
owners remain unchanged; prior Copper tests remain prior evidence, not new runs.

### Fresh Gradle SDK model and closed creation — task7.88 result

<!-- impl: script/test-groovy-gradle-provider.py#fresh_gradle_fixture_bundle -->

2026-10-17: unchanged production0.33.0 SHA-256 a5163a00dd384bae30ecae785acf4922320dedd8ad96cdb24244dc910b4784ae plus a Gradle-probe-owned test-only SDK delegate pass66 checks across cached/missing-output fresh/restart journeys. Exact Buildship3.1.12 immutable source/blob/API and Java17 warnings-denied delegate compilation qualify GradleBuild.withConnection(EclipseModelUtils.queryModels), authoritative custom Eclipse names and composite module scope. A disposable closed namespace reservation changes the returned model name from groovy-eclipse-library to selection-primary-groovy-eclipse-library; model queries preserve its description and removal restores the original name. Actual closed-resource creation uses the SDK name before any Java membership or prior .project, then existing explicit sync supplies required binary JAR/classpath readiness and known zero source errors. Checkpoints prove model discovery and SDK creation preserve initially cached/missing output status; artifact production belongs to explicit synchronization in these fixtures. Restore, repeat sync, source/report integrity, unlink/reimport and cold description recovery pass. Production0.33 correctly refuses a missing description without seeding/importing. Namespace collision is a read-only name-model control; creation while the reservation remains held is not qualified. One failed cohort leaves a disposable reservation directory between cells; explicit owned metadata cleanup fixes it, after proving SDK release preserved bytes. Earlier62-check cohort is retained separately; final66 source bindings match with4GiB reserve minimum9635557376 bytes. The existing nonfresh probe AST is identical after inactive opt-in removal; generic Maven fixture, production Java and Rust/Copper owners are unchanged. Python/CLI/digest/format/boundary/both complete canonical validators/manual audit and verified archive recorded in .agents/evidence-archives/groovy-jvm/gradle-fresh-module-provider-20261017. This completes SDK qualification only: fresh production support, typed caller/saved policy, live collision seed, Ignore/orphan controls and7.3/7.4/8.7/full parity remain open. No native/physical remote/accessibility/memory/platform/terms/default delivery/packaging/Actions/release claim. VoiceOver off/deferred; jvm planned.

Manual decomposition audit: one SDK identity/creation qualification boundary; the
existing generic Eclipse factory is consumed unchanged and all Gradle-specific
source transformations and journeys remain in the Gradle probe. Its model helper
is a pinned internal Buildship API, not a public provider SDK. Test-only creation
is not production acceptance. The next provider correction must retain whole
request validation, current SDK ownership and cancellation, and separately qualify
creation with a live name conflict. SDK display/resource names are not invented
editor identities. Existing LSP/task/settings and shared tree owners retain
policy, lifecycle and presentation. No Cargo model, second index/store/engine
or panel framework is added. Previous Copper regression evidence is unchanged,
not rerun or upgraded by this test-only task.

### Fresh Gradle module states and collision artifacts — task7.89

<!-- impl: tooling/jvm/gradle-model/GradleImportHandler.java#applyModuleStates -->
<!-- impl: tooling/jvm/gradle-model/GradleImportHandler.java#refreshCollidingClosedArtifacts -->

Development provider0.34.0 (SHA-256
`81a255652ee8f78a2ea9a83509e69891ebd3e2bcb82080e3a99f57dc2a2eabfc`)
uses the pinned configured Buildship Eclipse model to create genuinely fresh
closed resources with authoritative names. Existing descriptions and the six-field
version1 module-state receipt retain their prior contract. Canonical module URIs
remain editor identities; SDK resource names can change with the namespace.
First restore without a description refuses; the existing root import owner
retains ordinary initial opening. Whole-target preflight and the workspace fence
reject new descriptions, changed scope/ownership/configuration, aliases and
cancellation before creation; errors expose bounded recovery guidance.

The live-name-conflict missing-output fixture exposed a pinned Gradle8.13 SDK
scheduling defect: raw workspace names determine open status before Eclipse name
deduplication, so the required closed dependency JAR was omitted. After normal
Buildship synchronization, the existing provider import owner now uses the
actual SDK naming model followed by its official closed-dependency model for
only affected closed resources. Actual open resources remain in that operation's
selection; unaffected closed producers remain the normal sync owner's work.
No task path is guessed. A private model in the existing Gradle plugin rejects
empty/unexpected task selection during projects-loaded, before Gradle could run
default tasks. Existing init-script creation/cleanup is shared and arguments
are appended to preserve Buildship's offline/JDK/customizer/init settings.
Existing JDT refreshExternalArchives owns archive/index refresh; explicit
org.eclipse.jdt.core OSGi visibility uses the already installed SDK bundle.
No additional download, result store, index, generic framework or build engine.

Final fresh-mode checks distinguish SDK restore jobs from explicit sync. Opening
a resource after cold restart schedules one SDK synchronization; the fixture
waits for bounded Buildship job quiescence and retains that independent count.
Every explicit sync runs its synchronization task and unaffected composite
producer exactly once, and the affected producer exactly once while closed.
The default-task canary stays absent. Reproducible fixture JARs permit strict
content hashes despite forced producer execution. Earlier counter/escaping/
protocol/OSGi failures remain failed evidence, alongside the original real
missing-artifact defect; no failed cohort is promoted to a pass.

The generic Maven fixture and Rust/Copper owners are unchanged. Existing Gradle
metadata/graph/discovery class bodies are source-identical; inactive fresh-mode
probe comparison additionally removes an uncalled namespace helper and inlines
one exact local list alias. Twelve unchanged provider payload entries are
byte-identical; changed debug-line tables are not described as byte-equivalent.
This is provider qualification only. Typed callers, saved selection replay,
Ignore/orphan UI and full7.3/7.4/8.7 remain incomplete. Tasks/selected tests/DAP,
physical local/remote workflows, platform/accessibility/memory, exact provider
terms/default delivery and production identity/packaging remain separate gates.
Actions stopped, VoiceOver off/deferred, jvm planned; no release.

Qualification: **122 fresh-state +90 existing-state +35 output/model SDK checks**
across ten journeys, plus **13 direct admission/linkage +8 selection-guard
controls**, all pass on the final0.34 binary. The268 count is assertions/controls,
not distinct test functions. Java17 warnings-denied compilation, four CLI
admission controls, Python syntax, formatting, feature boundaries, both canonical
validators and manual decomposition are retained with source/artifact bindings in
[task7.89 evidence](../../../.agents/evidence-archives/groovy-jvm/gradle-fresh-module-state-provider-20261017/README.md).
Fresh-run minimum free capacity9779359744 bytes;4GiB reserve unchanged.

### Retained Gradle root policy qualification — 2026-10-17

2026-10-17: existing LspStore applies explicit local retained Gradle root policy through typed jvm_project adapters. Visible root selection and all five capabilities are checked before requests; root read, child close, forced SDK root-close preflight, removed-ignore root/child recovery, ordinary default import and final root re-read stay in the existing task. Ignored final Closed/Absent and default Open are distinct; initial retained-ignore still requires metadata. Permission/settings/host/server/token/worktree fences, cancellation, failure/sanitized errors and retry remain. Production panel/shared tree/settings/Tasks/results/DAP/provider Java/catalog owners are unchanged; no new infrastructure.107 distinct functions pass:39 lifecycle/panel, one Copper workspace, one Copper buffer,17 model/phase,24 structured results and25 Copper provider, with one existing Copper opt-in ignored. Affected GPUI20, ordinary tests once, Copper provider seed0. Final unsupported-root wording and assertions then pass eight overlapping root controls plus strict all-target/all-feature release project/jvm_ui repository Clippy; exact delta is one error string/assertions and unchanged LSP orchestration. The broader107 matrix and its original execution/source scope are retained, not rerun or mislabeled as final-source executions. Both strict lint runs, formatting/boundary/Python/both complete canonical validators/manual ownership/decomposition/archive checks pass. Thirteen shared/provider owners remain byte-identical. Initial policy-filename and namespace-tally runner failures are retained; the latter reuses source/environment-identical raw passing executions.4GiB reserve holds, minima 10488819712/10071359488 bytes. Archive84 entries SHA-256 f5e553a3ad2aced159ecfd07ab96cbc6bb18f2bb2f0de838260c4de1e25b76d9. Only7.99 completes. Fresh ignored/no-metadata, ordinary startup/nested/link/orphan policy7.3, causal7.4 and full controls8.7 retain their gates. No paired installed SDK/editor/native/physical remote/accessibility/resource/platform/terms/default delivery/identity/packaging/Actions/full-parity/release qualification. VoiceOver deferred/off; jvm planned.

[Evidence](../../../.agents/evidence-archives/groovy-jvm/gradle-root-policy-20261017/README.md).

### Joint controlled-provider startup qualification — 2026-10-17

2026-10-17: existing JvmStartupImportControls now adds an explicit empty projectConfigurations list only when both Maven and Gradle have saved selection/exclusion controls. No-intent and single-provider options remain exact; user advanced-option conflicts still refuse. Actual initialize-request software fixture verifies trust before startup, both-to-Maven-only/Gradle-only/default/back-to-both settings replacement, correct provider flags and retained unrelated options. Explicit empty selections and exclusion intent have pure coverage. Six focused functions pass seed0; final matrix passes26 distinct active functions with affected GPUI20 schedules and ordinary tests once, including existing startup failure/retry/revocation/stop/current-host and Copper buffer/no-Java workspace controls. Strict all-target/all-feature release project/jvm_ui repository Clippy, formatting/boundary/Python/both complete canonical validators/manual decomposition/source/archive integrity pass. Production LspStore is byte-identical; thirteen shared/provider owners unchanged. No new state owner, project index, engine, result store, generic panel framework or public SDK. G07/G09 stale wording is reconciled with delivered7.98–7.100; all57 reference capability IDs/states remain unchanged and incomplete reference outcomes stay open.4GiB reserve holds, minima9842540544/7044956160 bytes. Archive44 entries SHA-256 919b47d9ec83f9807077ebca0c2d21b1027f2880a9c3834e4cae5a3b9f060d57. Only7.101 completes. SDK7.100 remains separate initialization-contract evidence; no new SDK execution or paired installed editor/provider/native/physical remote workflow is claimed. Empty configurations do not close retained resources; one-provider metadata suppression/uncontrolled default replay, retained automatic reconciliation, fresh/nested/link/empty-selection/required/orphan policy, causal sync7.4 and full7.3/8.7 remain open. VoiceOver deferred/off, Actions stopped and jvm planned; resource/platform/terms/default delivery/identity/package/full-parity/release gates remain open.

[Evidence](../../../.agents/evidence-archives/groovy-jvm/joint-startup-controls-20261017/README.md).

### Automatic saved Gradle policy — qualified software correction7.102

2026-10-17: pending saved Gradle startup now uses the existing GradleModules target used by explicit import. Five software scenarios observe automatic read/module delta/import-or-root-close/final-root ordering from settings saved before Java startup, retained child exclusions and root ignore, missing capability zero dispatch, fresh ignored-root refusal without import, and failed delta with explicit retry. Seed0 raw5 pass; its inherited wrong namespace causes the tally runner to refuse, retained and corrected before final qualification. Final matrix passes49 distinct active functions, with affected GPUI20 schedules and ordinary tests once: existing startup/LSP settings/trust/restart/failure and Copper buffer ownership, Gradle module/root cancellation/settings/host/private/missing/orphan guards, SDK-metadata watcher and Copper no-Java workspace regression. Strict project/jvm_ui all-target/all-feature release repository Clippy, formatting/boundary/both complete canonical validators/manual ownership/decomposition and source/executable/archive integrity pass. Automatic target now shares the guarded policy; obsolete ordinary target/prepared branch removed after a focused dead-code warning. Dependencies panel production, nine audited shared/provider owners, provider0.37 and catalog remain unchanged. Only unadmitted Groovy waiters are cancelled to correct an inherited policy filename and then fix the actual dead-code warning; no foreign job stopped. Initial inverse-source audit omits rustfmt line wrapping and refuses before mutation, then exact inverse verifies.4GiB reserve holds, minimum113913614336 bytes. Archive45 entries SHA-256 41dc90da290e0e51047201ff6fca4f4ad817c9a2ea56c210b734b8ef0874e0bd. No new command/schema/index/store/executor/framework/public SDK. Only7.102 completes; full7.3 retained/default/nested/link/empty-selection/required/orphan policy, one-provider metadata suppression/uncontrolled replay, causal7.4 and8.7 remain open. Existing SDK evidence remains separate; no new real paired SDK/editor/native/physical remote/accessibility/memory/platform/terms/default delivery/identity/package/full-parity/release qualification. Actions stopped, VoiceOver off/deferred and jvm planned.

[Evidence](../../../.agents/evidence-archives/groovy-jvm/gradle-startup-policy-20261017/README.md).

### Gradle default-root discovery — qualified SDK command7.103

The development-only0.38 provider exposes groovy.gradle.defaultImportRoots in the existing GradleImportHandler. Gradle retains its four descriptor names, nesting, depth and provider-specific exclusions; Maven retains POM semantics. Only the already-used SDK detector reflection primitive is shared in the current provider bundle. The query evaluates bounded explicit candidates, performs no recursive production scan, build-script execution or preference toggle, and preserves exact initialized workspace-root authority. Retained modules do not become independent build roots; nested mutation still needs a separate selected-build identity. No second project index/store/executor or public SDK is added.

2026-10-18: development provider0.38 adds groovy.gradle.defaultImportRoots in the existing GradleImportHandler. The bounded explicit-candidate query uses pinned JDTLS detector exclusions/nesting/depth and foreign-provider resource ownership without scanning, executing scripts or toggling import preferences. Exact initializedRoot authority and separate retained-module ownership remain intact. Pinned Java-17 compilation with all warnings denied passes;440 build and449 runtime bindings verify unchanged. Real externally denied-network SDK fixtures pass85 Gradle checks and83 affected Maven default-root checks over eight fresh/restart journeys. Four descriptor kinds, duplicate-directory descriptors, depth/default/host exclusions and negation, retained/foreign ownership, malformed/aliased/foreign/over-budget/incomplete requests, actual held cancellation/input removal and recovery, unchanged membership/metadata/build bytes and empty Gradle worker home are covered. Both direct pre-cancellation checks exit0 without raw causes. Eighteen positive Gradle scanner comparisons and13 retained build files reverify. Initial runtime exits1 after24 first-journey checks because fixture cleanup reads closed stderr; its exact executed source/logs are retained. Capturing stderr before shutdown fixes the fixture; the complete fresh-profile retry exits0. No foreign job stopped;4GiB reserve holds, minimum67170414592 bytes. Existing phase bodies/Maven interpretation and26 other probe functions preserve their prior semantics;43 qualified Rust/catalog bindings are unchanged, with no new Copper runtime rerun claimed. Whole formatting/shared boundary/both complete canonical validators/manual ownership/decomposition/source/command/archive integrity pass. Archive143 entries SHA-256014c09ae0ef70c56c1a0a7d5ef126f0c3f8a84c2a3f0aabfbaabd4335c611eb2. Only7.103 completes. Private model/consumer7.104–7.105 remain pending; full7.3 startup/default/nested/link/empty-selection/required/orphan reconciliation remains open. All capability table states and catalog remain unchanged. No installed/default delivery, paired editor/provider, native/remote/platform/accessibility/memory/terms/package/full-parity or release qualification. Actions stopped, VoiceOver deferred/off and jvm planned.

[Evidence](../../../.agents/evidence-archives/groovy-jvm/gradle-default-roots-20261017/README.md).

### Default Gradle policy consumer — qualified software task7.104

2026-10-18: the existing GradleModules import operation now resolves provider-owned default roots before any root/module read or mutation when saved ignore intent has no explicit selected_build_files. The private jvm_project codec uses bounded visible descriptor identities and exact reply flags; current settings, trust, host, server, phase token and captured entries are rechecked across awaits. Only the exact initialized workspace root is admitted. Empty, nested or other-root outcomes remain actionable unsupported failures, with no mutation/import. Explicit selections preserve their existing behavior. Four model and12 lifecycle cases extend existing owners; production Dependencies panel and Maven default-root codec are byte-identical, and40 other qualified Rust/catalog inputs remain unchanged. Final qualification passes78 distinct functions:71 GPUI functions each on exact seeds0–19 and7 ordinary functions once, including existing Maven/default/startup, Copper buffer and no-Java workspace regressions. The first37 passing executions are retained from checks-v4: its wrapper failed only its multiline-output tally; checks-v5 revalidates that source-identical raw log and executes41 additional functions, rather than claiming a second37 execution. Strict all-target/all-feature release project/jvm_ui repository Clippy, whole formatting, shared feature boundary, both complete canonical validators, manual ownership/decomposition and archive/source checks pass. The unused-private-code warning prompted reconciliation of unstarted7.105 into this existing operation; the test-only trait import failure and parser/preflight failures are retained with exact source and corrected retries. No lint suppression, public adapter API, scanner, second model/index/store/engine or panel framework is added. Final57 source/recipe bindings verify unchanged;4GiB reserve holds, minimum59844214784 free bytes. Archive85 entries SHA-256 b978b702ad9076e7dd02953689e02d96195a3a1c5b931424679b4c0d14a51f08. Only7.104 completes;7.105 remains explicitly superseded. Full7.3 startup/nested/link/empty-selection/required/orphan/uncontrolled-provider replay and every reference capability state remain open. Provider0.38 SDK evidence7.103 is separate; no new paired SDK/editor, native, remote, accessibility, alpha-memory, platform, distribution/terms, identity, package, full-parity or release qualification is claimed. Actions stopped, VoiceOver off/deferred and jvm planned.

[Evidence](../../../.agents/evidence-archives/groovy-jvm/gradle-default-root-policy-20261018/README.md).

### Explicit single-provider startup configurations — qualified SDK task7.106

2026-10-18: the existing opt-in SDK startup fixture now qualifies explicit Maven POM or Gradle settings-file configurations while the other importer is disabled and both providers have retained Eclipse metadata. A separately initialized plain Java project is included through its .project. Real pinned cached JDTLS/Buildship/M2E execution passes29 assertions across eight journeys: seed/repair, Maven-only fresh/restart, Gradle-only fresh/restart, both-disabled with explicit files, empty list with both enabled, and removed-list retained-metadata observation. The first two profiles admit only the intended provider plus plain Java; both-disabled admits only plain Java, and explicit empty admits none. Removal witnesses both providers' Java metadata, not full build synchronization. Sixteen source/build/generated inputs remain byte-exact; two plain-Java SDK metadata inputs have16 bounded semantic snapshots. JDTLS formats .project and adds its exact managed regex resource filter; only formatting and that filter are allowed, with unchanged name/natures/classpath. A pure preflight executes the actual fixture function on captured metadata and rejects foreign nature, widened filter, oversized metadata and changed source. Both early SDK preservation failures and the helper-placement/receipt preflight failure remain retained; the first temporary mismatch was cleaned before capture, and the second supplies exact expected/actual bytes. Final451 source/SDK/recipe bindings are unchanged,4GiB reserve holds with minimum57944272896 free bytes, and isolated owned Gradle shutdown exits0. Syntax/help, three real CLI refusals, inactive-mode AST equivalence, both complete canonical validators/manual ownership/decomposition/source/archive checks pass. All74 bindings from the delivered92-function postmerge Groovy/Copper/Tasks matrix remain current; no new Rust execution is claimed for this SDK-only change. No production Rust/provider-Java/settings/catalog behavior changes or new framework/index/store/engine/SDK. Archive86 entries SHA-256 a9fedefac0fca7ad1fbb2ca1687062eccfa4cfcf57a29f241edfb0c10dccbde2. Only7.106 completes its pinned explicit-configuration contract. Single-provider production metadata suppression/default replay, default detector/exclusion/nesting/depth equivalence, retained-resource closing and full7.3 remain open. Native/remote/physical/accessibility/alpha-memory/platform/distribution/identity/packaging/Actions/full-parity/release gates are unchanged; VoiceOver deferred/off and jvm planned.

[Evidence](../../../.agents/evidence-archives/groovy-jvm/single-provider-startup-configurations-20261018/README.md).

### Scoped selected Gradle phase — qualified SDK task7.107

2026-10-18: development provider0.39 adds internal groovy.gradle.importSelected.v2 with separate workspaceRootUri/buildRootUri and explicit offline=true. The legacy initializedRoot owner stays byte-identical; both imports share the existing Buildship synchronization/closed-artifact path. Canonical scope, workspace/build directory identities, four root descriptors bounded to4MiB each with file keys/times/sizes/SHA-256, current cached host settings and exclusive metadata/namespace/orphan guards are checked without changing workspace folders or importer flags. Version2 returns both roots, phase completion and fullSynchronizationKnown=false. Java17-target warnings-denied build passes with441 current bindings; JAR SHA-256266891c6090f10cf00b3b6f1766dc7d9219df2024e769bb6884a8590e92ad919. Final scoped SDK execution passes84 assertions/four journeys: retained fresh-profile/restart plus a pristine nested root and modules with no retained descriptions. The Maven sibling,15 source/build/generated hashes and host root/settings oracle remain intact. Malformed/foreign/offline/alias/missing/oversized/broken/dual metadata, orphan refusal, five actual held cancellation/input/workspace-identity/build-identity/host-setting controls and explicit recoveries pass; no unrequested default task or test report appears. Independent build replacement records unchanged workspace identity and changed build identity. Each owned Gradle stop exits0. The earlier60-check/seven-journey affected legacy SDK execution is retained, not rerun or relabeled:449 other inputs remain exact and only two opt-in fixture/helper paths change with exact inactive AST/held-helper and seven generated source/command/manifest comparisons. Initial60/three and76/four scoped cohorts remain historical executions, not extra final assertions. Final451 bindings hold;4GiB reserve minimum57393713152 bytes. Existing provider phase tail, initializedRoot/Maven/catalog owners and shared Rust sources remain unchanged;71 prior postmerge bindings are current and exactly three intended provider paths change. No new Rust/Copper runtime rerun is claimed for this provider-only increment. Syntax/help/two real CLI refusals, shared boundary, both canonical validators, manual ownership/decomposition/source/command/archive checks pass. Prepared absolute build argv and actual relative argv resolve identically; the actual wrapper receipt remains command authority. Archive121 entries SHA-256 c8bed2e06fb24bf661b9f855fb264991adc5ec831b8a6d6ae237b3c123eeee9a. Only7.107 completes this provider phase. Typed product callers, nested root/module policy, mixed/composite/default replay, full7.3 and all native/remote/platform/accessibility/resource/delivery/identity/package/Actions/full-parity/release gates remain open. VoiceOver off/deferred, jvm planned; no release.

[Evidence](../../../.agents/evidence-archives/groovy-jvm/scoped-gradle-import-20261018/README.md).

## Scoped Gradle caller checkpoint — 2026-10-18

Task7.108 adds a bounded local consumer for existing initialized authorities and explicit nonoverlapping nested selections without exclusions. Namespace reads, input/trust/settings/generation and host-scope fences, cancellation/timeout, retained Maven/plain Java identities and recovery are qualified through existing Project/LspStore owners. The existing namespace coverage witness is distinct from stricter classified-build/worktree-root coverage; existing full-ownership consumers retain their meaning. Local evidence covers95 unique functions (79 GPUI seeds0–19,16 ordinary) with94 unchanged executions retained after a one-test clone-only lint fix, one ordinary rerun and passing final strict owner lint.

Every reference capability state in the matrix remains unchanged. This bounded operation does not complete nested root/module/exclusion/link/default/empty/mixed/composite policy7.3, its full shared-tree workflow8.7, paired installed SDK/editor, local/native/remote journeys, diagrams/window modes, Debug, physical/accessibility/resource/platform, delivery/identity/package, Actions or release gates. jvm remains planned, VoiceOver off/deferred and Actions stopped. [Evidence](../../../.agents/evidence-archives/groovy-jvm/scoped-gradle-caller-20261018/README.md).

## Scoped SDK root-state checkpoint — 2026-10-18

Task7.109 qualifies the internal0.40 root-state SDK phase within an already initialized parent authority, reusing scoped import validation and resource read/apply owners. Final exact artifact8b26a235 passes170 scoped-root/four,84 scoped-import/four and118 legacy-root/three SDK journeys, with12 held failures/recoveries and actual Maven/plain/source/host preservation. Test-only child closing supplies a precondition; scoped module SDK policy and typed root application sequencing remain7.3/8.7. All reference matrix states remain unchanged. Paired installed/default, native/remote/platform/accessibility/RSS, delivery terms, identity/package, Actions, full parity and release gates remain open; jvm planned, VoiceOver deferred/off, Actions stopped. [Evidence](../../../.agents/evidence-archives/groovy-jvm/scoped-gradle-root-policy-20261018/README.md).

## Scoped SDK module-state checkpoint — 2026-10-18

Internal0.41 task7.110 passes656 assertions/15 cached SDK journeys:284 module,170 root,84 import and118 legacy. Sixteen held module failures include actual same-path child replacement and child inputs; an earlier forbidden successful reply is retained as a failed negative control before the final guard. Transient resource/target identities bind before model work in the current provider; existing resource/tree/Tasks/Tests/DAP/remote/catalog owners remain. This SDK phase completes no reference matrix row. Typed module/root application sequencing, full policy/parity, installed-default/native/remote/platform/accessibility/RSS, delivery/identity/package, Actions and release gates remain open. jvm planned; VoiceOver off/deferred; Actions stopped. [Evidence](../../../.agents/evidence-archives/groovy-jvm/scoped-gradle-module-policy-20261018/README.md).

## Explicit nested scoped policy checkpoint — 2026-10-18

Task7.111 qualifies one bounded local LSP operation for explicit independent nested roots with module/whole-root/parent ignore and restore, current namespace/folder/settings/input/trust/token guards and final policy verification. The existing state validators, operation/result owners and production shared-tree panel remain. Direct0.41 SDK339/four journeys and local103 functions (85 GPUI exact0–19,18 ordinary), Copper/Maven/default/legacy regressions and strict owner lint pass on the retained pre-integration source. Full7.3/8.7/default/empty/link/required/orphan/composite, installed/native/remote/platform/accessibility/RSS, delivery/identity/package/Actions and release gates remain open. All reference capability matrix states stay unchanged. jvm planned, VoiceOver off/deferred, Actions stopped; no release. Post-integration checks must be recorded separately. [Evidence](../../../.agents/evidence-archives/groovy-jvm/scoped-gradle-policy-caller-20261018/README.md).

### Post-integration scoped policy qualification — 2026-10-18

2026-10-18 post-integration: normal merge2c1483bd9106ec0a093b05c64ea00a21d0ca97c5 preserves both a40e1439ad and remote6bccf3e2f4 without conflicts. Incoming Cargo/Task/terminal/coverage/remote changes alter five prior Rust bindings; all incoming runtime/config paths are now included in73 current bindings. New current-source executions pass103 functions (85 GPUI exact0–19,18 ordinary) and strict project/jvm_ui repository release/all-target/all-feature/deny-warnings Clippy. Formatting, feature boundary, both canonical validators, source ownership audit, ancestry and diff checks pass. Groovy runtime remains byte-identical; all451 SDK bindings remain current, so SDK339/four evidence is retained, not relabeled as a new execution. Sampled4GiB guard minimum26659758080 bytes. Separate post-integration archive18 entries SHA-256 5f699303a49099e7d6d862c7f9aff8dd8fe6dd988973a0015b242322c1da134b, every payload verified. Primary194-entry archive remains unchanged. jvm planned, no targets; Actions stopped, VoiceOver off/deferred, all full-policy/parity/native/remote/platform/accessibility/RSS/delivery/identity/package/release gates remain. No release. [Evidence](../../../.agents/evidence-archives/groovy-jvm/scoped-gradle-policy-caller-20261018/README.md).

## Provider-default scoped policy checkpoint — 2026-10-18

Task7.112 joins strict cached provider-default root discovery to the existing scoped root/module/import operation, preserving current authority and refusing ambiguous/empty/unselected ownership. Current internal0.42 fixes closed-resource detector exclusions through retained SDK descriptions. Current headless SDK433/eight journeys and software108 functions (90 GPUI exact0–19,18 ordinary), strict owner Clippy and Copper/Maven/explicit/legacy regressions pass. This bounded increment does not complete default startup/control replay, empty/link/required/orphan/composite, full7.3/8.7, installed/native/remote/physical/platform/accessibility/RSS/delivery/identity/package/Actions/full parity or release. All61 reference capability rows retain their states. jvm planned/no targets, VoiceOver off/deferred and Actions stopped; no native candidate launched or release published. [Evidence](../../../.agents/evidence-archives/groovy-jvm/default-scoped-gradle-policy-20261018/README.md).

### Existing namespace owner correction — 2026-10-18

Task7.113 removes an actual duplicate namespace helper/pass added in7.112, plus its direct mirror assertions:24 removed lines. Existing validate_ownership_reply already requires bounded unique canonical Gradle identities under initialized scope and a resolved selected root, then confines current-worktree ProjectPaths. Nonempty default roots resolve before ownership queries; admitted replies necessarily pass the removed check. Source comparison proves the strict validator unchanged and the LSP source changes only by removal of that one call. Provider, SDK scripts and the entire panel source remain byte-identical. No replacement helper, framework, index, store, engine or API is added. Current source passes24 existing functions:21 GPUI each exact0–19 and3 ordinary, including default/explicit/namespace policy, shared Copper buffer preservation and Copper no-Java startup. Strict project/jvm_ui repository Clippy passes release/all-targets/all-features/deny-warnings. All73 current Rust bindings hold. Retained0.42 SDK433/eight journeys and444 build/452 SDK bindings remain current; they are not rerun or relabeled, and the broader108-function result remains its actual7.112 execution. Formatting, feature boundary, both canonical validators, manual ownership/decomposition/source/archive and diff checks pass. Initial temporary path-shape preflight failed before guard/compiler creation and is classified separately. Sampled4GiB reserve minimum22274785280 bytes. Archive30 entries SHA-256 8db0c119d1a3de0bf2555ecc2bd0792dd1b9eeebf611c2e8e08901f9abb15d07, every payload verified. All other existing task/parity states and default startup/control replay, empty/link/required/orphan/composite, full7.3/8.7, installed/native/remote/physical/platform/accessibility/RSS/delivery/terms/identity/package/Actions/full-parity/release gates remain open. jvm planned/no targets; VoiceOver off/deferred, Actions stopped. No native candidate or release. Broad goal stays active. [Evidence](../../../.agents/evidence-archives/groovy-jvm/scoped-gradle-namespace-owner-20261018/README.md).

### Cached build-root association read — 2026-10-18

Task7.114 qualifies internal groovy.gradle.importedBuildRoots through the existing ProjectUnlinkHandler identity owner. Only active/open Gradle resources are enriched with cached Buildship buildRootUri; closed/unregistered resources stay retained data, not fabricated active roots. The new seven-field cached/read-only envelope has bounded canonical unique projectUri/buildRootUri/javaNature records and truthful partial/fullSynchronizationKnown flags. Missing cached configuration and unavailable resources report incomplete data, never authoritative empty success. Existing Maven/Gradle v1 envelopes and two-field records are preserved through the same reader; its false-branch statement projection is exact. Existing provider canonical-directory/directory-identity helpers are shared at package scope with unchanged behavior. Root authority is captured before the existing resource-rule wait; active module/build identities and cached associations are rechecked. No Tooling API model, import/synchronize/reopen, root heuristic, new index/store/engine/framework or public SDK is added.

Current0.43 SHA-2562b4287309fd9352e987a34672a59da87d56e9a48c3255bc5a8aa1a2b3af5b094 passes warnings-denied Java17-target build with444 bindings,33 major61 classes using retained JBR compiler (not a Java17 compiler/runtime claim). New current externally denied-network SDK executions pass384 cached/scoped policy checks/four journeys,125 pure read checks/four fresh/restart journeys and21 Maven v1 ownership/required-library checks/two cached/missing-cache journeys:530 total/ten journeys. Actual cached root/module associations, closed/pristine/cold/restore states and unknown configuration preserve namespace, Maven/plain Java, source/generated/build inputs, metadata and host settings. The pure cohort leaves the worker home empty. Three held association controls include actual cancellation and workspace-identity errors plus a missing-module partial result, each with recovery. Eight malformed/aliased/unnormalized/foreign/nonfile/budget/noncached/missing authority cases repeat across the four pure journeys. Owned Gradle stop exits0.

First fed1 candidate498 checks are historical. Its added alias negative fails after one recorded check and captures a forbidden successful alias response; the replacement uses the existing strict canonicalDirectory guard after the v1 early return. That actual counterexample, both source/artifact variants and current530 are retained distinctly. A source-audit inventory preflight initially expected unbound provider/fixture paths among prior mixed Rust bindings; the corrected audit verifies the bound intersection and all shared runtime sources, not a runtime failure/pass. Current444 build and452/452 SDK bindings hold. Shared Rust/Copper/panel/Tasks/Tests/DAP/remote/catalog sources remain unchanged from d004/7.113;24 functions (21 GPUI exact0–19,3 ordinary) and strict owner Clippy are retained, not newly executed.70 of73 prior mixed bindings remain current; three intentional provider changes are separated. Formatting, syntax, feature boundary, both canonical validators, manual ownership/decomposition/source/archive and diff checks pass. Sampled4GiB guard minimum22203961344 bytes. Archive244 entries SHA-256 3ee529c530a21349d86154764e6bfa30533ca6654cc90c421a90c7700b67a5f9, all payloads verified; binaries/SDK/profiles/caches remain locally retained.

Only7.114 provider read completes. The typed empty/link consumer and full7.3/8.7/default startup/control replay/required/orphan/composite, installed/native/remote/physical/platform/accessibility/RSS/delivery/terms/identity/package/Actions/full-parity/release gates remain open. All reference capability states stay unchanged. jvm planned/no targets; Actions stopped, VoiceOver off/deferred. No native candidate launched or release published; broad goal remains active. [Evidence](../../../.agents/evidence-archives/groovy-jvm/cached-gradle-build-associations-20261018/README.md).

### Explicit empty Gradle selection — 2026-10-18

Task7.115 now applies explicit empty Gradle selection using qualified cached build-root facts, current v1 identity completeness and the existing source-preserving unlink/LSP owners. No model/import is dispatched for empty intent. Current113 software functions (95 GPUI exact0–19,18 ordinary), strict owner Clippy and423 SDK checks/four journeys pass; three poisoned-build probes observe zero model markers. Provider0.43 is unchanged. Partial failure/retry and override removal/default restoration have software evidence; native/remote/default startup/link/required/orphan/composite and all remaining full7.3/parity/release gates stay open.61 reference rows unchanged; jvm planned/no targets, Actions stopped, VoiceOver off/deferred. This is implementation progress, not installed parity or release readiness. [Evidence](../../../.agents/evidence-archives/groovy-jvm/empty-gradle-selection-20261018/README.md).

### Independent Gradle selection transitions — 2026-10-18

Task7.116 adds bounded local nonempty independent Gradle selection transitions through current cached association/v1 identity completeness and existing source-preserving unlink, followed by selected import/policy. Selected-root ignore remains retained close/restore, and overlapping/ambiguous roots refuse. Current117 software functions (99 GPUI exact0–19,18 ordinary), strict owner Clippy and468 SDK checks/four journeys pass. Three completed poisoned-omitted-build probes observe zero model markers. The extra independent fixture is observed before explicit disposable cleanup; independent-build restart and full7.3 startup/control replay/link/required/orphan/composite remain open. No production tree/provider Java/Tasks/Tests/DAP/remote/catalog owner changes.61 reference capability states unchanged; native/physical/platform/AT/RSS/delivery/terms/package/Actions/full-parity/release gates stay open. jvm planned/no targets, Actions stopped, VoiceOver off/deferred. [Evidence](../../../.agents/evidence-archives/groovy-jvm/gradle-selection-transition-20261018/README.md).

### Scoped Gradle details provider qualification — 2026-10-18

Task 7.117 qualifies bounded local nested Gradle details through existing model/scoped authority owners. Development 0.44.0 SHA-256 `882a824d6f465942243a096a724fb636cb76eeeddb25d0d7eab977d9b2a9f6ef` passes Java17-target warnings-denied build and 529 SDK checks/four journeys, including six held authority/cancel/timeout controls and failure/restart recovery; failed cohorts remain separate. Current affected post-integration 17 GPUI functions pass exact 0–19 plus strict editor/project/jvm_ui/vim lint. The historical117-function matrix remains distinct.7.118 product consumption, full 7.1/parity, scoped outputs/graphs, native/remote/platform/accessibility/RSS/terms/default delivery/packaging/release remain open. Model configuration executes trusted scripts and may write caches; no global read-only/cached-only or synchronization claim. Existing Tasks/Tests/DAP/tree/extension/remote/catalog owners unchanged. All 61 reference states unchanged; jvm planned/no targets, Actions stopped, VoiceOver off/deferred. [Evidence](../../../.agents/evidence-archives/groovy-jvm/scoped-gradle-details-provider-20261018/README.md).

### Module-fenced details provider qualification — 2026-10-18

Task 7.119 qualifies internal v4 registered-module input/directory authority on development 0.45.0 (`b8b6b845afb3afb7ac73fda6014904f67ad6956ef375a23e5f9e6f247c4c83bd`): warnings-denied Java17-target build, 537 SDK assertions/four journeys, ten held failure/recovery controls, same-root v4/v3/legacy equality and owned stop 0. Existing model/context/module helpers and legacy semantics are preserved. Retained unchanged-Rust evidence is 17 GPUI functions exact seeds 0–19 plus strict four-owner lint; no new Rust execution claim. Consumer 7.118 and every full-parity/native/remote/platform/accessibility/RSS/terms/delivery/packaging/release gate stay open. V3/0.44 remains root-scoped; v4 is bounded registered-module freshness, not a universal input snapshot. All 61 reference states unchanged; jvm planned/no targets, Actions stopped, VoiceOver off/deferred. [Evidence](../../../.agents/evidence-archives/groovy-jvm/module-fenced-gradle-details-provider-20261018/README.md).

## Scoped nested Gradle details consumer — 2026-10-18

Task7.118 supplies the bounded local scoped-details consumer through the existing explicit LSP refresh, cached associations, initialized-folder/worktree witnesses and Gradle merger. Require v4 for nested-module freshness; v3-only nested providers refuse before configuration, exact-root legacy compatibility remains, and empty selection evaluates no model. Active cached module coverage, six-field envelope/URI/budget validation and current settings/trust/host/server/folder/input/generation fences gate atomic publication. Per-build initialized scopes retain shared source declarations within the owning worktree. The existing URI encoder now matches actual Java NIO path spelling; strict Java File spellings are admitted only for inner Gradle identities, with an11-case retained JBR codec oracle. Shared permission retirement runs before metadata-specific soft failure; the original missing-trust seed0 regression is repaired without weakening its assertions. Current software checks pass227 distinct functions (129 GPUI exact0–19,98 ordinary), an additional exactseed0 case and strict Project/JVM UI owner lint;87 bindings unchanged and4GiB reserve held. Prior failed compilation/parser/debounce/admission/wording/permission cohorts remain distinct. No new provider SDK run or native app launch;0.45 producer qualification remains separate. Scoped outputs/graphs, closed/ignore/composite/default policy, full causal sync/settings, paired installed/native/remote/platform/accessibility/RSS/terms/identity/packaging/Actions/full-parity/release gates stay open. All61 reference dispositions unchanged; jvm planned/no targets, VoiceOver off/deferred, Actions stopped. Final source/manual/format/canonical/archive checks are recorded with the qualification.

Source/manual/decomposition/format/boundary/canonical/archive checks pass. Immutable193-payload archive SHA-256 `cf93f27a56d31d5fcccff35c6798ab8cf62cd0f11e4d4a05590c7789afc2b227`. [Consumer evidence](../../../.agents/evidence-archives/groovy-jvm/scoped-gradle-details-consumer-20261018/README.md). Only7.118 closes; current installed/native/remote/full-parity/release gates remain open.

## Cached composite ownership qualification — 2026-10-18

Task7.120 passes34 real cached SDK checks across fresh/restart, with four ownership witnesses,12 cached namespace reads and four fenced v4 models. Buildship associates the included non-Java project with the primary connection root; the Tooling API retains its distinct included-build root and project path. Independent v1 namespace equality, source/resource/generated preservation and absent default-task markers pass before mutation and after authorized SDK repair. The fixture stops only its owned Gradle workers, exit0. Development0.45 remains unchanged;448 guarded inputs and the additional prelaunch client source witness match. The4GiB reserve holds. This is provider identity evidence only: the current7.118 product consumer still refuses included participant roots, and7.121 owns that admission. Full7.1/7.3 policy, installed/native/remote/platform/accessibility/RSS/terms/delivery/packaging/full-parity/release gates remain open. All61 reference states remain unchanged; jvm planned/no targets, Actions stopped, VoiceOver off/deferred. [Evidence](../../../.agents/evidence-archives/groovy-jvm/composite-gradle-build-associations-20261018/README.md).

## Cached composite topology research — 2026-10-18

Pinned Buildship bytecode exposes a `FROM_CACHE_ONLY` path that returns an existing cache entry without invoking its loader. This is API research, not a runtime capability or freshness guarantee. Cache completeness, absent grouping roots, restart, cancellation and replacement still need actual SDK qualification. Canonical tasks 7.124–7.126 sequence that qualification, the existing provider response and scoped consumer admission; full parity and all native/remote/platform/release gates remain open. [Retained API review](../../../.agents/evidence-archives/groovy-jvm/composite-topology-api-review-20261018/README.md).
