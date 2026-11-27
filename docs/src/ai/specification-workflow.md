# Specification workflow

Describe a feature in a built-in agent conversation in Copper or Groovy. The agent can start a specification when the request involves design choices or unclear acceptance behavior. It asks about consequential choices in chat and applies the project's constitution and instructions. Quick questions and small edits can use ordinary chat. The workflow uses the Write profile or a profile with the specification workflow tool enabled.

The feature header shows the current phase and next action. **Open spec** opens the editable Markdown source. Use **Details** to see the current, reviewed and approved revisions, tasks and acceptance evidence. **Discuss findings** opens the conversation about blocking findings. Revise and rerun the review after resolving them. **Approve revision** in the conversation records the decision on the exact reviewed revision.

Use **Resume Specification**, **Review Specification**, or **Check Specification Drift** from the Agent Panel menu or command palette. To select a particular existing specification, open its Markdown file before invoking the action. Drift review compares the document with the current code or tests explicitly selected for review. The agent asks before sending those references.

## Work in the editor

Open a feature's `tasks.md`, `spec.md`, `plan.md`, or `verification.md` to see the feature bar below the editor breadcrumbs. **Requirements** and **Design** navigate the editable spec. **Open plan** appears in Design when the plan exists. Use **Agent: Focus Specification Controls**, then the Left and Right arrows and Enter to use the feature bar. Use **Agent: Open Specification Task Actions** for the task at the editor cursor, then the Up and Down arrows and Enter to select an action.

**Start task** appears above an eligible task checkbox. Click the checkbox to edit the Markdown marker, or use **Check task** or **Uncheck task** from the keyboard action picker. Undo restores the previous marker. The gutter chevron folds a task's details and descendants. **View session** opens the recorded turn when its built-in conversation is available in this project. **View evidence** opens the latest recorded check for that task. Command-click an acceptance criterion or property ID in the Markdown to jump to its definition, or choose the ID in the keyboard picker. A completed checkbox earns **Verified** only when the current approved revision has passing evidence for each criterion and property assigned to the task.

Write prerequisites on the task line or on an indented line as `Depends on: T001, T002`. Click an existing prerequisite ID in the blocked status to jump to its task. **Run remaining tasks** starts the first eligible task in file order, waits for its completion, and checks the saved approval and dependencies again before proceeding. Missing or unfinished prerequisites block dependent tasks. A changed specification pauses execution. Use **Open review** to go to the existing conversation for the revision review and approval.

**Sync files** reloads clean open feature files from disk. It preserves unsaved buffers and identifies conflicts when a file changed on disk as well. Save or resolve the affected buffers before running tasks.

## Write testable requirements

Use EARS-style criteria when the event, condition, or state makes the expected response clearer. Write meaningful invariants as separately identified properties in the canonical spec. Goldfish checks the link to the acceptance criterion and whether the input domain, preconditions, and observable result are precise enough to test.

- **AC-001** — When duplicate IDs appear in a CSV import, the importer shall keep the first valid incoming row for each ID and preserve records already in the store.
- **PROP-001 (AC-001)** — For any ordered sequence of CSV rows with integer IDs and names, an ID absent from the initial store maps to its earliest valid row, if one exists. A valid row has a positive integer ID and a nonblank name. An invalid row never reserves an ID. An ID present in the initial store retains its original value.
- **T002 (AC-001, PROP-001)** — Check duplicate precedence across sequences of valid and invalid rows and both empty and prepopulated stores. The terminal receipt records both IDs when the check asserts the rule.

In Markdown, put each property on its own line beginning with its `PROP-001`-style ID and include the linked acceptance IDs on that line. Task links use those same IDs. A passing receipt for the acceptance criterion without the linked property leaves the property unverified.

## Project format

New features use `specs/<number>-<feature>/`. Existing Spec Kit projects can use `.specify/feature.json` and its `feature_directory` field to identify a feature. When several features are available, select the one you want. Made also applies `.specify/memory/constitution.md`, top-level `AGENTS.md` and `.rules` if present.

| File | Contents |
| --- | --- |
| `spec.md` | The editable feature specification. Made uses `AC-001`-style acceptance IDs and accepts existing Spec Kit `SC-001`-style IDs. |
| `plan.md` | Technical plan derived from the approved revision, with optional research, data model, quickstart or contract files. |
| `tasks.md` | Spec Kit checkboxes with `T001`-style task IDs and acceptance IDs. Existing checkboxes are preserved. |
| `made-workflow.json` | Format version, relative specification path, per-pass review findings, reference paths and hashes, approvals, registered plan and task hashes, observed checks, and local conversation pointers for task and review turns. Existing Markdown documents with other names get a sibling `<filename>.made-workflow.json`. |
| `verification.md` | Generated check results and the status of each acceptance criterion. The report is tied to a specification revision. |

An approval binds a saved specification and its governing project files. Made normalizes line endings when calculating hashes. Other edits require review and approval of the new revision. Earlier approvals and findings remain available in the workflow record. Existing plans and tasks can be registered after review. Made asks the agent to reconcile them in place if their contents change.

Workflow artifacts are plain project files and can be reviewed using version control. Screens, model requests and raw author conversations are outside this format. Check output recorded in the workflow comes from the built-in terminal tool. Edit the spec through the editor and save it before running review or approval.

## Review and provider scope

Goldfish passes run through a separate tool-free model request with the saved document, that pass's criteria, and the displayed reference snapshots. The comprehension pass sees the document alone. The requests omit author messages, summaries, global rules, automatic memory, attachments, skills, and context servers. The author receives findings in the original chat.

Managed review supports OpenAI-compatible Chat Completions adapters configured without custom headers. The author and reviewer use the selected model with separate requests. The endpoint controls server-side retention and inference behavior. External ACP agents control their own prompts and tools. This workflow is available to the built-in agent in a trusted local project, independently of Multiplayer and collaboration sign-in.

## Upstream provenance

- GitHub Spec Kit informed the constitution, specification, planning and task formats and the requirements-to-verification flow. Made implements these stages in Rust and uses the existing project files without invoking Specify, extension hooks or branch helpers. [Pinned Spec Kit revision](https://github.com/github/spec-kit/tree/f2010547121b3351e4923fe467205079e4593a32) · [MIT license](https://github.com/github/spec-kit/blob/f2010547121b3351e4923fe467205079e4593a32/LICENSE)
- Google's Elephant–Goldfish Model informed the context split, independent review passes, human resolution, approval gate and drift review. Made uses its own prompt text and one shared specification and review record. [Pinned EGM revision](https://github.com/google/elephant-goldfish-model/tree/46933754e63c0424daa9c270d7e0734245d79560) · [Apache-2.0 license](https://github.com/google/elephant-goldfish-model/blob/46933754e63c0424daa9c270d7e0734245d79560/LICENSE)

Update these pins deliberately. Review upstream workflow and license changes, document the adaptations, and rerun the existing-project, revision and reviewer-request checks before updating either pin.
