# Tracked collaboration validation inputs

These files are the tracked owners of the collaboration requirements, design,
task ledger, catalogs and audit inputs consumed by repository checks. The migration
from local `.agents/specs/collaborative-workspace` preserves task identifiers,
completion states and historical evidence verbatim. Links and quoted commands
inside historical records identify their original authorities; they do not make
`.agents/` a build dependency. Full source inventory still requires the separately
owned Buzz source at `projects/buzz`, and CI reports that absence explicitly.

Run `python3 script/test-collaborative-workspace-inventory.py` and
`python3 script/validate-spec.py docs/collaboration/collaborative-workspace`.
`script/check-collaborative-workspace-inventory` uses these tracked inputs.
The frozen client fixture consumed by release contracts is tracked under
`crates/collaboration_domain/test_fixtures/frozen-clients.json`.

The repository-root `.agents/` directory is ignored. Removing its files from the
index keeps local copies; earlier commits still contain the formerly tracked files.
