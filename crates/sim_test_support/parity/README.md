# Sim build and conformance inputs

These catalogs, policies and fixtures are tracked inputs to the existing Sim
runtime, CLI and conformance tests. They were copied from local agent planning
storage so that normal checkouts do not require `.agents/`. The original local
files remain unchanged. `COMFY-*` task/capability identifiers and upstream ComfyUI
names identify the original compatibility inventory; they are not UI branding.
First-party CLI catalog filenames now use `sim-cli-*`, consistently across owners.

Historical milestone documents and signed device receipts retain their recorded
states and provenance. Relocation does not recertify hardware, prove parity, or
turn old evidence into a current-source execution. The ownership scanner excludes
this `parity` fixture directory from production symbol discovery.

Only inputs used by tracked consumers and their Python imports are included.
Optional full catalog regeneration still requires the reviewed upstream ComfyUI
source checkout and planning inputs named by those tools. Those external sources
are not silently substituted. Use `python3 script/check-repository-inputs` to
verify tracked static include inputs and absence of `.agents/` from Git's index.

The CLI reconciliation filename/hash table binds the relocated `sim-cli-*`
catalogs. Inventory counts, task IDs and parity completion states are retained;
refreshing this table is not a new parity or live-service qualification.
