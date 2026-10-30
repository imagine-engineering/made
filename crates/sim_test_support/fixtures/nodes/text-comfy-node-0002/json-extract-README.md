This is a bounded Task 676 correction for COMFY-NODE-0276 JsonExtractString.
Task 676 remains open for source boundaries missed by its earlier acceptance.

The original fixture.json is preserved byte-for-byte at SHA-256
2e288b0974d0a811f55505031360e7f5c41cb95493d2326bafddcd2e832d4af9.
The new generator executes the unchanged JsonExtractString class AST from pinned
sim_extras/nodes_string.py, SHA-256
bb01963178f28efc6e3a9578aecb89f53f8265e89c0518da92ec21c4955e85f1.
Only the ComfyNode base and NodeOutput wrapper are captured; json.loads and str
are the real Python standard-library implementations. This proves the original
execute body, not full Comfy imports, prompt scheduling or app behavior.

Use /home/asvegah/repos/sim/target/sim-oracle-venv/bin/python with
`generate_json_extract_source.py --check`. The recorded interpreter is Python
3.12.14 with Unicode 15.0.0. Initial generation and immutable reproduction passed.
The fixture contains 63 ordinary source outputs covering direct values, all ASCII
controls, separately escaped DEL/C1 controls, quoted/backslash-containing strings, nested
containers, dictionary insertion order, malformed JSON, null/missing keys,
nonobjects, duplicate keys and an empty key. Sixteen additional capability cases
retain fifteen actual source outputs and one actual ValueError at the default Python
integer digit limit. They are not included in the 63-case native acceptance set.

The native correction removes the leaf-local Python representation functions
and delegates through the existing NativeTextFormatter owner, including its
cancellation and result bound. JsonExtractString alone advances its implementation
identity to source-bb019631-json-repr-v2 so old cached text cannot survive the
behavior correction. No canonical formatter, JSON parser, schema or other node
implementation changes are made.

The final intended-worktree validation passed two focused tests, seven text-family
tests, and 201 node library plus four integration tests. The normal repository
Clippy check for sim_nodes and sim_test_support, workspace formatting,
immutable source reproduction and diff checks passed. The focused source matrix
requires exact outputs for all 63 ordinary cases and separately proves exact
helper-body rendering plus typed input rejection for the four original raw DEL/C1
inputs. The second test checks cancellation before publication, the canonical
16MiB formatter result limit independently of the canonical 1MiB node input/output
admission, cache identity and recovery with a source-sensitive quoted/control
input. No tolerances or native-calibrated expected outputs are used.

Remaining boundaries include Python nonprintable Unicode representation,
source-valid NUL in direct STRING outputs, arbitrary-size integers, nonfinite JSON extensions, exponent display, deep JSON
and Python's uncaught ValueError for oversized integer literals. The capability
cases document original-source behavior and must not be counted as equivalent.
The four raw DEL/C1 admission failures are runtime-proven; the other capability
cases have no native equivalence acceptance in this batch. Some boundaries can currently
produce differing values or an empty native result. Shared owner repairs and
complete row/task acceptance remain with the coordinator. The separate source
CaseConverter title/context defect is outside this correction.

The initial native focused run passed cancellation/bounds/recovery but failed
canonical STRING admission in the exact-source matrix. The diagnostic rerun
identified c1-control-7f: Python ensure_ascii=False leaves literal DEL in the JSON
input, which native workflow STRING rejects before the node body. The four
original DEL/C1 inputs remain unchanged and are recorded as admission gates. Four
new escaped-del-c1-* source cases execute the original class using distinct JSON
texts containing Unicode escapes; they are not claimed equivalent native input
representations of the rejected originals. All 75 initial source input/output/error
objects remain unchanged; the appended set has 79 source outcomes. The native
regression separately requires exact helper-body rendering and typed input
rejection for all four original raw DEL/C1 cases. Initial admission, diagnostic admission and stale capability-count assertion RED
logs are preserved in ignored target/sim-validation/json-extract evidence.
The final count-corrected runner completed with every required gate green;
its four before/after file hashes remained identical during validation.
