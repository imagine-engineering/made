This is the Unicode-only corrective scope for canonical Task 413.

The source is unchanged StringFormat class AST from sim_extras/nodes_string.py,
SHA-256 bb01963178f28efc6e3a9578aecb89f53f8265e89c0518da92ec21c4955e85f1.
The original class delegates to actual Python str.format. Only descriptor
annotations and the NodeOutput wrapper are captured; full Comfy imports, node
scheduling and app behavior are not claimed.

Generate and immutably check using the pinned interpreter:
/home/asvegah/repos/sim/target/sim-oracle-venv/bin/python generate_source.py
/home/asvegah/repos/sim/target/sim-oracle-venv/bin/python generate_source.py --check
The generator rejects an interpreter other than Python 3.12.14 / Unicode 15.0.0.
--write-table also writes the generated nonprintable ranges into the existing
canonical text_format.rs owner. An immutable --check never writes that owner.

The oracle covers all 1,112,064 valid Unicode scalars, excluding only the 2,048
UTF-16 surrogate codepoints that native UTF-8 strings cannot represent. It records
713 nonprintable ranges, a printability bitmap digest, and independently obtained
original-source repr/ascii output digests. Each output is framed by little-endian
u32 codepoint, little-endian u32 UTF-8 byte count and output bytes. These rendered
output digests independently verify more than consistency with a generated table.
The 32 ordinary source cases exercise Latin-1 escapes, format/unassigned/private-use
characters, separators, printable emoji, quote selection, nested dictionaries and
lists, escaped conversion width and precision. U+1C89 remains unassigned in this
pinned source despite newer Unicode category libraries.

The production correction lives only in the canonical quotation owner. Its
Unicode 15 predicate has source-generated data and no new crate dependency;
Rust/dependency Unicode versions are not assumed to match Python. ASCII conversion
uses source \xNN below U+0100, \uNNNN in the BMP and \UNNNNNNNN above it.
Two earlier inline expectations used \u00e9 for Latin-1 ASCII conversion, whereas
actual original source uses \xe9; those assertions are corrected by source evidence.

Fresh intended-worktree Rust validation passed all ten canonical formatter tests,
including complete scalar printability and both independent original rendered
output digests, all 32 source cases, escaped-output bounds, pre-cancellation and
recovery. The existing JsonExtractString consumer family passed 7/7 and the
StringFormat consumer family passed 5/5. Full sim_nodes all-targets passed
204 library and four integration tests, with no failures or ignored tests.
Normal repository warnings-denied release/all-target/all-feature Clippy for
sim_nodes/sim_test_support, workspace formatting, immutable source reproduction
and diff checks all passed on the first native attempt. Ignored validation evidence
is retained under target/sim-validation/text-format-unicode/initial. Every owned
file hash, the prior JsonExtract checkpoint and the numeric production code stayed
unchanged throughout the readers; only this results paragraph was updated afterward.
No tolerance or native-calibrated source gold is used. This does not certify app
visuals, registry scheduling or the remaining source representation boundaries.

Numeric renderer, decimal JSON parser, huge integers, nonfinite admission,
workflow control/NUL input/output admission, surrogate representation and source
integer/depth errors remain outside this correction. The prior JsonExtract four-file
checkpoint and its 79 original source outcomes remain immutable. Task 413/676
full closure and dependent consumer cache invalidation remain coordinator-owned.

Coordinator extension executes the same unchanged original StringFormat class for 2,171 finite binary64 bit patterns and four nested containers. Values are reconstructed from hexadecimal bits to keep decimal parsing outside this proof. The original 32 Unicode cases, all-scalar digests and 713 source ranges remain unchanged. Native validation of this extension is pending. Numeric precision modes, arbitrary integers, decimal parsing, nonfinite workflow admission and lone surrogates remain separate gaps.
