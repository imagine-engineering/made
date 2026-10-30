# Actual aiohttp multipart wire reference

`generate_source_multipart.py` captures actual aiohttp 3.11.8 FormData bytes
with a fixed boundary. This is the minimum aiohttp version declared by the
pinned Comfy source; it does not claim every allowed newer dependency version.
The JSON binds the FormData and multipart implementation hashes and producer
hash. Rust compares the full body at exact bytes, including header order,
UTF-8 text, repeated files, empty text and empty files. Canonical body bounds
are checked at the exact serialized length and one byte below it. Control
characters in media types fail before encoding.

Use the isolated source environment for immutable verification:

```sh
PYTHONPATH=/tmp/sim-aiohttp-source-probe target/sim-oracle-venv/bin/python \
  crates/sim_test_support/fixtures/providers/common/generate_source_multipart.py --check
```

The initial sandbox producer stalled in thread-backed file reads and was
terminated before retry outside sandbox. No HTTP request was sent. Reference
capture completed before Rust readers started; subsequent checks are read-only.
This evidence verifies encoding, not production guest/WIT or upload binding.

The same producer also captures eight actual string-only FormData requests as
URL-encoded bodies. The canonical ordered string encoder preserves duplicate
names, empty names/values, UTF-8, quote-plus spaces, literal plus signs, percent
signs and uppercase hex escapes. Runtime compares all eight bodies at exact
bytes, checks exact limits and one byte below, and rejects a zero body limit.
The original three multipart case objects remain byte-identical; only the
producer hash and appended URL-encoded records change.

Sonilo source conformance delegates ordered string fields to this canonical
encoder and compares its result with the original aiohttp wire capture. This
does not certify Python scalar conversion, permissive base64, HTTP charset,
production receipt/clock binding, native audio decode/allocation, or a compiled
guest. Those callbacks remain explicitly unbound until separately validated.
