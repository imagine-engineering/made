# OpenAI pinned source request and helper conformance

Task453 leaf: eight source nodes and eight reviewed catalog rows, including
COMFY-API-EXT-0226 GET `/proxy/openai/v1/responses/{response_id}` from
nodes_openai.py:1182 and ApiEndpoint's default GET. Catalog226 digest:
`b230427e7db04b79515b2ef08698570070fe25fa408888d2ec97000eb18cecf0`.
The synthetic videos prefix is a tombstone, never an executable grant.

The immutable oracle executes pinned AST node methods, unmodified Pydantic API
models, actual Torch/Pillow image/mask/base64/file helpers and actual aiohttp
3.11.8 multipart serialization. Only service/poll/download and schema imports
are replaced. Fixed multipart boundary normalizes a random transport delimiter;
PNG pixels and wire bytes are source generated. 117 successes and 35 ordered
preflight failures cover schema options, model/background/custom dimensions,
ignored seed/persist_context, batch/mask, Unicode/empty prompts, advanced config,
file-chain prepending and local helpers. Four actual Chat response cases verify
first message/text, null => "None", fallback text and missing-message failures.
Six image-response cases independently prove base64 preference, URL download,
RGBA values, first-image resize/stack and corrupt/missing media rejection.

Regenerate/check only while no fixture readers are active:

```
PYTHONPATH=/tmp/sim-aiohttp-source-probe /home/asvegah/repos/sim/target/sim-oracle-venv/bin/python crates/sim_test_support/fixtures/providers/openai/generate_source_contract.py --check
```

All source hashes and library versions are recorded in source-contract.json.
The extra aiohttp directory contains the already cached pure Python runtime;
there is no repository dependency or network service requirement.

Production transport preserves sequential requests and distinct receipts,
source deadlines/retries, cancellation, bounded response/chunk/resource limits,
chat's overridden incomplete/completed statuses and first text extraction,
Sora estimate/interval and authenticated relative video content download,
credential-free image URL downloads and base64 selectors bound to the retained
parent JSON response. It requests one caller-authorized maximum cost independently
of source image token quote; unavailable quotes remain explicit, including
DALL-E with no source extractor. Input encoding, Model/file reads, live progress,
price reporting and native output materialization are typed owner callbacks.
No native Tensor/Video/Model placeholder is produced. Local helpers perform no
HTTP or cost request and keep existing OPENAI_CHAT_CONFIG/OPENAI_INPUT_FILES IDs.

Remaining acceptance: bind those callbacks to caller-owned native handles and
verified receipts; canonical base64 decode plus RGBA/no-orientation decoder,
first-image bilinear/center resize and stacking; exact source uint8 truncation,
DALL-E overlay alpha and zero-RGB masks; actual input filesystem authority and
local Model lineage; compiled guest routes/ABI, ticker, cancellation/rollback,
restart/replay and app visual checks. Mask/count validation currently precedes
canonical image preparation; source GPT methods encode base images before
rejecting some invalid masks. This source failure preparation-order difference
remains explicit until native preparation phases are integrated. Resource bounds
are independent fail-closed limits, not a claim that source accepts those bounds.
Initial oracle/import/probe failures remain in durable validation logs; the
first wire probe incorrectly used FormData() and was corrected to the actual
client.py:671 FormData(default_to_multipart=True) before fixture freeze.
