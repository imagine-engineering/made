Bria source request and transport fixtures

Task 438 is partially implemented. This directory and bria.rs are discovered
by the existing Task 434 nested harness without shared manifest, lockfile or
registry edits. There are six source nodes and seven route rows: six executable
rows and one non-executable synthetic status-prefix tombstone. The verified
image-edit route has append-only catalog identity COMFY-API-EXT-0218; prior
identities remain unchanged and COMFY-API-EXT-0015 remains a tombstone.

The oracle executes actual pinned node.execute AST, request-model defaults and
video-duration validation. It captures mask-before-image upload order, exact
moderation field mappings, optional instruction/negative-prompt omissions,
background-image RGB projection, MP4/H264 input encoding intent, transparent
WebM/VP9 output intent, actual request-id polling paths and preflight failures
with zero uploads and requests. Storage/media encoding and codecs are explicit
probe boundaries. Run generate_source_contract.py --check, optionally with
--source-root pointing to the pinned development source tree.

BriaTransport delegates prepared uploads, clocks, retry handles, interruption
and delays to the canonical host. Image uploads preserve MIME=None: storage
creation omits content_type and presigned PUT suppresses automatic Content-Type.
Video upload requests require canonical MP4/H264 bytes with video/mp4 MIME.
The canonical preparer must select the first image batch member, convert masks,
respect the source 2048*2048 image pixel bound and project background images to
RGB. The bounded adapter additionally rejects nonfinite/nonpositive durations
and limits aggregate prepared/download bytes to 16 MiB; these are host bounds.
Generation, polling and download retries preserve separate source policies.
Queued status polls do not consume the 480 non-queued poll budget; a separate
4096-call host safety bound prevents an unbounded source queue loop. Relative
cloud requests use capability placeholders with token-before-key selection;
absolute download requests receive no credentials. The host owns endpoint,
credential, total cost, retry idempotency and output-allocation authorization.

DownloadedMedia retains the actual response bytes, terminal receipt, source URL,
submission warning, unused source status_url and structured-prompt output.
BriaMediaMaterializer must delegate actual allocation to the canonical native
owners. Images require RGBA F32 decoding; ordinary video requires the canonical
VIDEO owner; transparent VP9 requires the actual alpha decoder and RGB IMAGE
plus MASK=1-alpha, including the source empty-frame edge case. Image edit
requires IMAGE and STRING to publish atomically. Decoder-failure tests publish
no output; successful native media allocation is not claimed.

The real PNG download fixture is reused by reference from BFL. No PNG bytes are
passed off as an encoded video. Existing structural demux fixtures are not
certified video files and are not used as valid Bria media. Actual input
preparation, storage/WASM guest bridging, native media/alpha materialization,
signing payload and real worker restart still require coordinator integration.
There are no live paid calls, credentials or production signatures here.

A bounded status-normalization follow-up uses Python strip semantics, including U+001C–U+001F, before the existing lowercase/default-status classification. All six original node poll extractors select the raw status; that string remains preserved in task results and provider errors. U+200B, U+FEFF and internal controls are not stripped. Three added native regressions pass in the current fourteen-case Bria suite (audit/2026-10-03-provider-source-whitespace.md): all-node status/result matrices, actual image invoke completion/failure boundaries with authentic downloaded bytes, and queued polls beyond the processing budget with canonical delay/pre-start cancellation. The separate genuine-source observations execute the original API models, each unchanged status lambda and normalization/default-set helpers; The later72-scenario capture also executes unchanged original async poll_op/poll_op_raw with HTTP, clock and UI mocks; it does not establish real clock/UI or full provider execution. Native strict response/URL/capacity admission, cost/authority and preparation phases are unchanged. Source JSON/media/generator/conformance claims and other providers remain untouched. Full canonical native image/video/audio, guest/WIT/deployment, real clock/UI, worker restart and visual acceptance remain partial.
