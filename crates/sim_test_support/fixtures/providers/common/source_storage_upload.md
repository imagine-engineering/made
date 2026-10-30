The producer executes the pinned UploadRequest/UploadResponse models and
upload_file_to_simapi body. It captures the actual upload_file prefix after
its content_type branch, before HTTP dispatch. Five cases cover omitted MIME,
explicit PNG/MP4 MIME and an empty generic file. Payload bytes are generic;
these cases do not establish native image encoding or upload success.

A separate actual Anthropic _build_image_content_blocks execution captures
its upload helper keyword arguments: mime_type is omitted. Encoded PNG media
identity remains image/png; storage creation omits content_type and signed PUT
must suppress automatic Content-Type. Rust tests verify the shared DTO policy
and Anthropic storage construction. Production native/guest HTTP bindings,
media allocation and real service acceptance remain unverified.
