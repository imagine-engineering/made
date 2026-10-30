ARG MIGRATION_BASE_IMAGE=postgres:17-alpine
FROM ${MIGRATION_BASE_IMAGE}

RUN apk add --no-cache python3 \
    && addgroup -S -g 65532 migration \
    && adduser -S -D -H -u 65532 -G migration migration

WORKDIR /opt/zed-collaboration-migrations
COPY deploy/collaboration/migrations/migrate.py ./migrate.py
COPY deploy/collaboration/migrations/manifest.json ./manifest.json
COPY crates/collab/migrations/*.up.sql /migrations/
COPY crates/collab/migrations/*.down.sql /migrations/

USER 65532:65532
ENTRYPOINT ["python3", "/opt/zed-collaboration-migrations/migrate.py"]
CMD ["up"]
